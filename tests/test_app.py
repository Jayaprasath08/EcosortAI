"""Smoke tests. Run:  python -m unittest discover -s tests -v
The real MobileNetV2 is replaced by a stand-in so tests run offline; the label-mapping logic is tested directly."""
import io, os, re, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import app as app_module
import classifier as clf


class FakeClassifier:
    def predict(self, path):
        return {"category": "plastic", "confidence": 0.8, "top_label": "water_bottle", "note": "stub"}


class EcoSortTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        app_module.classifier = FakeClassifier()
        self.app = app_module.create_app({
            "TESTING": True, "DB_PATH": os.path.join(t, "t.db"), "UPLOAD_DIR": os.path.join(t, "up"),
            "BACKUP_DIR": os.path.join(t, "bk"), "REPORT_DIR": os.path.join(t, "rp"), "SECRET_KEY": "test"})
        self.c = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def token(self, path="/login"):
        html = self.c.get(path).get_data(as_text=True)
        return re.search(r'name="csrf_token" value="([0-9a-f]+)"', html).group(1)

    def login(self, user="admin", pw="Admin@123"):
        return self.c.post("/login", data={"username": user, "password": pw, "csrf_token": self.token()})

    def post(self, url, data, page="/records", **kw):
        data = dict(data, csrf_token=self.token(page))
        return self.c.post(url, data=data, **kw)

    REC = {"category": "plastic", "source": "Bin A", "location": "Market", "weight_kg": "3.5", "collected_on": "2024-01-10", "notes": ""}

    def test_requires_login(self):
        self.assertEqual(self.c.get("/").status_code, 302)

    def test_csrf_enforced(self):
        self.assertEqual(self.c.post("/login", data={"username": "admin", "password": "Admin@123"}).status_code, 400)

    def test_login_good_and_bad(self):
        self.assertEqual(self.c.post("/login", data={"username": "admin", "password": "x", "csrf_token": self.token()}).status_code, 401)
        self.assertEqual(self.login().status_code, 302)
        self.assertEqual(self.c.get("/").status_code, 200)

    def test_lockout_after_failures(self):
        for _ in range(5):
            self.c.post("/login", data={"username": "staff", "password": "bad", "csrf_token": self.token()})
        self.assertEqual(self.c.post("/login", data={"username": "staff", "password": "Staff@123", "csrf_token": self.token()}).status_code, 429)

    def test_passwords_are_hashed(self):
        import sqlite3
        h = sqlite3.connect(self.app.config["DB_PATH"]).execute("SELECT password_hash FROM users WHERE username='admin'").fetchone()[0]
        self.assertNotIn("Admin@123", h)

    def test_staff_cannot_open_admin_pages(self):
        self.login("staff", "Staff@123")
        for url in ("/audit", "/backup", "/users"):
            self.assertEqual(self.c.get(url).status_code, 403, url)

    def test_record_validation_and_csv(self):
        self.login()
        self.post("/records", dict(self.REC, weight_kg="-4"))
        self.post("/records", dict(self.REC, category="gold"))
        self.post("/records", dict(self.REC, collected_on="2999-01-01"))
        self.post("/records", dict(self.REC, source="=HYPERLINK(1)"))
        self.post("/records", self.REC)
        body = self.c.get("/records/export.csv").get_data(as_text=True).strip().splitlines()
        self.assertEqual(len(body), 3)             # header + 2 valid rows
        self.assertTrue(any("'=HYPERLINK" in l for l in body))   # CSV injection neutralised

    def test_staff_cannot_delete_but_admin_can(self):
        self.login()
        self.post("/records", self.REC)
        self.c.post("/logout", data={"csrf_token": self.token("/records")})
        self.login("staff", "Staff@123")
        self.assertEqual(self.post("/records/1/delete", {}).status_code, 403)
        self.c.post("/logout", data={"csrf_token": self.token("/records")})
        self.login()
        self.assertEqual(self.post("/records/1/delete", {}).status_code, 302)

    def image_bytes(self, fmt="PNG"):
        b = io.BytesIO(); Image.new("RGB", (40, 40), "green").save(b, fmt); b.seek(0); return b

    def test_upload_classify_and_save(self):
        self.login()
        r = self.post("/classify", {"image": (self.image_bytes(), "bottle.png")}, page="/classify", content_type="multipart/form-data")
        self.assertEqual(r.status_code, 302)
        page = self.c.get(r.headers["Location"]).get_data(as_text=True)
        self.assertIn("plastic", page)
        s = self.post("/classify/1/save", dict(self.REC), page=r.headers["Location"])
        self.assertEqual(s.status_code, 302)
        self.assertIn("Bin A", self.c.get("/records").get_data(as_text=True))

    def test_upload_rejects_non_images(self):
        self.login()
        for name, data in (("x.txt", io.BytesIO(b"hello")), ("x.png", io.BytesIO(b"not an image"))):
            r = self.post("/classify", {"image": (data, name)}, page="/classify", content_type="multipart/form-data", follow_redirects=True)
            self.assertIn("class=\"flash error\"", r.get_data(as_text=True))

    def test_backup_restore_roundtrip(self):
        self.login()
        self.post("/records", self.REC)
        self.post("/backup", {"action": "create"}, page="/backup")
        name = os.listdir(self.app.config["BACKUP_DIR"])[0]
        self.post("/records/1/delete", {})
        self.assertNotIn("Bin A", self.c.get("/records").get_data(as_text=True))
        self.post("/backup", {"action": "restore", "name": name}, page="/backup")
        self.assertIn("Bin A", self.c.get("/records").get_data(as_text=True))

    def test_restore_rejects_bad_names(self):
        self.login()
        r = self.post("/backup", {"action": "restore", "name": "../../etc/passwd"}, page="/backup", follow_redirects=True)
        self.assertIn("Invalid backup name", r.get_data(as_text=True))

    def test_audit_trail(self):
        self.login()
        self.post("/records", self.REC)
        page = self.c.get("/audit").get_data(as_text=True)
        for a in ("LOGIN_SUCCESS", "RECORD_ADDED"):
            self.assertIn(a, page)

    def test_label_mapping(self):
        r = clf.map_labels([("water_bottle", 0.6), ("pop_bottle", 0.1), ("beer_bottle", 0.05)])
        self.assertEqual(r["category"], "plastic")
        self.assertEqual(clf.map_labels([("tiger", 0.9)])["category"], "other")
        self.assertEqual(clf.map_labels([("banana", 0.5)])["category"], "organic")


if __name__ == "__main__":
    unittest.main()
