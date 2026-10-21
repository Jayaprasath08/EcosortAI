"""EcoSort AI - waste classification and collection monitoring (Flask + SQLite + MobileNetV2)."""
import csv, io, os, secrets, sqlite3, uuid
from datetime import date, datetime, timedelta
from functools import wraps

from flask import (Flask, Response, abort, flash, g, redirect, render_template, request,
                   send_from_directory, session, url_for)
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

import backup as backup_tools
from classifier import ModelUnavailable, classifier

BASE = os.path.dirname(os.path.abspath(__file__))
CATEGORIES = ["plastic", "paper", "metal", "organic", "glass", "other"]
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_FAILED_LOGINS, LOCK_MINUTES = 5, 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin','staff')));
CREATE TABLE IF NOT EXISTS waste_records(
  id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL, source TEXT NOT NULL,
  location TEXT NOT NULL, weight_kg REAL NOT NULL, collected_on TEXT NOT NULL, notes TEXT DEFAULT '',
  created_by TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS classifications(
  id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL, predicted_category TEXT,
  confidence REAL, top_label TEXT, model_note TEXT, confirmed_category TEXT,
  created_by TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, username TEXT, action TEXT NOT NULL,
  detail TEXT, ip TEXT);
"""


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_app(test_config=None):
    app = Flask(__name__)
    inst = os.path.join(BASE, "instance")
    os.makedirs(inst, exist_ok=True)
    app.config.update(
        DB_PATH=os.path.join(inst, "ecosort.db"), UPLOAD_DIR=os.path.join(inst, "uploads"),
        BACKUP_DIR=os.path.join(BASE, "backups"), REPORT_DIR=os.path.join(BASE, "reports"),
        MAX_CONTENT_LENGTH=5 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax", PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
        SEED_ADMIN_PASSWORD=os.environ.get("ECOSORT_ADMIN_PASSWORD", "Admin@123"),
        SEED_STAFF_PASSWORD=os.environ.get("ECOSORT_STAFF_PASSWORD", "Staff@123"),
    )
    key_file = os.path.join(inst, "secret.key")
    if os.environ.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = os.environ["SECRET_KEY"]
    else:
        if not os.path.exists(key_file):
            with open(key_file, "w") as f:
                f.write(secrets.token_hex(32))
        with open(key_file) as f:
            app.config["SECRET_KEY"] = f.read().strip()
    if test_config:
        app.config.update(test_config)
    for d in (app.config["UPLOAD_DIR"], app.config["BACKUP_DIR"], app.config["REPORT_DIR"]):
        os.makedirs(d, exist_ok=True)

    # ---------- database helpers ----------
    def db():
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DB_PATH"])
            g.db.row_factory = sqlite3.Row
        return g.db

    @app.teardown_appcontext
    def close_db(_):
        conn = g.pop("db", None)
        if conn:
            conn.close()

    def init_db():
        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            conn.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                         ("admin", generate_password_hash(app.config["SEED_ADMIN_PASSWORD"]), "admin"))
            conn.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                         ("staff", generate_password_hash(app.config["SEED_STAFF_PASSWORD"]), "staff"))
        conn.commit()
        conn.close()

    init_db()

    def audit(action, detail="", username=None):
        db().execute("INSERT INTO audit_logs(ts,username,action,detail,ip) VALUES(?,?,?,?,?)",
                     (now(), username or session.get("username"), action, detail[:300], request.remote_addr))
        db().commit()

    # ---------- security helpers ----------
    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def protect_post():
        if request.method == "POST":
            sent = request.form.get("csrf_token", "")
            if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
                abort(400, "Invalid or missing CSRF token")

    def login_required(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            if "user_id" not in session:
                return redirect(url_for("login"))
            return fn(*a, **kw)
        return wrapper

    def admin_required(fn):
        @wraps(fn)
        @login_required
        def wrapper(*a, **kw):
            if session.get("role") != "admin":
                audit("ACCESS_DENIED", request.path)
                abort(403)
            return fn(*a, **kw)
        return wrapper

    @app.context_processor
    def inject_user():
        return {"current_user": session.get("username"), "role": session.get("role")}

    # ---------- validation ----------
    def validate_record(form):
        errors, c = [], {}
        c["category"] = form.get("category", "")
        if c["category"] not in CATEGORIES:
            errors.append("Choose a valid waste category.")
        for field, label in (("source", "Source"), ("location", "Location")):
            c[field] = form.get(field, "").strip()
            if not 1 <= len(c[field]) <= 80:
                errors.append(f"{label} is required (max 80 characters).")
        try:
            c["weight_kg"] = float(form.get("weight_kg", ""))
            if not 0 < c["weight_kg"] <= 10000:
                raise ValueError
        except ValueError:
            errors.append("Weight must be a number above 0 and at most 10000 kg.")
        c["collected_on"] = form.get("collected_on", "")
        try:
            if datetime.strptime(c["collected_on"], "%Y-%m-%d").date() > date.today():
                errors.append("Collection date cannot be in the future.")
        except ValueError:
            errors.append("Collection date must be a valid date (YYYY-MM-DD).")
        c["notes"] = form.get("notes", "").strip()[:200]
        return errors, c

    def insert_record(c):
        db().execute("INSERT INTO waste_records(category,source,location,weight_kg,collected_on,notes,created_by,created_at)"
                     " VALUES(?,?,?,?,?,?,?,?)", (c["category"], c["source"], c["location"], c["weight_kg"],
                                                  c["collected_on"], c["notes"], session["username"], now()))
        db().commit()

    # ---------- auth ----------
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()[:50]
            since = (datetime.now() - timedelta(minutes=LOCK_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
            failed = db().execute("SELECT COUNT(*) FROM audit_logs WHERE action='LOGIN_FAILED' AND username=? AND ts>=?",
                                  (username, since)).fetchone()[0]
            if failed >= MAX_FAILED_LOGINS:
                audit("LOGIN_BLOCKED", "too many failed attempts", username)
                flash(f"Too many failed attempts. Try again in {LOCK_MINUTES} minutes.", "error")
                return render_template("login.html"), 429
            user = db().execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
                session.clear()
                session.permanent = True
                session.update(user_id=user["id"], username=user["username"], role=user["role"])
                audit("LOGIN_SUCCESS", f"role={user['role']}")
                return redirect(url_for("dashboard"))
            audit("LOGIN_FAILED", "bad credentials", username)
            flash("Invalid username or password.", "error")
            return render_template("login.html"), 401
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        audit("LOGOUT")
        session.clear()
        return redirect(url_for("login"))

    # ---------- dashboard ----------
    @app.route("/")
    @login_required
    def dashboard():
        since = (date.today() - timedelta(days=30)).isoformat()
        by_cat = db().execute("SELECT category, COUNT(*) n, ROUND(SUM(weight_kg),2) kg FROM waste_records "
                              "WHERE collected_on>=? GROUP BY category ORDER BY kg DESC", (since,)).fetchall()
        totals = db().execute("SELECT COUNT(*) n, COALESCE(ROUND(SUM(weight_kg),2),0) kg FROM waste_records").fetchone()
        recent = db().execute("SELECT * FROM classifications ORDER BY id DESC LIMIT 5").fetchall()
        max_kg = max([r["kg"] for r in by_cat], default=1) or 1
        return render_template("dashboard.html", by_cat=by_cat, totals=totals, recent=recent, max_kg=max_kg)

    # ---------- AI classification ----------
    @app.route("/classify", methods=["GET", "POST"])
    @login_required
    def classify():
        if request.method == "POST":
            f = request.files.get("image")
            if not f or not f.filename:
                flash("Choose an image first.", "error")
                return redirect(url_for("classify"))
            ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
            if ext not in ALLOWED_EXT:
                flash("Only PNG and JPG images are allowed.", "error")
                return redirect(url_for("classify"))
            name = f"{uuid.uuid4().hex}.{ext}"
            path = os.path.join(app.config["UPLOAD_DIR"], name)
            f.save(path)
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:
                os.remove(path)
                flash("That file is not a valid image.", "error")
                return redirect(url_for("classify"))
            try:
                res = classifier.predict(path)
            except ModelUnavailable as exc:
                res = {"category": None, "confidence": None, "top_label": "",
                       "note": "AI model unavailable (" + str(exc)[:120] + "). Choose the category manually."}
            cur = db().execute("INSERT INTO classifications(filename,predicted_category,confidence,top_label,model_note,created_by,created_at)"
                               " VALUES(?,?,?,?,?,?,?)", (name, res["category"], res["confidence"], res["top_label"],
                                                          res["note"], session["username"], now()))
            db().commit()
            audit("IMAGE_CLASSIFIED", f"id={cur.lastrowid} predicted={res['category']}")
            return redirect(url_for("classification", cid=cur.lastrowid))
        rows = db().execute("SELECT * FROM classifications ORDER BY id DESC LIMIT 20").fetchall()
        return render_template("classify.html", rows=rows)

    @app.route("/classify/<int:cid>")
    @login_required
    def classification(cid):
        row = db().execute("SELECT * FROM classifications WHERE id=?", (cid,)).fetchone() or abort(404)
        return render_template("result.html", c=row, categories=CATEGORIES, today=date.today().isoformat())

    @app.route("/classify/<int:cid>/save", methods=["POST"])
    @login_required
    def save_classification(cid):
        row = db().execute("SELECT * FROM classifications WHERE id=?", (cid,)).fetchone() or abort(404)
        errors, c = validate_record(request.form)
        if errors:
            for e in errors:
                flash(e, "error")
            return redirect(url_for("classification", cid=cid))
        insert_record(c)
        db().execute("UPDATE classifications SET confirmed_category=? WHERE id=?", (c["category"], cid))
        db().commit()
        audit("RECORD_FROM_CLASSIFICATION", f"id={cid} category={c['category']} kg={c['weight_kg']}")
        flash("Collection record saved.", "ok")
        return redirect(url_for("records"))

    @app.route("/uploads/<name>")
    @login_required
    def uploaded(name):
        return send_from_directory(app.config["UPLOAD_DIR"], name)

    # ---------- collection records ----------
    @app.route("/records", methods=["GET", "POST"])
    @login_required
    def records():
        if request.method == "POST":
            errors, c = validate_record(request.form)
            if errors:
                for e in errors:
                    flash(e, "error")
            else:
                insert_record(c)
                audit("RECORD_ADDED", f"category={c['category']} kg={c['weight_kg']} at={c['location']}")
                flash("Record added.", "ok")
            return redirect(url_for("records"))
        rows = db().execute("SELECT * FROM waste_records ORDER BY collected_on DESC, id DESC LIMIT 200").fetchall()
        return render_template("records.html", rows=rows, categories=CATEGORIES, today=date.today().isoformat())

    @app.route("/records/<int:rid>/delete", methods=["POST"])
    @admin_required
    def delete_record(rid):
        db().execute("DELETE FROM waste_records WHERE id=?", (rid,))
        db().commit()
        audit("RECORD_DELETED", f"id={rid}")
        flash("Record deleted.", "ok")
        return redirect(url_for("records"))

    def records_csv():
        rows = db().execute("SELECT id,category,source,location,weight_kg,collected_on,notes,created_by FROM waste_records "
                            "ORDER BY collected_on, id").fetchall()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "category", "source", "location", "weight_kg", "collected_on", "notes", "created_by"])
        for r in rows:
            # prefix formula-like cells so spreadsheets do not execute them (CSV injection)
            w.writerow([("'" + str(v)) if isinstance(v, str) and v[:1] and v[:1] in "=+-@" else v for v in tuple(r)])
        return buf.getvalue()

    @app.route("/records/export.csv")
    @login_required
    def export_csv():
        audit("CSV_EXPORT")
        return Response(records_csv(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=collection_records.csv"})

    # ---------- admin: users, audit, backup ----------
    @app.route("/users", methods=["GET", "POST"])
    @admin_required
    def users():
        if request.method == "POST":
            u, p, r = request.form.get("username", "").strip(), request.form.get("password", ""), request.form.get("role", "")
            if not (3 <= len(u) <= 30 and u.replace("_", "").isalnum()):
                flash("Username must be 3-30 letters, digits or underscores.", "error")
            elif len(p) < 8 or r not in ("admin", "staff"):
                flash("Password needs at least 8 characters and the role must be admin or staff.", "error")
            else:
                try:
                    db().execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                                 (u, generate_password_hash(p), r))
                    db().commit()
                    audit("USER_CREATED", f"username={u} role={r}")
                    flash("User created.", "ok")
                except sqlite3.IntegrityError:
                    flash("That username already exists.", "error")
            return redirect(url_for("users"))
        return render_template("users.html", rows=db().execute("SELECT id,username,role FROM users").fetchall())

    @app.route("/audit")
    @admin_required
    def audit_page():
        rows = db().execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 300").fetchall()
        return render_template("audit.html", rows=rows)

    @app.route("/backup", methods=["GET", "POST"])
    @admin_required
    def backup_page():
        if request.method == "POST":
            action = request.form.get("action")
            try:
                if action == "create":
                    name, remote = backup_tools.create_backup(app.config["DB_PATH"], app.config["BACKUP_DIR"])
                    audit("BACKUP_CREATED", f"{name} cloud={remote}")
                    flash(f"Backup {name} created" + (" and uploaded to S3." if remote else " (local only)."), "ok")
                elif action == "restore":
                    name = request.form.get("name", "")
                    backup_tools.restore_backup(app.config["DB_PATH"], app.config["BACKUP_DIR"], name)
                    audit("BACKUP_RESTORED", name)
                    flash(f"Database restored from {name}. A safety copy of the previous database was kept.", "ok")
                elif action == "report":
                    name, remote = backup_tools.save_report(app.config["REPORT_DIR"], records_csv())
                    audit("REPORT_SAVED", f"{name} cloud={remote}")
                    flash(f"Collection report {name} saved" + (" and uploaded to S3." if remote else " (local only)."), "ok")
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                flash(str(exc), "error")
            return redirect(url_for("backup_page"))
        return render_template("backup.html", backups=backup_tools.list_files(app.config["BACKUP_DIR"], ".db"),
                               reports=backup_tools.list_files(app.config["REPORT_DIR"], ".csv"),
                               s3=backup_tools.s3_configured())

    @app.errorhandler(403)
    def forbidden(_):
        return render_template("error.html", code=403, msg="You do not have permission to open this page."), 403

    @app.errorhandler(400)
    def bad_request(e):
        return render_template("error.html", code=400, msg=getattr(e, "description", "Bad request")), 400

    @app.errorhandler(413)
    def too_large(_):
        return render_template("error.html", code=413, msg="Image is larger than 5 MB."), 413

    return app


if __name__ == "__main__":
    create_app().run(debug=False)
