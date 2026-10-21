"""Backup / restore of the SQLite database and collection reports (local folder, optional AWS S3)."""
import os, re, sqlite3
from datetime import datetime

SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def s3_configured():
    return bool(os.environ.get("ECOSORT_S3_BUCKET"))


def _upload(path):
    """Upload a file to S3 when ECOSORT_S3_BUCKET is set. Returns True if uploaded, False if local-only."""
    if not s3_configured():
        return False
    try:
        import boto3
        boto3.client("s3", region_name=os.environ.get("AWS_REGION")).upload_file(
            path, os.environ["ECOSORT_S3_BUCKET"], "ecosort/" + os.path.basename(path))
        return True
    except Exception as exc:
        raise RuntimeError(f"Saved locally, but the S3 upload failed: {exc}") from exc


def create_backup(db_path, backup_dir, prefix="ecosort"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{prefix}_{stamp}.db"          # microseconds make names unique, so nothing is overwritten
    dest = os.path.join(backup_dir, name)
    src, dst = sqlite3.connect(db_path), sqlite3.connect(dest)
    with dst:
        src.backup(dst)           # consistent online copy
    src.close(); dst.close()
    return name, _upload(dest)


def _check_name(backup_dir, name):
    if not SAFE_NAME.match(name or "") or not name.endswith(".db"):
        raise ValueError("Invalid backup name.")
    path = os.path.join(backup_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError("Backup file not found.")
    return path


def restore_backup(db_path, backup_dir, name):
    path = _check_name(backup_dir, name)
    probe = sqlite3.connect(path)
    try:
        ok = probe.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        tables = {r[0] for r in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.DatabaseError:
        ok, tables = False, set()
    finally:
        probe.close()
    if not ok or not {"users", "waste_records", "audit_logs"} <= tables:
        raise ValueError("That file is not a valid EcoSort database backup.")
    create_backup(db_path, backup_dir, prefix="prerestore")   # safety copy of the current database
    src, dst = sqlite3.connect(path), sqlite3.connect(db_path)
    with dst:
        src.backup(dst)
    src.close(); dst.close()


def save_report(report_dir, csv_text):
    name = datetime.now().strftime("collection_report_%Y%m%d_%H%M%S.csv")
    path = os.path.join(report_dir, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(csv_text)
    return name, _upload(path)


def list_files(folder, suffix):
    if not os.path.isdir(folder):
        return []
    return sorted((f for f in os.listdir(folder) if f.endswith(suffix)), reverse=True)
