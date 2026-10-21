# EcoSort AI - Smart Waste Classification and Monitoring

Flask + SQLite web app. Upload a photo of a waste item, get a category estimate from a pretrained
**MobileNetV2**, confirm it, and save it as a collection record. Includes login, roles, audit log,
CSV export and backup/restore with optional AWS S3 storage.

## Modules
| Module | What it does |
|---|---|
| Data | Collection records (category, source, location, weight, date), classification history, dashboard totals, CSV export |
| Cloud | Database backup/restore and saved CSV reports in local folders; uploads to S3 when `ECOSORT_S3_BUCKET` is set |
| AI | MobileNetV2 (ImageNet weights) -> ImageNet labels mapped to plastic / paper / metal / organic / glass, else "other" |
| Security | Password hashing, admin/staff roles, CSRF tokens, upload checks, login lockout, audit log |

## Run (Windows PowerShell)
```
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000. Demo accounts: `admin` / `Admin@123` and `staff` / `Staff@123`.
**Change them** (set `ECOSORT_ADMIN_PASSWORD` / `ECOSORT_STAFF_PASSWORD` before the first run, or add users on the Users page).
The first classification downloads the MobileNetV2 weights (~14 MB, needs internet) and may take a minute.
If TensorFlow or the weights are unavailable, the app still runs and asks you to pick the category manually.

## Run the tests (no TensorFlow or internet needed)
```
python -m unittest discover -s tests -v
```
14 tests cover login, lockout, CSRF, role restrictions, validation, CSV, uploads, backup/restore and the label mapping.
The tests replace the real model with a stand-in, so they do **not** measure classification accuracy.

## Optional AWS S3
```
pip install boto3
set ECOSORT_S3_BUCKET=your-private-bucket
set AWS_REGION=ap-south-1
aws configure
```
Backups and reports are then also uploaded under `ecosort/`. Keep the bucket private and use an IAM user or role
that can only list/put/get in that bucket. Hosting the app on a cloud service is a separate step you must configure yourself.

## Important limitations (state these in your report)
- MobileNetV2 was trained on ImageNet, not on waste. The category comes from a keyword mapping of ImageNet labels, so it
  works for clear single items (bottles, fruit, cartons, cans) and is unreliable for crushed, dirty or mixed waste.
  Always let the user confirm; the confirmed category is stored next to the prediction so accuracy can be measured later.
- **Improving accuracy:** fine-tune MobileNetV2 on a labelled waste dataset (e.g. TrashNet or TACO), save the model,
  and replace `WasteClassifier.predict` in `classifier.py` to load it.
- SQLite and local folders suit a single-instance demo. For a real deployment use HTTPS, a production server (not `python app.py`),
  and persistent storage for `instance/`.
- Run each feature yourself and capture screenshots before submitting.

## Files
`app.py` routes and security - `classifier.py` AI - `backup.py` backup/S3 - `templates/`, `static/` UI - `tests/` tests
