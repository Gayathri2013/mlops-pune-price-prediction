"""MLOps lab scripts for the Pune Price Prediction project.

Each module is runnable standalone from the project root:
    python -m mlops.train
    python -m mlops.mlflow_train
    python -m mlops.mlflow_sweep
    python -m mlops.mlflow_query
    python -m mlops.pycaret_benchmark
"""

# Force UTF-8 on stdout/stderr. On Windows under DVC (or any non-TTY pipe),
# Python defaults to cp1252, which crashes on the → / ✅ / ₹ characters used
# throughout these scripts. Reconfiguring here covers every `python -m mlops.x`
# entry point in one place.
import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# Load .env from the project root (one level up from this package) so
# DAGSHUB_USER / DAGSHUB_TOKEN / MLFLOW_TRACKING_URI etc. don't need to be
# exported manually each shell. Real env vars take precedence over .env.
# python-dotenv is optional — if missing, .env is silently ignored.
try:
    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# If DAGSHUB_* credentials are present and MLFLOW_TRACKING_URI is NOT,
# auto-derive the MLflow URI + basic-auth vars so every `python -m mlops.x`
# logs to DagsHub instead of the local SQLite DB. To force local logging
# despite DagsHub creds, set MLFLOW_TRACKING_URI explicitly (in .env or shell).
import os as _os
if (
    not _os.environ.get("MLFLOW_TRACKING_URI")
    and _os.environ.get("DAGSHUB_USER")
    and _os.environ.get("DAGSHUB_TOKEN")
    and _os.environ.get("DAGSHUB_REPO")
):
    _os.environ["MLFLOW_TRACKING_URI"] = (
        f"https://dagshub.com/{_os.environ['DAGSHUB_USER']}/"
        f"{_os.environ['DAGSHUB_REPO']}.mlflow"
    )
    _os.environ.setdefault("MLFLOW_TRACKING_USERNAME", _os.environ["DAGSHUB_USER"])
    _os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", _os.environ["DAGSHUB_TOKEN"])
