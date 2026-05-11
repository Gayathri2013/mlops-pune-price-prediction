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
