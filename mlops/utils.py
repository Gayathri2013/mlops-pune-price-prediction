"""Shared utilities for every MLOps script in this project.

Functional only — no classes, in line with project convention.

The single source of truth for:
  * how features and target are loaded
  * how a regressor is scored
  * how the Lab-3 Voting Regressor is built
  * how a 95% prediction interval is derived from training residuals
  * how params.yaml is read

Everything in mlops/*.py imports from here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Paths (resolved relative to project root) ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "model_features.csv"
TARGET_NPY = PROJECT_ROOT / "model_target.npy"
MODEL_DIR = PROJECT_ROOT / "model"
METRICS_DIR = PROJECT_ROOT / "metrics"
PARAMS_YAML = PROJECT_ROOT / "params.yaml"


# ── Config loading ────────────────────────────────────────────────────────────
def load_params() -> dict[str, Any]:
    """Read params.yaml. Falls back to sensible defaults so any script can run
    even if params.yaml is missing (e.g. a learner copies just one .py file)."""
    defaults = {
        "data": {"test_size": 0.2, "random_state": 42},
        "ridge": {"alpha": 1.0},
        "lasso": {"alpha": 0.1, "max_iter": 10_000},
        "voting": {"weights": None},
        "interval": {"confidence": 0.95},
        "sweep": {"alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
        "mlflow": {"experiment_name": "M2_Pune_Real_Estate_Price"},
    }
    if PARAMS_YAML.exists():
        with open(PARAMS_YAML) as fh:
            user = yaml.safe_load(fh) or {}
        # Deep-merge: user values override defaults at the leaf level.
        for top_key, top_val in user.items():
            if isinstance(top_val, dict) and top_key in defaults:
                defaults[top_key].update(top_val)
            else:
                defaults[top_key] = top_val
    return defaults


# ── Data loading ──────────────────────────────────────────────────────────────
def load_features_and_target() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the Lab 2 feature matrix and target vector from project root.

    Raises a clear, actionable error if either is missing.
    """
    missing = []
    if not FEATURES_CSV.exists():
        missing.append(str(FEATURES_CSV.name))
    if not TARGET_NPY.exists():
        missing.append(str(TARGET_NPY.name))
    if missing:
        raise FileNotFoundError(
            f"Missing required input file(s): {', '.join(missing)}\n"
            "These are produced by Module 2, Lab 2 (NLP_Feature_Engineering).\n"
            "Re-run Lab 2 and copy the outputs to the project root, or "
            "`dvc pull` if the project is connected to a DVC remote."
        )
    X = pd.read_csv(FEATURES_CSV)
    y = np.load(TARGET_NPY)
    return X, y


def split_data(
    X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Train/test split — same parameters Lab 3 uses, so any score from any
    script in this project is directly comparable to Lab 3's reported numbers."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_regressor(model, X_eval: pd.DataFrame, y_eval: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, R² on the eval set. Returns plain floats so the
    dict can be JSON-serialized for DVC metrics or MLflow logging."""
    pred = model.predict(X_eval)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_eval, pred))),
        "mae": float(mean_absolute_error(y_eval, pred)),
        "r2": float(r2_score(y_eval, pred)),
    }


# ── Training ──────────────────────────────────────────────────────────────────
def build_voting_regressor(ridge_alpha: float, lasso_alpha: float, lasso_max_iter: int = 10_000):
    """Construct the Lab-3 Voting Regressor (LR + Ridge + Lasso).

    This is the exact estimator that Lab 3 trained and that src/inference.py
    serves. Wrapping it here means any future hyperparameter change happens
    in params.yaml only — no code edits needed.
    """
    return VotingRegressor(
        estimators=[
            ("lr", LinearRegression()),
            ("ridge", Ridge(alpha=ridge_alpha)),
            ("lasso", Lasso(alpha=lasso_alpha, max_iter=lasso_max_iter)),
        ]
    )


def train_voting_regressor(X_train, y_train, params: dict[str, Any]):
    """Fit the Voting Regressor with hyperparameters from params.yaml."""
    model = build_voting_regressor(
        ridge_alpha=params["ridge"]["alpha"],
        lasso_alpha=params["lasso"]["alpha"],
        lasso_max_iter=params["lasso"]["max_iter"],
    )
    model.fit(X_train, y_train)
    return model


# ── Confidence interval (Lab 3, §9) ───────────────────────────────────────────
def compute_interval_estimate(
    y_true: np.ndarray, y_pred: np.ndarray, confidence: float = 0.95
) -> dict[str, float]:
    """Parametric prediction-interval width derived from training residuals.

    Returns the same dict shape that src/inference.py expects to load from
    `model/interval_est.pkl`, so re-training swaps the file in place.
    """
    residuals = y_true - y_pred
    residual_std = float(np.std(residuals))
    z_score = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    return {"z_score": z_score, "residual_std": residual_std, "pi": confidence}


# ── Persistence ───────────────────────────────────────────────────────────────
def save_model_artifacts(model, interval_est: dict, model_dir: Path = MODEL_DIR) -> dict[str, Path]:
    """Persist the trained model + interval to disk.

    Note: we save into the SAME filenames that src/inference.py loads. That
    way a fresh `python -m mlops.train` run produces a model the FastAPI app
    can serve immediately — no glue code, no name translation.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "property_price_prediction_voting.sav"
    interval_path = model_dir / "interval_est.pkl"
    joblib.dump(model, model_path)
    joblib.dump(interval_est, interval_path)
    return {"model": model_path, "interval": interval_path}


def write_metrics_json(metrics: dict, name: str = "train_metrics.json") -> Path:
    """Write a flat JSON file under metrics/ that DVC can read with
    `dvc metrics show`. Anything in the metrics dict will appear in the
    DVC metrics dashboard."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / name
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    return path
