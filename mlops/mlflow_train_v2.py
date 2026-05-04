"""Train the Voting Regressor with MLflow tracking — v2.

================================================================================
 VERSION HISTORY
================================================================================

v1 (mlflow_train.py)
--------------------
  * Logged hyperparameters, test metrics (RMSE/MAE/R²), and the model.
  * Logged a residuals + predicted-vs-actual diagnostic plot.
  * Used a model signature inferred from training data.
  * Limitation: only test-set metrics. A run with test R² = 0.85 and a
    train R² of 0.86 (healthy) looks identical in the UI to a run with
    train R² = 0.99 (overfit) — same row, same column, no warning.

v2 (this version) — overfitting + registry
-------------------------------------------
  * NEW: scores on the TRAIN set as well, logs train_rmse / train_mae /
    train_r2. Same lesson as pycaret_benchmark v2 — without train metrics
    you can't see overfitting in the UI.
  * NEW: logs `train_test_r2_gap` as a metric. Sort by it in the MLflow UI
    to spot overfit runs at a glance.
  * NEW: prints a verdict (healthy / watch / OVERFIT) based on the gap.
  * NEW: registers the trained model in the MLflow Model Registry under a
    stable name. After a few runs you can promote the best one to "Staging"
    or "Production" via the UI, and the FastAPI app can load it by name
    instead of by file path. This is the bridge from "tracked experiments"
    to "deployed models" — the headline benefit of MLflow over plain
    tracking.

================================================================================
 USAGE
================================================================================

Run from the project root:
    python -m mlops.mlflow_train_v2

View results:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    open http://localhost:5000

By default this logs to ./mlflow.db (SQLite). To log to DagsHub instead:
    python -m mlops.dagshub_setup --check
beforehand (it sets MLFLOW_TRACKING_URI from your env vars).

Promoting a run to Production via the UI:
    1. Open the registered model `pune_price_voting_regressor`
    2. Click the version you want to promote
    3. Set its stage to "Production"
    4. In FastAPI: load with `mlflow.pyfunc.load_model("models:/pune_price_voting_regressor/Production")`
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from mlops.utils import (
    PROJECT_ROOT,
    compute_interval_estimate,
    load_features_and_target,
    load_params,
    save_model_artifacts,
    score_regressor,
    split_data,
    train_voting_regressor,
)

REGISTERED_MODEL_NAME = "pune_price_voting_regressor"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _verdict(gap: float) -> str:
    """Same thresholds as pycaret_benchmark_v2 — keep the curriculum consistent."""
    if gap < 0.05:
        return "healthy"
    if gap < 0.10:
        return "watch"
    return "OVERFIT"


def _make_diagnostic_plot(model, X_test, y_test, out_path: Path) -> Path:
    """Two-panel diagnostic: residuals + predicted-vs-actual."""
    pred = model.predict(X_test)
    resid = y_test - pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(pred, resid, alpha=0.6, color="#2563eb")
    axes[0].axhline(0, color="#f43f5e", ls="--", lw=1.5)
    axes[0].set_xlabel("Predicted Price (₹L)")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    axes[0].set_title("Residuals vs Predicted")
    axes[1].scatter(y_test, pred, alpha=0.6, color="#10b981")
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    axes[1].plot(lims, lims, color="#f43f5e", ls="--", lw=1.5, label="y = x")
    axes[1].set_xlabel("Actual Price (₹L)")
    axes[1].set_ylabel("Predicted Price (₹L)")
    axes[1].set_title("Predicted vs Actual")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> str:
    """Run training inside a tracked MLflow run. Returns the MLflow run_id."""
    params = load_params()

    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment   : {params['mlflow']['experiment_name']}")
    print(f"Registered model    : {REGISTERED_MODEL_NAME}")

    X, y = load_features_and_target()
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )

    with mlflow.start_run(run_name="voting_regressor") as run:
        # 1. Train.
        model = train_voting_regressor(X_train, y_train, params)

        # 2. Hyperparameters + run metadata.
        mlflow.log_params({
            "model_type":     "VotingRegressor",
            "estimators":     "LinearRegression+Ridge+Lasso",
            "ridge_alpha":    params["ridge"]["alpha"],
            "lasso_alpha":    params["lasso"]["alpha"],
            "lasso_max_iter": params["lasso"]["max_iter"],
            "test_size":      params["data"]["test_size"],
            "random_state":   params["data"]["random_state"],
            "n_features":     X_train.shape[1],
            "n_train":        len(X_train),
            "n_test":         len(X_test),
        })

        # 3. Score on TRAIN and TEST (v2 addition).
        train_metrics = score_regressor(model, X_train, y_train)
        test_metrics  = score_regressor(model, X_test,  y_test)
        gap = train_metrics["r2"] - test_metrics["r2"]
        verdict = _verdict(gap)

        # Log every metric with a clear prefix so the MLflow UI groups them.
        mlflow.log_metrics({
            "train_rmse":         train_metrics["rmse"],
            "train_mae":          train_metrics["mae"],
            "train_r2":           train_metrics["r2"],
            "test_rmse":          test_metrics["rmse"],
            "test_mae":           test_metrics["mae"],
            "test_r2":            test_metrics["r2"],
            "train_test_r2_gap":  gap,
            # Back-compat aliases — v1 logged these names; preserve for old queries.
            "rmse":               test_metrics["rmse"],
            "mae":                test_metrics["mae"],
            "r2":                 test_metrics["r2"],
        })

        # 4. Prediction interval (unchanged from v1).
        train_pred = model.predict(X_train)
        interval_est = compute_interval_estimate(
            y_train, train_pred, confidence=params["interval"]["confidence"]
        )
        mlflow.log_metric(
            "interval_margin", interval_est["z_score"] * interval_est["residual_std"]
        )

        # 5. Tags — including the v2 verdict so you can filter by it in the UI.
        mlflow.set_tags({
            "module":                  "M2_Lab5",
            "dataset":                 "Pune Real Estate v1",
            "candidate_for_registry":  "true",
            "overfit_verdict":         verdict,
        })

        # 6. Diagnostic plot.
        with tempfile.TemporaryDirectory() as tmp:
            plot_path = _make_diagnostic_plot(model, X_test, y_test,
                                              Path(tmp) / "diagnostics.png")
            mlflow.log_artifact(str(plot_path), artifact_path="diagnostics")

        # 7. Log AND register the model. `registered_model_name` is the v2
        #    addition — MLflow auto-creates the registered model on first
        #    run and adds a new version on each subsequent run. After the
        #    first run, open the UI's "Models" tab to see it.
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.iloc[:2],
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # 8. Save artifacts to model/ for the FastAPI app.
        save_model_artifacts(model, interval_est)

        run_id = run.info.run_id
        print()
        print(f"✅ Run logged: {run_id}")
        print(f"   Train R²: {train_metrics['r2']:.4f}  |  "
              f"Test R²: {test_metrics['r2']:.4f}  |  "
              f"Gap: {gap:+.4f}  ({verdict})")
        print(f"   Test RMSE: ₹{test_metrics['rmse']:.2f}L  |  "
              f"Test MAE: ₹{test_metrics['mae']:.2f}L")
        print(f"   Model URI:   runs:/{run_id}/model")
        print(f"   Registry:    models:/{REGISTERED_MODEL_NAME}/<latest version>")
        print()
        print("   To promote this run to Production:")
        print(f"     1. mlflow ui --backend-store-uri sqlite:///{(PROJECT_ROOT / 'mlflow.db').name}")
        print(f"     2. Models tab → {REGISTERED_MODEL_NAME} → set stage to 'Production'")
        return run_id


if __name__ == "__main__":
    main()
