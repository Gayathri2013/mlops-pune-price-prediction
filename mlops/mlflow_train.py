"""Train the Voting Regressor with full MLflow tracking.

Same training logic as mlops/train.py, but every parameter, metric, plot,
and the model itself is logged to MLflow. Use this when you want a
permanent record of the run; use train.py when you just need the artifact.

Run from the project root:
    python -m mlops.mlflow_train

View results:
    mlflow ui --backend-store-uri file:./mlruns
    open http://localhost:5000

By default this logs to ./mlruns. To log to DagsHub instead, run:
    python -m mlops.dagshub_setup --check
beforehand (it sets MLFLOW_TRACKING_URI from your env vars).
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


def _make_diagnostic_plot(model, X_test, y_test, out_path: Path) -> Path:
    """Two-panel diagnostic: residuals + predicted-vs-actual.

    Saved to disk so we can attach it to the MLflow run as an artifact.
    """
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


def main() -> str:
    """Run training inside a tracked MLflow run. Returns the MLflow run_id."""
    params = load_params()

    # If MLFLOW_TRACKING_URI is set in the environment (e.g. by
    # mlops.dagshub_setup) we honour it. Otherwise, default to a local SQLite
    # store. SQLite is the recommended local backend in MLflow 3.x — the older
    # file:./mlruns store is being deprecated. mlflow.db is one file you can
    # delete to reset, and `mlflow ui --backend-store-uri sqlite:///mlflow.db`
    # will pick it up.
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment   : {params['mlflow']['experiment_name']}")

    X, y = load_features_and_target()
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )

    with mlflow.start_run(run_name="voting_regressor") as run:
        # 1. Train.
        model = train_voting_regressor(X_train, y_train, params)

        # 2. Log every hyperparameter that could affect the result.
        mlflow.log_params({
            "model_type": "VotingRegressor",
            "estimators": "LinearRegression+Ridge+Lasso",
            "ridge_alpha": params["ridge"]["alpha"],
            "lasso_alpha": params["lasso"]["alpha"],
            "lasso_max_iter": params["lasso"]["max_iter"],
            "test_size": params["data"]["test_size"],
            "random_state": params["data"]["random_state"],
            "n_features": X_train.shape[1],
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        # 3. Score on test set + log all metrics.
        metrics = score_regressor(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # 4. Derive the prediction interval and log its width as a metric.
        train_pred = model.predict(X_train)
        interval_est = compute_interval_estimate(
            y_train, train_pred, confidence=params["interval"]["confidence"]
        )
        mlflow.log_metric(
            "interval_margin", interval_est["z_score"] * interval_est["residual_std"]
        )

        # 5. Tags (free-form key/value, useful for filtering runs in the UI).
        mlflow.set_tags({
            "module": "M2_Lab5",
            "dataset": "Pune Real Estate v1",
            "candidate_for_registry": "true",
        })

        # 6. Diagnostic plot as an artifact.
        with tempfile.TemporaryDirectory() as tmp:
            plot_path = _make_diagnostic_plot(model, X_test, y_test,
                                              Path(tmp) / "diagnostics.png")
            mlflow.log_artifact(str(plot_path), artifact_path="diagnostics")

        # 7. Log the model with a signature + sample input. The signature is
        #    what enables one-click `mlflow models serve` later.
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",                      # MLflow 3.x param (was artifact_path)
            signature=signature,
            input_example=X_train.iloc[:2],
        )

        # 8. Save the same artifacts to model/ so the FastAPI app can serve
        #    this run immediately. (DVC will pick this up too.)
        save_model_artifacts(model, interval_est)

        run_id = run.info.run_id
        print()
        print(f"✅ Run logged: {run_id}")
        print(f"   RMSE = ₹{metrics['rmse']:.2f}L | MAE = ₹{metrics['mae']:.2f}L | R² = {metrics['r2']:.4f}")
        print(f"   Model URI: runs:/{run_id}/model")
        return run_id


if __name__ == "__main__":
    main()
