"""Sweep Ridge alphas — one MLflow run per alpha value.

For each alpha in params.yaml `sweep.alphas`, train a fresh Voting Regressor
(LR + Ridge(alpha) + Lasso(default)), score it on the held-out test set,
and log the run to MLflow. After this script finishes you can:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

…sort by `metrics.r2` to find the best alpha, or use the parallel-coordinates
plot to see the alpha-vs-R² curve at a glance.

Run from the project root:
    python -m mlops.mlflow_sweep
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from mlops.utils import (
    PROJECT_ROOT,
    build_voting_regressor,
    load_features_and_target,
    load_params,
    score_regressor,
    split_data,
)


def main() -> list[dict]:
    """Run one MLflow run per alpha. Returns a list of summary dicts."""
    params = load_params()

    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    X, y = load_features_and_target()
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )

    alphas = params["sweep"]["alphas"]
    lasso_alpha = params["lasso"]["alpha"]
    lasso_max_iter = params["lasso"]["max_iter"]

    print(f"Sweeping {len(alphas)} ridge alphas: {alphas}")
    print(f"(Lasso alpha pinned at {lasso_alpha} for the sweep — only Ridge varies)")
    print()

    results = []
    for alpha in alphas:
        with mlflow.start_run(run_name=f"sweep_ridge_alpha_{alpha}") as run:
            model = build_voting_regressor(alpha, lasso_alpha, lasso_max_iter)
            model.fit(X_train, y_train)
            metrics = score_regressor(model, X_test, y_test)

            mlflow.log_params({
                "model_type": "VotingRegressor",
                "ridge_alpha": alpha,
                "lasso_alpha": lasso_alpha,
                "n_features": X_train.shape[1],
            })
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"module": "M2_Lab5", "sweep": "ridge_alpha"})

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, name="model", signature=signature)

            row = {"alpha": alpha, **metrics, "run_id": run.info.run_id}
            results.append(row)
            print(f"  α={alpha:<8}  RMSE = ₹{metrics['rmse']:6.2f}L   "
                  f"R² = {metrics['r2']:.4f}   run = {run.info.run_id[:8]}")

    # Best by R²
    best = max(results, key=lambda r: r["r2"])
    print()
    print(f"⭐ Best ridge alpha: {best['alpha']}  →  R² = {best['r2']:.4f}")
    print(f"   Run ID: {best['run_id']}")
    print()
    # print("Inspect the full sweep with:  mlflow ui --backend-store-uri file:./mlruns")
    return results


if __name__ == "__main__":
    main()
