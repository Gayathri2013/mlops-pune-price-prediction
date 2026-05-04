"""Train the Voting Regressor — the script DVC reproduces.

This is the "production trainer". It reads params.yaml, loads the Lab 2
feature matrix, trains the Lab 3 Voting Regressor, computes a 95%
prediction interval from training residuals, and writes:

  model/property_price_prediction_voting.sav
  model/interval_est.pkl
  metrics/train_metrics.json

Run from the project root:
    python -m mlops.train

This script has NO MLflow logging on purpose — keep it minimal. For
experiment tracking, use mlops/mlflow_train.py instead. DVC tracks the
artifacts; MLflow tracks the experiments.
"""

from __future__ import annotations

from mlops.utils import (
    compute_interval_estimate,
    load_features_and_target,
    load_params,
    save_model_artifacts,
    score_regressor,
    split_data,
    train_voting_regressor,
    write_metrics_json,
)


def main() -> dict[str, float]:
    """End-to-end training run. Returns the test-set metric dict."""
    print("=" * 70)
    print(" Pune Real Estate — Voting Regressor training run")
    print("=" * 70)

    # 1. Read config.
    params = load_params()
    print(f"\n[1/5] Loaded params.yaml")
    print(f"      ridge.alpha   = {params['ridge']['alpha']}")
    print(f"      lasso.alpha   = {params['lasso']['alpha']}")
    print(f"      test_size     = {params['data']['test_size']}")
    print(f"      random_state  = {params['data']['random_state']}")

    # 2. Load data.
    X, y = load_features_and_target()
    print(f"\n[2/5] Loaded data: X = {X.shape}, y = {y.shape}")

    # 3. Split + train.
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )
    print(f"\n[3/5] Train/test split: {len(X_train)} / {len(X_test)} rows")

    model = train_voting_regressor(X_train, y_train, params)
    print(f"      Fit complete: {type(model).__name__}")

    # 4. Score on test + derive interval from training residuals.
    test_metrics = score_regressor(model, X_test, y_test)
    train_pred = model.predict(X_train)
    interval_est = compute_interval_estimate(
        y_train, train_pred, confidence=params["interval"]["confidence"]
    )
    print(f"\n[4/5] Test set metrics:")
    for key, val in test_metrics.items():
        print(f"        {key:5s} = {val:.4f}")
    margin = interval_est["z_score"] * interval_est["residual_std"]
    print(f"      95% prediction interval: ±₹{margin:.2f}L")

    # 5. Persist. The filenames here match what src/inference.py loads,
    #    so the FastAPI app picks up the new model the moment it restarts.
    paths = save_model_artifacts(model, interval_est)
    metrics_payload = {
        **test_metrics,
        "ridge_alpha": params["ridge"]["alpha"],
        "lasso_alpha": params["lasso"]["alpha"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "interval_margin": margin,
    }
    metrics_path = write_metrics_json(metrics_payload, name="train_metrics.json")

    print(f"\n[5/5] Artifacts written:")
    print(f"        {paths['model'].relative_to(paths['model'].parent.parent)}")
    print(f"        {paths['interval'].relative_to(paths['interval'].parent.parent)}")
    print(f"        {metrics_path.relative_to(metrics_path.parent.parent)}")
    print("\n✅ Done. Restart the FastAPI app to serve the new model.")
    return test_metrics


if __name__ == "__main__":
    main()
