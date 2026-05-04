"""PyCaret AutoML benchmark — compares against the Lab 3 Voting Regressor.

This script answers: "Is our manually-built Voting Regressor competitive
with what an AutoML library can produce on the same dataset?"

Workflow:
  1. Load the Lab 2 feature matrix and target
  2. Use the SAME train/test split as Lab 3 (same random_state)
  3. Run PyCaret: setup → compare_models → tune → finalize → save
  4. Score the PyCaret champion on Lab 3's test set
  5. Score the existing model/property_price_prediction_voting.sav on the same set
  6. Print a head-to-head table

Run from the project root:
    python -m mlops.pycaret_benchmark

Note: PyCaret's first install pulls a heavy dependency tree (XGBoost,
LightGBM, CatBoost, etc.) — typically 2-3 minutes the first time.
After that, this script runs in 60-90 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from mlops.utils import (
    METRICS_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    load_features_and_target,
    load_params,
    score_regressor,
    split_data,
)


def main() -> dict:
    # PyCaret is imported inside main() so a learner who hasn't installed it
    # gets a clean error here, not at import time.
    try:
        from pycaret.regression import (  # noqa: F401
            compare_models, finalize_model, predict_model, pull,
            save_model, setup, tune_model,
        )
    except ImportError as exc:
        raise SystemExit(
            "PyCaret is not installed. Run:\n"
            "    pip install -r requirements-mlops.txt\n"
            "or:\n"
            "    pip install pycaret"
        ) from exc

    params = load_params()
    print("=" * 70)
    print(" PyCaret AutoML benchmark vs the Lab 3 Voting Regressor")
    print("=" * 70)

    # 1. Load data + same split as Lab 3.
    X, y = load_features_and_target()
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )
    print(f"\nLoaded data: train = {len(X_train)}, test = {len(X_test)}")

    # 2. PyCaret needs a single dataframe with the target as a column.
    train_df = X_train.copy()
    train_df["price"] = y_train

    # 3. setup() — PyCaret's preprocessing entry point.
    print("\n[1/5] PyCaret setup() — preprocessing pipeline initializing…")
    setup(
        data=train_df,
        target="price",
        session_id=params["data"]["random_state"],
        train_size=0.8,
        normalize=True,
        verbose=False,
        html=False,
        log_experiment=False,    # we run a separate MLflow script
    )

    # 4. compare_models — train and rank ~20 algorithms by R².
    print("[2/5] compare_models(n_select=3, sort='R2') — running ~20 algorithms…")
    top_3 = compare_models(n_select=10, sort="R2", turbo=True)
    leaderboard = pull()
    print("\n   Top 5 by R²:")
    print(leaderboard.head(5).to_string())

    # 5. Tune the winner.
    best = top_3[0] if isinstance(top_3, list) else top_3
    print(f"\n[3/5] tune_model({type(best).__name__}, n_iter=20)…")
    tuned = tune_model(best, n_iter=20, optimize="R2", choose_better=True, verbose=False)

    # 6. Finalize (refit on full training pool) and save the pipeline.
    print(f"[4/5] finalize_model() → refit on full training data")
    final = finalize_model(tuned)
    pycaret_path = MODEL_DIR / "pycaret_pune_real_estate"
    save_model(final, str(pycaret_path))
    print(f"      Saved: {pycaret_path}.pkl  (preprocessing + model in one file)")

    # 7. Score PyCaret's final pipeline on Lab 3's test set.
    print("\n[5/5] Head-to-head on the Lab 3 test set")
    pycaret_pred_df = predict_model(final, data=X_test, verbose=False)
    pred_col = "prediction_label" if "prediction_label" in pycaret_pred_df.columns else "Label"
    pycaret_pred = pycaret_pred_df[pred_col].values
    pycaret_metrics = {
        "rmse": float(((y_test - pycaret_pred) ** 2).mean() ** 0.5),
        "r2":   float(1 - ((y_test - pycaret_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()),
    }

    # 8. Score the EXISTING Lab 3 Voting Regressor on the SAME test set.
    lab3_path = MODEL_DIR / "property_price_prediction_voting.sav"
    if lab3_path.exists():
        lab3_model = joblib.load(lab3_path)
        lab3_metrics = score_regressor(lab3_model, X_test, y_test)
    else:
        print(f"⚠️  Lab 3 model not found at {lab3_path} — run `python -m mlops.train` first.")
        lab3_metrics = None

    # 9. Print + save the comparison.
    print()
    print("┌" + "─" * 60 + "┐")
    print(f"│ {'Model':<30s}{'RMSE (₹L)':>13s}{'R²':>15s} │")
    print("├" + "─" * 60 + "┤")
    if lab3_metrics:
        print(f"│ {'Lab 3 manual VotingRegressor':<30s}"
              f"{lab3_metrics['rmse']:>13.2f}{lab3_metrics['r2']:>15.4f} │")
    print(f"│ {'PyCaret final ' + type(final).__name__:<30s}"
          f"{pycaret_metrics['rmse']:>13.2f}{pycaret_metrics['r2']:>15.4f} │")
    print("└" + "─" * 60 + "┘")

    if lab3_metrics:
        delta = pycaret_metrics["r2"] - lab3_metrics["r2"]
        if abs(delta) < 0.01:
            verdict = "≈ tied — Lab 3's manual pipeline matches AutoML"
        elif delta > 0:
            verdict = f"PyCaret wins by ΔR² = +{delta:.4f}"
        else:
            verdict = f"Lab 3 wins by ΔR² = {abs(delta):.4f}"
        print(f"\n   Verdict: {verdict}")

    # Persist comparison metrics for DVC.
    payload = {
        "pycaret_model": type(final).__name__,
        "pycaret_rmse": pycaret_metrics["rmse"],
        "pycaret_r2":   pycaret_metrics["r2"],
        "lab3_rmse":    lab3_metrics["rmse"] if lab3_metrics else None,
        "lab3_r2":      lab3_metrics["r2"]   if lab3_metrics else None,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "pycaret_benchmark.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n📊 Comparison saved: {out.relative_to(PROJECT_ROOT)}")
    return payload


if __name__ == "__main__":
    main()
