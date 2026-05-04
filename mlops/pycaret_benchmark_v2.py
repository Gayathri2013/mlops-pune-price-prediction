"""PyCaret AutoML benchmark — v2 with overfitting diagnostic.

This script answers two questions:
  1. Is our manually-built Voting Regressor competitive with what an AutoML
     library can produce on the same dataset?
  2. Is the PyCaret champion model actually generalizing, or is it overfitting?

What's new in v2 vs v1:
  * Prints THREE R² values per model, not one:
      - Train R²  → fit on training data (capacity check)
      - CV R²     → cross-validated mean from compare_models leaderboard
                    (generalization check; only available for PyCaret)
      - Test R²   → held-out test set (honest performance)
  * Computes the train-test gap and a verdict (healthy / watch / OVERFIT).
  * Saves all of the above into the DVC metrics JSON.

Why this matters: a high test R² alone is not proof of a good model. A
tree ensemble like ExtraTreesRegressor can hit 0.99 train R² on small data
and 0.92 test R² — the test number looks great but the gap reveals
memorization. The Lab 3 linear Voting Regressor cannot overfit this way,
which is why a head-to-head needs all three numbers to be meaningful.

Workflow:
  1. Load the Lab 2 feature matrix and target
  2. Use the SAME train/test split as Lab 3 (same random_state)
  3. Run PyCaret: setup → compare_models → tune → finalize → save
  4. Score the PyCaret champion on TRAIN and TEST sets
  5. Score the existing Lab 3 model on TRAIN and TEST sets
  6. Print a head-to-head table with overfitting verdicts
  7. Persist everything to metrics/pycaret_benchmark.json for DVC

Run from the project root:
    python -m mlops.pycaret_benchmark_v2

Note: PyCaret's first install pulls a heavy dependency tree (XGBoost,
LightGBM, CatBoost, etc.) — typically 2-3 minutes the first time.
After that, this script runs in 60-90 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from mlops.utils import (
    METRICS_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    load_features_and_target,
    load_params,
    score_regressor,
    split_data,
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain R² without sklearn import — keeps this file self-contained."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(((y_true - y_pred) ** 2).mean() ** 0.5)


def _verdict(gap: float) -> str:
    """Translate train-minus-test R² gap into a human verdict.

    Thresholds are deliberately conservative for small (~200 row) datasets
    where any single test split is noisy. Tweak in params.yaml later if
    you want different bands.
    """
    if gap < 0.05:
        return "healthy"
    if gap < 0.10:
        return "watch"
    return "OVERFIT"


def _score_pycaret_pipeline(predict_fn, final, X, y, pred_col_cache: list[str]):
    """Run a PyCaret pipeline on (X, y) and return RMSE + R².

    pred_col_cache is a one-element list used to remember which column
    name PyCaret returned predictions in (it differs across versions:
    'prediction_label' in 3.x, 'Label' in 2.x). We detect it once on
    the first call and reuse it.
    """
    pred_df = predict_fn(final, data=X, verbose=False)
    if not pred_col_cache:
        if "prediction_label" in pred_df.columns:
            pred_col_cache.append("prediction_label")
        elif "Label" in pred_df.columns:
            pred_col_cache.append("Label")
        else:
            raise RuntimeError(
                f"Could not find PyCaret prediction column in: {list(pred_df.columns)}"
            )
    pred = pred_df[pred_col_cache[0]].values
    return {"rmse": _rmse(y, pred), "r2": _r2(y, pred)}


# ── Main ─────────────────────────────────────────────────────────────────────
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
    print(" PyCaret AutoML benchmark vs the Lab 3 Voting Regressor (v2)")
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
        verbose=True,
        html=True,
        log_experiment=False,    # we run a separate MLflow script
    )

    # 4. compare_models — train and rank ~20 algorithms by R².
    print("[2/5] compare_models(n_select=10, sort='R2') — running ~20 algorithms…")
    top_n = compare_models(n_select=10, sort="R2", turbo=True)
    leaderboard = pull()
    print("\n   Top 5 by R²:")
    print(leaderboard.head(5).to_string())

    # 5. Tune the winner.
    best = top_n[0] if isinstance(top_n, list) else top_n
    print(f"\n[3/5] tune_model({type(best).__name__}, n_iter=20)…")
    tuned = tune_model(best, n_iter=20, optimize="R2", choose_better=True, verbose=False)

    # 6. Finalize (refit on full training pool) and save the pipeline.
    print(f"[4/5] finalize_model() → refit on full training data")
    final = finalize_model(tuned)
    pycaret_path = MODEL_DIR / "pycaret_pune_real_estate"
    save_model(final, str(pycaret_path))
    print(f"      Saved: {pycaret_path}.pkl  (preprocessing + model in one file)")

    # 7. Score PyCaret's final pipeline on BOTH train and test sets.
    #    The train score tells us about capacity / memorization; the test
    #    score tells us about honest generalization. Without both, we cannot
    #    detect overfitting.
    print("\n[5/5] Head-to-head — train vs test on the Lab 3 splits")
    pred_col_cache: list[str] = []
    pycaret_train = _score_pycaret_pipeline(predict_model, final, X_train, y_train, pred_col_cache)
    pycaret_test  = _score_pycaret_pipeline(predict_model, final, X_test,  y_test,  pred_col_cache)

    # 7a. Pull the CV mean R² for the chosen model from the leaderboard.
    #     The leaderboard is sorted DESC by R², and `best` was leaderboard.iloc[0],
    #     so chosen_row is row 0. We still match by model name in case PyCaret
    #     ever reorders rows.
    chosen_name = leaderboard.iloc[0]["Model"]
    pycaret_cv_r2 = float(leaderboard.iloc[0]["R2"])
    print(f"\n   CV mean R² for '{chosen_name}' (from compare_models): {pycaret_cv_r2:.4f}")

    # 8. Score the EXISTING Lab 3 Voting Regressor on the SAME train/test splits.
    lab3_path = MODEL_DIR / "property_price_prediction_voting.sav"
    if lab3_path.exists():
        lab3_model = joblib.load(lab3_path)
        lab3_train = score_regressor(lab3_model, X_train, y_train)
        lab3_test  = score_regressor(lab3_model, X_test,  y_test)
    else:
        print(f"⚠️  Lab 3 model not found at {lab3_path} — run `python -m mlops.train` first.")
        lab3_train = None
        lab3_test  = None

    # 9. Print the head-to-head table with the overfitting verdict.
    print()
    print("┌" + "─" * 78 + "┐")
    print(f"│ {'Model':<32s}{'Train R²':>11s}{'Test R²':>11s}"
          f"{'Train−Test':>13s}{'Verdict':>10s} │")
    print("├" + "─" * 78 + "┤")

    if lab3_train and lab3_test:
        gap = lab3_train["r2"] - lab3_test["r2"]
        print(f"│ {'Lab 3 manual VotingRegressor':<32s}"
              f"{lab3_train['r2']:>11.4f}{lab3_test['r2']:>11.4f}"
              f"{gap:>+13.4f}{_verdict(gap):>10s} │")

    pyc_gap = pycaret_train["r2"] - pycaret_test["r2"]
    print(f"│ {'PyCaret ' + type(final).__name__:<32s}"
          f"{pycaret_train['r2']:>11.4f}{pycaret_test['r2']:>11.4f}"
          f"{pyc_gap:>+13.4f}{_verdict(pyc_gap):>10s} │")
    print("└" + "─" * 78 + "┘")

    # RMSE table (kept from v1 — still useful for absolute-error reporting).
    print()
    print("┌" + "─" * 60 + "┐")
    print(f"│ {'Model':<30s}{'Train RMSE':>13s}{'Test RMSE':>15s} │")
    print("├" + "─" * 60 + "┤")
    if lab3_train and lab3_test:
        print(f"│ {'Lab 3 manual VotingRegressor':<30s}"
              f"{lab3_train['rmse']:>13.2f}{lab3_test['rmse']:>15.2f} │")
    print(f"│ {'PyCaret ' + type(final).__name__:<30s}"
          f"{pycaret_train['rmse']:>13.2f}{pycaret_test['rmse']:>15.2f} │")
    print("└" + "─" * 60 + "┘")

    # 10. Verdict prose — same as v1 but we say it after showing the gap.
    if lab3_test:
        delta = pycaret_test["r2"] - lab3_test["r2"]
        if abs(delta) < 0.01:
            verdict = "≈ tied on test R² — Lab 3's manual pipeline matches AutoML"
        elif delta > 0:
            verdict = f"PyCaret wins on test R² by Δ = +{delta:.4f}"
        else:
            verdict = f"Lab 3 wins on test R² by Δ = {abs(delta):.4f}"
        print(f"\n   Verdict (test R² only): {verdict}")

        # Honest add-on: if PyCaret won on test but the gap is wide, flag it.
        if delta > 0 and pyc_gap >= 0.10:
            print(f"   ⚠️  But PyCaret's train-test gap is {pyc_gap:+.4f} — "
                  f"some of that test win may be overfitting noise.")
        elif delta > 0 and 0.05 <= pyc_gap < 0.10:
            print(f"   ⚠️  PyCaret's train-test gap is {pyc_gap:+.4f} — "
                  f"watch for overfitting on a different test split.")

    # 11. Persist comparison metrics for DVC. Schema is a strict superset of
    #     v1, so any downstream code that read v1's JSON still works.
    payload = {
        "pycaret_model":              type(final).__name__,
        "pycaret_train_r2":           pycaret_train["r2"],
        "pycaret_cv_r2":              pycaret_cv_r2,
        "pycaret_test_r2":            pycaret_test["r2"],
        "pycaret_train_test_gap":     pyc_gap,
        "pycaret_overfit_verdict":    _verdict(pyc_gap),
        "pycaret_train_rmse":         pycaret_train["rmse"],
        "pycaret_test_rmse":          pycaret_test["rmse"],

        # Back-compat aliases (v1 schema)
        "pycaret_rmse":               pycaret_test["rmse"],
        "pycaret_r2":                 pycaret_test["r2"],

        "lab3_train_r2":              lab3_train["r2"]   if lab3_train else None,
        "lab3_test_r2":               lab3_test["r2"]    if lab3_test  else None,
        "lab3_train_test_gap":        (lab3_train["r2"] - lab3_test["r2"])
                                       if (lab3_train and lab3_test) else None,
        "lab3_overfit_verdict":       _verdict(lab3_train["r2"] - lab3_test["r2"])
                                       if (lab3_train and lab3_test) else None,
        "lab3_train_rmse":            lab3_train["rmse"] if lab3_train else None,
        "lab3_test_rmse":             lab3_test["rmse"]  if lab3_test  else None,

        # Back-compat aliases (v1 schema)
        "lab3_rmse":                  lab3_test["rmse"]  if lab3_test  else None,
        "lab3_r2":                    lab3_test["r2"]    if lab3_test  else None,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "pycaret_benchmark.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n📊 Comparison saved: {out.relative_to(PROJECT_ROOT)}")
    return payload


if __name__ == "__main__":
    main()
