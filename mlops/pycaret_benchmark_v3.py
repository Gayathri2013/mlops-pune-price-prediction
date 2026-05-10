"""PyCaret AutoML benchmark — v3 with deployment refit on full data.

================================================================================
 VERSION HISTORY
================================================================================

v1 (original)
-------------
  * Ran PyCaret: setup → compare_models → tune → finalize → save
  * Scored the finalized PyCaret pipeline on the Lab 3 test set ONLY
  * Compared that test R² against the Lab 3 manual VotingRegressor
  * Persisted the comparison to metrics/pycaret_benchmark.json
  * Limitation: a single test R² number can hide overfitting. A tree
    ensemble like ExtraTrees can hit 0.99 train R² and 0.92 test R²;
    the test number alone won't tell you the gap.

v2 (overfitting diagnostic)
---------------------------
  * Added Train R² scoring for both PyCaret and Lab 3 models
  * Pulled CV mean R² for the chosen PyCaret model from the leaderboard
  * Computed train-test gap and printed a verdict: healthy / watch / OVERFIT
  * Added a second-line warning: if PyCaret wins on test R² but its train-test
    gap is wide, flagged that some of the win may be overfitting noise
  * Expanded metrics/pycaret_benchmark.json with all three R² values per model
    (back-compatible: v1 keys are kept as aliases)

v3 (this version) — deployment refit
-------------------------------------
  * NEW: After scoring on the test set, retrains the final PyCaret pipeline
    on the FULL dataset (train + test combined) and saves it separately as
    the deployment artifact.
  * Why: the test set has done its one job — it gave us an honest performance
    estimate. We don't need it to be held out anymore. The deployed model
    should learn from every row of signal we have, including those 40 rows.
  * Discipline: no re-tuning, no algorithm change. Same tuned hyperparameters,
    just refit on more data. The reported test R² becomes our "expected
    production performance" for the deployment model.
  * Two model files are saved:
      - pycaret_pune_real_estate.pkl         (scored on test; for audit)
      - pycaret_pune_real_estate_DEPLOY.pkl  (refit on all data; for FastAPI)
    The naming makes the distinction unambiguous: anyone asking "which model
    produced the 0.92 test score?" gets a clear answer.
  * The deployment artifact path is added to the metrics JSON.

================================================================================
 THE LIFECYCLE THIS SCRIPT DEMONSTRATES
================================================================================

  1. Split           → train (159) / test (40)
  2. Select          → compare_models, tune_model on the 159
  3. Finalize        → refit best model on full 159-row training pool
  4. Score           → predict on the 40-row test set ONCE; record the number
  5. Deploy refit    → with all decisions frozen, refit the same configuration
                       on all 199 rows for the artifact that ships
  6. Save two models → audit copy + deployment copy

The rule that makes this valid: NO decisions are made after step 4. If the
test number disappoints, we do NOT try a new algorithm or re-tune — that
would contaminate the test set. We commit to the result and ship.

================================================================================
 USAGE
================================================================================

Run from the project root:
    python -m mlops.pycaret_benchmark_v3

The deployment artifact is written to:
    model/pycaret_pune_real_estate_DEPLOY.pkl

To serve it from the existing FastAPI app, point inference.py at this file
instead of the Lab 3 .sav file (or do a side-by-side A/B).

Note: PyCaret's first install pulls a heavy dependency tree (XGBoost,
LightGBM, CatBoost, etc.) — typically 2-3 minutes the first time.
After that, this script runs in 90-120 seconds (slightly longer than
v1/v2 because of the second setup + finalize on the full dataset).
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
    where any single test split is noisy.
    """
    if gap < 0.05:
        return "healthy"
    if gap < 0.10:
        return "watch"
    return "OVERFIT"


def _score_pycaret_pipeline(predict_fn, model, X, y, pred_col_cache: list[str]):
    """Run a PyCaret pipeline on (X, y) and return RMSE + R².

    pred_col_cache is a one-element list used to remember which column
    name PyCaret returned predictions in (it differs across versions:
    'prediction_label' in 3.x, 'Label' in 2.x). We detect it once on
    the first call and reuse it.
    """
    pred_df = predict_fn(model, data=X, verbose=False)
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


def _patch_pycaret_mlflow_compat() -> None:
    # Newer MLflow turned _active_run_stack into a ThreadLocalVariable and no
    # longer pushes onto it from mlflow.start_run. PyCaret's mlflow_logger still
    # treats it as a list and indexes [-1] after start_run — both break.
    # Fix: rebind it to a real list (so set_active_mlflow_run's append/remove
    # work), and replace clean_active_mlflow_run with a no-op (mlflow.start_run
    # now manages its own active-run state, so wrapping it is unnecessary).
    from contextlib import contextmanager
    import pycaret.loggers.mlflow_logger as ml

    ml._active_run_stack = []

    @contextmanager
    def _noop():
        yield

    ml.clean_active_mlflow_run = _noop


def _silence_noisy_loggers() -> None:
    # MLflow chats at WARNING with deprecation/safety notes on every model save.
    import logging
    logging.getLogger("mlflow").setLevel(logging.ERROR)

    # LightGBM emits "No further splits with positive gain" from the C++ side
    # via its registered logger — replace with a sink that drops everything.
    try:
        import lightgbm as lgb

        class _SilentLogger:
            def info(self, msg): pass
            def warning(self, msg): pass

        lgb.register_logger(_SilentLogger())
    except ImportError:
        pass


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> dict:
    # PyCaret is imported inside main() so a learner who hasn't installed it
    # gets a clean error here, not at import time.
    try:
        from pycaret.regression import (  # noqa: F401
            compare_models, finalize_model, predict_model, pull,
            save_model, setup, tune_model,
        )
        _patch_pycaret_mlflow_compat()
        _silence_noisy_loggers()
    except ImportError as exc:
        raise SystemExit(
            "PyCaret is not installed. Run:\n"
            "    pip install -r requirements-mlops.txt\n"
            "or:\n"
            "    pip install pycaret"
        ) from exc

    # Point PyCaret's MlflowLogger at the same SQLite backend the rest of the
    # project uses, so PyCaret runs show up in `mlflow ui --backend-store-uri
    # sqlite:///mlflow.db` alongside mlflow_train.py runs.
    import os
    import mlflow
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}")

    params = load_params()
    print("=" * 70)
    print(" PyCaret AutoML benchmark vs the Lab 3 Voting Regressor (v3)")
    print(" — with deployment refit on full data")
    print("=" * 70)

    # 1. Load data + same split as Lab 3.
    X, y = load_features_and_target()
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )
    print(f"\nLoaded data: train = {len(X_train)}, test = {len(X_test)}, "
          f"full = {len(X)}")

    # 2. PyCaret needs a single dataframe with the target as a column.
    train_df = X_train.copy()
    train_df["price"] = y_train

    # 3. setup() — PyCaret's preprocessing entry point. Fit on TRAIN only;
    #    the test set must remain unseen at this stage.
    print("\n[1/6] PyCaret setup() on training pool — preprocessing initializing…")
    setup(
        data=train_df,
        target="price",
        session_id=params["data"]["random_state"],
        train_size=0.8,
        normalize=True,
        verbose=True,
        html=True,
        log_experiment=True,
        experiment_name="M2_Pune_Real_Estate_Price_PyCaret",
    )

    # 4. compare_models — train and rank ~20 algorithms by R².
    print("[2/6] compare_models(n_select=10, sort='R2') — running ~20 algorithms…")
    top_n = compare_models(n_select=10, sort="R2", turbo=True)
    leaderboard = pull()
    print("\n   Top 5 by R²:")
    print(leaderboard.head(5).to_string())

    # 5. Tune the winner.
    best = top_n[0] if isinstance(top_n, list) else top_n
    print(f"\n[3/6] tune_model({type(best).__name__}, n_iter=20)…")
    tuned = tune_model(best, n_iter=20, optimize="R2", choose_better=True, verbose=False)

    # 6. Finalize on the training pool (this is the model we SCORE).
    print(f"[4/6] finalize_model() → refit on full training pool (159 rows)")
    scored_model = finalize_model(tuned)
    scored_path = MODEL_DIR / "pycaret_pune_real_estate"
    save_model(scored_model, str(scored_path))
    print(f"      Saved (audit copy): {scored_path}.pkl")

    # 7. Score on BOTH train and test sets — the v2 diagnostic.
    print("\n[5/6] Head-to-head — train vs test on the Lab 3 splits")
    pred_col_cache: list[str] = []
    pycaret_train = _score_pycaret_pipeline(predict_model, scored_model, X_train, y_train, pred_col_cache)
    pycaret_test  = _score_pycaret_pipeline(predict_model, scored_model, X_test,  y_test,  pred_col_cache)

    chosen_name = leaderboard.iloc[0]["Model"]
    pycaret_cv_r2 = float(leaderboard.iloc[0]["R2"])
    print(f"\n   CV mean R² for '{chosen_name}' (from compare_models): {pycaret_cv_r2:.4f}")

    # 8. Score the EXISTING Lab 3 Voting Regressor on the SAME splits.
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
    print(f"│ {'PyCaret ' + type(scored_model).__name__:<32s}"
          f"{pycaret_train['r2']:>11.4f}{pycaret_test['r2']:>11.4f}"
          f"{pyc_gap:>+13.4f}{_verdict(pyc_gap):>10s} │")
    print("└" + "─" * 78 + "┘")

    # RMSE table.
    print()
    print("┌" + "─" * 60 + "┐")
    print(f"│ {'Model':<30s}{'Train RMSE':>13s}{'Test RMSE':>15s} │")
    print("├" + "─" * 60 + "┤")
    if lab3_train and lab3_test:
        print(f"│ {'Lab 3 manual VotingRegressor':<30s}"
              f"{lab3_train['rmse']:>13.2f}{lab3_test['rmse']:>15.2f} │")
    print(f"│ {'PyCaret ' + type(scored_model).__name__:<30s}"
          f"{pycaret_train['rmse']:>13.2f}{pycaret_test['rmse']:>15.2f} │")
    print("└" + "─" * 60 + "┘")

    # 10. Verdict prose with overfitting honesty.
    if lab3_test:
        delta = pycaret_test["r2"] - lab3_test["r2"]
        if abs(delta) < 0.01:
            verdict = "≈ tied on test R² — Lab 3's manual pipeline matches AutoML"
        elif delta > 0:
            verdict = f"PyCaret wins on test R² by Δ = +{delta:.4f}"
        else:
            verdict = f"Lab 3 wins on test R² by Δ = {abs(delta):.4f}"
        print(f"\n   Verdict (test R² only): {verdict}")

        if delta > 0 and pyc_gap >= 0.10:
            print(f"   ⚠️  But PyCaret's train-test gap is {pyc_gap:+.4f} — "
                  f"some of that test win may be overfitting noise.")
        elif delta > 0 and 0.05 <= pyc_gap < 0.10:
            print(f"   ⚠️  PyCaret's train-test gap is {pyc_gap:+.4f} — "
                  f"watch for overfitting on a different test split.")

    # ─── 11. v3 ADDITION: deployment refit on the FULL dataset ───────────────
    #
    # The test set has done its one job — it gave us the honest performance
    # number we just printed. From this point on, we treat that number as
    # frozen and refit the same configuration on every row of data we have.
    #
    # Discipline reminder:
    #   * Same algorithm (whatever `tuned` is)
    #   * Same hyperparameters (no re-tuning)
    #   * Bigger training data (199 rows instead of 159)
    #   * No more decisions based on the test set — we already committed
    #
    # The deployed model will, in expectation, be slightly better than the
    # 0.92 test R² we reported, because it learned from 25% more rows. We
    # do NOT re-score it; the test set is no longer a valid hold-out for it.
    print("\n[6/6] Deployment refit — retraining on FULL dataset (train + test)")
    print("       Same algorithm, same hyperparameters; only the data is bigger.")

    full_df = X.copy()
    full_df["price"] = y

    # Re-run setup on the full data so the preprocessing pipeline (scaling,
    # encoding) is fit on what the deployed model will see in production.
    # train_size=0.99 because PyCaret requires *some* internal holdout —
    # we can't pass 1.0. The 1% sliver is not used for any decision; we've
    # already made every decision we're going to make.
    setup(
        data=full_df,
        target="price",
        session_id=params["data"]["random_state"],
        train_size=0.99,
        normalize=True,
        verbose=False,
        html=False,
        log_experiment=False,
    )

    deployment_model = finalize_model(tuned)
    deployment_path = MODEL_DIR / "pycaret_pune_real_estate_DEPLOY"
    save_model(deployment_model, str(deployment_path))
    print(f"       Saved (deployment): {deployment_path}.pkl")
    print(f"       Expected production R² ≈ {pycaret_test['r2']:.4f} "
          f"(from step 5; deployed model is fit on more data so likely slightly better)")

    # 12. Persist comparison metrics for DVC.
    payload = {
        # ── PyCaret diagnostics (v2 + v3) ────────────────────────────────
        "pycaret_model":              type(scored_model).__name__,
        "pycaret_train_r2":           pycaret_train["r2"],
        "pycaret_cv_r2":              pycaret_cv_r2,
        "pycaret_test_r2":            pycaret_test["r2"],
        "pycaret_train_test_gap":     pyc_gap,
        "pycaret_overfit_verdict":    _verdict(pyc_gap),
        "pycaret_train_rmse":         pycaret_train["rmse"],
        "pycaret_test_rmse":          pycaret_test["rmse"],

        # ── v3: artifact paths ───────────────────────────────────────────
        "pycaret_scored_artifact":    f"{scored_path}.pkl",
        "pycaret_deploy_artifact":    f"{deployment_path}.pkl",
        "pycaret_n_train_scored":     len(X_train),
        "pycaret_n_train_deployed":   len(X),

        # ── Back-compat aliases (v1 schema) ──────────────────────────────
        "pycaret_rmse":               pycaret_test["rmse"],
        "pycaret_r2":                 pycaret_test["r2"],

        # ── Lab 3 diagnostics (v2) ───────────────────────────────────────
        "lab3_train_r2":              lab3_train["r2"]   if lab3_train else None,
        "lab3_test_r2":               lab3_test["r2"]    if lab3_test  else None,
        "lab3_train_test_gap":        (lab3_train["r2"] - lab3_test["r2"])
                                       if (lab3_train and lab3_test) else None,
        "lab3_overfit_verdict":       _verdict(lab3_train["r2"] - lab3_test["r2"])
                                       if (lab3_train and lab3_test) else None,
        "lab3_train_rmse":            lab3_train["rmse"] if lab3_train else None,
        "lab3_test_rmse":             lab3_test["rmse"]  if lab3_test  else None,

        # ── Back-compat aliases (v1 schema) ──────────────────────────────
        "lab3_rmse":                  lab3_test["rmse"]  if lab3_test  else None,
        "lab3_r2":                    lab3_test["r2"]    if lab3_test  else None,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "pycaret_benchmark.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n📊 Comparison saved: {out.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"   Audit model       : {scored_path.name}.pkl  "
          f"(fit on {len(X_train)} rows, scored on {len(X_test)})")
    print(f"   Deployment model  : {deployment_path.name}.pkl  "
          f"(fit on {len(X)} rows, NOT scored)")
    print(f"   Reported test R²  : {pycaret_test['r2']:.4f}  ← what to put on the slide")
    print(f"   Train-test gap    : {pyc_gap:+.4f}  ({_verdict(pyc_gap)})")
    print()
    print("   To deploy: point src/inference.py at the DEPLOY .pkl and restart FastAPI.")
    return payload


if __name__ == "__main__":
    main()
