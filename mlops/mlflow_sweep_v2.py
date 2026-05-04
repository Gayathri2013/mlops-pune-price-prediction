"""Sweep Ridge alphas — one MLflow run per alpha, with train + CV + test (v2).

================================================================================
 VERSION HISTORY
================================================================================

v1 (mlflow_sweep.py)
--------------------
  * For each alpha, fit a fresh Voting Regressor (LR + Ridge(alpha) + Lasso),
    score on the held-out test set, log the run.
  * Limitation: a single test score per alpha is noisy at 159 train rows.
    `alpha=10` might "win" purely because the test split happened to be
    friendly to it. The leaderboard becomes a coin flip in disguise.

v2 (this version) — train + CV + test per alpha
-----------------------------------------------
  * For each alpha, logs THREE R²s:
      - train_r2  (fit on full training set)
      - cv_r2_mean / cv_r2_std (5-fold cross-validation, more stable than test)
      - test_r2   (held-out 40-row test set)
  * Logs train_test_r2_gap for overfitting detection.
  * Picks the "best alpha" using CV mean R², not test R² — this is the
    statistically honest choice. The test-set winner is reported alongside
    so you can see when they disagree (which they often will on small data).
  * Tags each run with overfit_verdict (healthy / watch / OVERFIT) so you
    can filter the UI by verdict.
  * Each run name now includes the alpha for readability in the runs table.

The cross_val_score adds ~5x to per-alpha runtime (5 folds), but at
~0.1s per fit it's still under a minute for the full sweep. The honesty
upgrade is worth it.

================================================================================
 USAGE
================================================================================

Run from the project root:
    python -m mlops.mlflow_sweep_v2

Then:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    open http://localhost:5000

In the UI, filter `tags.sweep = 'ridge_alpha_v2'` to see this sweep's runs.
Sort by `metrics.cv_r2_mean DESC` to find the best alpha by CV.
Sort by `metrics.train_test_r2_gap DESC` to spot overfit alphas.
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import KFold, cross_val_score

from mlops.utils import (
    PROJECT_ROOT,
    build_voting_regressor,
    load_features_and_target,
    load_params,
    score_regressor,
    split_data,
)


def _verdict(gap: float) -> str:
    """Same thresholds as pycaret_benchmark_v2 / mlflow_train_v2 — keep consistent."""
    if gap < 0.05:
        return "healthy"
    if gap < 0.10:
        return "watch"
    return "OVERFIT"


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
    print(f"5-fold CV per alpha; train + CV + test scores logged.\n")

    print(f"  {'α':<10s}{'Train R²':>10s}{'CV R²':>14s}{'Test R²':>10s}"
          f"{'T−Test':>10s}{'Verdict':>10s}{'Run':>10s}")
    print("  " + "─" * 78)

    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=params["data"]["random_state"])
    results = []

    for alpha in alphas:
        with mlflow.start_run(run_name=f"sweep_ridge_alpha_{alpha}") as run:
            # 1. Fit on the training pool.
            model = build_voting_regressor(alpha, lasso_alpha, lasso_max_iter)
            model.fit(X_train, y_train)

            # 2. Train + test scoring.
            train_m = score_regressor(model, X_train, y_train)
            test_m  = score_regressor(model, X_test,  y_test)
            gap = train_m["r2"] - test_m["r2"]
            verdict = _verdict(gap)

            # 3. 5-fold CV on the training pool (a fresh model per fold so we
            #    don't leak the already-fit weights into the CV estimate).
            cv_model = build_voting_regressor(alpha, lasso_alpha, lasso_max_iter)
            cv_scores = cross_val_score(cv_model, X_train, y_train,
                                        cv=cv_splitter, scoring="r2", n_jobs=1)
            cv_mean = float(cv_scores.mean())
            cv_std  = float(cv_scores.std())

            # 4. Log everything.
            mlflow.log_params({
                "model_type":     "VotingRegressor",
                "ridge_alpha":    alpha,
                "lasso_alpha":    lasso_alpha,
                "n_features":     X_train.shape[1],
                "cv_folds":       5,
            })
            mlflow.log_metrics({
                "train_r2":           train_m["r2"],
                "train_rmse":         train_m["rmse"],
                "test_r2":            test_m["r2"],
                "test_rmse":          test_m["rmse"],
                "cv_r2_mean":         cv_mean,
                "cv_r2_std":          cv_std,
                "train_test_r2_gap":  gap,
                # v1 back-compat aliases
                "rmse":               test_m["rmse"],
                "r2":                 test_m["r2"],
            })
            mlflow.set_tags({
                "module":           "M2_Lab5",
                "sweep":            "ridge_alpha_v2",
                "overfit_verdict":  verdict,
            })

            # 5. Log model with a signature.
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, name="model", signature=signature)

            row = {
                "alpha": alpha,
                "train_r2": train_m["r2"],
                "cv_r2_mean": cv_mean,
                "cv_r2_std": cv_std,
                "test_r2": test_m["r2"],
                "test_rmse": test_m["rmse"],
                "gap": gap,
                "verdict": verdict,
                "run_id": run.info.run_id,
            }
            results.append(row)

            print(f"  α={alpha:<8}{train_m['r2']:>10.4f}"
                  f"{cv_mean:>9.4f}±{cv_std:.3f}"
                  f"{test_m['r2']:>10.4f}{gap:>+10.4f}"
                  f"{verdict:>10s}{run.info.run_id[:8]:>10s}")

    # ── Picking the winner — CV first, test second ─────────────────────────
    best_cv   = max(results, key=lambda r: r["cv_r2_mean"])
    best_test = max(results, key=lambda r: r["test_r2"])

    print()
    print(f"⭐ Best by CV  (recommended): α={best_cv['alpha']}  "
          f"CV R²={best_cv['cv_r2_mean']:.4f}±{best_cv['cv_r2_std']:.3f}  "
          f"Test R²={best_cv['test_r2']:.4f}  ({best_cv['verdict']})")
    print(f"   Best by Test (noisier):    α={best_test['alpha']}  "
          f"Test R²={best_test['test_r2']:.4f}  "
          f"CV R²={best_test['cv_r2_mean']:.4f}  ({best_test['verdict']})")

    if best_cv["alpha"] != best_test["alpha"]:
        print()
        print("   ℹ️  CV and test winners disagree. With only 40 test rows that's normal.")
        print("       Trust the CV winner — it averages across 5 folds and is more stable.")

    print()
    print(f"Inspect the full sweep:  mlflow ui --backend-store-uri sqlite:///"
          f"{(PROJECT_ROOT / 'mlflow.db').name}")
    print(f"Filter the runs:         tags.sweep = 'ridge_alpha_v2'")
    return results


if __name__ == "__main__":
    main()
