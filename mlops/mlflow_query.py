"""Query the MLflow tracking store from code.

Reads the experiment named in params.yaml, prints a leaderboard of all runs
(sorted by R²), and writes a Ridge-alpha-vs-R² plot to metrics/ridge_sweep.png.

This is the everyday command for "which model performed best?" and proves
that you don't need to babysit the MLflow UI to do real analysis — every
column is queryable from a one-liner.

Run from the project root:
    python -m mlops.mlflow_query
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from mlops.utils import METRICS_DIR, PROJECT_ROOT, load_params


def main() -> pd.DataFrame:
    params = load_params()

    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    exp = mlflow.get_experiment_by_name(params["mlflow"]["experiment_name"])
    if exp is None:
        print(f"❌ Experiment '{params['mlflow']['experiment_name']}' not found.")
        print("   Run `python -m mlops.mlflow_train` or `python -m mlops.mlflow_sweep` first.")
        return pd.DataFrame()

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["metrics.r2 DESC"])
    if runs.empty:
        print("⚠️  Experiment exists but has no runs yet.")
        return runs

    # ── Top-level leaderboard ───────────────────────────────────────────────
    cols = [c for c in (
        "tags.mlflow.runName", "params.model_type", "params.ridge_alpha",
        "params.lasso_alpha", "metrics.rmse", "metrics.r2", "run_id"
    ) if c in runs.columns]
    print("\n=== Leaderboard (top 10 by R²) ===")
    print(runs[cols].head(10).to_string(index=False))

    # ── Sweep-only filter ────────────────────────────────────────────────────
    sweep = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.sweep = 'ridge_alpha'",
        order_by=["params.ridge_alpha ASC"],
    )

    if not sweep.empty and "params.ridge_alpha" in sweep.columns:
        sweep = sweep.copy()
        sweep["alpha"] = sweep["params.ridge_alpha"].astype(float)
        sweep = sweep.sort_values("alpha")
        print("\n=== Ridge alpha sweep, ordered by alpha ===")
        print(sweep[["alpha", "metrics.rmse", "metrics.r2"]].to_string(index=False))

        # ── Plot the sweep curve ────────────────────────────────────────────
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = METRICS_DIR / "ridge_sweep.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sweep["alpha"], sweep["metrics.r2"], "o-", color="#2563eb",
                lw=2, markersize=10)
        ax.set_xscale("log")
        ax.set_xlabel("Ridge α (log scale)")
        ax.set_ylabel("Test R²")
        ax.set_title("Ridge Alpha Sweep — R² vs α", fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(plot_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n📊 Plot saved: {plot_path.relative_to(PROJECT_ROOT)}")

    print(f"\nTotal runs in experiment: {len(runs)}")
    return runs


if __name__ == "__main__":
    main()
