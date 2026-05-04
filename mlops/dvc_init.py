"""Initialize DVC for this project (one-time setup).

This script is what a learner runs on their machine the first time they
want DVC tracking on this project. It is idempotent — running it twice
is safe; nothing is overwritten.

Steps:
  1. Verify a Git repo exists (DVC requires Git)
  2. `dvc init` if not already initialized
  3. `dvc add` the existing data + model artifacts so they're tracked
  4. Configure a default LOCAL remote (./dvc_remote) for the demo
  5. Print next steps

To use a real remote (S3 / DagsHub) instead of the local folder, run:
    python -m mlops.dagshub_setup --print-dvc-cmds

Run from the project root:
    python -m mlops.dvc_init

Note: this script invokes the `dvc` and `git` CLIs via subprocess. They
must be installed and on PATH. `pip install -r requirements-mlops.txt`
takes care of that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files we want under DVC tracking. Each path is relative to project root.
# Order matters slightly: data first, then auxiliary model artifacts.
#
# Two model files are intentionally NOT in this list:
#   * model/property_price_prediction_voting.sav
#   * model/interval_est.pkl
# Those are declared as outputs of the `train` stage in dvc.yaml, so DVC
# tracks them automatically when you run `dvc repro`. Trying to `dvc add`
# them here would be an error.
TRACKED_PATHS = [
    "model_features.csv",
    "model_target.npy",
    "model/count_vectorizer.pkl",
    "model/sub_area_price_map.pkl",
    "model/amenities_score_price_map.pkl",
    "model/all_feature_names.pkl",
    "model/feature_cols.pkl",
]

LOCAL_REMOTE_PATH = PROJECT_ROOT / "dvc_remote"


def _run(cmd: list[str], check: bool = True, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    """Wrapper around subprocess.run that prints what it ran."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd), check=check, capture_output=True, text=True)


def _git_repo_exists() -> bool:
    return (PROJECT_ROOT / ".git").is_dir()


def _dvc_initialized() -> bool:
    return (PROJECT_ROOT / ".dvc").is_dir()


def init_git_if_missing() -> None:
    """Create a fresh Git repo if one isn't there. Sets minimal user config
    so DVC's pre-commit hook doesn't complain on a clean machine."""
    if _git_repo_exists():
        print("[1/4] ✅ Git repo already exists")
        return
    print("[1/4] No Git repo — initializing one")
    _run(["git", "init", "-q"])
    # Minimal config so commits work without prompting
    _run(["git", "config", "user.email", "lab@example.com"])
    _run(["git", "config", "user.name",  "Lab Student"])


def init_dvc_if_missing() -> None:
    if _dvc_initialized():
        print("[2/4] ✅ DVC already initialized")
        return
    print("[2/4] Initializing DVC")
    _run(["dvc", "init", "-q"])
    # Stage the .dvc/ scaffolding for git
    _run(["git", "add", ".dvc/.gitignore", ".dvc/config", ".dvcignore"], check=False)


def add_artifacts() -> list[str]:
    """Run `dvc add` on each tracked file. Returns the list that succeeded."""
    print("[3/4] Registering existing artifacts with DVC")
    succeeded = []
    for rel in TRACKED_PATHS:
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"      ⚠️  skip {rel} (not present)")
            continue
        # Skip if already tracked (the .dvc pointer file already exists)
        if (PROJECT_ROOT / f"{rel}.dvc").exists():
            print(f"      ✓ {rel} (already tracked)")
            succeeded.append(rel)
            continue
        result = _run(["dvc", "add", rel], check=False)
        if result.returncode == 0:
            print(f"      ✅ added {rel}")
            succeeded.append(rel)
        else:
            print(f"      ❌ failed to add {rel}: {result.stderr.strip()}")
    return succeeded


def configure_local_remote() -> None:
    """Configure a local-folder remote for the demo workflow.

    For real projects you'd swap this for s3:// or DagsHub. Run
    `python -m mlops.dagshub_setup --print-dvc-cmds` to see those commands.
    """
    print("[4/4] Configuring a LOCAL remote at ./dvc_remote")
    LOCAL_REMOTE_PATH.mkdir(parents=True, exist_ok=True)

    result = _run(["dvc", "remote", "list"], check=False)
    existing = result.stdout
    if "local_storage" in existing:
        print("      ✓ remote 'local_storage' already configured")
        return
    _run(["dvc", "remote", "add", "-d", "local_storage", str(LOCAL_REMOTE_PATH)])
    print(f"      ✅ remote 'local_storage' → {LOCAL_REMOTE_PATH}")


def main() -> int:
    print("=" * 70)
    print(" DVC initialization for the Pune Price Prediction project")
    print("=" * 70)
    print()

    try:
        init_git_if_missing()
        init_dvc_if_missing()
        added = add_artifacts()
        configure_local_remote()
    except subprocess.CalledProcessError as exc:
        print(f"\n❌ Command failed: {exc}")
        print(exc.stderr if exc.stderr else "")
        return 1
    except FileNotFoundError as exc:
        print(f"\n❌ Required CLI not found: {exc}")
        print("   Install dvc and git, then re-run this script.")
        return 1

    print()
    print("=" * 70)
    print(" Next steps")
    print("=" * 70)
    print()
    print(f"  Pointer files created  : {len(added)}")
    print()
    print("  Commit the pointers to Git:")
    print("    git add *.dvc model/*.dvc model/.gitignore .gitignore")
    print("    git commit -m 'Track Pune dataset and model artifacts with DVC'")
    print()
    print("  Push the actual data to the local remote:")
    print("    dvc push")
    print()
    print("  Run the full pipeline (train → metrics):")
    print("    dvc repro")
    print("    dvc metrics show")
    print()
    print("  Switch to DagsHub instead of local:")
    print("    python -m mlops.dagshub_setup --print-dvc-cmds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
