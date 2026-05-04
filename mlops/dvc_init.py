"""Initialize DVC for this project (one-time setup) — v2 with full pipeline.

This script bootstraps a fresh laptop into the DVC workflow. It is
idempotent — running it twice is safe; nothing gets overwritten.

What changed vs v1:
  * v1 ran `dvc add` on model_features.csv, model_target.npy, and the five
    helper .pkls in model/. That made sense when those files were produced
    by notebooks outside DVC's awareness.
  * v2 (this version) recognizes that those files are now declared as
    OUTPUTS of the `features` stage in dvc.yaml. DVC manages them
    automatically — `dvc add`-ing them would actually be an error
    ("output already tracked").
  * What this script `dvc add`s now: only the raw Excel input. Everything
    downstream is produced by `dvc repro`.

Steps:
  1. Verify a Git repo exists (DVC requires Git)
  2. `dvc init` if not already initialized
  3. `dvc add` ONLY the raw input file (Pune Real Estate Data.xlsx)
  4. Configure a default LOCAL remote (./dvc_remote) for the demo
  5. Print next steps — including running `dvc repro` to materialize
     all the downstream artifacts

To use a real remote (S3 / DagsHub) instead of the local folder, run:
    python -m mlops.dagshub_setup --print-dvc-cmds

Run from the project root:
    python -m mlops.dvc_init
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files we want under DVC tracking via `dvc add`. These are INPUTS to the
# pipeline, not outputs. Pipeline outputs (model_features.csv, the .pkls,
# the .sav, etc.) are tracked automatically because they're declared in
# dvc.yaml — `dvc add` would conflict with that.
TRACKED_INPUTS = [
    "Pune Real Estate Data.xlsx",
]

LOCAL_REMOTE_PATH = PROJECT_ROOT / "dvc_remote"


def _run(cmd: list[str], check: bool = True, cwd: Path = PROJECT_ROOT):
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
    _run(["git", "config", "user.email", "lab@example.com"])
    _run(["git", "config", "user.name",  "Lab Student"])


def init_dvc_if_missing() -> None:
    if _dvc_initialized():
        print("[2/4] ✅ DVC already initialized")
        return
    print("[2/4] Initializing DVC")
    _run(["dvc", "init", "-q"])
    _run(["git", "add", ".dvc/.gitignore", ".dvc/config", ".dvcignore"], check=False)


def add_inputs() -> list[str]:
    """Run `dvc add` on each raw input. Returns the list that succeeded."""
    print("[3/4] Registering raw input(s) with DVC")
    succeeded = []
    for rel in TRACKED_INPUTS:
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"      ⚠️  skip {rel} (not present)")
            continue
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
    """Configure a local-folder remote for the demo workflow."""
    print("[4/4] Configuring a LOCAL remote at ./dvc_remote")
    LOCAL_REMOTE_PATH.mkdir(parents=True, exist_ok=True)

    result = _run(["dvc", "remote", "list"], check=False)
    if "local_storage" in result.stdout:
        print("      ✓ remote 'local_storage' already configured")
        return
    _run(["dvc", "remote", "add", "-d", "local_storage", str(LOCAL_REMOTE_PATH)])
    print(f"      ✅ remote 'local_storage' → {LOCAL_REMOTE_PATH}")


def main() -> int:
    print("=" * 70)
    print(" DVC initialization for the Pune Price Prediction project (v2)")
    print("=" * 70)
    print()

    try:
        init_git_if_missing()
        init_dvc_if_missing()
        added = add_inputs()
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
    print(f"  Raw inputs registered : {len(added)}")
    print()
    print("  Commit the DVC scaffolding to Git:")
    print("    git add *.dvc .gitignore .dvc/.gitignore .dvc/config .dvcignore dvc.yaml")
    print("    git commit -m 'Add DVC pipeline (clean → features → train)'")
    print()
    print("  Run the FULL pipeline (raw Excel → trained model) in one command:")
    print("    dvc repro")
    print()
    print("  Inspect the dependency graph:")
    print("    dvc dag")
    print()
    print("  See current metrics:")
    print("    dvc metrics show")
    print()
    print("  Push everything (raw input + all pipeline outputs) to the local remote:")
    print("    dvc push")
    print()
    print("  Switch to DagsHub instead of local:")
    print("    python -m mlops.dagshub_setup --print-dvc-cmds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
