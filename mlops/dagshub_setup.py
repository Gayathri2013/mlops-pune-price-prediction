"""DagsHub configuration helper.

DagsHub provides hosted Git + MLflow + DVC. To point this project at a
DagsHub repo, you need three credentials and one URL:

    DAGSHUB_USER     — your DagsHub username
    DAGSHUB_TOKEN    — access token from https://dagshub.com/user/settings/tokens
    DAGSHUB_REPO     — repo name (e.g. "pune-real-estate-mlops")

Set them as environment variables (recommended) or edit them into a
`.env` file at the project root. Then:

    python -m mlops.dagshub_setup --check          # verify env vars are set
    python -m mlops.dagshub_setup --configure      # set MLflow URI + auth env vars
                                                    # for the current Python process
    python -m mlops.dagshub_setup --print-dvc-cmds # print exact dvc commands to run

This script is functional and side-effect-light: --configure only sets
process-local environment variables. It does NOT modify ~/.bashrc, your
DVC config, or anything outside this Python process.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def _read_env() -> dict[str, Optional[str]]:
    """Pull DagsHub credentials from the environment. None if unset."""
    return {
        "user":  os.environ.get("DAGSHUB_USER"),
        "token": os.environ.get("DAGSHUB_TOKEN"),
        "repo":  os.environ.get("DAGSHUB_REPO"),
    }


def check() -> bool:
    """Return True iff all three env vars are set. Print a friendly status."""
    env = _read_env()
    print("=== DagsHub credentials ===")
    all_set = True
    for key, val in env.items():
        if val:
            shown = val if key != "token" else f"{val[:4]}…{val[-2:]}"
            print(f"  ✅ DAGSHUB_{key.upper():<6s} = {shown}")
        else:
            print(f"  ❌ DAGSHUB_{key.upper():<6s} = (unset)")
            all_set = False
    if not all_set:
        print()
        print("To set these (Linux/macOS bash):")
        print("    export DAGSHUB_USER=your_username")
        print("    export DAGSHUB_TOKEN=your_token")
        print("    export DAGSHUB_REPO=pune-real-estate-mlops")
        print()
        print("On Windows PowerShell:")
        print("    $env:DAGSHUB_USER    = 'your_username'")
        print("    $env:DAGSHUB_TOKEN   = 'your_token'")
        print("    $env:DAGSHUB_REPO    = 'pune-real-estate-mlops'")
    return all_set


def configure_mlflow() -> Optional[str]:
    """Set MLFLOW_TRACKING_URI + auth env vars for the current process.

    Returns the configured URI on success, None otherwise.
    """
    env = _read_env()
    if not (env["user"] and env["token"] and env["repo"]):
        print("❌ Cannot configure — run `python -m mlops.dagshub_setup --check` first.")
        return None

    uri = f"https://dagshub.com/{env['user']}/{env['repo']}.mlflow"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = env["user"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = env["token"]

    print(f"✅ MLFLOW_TRACKING_URI = {uri}")
    print(f"✅ MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD set")
    print()
    print("Subsequent calls in THIS Python process will log to DagsHub.")
    print("To make it permanent, export the same three env vars in your shell")
    print("startup file (.bashrc / .zshrc / PowerShell profile).")
    return uri


def print_dvc_commands() -> None:
    """Print the exact dvc commands needed to wire this repo to DagsHub.

    We don't run them here because dvc commands modify .dvc/config which
    is best done by the learner, with full visibility into what changes.
    """
    env = _read_env()
    if not (env["user"] and env["token"] and env["repo"]):
        print("❌ Run `python -m mlops.dagshub_setup --check` first.")
        return

    user, token, repo = env["user"], env["token"], env["repo"]
    print("=== DVC remote configuration commands ===")
    print()
    print("Run these once, from the project root, after `dvc init`:")
    print()
    print(f"    dvc remote add -d origin https://dagshub.com/{user}/{repo}.dvc")
    print(f"    dvc remote modify origin --local auth basic")
    print(f"    dvc remote modify origin --local user {user}")
    print(f"    dvc remote modify origin --local password {token}")
    print()
    print("Verify with:")
    print("    dvc remote list")
    print()
    print("Then push your data:")
    print("    dvc add model_features.csv model_target.npy model/")
    print("    dvc push")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="Show which env vars are set")
    parser.add_argument("--configure", action="store_true",
                        help="Set MLflow env vars in the current Python process")
    parser.add_argument("--print-dvc-cmds", action="store_true",
                        help="Print the dvc commands needed for the DagsHub remote")
    args = parser.parse_args()

    if not any((args.check, args.configure, args.print_dvc_cmds)):
        # No flag → show all
        check()
        print()
        print_dvc_commands()
        return 0

    if args.check:
        return 0 if check() else 1
    if args.configure:
        return 0 if configure_mlflow() else 1
    if args.print_dvc_cmds:
        print_dvc_commands()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
