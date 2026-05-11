# Module 2 — Lab 5: MLOps Tools (Hands-On)

> The companion notebook (`M2_Lab5_MLOps_Tools_Primers.ipynb`) is the conceptual reference;
> **this document is what you do at the keyboard.**

By the end of this lab, the Pune Price Prediction project you deployed in Lab 4 will have:

- A **PyCaret** AutoML benchmark that compares ~20 algorithms against your manual model — with overfitting diagnostics and a deployment-grade refit
- A reproducible training pipeline driven by **DVC** (clean → features → train, three stages, one DAG)
- Every training run logged to **MLflow** with parameters, metrics, model artifacts, and a model **registry**
- Credentials and commands ready to push everything to **DagsHub** for team collaboration

The work is split into 8 sections. Each is a self-contained chunk of 15–25 minutes. Total time: ~3 hours including reading.

> **Versioned scripts.** You'll see `_v1`, `_v2`, `_v3` suffixes throughout. These are not parallel implementations — they're a learning progression. Each version adds one important concept on top of the last. Run them in order; don't skip to the latest.

---

## Prerequisites

- You finished **Labs 1, 2, 3, and 4** of Module 2 — the FastAPI app in `src/` loads and serves predictions correctly.
- Python 3.10–3.12 is on PATH. (PyCaret does not support 3.13+ yet.)
- `git` and `dvc` are on PATH.
- A clean shell (terminal / PowerShell) opened at the project root.

Sanity check before starting:

```bash
python --version          # 3.10, 3.11 or 3.12
git --version             # any
dvc --version             # any 3.x
ls model/                 # should show the .pkl/.sav files from Labs 2-3
ls model_features.csv     # should be present
uvicorn src.app:app --reload   # the Lab 4 API still serves
```

If anything above fails, fix it before continuing.

---

## Section 0 — Install MLOps tooling

The Lab 4 service requires the packages in `requirements.txt`. Lab 5 adds a heavier set — kept in a separate file so the production API container stays small.

```bash
# Activate your venv first
pip install -r requirements-mlops.txt
```

PyCaret pulls XGBoost, LightGBM, CatBoost, Yellowbrick, and a pinned scikit-learn. **First-time install takes 2–3 minutes.** That's normal.

Verify everything imports:

```bash
python -c "import mlflow, dvc, pycaret, dagshub; print('OK')"
```

---

## Section 1 — PyCaret AutoML Benchmark (~30 min)

The first MLOps question worth answering on any project is:

> **What R² is even possible on this dataset?**

If 20 well-tuned algorithms top out at R² = 0.65, your manual pipeline is unlikely to hit 0.85. Conversely, if PyCaret reaches 0.92 in two minutes, your Lab 3 model has clear room to grow.

There are three PyCaret scripts in `mlops/` — run them in order. Each adds one concept.

### 1.1 — `pycaret_benchmark.py` (v1): first benchmark

```bash
python -m mlops.pycaret_benchmark
```

This will:

1. Load `model_features.csv` + `model_target.npy` (Lab 2 outputs)
2. Use the **same train/test split as Lab 3** (`random_state=42`)
3. Run PyCaret `setup` → `compare_models` → `tune_model` → `finalize_model` → `save_model`
4. Score PyCaret's champion on Lab 3's exact test set
5. Score your existing `model/property_price_prediction_voting.sav` on the same set
6. Print a head-to-head comparison and save it to `metrics/pycaret_benchmark.json`

Expected output (numbers vary by run):

```
┌─────────────────────────────────────────────────────────────┐
│ Model                              RMSE (₹L)             R² │
├─────────────────────────────────────────────────────────────┤
│ Lab 3 manual VotingRegressor          15.75         0.8519 │
│ PyCaret final Pipeline                11.79         0.9171 │
└─────────────────────────────────────────────────────────────┘
   Verdict: PyCaret wins by ΔR² = +0.0652
```

Looks great. But it's *one test number*. Is the PyCaret model actually generalizing, or is it memorizing the training data? You can't tell from a single test R². That's what v2 fixes.

### 1.2 — `pycaret_benchmark_v2.py`: overfitting diagnostic

```bash
python -m mlops.pycaret_benchmark_v2
```

Same workflow, but now scores on BOTH the training and test sets, and prints a verdict per model:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Model                          Train R²    Test R²   Train−Test    Verdict │
├──────────────────────────────────────────────────────────────────────────────┤
│ Lab 3 manual VotingRegressor    0.8612     0.8519     +0.0093      healthy │
│ PyCaret ExtraTreesRegressor     0.9821     0.9171     +0.0650        watch │
└──────────────────────────────────────────────────────────────────────────────┘

   CV mean R² for 'Extra Trees Regressor' (from compare_models): 0.8207
```

Three numbers now, each meaning something different:

- **Train R²** — capacity check. Should be high. If it's not, the model is broken.
- **CV R²** — generalization estimate. Averaged across 5 CV folds, more stable than test.
- **Test R²** — honest performance on data the model has never seen.

The **gap** (train minus test) is the diagnostic. Under 0.05 is healthy. Above 0.10 is overfitting. The Lab 3 linear Voting Regressor cannot really overfit a 159-row dataset because of L1/L2 regularization, so its gap is tiny. ExtraTrees grows trees to full depth and finds real signal *plus* memorizes some noise — hence the bigger gap.

A high test R² alone is not proof of a good model. v2 makes that visible.

### 1.3 — `pycaret_benchmark_v3.py`: deployment refit

The test R² in v2 is your *honest performance estimate*. Once you've recorded it, the test set has done its one job. The model you ship should learn from every row of signal you have — including those 40 test rows.

```bash
python -m mlops.pycaret_benchmark_v3
```

v3 prints the same diagnostic table as v2, then does one more step: it **refits the same algorithm with the same hyperparameters on the full 199-row dataset** and saves it as a separate artifact:

```
[6/6] Deployment refit — retraining on FULL dataset (train + test)
       Same algorithm, same hyperparameters; only the data is bigger.
       Saved (deployment): /…/model/pycaret_pune_real_estate_DEPLOY.pkl
       Expected production R² ≈ 0.9171 (from step 5; deployed model is fit
       on more data so likely slightly better)

======================================================================
 Summary
======================================================================
   Audit model       : pycaret_pune_real_estate.pkl  (fit on 159 rows, scored on 40)
   Deployment model  : pycaret_pune_real_estate_DEPLOY.pkl  (fit on 199 rows, NOT scored)
   Reported test R²  : 0.9171  ← what to put on the slide
   Train-test gap    : +0.0654  (watch)
```

Two model files, two purposes:

- `pycaret_pune_real_estate.pkl` — the *audit copy*. Trained on 159 rows, scored on 40. This is the model that produced the reportable R².
- `pycaret_pune_real_estate_DEPLOY.pkl` — the *deployment copy*. Trained on 199 rows, never scored. This is what your FastAPI app should load.

The discipline rule: **you do not change anything based on what the test R² turned out to be.** No re-tuning, no algorithm swap. The test set is touched exactly once, and then all decisions are frozen. Refitting on the full data is mechanically applying the same choice with more rows — it does not contaminate the test estimate.

### Comparison summary

| Script | What it adds | When to use it |
|---|---|---|
| `pycaret_benchmark.py` | Baseline head-to-head, single test R² | First exposure to PyCaret |
| `pycaret_benchmark_v2.py` | Train R², CV R², train-test gap verdict | Production model selection |
| `pycaret_benchmark_v3.py` | Deployment refit on full data; saves separate `_DEPLOY` artifact | Producing the model that actually ships |

### What this is NOT

PyCaret hides preprocessing decisions (encoding, scaling, outlier handling) in `setup()`. Use it to **set the bar**, not to replace your transparent pipeline. If you adopt PyCaret's model, audit its preprocessing first.

---

## Section 2 — MLflow: log one training run (~25 min)

You ran 25 PyCaret experiments in Section 1. Quick test — without scrolling back, can you tell which hyperparameters produced the 7th-best model? You can't. That's the problem MLflow solves.

There are two MLflow training scripts. Run them in order.

### 2.1 — `mlflow_train.py` (v1): the primitives

This is the same training as `mlops/train.py` but wrapped in `mlflow.start_run()`. It logs:

- Every hyperparameter (ridge_alpha, lasso_alpha, n_features, n_train, n_test, …)
- All test-set metrics (rmse, mae, r2, interval_margin)
- A diagnostic plot (residuals + predicted-vs-actual)
- The model itself with a signature and input example
- Free-form tags (module, dataset)

Run it:

```bash
python -m mlops.mlflow_train
```

Output:

```
MLflow tracking URI : sqlite:///.../mlflow.db
MLflow experiment   : M2_Pune_Real_Estate_Price
✅ Run logged: 31f9b8e94e7b474eaa46ea50014e23b8
   RMSE = ₹15.75L | MAE = ₹11.46L | R² = 0.8519
   Model URI: runs:/31f9b8e94e7b474eaa46ea50014e23b8/model
```

Open the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

Click the run. See params, metrics, tags, the diagnostic plot, the model artifact with its signature.

Run it 3–4 more times with different `ridge.alpha` values in `params.yaml`. Refresh the UI. Sort by R². Compare two runs side-by-side. *This is what experiment tracking is for.*

### 2.2 — `mlflow_train_v2.py`: overfitting diagnostic + Model Registry

v1 logs test metrics. A run with test R² = 0.85 and train R² = 0.86 (healthy) looks identical in the UI to a run with test R² = 0.85 and train R² = 0.99 (overfit). v2 closes that gap.

Run it:

```bash
python -m mlops.mlflow_train_v2
```

Three new things land in the UI:

**1. Train metrics alongside test.** New columns: `train_rmse`, `train_mae`, `train_r2`, plus `train_test_r2_gap`. Sort by `train_test_r2_gap DESC` to spot overfit runs at a glance.

**2. An `overfit_verdict` tag.** Each run is tagged `healthy`, `watch`, or `OVERFIT` based on the gap. Filter the runs view with `tags.overfit_verdict = "OVERFIT"` and only the bad runs show up.

**3. The Model Registry.** This is the big one. Click the **Models** tab on the left sidebar of the UI. There's a registered model called `pune_price_voting_regressor` with version 1. Run `mlflow_train_v2.py` again — version 2 appears.

The registry is what turns MLflow from a logbook into deployment infrastructure. Right now your FastAPI app loads `model/property_price_prediction_voting.sav` from disk. With the registry you can change that to:

```python
mlflow.pyfunc.load_model("models:/pune_price_voting_regressor/Production")
```

Train ten new models. Promote the best one to "Production" via the UI. Restart FastAPI. *Done — no code change, no file copying.* The model name is the contract; the version under that name is what changes.

To promote a run to Production:

1. Open the registered model `pune_price_voting_regressor`
2. Click the version you want
3. Set its stage to "Production"

### Comparison summary

| Script | What it adds |
|---|---|
| `mlflow_train.py` | Logs params, test metrics, model artifact, signature |
| `mlflow_train_v2.py` | + train metrics, train-test gap, overfit verdict, **Model Registry registration** |

---

## Section 3 — MLflow: sweep Ridge alphas (~20 min)

In Lab 3 you tuned Ridge α manually with a `for` loop printing to stdout. With MLflow, **every alpha becomes its own run** — and you can compare them visually in the UI.

### 3.1 — `mlflow_sweep.py` (v1): basic sweep

```bash
python -m mlops.mlflow_sweep
```

Loops over `params.yaml sweep.alphas` (default: 6 values from 0.01 to 1000) and creates one MLflow run per alpha. Each run is tagged with `sweep=ridge_alpha` so you can filter.

But there's a problem: with 40 test rows, the test R² per alpha is noisy. `alpha=10` might "win" purely because the test split happened to be friendly to it. We need a more stable signal.

### 3.2 — `mlflow_sweep_v2.py`: sweep with cross-validation

```bash
python -m mlops.mlflow_sweep_v2
```

Same sweep, but logs THREE R² values per alpha:

- `train_r2` — fit on full training set
- `cv_r2_mean`, `cv_r2_std` — 5-fold cross-validation, more stable than test
- `test_r2` — held-out 40-row test set

Plus `train_test_r2_gap` for overfitting detection per alpha, and an `overfit_verdict` tag.

The script picks the winner using **CV mean R²**, not test R² — this is the statistically honest choice. The test-set winner is reported alongside so you can see when they disagree (which they often will on small data).

Sample output:

```
  α          Train R²        CV R²    Test R²    T−Test   Verdict       Run
  ──────────────────────────────────────────────────────────────────────────────
  α=0.01       0.9234   0.7891±0.034     0.8456   +0.0778     watch   a1b2c3d4
  α=0.1        0.9012   0.8123±0.029     0.8512   +0.0500     watch   b2c3d4e5
  α=1.0        0.8612   0.8345±0.022     0.8519   +0.0093   healthy   c3d4e5f6
  α=10.0       0.8521   0.8401±0.018     0.8538   −0.0017   healthy   d4e5f6g7
  α=100.0      0.7234   0.6912±0.045     0.7123   +0.0111   healthy   e5f6g7h8
  α=1000.0     0.5123   0.4823±0.062     0.4912   +0.0211   healthy   f6g7h8i9

⭐ Best by CV  (recommended): α=10.0  CV R²=0.8401±0.018  Test R²=0.8538  (healthy)
   Best by Test (noisier):    α=10.0  Test R²=0.8538  CV R²=0.8401  (healthy)
```

In the MLflow UI, filter `tags.sweep = 'ridge_alpha_v2'` to see this sweep's runs. Sort by `metrics.cv_r2_mean DESC` to find the best alpha by CV.

### 3.3 — Query the experiment programmatically

```bash
python -m mlops.mlflow_query
```

Prints the leaderboard, isolates the sweep runs, and saves a sweep curve to `metrics/ridge_sweep.png`. This is the kind of analysis that becomes a one-line query when your runs are tracked — you do not need to click around in the UI for it.

### Comparison summary

| Script | What it adds |
|---|---|
| `mlflow_sweep.py` | One run per alpha, test R² only |
| `mlflow_sweep_v2.py` | + train R², 5-fold CV R², train-test gap, picks winner by CV (not test) |
| `mlflow_query.py` | Programmatic leaderboard + sweep curve plot |

### Resetting MLflow tracking (start clean)

After exploring a few sweeps and benchmarks, your dashboard will accumulate runs you no longer need. To wipe everything and start fresh, you must delete **both** stores:

- **`mlflow.db`** — the SQLite database holding run metadata, params, and metrics
- **`mlruns/`** — the filesystem directory holding logged model artifacts and plots

Orphaning one without the other leaves dead weight (artifact files with no parent run, or a DB referencing files that no longer exist).

**Step 1 — stop the UI first.** On Windows the running `mlflow ui` process holds a lock on `mlflow.db`; deleting it while the UI is up fails with "file in use." Press `Ctrl+C` in the terminal running `mlflow ui`.

**Step 2 — delete the stores from the project root:**

```powershell
# Windows PowerShell
Remove-Item -Force .\mlflow.db
Remove-Item -Recurse -Force .\mlruns
```

```bash
# macOS / Linux
rm -f mlflow.db
rm -rf mlruns
```

**Step 3 — restart the UI:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The next training/benchmark run will recreate `mlflow.db` and `mlruns/` automatically. All scripts (v1, v2, sweep, PyCaret) write to the same MLflow tracking store; the experiment names differ:

| Command | Experiment name |
|---|---|
| `python -m mlops.mlflow_train` | `M2_Pune_Real_Estate_Price` |
| `python -m mlops.mlflow_train_v2` | `M2_Pune_Real_Estate_Price` (registers model `pune_price_voting_regressor`) |
| `python -m mlops.mlflow_sweep` | `M2_Pune_Real_Estate_Price` (tag `sweep=ridge_alpha`) |
| `python -m mlops.mlflow_sweep_v2` | `M2_Pune_Real_Estate_Price` (tag `sweep=ridge_alpha_v2`) |
| `python -m mlops.pycaret_benchmark_v3` | `M2_Pune_Real_Estate_Price_PyCaret` (when `log_experiment=True`) |

> **Warning:** the reset is destructive — every prior run and logged model artifact is lost. The `model/*.pkl` files in your project root are separate and won't be touched.

---

## Section 4 — Clean up before DVC (~10 min, IMPORTANT)

Before DVC can take over data versioning, we need to clean up Git. During Labs 1–4 you accidentally committed several files that DVC will now manage. Leaving them in Git creates a conflict: two systems trying to track the same files, and Git always wins by default. DVC tracking won't kick in until Git lets go.

### What needs to leave Git

These files are declared as outputs of `dvc.yaml` stages (see Section 5). They must be removed from Git tracking but **kept on disk**:

**Outputs of the `features` stage:**

- `model_features.csv`
- `model_target.npy`
- `model/count_vectorizer.pkl`
- `model/sub_area_price_map.pkl`
- `model/amenities_score_price_map.pkl`
- `model/feature_cols.pkl`
- `model/all_feature_names.pkl`

**Outputs of the `train` stage:**

- `model/property_price_prediction_voting.sav`
- `model/interval_est.pkl`

**PyCaret outputs** (not in DVC pipeline, regenerable by running the script):

- `model/pycaret_pune_real_estate.pkl`
- `model/pycaret_pune_real_estate_DEPLOY.pkl`

### The cleanup commands

Run these from the project root, in order:

```bash
# 1. Stop any running mlflow ui / uvicorn / pycaret processes first (Ctrl+C in their terminals)

# 2. Remove the files from Git's index (keeps them on disk — note the --cached flag)
git rm --cached model_features.csv model_target.npy
git rm --cached model/count_vectorizer.pkl model/sub_area_price_map.pkl
git rm --cached model/amenities_score_price_map.pkl
git rm --cached model/feature_cols.pkl model/all_feature_names.pkl
git rm --cached model/property_price_prediction_voting.sav
git rm --cached model/interval_est.pkl
git rm --cached model/pycaret_pune_real_estate.pkl
git rm --cached model/pycaret_pune_real_estate_DEPLOY.pkl
```

On Windows PowerShell, the commands are identical (forward slashes work).

If any file is already missing from Git's index, `git rm --cached` will say "did not match any files" — that's fine, ignore the message.

### Update `.gitignore`

Add these patterns to `.gitignore` so the files stay out of future commits. If `.gitignore` does not exist yet, create it.

```gitignore
# Pipeline outputs — managed by DVC (declared in dvc.yaml outs)
model_features.csv
model_target.npy
data_cleaned.csv

# Anything inside model/ except the .dvc pointer files DVC may create
model/*.pkl
model/*.sav

# But keep DVC pointer files in Git — they're the contract between Git and DVC
!*.dvc
!.dvc/

# MLflow local stores
mlflow.db
mlruns/

# DVC local cache + workspace
.dvc/cache/
.dvc/tmp/

# Python
__pycache__/
*.pyc
.venv/

# OS noise
.DS_Store
Thumbs.db
```

### Commit the cleanup

```bash
git add .gitignore
git commit -m "Remove DVC-managed artifacts from Git tracking before DVC takes over"
```

Verify Git no longer tracks the files:

```bash
git ls-files | grep -E "model_features\.csv|\.pkl|\.sav"
# Should print nothing
```

The files are still on your disk:

```bash
ls model/
# Still see all the .pkl/.sav files
```

> **Note on Git history.** `git rm --cached` removes the files from your *current and future* commits, but they remain in your Git *history* (consuming repo space). For a learning project that's fine. To purge them from history entirely, you'd use `git filter-repo`, which is out of scope for this lab. If the repo is small (<100 MB), don't bother.

You're now ready for DVC to take over.

---

## Section 5 — DVC: version the data and the pipeline (~30 min)

You've been editing the trained model in place. That's fine for one engineer, but the moment you have a teammate, "the model" becomes a moving target. DVC fixes this with two ideas:

1. **Pointer files** — large data and model files get a tiny `.dvc` companion tracked in Git; the actual file lives in a remote.
2. **Pipeline reproduction** — `dvc.yaml` declares stages with deps and outs; `dvc repro` re-runs only the stages whose inputs changed.

### 5.1 — One-time setup

```bash
python -m mlops.dvc_init
```

This script:

1. Creates a Git repo if one isn't there (`.git/`)
2. Initializes DVC (`.dvc/`)
3. Runs `dvc add` on the raw input file (`Pune Real Estate Data.xlsx`)
4. Configures a **local-folder remote** at `./dvc_remote/` for the demo

> **What it does NOT do.** Earlier versions of `dvc_init.py` ran `dvc add` on the helper pkls and feature CSVs as well. Those are now **outputs of the `features` stage** in `dvc.yaml` — DVC tracks them automatically when you run `dvc repro`. Trying to `dvc add` a pipeline output is an error.

### 5.2 — The three-stage pipeline

Open `dvc.yaml`. Three stages declared in dependency order:

```
clean    raw Excel       → data_cleaned.csv
features data_cleaned    → model_features.csv + helper pkls (×5)
train    feature matrix  → trained model + interval estimate + metrics
```

See the dependency graph:

```bash
dvc dag
```

This is the structural upgrade over earlier versions of `dvc.yaml`, which only declared a `train` stage. Now the **entire path** from raw Excel to trained model is one DVC graph. Change anything → only the affected stages re-run.

### 5.3 — Reproduce the pipeline

```bash
dvc repro
```

First run executes all three stages from scratch. Watch the output:

```
Running stage 'clean':
> python -m mlops.clean_data
   …writes data_cleaned.csv

Running stage 'features':
> python -m mlops.build_features
   …writes model_features.csv, model_target.npy, model/*.pkl

Running stage 'train':
> python -m mlops.train
   …writes model/property_price_prediction_voting.sav, model/interval_est.pkl
   …writes metrics/train_metrics.json
```

Now run it again:

```bash
dvc repro
```

Nothing happens. DVC hashed every input — the Excel, every script, every parameter — compared to last time, found no changes, and skipped everything. *Caching by content hash.*

### 5.4 — Selective re-execution (the "Make for ML" demo)

```bash
touch mlops/build_features.py    # mark as modified
dvc repro
```

Only `features` and `train` re-run. The `clean` stage is skipped because nothing it depends on changed.

```bash
# Edit params.yaml: change ridge.alpha from 1.0 to 10.0
dvc repro
```

Only `train` re-runs. Cleaning and feature engineering are skipped because they don't depend on hyperparameters.

```bash
dvc metrics show
# ridge_alpha: 10.0   r2: 0.85387   rmse: 15.64829   ...

dvc metrics diff HEAD~1   # if you committed the previous run
# Shows the metric differences between commits
```

Restore α=1.0 and `dvc repro` once more to leave the project in a clean state.

### 5.5 — Time travel for data

This is the demo that makes DVC click.

```bash
# Make a change, run the pipeline, commit
# Edit params.yaml: ridge.alpha → 10.0
dvc repro
git add params.yaml dvc.lock metrics/train_metrics.json
git commit -m "Tune alpha to 10.0"
cat metrics/train_metrics.json
# Shows the new metrics (alpha=10)

# Now travel backward
git checkout HEAD~1
dvc checkout
cat metrics/train_metrics.json
# Shows the OLD metrics — the file on disk physically reverted
```

You went back one commit. The metrics file, the trained model, the pickled vectorizer — all *physically reverted* to the bytes they had before. Time travel for data.

```bash
# Go forward again
git checkout main
dvc checkout
```

### 5.6 — Push to the local remote

```bash
dvc push
```

The actual files (CSVs, pkls, .sav) are uploaded to the local remote at `./dvc_remote/`. Browse it:

```bash
ls dvc_remote/
# Files are stored under their content hash, not their original names
```

For a teammate to reproduce your work from a fresh clone:

```bash
git clone <your-repo-url>
cd <repo>
dvc pull           # downloads all the actual data from the remote
dvc repro          # reproduces the pipeline to identical numbers
```

That's the full DVC lifecycle.

---

## Section 6 — DagsHub: collaboration (~20 min)

Local DVC + local MLflow is fine for one engineer. For a team, you want remote storage with a UI. **DagsHub** provides Git + MLflow + DVC hosted in one platform — free tier is enough for this project.

### 6.1 — Sign up and create the repo

1. Sign up at **https://dagshub.com** (you can use your GitHub login)
2. Click "Create" → **"Connect a repository"** → select your existing GitHub repo
   - DagsHub will *mirror* your GitHub repo. Same code, same commits, same history. You don't migrate — you augment.
3. Generate an access token at **Profile → Settings → Tokens**
4. Note your repo's three URLs:
   - Git: `https://dagshub.com/<user>/<repo>.git`
   - MLflow: `https://dagshub.com/<user>/<repo>.mlflow`
   - DVC: `https://dagshub.com/<user>/<repo>.dvc`

### 6.2 — Set environment variables

```bash
# Linux / macOS
export DAGSHUB_USER=your_username
export DAGSHUB_TOKEN=your_token
export DAGSHUB_REPO=pune-real-estate-mlops

# Windows PowerShell
$env:DAGSHUB_USER  = "your_username"
$env:DAGSHUB_TOKEN = "your_token"
$env:DAGSHUB_REPO  = "pune-real-estate-mlops"
```

Verify:

```bash
python -m mlops.dagshub_setup --check
```

You should see three ✅. If not, fix the env vars and re-run.

### 6.3 — Wire up the DVC remote

```bash
python -m mlops.dagshub_setup --print-dvc-cmds
```

Copy the four `dvc remote ...` commands it prints and run them. They:

1. Add `origin` as the default DVC remote pointing at DagsHub
2. Configure basic auth with your username and token

Then push:

```bash
dvc push
git push -u origin main
```

### 6.4 — Wire up the MLflow tracking server

The MLflow scripts in `mlops/` honour `MLFLOW_TRACKING_URI` from the environment. Set it for the current shell:

```bash
python -m mlops.dagshub_setup --configure
```

Re-run a trainer in the *same* shell:

```bash
python -m mlops.mlflow_train_v2
```

Open `https://dagshub.com/<user>/<repo>` and click the **Experiments** tab. Your run is there — visible to anyone with access to the repo. The same code logged to a different backend, with zero code changes, because the URI is environment-driven.

### 6.5 — Tour the DagsHub UI

Click through the tabs:

- **Files** — your code, README, params.yaml. Looks like GitHub.
- **Data files** — your `model_features.csv`, model artifacts. Downloadable. Versioned.
- **Experiments** — every MLflow run. Sortable, filterable, hosted, permanent.
- **Models** — the registered `pune_price_voting_regressor` with its versions and stages.

Code, data, experiments, registered models. **Four tabs. One URL. One login.**

### 6.6 — The collaboration test

To prove the loop works, simulate a teammate. In a new directory:

```bash
git clone https://dagshub.com/$DAGSHUB_USER/$DAGSHUB_REPO
cd $DAGSHUB_REPO
pip install -r requirements.txt -r requirements-mlops.txt
dvc pull
python -m mlops.train
cat metrics/train_metrics.json
# Same metrics as your original
```

Three commands and a `pip install`. No setup wiki. No credential negotiation.

### 6.7 — Permanent setup

To make MLflow always log to DagsHub from this project, export the env vars in your shell startup (`.bashrc` / `.zshrc` / PowerShell profile) or use a `.env` file with a tool like `direnv`.

---

## Section 7 — End-to-end recap

Here's what your project does now that didn't before Lab 5:

```
┌─────────────────────────────────────────────────────────────────────┐
│  YOU edit params.yaml                                                │
│  $ vim params.yaml          # change ridge.alpha 1.0 → 10.0         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  $ dvc repro                                                         │
│   → DVC sees ridge.alpha changed                                     │
│   → re-runs ONLY the train stage (clean and features are skipped)    │
│   → writes new model/property_price_prediction_voting.sav            │
│   → writes new metrics/train_metrics.json                            │
│   → updates dvc.lock                                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  $ python -m mlops.mlflow_train_v2                                   │
│   → logs the new run to MLflow with train + test metrics + plots     │
│   → registers a new model version under pune_price_voting_regressor  │
│   → if better, promote it to Production in the UI                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  $ uvicorn src.app:app --reload                                      │
│   → src/inference.py loads models:/pune_price_voting_regressor/Prod  │
│   → API serves predictions from the promoted version                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  $ git add . && git commit -m "Tune ridge alpha → 10.0"              │
│  $ git push                # code change goes to DagsHub             │
│  $ dvc push                # new model + metrics go to DagsHub       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TEAMMATE on a different machine                                     │
│  $ git clone … && dvc pull                                           │
│  → Gets your code, your data, your trained model.                    │
│  → `dvc repro` reproduces your numbers.                              │
│  → MLflow UI at dagshub.com shows the run history.                   │
└─────────────────────────────────────────────────────────────────────┘
```

That's reproducible MLOps in five tools (Git, DVC, MLflow, DagsHub, FastAPI).

---

## Try on your own

1. **Add a `pycaret_benchmark` stage to `dvc.yaml`.** Make it depend on `model_features.csv` and `model_target.npy`; declare `metrics/pycaret_benchmark.json` as a metric. Now `dvc repro` includes AutoML benchmarking.
2. **Tag every MLflow run with the DVC data hash.** Read `dvc.lock`, extract the hash for `model_features.csv`, and call `mlflow.set_tag("data_hash", hash)`. Every model is now traceable to the exact data version it was trained on.
3. **Wire the FastAPI app to the Model Registry.** Replace `joblib.load("model/...")` in `src/inference.py` with `mlflow.pyfunc.load_model("models:/pune_price_voting_regressor/Production")`. Promote a new version in the UI, restart FastAPI, see predictions change with no code edit.
4. **Use DVC `plots`.** Write per-fold metrics as a CSV from `mlops/train.py`, declare it under `plots:` in `dvc.yaml`, and use `dvc plots show` for a one-command visualization. `dvc plots diff` compares two commits visually.
5. **Add `pycaret_benchmark_v3` to the pipeline** so the deployment model is auto-rebuilt whenever data changes. (Watch for runtime — PyCaret is slow.)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `python -m mlops.train` says "Missing required input file(s)" | `model_features.csv` or `model_target.npy` not in the project root | Run `dvc repro` to regenerate them, or `dvc pull` if you've configured a remote |
| `dvc repro` says "failed to reproduce 'train'" with import errors | `requirements-mlops.txt` not installed | `pip install -r requirements-mlops.txt` |
| `dvc add model/foo.pkl` says "output 'model/foo.pkl' is already specified in 'dvc.yaml'" | The file is a pipeline output, not an input — DVC tracks it via the stage definition | Don't `dvc add` pipeline outputs. `dvc repro` is enough. |
| MLflow UI shows no runs | You ran the script in one shell and `mlflow ui` in another with a different working dir | Run both from the project root, or pass `--backend-store-uri sqlite:///<absolute-path>/mlflow.db` |
| `dvc push` says "Authentication failed" | Wrong DagsHub token, or token expired | Regenerate a token, re-run `python -m mlops.dagshub_setup --print-dvc-cmds`, re-run the four `dvc remote` commands |
| `pip install pycaret` fails | scikit-learn version conflict, or Python 3.13+ | PyCaret needs Python 3.9–3.12; check your Python version |
| FastAPI returns 500 after `dvc repro` | New model has a different feature count than `count_vectorizer.pkl` expects | DVC tracks helpers separately from the model — make sure you didn't manually delete a `.pkl`. Run `dvc repro --force` to regenerate everything. |
| `git rm --cached` says "did not match any files" | The file is already untracked or already gitignored | Safe to ignore — it just means the cleanup is a no-op for that file |
| `mlflow_train_v2.py` doesn't show up in the Models tab | Model registry only initializes after first successful `log_model` with `registered_model_name` | Re-run the script; if it still doesn't show up, check your MLflow version (3.x required) |

---

**Module 2 complete.** You now have a project that handles a full ML lifecycle: clean → engineer → train → serve → version → track → collaborate. Module 3 picks up by deploying this exact project to AWS.
