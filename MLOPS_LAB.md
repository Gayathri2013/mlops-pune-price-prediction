# Module 2 — Lab 5: MLOps Tools (Hands-On Project)

> Lab 5 of the MLOps course. The companion notebook
> (`M2_Lab5_MLOps_Tools_Primers.ipynb`) is the conceptual reference;
> **this document is what you do at the keyboard.**

By the end of this lab, your Pune Price Prediction project — the same one you
deployed in Lab 4 — will have:

- A reproducible training pipeline driven by **DVC**
- Every training run logged to **MLflow** with parameters, metrics, model, and plots
- A **PyCaret** AutoML benchmark proving (or refuting) that your manual model is competitive
- Credentials and commands ready to push everything to **DagsHub** for team collaboration

The work is split into 6 sections. Each one is a self-contained chunk you
can do in 15–25 minutes. Total time: ~2 hours including reading.

---

## Prerequisites

- You finished **Labs 1, 2, 3, and 4** of Module 2 — the FastAPI app in `src/`
  loads and serves predictions correctly.
- Python 3.10 or newer is on PATH.
- `git` is on PATH.
- A clean shell (terminal / PowerShell) opened at the project root.

Sanity check before starting:

```bash
python --version          # 3.10+
git --version             # any
ls model/                 # should show 7 .pkl/.sav files
ls model_features.csv     # should be present
uvicorn src.app:app --reload   # the Lab 4 API still serves
```

If anything above fails, fix it before continuing.

---

## Section 0 — Install MLOps tooling

The Lab 4 service requires the packages in `requirements.txt`. Lab 5 adds a
heavier set — kept in a separate file so the production API container stays small.

```bash
# Activate your venv first
pip install -r requirements-mlops.txt
```

PyCaret pulls XGBoost, LightGBM, CatBoost, Yellowbrick, and a pinned
scikit-learn. **First-time install takes 2–3 minutes.** That's normal.

Verify everything imports:

```bash
python -c "import mlflow, dvc, pycaret, dagshub; print('OK')"
```

---

## Section 1 — PyCaret AutoML Benchmark (~25 min)

The first MLOps question worth answering on any project is:

> **What R² is even possible on this dataset?**

If 20 well-tuned algorithms top out at R² = 0.65, your manual pipeline is
unlikely to hit 0.85. Conversely, if PyCaret reaches 0.92 in two minutes,
your Lab 3 model has clear room to grow.

### Run the benchmark

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

### Expected output

A leaderboard of ~15 algorithms ranked by R², followed by:

```
┌─────────────────────────────────────────────────────────────┐
│ Model                              RMSE (₹L)             R² │
├─────────────────────────────────────────────────────────────┤
│ Lab 3 manual VotingRegressor          15.75         0.8519 │
│ PyCaret final XGBRegressor            14.12         0.8810 │
└─────────────────────────────────────────────────────────────┘
   Verdict: PyCaret wins by ΔR² = +0.0291
```

Numbers will vary by run (PyCaret's tune step uses random search). What
matters is the gap. A sub-0.03 gap means your manual pipeline is doing
its job; a 0.10+ gap means a tree-based model would be a clear upgrade.

### Why Lab 3 vs PyCaret on the same split is fair

- Same `X_train`, `X_test` (identical `random_state`)
- Same `y_train`, `y_test`
- Both report metrics on the held-out 40 rows

### What this is NOT

PyCaret hides preprocessing decisions (encoding, scaling, outlier handling)
in `setup()`. Use it to **set the bar**, not to replace your transparent
pipeline. If you adopt PyCaret's model, audit its preprocessing first.

---

## Section 2 — MLflow: log one model carefully (~15 min)

`mlops/mlflow_train.py` does the same training as `mlops/train.py` but wraps
it in `mlflow.start_run()` and logs:

- Every hyperparameter (ridge_alpha, lasso_alpha, n_features, n_train, n_test, …)
- All test-set metrics (rmse, mae, r2, interval_margin)
- Diagnostic plot (residuals + predicted-vs-actual)
- The model itself with a signature and input example
- Free-form tags (module, dataset, candidate_for_registry)

### Run it

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

### View it in the UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

In the UI:
- Click the run name
- See params, metrics, tags, the diagnostic plot, and the model signature
- The **Model URI** (`runs:/<id>/model`) is your permanent address for that
  artifact — it works in `mlflow.sklearn.load_model(...)` from any machine.

---

## Section 3 — MLflow: sweep Ridge alphas (~15 min)

In Lab 3 you tuned Ridge α manually with a `for` loop printing to stdout.
With MLflow, **every alpha becomes its own run** — and you can compare them
visually in the UI.

### Run the sweep

```bash
python -m mlops.mlflow_sweep
```

This loops over `params.yaml sweep.alphas` (default: 6 values from 0.01 to
1000) and creates one MLflow run per alpha. Each run is tagged with
`sweep=ridge_alpha` so you can filter just these later.

### Query the experiment programmatically

```bash
python -m mlops.mlflow_query
```

Prints the leaderboard, isolates the sweep runs, and saves a sweep curve to
`metrics/ridge_sweep.png`. This is the kind of analysis that becomes a
one-line query when your runs are tracked.

### Inspect in the UI

In the MLflow UI, select all 6 sweep runs, click **Compare**, and look at the
parallel-coordinates plot. You'll see the R²-vs-α curve at a glance.

> **What you should find on the default dataset:** α=10.0 marginally beats
> α=1.0 (R² ~0.854 vs ~0.852). It's a tiny gap, but real — and only visible
> *because* you tracked all the runs.

---

## Section 4 — DVC: version the data and the pipeline (~25 min)

You've been editing the trained model in place. That's fine for one
engineer, but the moment you have a teammate, "the model" becomes a moving
target. DVC fixes this with two ideas:

1. **Pointer files** — every data/model file gets a tiny `.dvc` companion
   tracked in Git; the actual file lives in a remote.
2. **Pipeline reproduction** — `dvc.yaml` declares stages with deps and
   outs; `dvc repro` re-runs only the stages whose inputs changed.

### One-time setup

```bash
python -m mlops.dvc_init
```

This script:

1. Creates a Git repo if one isn't there (`.git/`)
2. Initializes DVC (`.dvc/`)
3. Runs `dvc add` on the 7 input/auxiliary artifacts:
   - `model_features.csv`, `model_target.npy`
   - `model/count_vectorizer.pkl`, `model/sub_area_price_map.pkl`,
     `model/amenities_score_price_map.pkl`, `model/all_feature_names.pkl`,
     `model/feature_cols.pkl`
4. Configures a **local-folder remote** at `./dvc_remote/` for the demo

The two trained model files (`property_price_prediction_voting.sav`,
`interval_est.pkl`) are **deliberately not** added by this script — they're
declared as **outputs** of the `train` stage in `dvc.yaml` and tracked by
the pipeline.

### Commit the pointers

```bash
git add *.dvc model/*.dvc model/.gitignore .gitignore .dvcignore .dvc/config dvc.yaml params.yaml
git add mlops/ src/ frontend/ requirements*.txt README.md MLOPS_LAB.md
git commit -m "Initial commit: Pune Price Prediction with DVC + MLflow"
```

### Push the data to the remote

```bash
dvc push
```

Inspect the remote — every file is stored under its content hash:

```bash
ls dvc_remote/
```

### Run the pipeline through DVC

```bash
dvc repro
```

This:
- Reads `dvc.yaml`
- Sees that the `train` stage's deps haven't changed yet → does nothing
  (or runs once on a fresh checkout)
- Updates `dvc.lock` with the exact hashes of inputs and outputs

Now change something:

```bash
# Edit params.yaml → change ridge.alpha from 1.0 to 10.0
dvc repro
```

DVC detects the param change, re-runs `mlops/train.py`, writes a new
model and new metrics. You'll see:

```bash
dvc metrics show
# ridge_alpha: 10.0   r2: 0.85387   rmse: 15.64829   ...
```

Restore α=1.0 and `dvc repro` once more to leave the project in a clean state.

### See the dependency graph

```bash
dvc dag
```

Reads `dvc.yaml` and prints the DAG. With one `train` stage, it's small.
In a fully scripted project (clean → features → train → evaluate) it'd
have 4 nodes.

---

## Section 5 — DagsHub: collaboration (~15 min)

Local DVC + local MLflow is fine for one engineer. For a team, you want
remote storage. **DagsHub** provides Git + MLflow + DVC hosted in one
platform — free tier is enough for this project.

### Sign up + get credentials

1. Sign up at **https://dagshub.com** (you can use your GitHub login)
2. Create a new repository named e.g. `pune-real-estate-mlops`
3. Generate an access token at **Profile → Settings → Tokens**
4. Note your repo's three URLs:
   - Git: `https://dagshub.com/<user>/<repo>.git`
   - MLflow: `https://dagshub.com/<user>/<repo>.mlflow`
   - DVC: `https://dagshub.com/<user>/<repo>.dvc`

### Set environment variables

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

### Verify

```bash
python -m mlops.dagshub_setup --check
```

You should see three ✅. If not, fix the env vars and re-run.

### Wire up the DVC remote

```bash
python -m mlops.dagshub_setup --print-dvc-cmds
```

Copy the four `dvc remote ...` commands it prints and run them. They:

1. Add `origin` as the default remote pointing at DagsHub
2. Configure basic auth with your username and token

Then push:

```bash
dvc push    # data + helpers go to DagsHub
git remote add origin https://dagshub.com/$DAGSHUB_USER/$DAGSHUB_REPO.git
git push -u origin main
```

### Wire up the MLflow tracking server

The MLflow scripts in `mlops/` honour `MLFLOW_TRACKING_URI` from the
environment. If you set the three env vars above and then run:

```bash
python -m mlops.dagshub_setup --configure
```

…the URI and auth env vars are set for the current Python process.
Re-run the trainer in the *same* shell:

```bash
python -m mlops.mlflow_train
```

Now open `https://dagshub.com/<user>/<repo>` and check the **Experiments** tab.
Your run is there — visible to anyone with access to the repo.

### Permanent setup

To make MLflow always log to DagsHub from this project, export the env
vars in your shell startup (`.bashrc` / `.zshrc` / PowerShell profile)
or use a `.env` file with a tool like `direnv`.

---

## Section 6 — End-to-end recap

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
│   → re-runs mlops/train.py                                           │
│   → writes new model/property_price_prediction_voting.sav            │
│   → writes new metrics/train_metrics.json                            │
│   → updates dvc.lock                                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  $ python -m mlops.mlflow_train                                      │
│   → logs the new run to MLflow with full params + metrics + plots    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  $ uvicorn src.app:app --reload                                      │
│   → src/inference.py loads the NEW model/*.sav                       │
│   → API serves predictions from the new model immediately            │
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
│  $ git pull && dvc pull                                              │
│  → Gets your code, your data, your trained model.                    │
│  → `dvc repro` reproduces your numbers.                              │
│  → MLflow UI at dagshub.com shows the run history.                   │
└─────────────────────────────────────────────────────────────────────┘
```

That's reproducible MLOps in five tools (Git, DVC, MLflow, DagsHub, FastAPI).

---

## Try on your own

1. **Add a `clean` and `features` stage to `dvc.yaml`** that wrap the Lab 1
   and Lab 2 notebooks with `jupyter nbconvert --execute`. Now `dvc repro`
   reproduces the *entire* workflow from raw Excel to deployed model.
2. **Tag every MLflow run with the DVC data hash.** Read `dvc.lock`,
   extract the hash for `model_features.csv`, and call
   `mlflow.set_tag("data_hash", hash)`. Now every model is traceable to
   the exact data version it was trained on.
3. **Promote the best run to a Model Registry.** In the MLflow UI, click
   **Register Model** on your best run, name it
   `pune_real_estate_price`, and mark version 1 as **Production**. Update
   `src/inference.py` to load via
   `mlflow.sklearn.load_model("models:/pune_real_estate_price/Production")`
   instead of `joblib.load("model/...")`.
4. **Use PyCaret's MLflow integration.** Add `log_experiment="mlflow"` to
   the `setup()` call in `mlops/pycaret_benchmark.py`. Every algorithm in
   `compare_models()` will land in MLflow as its own run — instant
   leaderboard with no manual logging.
5. **Use DVC `plots`.** Write per-fold metrics as a CSV from `mlops/train.py`,
   declare it under `plots:` in `dvc.yaml`, and use
   `dvc plots show` for a one-command visualization.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `python -m mlops.train` says "Missing required input file(s)" | `model_features.csv` or `model_target.npy` not in the project root | Re-run Lab 2, or `dvc pull` if you've configured a remote |
| `dvc repro` says "ERROR: failed to reproduce 'train'" with import errors | `requirements-mlops.txt` not installed | `pip install -r requirements-mlops.txt` |
| MLflow UI shows no runs | You ran the script in one shell and `mlflow ui` in another with a different working dir | Run both from the project root, or pass `--backend-store-uri sqlite:///<absolute-path>/mlflow.db` |
| `dvc push` says "Authentication failed" | Wrong DagsHub token, or token expired | Regenerate a token, re-run `python -m mlops.dagshub_setup --print-dvc-cmds`, run the four commands |
| `pip install pycaret` fails | scikit-learn version conflict, or Python 3.13+ | PyCaret needs Python 3.9–3.12; check your Python version |
| FastAPI returns 500 after `dvc repro` | New model has a different feature count than `count_vectorizer.pkl` expects | DVC tracks helpers separately from the model — make sure you didn't manually delete a `.pkl` |

---

**Module 2 complete.** You now have a project that handles a full ML
lifecycle: clean → engineer → train → serve → version → track → collaborate.
Module 3 picks up by deploying this exact project to AWS.
