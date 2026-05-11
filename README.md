# Pune Real Estate Price Prediction — End-to-End MLOps Project

A complete machine-learning project on a real-world dataset of 200 residential
properties in Pune, India. This repo is the consolidated output of Module 2
of the MLOps course:

| Lab | What it produces |
|---|---|
| Lab 1 | Cleaned dataset (`data_cleaned.csv`) — *notebook, not in this repo* |
| Lab 2 | Feature matrix (`model_features.csv`, `model_target.npy`) + helper artifacts (`model/*.pkl`) |
| Lab 3 | Trained Voting Regressor (`model/property_price_prediction_voting.sav`) + 95% interval estimator |
| **Lab 4** | **FastAPI prediction service (`src/`) + browser frontend (`frontend/`)** |
| **Lab 5** | **MLOps tooling — PyCaret / MLflow / DVC / DagsHub (`mlops/`, `dvc.yaml`, `params.yaml`)** |

---
```bash

py --list
winget install --id Python.Python.3.11
cd "C:\venvs"
py -3.11 -m venv mlops-pune-price
cd "c:\Users\prash\OneDrive\2026\Training\K21Academy\MLOps\Module 2\MLOPS_Pune_Price_Prediction_Project"
C:\venvs\mlops-pune-price\Scripts\Activate.ps1
pip install requirements.txt
pip install requirements-mlops.txt

```




## Project structure

```
Pune_Price_Prediction_Project/
├── frontend/                       # Lab 4 — browser UI
│   ├── index.html
│   ├── results.html
│   ├── script.js
│   └── style.css
│
├── src/                            # Lab 4 — FastAPI service
│   ├── app.py                      # endpoints
│   ├── inference.py                # model loading + prediction pipeline
│   ├── schemas.py                  # Pydantic input/output schemas
│   └── test_api.py                 # client tests
│
├── model/                          # Trained artifacts (Labs 1-3 outputs)
│   ├── property_price_prediction_voting.sav   # ← DVC pipeline output
│   ├── interval_est.pkl                       # ← DVC pipeline output
│   ├── count_vectorizer.pkl                   # ← DVC-tracked
│   ├── sub_area_price_map.pkl                 # ← DVC-tracked
│   ├── amenities_score_price_map.pkl          # ← DVC-tracked
│   ├── all_feature_names.pkl                  # ← DVC-tracked
│   └── feature_cols.pkl                       # ← DVC-tracked
│
├── mlops/                          # Lab 5 — all MLOps scripts
│   ├── utils.py                    # shared helpers (data, scoring, training)
│   ├── train.py                    # plain trainer (the one DVC drives)
│   ├── mlflow_train.py             # train + log everything to MLflow
│   ├── mlflow_sweep.py             # one MLflow run per Ridge alpha
│   ├── mlflow_query.py             # search_runs leaderboard + plot
│   ├── pycaret_benchmark.py        # AutoML benchmark vs the Lab 3 model
│   ├── dagshub_setup.py            # env-driven DagsHub config helper
│   └── dvc_init.py                 # one-time DVC initialization
│
├── metrics/                        # DVC-readable JSON metrics + plots
├── model_features.csv              # ← DVC-tracked input
├── model_target.npy                # ← DVC-tracked input
├── dvc.yaml                        # pipeline definition
├── params.yaml                     # all hyperparameters in one place
├── requirements.txt                # production + notebook deps (Labs 1-4)
├── requirements-mlops.txt          # MLOps tooling deps (Lab 5)
├── .gitignore
├── .dvcignore
├── README.md                       # this file
└── MLOPS_LAB.md                    # step-by-step Lab 5 guide
```

---

## Quick start

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd Pune_Price_Prediction_Project

python -m venv .venv
# Activate:
#   Windows PowerShell : .venv\Scripts\Activate.ps1
#   Windows Cmd        : .venv\Scripts\activate
#   macOS / Linux      : source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the FastAPI service (Lab 4)

```bash
uvicorn src.app:app --reload
# Open http://localhost:8000/docs for the interactive API
# Open frontend/index.html in a browser to use the UI
```

Test it:

```bash
python src/test_api.py
```

### 3. Set up the MLOps tooling (Lab 5)

```bash
pip install -r requirements-mlops.txt
```

Then follow [`MLOPS_LAB.md`](./MLOPS_LAB.md) for the full step-by-step walkthrough.
The TL;DR:

```bash
# Initialize DVC + Git tracking on this project (one-time)
python -m mlops.dvc_init

# Run the MLOps pipeline through DVC
dvc repro                             # train.py runs, model/ is updated
dvc metrics show                      # see the latest test-set metrics

# Track experiments in MLflow
python -m mlops.mlflow_train          # log one careful run
python -m mlops.mlflow_sweep          # log one run per ridge alpha
python -m mlops.mlflow_query          # leaderboard + plot
mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://localhost:5000

# AutoML benchmark — does our model match what AutoML can produce?
python -m mlops.pycaret_benchmark

# To use DagsHub instead of local MLflow + DVC, set credentials:
export DAGSHUB_USER=...   DAGSHUB_TOKEN=...   DAGSHUB_REPO=...
python -m mlops.dagshub_setup --check
python -m mlops.dagshub_setup --print-dvc-cmds
```

### 4. Resetting MLflow tracking (start clean)

Want a fresh MLflow dashboard? You need to delete **both** stores — the
SQLite DB (run metadata) and the `mlruns/` directory (logged model
artifacts and plots). Orphaning one without the other leaves dead weight
and broken artifact links.

**Step 1 — stop the UI first.** On Windows the running `mlflow ui`
process holds a lock on `mlflow.db`; deleting it while the UI is up
fails with "file in use." `Ctrl+C` in that terminal.

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

The next training/benchmark run will recreate `mlflow.db` and
`mlruns/` automatically. After this reset, all three logging sources
write to the same store:

| Command | Experiment name |
|---|---|
| `python -m mlops.mlflow_train` | `M2_Pune_Real_Estate_Price` |
| `python -m mlops.mlflow_train_v2` | `M2_Pune_Real_Estate_Price_v2` |
| `python -m mlops.pycaret_benchmark_v3` | `M2_Pune_Real_Estate_Price_PyCaret` |

**Warning:** this is destructive — every prior run and logged model
artifact is lost. The `model/*.pkl` files in your repo are separate
and won't be touched.

---

## Configuration

All hyperparameters live in **`params.yaml`** at the project root. Edit
values there and DVC will detect the change and re-train on the next
`dvc repro`. Code reads these via `mlops.utils.load_params()`.

```yaml
data:
  test_size: 0.2
  random_state: 42

ridge:
  alpha: 1.0

lasso:
  alpha: 0.1
  max_iter: 10000

interval:
  confidence: 0.95

sweep:
  alphas: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

mlflow:
  experiment_name: M2_Pune_Real_Estate_Price
```

---

## Reproducibility — where each piece lives

| Artifact | Storage | Versioned by |
|---|---|---|
| Source code | Git | Git |
| `model_features.csv`, `model_target.npy` | DVC remote | DVC content hash |
| `model/*.pkl` (helpers) | DVC remote | DVC content hash |
| `model/*.sav` (trained model) | DVC remote, declared as `train` stage output | DVC pipeline output |
| Hyperparameters, run metadata | `params.yaml` (Git) + MLflow run | Git + MLflow run_id |
| Experiment tracking | `mlflow.db` locally, DagsHub remotely | MLflow run_id |
| Pipeline definition | `dvc.yaml` (Git) | Git |

A teammate can reproduce the entire project with:

```bash
git clone <repo>
cd Pune_Price_Prediction_Project
pip install -r requirements.txt -r requirements-mlops.txt
dvc pull          # fetches data + model from the DVC remote
dvc repro         # re-runs the pipeline → identical numbers
```

---

## Test results (default `params.yaml`)

| Metric | Value |
|---|---|
| Test R² | 0.852 |
| Test RMSE | ₹15.75 lakhs |
| Test MAE | ₹11.46 lakhs |
| 95% prediction interval width | ±₹31.35 lakhs |
| Train / test split | 159 / 40 |

A small Ridge α-sweep (α ∈ {0.01 … 1000}) finds α=10.0 marginally better
(R² = 0.854). Run `python -m mlops.mlflow_sweep` followed by
`python -m mlops.mlflow_query` to verify on your machine.

---

## License

For training/educational use as part of the MLOps course (Module 2).
Pune real estate dataset is proprietary and not redistributed; bring
your own copy if you want to re-run Labs 1 and 2 from scratch.
