# Customer Churn Prediction System

A production-grade ML system for predicting customer churn in subscription businesses.
Includes a full training pipeline, REST API, and interactive Streamlit dashboard.

---

## Architecture Overview

```
Raw Data
   │
   ▼
DataProcessor          ← cleaning, splitting
   │
   ▼
FeatureEngineer        ← encoding, scaling, SMOTE
   │
   ▼
ModelTrainer           ← XGBoost / LightGBM / LogReg
   │
   ▼
HyperparamOptimizer    ← Optuna Bayesian search
   │
   ▼
Best Model  ──────┬──────────────────────────┐
                  ▼                          ▼
          FastAPI REST API          Streamlit Dashboard
```

---

## Project Structure

```
customer-churn-ml/
│
├── data/
│   └── generate_data.py      # Synthetic dataset generator
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Cleaning & train/val/test splitting
│   ├── feature_engineering.py# Encoding, scaling, SMOTE
│   ├── train_model.py        # Training & cross-validation
│   ├── optimize_model.py     # Optuna hyperparameter search
│   ├── predict.py            # Inference engine + SHAP explanations
│   └── run_training.py       # End-to-end pipeline orchestrator
│
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI application
│   └── schemas.py            # Pydantic request / response models
│
├── dashboard/
│   └── app.py                # Streamlit dashboard
│
├── models/                   # Serialised model artefacts (git-ignored)
├── validate.py               # Data & model quality gates
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python data/generate_data.py --rows 50000 --seed 42
```

### 3. Train the model

```bash
python src/run_training.py --seed 42
```

### 4. Validate data & model

```bash
python validate.py --mode all
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Key ML Choices

| Concern | Choice | Reason |
|---|---|---|
| Primary model | XGBoost / LightGBM | Best AUC on tabular churn data |
| Class imbalance | SMOTE + class weights | ~20 % churn rate |
| HPO | Optuna TPE | Sample-efficient Bayesian search |
| Explainability | SHAP TreeExplainer | Fast, exact for tree models |
| Experiment tracking | MLflow | Industry standard |
| Serving | FastAPI | Async, auto-docs, type-safe |

---

## Evaluation Metrics

- **ROC-AUC** — primary ranking metric (target ≥ 0.80)
- **PR-AUC** — important for imbalanced classes (target ≥ 0.55)
- **F1 @ 0.5 threshold** — balanced precision / recall
- **Profit curve** — business-value-aware threshold selection

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PREDICTION_MODE` | `direct` | `direct` or `api` for dashboard |
| `API_URL` | `http://localhost:8000` | FastAPI base URL |
| `MODEL_PATH` | `models/best_model.joblib` | Path to model artefact |
| `PIPELINE_PATH` | `models/feature_pipeline.joblib` | Path to feature pipeline |
| `MLFLOW_TRACKING_URI` | `mlruns` | MLflow tracking directory |

---

## Development Roadmap

- [ ] Step 1 — Project setup ✅
- [ ] Step 2 — Synthetic data generation
- [ ] Step 3 — Data processing pipeline
- [ ] Step 4 — Feature engineering pipeline
- [ ] Step 5 — Model training & evaluation
- [ ] Step 6 — Hyperparameter optimisation
- [ ] Step 7 — FastAPI service
- [ ] Step 8 — Streamlit dashboard
- [ ] Step 9 — Validation suite
- [ ] Step 10 — Docker & deployment
