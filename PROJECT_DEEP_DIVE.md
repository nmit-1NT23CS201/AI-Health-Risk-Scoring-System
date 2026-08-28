# Project Deep Dive

## 1. Executive Summary

This repository implements an AI-powered health risk scoring prototype for cardiovascular risk estimation. The system combines a machine learning model, explainability tooling, a FastAPI backend, and a Next.js dashboard to let a user enter health-related parameters, get a risk score, view SHAP-based contributors, and receive personalized suggestions.

The codebase is a working prototype rather than a production healthcare system. It uses a synthetic dataset in `dataset/indian_health_risk_dataset.csv`, trains a `RandomForestRegressor` pipeline, saves the model to `models/random_forest_model.pkl`, exposes predictions via `api_server.py`, and renders the interface using `app/page.tsx` (Next.js) and `app.py` (Streamlit, alternative UI). The backend also stores assessment history in SQLite via `src/database.py`.

The project is clearly oriented toward demo/prototype use and academic validation rather than production deployment. There is no authentication layer, no CI/CD pipeline, no Docker configuration, no deployment manifest, and no formal automated test suite found in the repository.

## 2. Project at a Glance

| Item | Evidence in repository |
|---|---|
| Primary purpose | Predict cardiovascular / health risk using ML and explainability |
| Data source | `dataset/indian_health_risk_dataset.csv` |
| Model type | `RandomForestRegressor` in `src/train_model.py` and `src/utils.py` |
| Explainability | SHAP via `src/shap_explainer.py` and `src/visualization.py` |
| Backend | FastAPI in `api_server.py` |
| Frontend | Next.js app in `app/page.tsx`; Streamlit app in `app.py` |
| Storage | SQLite database at `database/healthrisk.db` |
| Persistence | `src/database.py` with `assessments` table |
| Model artifact | `models/random_forest_model.pkl` |
| Deployment config | Not found |
| Test suite | Not found |
| Auth/authorization | Not implemented |
| Environment files | None found in repo root |

## 3. Problem Statement

The project addresses the need for a transparent, low-friction health-risk assessment tool that can:

- accept clinical and lifestyle indicators,
- estimate a numeric health risk score,
- classify the result into Low/Medium/High risk bands,
- show which features most influenced the result,
- recommend likely next steps.

The repository is explicit that the project is a prototype for preventive healthcare analytics and explainable AI. The design documentation files in `docs/` describe it as a proof-of-concept and a sensor-augmented personalized risk scoring system using explainable machine learning.

The system is aimed at demo use, educational demonstration, and internal prototype evaluation. It is not framed as a regulated medical decision-support product.

## 4. Core Features

The repository implements the following core capabilities:

1. Health risk prediction
   - Based on user inputs such as age, BMI, blood pressure, cholesterol, smoking status, physical activity, family history, heart rate, and HRV values.
   - Implemented in `src/feature_engineering.py`, `src/train_model.py`, `src/utils.py`, and `api_server.py`.

2. Risk classification
   - Numeric scores are mapped into `Low Risk`, `Medium Risk`, and `High Risk`.
   - Logic appears in `src/risk_interpreter.py` and `api_server.py`.

3. Explainable AI with SHAP
   - `src/shap_explainer.py` loads a trained pipeline and computes SHAP values.
   - `api_server.py` generates summary, bar, and waterfall plots as base64-encoded PNGs for the frontend.
   - `app/page.tsx` renders these images in the browser.

4. Personalized health recommendations
   - The backend builds a recommendation list from risk factors such as smoking, BMI, blood pressure, cholesterol, physical activity, and SpO2.
   - Implemented in `api_server.py` and `app.py`.

5. Assessment history persistence
   - Each prediction saves a record to the SQLite `assessments` table.
   - The API exposes `/history` and `/history/{assessment_id}` for listing and deletion.

6. Model performance reporting
   - `src/model_evaluation.py` computes regression and classification metrics and saves them to `outputs/metrics.json`.
   - The frontend fetches `/model-metrics` and displays the metrics and confusion matrix.

7. Dual UI support
   - `app/page.tsx` is the primary browser interface.
   - `app.py` is a Streamlit interface that mirrors the patient form and prediction flow locally.

## 5. Technology Stack

### Programming languages

- Python: used for the ML pipeline, API backend, and data processing.
- TypeScript: used for the Next.js frontend.
- JavaScript/React: used by the Next.js app.

### Frontend

- Next.js 16 (from `package.json`): `"next": "^16.2.7"`
- React 18.3.1, React DOM 18.3.1
- TypeScript 5.5.4
- CSS custom styling in `app/globals.css` and inline component styling in `app/page.tsx`

### Backend

- FastAPI
- Uvicorn
- Pydantic models for request/response validation
- SQLite

### ML/data libraries

- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- shap
- streamlit

### Build tooling

- Next.js build pipeline (`npm run build`)
- Python environment with `requirements.txt`

### Versions

The repository shows explicit versions only for the frontend package set and a few Python dependency names without exact pins. `package.json` declares:

- Next.js: `^16.2.7`
- React: `18.3.1`
- React DOM: `18.3.1`
- TypeScript: `5.5.4`

`requirements.txt` contains dependency names but not pinned versions:

- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- shap
- streamlit
- fastapi
- uvicorn

This suggests a lightweight prototype environment rather than a fully locked production stack.

## 6. Repository Structure

The repository is laid out as a multi-layer prototype with both Python and Next.js components.

```text
AI-Health-Risk-Scoring-System/
├── app/                          # Next.js frontend app
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── database/                    # SQLite DB directory
│   └── healthrisk.db
├── dataset/                     # ML dataset
│   └── indian_health_risk_dataset.csv
├── docs/                        # Project documentation set
│   ├── Dataset and Feature Engineering.md
│   ├── Frontend and DemoFlow.md
│   ├── Functional Requirement and System Design Document,.md
│   ├── ML and SHAP Explainability.md
│   ├── Phasewise Execution.md
│   └── project_summary.md
├── models/                      # Serialized model artifact
│   └── random_forest_model.pkl
├── outputs/                     # Generated metrics/plots (not fully tracked in repo)
│   ├── confusion_matrix.png
│   └── metrics.json
├── src/                         # ML/data utilities and backend helper logic
│   ├── data_loader.py
│   ├── database.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── risk_interpreter.py
│   ├── shap_explainer.py
│   ├── train_model.py
│   ├── utils.py
│   └── visualization.py
├── .gitignore
├── README.md
├── api_server.py                # FastAPI app
├── app.py                       # Streamlit alternative frontend
├── main.py                      # CLI explainability pipeline
├── next.config.js
├── next-env.d.ts
├── package-lock.json
├── package.json
├── requirements.txt
├── tsconfig.json
├── PROJECT_DEEP_DIVE.md         # This document
└── venv/
```

### Major directories by role

- `src/`: ML pipeline, preprocessing, explainability, and utility modules.
- `app/`: Next.js dashboard.
- `database/`: SQLite DB persistence.
- `models/`: persisted trained ML model.
- `dataset/`: input dataset.
- `outputs/`: generated metrics and plots.
- `docs/`: project design and architecture notes.

### Excluded/generated directories

- `.next/`, `node_modules/`, `venv/`, `__pycache__/` are present and excluded from analysis as generated or environment-specific directories.
- `outputs/` was inspected because it is relevant to the model performance metrics and generated visualizations, but it is not treated as source code.
- The root `.gitignore` explicitly ignores `node_modules`, `.next`, `.env.local`, and `outputs/`, which indicates they are generated or environment-specific and not part of source control.

## 7. Architecture Overview

The architecture is a layered prototype:

1. User input layer
   - Next.js form in `app/page.tsx`
   - Streamlit form in `app.py`

2. API layer
   - FastAPI app in `api_server.py`

3. ML/inference layer
   - Data preprocessing in `src/preprocessing.py`
   - Feature selection in `src/feature_engineering.py`
   - Model loading / inference in `src/shap_explainer.py` and `api_server.py`
   - Risk interpretation in `src/risk_interpreter.py`

4. Data persistence layer
   - SQLite `assessments` store in `src/database.py`

5. Visualization layer
   - Matplotlib + SHAP plot generation in `src/visualization.py` and `api_server.py`

### High-level flow

```text
User
  ↓
Next.js dashboard (`app/page.tsx`) or Streamlit (`app.py`)
  ↓
POST /predict or direct local prediction logic
  ↓
FastAPI or local Python pipeline
  ↓
Preprocessor + RandomForestRegressor
  ↓
Risk score & risk level
  ↓
SHAP explainability + recommendations
  ↓
SQLite persistence + JSON response + charts
  ↓
Rendered dashboard
```

## 8. Architecture Diagram

```text
+----------------------------------------------------+
| User / Clinician                                   |
| Inputs patient health data                          |
+-------------------------------+--------------------+
                                |
                                v
+-------------------------------+--------------------+
| Frontend UI                                       |
| `app/page.tsx` (Next.js)                          |
| `app.py` (Streamlit alternative)                  |
+-------------------------------+--------------------+
                                |
                                v
+-------------------------------+--------------------+
| API Layer                                         |
| `api_server.py`                                   |
| Endpoints: /predict, /history, /model-metrics      |
+-------------------------------+--------------------+
                                |
                                v
+-------------------------------+--------------------+
| ML Pipeline                                       |
| `src/feature_engineering.py`                      |
| `src/preprocessing.py`                            |
| `src/train_model.py`                              |
| `src/shap_explainer.py`                           |
| `src/risk_interpreter.py`                          |
+-------------------------------+--------------------+
                                |
                                v
+-------------------------------+--------------------+
| Model Artifact                                     |
| `models/random_forest_model.pkl`                  |
+-------------------------------+--------------------+
                                |
                                v
+-------------------------------+--------------------+
| Data Storage                                      |
| SQLite `database/healthrisk.db`                   |
| `assessments` table                                |
+-------------------------------+--------------------+
                                |
                                v
+-------------------------------+--------------------+
| Output                                            |
| risk score, explanations, recommendations, plots  |
+----------------------------------------------------+
```

## 9. End-to-End Application Flow

### Normal prediction flow

The primary runtime path is the Next.js dashboard calling the FastAPI backend.

1. User opens the app in the browser.
2. `app/page.tsx` builds a JSON payload matching `PredictionInput` in `api_server.py`.
3. `fetch("http://127.0.0.1:8000/predict")` is sent with `Content-Type: application/json`.
4. `api_server.py` validates the payload against the Pydantic model.
5. The request enters `predict()`.
6. `_get_pipeline()` loads the model once and caches it.
7. `_prepare_input_df()` converts the payload into a pandas DataFrame with the required `FEATURE_COLUMNS`.
8. `pipeline.predict(input_df)` produces a scalar risk score.
9. `classify_risk()` maps the float to `Low Risk`, `Medium Risk`, or `High Risk`.
10. The code then calls `prepare_shap_inputs()`, `compute_shap_values()`, and `top_contributors()` to determine feature-level influences.
11. `generate_insights()` prepares textual explanations.
12. `_generate_recommendations()` assembles a list of action suggestions.
13. `save_assessment()` writes the prediction result to SQLite.
14. The backend generates summary, bar, and waterfall plots from Matplotlib/SHAP and converts them to base64.
15. `PredictionResponse` is returned as JSON.
16. The front-end renders the prediction, charts, recommendations, and history.

### History flow

- `GET /history` calls `get_assessment_history()` from `src/database.py`.
- Results are converted from SQLite rows into JSON with ISO timestamps in Asia/Kolkata timezone.
- `DELETE /history/{id}` removes a single assessment.
- `DELETE /history` clears all assessments.

### Metrics flow

- `GET /model-metrics` reads `outputs/metrics.json` and `outputs/confusion_matrix.png`.
- The data is base64-encoded and returned to the frontend.
- The frontend displays regression metrics and confusion matrix results.

## 10. Major Workflows

### Workflow A: Model training and evaluation

Source:
- `main.py`
- `src/train_model.py`
- `src/model_evaluation.py`
- `src/data_loader.py`
- `src/feature_engineering.py`
- `src/preprocessing.py`

Purpose:
- Train the risk-scoring model and measure it.

Flow:
- `load_dataset()` reads `dataset/indian_health_risk_dataset.csv`.
- `select_features()` extracts `X` and `y` using `FEATURE_COLUMNS` and `TARGET_COLUMN`.
- `split_data()` performs a train/test split.
- `build_preprocessor()` applies `OneHotEncoder` for categorical data and `StandardScaler` for numeric data.
- `train_model()` wraps the preprocessor and `RandomForestRegressor` in a scikit-learn `Pipeline`.
- `evaluate_model()` computes MAE, MSE, RMSE, R2, and classification metrics.
- The result is serialized by `src/model_evaluation.py` to `outputs/metrics.json`.

### Workflow B: CLI explainability pipeline

Source:
- `main.py`
- `src/shap_explainer.py`
- `src/visualization.py`
- `src/risk_interpreter.py`

Purpose:
- Generate SHAP outputs and explanation text samples for a chosen data point.

Flow:
- `run_pipeline()` loads the dataset and already-trained model.
- `generate_predictions()` creates score predictions for the test split.
- `prepare_shap_inputs()` gets the transformed feature matrix.
- `compute_shap_values()` calls SHAP `TreeExplainer`.
- `top_contributors()` finds the biggest positive and negative contributors.
- `build_report()` writes a text report and `save_*_plot()` saves visualizations to `outputs/`.

### Workflow C: API prediction and persistence

Source:
- `api_server.py`
- `src/database.py`
- `src/shap_explainer.py`

Purpose:
- Accept a patient profile, get a prediction, store it, and return explanations.

Flow:
- `PredictionInput` validates age, BMI, BP, cholesterol, etc.
- `predict()` receives the payload and runs the model.
- `save_assessment()` writes a record to SQLite.
- `top_contributors()` and `generate_insights()` produce explanations.
- `_render_summary_plot()`, `_render_bar_plot()`, and `_render_waterfall_plot()` generate charts.
- `PredictionResponse` returns JSON and embedded base64 plot strings.

### Workflow D: Browser UI interaction

Source:
- `app/page.tsx`

Purpose:
- Let the user test predictions and see history.

Flow:
- `handleSubmit()` sends the JSON payload to `/predict`.
- The response is saved in React state.
- `loadHistory()` populates the assessment table.
- `loadMetrics()` fetches `/model-metrics`.
- The page displays the risk score, clusters of feature contributors, SHAP plots, recommendations, and confusion matrix.

## 11. Frontend Architecture

The frontend is split between a production-style Next.js dashboard and a Streamlit alternative.

### Next.js app details

File: `app/page.tsx`

Highlights:
- Client component (`"use client"`)
- Defines the `PredictionInput` shape and the API payload contract
- Uses `useState` and `useEffect` for form state, history, and metrics
- Uses `fetch` to communicate with FastAPI
- Includes preset values for high-risk and low-risk profiles
- Renders result cards, recommendations, charts, and history table

The UI is styled with CSS classes in `app/page.tsx` and `app/globals.css` rather than a CSS-in-JS library. The design is glassmorphic and dark-mode themed.

### Streamlit app details

File: `app.py`

This file is a second interface that mirrors the same core workflow, but keeps execution in Python. It:

- sets a dark, dashboard-like theme,
- builds a sidebar input form,
- generates risk predictions from the model,
- shows SHAP plots with Matplotlib,
- renders recommendations and explanatory markdown.

This appears to be an alternate prototype frontend or historical/parallel UI implementation.

### Frontend/back-end boundary

- `app/page.tsx` calls the FastAPI backend at `http://127.0.0.1:8000` by default.
- The backend route is defined in `api_server.py`.
- The app uses `NEXT_PUBLIC_API_BASE_URL` if configured, otherwise defaults to localhost:8000.

## 12. Backend Architecture

File: `api_server.py`

This is the main API server.

### Important backend responsibilities

- Creates tables on import via `create_tables()`
- Defines all Pydantic models for validation and response serialization
- Configures CORS for browser access from `http://localhost:3000` and `http://127.0.0.1:3000`
- Loads the saved model from `models/random_forest_model.pkl`
- Builds SHAP-based explanations and plots
- Persists assessment history in SQLite
- Serves metrics and confusion matrix output for the UI

### Critical code points

- `_get_paths()` resolves the dataset/model paths
- `_get_pipeline()` caches model loading
- `_prepare_input_df()` turns the request payload into a DataFrame
- `_generate_recommendations()` creates actionable advice
- `_render_summary_plot()` and related helpers generate base64 images
- `root()` and `health_check()` provide minimal service checks
- `history()` returns assessment rows
- `predict()` is the main prediction endpoint

### Separation of concerns

The file mixes API routing, model logic, plotting, recommendation generation, and persistence in one module. This is pragmatic for a prototype, but it is not a strong separation-of-concerns architecture. The code is functionally cohesive but not cleanly distributed across multiple service layers.

## 13. API Documentation

### GET `/`

Purpose:
- Health check / root endpoint.

Implementation:
- `api_server.py` -> `root()`

Response:
```json
{"status": "ok"}
```

### GET `/health`

Purpose:
- Liveness check for the backend.

Response:
```json
{"status": "ok"}
```

### GET `/history`

Purpose:
- Return recent assessment history.

Behavior:
- Calls `get_assessment_history()` from `src/database.py`
- Returns rows with `id`, `age`, `risk_score`, `risk_level`, and ISO timestamp in `Asia/Kolkata`

### DELETE `/history/{assessment_id}`

Purpose:
- Delete a single saved assessment.

Behavior:
- Calls `delete_assessment()`.
- Returns `404` if the ID does not exist.

### DELETE `/history`

Purpose:
- Clear all assessment history.

Behavior:
- Calls `delete_all_assessments()`.

### GET `/model-metrics`

Purpose:
- Return model performance metrics and a confusion matrix image.

Behavior:
- Reads `outputs/metrics.json` and `outputs/confusion_matrix.png`.
- Returns JSON with `regression_metrics`, `classification_metrics`, and a base64-encoded confusion matrix.
- Raises `404` if the generated files are missing.

### POST `/predict`

Purpose:
- Predict risk from patient attributes.

Request body (validated by `PredictionInput`):
```json
{
  "age": 40,
  "gender": "Male",
  "bmi": 25,
  "systolic_bp": 125,
  "diastolic_bp": 80,
  "cholesterol_mg_dl": 190,
  "smoking": "No",
  "alcohol_consumption": "No",
  "physical_activity": "Medium",
  "family_history": "No",
  "heart_rate_bpm": 76,
  "sdnn_hrv": 55,
  "rmssd_hrv": 50,
  "spo2": 97
}
```

Validation rules:
- `age` between 18 and 80
- `bmi` between 15 and 45
- `systolic_bp` between 90 and 200
- `diastolic_bp` between 60 and 130
- `cholesterol_mg_dl` between 120 and 320
- `heart_rate_bpm` between 50 and 130
- `sdnn_hrv` and `rmssd_hrv` between 10 and 140
- `spo2` between 90 and 100

Processing flow:
- loads model,
- creates a DataFrame,
- computes prediction,
- classifies risk,
- saves to SQLite,
- computes SHAP values,
- computes top positive/negative contributors,
- builds insights and recommendations,
- generates visualizations,
- returns a `PredictionResponse`.

Response shape:
```json
{
  "risk_score": 72.1,
  "risk_level": "High Risk",
  "positive_contributors": [{"feature": "Smoking", "value": 8.2}],
  "negative_contributors": [{"feature": "Physical Activity", "value": -3.1}],
  "insights": ["..."],
  "recommendations": ["..."],
  "summary_plot": "base64...",
  "bar_plot": "base64...",
  "waterfall_plot": "base64..."
}
```

### Authentication requirements

No authentication or authorization is implemented anywhere in the API or frontend. The backend allows CROSS-origin requests from localhost:3000 and 127.0.0.1:3000 only, but there are no user identity or role checks.

## 14. Database Architecture

### Database technology

- SQLite
- File: `database/healthrisk.db`

### Schema

The `assessments` table is created in `src/database.py` with the following fields:

- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `age` INTEGER
- `gender` TEXT
- `bmi` REAL
- `systolic_bp` INTEGER
- `diastolic_bp` INTEGER
- `cholesterol_mg_dl` REAL
- `smoking` TEXT
- `alcohol_consumption` TEXT
- `physical_activity` TEXT
- `family_history` TEXT
- `heart_rate_bpm` REAL
- `sdnn_hrv` REAL
- `rmssd_hrv` REAL
- `spo2` REAL
- `risk_score` REAL
- `risk_level` TEXT
- `model_version` TEXT
- `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP

### CRUD operations

- Create table: `create_tables()`
- Insert: `save_assessment()`
- Read: `get_assessment_history()`
- Delete one: `delete_assessment()`
- Delete all: `delete_all_assessments()`

### Migrations

There are no migration files or migration tooling in the repository. SQLite tables are created on import by `create_tables()` in the FastAPI backend.

### Database usage in the application

The app stores completed risk assessment records so the dashboard can render a recent history list. There is no relational modeling beyond one table and no foreign keys or normalized schema.

## 15. Data Flow

### Source data

Dataset source:
- `dataset/indian_health_risk_dataset.csv`

The loader in `src/data_loader.py` prints dataset shape, columns, and summary statistics, and checks for missing values.

### Data lifecycle

```text
CSV dataset
  ↓
`src/data_loader.py` -> load_dataset()
  ↓
`src/feature_engineering.py` -> select_features()
  ↓
`src/preprocessing.py` -> ColumnTransformer (OneHotEncoder + StandardScaler)
  ↓
`src/train_model.py` -> RandomForestRegressor pipeline
  ↓
`api_server.py` / `app.py` -> prediction for new patient input
  ↓
`src/shap_explainer.py` -> SHAP values and feature affects
  ↓
`src/database.py` -> save_assessment()
  ↓
Frontend renders result + history + metrics
```

### Data structure

The primary input vector is defined in `src/feature_engineering.py` as:

- `age`
- `bmi`
- `systolic_bp`
- `diastolic_bp`
- `cholesterol_mg_dl`
- `smoking`
- `alcohol_consumption`
- `physical_activity`
- `family_history`
- `heart_rate_bpm`
- `sdnn_hrv`
- `rmssd_hrv`
- `spo2`

Target:
- `risk_score`

### Validation rules

The backend enforces numeric ranges via `pydantic.Field(...)` constraints in `PredictionInput`.

### Transformations

- categorical values normalized to `Yes` / `No` in `_normalize_yes_no()`
- numeric columns coerced via `pd.to_numeric(..., errors="coerce")`
- categorical features encoded via `OneHotEncoder(handle_unknown="ignore")`
- numeric features scaled via `StandardScaler()`

## 16. Machine Learning / Data Science Pipeline

### Model choice

This repository uses a `RandomForestRegressor` trained on a tabular health features dataset. The reason is documented in `docs/ML and SHAP Explainability.md` and is supported by the implementation: tree-based models work well with mixed numeric/categorical features and support SHAP explainability.

### Feature engineering

`src/feature_engineering.py` defines:

- `FEATURE_COLUMNS`
- `TARGET_COLUMN = "risk_score"`

`select_features(df)` returns `X` and `y`.

### Preprocessing

`src/preprocessing.py` implements:

- `build_preprocessor()`
- `get_feature_groups()`
- `ColumnTransformer` with separate pipelines for categorical and numeric features

Categorical pipeline:
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

Numeric pipeline:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`

### Training

`src/train_model.py`:

- creates a `RandomForestRegressor(random_state=random_state, n_estimators=100)`
- wraps it in a scikit-learn pipeline
- fits it to `X_train` and `y_train`

### Evaluation

`src/model_evaluation.py` calculates:

- MAE
- MSE
- RMSE
- R2
- accuracy
- precision
- recall
- f1
- confusion matrix

The repo also keeps generated metrics in `outputs/metrics.json`:

```json
{
  "regression_metrics": {
    "MAE": 11.255166666666675,
    "MSE": 172.04071076666682,
    "RMSE": 13.116429040202474,
    "R2": 0.7859468992600477
  },
  "classification_metrics": {
    "accuracy": 0.6666666666666666,
    "precision": 0.8333333333333334,
    "recall": 0.6666666666666666,
    "f1": 0.6656746031746031
  }
}
```

### Explainability

`src/shap_explainer.py`:

- loads the pipeline from disk,
- extracts the trained model,
- uses `shap.TreeExplainer(model)`,
- computes SHAP values for processed inputs,
- returns feature names and SHAP values.

`src/risk_interpreter.py`:

- classifies risk scores into three bands,
- extracts top contributors,
- formats feature names for readability,
- generates text insights based on patient values.

### Model artifact storage

- Saved path: `models/random_forest_model.pkl`
- Loaded by `api_server.py` and `main.py`

### Inference pipeline

A live prediction does the following:

`PredictionInput` -> DataFrame -> model.predict() -> risk score -> `classify_risk()` -> SHAP values -> contributor list -> recommendations -> response

## 17. Module-by-Module Breakdown

### `README.md`

Purpose:
- project overview, quick install instructions, and run commands.

### `api_server.py`

Purpose:
- primary backend service and API surface.

Key elements:
- Pydantic models for request/response validation
- CORS setup
- cached model loader
- database integration
- plotting logic
- endpoints

### `app.py`

Purpose:
- alternative Streamlit UI for local demo/testing.

Key elements:
- dashboard styling
- form builders
- prediction logic
- SHAP plot rendering
- recommendations output

### `main.py`

Purpose:
- CLI script to run the interpretability pipeline over a selected sample.

### `src/data_loader.py`

Purpose:
- load CSV data, print summary, warn on missing values.

### `src/feature_engineering.py`

Purpose:
- define feature columns and target.

### `src/preprocessing.py`

Purpose:
- prepare and normalize feature space for the model.

### `src/train_model.py`

Purpose:
- train a `RandomForestRegressor` pipeline.

### `src/utils.py`

Purpose:
- save data and directories for model output generation.

### `src/predict.py`

Purpose:
- quick function to generate raw predictions for test data.

### `src/shap_explainer.py`

Purpose:
- model loading and SHAP value computation.

### `src/risk_interpreter.py`

Purpose:
- risk band logic, contributor analysis, and summary text generation.

### `src/visualization.py`

Purpose:
- save summary, bar, and waterfall plots to file.

### `src/model_evaluation.py`

Purpose:
- evaluate classification and regression metrics, save confusion matrix and JSON metrics.

### `src/database.py`

Purpose:
- SQLite persistence layer for historical assessments.

### `app/page.tsx`

Purpose:
- main browser user interface for predictions and metrics.

### `app/layout.tsx`

Purpose:
- application shell and metadata declaration.

### `app/globals.css`

Purpose:
- UI styling for the Next.js app.

## 18. Important Functions, Classes, and Components

### `PredictionInput` in `api_server.py`

Purpose:
- validated request payload.

Fields:
- age, gender, bmi, blood pressure, cholesterol, smoking, alcohol consumption, activity level, family history, heart rate, HRV, SpO2.

### `predict()` in `api_server.py`

Purpose:
- main API prediction endpoint.

Inputs:
- `payload: PredictionInput`

Process:
- loads model,
- converts to DataFrame,
- gets raw risk score,
- saving to SQLite,
- computes SHAP values,
- identifies top contributors,
- generates text insights and recommendations,
- renders plots,
- returns JSON.

Output:
- `PredictionResponse`

### `save_assessment()` in `src/database.py`

Purpose:
- persist patient assessment rows.

Notes:
- stores a `model_version` of `"RF_v1"` for each row.
- does not currently store any user identity or audit trail beyond the input values.

### `classify_risk()` in `src/risk_interpreter.py`

Purpose:
- convert a numeric score to a label.

Logic:
- 0–34 -> Low Risk
- 35–69 -> Medium Risk
- 70–100 -> High Risk

### `top_contributors()` in `src/risk_interpreter.py`

Purpose:
- find the most important features from SHAP contributions.

Implementation detail:
- sorts by absolute SHAP magnitude,
- uses a deduping approach based on feature family name,
- splits into positive and negative contributors.

### `generate_insights()` in `src/risk_interpreter.py`

Purpose:
- produce human-readable descriptions of major contributors.

Notes:
- includes condition-based logic for age, BMI, cholesterol, BP, smoking, physical activity, SpO2, SDNN, RMSSD.
- relies on the input payload object.

### `load_model()` in `src/shap_explainer.py`

Purpose:
- load the persisted model from disk using `joblib`.

### `prepare_shap_inputs()` in `src/shap_explainer.py`

Purpose:
- transform the input into a feature matrix that matches the model pipeline.

### `compute_shap_values()` in `src/shap_explainer.py`

Purpose:
- build `shap.TreeExplainer` and compute SHAP values.

### `HealthRisk` dashboard form in `app/page.tsx`

Purpose:
- UI for collecting demographics, lifestyle, clinical, and physiological metrics.

Important behavior:
- range sliders and numeric inputs
- preset buttons for custom/high/low risk
- fetch prediction calls and state updates

## 19. Configuration and Environment

### Environment variables

The codebase does not include `.env`, `.env.example`, or any explicit environment configuration file. The frontend uses:

- `process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000"`

There are no other environment variables referenced in the repo.

### Runtime assumptions

- API backend expected at `http://127.0.0.1:8000` by default.
- Frontend expected at `http://localhost:3000` by the CORS configuration.
- Model file expected at `models/random_forest_model.pkl`.
- Dataset expected at `dataset/indian_health_risk_dataset.csv`.
- SQLite database expected at `database/healthrisk.db`.

### Secrets and credentials

No secrets, API keys, passwords, JWT secrets, or cloud credentials were found in the repository. The codebase does not appear to integrate with any external secret store or authentication service.

### Feature flags

No feature-flag mechanism is present.

## 20. Dependencies

| Technology | Version / evidence | Purpose | Where used |
|---|---|---|---|
| Python | Not pinned in requirements | runtime for ML + server | `api_server.py`, `src/*.py`, `app.py` |
| Next.js | `^16.2.7` | frontend framework | `package.json`, `app/` |
| React | `18.3.1` | component rendering | `app/page.tsx` |
| TypeScript | `5.5.4` | typed frontend | `tsconfig.json`, `app/page.tsx` |
| FastAPI | in `requirements.txt` | API layer | `api_server.py` |
| Uvicorn | in `requirements.txt` | ASGI server | recommended run command |
| pandas | in `requirements.txt` | tabular data processing | dataset load + DataFrame logic |
| numpy | in `requirements.txt` | numeric arrays and transformations | ML and SHAP |
| scikit-learn | in `requirements.txt` | preprocessing, model, metrics | `src/preprocessing.py`, `src/train_model.py`, `src/model_evaluation.py` |
| joblib | in `requirements.txt` | save/load model | `src/shap_explainer.py`, `src/utils.py` |
| matplotlib | in `requirements.txt` | plotting | `api_server.py`, `src/visualization.py`, `app.py` |
| SHAP | in `requirements.txt` | explainability | `src/shap_explainer.py` |
| Streamlit | in `requirements.txt` | alternate dashboard | `app.py` |
| SQLite3 | builtin Python module | persistence | `src/database.py` |

## 21. Setup and Installation

### Prerequisites

The repository expects:

- Python environment
- Node.js / npm for Next.js
- Access to the local dataset and model files

No explicit Docker or cloud prerequisites are present.

### Install Python dependencies

Command from `README.md` and `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Install frontend dependencies

Command from `README.md`:

```bash
npm install
```

### Configure environment

No `.env.example` or service config file exists. The default behavior is hardcoded in code:

- Frontend API base: `http://127.0.0.1:8000`
- FastAPI CORS allowed origins: `http://localhost:3000`, `http://127.0.0.1:3000`

The recommended configuration is therefore either:

- leave defaults as-is, or
- set `NEXT_PUBLIC_API_BASE_URL` before running the frontend.

### Initialize database

The table is created automatically on import by `create_tables()` in `src/database.py`.

### Run backend

From README and code:

```bash
python api_server.py
```

or

```bash
uvicorn api_server:app --reload
```

### Run frontend

```bash
npm run dev
```

Then open:

```text
http://localhost:3000
```

### Run alternative Streamlit frontend

```bash
streamlit run app.py
```

### Run training / evaluation pipeline

`main.py` gives a CLI entry point:

```bash
python main.py --test-size 0.2 --random-state 42 --sample-index 0
```

### Build the UI

```bash
npm run build
```

### Run the app in production mode

```bash
npm run start
```

## 22. Running the Application

### Typical developer flow

1. `pip install -r requirements.txt`
2. `npm install`
3. Start backend: `python api_server.py`
4. Start frontend: `npm run dev`
5. Open localhost:3000
6. Submit a form to hit `/predict`
7. View result, history, and model metrics

### Observed runtime behavior

The system is designed to work with a local dataset and model artifact. There are no deployed services or remote cloud hooks. The backend uses the local file system for model loading and metrics retrieval.

## 23. Testing

### Test infrastructure

No automated test files or test frameworks were found. Searches for `pytest`, `unittest`, `test_`, and `*.test.*` patterns returned no meaningful matches.

### What this means

- There are no unit tests.
- There are no integration tests.
- There are no end-to-end tests.
- There is no CI validation found in the repo.

### What is verified instead

The project has example usage and generated outputs:

- `outputs/metrics.json`
- `outputs/confusion_matrix.png`
- documentation files in `docs/`
- a demo-oriented UI in `app/page.tsx` and `app.py`

### Coverage gaps

This is a significant limitation:

- no test for API validation,
- no test for DB logic,
- no test for prediction edge cases,
- no regression check after model update,
- no security tests.

The repository appears to rely on manual validation and live demo runs rather than automated quality checks.

## 24. Security Review

### Authentication and authorization

Status: Not implemented.

Evidence:
- no user auth modules or route guards,
- no JWT, session, OAuth, or login flow,
- no access-control checks on `/predict` or `/history`.

Severity: HIGH

Reason:
- the backend exposes data collection and history endpoints without any access control. Because the repo is a local prototype, this is less critical than production deployment, but it is still a real gap.

### Input validation

Status: Partially implemented.

Evidence:
- `PredictionInput` in `api_server.py` validates numeric ranges and data types.
- the app also limits form values in the UI.

Strength:
- prevents obvious invalid ranges.

Weakness:
- validation is not deep medical validation and is not tied to stronger domain constraints.

Severity: MEDIUM

### Secret handling

Status: Clean, based on repository evidence.

Evidence:
- no `.env` files with credentials,
- no embedded API keys in code,
- no secret references in the repository.

Severity: INFORMATIONAL

### SQL injection / command injection

SQL injection risk appears low in the repository because the code uses parameterized SQLite queries in `src/database.py`.

Command injection risk is also low because no shell commands are dynamically constructed from user input in the backend path. However, the project does not implement a more robust security boundary for untrusted inputs.

Severity: LOW / INFORMATIONAL

### CORS

Status: configured.

Evidence:
```python
allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"]
```

This is a standard browser CORS restriction but not a full security model.

### Data exposure

Status: moderate.

The system stores patient data in a local SQLite file. The data includes age, gender, blood pressure, cholesterol, smoking status, and similar health indicators. This is not necessarily a vulnerability in a local prototype, but it is sensitive personal data being kept without a stronger privacy model.

Severity: MEDIUM

### Hardcoded values

The app has hardcoded default demo profiles and API URLs. This is appropriate for demo use but not robust for production deployment.

Severity: LOW

### Missing file validation

The backend reads model and metrics files from hardcoded local paths and raises `FileNotFoundError` or `HTTPException` when absent. This is straightforward but not hardened for deployment.

## 25. Code Quality Review

### Strengths

- Clear separation between dataset, preprocessing, feature engineering, model training, explainability, and UI.
- Real implementation rather than a mock-only prototype.
- Use of a proper `Pipeline` object for model training.
- Good use of SHAP to improve interpretability.
- SQLite persistence for assessment history.
- Next.js frontend with a clean interactive form and real API use.

### Weaknesses

- `api_server.py` mixes API routes, plotting, recommendations, persistence, and model logic in a single file.
- There is no service layer or separation into controllers, services, and repositories.
- `src/risk_interpreter.py` has redundant logic and some formatting inconsistencies.
- `generate_insights()` contains repetitive branching and a duplicated comment `# AGE` twice.
- No tests exist to protect API and model correctness.
- No environment management or dependency pinning is enforced.
- There is no authentication or authorization.
- Project is a demonstrator, not a production-ready system.

### Maintainability

The code is understandable and readable, but the architecture is more of a monolithic prototype than well-layered software architecture. It is maintainable for a school project or proof-of-concept but would need refactoring for larger user loads or team maintenance.

## 26. Implemented vs Planned

| Feature | Current Status | Evidence | Notes |
|---|---|---|---|
| Health risk prediction | Implemented | `api_server.py`, `src/train_model.py`, `app/page.tsx` | Real prediction flow exists |
| Risk classification | Implemented | `src/risk_interpreter.py` | Low/Medium/High logic present |
| SHAP explainability | Implemented | `src/shap_explainer.py` | TreeExplainer used |
| SHAP plot rendering | Implemented | `api_server.py` + `src/visualization.py` | summary, bar, waterfall |
| Recommendation generation | Implemented | `api_server.py`, `app.py` | lifestyle-based suggestions |
| Assessment history | Implemented | `src/database.py` | SQLite save/list/delete |
| Model metrics endpoint | Implemented | `api_server.py` | reads generated metrics files |
| Next.js frontend | Implemented | `app/page.tsx`, `app/layout.tsx` | active browser UI |
| Streamlit frontend | Implemented | `app.py` | alternative or parallel demo UI |
| Authentication | Planned / not implemented | no auth code or routes | absent from repo |
| Deployment pipeline | Planned / not implemented | no Dockerfiles, no cloud config, no CI | no deployment manifest |
| Production security hardening | Planned / not implemented | no auth, no secret mgmt, no validation beyond ranges | prototype level |
| Automated tests | Not implemented | no `/tests`, no pytest files | manual validation only |
| Real medical-grade model validation | Not implemented | no clinical dataset or validation workflow | synthetic dataset only |
| Cloud/production environment | Not implemented | no infra config files found | not determinable from codebase |

## 27. Technical Debt

The codebase contains several signs of technical debt typical of a prototype:

- A single-file API module handling too many responsibilities.
- Duplicate or repeated logic in risk insights generation.
- Hardcoded localhost endpoints and default values.
- SQLite schema created at runtime rather than using migration tooling.
- No formal test suite.
- No version pinning for Python dependencies.
- `app.py` and `api_server.py` both implement similar logic, which increases maintenance overhead.
- The repository includes documentation files that describe architecture but they are not active code enforcement.

## 28. Known Limitations

1. Synthetic dataset only
   - `dataset/indian_health_risk_dataset.csv` appears synthetic and not a clinical dataset from a real healthcare system.

2. No real auth or user identity model
   - not suitable for production or multi-user health record storage.

3. Local-only deployment assumptions
   - `localhost` URLs are hard-coded.

4. No persistence for model training metadata
   - no model registry or versioning beyond `model_version` in SQLite.

5. No CI/CD or automated deployment
   - absent from the repo.

6. No test coverage
   - manual validation only.

7. No compliance frameworks or medical safety controls
   - this is a prototype and should not be treated as a medical product.

## 29. Risks

### Architectural risks

- Prototype architecture may become brittle as more features are added.
- API and UI responsibilities are not cleanly separated.

### Data risks

- Data quality is only as good as the synthetic dataset and preprocessing rules.
- No validation of data provenance or input source trust.

### Security risks

- no auth, no rate limiting, and no secure secrets configuration.

### Operational risks

- no CI pipeline or automated test checks.
- no production deployment config.

### Model risks

- the model may be overfit to synthetic patterns and not generalize to real-world patient data.
- there is no calibration or external validation against real clinical outcomes.

## 30. Important Design Decisions

1. Use of Random Forest + SHAP
   - the model is tree-based to support SHAP explainability efficiently.

2. Use of SQLite for persistence
   - a simple and local persistence layer fits a demo or prototype.

3. CORS restricted to localhost
   - safe enough for local dev but not production-oriented.

4. Frontend and backend split by technology
   - Next.js handles browser-based interaction while FastAPI exposes the model service.

5. Dual UI strategy
   - Next.js for modern dashboard and Streamlit for prototyping or quick local experiments.

6. Synthetic health-score target
   - the score is framed as a risk score rather than a clinician-calibrated output.

## 31. Developer Guide

### Typical workflow for a developer working in this repo

1. Understand the model definition in `src/train_model.py` and `src/preprocessing.py`.
2. Confirm the feature schema in `src/feature_engineering.py`.
3. Run training / evaluation pipeline with `python main.py` or `src/model_evaluation.py`.
4. Start the backend with `python api_server.py`.
5. Run the frontend with `npm run dev`.
6. Use the form to generate predictions and verify history persistence.
7. Inspect `outputs/metrics.json` and `outputs/confusion_matrix.png` for evaluation results.
8. Use the docs in `docs/` for functional and architecture context.

### Typical debugging points

- Model missing: check `models/random_forest_model.pkl` exists.
- Data missing: check dataset path and column names.
- API not reachable: ensure backend is running on 8000.
- CORS errors: confirm frontend is using localhost:3000 or 127.0.0.1:3000 and backend config matches.
- History not showing: check SQLite DB file and `create_tables()`.

## 32. Troubleshooting

### Problem: `FileNotFoundError` for model or dataset

Likely cause:
- model or dataset not generated or placed in expected location.

Check:
- `models/random_forest_model.pkl`
- `dataset/indian_health_risk_dataset.csv`

### Problem: frontend cannot reach backend

Likely cause:
- backend not started or wrong URL.

Check:
- `app/page.tsx` `API_BASE` constant
- backend origin in `api_server.py`

### Problem: metrics endpoint returns 404

Likely cause:
- metrics not generated or `outputs/` missing.

Check:
- `outputs/metrics.json`
- `outputs/confusion_matrix.png`

### Problem: prediction errors

Likely cause:
- feature mismatch or invalid range values.

Check:
- `PredictionInput` field constraints
- `FEATURE_COLUMNS` in `src/feature_engineering.py`
- model training pipeline step naming (`preprocessor`, `model`)

### Problem: database errors

Likely cause:
- database directory or database file absent or lock issues.

Check:
- `src/database.py`
- `database/healthrisk.db`

## 33. Glossary

- SHAP: Shapley Additive Explanations, a model-interpretability method used to quantify feature contributions.
- Random Forest: ensemble of decision trees used for regression or classification.
- Feature engineering: selection and preparation of variables for training.
- OneHotEncoder: categorical encoding used to convert strings into binary indicator columns.
- StandardScaler: standardizes numeric features to a mean of zero and variance of one.
- Risk score: numeric output in the 0–100 scale.
- Risk level: Low / Medium / High output label derived from the risk score.
- Model artifact: saved machine-learning object (`.pkl`) loaded later for inference.
- Assessment history: persisted set of prior user/submitted risk assessments.

## 34. Complete System Summary

This repository is a prototype AI health risk scoring system built around a synthetic healthcare dataset and a trained random forest model. It includes the full flow from data ingestion to preprocessing, model training and evaluation, risk prediction, SHAP explainability, and persistence of assessment history in SQLite. The backend is a FastAPI service, while the user interface is implemented in both Next.js and Streamlit, enabling browser-based and local Python-based interactions.

The strongest implemented capabilities are:

- prediction from user health inputs,
- low/medium/high classification,
- SHAP-based explanation and chart generation,
- recommendation generation,
- SQLite history tracking,
- metrics output and confusion-matrix display.

The strongest limitations are:

- no production security model,
- no authentication or authorization,
- no automated testing,
- no deployment configuration,
- no real clinical data or formal validation pipeline,
- no environment file or dependency pinning.

The repository should be understood as a functional educational and demo prototype rather than a medically validated or production-ready healthcare system.

## Documentation Confidence

### Fully verified areas

- Repository structure and file inventory
- Dataset and model artifact locations
- FastAPI endpoints and request/response shapes
- Frontend form and prediction flow in `app/page.tsx`
- Training/evaluation pipeline in `src/*.py`
- SQLite schema and basic CRUD operations
- Risk classification and SHAP logic in the implementation
- Presence/absence of deployment, Docker, CI, and auth components

### Partially verified areas

- Actual model performance on unseen clinical data: the repo contains metrics but not a full benchmarking/reporting pipeline beyond the generated output.
- Deployment readiness: the repo clearly lacks config files, but exact production deployment steps are not specified.
- Streaming/live production behavior: the project is intended for local demo use, but no production runtime environment was found.

### Areas not determinable from the codebase

- Real-world clinical validity of the risk model
- Real patient data source or governance model
- Deployment cloud provider or environment
- User identity model and organization-level access policy
- Whether the project was meant to be a medical product versus an academic prototype
- Exact training procedure on the underlying synthetic dataset beyond what is present in the code and docs

### Necessary assumptions and inferences

- The system is a prototype/academic project rather than a production medical system, based on the included documentation and absence of deployment, auth, and testing infrastructure.
- The model is intended for local demonstration and prototyping, because all required assets are local files and no external services are configured.
- The use of a synthetic dataset and local evaluation output suggests the project is designed for demonstration and learning rather than clinical deployment.

🟢 Stage 1 — Dataset 2.0

1–2 weeks

identify credible datasets
ICMR-INDIAB
WHO STEPS / Indian NCD data
NHANES
potentially MIMIC
establish provenance
harmonize schemas
normalize units
remove identifiers
handle missing values
perform EDA
define target properly

The ICMR repository is especially worth investigating for your Indian-population positioning.

↓

🟣 Stage 2 — ML 2.0

1 week

Build:

Baseline
Linear
Random Forest
Extra Trees
XGBoost
HistGradientBoosting

Then:

5-fold CV
↓
Hyperparameter tuning
↓
Held-out test
↓
External validation
↓
Calibration
↓
SHAP

↓

🟠 Stage 3 — Blood Report Engine

1–2 weeks

Upload
 ↓
PDF/image extraction
 ↓
OCR fallback
 ↓
Biomarker parser
 ↓
Unit normalization
 ↓
Confidence
 ↓
Human confirmation
 ↓
Lab risk model

↓

🔴 Stage 4 — Backend architecture

3–5 days

Refactor:

api_server.py

into:

routers
services
ML
schemas
database
report processing

↓

🟡 Stage 5 — UI 2.0

1–2 weeks

Build:

Dashboard
Assessment
Blood Report
History
Insights
Model Performance

with:

Framer Motion / equivalent animation layer
proper design system
responsive layout
animated risk visualization
interactive charts
skeleton loading
transitions
report extraction animation
polished empty/error states

↓

⚫ Stage 6 — Integration + evaluation

1 week

Test the entire chain:

Manual input
       ↓
Prediction
       ↓
SHAP
       ↓
Recommendations
       ↓
History

and:

Blood PDF
       ↓
OCR
       ↓
Biomarkers
       ↓
Risk model
       ↓
Explanation
       ↓
Dashboard