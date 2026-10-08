# Repository Cleanup & Pre-Integration Source-of-Truth Audit (Pass 1: Analysis)

**Project**: AI Health Risk Scoring System  
**Audit Stage**: Pre-Integration Repository Cleanup & Source-of-Truth Audit (Pass 1)  
**Execution Timestamp**: October 2026  
**Auditor**: Senior ML Research Engineer & Systems Reviewer  

---

## 1. Repository Inventory

The repository contains **317 Git-tracked files** spanning four primary directory structures plus root assets:

| Top-Level Directory | Tracked Files | Role & Description |
|---|:---:|---|
| **`.` (Root)** | 12 | Research pipeline scripts (`shap_compute.py`, `shap_plot_figure3.py`), locked SHAP data/figures (`fig3_global_shap_feature_importance.*`, `shap_*.png`), canonical system description (`PROJECT_SYSTEM_DESCRIPTION.md`), `.gitignore`, and `README.md`. |
| **`backend/`** | 289 | Data pipelines, raw/processed NHANES datasets, Stage 1D/1E/1F/1G research models and evaluation reports, SQLite database, FastAPI server (`backend/api/`), Streamlit app (`backend/api/app.py`), and dependencies. |
| **`frontend/`** | 9 | Next.js 16 interactive dashboard (`frontend/app/page.tsx`, `layout.tsx`, `globals.css`, `package.json`, etc.). |
| **`docs/`** | 6 | Early Model V1 markdown documents describing the legacy synthetic Indian health dataset and wearable HRV features. |
| **`dataset/`** | 1 | Legacy Model V1 synthetic CSV (`dataset/indian_health_risk_dataset.csv`, 151 rows). |

In addition, untracked generated artifacts exist in the local workspace:
- Orphan build directories in root: `node_modules/` (331.6 MB) and `.next/` (87.4 MB) left behind from before the project migrated to `frontend/`.
- Active build directories in `frontend/`: `frontend/node_modules/` (331.6 MB) and `frontend/.next/` (63.8 MB).
- Python bytecode caches: Local untracked `__pycache__` directories in root, `backend/`, `backend/api/`, `backend/database/`, `backend/ml/`, and `backend/ml/scripts/`.

---

## 2. Current Authoritative Sources of Truth

The canonical, locked research pipeline and architectural blueprint for the project are defined strictly by:

1. **System & Architectural Blueprint**:
   - [`PROJECT_SYSTEM_DESCRIPTION.md`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/PROJECT_SYSTEM_DESCRIPTION.md): Authoritative definition of the V2 research system, CDC NHANES 2021–2023 cohort, two-tier Mode A/B feature architecture, calibrated classifiers, operating thresholds, and product vision.
2. **Dataset & Partitioning Specifications**:
   - Raw CDC NHANES XPT files: [`backend/ml/data/raw/nhanes_2021_2023/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/raw/nhanes_2021_2023/) (16 files).
   - Dataset Manifest: [`backend/ml/data/processed/nhanes_2021_2023/DATASET_MANIFEST.md`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/DATASET_MANIFEST.md).
   - Six Processed Parquet Datasets: `backend/ml/data/processed/nhanes_2021_2023/nhanes_*.parquet` ($N=7,809$ each).
   - Participant-Level Splits: `backend/ml/data/processed/nhanes_2021_2023/splits/*.csv` (70% Train, 15% Val, 15% Test with zero overlap).
3. **Locked Model Artifacts**:
   - Six Final Production-Candidate Pipelines in [`backend/ml/evaluation/stage_1f/models/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/models/):
     - CVD Mode A: `cvd_mode_a_calibrated.joblib` (Random Forest, 13 predictors, Sigmoid, threshold 0.13)
     - CVD Mode B: `cvd_mode_b_calibrated.joblib` (Random Forest, 29 predictors, Sigmoid, threshold 0.15)
     - Diabetes Mode A: `diabetes_mode_a_calibrated.joblib` (HistGradientBoosting, 13 predictors, Sigmoid, threshold 0.15)
     - Diabetes Mode B: `diabetes_mode_b_calibrated.joblib` (XGBoost, 27 predictors, Sigmoid, threshold 0.17)
     - Hypertension Mode A: `hypertension_mode_a_calibrated.joblib` (Logistic Regression, 11 predictors, Sigmoid, threshold 0.41)
     - Hypertension Mode B: `hypertension_mode_b_calibrated.joblib` (Random Forest, 27 predictors, Sigmoid, threshold 0.52)
4. **Authoritative Research Evaluation Reports**:
   - Stage 1F: `backend/ml/evaluation/stage_1f/STAGE_1F_CALIBRATION_THRESHOLD_REPORT.md`
   - Stage 1G-1: `backend/ml/evaluation/stage_1g/STAGE_1G_INTEGRITY_AUDIT.md`
   - Stage 1G-2: `backend/ml/evaluation/stage_1g/STAGE_1G_PERFORMANCE_ANALYSIS.md`
   - Stage 1G-3: `backend/ml/evaluation/stage_1g/STAGE_1G_SHAP_ANALYSIS.md`
   - Stage 1G-4: `backend/ml/evaluation/stage_1g/STAGE_1G-4_FINAL_RESEARCH_READINESS_AUDIT.md`
5. **Authoritative Explainability & Publication Figures**:
   - `shap_global_values.csv` & `shap_global_values.json` (60 locked records).
   - Figure 3: `fig3_global_shap_feature_importance.png`, `.pdf`, `.svg`.
   - Figure 4: `shap_waterfall_plot.png` (SEQN 130670).
   - Directional Summary Beeswarm: `shap_summary_plot.png`.

---

## 3. V1 Legacy Sources & Active Runtime Dependencies

A critical finding of this audit is that **the repository currently contains two distinct operational layers**:

1. **The Frozen V2 Research Pipeline**: Fully verified, audited, and locked in `backend/ml/data/`, `backend/ml/evaluation/`, and root SHAP scripts.
2. **The Active V1 Application Runtime**: The current FastAPI server (`backend/api/api_server.py`) and Next.js frontend (`frontend/app/page.tsx`) are currently wired to the legacy V1 single-model architecture. Specifically:
   - `api_server.py` line 87 loads `backend/models/random_forest_model.pkl`.
   - `api_server.py` line 86 resolves `dataset/indian_health_risk_dataset.csv`.
   - `api_server.py` line 237–238 loads `backend/outputs/metrics.json` and `backend/outputs/confusion_matrix.png` for `/model-metrics`.
   - `api_server.py` lines 27–34 imports `backend.ml.feature_engineering`, `backend.ml.risk_interpreter`, and `backend.ml.shap_explainer`.
   - `frontend/app/page.tsx` submits `PredictionInput` with V1 fields (`sdnn_hrv`, `rmssd_hrv`, etc.) to `/predict`.

**CRITICAL SAFETY CONCLUSION**:
Because the very next phase is **FastAPI V2 Integration** followed by **Next.js V2 Integration**, deleting `random_forest_model.pkl`, `metrics.json`, `confusion_matrix.png`, `indian_health_risk_dataset.csv`, or the root `backend/ml/*.py` modules right now would instantly break the functioning full-stack application and prevent local testing. These files must be categorized as **KEEP — SUPPORTING (ACTIVE V1 RUNTIME)** and retained until the FastAPI V2 refactoring cleanly replaces them.

In contrast, the markdown documentation files in `docs/` are **not** imported by any code; they are purely static text files describing the obsolete V1 prototype and presenting false sources of truth. They are prime candidates for archival into `docs/legacy_v1/`.

---

## 4. Duplicate Sources

1. **Root `node_modules/` vs. `frontend/node_modules/`**:
   - Root `node_modules/` (331.6 MB) has no `package.json` in root and is an orphaned duplicate from prior directory restructuring.
   - `frontend/node_modules/` is the active, tracked Next.js installation configured by `frontend/package.json`.
2. **Root `.next/` vs. `frontend/.next/`**:
   - Root `.next/` (87.4 MB) is an orphaned build folder.
   - `frontend/.next/` (63.8 MB) is the active Next.js build cache.
3. **Stage 1G Figures vs. Root Figures**:
   - `fig3_global_shap_feature_importance.png`, `shap_summary_plot.png`, and `shap_waterfall_plot.png` exist in both the root directory (for publication/paper access) and `backend/ml/evaluation/stage_1g/figures/` (for evaluation stage packaging). Both copies are intentional and should be retained.

---

## 5. Generated Files

1. **Untracked Orphan Node/Next Directories**: Root `node_modules/` and root `.next/`.
2. **Untracked Python Bytecode Caches**: Local `__pycache__` directories in root and backend packages.
3. **Git-Tracked Research Stage Outputs**: `backend/outputs/model_v2_baseline/` (Stage 1D benchmark outputs, 102 files) and `backend/outputs/model_v2_xgboost_baseline/` (Stage 1E-1 benchmark outputs, 23 files). These are tracked research provenance artifacts documenting baseline progression.

---

## 6. Potential Deletion Candidates

Applying the strict standard:
**SAFE TO DELETE = YES and CONFIDENCE = HIGH**:

### 1. Root Orphan `node_modules/` Directory
- **PATH**: `node_modules/` (workspace root)
- **CATEGORY**: Generated / Orphan Build Artifact (Untracked)
- **WHY OBSOLETE**: Leftover 331 MB build directory from before Next.js was moved into `frontend/`. Root has no `package.json`.
- **REFERENCED BY**: None.
- **REPLACED BY**: `frontend/node_modules/`.
- **SAFE TO DELETE**: YES.
- **CONFIDENCE**: **HIGH**.

### 2. Root Orphan `.next/` Directory
- **PATH**: `.next/` (workspace root)
- **CATEGORY**: Generated / Orphan Build Artifact (Untracked)
- **WHY OBSOLETE**: Leftover 87 MB build folder from prior root setup. Active Next.js app builds in `frontend/.next/`.
- **REFERENCED BY**: None.
- **REPLACED BY**: `frontend/.next/`.
- **SAFE TO DELETE**: YES.
- **CONFIDENCE**: **HIGH**.

### 3. Stale Local Python `__pycache__` Directories
- **PATH**: `__pycache__/`, `backend/__pycache__/`, `backend/api/__pycache__/`, `backend/database/__pycache__/`, `backend/ml/__pycache__/`, `backend/ml/scripts/__pycache__/`
- **CATEGORY**: Generated / Bytecode Cache (Untracked)
- **WHY OBSOLETE**: Contains compiled `.cpython-*.pyc` files from older runs. Automatically re-generated on demand.
- **REFERENCED BY**: None (runtime cache only).
- **REPLACED BY**: Fresh bytecode compilation by active Python interpreter.
- **SAFE TO DELETE**: YES.
- **CONFIDENCE**: **HIGH**.

---

## 7. Potential Archive Candidates

The six legacy V1 documentation files in `docs/` describe obsolete synthetic Indian datasets, universal 0–100 health scores, and simulated wearable HRV features. Archiving them into `docs/legacy_v1/` with a clear deprecation notice cleanses `docs/` of conflicting claims while preserving historical context:

### 1. `docs/Dataset and Feature Engineering.md`
- **PATH**: `docs/Dataset and Feature Engineering.md`
- **CATEGORY**: Obsolete V1 Documentation
- **WHY OBSOLETE**: Describes 151-row synthetic Indian dataset and simulated HRV features.
- **REFERENCED BY**: None in active code.
- **REPLACED BY**: `backend/ml/data/processed/nhanes_2021_2023/DATASET_MANIFEST.md` & `PROJECT_SYSTEM_DESCRIPTION.md`.
- **DESTINATION**: `docs/legacy_v1/Dataset and Feature Engineering.md`
- **SAFE TO ARCHIVE**: YES.
- **CONFIDENCE**: **HIGH**.

### 2. `docs/Frontend and DemoFlow.md`
- **PATH**: `docs/Frontend and DemoFlow.md`
- **CATEGORY**: Obsolete V1 Documentation
- **WHY OBSOLETE**: Describes single 0–100 risk score gauge demo flow.
- **REFERENCED BY**: None in active code.
- **REPLACED BY**: `PROJECT_SYSTEM_DESCRIPTION.md` Section 18.
- **DESTINATION**: `docs/legacy_v1/Frontend and DemoFlow.md`
- **SAFE TO ARCHIVE**: YES.
- **CONFIDENCE**: **HIGH**.

### 3. `docs/Functional Requirement and System Design Document,.md`
- **PATH**: `docs/Functional Requirement and System Design Document,.md`
- **CATEGORY**: Obsolete V1 Documentation (Note typo comma in filename)
- **WHY OBSOLETE**: Outdated V1 design spec.
- **REFERENCED BY**: None in active code.
- **REPLACED BY**: `PROJECT_SYSTEM_DESCRIPTION.md`.
- **DESTINATION**: `docs/legacy_v1/Functional Requirement and System Design Document,.md`
- **SAFE TO ARCHIVE**: YES.
- **CONFIDENCE**: **HIGH**.

### 4. `docs/ML and SHAP Explainability.md`
- **PATH**: `docs/ML and SHAP Explainability.md`
- **CATEGORY**: Obsolete V1 Documentation
- **WHY OBSOLETE**: Describes V1 regressor SHAP explainer on synthetic data.
- **REFERENCED BY**: None in active code.
- **REPLACED BY**: `PROJECT_SYSTEM_DESCRIPTION.md` Section 16 & `STAGE_1G_SHAP_ANALYSIS.md`.
- **DESTINATION**: `docs/legacy_v1/ML and SHAP Explainability.md`
- **SAFE TO ARCHIVE**: YES.
- **CONFIDENCE**: **HIGH**.

### 5. `docs/Phasewise Execution.md`
- **PATH**: `docs/Phasewise Execution.md`
- **CATEGORY**: Obsolete V1 Documentation
- **WHY OBSOLETE**: Early prototype development phases.
- **REFERENCED BY**: None in active code.
- **REPLACED BY**: `PROJECT_SYSTEM_DESCRIPTION.md` Section 23.
- **DESTINATION**: `docs/legacy_v1/Phasewise Execution.md`
- **SAFE TO ARCHIVE**: YES.
- **CONFIDENCE**: **HIGH**.

### 6. `docs/project_summary.md`
- **PATH**: `docs/project_summary.md`
- **CATEGORY**: Obsolete V1 Documentation
- **WHY OBSOLETE**: Early summary with obsolete directory structure (`src/`, `models/`).
- **REFERENCED BY**: None in active code.
- **REPLACED BY**: `PROJECT_SYSTEM_DESCRIPTION.md`.
- **DESTINATION**: `docs/legacy_v1/project_summary.md`
- **SAFE TO ARCHIVE**: YES.
- **CONFIDENCE**: **HIGH**.

---

## 8. Files That Must Definitely Be Preserved

### Group 1: Core V2 Machine Learning Pipeline (PROTECTED)
- All 16 CDC NHANES raw files in `backend/ml/data/raw/nhanes_2021_2023/`.
- All 6 processed Parquet datasets and split CSVs in `backend/ml/data/processed/nhanes_2021_2023/`.
- All Stage 1C data scripts: `build_nhanes_dataset.py`, `nhanes_harmonizer.py`, `nhanes_loader.py`, `nhanes_targets.py`, `nhanes_quality_checks.py`.
- All Stage 1D/1E/1F model scripts: `benchmark_models.py`, `benchmark_xgboost.py`, `optimize_models.py`, `optimize_random_forest.py`, `calibrate_and_optimize_thresholds.py`.
- All 12 calibrated model binaries (`.joblib` and `.pkl`) in `backend/ml/evaluation/stage_1f/models/`.
- All Stage 1E, 1F, and 1G reports, JSONs, CSVs, and figures in `backend/ml/evaluation/`.
- Root SHAP computation & plotting scripts: `shap_compute.py`, `shap_plot_figure3.py`.
- Locked SHAP outputs & figures: `shap_global_values.csv`, `shap_global_values.json`, `fig3_global_shap_feature_importance.*`, `shap_summary_plot.png`, `shap_waterfall_plot.png`.
- Authoritative documentation: `PROJECT_SYSTEM_DESCRIPTION.md`, `README.md`.

### Group 2: Frontend & Full-Stack Application (PROTECTED)
- Entire `frontend/` directory (`app/`, `package.json`, `tsconfig.json`, `next.config.js`, etc.).
- Active dependencies: `backend/requirements.txt` (including newly verified `xgboost` and `pyarrow`).
- Database module: `backend/database/database.py`, `backend/database/healthrisk.db`.
- API entry points: `backend/api/api_server.py`, `backend/api/app.py`.

### Group 3: Active V1 Runtime Dependencies (PROTECTED Until Phase 2)
- `backend/models/random_forest_model.pkl` (loaded by `api_server.py`).
- `backend/outputs/metrics.json` & `backend/outputs/confusion_matrix.png` (loaded by `api_server.py` `/model-metrics`).
- `dataset/indian_health_risk_dataset.csv` (referenced by `api_server.py`).
- `backend/ml/*.py` (`feature_engineering.py`, `risk_interpreter.py`, `shap_explainer.py`, `data_loader.py`, `predict.py`, `preprocessing.py`, `train_model.py`, `utils.py`, `visualization.py`, `model_evaluation.py`).

### Group 4: Historical Stage Benchmarks (KEEP — SUPPORTING)
- `backend/outputs/model_v2_baseline/` (Stage 1D benchmark evidence).
- `backend/outputs/model_v2_xgboost_baseline/` (Stage 1E-1 XGBoost benchmark evidence).
- `backend/ml/scripts/main.py` (V1 CLI test harness).
- `backend/ml/scripts/_inspect_data.py` (Stage 1E-2 data exploration script).
- `backend/ml/data/raw/icmr_indiab/` (`sample.dta`, `meta.pdf` reference cohort).

---

## 9. Dependency and Reference Analysis

```
[ Next.js Client (frontend/app/page.tsx) ]
           │
           │ HTTP POST /predict & GET /model-metrics
           ▼
[ FastAPI Server (backend/api/api_server.py) ]
     ├── imports: backend.database.database
     ├── imports: backend.ml.feature_engineering
     ├── imports: backend.ml.risk_interpreter
     ├── imports: backend.ml.shap_explainer
     ├── loads:   backend/models/random_forest_model.pkl   <-- REQUIRED UNTIL PHASE 2
     ├── loads:   backend/outputs/metrics.json             <-- REQUIRED UNTIL PHASE 2
     ├── loads:   backend/outputs/confusion_matrix.png     <-- REQUIRED UNTIL PHASE 2
     └── path to: dataset/indian_health_risk_dataset.csv   <-- REQUIRED UNTIL PHASE 2

[ Frozen V2 Research Pipeline (Independent & Locked) ]
     ├── Data:    backend/ml/data/processed/nhanes_2021_2023/
     ├── Models:  backend/ml/evaluation/stage_1f/models/
     ├── Reports: backend/ml/evaluation/stage_1g/
     ├── SHAP:    shap_compute.py, shap_global_values.csv, fig3_global_shap_feature_importance.*
     └── Specs:   PROJECT_SYSTEM_DESCRIPTION.md
```

This structural analysis proves that deleting the V1 runtime assets prior to Phase 2 would sever the active application link, whereas archiving `docs/` and deleting orphan root build caches has zero impact on either subsystem.

---

## 10. Risk Assessment

| Action | Candidates | Risk Level | Mitigation / Guardrail |
|---|---|:---:|---|
| **Delete Untracked Orphan Caches** | Root `node_modules/`, Root `.next/`, `__pycache__/` | **VERY LOW** | These are untracked build caches. `frontend/node_modules/` and `frontend/.next/` remain intact. Python regenerates `.pyc` automatically. |
| **Archive Legacy Documentation** | 6 files in `docs/` $\rightarrow$ `docs/legacy_v1/` | **VERY LOW** | Zero code files import markdown documentation. Historical information is 100% preserved. Git tracks file moves smoothly. |
| **Retain Active V1 Runtime Assets** | `random_forest_model.pkl`, `metrics.json`, `confusion_matrix.png`, `dataset/` | **ZERO RISK** | Retaining these files guarantees that `api_server.py` and `frontend/app/page.tsx` remain fully functional until Phase 2 rewires them to V2. |

---

## Summary Lists

### DELETE LIST (Pass 2 Action)
1. `node_modules/` (Root untracked orphan directory, ~331.6 MB)
2. `.next/` (Root untracked orphan directory, ~87.4 MB)
3. Stale local Python `__pycache__` directories in repository (`__pycache__/`, `backend/__pycache__/`, `backend/api/__pycache__/`, `backend/database/__pycache__/`, `backend/ml/__pycache__/`, `backend/ml/scripts/__pycache__/`)

### ARCHIVE LIST (Pass 2 Action)
Move from `docs/` to `docs/legacy_v1/`:
1. `docs/Dataset and Feature Engineering.md`
2. `docs/Frontend and DemoFlow.md`
3. `docs/Functional Requirement and System Design Document,.md`
4. `docs/ML and SHAP Explainability.md`
5. `docs/Phasewise Execution.md`
6. `docs/project_summary.md`  
*(Plus create `docs/legacy_v1/README.md` explaining the deprecation)*

### KEEP LIST (Preserved Without Alteration)
- All 12 files in root (`PROJECT_SYSTEM_DESCRIPTION.md`, `README.md`, `shap_compute.py`, `shap_plot_figure3.py`, `fig3_global_shap_feature_importance.*`, `shap_*.png`, `shap_global_values.*`, `.gitignore`).
- All raw and processed NHANES data in `backend/ml/data/` (52 files).
- All research evaluation reports, models, and figures in `backend/ml/evaluation/` (75 files).
- All data and training scripts in `backend/ml/scripts/` (15 files).
- Entire Next.js application in `frontend/` (9 tracked files + `frontend/node_modules/` + `frontend/.next/`).
- FastAPI server and database in `backend/api/` and `backend/database/` (6 files).
- Active V1 runtime dependencies (`backend/models/random_forest_model.pkl`, `backend/outputs/metrics.json`, `backend/outputs/confusion_matrix.png`, `dataset/indian_health_risk_dataset.csv`, `backend/ml/*.py`).
- Baseline benchmark evidence in `backend/outputs/model_v2_baseline/` and `model_v2_xgboost_baseline/`.
