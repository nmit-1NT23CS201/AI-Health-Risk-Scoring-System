# Repository Cleanup & Pre-Integration Source-of-Truth Audit (Final Report)

**Project**: AI Health Risk Scoring System  
**Audit Stage**: Pre-Integration Repository Cleanup & Source-of-Truth Audit (Final Report)  
**Execution Timestamp**: October 2026  
**Auditor**: Senior ML Research Engineer & Systems Reviewer  

---

## 1. Files Deleted

The following generated, unreferenced, and orphan cache directories were safely deleted from the workspace:

| Path | Type | Disk Space Reclaimed | Reason for Deletion |
|---|---|:---:|---|
| `node_modules/` (Root) | Directory (Untracked) | ~331.6 MB | Redundant orphan build directory left over from prior directory restructuring. Root has no `package.json`; active frontend node modules reside in `frontend/node_modules/`. |
| `.next/` (Root) | Directory (Untracked) | ~87.4 MB | Redundant orphan build directory. Active Next.js build cache resides in `frontend/.next/`. |
| Local `__pycache__/` | Directories (Untracked) | ~5.2 MB | Stale Python bytecode caches across root, `backend/`, `backend/api/`, `backend/database/`, `backend/ml/`, and `backend/ml/scripts/`. |
| **Total Reclaimed** | | **~424.2 MB** | |

*Note*: Zero Git-tracked code, data, model, or research files were deleted.

---

## 2. Files Archived

Six legacy Model V1 markdown documents were moved from `docs/` into `docs/legacy_v1/` to eliminate conflicting sources of truth while preserving project evolution history:

| Original Path | Archived Destination | Description & Archival Rationale |
|---|---|---|
| `docs/Dataset and Feature Engineering.md` | `docs/legacy_v1/Dataset and Feature Engineering.md` | Outdated V1 doc describing 151-row synthetic Indian dataset and simulated HRV. Replaced by `DATASET_MANIFEST.md` and `PROJECT_SYSTEM_DESCRIPTION.md`. |
| `docs/Frontend and DemoFlow.md` | `docs/legacy_v1/Frontend and DemoFlow.md` | Outdated V1 demo flow for single 0–100 risk score gauge. Replaced by `PROJECT_SYSTEM_DESCRIPTION.md` Section 18. |
| `docs/Functional Requirement and System Design Document,.md` | `docs/legacy_v1/Functional Requirement and System Design Document,.md` | Outdated V1 requirements and system design specification (with legacy filename comma). |
| `docs/ML and SHAP Explainability.md` | `docs/legacy_v1/ML and SHAP Explainability.md` | Outdated V1 SHAP description for continuous regressor. Replaced by `STAGE_1G_SHAP_ANALYSIS.md` and `PROJECT_SYSTEM_DESCRIPTION.md` Section 16. |
| `docs/Phasewise Execution.md` | `docs/legacy_v1/Phasewise Execution.md` | Outdated V1 prototype roadmap. Replaced by `PROJECT_SYSTEM_DESCRIPTION.md` Section 23. |
| `docs/project_summary.md` | `docs/legacy_v1/project_summary.md` | Outdated V1 project summary with obsolete folder hierarchy (`src/`, `models/`). |
| *(New Asset)* | `docs/legacy_v1/README.md` | Clear deprecation notice explicitly stating that `docs/legacy_v1/` contains historical V1 assets superseded by V2. |

---

## 3. Files Retained (Authoritative & Supporting)

### 3.1 Locked V2 Research Pipeline (PROTECTED — 100% Retained)
- **Raw CDC NHANES Data**: All 16 `.xpt` files in [`backend/ml/data/raw/nhanes_2021_2023/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/raw/nhanes_2021_2023/).
- **Processed NHANES Datasets**: All 6 Parquet files in [`backend/ml/data/processed/nhanes_2021_2023/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/) ($N=7,809$ each).
- **Participant Split Files**: All 3 split CSVs in `backend/ml/data/processed/nhanes_2021_2023/splits/` (70% Train, 15% Val, 15% Test with zero overlap).
- **Locked V2 Model Binaries**: All 12 serialized model artifacts (`.joblib` and `.pkl`) in [`backend/ml/evaluation/stage_1f/models/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/models/).
- **Authoritative Research Reports**: Stage 1E, Stage 1F, and Stage 1G (1G-1, 1G-2, 1G-3, 1G-4) evaluation reports and JSON files in `backend/ml/evaluation/`.
- **SHAP Engine & Publication Figures**: `shap_compute.py`, `shap_plot_figure3.py`, `shap_global_values.csv`, `shap_global_values.json`, `fig3_global_shap_feature_importance.*` (PNG, PDF, SVG), `shap_summary_plot.png`, `shap_waterfall_plot.png`.
- **Canonical Blueprint**: [`PROJECT_SYSTEM_DESCRIPTION.md`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/PROJECT_SYSTEM_DESCRIPTION.md) and `README.md`.

### 3.2 Full-Stack Application & Active V1 Runtime (PROTECTED — Retained for Phase 2 Integration)
- **Frontend**: Entire [`frontend/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/frontend/) application (`app/page.tsx`, `layout.tsx`, `globals.css`, `package.json`, `frontend/node_modules/`, `frontend/.next/`).
- **Backend API & Database**: `backend/api/api_server.py`, `backend/api/app.py`, `backend/database/database.py`, `backend/database/healthrisk.db`, `backend/requirements.txt`.
- **Active V1 Runtime Assets**: `backend/models/random_forest_model.pkl`, `backend/outputs/metrics.json`, `backend/outputs/confusion_matrix.png`, `dataset/indian_health_risk_dataset.csv`, and root `backend/ml/*.py` modules (`feature_engineering.py`, `risk_interpreter.py`, `shap_explainer.py`, etc.).

### 3.3 Historical Research Provenance (KEEP — SUPPORTING)
- `backend/outputs/model_v2_baseline/` (Stage 1D benchmark outputs, 102 files).
- `backend/outputs/model_v2_xgboost_baseline/` (Stage 1E-1 XGBoost benchmark outputs, 23 files).
- `backend/ml/scripts/` (Stage 1C data preparation, Stage 1D/1E benchmarking, Stage 1F calibration scripts).
- `backend/ml/data/raw/icmr_indiab/` (`sample.dta`, `meta.pdf` reference cohort).

---

## 4. Why Each Deletion Was Safe

1. **Root `node_modules/`**: The root directory contains no `package.json` or `package-lock.json`. The active Node/Next.js environment is self-contained inside `frontend/` (`frontend/package.json` and `frontend/node_modules/`). Deleting the root duplicate had zero effect on the frontend, which builds and runs independently.
2. **Root `.next/`**: An orphaned build artifact from when Next.js was run at the repository root prior to project reorganization. Active builds occur inside `frontend/.next/`.
3. **Local `__pycache__/`**: Python bytecode caches are temporary compilation artifacts. Python automatically recompiles `.py` files into fresh `.pyc` files upon execution, ensuring zero stale bytecode from older Python installations.

---

## 5. Ambiguous Files Intentionally Retained

1. **`backend/models/random_forest_model.pkl`**: While an obsolete V1 model, `backend/api/api_server.py` line 87 currently loads this exact binary at server startup. Deleting it would break the live FastAPI server. It is retained until Phase 2 rewires `api_server.py` to the six V2 models.
2. **`backend/outputs/metrics.json` and `confusion_matrix.png`**: Loaded by `api_server.py`'s `/model-metrics` endpoint for `frontend/app/page.tsx`. Retained to maintain full-stack functionality until Phase 2 updates the metrics route to deliver V2 multi-disease performance.
3. **`dataset/indian_health_risk_dataset.csv`**: Referenced by `api_server.py`'s path resolver. Retained until Phase 2 removes this dependency.
4. **`backend/ml/scripts/main.py` & `backend/ml/scripts/_inspect_data.py`**: Standalone scripts under `backend/ml/scripts/` that do not harm the V2 pipeline and serve as historical testing/exploration harnesses.

---

## 6. Remaining V1 References & Transition Strategy

The remaining V1 references exist exclusively in the active application runtime layer:
- `backend/api/api_server.py` (`PredictionInput`, model loading, metric endpoints).
- `backend/api/app.py` (Streamlit dashboard interface).
- `frontend/app/page.tsx` (Single cardiovascular risk form input view).

**Transition Strategy for Next Phase (Phase 2 & 3)**:
1. **Phase 2 (FastAPI V2 Integration)**: Update `api_server.py` to import the six calibrated Stage 1F models from `backend/ml/evaluation/stage_1f/models/` and serve Mode A and Mode B predictions for CVD, Diabetes, and Hypertension. Once this is completed, `random_forest_model.pkl`, `metrics.json`, `confusion_matrix.png`, and `indian_health_risk_dataset.csv` can be permanently retired.
2. **Phase 3 (Next.js Multi-Disease Dashboard)**: Update `frontend/app/page.tsx` to provide Mode A / Mode B toggle controls and display multi-disease risk cards with V2 SHAP waterfall visualizations.

---

## 7. Remaining Duplicate Sources of Truth

- **None in Documentation**: Archiving `docs/*.md` into `docs/legacy_v1/` ensures that [`PROJECT_SYSTEM_DESCRIPTION.md`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/PROJECT_SYSTEM_DESCRIPTION.md) is the single authoritative architectural blueprint.
- **None in ML Evaluation**: All official V2 model performance metrics and SHAP values stem strictly from `backend/ml/evaluation/stage_1f/` and `stage_1g/`.
- **Publication Figures**: The dual presence of Figure 3 (`fig3_global_shap_feature_importance.*`), Figure 4 (`shap_waterfall_plot.png`), and the summary plot (`shap_summary_plot.png`) in both the root directory and `backend/ml/evaluation/stage_1g/figures/` is intentional to support IEEE paper manuscript compilation without traversing deep subdirectories.

---

## 8. Cleanup Scheduled for Subsequent Phases

The following cleanup actions are scheduled for execution during Phase 2 (FastAPI V2 Migration):
1. **Decommission V1 Model File**: Remove `backend/models/random_forest_model.pkl` once `api_server.py` loads the six V2 `.joblib` pipelines.
2. **Decommission V1 Output Metrics**: Remove `backend/outputs/metrics.json` and `backend/outputs/confusion_matrix.png` once `/model-metrics` serves V2 multi-disease performance from `stage_1g_performance_analysis.json`.
3. **Decommission V1 Dataset**: Remove `dataset/indian_health_risk_dataset.csv` once `api_server.py` no longer resolves its path.
4. **Decommission V1 ML Scripts**: Retire legacy modules `backend/ml/risk_interpreter.py`, `backend/ml/predict.py`, etc., after replacing them with V2 multi-disease inference utilities.

---

## 9. Validation Results

A 13-point automated validation suite was executed against the cleaned repository:

| Check # | Verification Item | Result | Details |
|:---:|---|:---:|---|
| **1** | Git Status | **PASS** | 6 files cleanly moved to `docs/legacy_v1/`, orphan directories eliminated. |
| **2** | Python Core Imports | **PASS** | `scikit-learn==1.9.0`, `xgboost==3.4.1`, `pyarrow==25.0.1`, `shap==0.52.0`, `fastapi`, `uvicorn` all operational. |
| **3** | FastAPI App Initialization | **PASS** | `backend.api.api_server.app` loads cleanly; SQLite database initializes; `/model-metrics` executes successfully. |
| **4** | Backend Structure | **PASS** | All required directories (`api/`, `database/`, `ml/`, `models/`, `outputs/`) intact. |
| **5** | Frontend Structure | **PASS** | `frontend/package.json`, `tsconfig.json`, `app/page.tsx`, `layout.tsx`, `globals.css` intact. |
| **6** | V2 Model Artifacts | **PASS** | All 6 `.joblib` pipelines load cleanly with exact feature counts (13, 29, 13, 27, 11, 27). |
| **7** | V2 Processed Datasets | **PASS** | All 6 Parquet files present with exact row counts ($N = 7,809$ each). |
| **8** | V2 Split Files | **PASS** | All 3 split CSVs present ($70/15/15$ split ratios, zero SEQN overlap). |
| **9** | Stage 1F Reports | **PASS** | Calibration report, results JSON, and comparison CSV intact. |
| **10** | Stage 1G Reports | **PASS** | All four Stage 1G reports (1G-1, 1G-2, 1G-3, 1G-4) and JSON manifests intact. |
| **11** | SHAP Outputs & Figures | **PASS** | `shap_global_values.csv`, `.json`, Fig. 3 (PNG, PDF, SVG), Fig. 4, and summary beeswarm intact. |
| **12** | Frontend Node Modules | **PASS** | `frontend/node_modules/` and `frontend/.next/` intact and unaffected. |
| **13** | Dependencies Specification | **PASS** | `backend/requirements.txt` contains complete dependencies including `xgboost` and `pyarrow`. |

---

## 10. Git Status Summary

```text
Changes to be committed:
  renamed: docs/Dataset and Feature Engineering.md -> docs/legacy_v1/Dataset and Feature Engineering.md
  renamed: docs/Frontend and DemoFlow.md -> docs/legacy_v1/Frontend and DemoFlow.md
  renamed: docs/Functional Requirement and System Design Document,.md -> docs/legacy_v1/Functional Requirement and System Design Document,.md
  renamed: docs/ML and SHAP Explainability.md -> docs/legacy_v1/ML and SHAP Explainability.md
  renamed: docs/Phasewise Execution.md -> docs/legacy_v1/Phasewise Execution.md
  renamed: docs/project_summary.md -> docs/legacy_v1/project_summary.md

Untracked files:
  docs/legacy_v1/README.md
  repository_cleanup_audit.md
  repository_cleanup_final.md
```

---

## Authoritative System Audit Verdicts

```
ML PIPELINE INTEGRITY:       PASS
APPLICATION INTEGRITY:       PASS
RESEARCH REPRODUCIBILITY:    PASS
FINAL CLEANUP STATUS:        CLEAN (READY FOR APPLICATION INTEGRATION)
```
