# Stage 1G-4: Final Research-Readiness Audit Report

**Project**: AI Health Risk Scoring System  
**Audit Stage**: Stage 1G-4 (Final Research-Readiness Audit of the Frozen V2 ML Pipeline)  
**Evaluator**: Senior ML Research Engineer & Research-Methodology Reviewer  
**Audit Scope**: Entire Frozen V2 ML Pipeline (Stages 1A through 1G-3)  
**Execution Timestamp**: October 2026  
**Final Decision**: **CONDITIONAL PASS** (Zero Methodological Blockers; Pipeline Ready for Freezing)  

---

## 1. Executive Summary

This audit constitutes the exhaustive, independent research-readiness verification of the frozen Model V2 machine learning pipeline within the **AI Health Risk Scoring System**. Spanning all six disease-mode configurations—Cardiovascular Disease (CVD Modes A & B), Diabetes Mellitus (Modes A & B), and Hypertension (Modes A & B)—this evaluation assesses dataset provenance, target construction, leakage quarantines, participant split independence, serialized model artifacts, calibration rigor, threshold selections, test metric reproducibility, Mode A vs. Mode B trade-offs, SHAP explainability fidelity, IEEE paper claims, artifact consistency, and computational reproducibility.

### Key Audit Determinations:
1. **Mathematical & Metric Reproducibility (PASS)**: Re-executing single-pass evaluation across all six locked models on the untouched 15% participant-level held-out test partitions reproduced all reported test metrics (ROC-AUC, PR-AUC, Brier score, ECE, sensitivity, specificity, precision, NPV, F1, balanced accuracy) with maximum absolute error $< 0.00005$, comfortably satisfying the stringent $\le 0.0001$ scientific tolerance.
2. **Methodological & Leakage Integrity (PASS)**: Zero target leakage or survey-design leakage exists. Target-defining diagnostic variables, treatment proxies, and sampling weights are strictly excluded from all predictor matrices. Feature counts match locked specifications exactly: CVD Mode A (13), CVD Mode B (29), Diabetes Mode A (13), Diabetes Mode B (27), Hypertension Mode A (11), and Hypertension Mode B (27).
3. **Partition Independence (PASS)**: Participant-level 70% Train, 15% Validation, and 15% Test splits exhibit strictly zero SEQN overlap. Validation-only out-of-fold calibration and threshold sweeping were maintained with zero test contamination.
4. **Explainability Layer (PASS)**: SHAP Tree and Linear explainers evaluate the underlying base estimators within their mathematical decision spaces prior to monotonic Platt sigmoid scaling. Zero V1 prototype features contaminate the calculations.
5. **Non-Blocking Presentation & Environmental Findings (WARNING)**: 
   - `backend/requirements.txt` omits `xgboost` and `pyarrow`, which are necessary for clean-environment dependency resolution.
   - The `docs/` directory contains legacy V1 documentation referring to a synthetic Indian dataset and wearable HRV features that could confuse outside readers if not clearly labeled as deprecated.
   - In `PROJECT_SYSTEM_DESCRIPTION.md`, Section 18's UI text mockup suggests additive percentage contributions for SHAP attributions, which should be aligned with the margin-space explanation methodology.
6. **Overall Readiness**: The machine learning research pipeline is technically sound, methodologically rigorous, and scientifically locked. Stage 1G-4 is assigned **CONDITIONAL PASS**.

---

## 2. Audit Scope

This audit is strictly evaluative and analytical:
- **No Retraining or Refitting**: All six final model artifacts remained locked and unaltered.
- **No Data or Split Adjustments**: Datasets, survey weights, and participant partitions remained frozen.
- **No Threshold or Calibration Re-tuning**: Locked operating thresholds ($0.13, 0.15, 0.15, 0.17, 0.41, 0.52$) and Platt calibrators were tested as-is.
- **No Automated Fixes**: Discrepancies or stale artifacts are reported transparently for subsequent integration phases.

---

## 3. Dataset Provenance

**Status**: **PASS**

### Verification of CDC NHANES 2021–2023 Cohort
- **Raw SAS Files**: The raw source directory [`backend/ml/data/raw/nhanes_2021_2023/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/raw/nhanes_2021_2023/) contains all 16 official CDC SAS transport (`.xpt`) files (`DEMO_L.xpt`, `BMX_L.xpt`, `BPXO_L.xpt`, `BPQ_L.xpt`, `SMQ_L.xpt`, `ALQ_L.xpt`, `PAQ_L.xpt`, `DIQ_L.xpt`, `MCQ_L.xpt`, `TCHOL_L.xpt`, `HDL_L.xpt`, `TRIGLY_L.xpt`, `GLU_L.xpt`, `GHB_L.xpt`, `BIOPRO_L.xpt`, `CBC_L.xpt`). File hashes and byte counts confirm they remain unaltered.
- **Adult Cohort Filter**: An explicit filter (`RIDAGEYR >= 20`) was applied to the initial 11,933 survey participants, yielding exactly **$N = 7,809$ adult participants** with age ranging from $20.0$ to $80.0$ years.
- **Processed Parquet Files**: All six disease-mode datasets in [`backend/ml/data/processed/nhanes_2021_2023/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/) (`nhanes_cvd_mode_a.parquet`, `nhanes_cvd_mode_b.parquet`, `nhanes_diabetes_mode_a.parquet`, `nhanes_diabetes_mode_b.parquet`, `nhanes_hypertension_mode_a.parquet`, `nhanes_hypertension_mode_b.parquet`) contain exactly 7,809 rows.
- **Survey Metadata Isolation**: Survey design variables (`SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU`) are preserved for statistical accountability but are quarantined from predictive feature vectors.
- **Isolation of Legacy Indian Dataset**: The legacy V1 file [`dataset/indian_health_risk_dataset.csv`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/dataset/indian_health_risk_dataset.csv) (151 rows) exists in the repository root but is completely quarantined from the V2 pipeline.

---

## 4. Target Integrity

**Status**: **PASS**

Target definitions and diagnostic proxy exclusions were inspected against CDC NHANES clinical guidelines:

### 4.1 Cardiovascular Disease (`target_cvd`)
- **Clinical Formulation**: Physician-diagnosed hard atherosclerotic/ischemic endpoints: Congestive Heart Failure (`MCQ160B`), Coronary Heart Disease (`MCQ160C`), Angina Pectoris (`MCQ160D`), Myocardial Infarction (`MCQ160E`), or Stroke (`MCQ160F`).
- **Epidemiological Breakdown**: Positive = $982$ ($12.58\%$), Negative = $6,825$ ($87.42\%$), Missing = $2$ ($N_{\text{valid}} = 7,807$).
- **Leakage Prevention**: All `MCQ160*` diagnostic questions and cholesterol treatment proxy `BPQ101D` are quarantined from both Mode A and Mode B predictors.

### 4.2 Type 2 Diabetes Mellitus (`target_diabetes`)
- **Clinical Formulation**: American Diabetes Association (ADA) multi-criteria standard: $\text{HbA1c} \ge 6.5\%$ (`LBXGH`) OR Fasting Plasma Glucose $\ge 126\text{ mg/dL}$ (`LBXGLU`) OR Doctor Diagnosed (`DIQ010`) OR Insulin Use (`DIQ050`) OR Oral Hypoglycemic Agents (`DIQ070`).
- **Epidemiological Breakdown**: Positive = $1,385$ ($17.74\%$), Negative = $6,421$ ($82.26\%$), Missing = $3$ ($N_{\text{valid}} = 7,806$).
- **Leakage Prevention**: Both diagnostic laboratory biomarkers (`hba1c`, `fasting_glucose`) and questionnaire history (`DIQ010`, `DIQ050`, `DIQ070`) are strictly quarantined from predictors.

### 4.3 Hypertension (`target_hypertension`)
- **Clinical Formulation**: Joint National Committee (JNC7) clinical threshold: $\text{Mean SBP} \ge 140\text{ mmHg}$ OR $\text{Mean DBP} \ge 90\text{ mmHg}$ OR Doctor Diagnosed (`BPQ020`).
- **Epidemiological Breakdown**: Positive = $3,331$ ($42.71\%$), Negative = $4,469$ ($57.29\%$), Missing = $9$ ($N_{\text{valid}} = 7,800$).
- **Leakage Prevention**: Oscillometric blood pressure means (`mean_sbp`, `mean_dbp`) and diagnosis history (`BPQ020`) are strictly quarantined from both Mode A and Mode B predictors.

---

## 5. Feature & Leakage Audit

**Status**: **PASS**

### Predictor Dimensions & Manifest Compliance
All six pipelines strictly adhere to the locked feature counts:

| Pipeline | Target Disease | Mode | Authoritative Predictor Count | Actual Parquet Predictors | Model Artifact `n_features_in_` | SHAP Matrix Dimension | Leakage Audit Result |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **CVD Mode A** | Cardiovascular Disease | Mode A | **13** | 13 | 13 | $(1172, 13)$ | **PASS** |
| **CVD Mode B** | Cardiovascular Disease | Mode B | **29** | 29 | 29 | $(1172, 29)$ | **PASS** |
| **Diabetes Mode A** | Diabetes Mellitus | Mode A | **13** | 13 | 13 | $(1171, 13)$ | **PASS** |
| **Diabetes Mode B** | Diabetes Mellitus | Mode B | **27** | 27 | 27 | $(1171, 27)$ | **PASS** |
| **Hypertension Mode A**| Hypertension | Mode A | **11** | 11 | 11 | $(1170, 11)$ | **PASS** |
| **Hypertension Mode B**| Hypertension | Mode B | **27** | 27 | 27 | $(1170, 27)$ | **PASS** |

### Quarantine Verification:
- **Zero V1 Contamination**: Regex searches across all active matrices, scripts, models, and JSON outputs confirmed 0 occurrences of V1 features (`SDNN`, `RMSSD`, `HRV`, `resting_heart_rate`, `peak_heart_rate`, V1 `family_history`).
- **Zero Survey Design Leakage**: Non-predictor survey metadata (`SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU`) is quarantined from model matrices across all six pipelines.

---

## 6. Split Integrity

**Status**: **PASS**

### Participant-Level Independence & Zero Overlap
Splits stored in [`backend/ml/data/processed/nhanes_2021_2023/splits/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/splits/) were audited:

| Disease Cohort | Split Index File | Train Partition (70%) | Validation Partition (15%) | Test Partition (15%) | Total Valid Participants | Train-Val Overlap | Train-Test Overlap | Val-Test Overlap |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cardiovascular Disease** | [`cvd_splits.csv`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/splits/cvd_splits.csv) | 5,464 | 1,171 | 1,172 | 7,807 | **0** | **0** | **0** |
| **Diabetes Mellitus** | [`diabetes_splits.csv`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/splits/diabetes_splits.csv) | 5,464 | 1,171 | 1,171 | 7,806 | **0** | **0** | **0** |
| **Hypertension** | [`hypertension_splits.csv`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/splits/hypertension_splits.csv) | 5,460 | 1,170 | 1,170 | 7,800 | **0** | **0** | **0** |

- **Stratification**: Target distributions across splits mirror population prevalences within $\pm 0.2\%$.
- **Test Integrity**: The 15% test partition remained untouched until final evaluation. All calibration parameters and threshold sweeps were computed strictly on validation partitions.

---

## 7. Model Artifact Integrity

**Status**: **PASS**

All six production-candidate artifacts in [`backend/ml/evaluation/stage_1f/models/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/models/) were loaded and audited:

| Configuration | Serialized Artifact File | Calibration Wrapper Class | Underlying Estimator Class | Preprocessing Pipeline Steps | Model Input Count | Status |
|---|---|---|---|---|:---:|:---:|
| **CVD Mode A** | `cvd_mode_a_calibrated.joblib` | `CalibratedClassifierCV` (method='sigmoid') | `RandomForestClassifier` | `Pipeline([('imputer', SimpleImputer(strategy='median'))])` | 13 | **PASS** |
| **CVD Mode B** | `cvd_mode_b_calibrated.joblib` | `CalibratedClassifierCV` (method='sigmoid') | `RandomForestClassifier` | `Pipeline([('imputer', SimpleImputer(strategy='median'))])` | 29 | **PASS** |
| **Diabetes Mode A** | `diabetes_mode_a_calibrated.joblib` | `CalibratedClassifierCV` (method='sigmoid') | `HistGradientBoostingClassifier` | Native tree split missing-value routing | 13 | **PASS** |
| **Diabetes Mode B** | `diabetes_mode_b_calibrated.joblib` | `CalibratedClassifierCV` (method='sigmoid') | `XGBClassifier` | Native tree split missing-value routing | 27 | **PASS** |
| **Hypertension Mode A** | `hypertension_mode_a_calibrated.joblib` | `CalibratedClassifierCV` (method='sigmoid') | `LogisticRegression` | `Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])` | 11 | **PASS** |
| **Hypertension Mode B** | `hypertension_mode_b_calibrated.joblib` | `CalibratedClassifierCV` (method='sigmoid') | `RandomForestClassifier` | `Pipeline([('imputer', SimpleImputer(strategy='median'))])` | 27 | **PASS** |

---

## 8. Calibration Integrity

**Status**: **PASS**

- **Validation OOF Protocol**: In Stage 1F, base models were fitted on the 70% training split. 5-fold out-of-fold cross-validation on the 15% validation partition compared Raw, Sigmoid (Platt), and Isotonic methods.
- **Empirical Selection**: Sigmoid calibration was selected across all six pipelines because isotonic regression exhibited step-function instability and higher Brier scores on sparse validation regions.
- **Refitting & Locking**: Calibrators were refit on the complete validation partition and frozen.
- **Zero Test Calibration**: The test set was evaluated once post-hoc; zero test data was used for calibration parameter estimation.

---

## 9. Threshold Integrity

**Status**: **PASS**

- **Locked Operating Points**:
  - CVD Mode A: **$0.13$**
  - CVD Mode B: **$0.15$**
  - Diabetes Mode A: **$0.15$**
  - Diabetes Mode B: **$0.17$**
  - Hypertension Mode A: **$0.41$**
  - Hypertension Mode B: **$0.52$**
- **Methodological Soundness**: Swept across 91 candidate thresholds ($0.05$ to $0.95$) strictly on validation out-of-fold predictions. Optimized Youden's $J$ Index subject to clinical screening recall constraints.
- **Test Set Adherence**: All reported test binary metrics (Sensitivity, Specificity, Precision, NPV, F1) use these exact locked values without post-hoc test optimization.

---

## 10. Final Test Metric Reproducibility

**Status**: **PASS**

Independent re-execution on the held-out test partitions verified complete metric reproducibility against authoritative Stage 1F and Stage 1G-2 records. All absolute differences are $\le 0.00005$ (well below the $0.0001$ tolerance limit):

```
+----------------------------------------------------------------------------------------------------+
|                         REPRODUCIBILITY BENCHMARK (INDEPENDENT VS REPORTED)                         |
+----------------------------------------------------------------------------------------------------+
  Metric            CVD Mode A        CVD Mode B      Diabetes Mode A   Diabetes Mode B    HTN Mode A        HTN Mode B
  --------------------------------------------------------------------------------------------------
  ROC-AUC           0.8137 vs 0.8137  0.8218 vs 0.8218  0.7957 vs 0.7957  0.8162 vs 0.8162  0.7662 vs 0.7662  0.7731 vs 0.7731
  PR-AUC            0.4159 vs 0.4159  0.4340 vs 0.4340  0.4348 vs 0.4348  0.4793 vs 0.4793  0.6541 vs 0.6541  0.6898 vs 0.6898
  Brier Score       0.0912 vs 0.0912  0.0906 vs 0.0906  0.1222 vs 0.1222  0.1203 vs 0.1203  0.1946 vs 0.1946  0.1916 vs 0.1916
  ECE               0.0125 vs 0.0125  0.0368 vs 0.0368  0.0250 vs 0.0250  0.0470 vs 0.0470  0.0331 vs 0.0331  0.0465 vs 0.0465
  Sensitivity       0.6824 vs 0.6824  0.6081 vs 0.6081  0.8173 vs 0.8173  0.6202 vs 0.6202  0.7615 vs 0.7615  0.6253 vs 0.6253
  Specificity       0.7803 vs 0.7803  0.8438 vs 0.8438  0.6397 vs 0.6397  0.8100 vs 0.8100  0.6647 vs 0.6647  0.7511 vs 0.7511
  Precision (PPV)   0.3098 vs 0.3098  0.3600 vs 0.3600  0.3288 vs 0.3288  0.4135 vs 0.4135  0.6281 vs 0.6281  0.6514 vs 0.6514
  NPV               0.9444 vs 0.9444  0.9371 vs 0.9371  0.9419 vs 0.9419  0.9080 vs 0.9080  0.7894 vs 0.7894  0.7294 vs 0.7294
  F1 Score          0.4262 vs 0.4262  0.4523 vs 0.4523  0.4690 vs 0.4690  0.4962 vs 0.4962  0.6884 vs 0.6884  0.6380 vs 0.6380
  Balanced Accuracy 0.7314 vs 0.7314  0.7259 vs 0.7259  0.7285 vs 0.7285  0.7151 vs 0.7151  0.7131 vs 0.7131  0.6882 vs 0.6882
  --------------------------------------------------------------------------------------------------
  Max Difference    0.000047          0.000050          0.000034          0.000049          0.000036          0.000050
  Audit Verdict     PASS              PASS              PASS              PASS              PASS              PASS
```

---

## 11. Mode A vs Mode B Analysis

**Status**: **PASS**

The empirical comparison between Non-Invasive (Mode A) and Biomarker-Enhanced (Mode B) reflects genuine clinical trade-offs:

1. **Discrimination Gains**: Adding circulating blood chemistry yields consistent, modest improvements in discrimination:
   - CVD: ROC-AUC $+0.0081$ ($0.8137 \rightarrow 0.8218$), PR-AUC $+0.0181$ ($0.4159 \rightarrow 0.4340$).
   - Diabetes: ROC-AUC $+0.0205$ ($0.7957 \rightarrow 0.8162$), PR-AUC $+0.0445$ ($0.4348 \rightarrow 0.4793$).
   - Hypertension: ROC-AUC $+0.0069$ ($0.7662 \rightarrow 0.7731$), PR-AUC $+0.0357$ ($0.6541 \rightarrow 0.6898$).
2. **Operating Point Trade-offs**: Mode B does **not** universally outperform Mode A on every metric:
   - At locked screening thresholds, Mode A provides superior sensitivity for initial population screening: Diabetes Mode A achieves **$81.7\%$ sensitivity** vs. **$62.0\%$** in Mode B; CVD Mode A achieves **$68.2\%$ sensitivity** vs. **$60.8\%$** in Mode B; Hypertension Mode A achieves **$76.2\%$ sensitivity** vs. **$62.5\%$** in Mode B.
   - Mode B tightens specificity (reducing false positives by $6.4\%$ in CVD and $17.0\%$ in Diabetes) and improves precision.
   - In Hypertension, Mode A achieves a higher F1 score ($0.6884$ vs. $0.6380$).
3. **Research Framing**: Mode A represents an effective, low-burden screening configuration for settings without lab testing, while Mode B refines risk stratification when laboratory results are available.

---

## 12. SHAP / Explainability Integrity

**Status**: **PASS**

Audited against Stage 1G-3 deliverables:
1. **Model & Data Integrity**: Explanations were computed on the locked test sets without retraining or refitting.
2. **Explanation Space**: Confirmed that SHAP Tree and Linear explainers evaluate the underlying base estimators in uncalibrated margin/log-odds space ($f(x)$ or vote proportions). Monotonicity with respect to calibrated probability is preserved without making invalid claims of direct probability additivity.
3. **Non-Causal Language**: Local and global attributions are framed as statistical predictive influence within the fitted model rather than biological etiologies.

---

## 13. Reproducibility Audit

**Status**: **WARNING (Non-Blocking)**

### Strengths:
- Fully deterministic random seeds (`random_state=42`).
- Fixed, persistent CSV split files with zero cross-set leakage.
- Self-contained Parquet datasets and serialized models.

### Identified Discrepancy:
- **`backend/requirements.txt` Dependency Gap**: The file lists `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `shap`, `streamlit`, `fastapi`, and `uvicorn`, but omits:
  - `xgboost` (required to load and evaluate Diabetes Mode B).
  - `pyarrow` or `fastparquet` (required to read Parquet datasets).
- *Recommendation*: Update `backend/requirements.txt` prior to final application deployment.

---

## 14. Artifact Consistency

**Status**: **WARNING (Non-Blocking)**

### Strengths:
- `backend/ml/evaluation/stage_1f/` and `stage_1g/` contain authoritative, internally consistent results, JSON matrices, and figures.

### Identified Inconsistencies:
1. **Stale Documentation in `docs/`**: The files in [`docs/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/docs/) (`Dataset and Feature Engineering.md`, `project_summary.md`, `ML and SHAP Explainability.md`, etc.) describe the obsolete V1 prototype (151-row synthetic Indian dataset, single 0–100 risk score, simulated HRV). They do not reflect the locked V2 pipeline.
2. **Legacy Model Files in Root**: `backend/models/random_forest_model.pkl` and `backend/outputs/metrics.json` are legacy V1 outputs.
3. *Recommendation*: Add a clear deprecation banner to `docs/` or relocate them to a `docs/legacy_v1/` archive so readers do not confuse them with the canonical V2 architecture described in `PROJECT_SYSTEM_DESCRIPTION.md`.

---

## 15. IEEE Paper Claim Audit

**Status**: **PASS WITH ADVISORIES**

Substantive claims in [`PROJECT_SYSTEM_DESCRIPTION.md`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/PROJECT_SYSTEM_DESCRIPTION.md) were audited against the locked V2 pipeline:

| Claim Category | Text / Statement in Document | Audit Finding | Classification |
|---|---|---|:---:|
| **Dataset Description** | CDC NHANES 2021–2023, adult cohort $N = 7,809$, age 20–80. | Verified exactly against Parquet schemas and raw files. | **CORRECT (E)** |
| **Disease Definitions** | Hard CVD (MCQ160B-F), Diabetes (ADA multi-criteria), Hypertension (JNC7). | Verified against target construction code and manifest. | **CORRECT (E)** |
| **Feature Spaces** | Mode A: 11–13 predictors; Mode B: 27–29 predictors. | Verified against Parquet schemas and model inputs. | **CORRECT (E)** |
| **Data Partitioning** | 70% Train, 15% Validation, 15% Test participant-level split. | Verified in split CSVs with zero SEQN overlap. | **CORRECT (E)** |
| **Model Selection** | CVD A/B (RF), Diabetes A (HGB), Diabetes B (XGB), HTN A (LR), HTN B (RF). | Verified against saved `.joblib` estimators. | **CORRECT (E)** |
| **Calibration Method** | Sigmoid (Platt) scaling across all six pipelines. | Verified against `model.method == 'sigmoid'`. | **CORRECT (E)** |
| **Locked Thresholds** | 0.13, 0.15, 0.15, 0.17, 0.41, 0.52. | Verified against evaluation records. | **CORRECT (E)** |
| **Final Test Metrics** | Table 15.1 numbers match reproduced values ($< 0.00005$ diff). | Verified programmatically on test set. | **CORRECT (E)** |
| **0.50 Threshold Impact** | Section 2 Line 73 states 0.50 cutoff produces "catastrophic under-diagnosis". | Overly dramatic phrasing for an IEEE paper. | **WORDING TO SOFTEN (D)** |
| **SHAP Impact Presentation** | Section 18 UI mockup shows `[+] Systolic BP (+14.2% risk contribution)`. | Implies direct additive percentage to probability. | **WORDING TO SOFTEN (D)** |
| **Planned Features** | Blood-test report OCR, multi-disease UI, FHIR EHR integration. | Transparently categorized as "Proposed/Vision". | **CORRECT (E)** |

---

## 16. Figure Audit

**Status**: **PASS WITH ADVISORIES**

### Figure Inventory:
1. **Figure 1 (NHANES Research Workflow)**: Currently exists as an ASCII workflow diagram in `PROJECT_SYSTEM_DESCRIPTION.md`. Needs high-resolution vector/raster graphic rendering for final IEEE layout.
2. **Figure 2 (System Architecture)**: Exists as a detailed ASCII architectural diagram in documentation. Vector rendering required for paper layout.
3. **Figure 3 (Global Six-Panel SHAP Feature Importance)**: Verified and locked in [`fig3_global_shap_feature_importance.png`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/fig3_global_shap_feature_importance.png), [`.pdf`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/fig3_global_shap_feature_importance.pdf), [`.svg`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/fig3_global_shap_feature_importance.svg). Contains verified V2 data only.
4. **Figure 4 (Individual Waterfall Explanation)**: Verified in [`shap_waterfall_plot.png`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/shap_waterfall_plot.png) for test participant SEQN 130670.
5. **Figure 5 (Risk Dashboard)**: ASCII mockup in documentation; frontend UI screenshot or vector rendering required for paper layout.
6. **Obsolete Figures**: `backend/outputs/confusion_matrix.png` is an obsolete V1 figure and must not be used in the IEEE paper.

---

## 17. Scientific Limitations

**Status**: **PASS**

The scientific positioning in `PROJECT_SYSTEM_DESCRIPTION.md` appropriately bounds the research scope:
- **Research Prototype Only**: Positioned as an investigational screening decision-support tool, not as a diagnostic medical device.
- **Cross-Sectional Boundary**: Predicts prevalent, undiagnosed condition risk rather than long-term 10-year incident events.
- **Demographic Boundary**: Derived from U.S. civilian NHANES data; external validation is required prior to deployment in international populations.
- **Imbalance & Thresholds**: Transparently discusses sensitivity-specificity trade-offs across non-default operating points.

---

## 18. Issues / Findings Summary

| Identifier | Finding Description | Severity | Impact | Resolution Recommendation |
|---|---|:---:|:---:|---|
| **ISSUE-1** | `backend/requirements.txt` lacks `xgboost` and `pyarrow`. | Low | Environment Setup | Add `xgboost>=2.0.0` and `pyarrow>=14.0.0` to `requirements.txt`. |
| **ISSUE-2** | `docs/` contains stale V1 documentation describing synthetic Indian dataset. | Low | Documentation Clarity | Add deprecation notice to `docs/` or move to `docs/legacy_v1/`. |
| **ISSUE-3** | UI mockup in `PROJECT_SYSTEM_DESCRIPTION.md` uses percentage signs for SHAP (`+14.2%`). | Low | Scientific Presentation | Revise mockup text to show relative weight or score contribution rather than absolute probability percentage. |
| **ISSUE-4** | Figures 1, 2, and 5 exist only as ASCII diagrams in documentation. | Low | Publication Preparation | Render vector/PNG graphics for Figures 1, 2, and 5 during paper layout. |

---

## 19. Final Readiness Decision

```
================================================================================
FINAL AUDIT DECISION: CONDITIONAL PASS (PIPELINE FROZEN & VERIFIED)
================================================================================
```

### Justification:
- **Zero Methodological Failures**: Zero data leakage, zero model refitting, 100% split independence, verified calibration, exact threshold adherence, and 100% test metric reproducibility ($< 0.00005$ difference).
- **All Findings Are Non-Blocking**: The four identified issues are non-blocking documentation, presentation, and packaging items that do not compromise model weights, data matrices, or experimental metrics.

---

## 20. Recommended Next Phase

> **"Freeze ML research pipeline and proceed to final paper integration + FastAPI/Next.js application integration."**

### Immediate Action Plan:
1. Lock all files under `backend/ml/evaluation/stage_1f/` and `backend/ml/evaluation/stage_1g/`.
2. Update `backend/requirements.txt` with `xgboost` and `pyarrow`.
3. Proceed to Phase 2: Wire the 6 calibrated Stage 1F model artifacts into `backend/api/api_server.py`.
4. Proceed to Phase 3: Upgrade the Next.js frontend to display multi-disease Mode A and Mode B assessment cards with SHAP visualizations.
5. Finalize the IEEE research paper manuscript using the verified metrics, figures, and non-causal interpretability framing.
