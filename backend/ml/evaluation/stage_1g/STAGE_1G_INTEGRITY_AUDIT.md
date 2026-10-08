# STAGE 1G-1 — Locked V2 Research Pipeline Integrity Audit

> **Stage:** 1G-1 (Final Integrity Audit of Locked V2 Pipeline)  
> **Source Population:** NHANES 2021–2023 (Adults $\ge 20$ years)  
> **scikit-learn Version:** `1.9.0` | **xgboost Version:** `3.4.1` | **Random Seed:** `42`  
> **Audit Status:** Complete | **Overall Result:** **PASS**

---

## 1. Audit Scope

This audit evaluates the integrity, reproducibility, and consistency of the six locked final **Model V2** candidate pipelines finalized in Stage 1F of the AI Health Risk Scoring System.

The audit verified:
1. **Model & Ancillary Artifacts:** Physical existence, binary loadability, file sizes, and formats (`.joblib`, `.pkl`, calibration CSVs, threshold sweep CSVs).
2. **Model Architectures & Ensembles:** Correct estimator families, pipeline wrappers, hyperparameters, and `FrozenEstimator` unwrapping within `CalibratedClassifierCV`.
3. **Feature Specifications & Alignments:** Strict 1-to-1 parity between processed Parquet datasets, recorded JSON feature sets, and model input expectations across Mode A and Mode B.
4. **Data Partitioning & Integrity:** Exact 70% Train / 15% Validation / 15% Test participant-level split counts, zero participant (`SEQN`) overlap between sets, and split consistency across modes.
5. **Probability Calibration:** Proper application of Platt/Sigmoid calibration, OOF validation selection, and final full-validation refitting for production deployment.
6. **Operating Decision Thresholds:** Verification of locked clinical screening thresholds against empirical 5-fold OOF validation sweeps and Youden's J optimization.
7. **Evaluation Metric Parity:** Recomputed independent test-set inference matching reported metrics in `stage_1f_results.json` and `stage_1f_comparison.csv` with zero discrepancies ($< 0.0001$).
8. **Leakage & Survey Variable Exclusion:** Strict absence of target diagnostic variables, survey design columns, and participant identifiers from all predictor matrices.
9. **V1 / Prototype Contamination:** Rigorous screening confirming total absence of deprecated prototype features (e.g., HRV metrics, SDNN, RMSSD, legacy family history, resting/peak HR).

---

## 2. Locked V2 Configurations

The six candidate disease-mode configurations established in Stage 1E and locked in Stage 1F:

| Pipeline | Disease Target | Mode | Candidate Model Architecture | Source Stage | Locked Calibration | Locked Threshold ($t_{\text{opt}}$) |
|---|---|---|---|---|---|---|
| **`cvd_mode_a`** | Cardiovascular Disease (`target_cvd`) | Mode A (Non-invasive) | Optimized Random Forest (`n_est=500`, `max_depth=8`, `log2`) | Stage 1E-3 | Sigmoid (Platt) | **0.13** |
| **`cvd_mode_b`** | Cardiovascular Disease (`target_cvd`) | Mode B (Biomarker-augmented) | Optimized Random Forest (`n_est=800`, `max_depth=8`, `feat=0.5`) | Stage 1E-3 | Sigmoid (Platt) | **0.15** |
| **`diabetes_mode_a`** | Diabetes Mellitus (`target_diabetes`) | Mode A (Non-invasive) | HistGradientBoosting (`lr=0.05`, `max_leaf=15`, `l2=10`) | Stage 1E-2 | Sigmoid (Platt) | **0.15** |
| **`diabetes_mode_b`** | Diabetes Mellitus (`target_diabetes`) | Mode B (Biomarker-augmented) | XGBoost (`n_est=500`, `max_depth=5`, `lr=0.01`, `sub=0.6`) | Stage 1E-2 | Sigmoid (Platt) | **0.17** |
| **`hypertension_mode_a`** | Hypertension (`target_hypertension`) | Mode A (Non-invasive) | Logistic Regression (`C=10.0`, `balanced`, `lbfgs`) | Stage 1E-2 | Sigmoid (Platt) | **0.41** |
| **`hypertension_mode_b`** | Hypertension (`target_hypertension`) | Mode B (Biomarker-augmented) | Optimized Random Forest (`n_est=800`, `max_depth=8`, `feat=0.5`) | Stage 1E-3 | Sigmoid (Platt) | **0.52** |

---

## 3. Artifact Verification

All required artifact directories, model files, calibration tables, and threshold sweep files exist, are accessible, and load cleanly without warnings.

### Artifact Status Table

| Pipeline | Model `.joblib` Path | Size (Bytes) | Model `.pkl` Path | Calibration CSV | Threshold Sweep CSV | Load Status | Check Result |
|---|---|---|---|---|---|---|---|
| **`cvd_mode_a`** | `backend/ml/evaluation/stage_1f/models/cvd_mode_a_calibrated.joblib` | 8,433,283 | Exists (8,386,154 B) | `cvd_mode_a_calibration_data.csv` (2,620 B) | `cvd_mode_a_threshold_sweep.csv` (7,189 B) | Loaded OK | **PASS** |
| **`cvd_mode_b`** | `backend/ml/evaluation/stage_1f/models/cvd_mode_b_calibrated.joblib` | 8,765,914 | Exists (8,695,625 B) | `cvd_mode_b_calibration_data.csv` (2,597 B) | `cvd_mode_b_threshold_sweep.csv` (6,864 B) | Loaded OK | **PASS** |
| **`diabetes_mode_a`** | `backend/ml/evaluation/stage_1f/models/diabetes_mode_a_calibrated.joblib` | 380,978 | Exists (366,774 B) | `diabetes_mode_a_calibration_data.csv` (2,684 B) | `diabetes_mode_a_threshold_sweep.csv` (7,031 B) | Loaded OK | **PASS** |
| **`diabetes_mode_b`** | `backend/ml/evaluation/stage_1f/models/diabetes_mode_b_calibrated.joblib` | 978,468 | Exists (978,262 B) | `diabetes_mode_b_calibration_data.csv` (2,331 B) | `diabetes_mode_b_threshold_sweep.csv` (7,398 B) | Loaded OK | **PASS** |
| **`hypertension_mode_a`** | `backend/ml/evaluation/stage_1f/models/hypertension_mode_a_calibrated.joblib` | 3,437 | Exists (2,534 B) | `hypertension_mode_a_calibration_data.csv` (2,850 B) | `hypertension_mode_a_threshold_sweep.csv` (7,430 B) | Loaded OK | **PASS** |
| **`hypertension_mode_b`** | `backend/ml/evaluation/stage_1f/models/hypertension_mode_b_calibrated.joblib` | 12,649,364 | Exists (12,579,566 B) | `hypertension_mode_b_calibration_data.csv` (2,967 B) | `hypertension_mode_b_threshold_sweep.csv` (7,450 B) | Loaded OK | **PASS** |

### Estimator Architecture Verification

- All 6 saved models are instances of `sklearn.calibration.CalibratedClassifierCV`.
- For models wrapping scikit-learn `Pipeline` (CVD Mode A, CVD Mode B, Hypertension Mode A, Hypertension Mode B), the pipeline steps and estimators were unwrapped from `FrozenEstimator` and verified:
  - `cvd_mode_a`: `Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', RandomForestClassifier())])`
  - `cvd_mode_b`: `Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', RandomForestClassifier())])`
  - `hypertension_mode_a`: `Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('clf', LogisticRegression())])`
  - `hypertension_mode_b`: `Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', RandomForestClassifier())])`
- For standalone models:
  - `diabetes_mode_a`: `HistGradientBoostingClassifier`
  - `diabetes_mode_b`: `XGBClassifier`
- **Result:** **PASS**

---

## 4. Feature Verification

Feature matrices extracted from `backend/ml/data/processed/nhanes_2021_2023/*.parquet` were cross-referenced against `stage_1f_results.json`.

### Feature Counts & Composition

| Pipeline | Predictor Count | Feature Set Description | Target Exclusion | Non-Predictor Design Columns Dropped | Check Result |
|---|---|---|---|---|---|
| **`cvd_mode_a`** | **13** | 13 non-invasive clinical & demographic features | `target_cvd` | `SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU` | **PASS** |
| **`cvd_mode_b`** | **29** | 13 non-invasive + 16 blood/urine biomarkers | `target_cvd` | `SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU` | **PASS** |
| **`diabetes_mode_a`** | **13** | 13 non-invasive clinical & demographic features | `target_diabetes` | `SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU` | **PASS** |
| **`diabetes_mode_b`** | **27** | 13 non-invasive + 14 non-glycemic biomarkers | `target_diabetes`, `hba1c`, `fasting_glucose` | `SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU` | **PASS** |
| **`hypertension_mode_a`** | **11** | 11 non-invasive features (BP readings excluded) | `target_hypertension`, `mean_sbp`, `mean_dbp` | `SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU` | **PASS** |
| **`hypertension_mode_b`** | **27** | 11 non-invasive + 16 blood/urine biomarkers | `target_hypertension`, `mean_sbp`, `mean_dbp` | `SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU` | **PASS** |

### Verified Exact Feature Names

- **CVD Mode A (13):**
  `['age', 'gender', 'education_level', 'poverty_income_ratio', 'bmi', 'waist_circumference', 'mean_sbp', 'mean_dbp', 'mean_pulse', 'smoking_status', 'alcohol_frequency', 'physical_activity_level', 'sedentary_minutes']`
- **CVD Mode B (29):**
  CVD Mode A (13) + `['hba1c', 'fasting_glucose', 'total_cholesterol', 'hdl_cholesterol', 'triglycerides', 'ldl_cholesterol', 'serum_creatinine', 'blood_urea_nitrogen', 'serum_uric_acid', 'alt_enzyme', 'ast_enzyme', 'hemoglobin', 'wbc_count', 'platelet_count', 'rdw', 'serum_albumin']`
- **Diabetes Mode A (13):**
  `['age', 'gender', 'education_level', 'poverty_income_ratio', 'bmi', 'waist_circumference', 'mean_sbp', 'mean_dbp', 'mean_pulse', 'smoking_status', 'alcohol_frequency', 'physical_activity_level', 'sedentary_minutes']`
- **Diabetes Mode B (27):**
  Diabetes Mode A (13) + `['total_cholesterol', 'hdl_cholesterol', 'triglycerides', 'ldl_cholesterol', 'serum_creatinine', 'blood_urea_nitrogen', 'serum_uric_acid', 'alt_enzyme', 'ast_enzyme', 'hemoglobin', 'wbc_count', 'platelet_count', 'rdw', 'serum_albumin']` *(Note: `hba1c` and `fasting_glucose` strictly omitted to avoid target circularity)*.
- **Hypertension Mode A (11):**
  `['age', 'gender', 'education_level', 'poverty_income_ratio', 'bmi', 'waist_circumference', 'mean_pulse', 'smoking_status', 'alcohol_frequency', 'physical_activity_level', 'sedentary_minutes']` *(Note: `mean_sbp` and `mean_dbp` strictly omitted to avoid target circularity)*.
- **Hypertension Mode B (27):**
  Hypertension Mode A (11) + `['hba1c', 'fasting_glucose', 'total_cholesterol', 'hdl_cholesterol', 'triglycerides', 'ldl_cholesterol', 'serum_creatinine', 'blood_urea_nitrogen', 'serum_uric_acid', 'alt_enzyme', 'ast_enzyme', 'hemoglobin', 'wbc_count', 'platelet_count', 'rdw', 'serum_albumin']`.

- Feature ordering and naming between Parquet columns and JSON metadata match **100%**.
- **Result:** **PASS**

---

## 5. Split Verification

The participant-level split files located under `backend/ml/data/processed/nhanes_2021_2023/splits/` were audited for row counts, ratio consistency, and participant disjointness.

| Disease Split File | Train Rows ($N_{\text{train}}$) | Validation Rows ($N_{\text{val}}$) | Test Rows ($N_{\text{test}}$) | Total Population | Train/Val/Test Ratios | Participant Disjointness (`SEQN` Overlap) | Check Result |
|---|---|---|---|---|---|---|---|
| **`cvd_splits.csv`** | 5,464 | 1,171 | 1,172 | 7,807 | 70.0% / 15.0% / 15.0% | **0 overlaps** ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$) | **PASS** |
| **`diabetes_splits.csv`** | 5,464 | 1,171 | 1,171 | 7,806 | 70.0% / 15.0% / 15.0% | **0 overlaps** ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$) | **PASS** |
| **`hypertension_splits.csv`** | 5,460 | 1,170 | 1,170 | 7,800 | 70.0% / 15.0% / 15.0% | **0 overlaps** ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$) | **PASS** |

- Partition row counts match the recorded pipeline observations in `stage_1f_results.json` identically.
- Mode A and Mode B for each respective disease use the exact identical participant partition files.
- **Result:** **PASS**

---

## 6. Calibration Verification

Probability calibration was audited for method selection, out-of-fold validation rigor, and deployment artifact refitting.

| Pipeline | Model Architecture | Evaluated Methods | Selected Method | Selection Rationale Verified | Refitted on Full Val Set | Check Result |
|---|---|---|---|---|---|---|
| **`cvd_mode_a`** | Optimized Random Forest | Raw, Sigmoid, Isotonic | **Sigmoid** | Brier: 0.0922, ECE: 0.0111 (Superior to Raw 0.0271, avoids Isotonic step-overfitting) | Yes (`X_val`, `y_val`) | **PASS** |
| **`cvd_mode_b`** | Optimized Random Forest | Raw, Sigmoid, Isotonic | **Sigmoid** | Brier: 0.0884, ECE: 0.0101 (Superior to Raw 0.0167, stable Platt mapping) | Yes (`X_val`, `y_val`) | **PASS** |
| **`diabetes_mode_a`** | HistGradientBoosting | Raw, Sigmoid, Isotonic | **Sigmoid** | Brier: 0.1231, ECE: 0.0373 (Lower Brier than Raw 0.1238, avoids step artifacts) | Yes (`X_val`, `y_val`) | **PASS** |
| **`diabetes_mode_b`** | XGBoost | Raw, Sigmoid, Isotonic | **Sigmoid** | Brier: 0.1136, ECE: 0.0346 (ECE reduced from 0.0395 raw, smooth sigmoid mapping) | Yes (`X_val`, `y_val`) | **PASS** |
| **`hypertension_mode_a`** | Logistic Regression | Raw, Sigmoid, Isotonic | **Sigmoid** | Brier: 0.1848, ECE: 0.0323 (ECE reduced from 0.0582 raw) | Yes (`X_val`, `y_val`) | **PASS** |
| **`hypertension_mode_b`** | Optimized Random Forest | Raw, Sigmoid, Isotonic | **Sigmoid** | Brier: 0.1772, ECE: 0.0334 (Maintains strong calibration, smooth mapping) | Yes (`X_val`, `y_val`) | **PASS** |

- Calibration method recorded in `stage_1f_results.json`, `stage_1f_comparison.csv`, and model objects is uniformly `'sigmoid'`.
- Calibrated probability tables in `backend/ml/evaluation/stage_1f/calibration/*_calibration_data.csv` correspond directly to the OOF validation curve evaluations.
- **Result:** **PASS**

---

## 7. Threshold Verification

Operating threshold sweeps conducted on 5-fold OOF validation probabilities were cross-checked against the threshold sweep tables and locked model metadata.

| Pipeline | Locked Threshold ($t_{\text{opt}}$) | Threshold Sweep CSV Path | Validation Youden's J ($J_{\text{OOF}}$) | Validation Sensitivity ($S_{\text{OOF}}$) | Validation Specificity ($S_{\text{OOF}}$) | Validation F1 ($F_{1,\text{OOF}}$) | Selection Criteria Compliance | Check Result |
|---|---|---|---|---|---|---|---|---|
| **`cvd_mode_a`** | **0.13** | `backend/ml/evaluation/stage_1f/thresholds/cvd_mode_a_threshold_sweep.csv` | 0.5194 | 0.8299 | 0.6895 | 0.4157 | Optimized Youden's J with Sens $\ge 0.65$ & Spec $\ge 0.65$ | **PASS** |
| **`cvd_mode_b`** | **0.15** | `backend/ml/evaluation/stage_1f/thresholds/cvd_mode_b_threshold_sweep.csv` | 0.5372 | 0.7755 | 0.7617 | 0.4515 | Optimized Youden's J with Sens $\ge 0.65$ & Spec $\ge 0.65$ | **PASS** |
| **`diabetes_mode_a`** | **0.15** | `backend/ml/evaluation/stage_1f/thresholds/diabetes_mode_a_threshold_sweep.csv` | 0.4788 | 0.8173 | 0.6615 | 0.4830 | Optimized Youden's J with Sens $\ge 0.65$ & Spec $\ge 0.65$ | **PASS** |
| **`diabetes_mode_b`** | **0.17** | `backend/ml/evaluation/stage_1f/thresholds/diabetes_mode_b_threshold_sweep.csv` | 0.5139 | 0.7933 | 0.7207 | 0.5140 | Optimized Youden's J with Sens $\ge 0.65$ & Spec $\ge 0.65$ | **PASS** |
| **`hypertension_mode_a`** | **0.41** | `backend/ml/evaluation/stage_1f/thresholds/hypertension_mode_a_threshold_sweep.csv` | 0.4616 | 0.7900 | 0.6716 | 0.7085 | Optimized Youden's J with Sens $\ge 0.65$ & Spec $\ge 0.65$ | **PASS** |
| **`hypertension_mode_b`** | **0.52** | `backend/ml/evaluation/stage_1f/thresholds/hypertension_mode_b_threshold_sweep.csv` | 0.4917 | 0.7260 | 0.7657 | 0.7118 | Optimized Youden's J with Sens $\ge 0.65$ & Spec $\ge 0.65$ | **PASS** |

- Locked thresholds match `stage_1f_results.json` and `stage_1f_comparison.csv` with zero discrepancies.
- **Result:** **PASS**

---

## 8. Metric Consistency Verification

To ensure zero fabrication, stale caching, or rounding divergence, an independent test evaluation script was executed against the locked model artifacts and held-out test partitions. The recalculated metrics were compared against `stage_1f_results.json` and `stage_1f_comparison.csv`.

| Pipeline | Metric | Recalculated Test Value | `stage_1f_results.json` | `stage_1f_comparison.csv` | Absolute Delta ($\Delta$) | Status |
|---|---|---|---|---|---|---|
| **`cvd_mode_a`** | ROC-AUC | 0.813737 | 0.8137 | 0.8137 | $< 0.00004$ | **PASS** |
| | PR-AUC | 0.415895 | 0.4159 | 0.4159 | $< 0.00001$ | **PASS** |
| | Brier Score | 0.091203 | 0.0912 | 0.0912 | $< 0.00001$ | **PASS** |
| | Sensitivity | 0.682432 | 0.6824 | 0.6824 | $< 0.00004$ | **PASS** |
| | Specificity | 0.780273 | 0.7803 | 0.7803 | $< 0.00003$ | **PASS** |
| | F1 Score | 0.426160 | 0.4262 | 0.4262 | $< 0.00004$ | **PASS** |
| **`cvd_mode_b`** | ROC-AUC | 0.821774 | 0.8218 | 0.8218 | $< 0.00003$ | **PASS** |
| | PR-AUC | 0.434043 | 0.4340 | 0.4340 | $< 0.00005$ | **PASS** |
| | Brier Score | 0.090647 | 0.0906 | 0.0906 | $< 0.00005$ | **PASS** |
| | Sensitivity | 0.608108 | 0.6081 | 0.6081 | $< 0.00001$ | **PASS** |
| | Specificity | 0.843750 | 0.8438 | 0.8438 | $< 0.00005$ | **PASS** |
| | F1 Score | 0.452261 | 0.4523 | 0.4523 | $< 0.00004$ | **PASS** |
| **`diabetes_mode_a`** | ROC-AUC | 0.795724 | 0.7957 | 0.7957 | $< 0.00003$ | **PASS** |
| | PR-AUC | 0.434801 | 0.4348 | 0.4348 | $< 0.00001$ | **PASS** |
| | Brier Score | 0.122221 | 0.1222 | 0.1222 | $< 0.00003$ | **PASS** |
| | Sensitivity | 0.817308 | 0.8173 | 0.8173 | $< 0.00001$ | **PASS** |
| | Specificity | 0.639668 | 0.6397 | 0.6397 | $< 0.00004$ | **PASS** |
| | F1 Score | 0.468966 | 0.4690 | 0.4690 | $< 0.00004$ | **PASS** |
| **`diabetes_mode_b`** | ROC-AUC | 0.816222 | 0.8162 | 0.8162 | $< 0.00003$ | **PASS** |
| | PR-AUC | 0.479349 | 0.4793 | 0.4793 | $< 0.00005$ | **PASS** |
| | Brier Score | 0.120335 | 0.1203 | 0.1203 | $< 0.00004$ | **PASS** |
| | Sensitivity | 0.620192 | 0.6202 | 0.6202 | $< 0.00001$ | **PASS** |
| | Specificity | 0.809969 | 0.8100 | 0.8100 | $< 0.00004$ | **PASS** |
| | F1 Score | 0.496154 | 0.4962 | 0.4962 | $< 0.00005$ | **PASS** |
| **`hypertension_mode_a`** | ROC-AUC | 0.766236 | 0.7662 | 0.7662 | $< 0.00004$ | **PASS** |
| | PR-AUC | 0.654116 | 0.6541 | 0.6541 | $< 0.00002$ | **PASS** |
| | Brier Score | 0.194612 | 0.1946 | 0.1946 | $< 0.00002$ | **PASS** |
| | Sensitivity | 0.761523 | 0.7615 | 0.7615 | $< 0.00003$ | **PASS** |
| | Specificity | 0.664680 | 0.6647 | 0.6647 | $< 0.00002$ | **PASS** |
| | F1 Score | 0.688372 | 0.6884 | 0.6884 | $< 0.00001$ | **PASS** |
| **`hypertension_mode_b`** | ROC-AUC | 0.773140 | 0.7731 | 0.7731 | $< 0.00004$ | **PASS** |
| | PR-AUC | 0.689837 | 0.6898 | 0.6898 | $< 0.00004$ | **PASS** |
| | Brier Score | 0.191647 | 0.1916 | 0.1916 | $< 0.00005$ | **PASS** |
| | Sensitivity | 0.625251 | 0.6253 | 0.6253 | $< 0.00005$ | **PASS** |
| | Specificity | 0.751118 | 0.7511 | 0.7511 | $< 0.00002$ | **PASS** |
| | F1 Score | 0.638037 | 0.6380 | 0.6380 | $< 0.00004$ | **PASS** |

- All recalculated values match stored values within numerical precision limits ($< 10^{-4}$).
- **Result:** **PASS**

---

## 9. Leakage Sanity Check

An exhaustive variable-level audit was conducted across all six predictor matrices to ensure zero leakage of survey design information, target variables, or diagnostic criteria.

### Excluded Non-Predictor Design Columns Checked
The following six survey-design and identifier variables were checked and confirmed **completely absent** from all model feature sets:
1. `SEQN` (Participant Sequence Number)
2. `WTINT2YR` (Interview 2-Year Sample Weight)
3. `WTMEC2YR` (Mobile Examination Center 2-Year Weight)
4. `WTSAF2YR` (Fasting Subsample 2-Year Weight)
5. `SDMVSTRA` (Masked Variance Pseudo-Stratum)
6. `SDMVPSU` (Masked Variance Pseudo-PSU)

### Excluded Disease Diagnostic Target Columns Checked
The following diagnostic and definition-defining variables were confirmed **completely absent** from predictors:
- **CVD Pipelines:**
  - `target_cvd` (Primary target label)
  - Diagnostic variables: `mcq160b` (Congestive heart failure), `mcq160c` (Coronary heart disease), `mcq160d` (Angina pectoris), `mcq160e` (Heart attack), `mcq160f` (Stroke), `mcq160a`, `mcq160m`, `mcq160k`, `mcq160l`, `mcq220`.
- **Diabetes Pipelines:**
  - `target_diabetes` (Primary target label)
  - Diagnostic variables: `diq010` (Doctor diagnosed diabetes), `did040` (Age diagnosed), `diq050` (Insulin use), `diq070` (Pills for diabetes).
  - Mode B Biomarker Guardrails: `lbxdba` / `hba1c` (Glycated hemoglobin) and `lbxglu` / `fasting_glucose` (Fasting glucose) are **strictly excluded** from Diabetes Mode B to avoid target-defining leakage.
- **Hypertension Pipelines:**
  - `target_hypertension` (Primary target label)
  - Diagnostic variables: `bpq020` (Doctor diagnosed hypertension), `bpq030`, `bpq040a`, `bpq050a` (Antihypertensive medication).
  - Mode A/B Blood Pressure Guardrails: `mean_sbp`, `mean_dbp`, `bpa_mean_sbp`, `bpa_mean_dbp` are **strictly excluded** from both Hypertension Mode A and Mode B to avoid circular target derivation.

- **Check Result:** **PASS** (Zero leakage detected).

---

## 10. V1 / Prototype Contamination Check

To guarantee that the research pipeline uses authentic, reproducible NHANES 2021–2023 clinical datasets rather than legacy prototype or synthetic variables, all features were scanned for V1 artifact names:

| Prototype / V1 Variable Tested | Status in V2 Predictors | Comment | Check Result |
|---|---|---|---|
| `SDNN` (Standard deviation of NN intervals) | **NOT FOUND** | Excluded; legacy V1 smartwatch prototype feature | **PASS** |
| `RMSSD` (Root mean square of successive differences) | **NOT FOUND** | Excluded; legacy V1 smartwatch prototype feature | **PASS** |
| `HRV` / `heart_rate_variability` | **NOT FOUND** | Excluded; legacy V1 prototype feature | **PASS** |
| Legacy Family History variables (`family_history_*`) | **NOT FOUND** | Excluded; subjective legacy questionnaire feature | **PASS** |
| Legacy Heart Rate variables (`resting_heart_rate`, `peak_hr`) | **NOT FOUND** | Replaced by standardized NHANES physical examination pulse (`mean_pulse`) | **PASS** |

- Zero deprecated V1 prototype features exist in any V2 model artifact or dataset.
- **Check Result:** **PASS**

---

## 11. Overall Audit Result

Across all 10 inspection dimensions and all six disease-mode candidate pipelines, the locked V2 research pipeline demonstrates total internal consistency, complete artifact reproducibility, zero data leakage, and rigorous split separation.

| Dimension | Scope | Result |
|---|---|---|
| 1. Artifact Verification | Model files, calibration CSVs, threshold sweeps exist and load | **PASS** |
| 2. Architecture Verification | Random Forest, HistGB, XGBoost, Logistic Regression match specifications | **PASS** |
| 3. Feature Set Parity | Parquet datasets match recorded JSON feature vectors 100% | **PASS** |
| 4. Partition Integrity | 70/15/15 participant split with zero participant overlap | **PASS** |
| 5. Calibration Mapping | Platt/Sigmoid calibration applied and refitted on full validation set | **PASS** |
| 6. Threshold Locking | Screening thresholds correspond to empirical OOF Youden's J sweep | **PASS** |
| 7. Metric Parity | Independent test inference matches published test metrics ($< 10^{-4}$) | **PASS** |
| 8. Leakage Sanity | Zero survey weights, zero target diagnostic circularity | **PASS** |
| 9. V1 Contamination | Zero legacy smartwatch / synthetic variables | **PASS** |

---

### OVERALL STATUS: **PASS**
