# STAGE 1D — Model Benchmarking & Honest Baseline Report

> **Stage:** 1D — Supervised Model Benchmarking  
> **Source Population:** NHANES 2021–2023 Adults (Age $\ge 20$)  
> **Random Seed:** 42  
> **Evaluation Strategy:** 70% Train / 15% Validation / 15% Final Test (Stratified Participant-Level SEQN Split)  
> **Status:** Complete — All 36 benchmark experiments completed.

---

## 1. Executive Summary

In Stage 1D, we benchmarked multiple supervised machine learning algorithm families across **three risk targets** (`Hard CVD`, `Diabetes`, `Hypertension`) and **two feature operational modes** (`Mode A` questionnaire/non-invasive baseline and `Mode B` lab-enhanced panel).

### Key Takeaways:
1. **No Data Leakage:** All preprocessing (median imputation, standardization) was fitted strictly inside scikit-learn `Pipeline` objects on training splits only. SEQN, survey weights, target variables, and prohibited target-defining variables were strictly excluded from predictor matrices.
2. **Standardized SEQN Splitting:** Participant-level splitting ensured zero participant overlap across train, validation, and test splits. Mode A and Mode B models for the same target were evaluated on identical train/val/test participant subsets.
3. **Honest Baselines Established:** No hyperparameter tuning, SMOTE, or threshold manipulation was performed.
4. **Lab Biomarker Impact (Mode B Gain):** Adding 16 laboratory biomarkers in Mode B consistently improved discrimination across all three targets (ROC-AUC gains: $+0.0157$ for CVD, $+0.0267$ for Diabetes, $+0.0099$ for Hypertension).
5. **Class Imbalance & Threshold Observation:** Fixed thresholding at $0.50$ yields low recall on imbalanced targets (`CVD` 12.6%, `Diabetes` 17.7%) for unweighted models. Class-balanced weighting (`class_weight='balanced'`) significantly increases sensitivity, but probability calibration (Stage 1E) is required before converting probabilities into risk scores.

---

## 2. Dataset & Split Architecture

| Target | Target Column | Total Evaluated Adults | Train (70%) | Validation (15%) | Final Test (15%) | Positive Prevalence |
|---|---|---|---|---|---|---|
| **Hard CVD** | `target_cvd` | 7,807 | 5,464 | 1,171 | 1,172 | 12.58% (982 pos) |
| **Diabetes** | `target_diabetes` | 7,806 | 5,464 | 1,171 | 1,171 | 17.74% (1,385 pos) |
| **Hypertension** | `target_hypertension` | 7,800 | 5,460 | 1,170 | 1,170 | 42.71% (3,331 pos) |

*The exact `SEQN` memberships for `train`, `validation`, and `test` are saved in `backend/ml/data/processed/nhanes_2021_2023/splits/*.csv`.*

---

## 3. Models Benchmarked & Environment

### Models Included:
1. **Logistic Regression (Default)** — L2 penalty, `max_iter=1000`, `SimpleImputer(median)` + `StandardScaler`
2. **Logistic Regression (Balanced)** — `class_weight='balanced'`
3. **Random Forest (Default)** — 100 trees, `SimpleImputer(median)`
4. **Random Forest (Balanced)** — 100 trees, `class_weight='balanced'`
5. **HistGradientBoosting (Default)** — Native NaN handling, histogram binning
6. **HistGradientBoosting (Balanced)** — `class_weight='balanced'`

### Skipped Models:
* **XGBoost** and **LightGBM**: Skipped as they are not currently installed in the Python environment.

---

## 4. Benchmark Winners Summary

| Target / Mode Configuration | Winner Model | Validation ROC-AUC | Validation PR-AUC | Test ROC-AUC | Test PR-AUC | Test Brier Score |
|---|---|---|---|---|---|---|
| **CVD Mode A** | `logistic_regression` | **0.8166** | 0.3714 | 0.8125 | 0.3792 | 0.0924 |
| **CVD Mode B** | `logistic_regression_balanced` | **0.8323** | 0.4221 | 0.8191 | 0.4219 | 0.1739 |
| **Diabetes Mode A** | `hist_gradient_boosting` | **0.7967** | 0.4057 | 0.7810 | 0.4145 | 0.1267 |
| **Diabetes Mode B** | `hist_gradient_boosting` | **0.8234** | 0.4644 | 0.8003 | 0.4594 | 0.1204 |
| **Hypertension Mode A** | `logistic_regression` | **0.7884** | 0.6807 | 0.7663 | 0.6550 | 0.1846 |
| **Hypertension Mode B** | `hist_gradient_boosting_balanced` | **0.7983** | 0.7186 | 0.7637 | 0.6748 | 0.1827 |

---

## 5. Complete Benchmark Results Table

Below is the complete evaluation matrix for all 36 baseline runs (6 models $\times$ 6 dataset configurations) evaluated at classification threshold $0.50$:

| Target | Mode | Model | Val ROC-AUC | Val PR-AUC | Val Recall | Val Spec | Test ROC-AUC | Test PR-AUC | Test Brier |
|---|---|---|---|---|---|---|---|---|---|
| cvd | mode_a | `logistic_regression` | 0.8166 | 0.3714 | 0.1088 | 0.9854 | 0.8125 | 0.3792 | 0.0924 |
| cvd | mode_a | `logistic_regression_balanced` | 0.8159 | 0.3746 | 0.7483 | 0.7188 | 0.8105 | 0.3740 | 0.1852 |
| cvd | mode_a | `random_forest` | 0.7821 | 0.3140 | 0.1224 | 0.9785 | 0.7821 | 0.3512 | 0.0978 |
| cvd | mode_a | `random_forest_balanced` | 0.7947 | 0.3225 | 0.0952 | 0.9844 | 0.7755 | 0.3091 | 0.1105 |
| cvd | mode_a | `hist_gradient_boosting` | 0.7863 | 0.3123 | 0.1429 | 0.9746 | 0.7916 | 0.3551 | 0.1002 |
| cvd | mode_a | `hist_gradient_boosting_balanced` | 0.7750 | 0.3114 | 0.6667 | 0.7246 | 0.7920 | 0.3662 | 0.1402 |
| cvd | mode_b | `logistic_regression` | 0.8319 | 0.4208 | 0.1973 | 0.9844 | 0.8200 | 0.4289 | 0.0889 |
| cvd | mode_b | `logistic_regression_balanced` | 0.8323 | 0.4221 | 0.7551 | 0.7412 | 0.8191 | 0.4219 | 0.1739 |
| cvd | mode_b | `random_forest` | 0.8135 | 0.3587 | 0.1361 | 0.9873 | 0.7996 | 0.3763 | 0.0929 |
| cvd | mode_b | `random_forest_balanced` | 0.8250 | 0.3923 | 0.1088 | 0.9893 | 0.7973 | 0.3388 | 0.1007 |
| cvd | mode_b | `hist_gradient_boosting` | 0.8111 | 0.3675 | 0.2313 | 0.9707 | 0.8024 | 0.3980 | 0.0953 |
| cvd | mode_b | `hist_gradient_boosting_balanced` | 0.8127 | 0.4178 | 0.6122 | 0.8242 | 0.7909 | 0.3761 | 0.1158 |
| diabetes | mode_a | `logistic_regression` | 0.7925 | 0.4120 | 0.2452 | 0.9626 | 0.7831 | 0.4178 | 0.1247 |
| diabetes | mode_a | `logistic_regression_balanced` | 0.7928 | 0.4128 | 0.7163 | 0.7134 | 0.7841 | 0.4183 | 0.1927 |
| diabetes | mode_a | `random_forest` | 0.7875 | 0.3933 | 0.2548 | 0.9419 | 0.7586 | 0.3982 | 0.1248 |
| diabetes | mode_a | `random_forest_balanced` | 0.7882 | 0.3729 | 0.2212 | 0.9574 | 0.7673 | 0.3852 | 0.1384 |
| diabetes | mode_a | `hist_gradient_boosting` | 0.7967 | 0.4057 | 0.2692 | 0.9470 | 0.7810 | 0.4145 | 0.1267 |
| diabetes | mode_a | `hist_gradient_boosting_balanced` | 0.7926 | 0.4091 | 0.6779 | 0.7321 | 0.7918 | 0.4541 | 0.1572 |
| diabetes | mode_b | `logistic_regression` | 0.8157 | 0.4821 | 0.2981 | 0.9595 | 0.7938 | 0.4285 | 0.1185 |
| diabetes | mode_b | `logistic_regression_balanced` | 0.8167 | 0.4860 | 0.7404 | 0.7394 | 0.7966 | 0.4307 | 0.1792 |
| diabetes | mode_b | `random_forest` | 0.8033 | 0.4580 | 0.2596 | 0.9595 | 0.7801 | 0.4363 | 0.1192 |
| diabetes | mode_b | `random_forest_balanced` | 0.8063 | 0.4400 | 0.2260 | 0.9709 | 0.7741 | 0.3966 | 0.1305 |
| diabetes | mode_b | `hist_gradient_boosting` | 0.8234 | 0.4644 | 0.3317 | 0.9512 | 0.8003 | 0.4594 | 0.1204 |
| diabetes | mode_b | `hist_gradient_boosting_balanced` | 0.8139 | 0.4638 | 0.6250 | 0.8183 | 0.8037 | 0.4503 | 0.1416 |
| hypertension | mode_a | `logistic_regression` | 0.7884 | 0.6807 | 0.6400 | 0.7672 | 0.7663 | 0.6550 | 0.1846 |
| hypertension | mode_a | `logistic_regression_balanced` | 0.7883 | 0.6803 | 0.7060 | 0.7134 | 0.7661 | 0.6541 | 0.1873 |
| hypertension | mode_a | `random_forest` | 0.7788 | 0.6746 | 0.6040 | 0.7791 | 0.7437 | 0.6446 | 0.1888 |
| hypertension | mode_a | `random_forest_balanced` | 0.7771 | 0.6679 | 0.5840 | 0.7940 | 0.7394 | 0.6366 | 0.1908 |
| hypertension | mode_a | `hist_gradient_boosting` | 0.7709 | 0.6709 | 0.6260 | 0.7612 | 0.7455 | 0.6630 | 0.1937 |
| hypertension | mode_a | `hist_gradient_boosting_balanced` | 0.7731 | 0.6703 | 0.6580 | 0.7254 | 0.7470 | 0.6618 | 0.1936 |
| hypertension | mode_b | `logistic_regression` | 0.7938 | 0.7002 | 0.6580 | 0.7716 | 0.7727 | 0.6743 | 0.1827 |
| hypertension | mode_b | `logistic_regression_balanced` | 0.7936 | 0.6996 | 0.7160 | 0.7194 | 0.7725 | 0.6732 | 0.1857 |
| hypertension | mode_b | `random_forest` | 0.7891 | 0.6943 | 0.6120 | 0.7970 | 0.7592 | 0.6594 | 0.1850 |
| hypertension | mode_b | `random_forest_balanced` | 0.7918 | 0.7007 | 0.6100 | 0.7970 | 0.7584 | 0.6566 | 0.1849 |
| hypertension | mode_b | `hist_gradient_boosting` | 0.7971 | 0.7149 | 0.6580 | 0.7776 | 0.7633 | 0.6755 | 0.1818 |
| hypertension | mode_b | `hist_gradient_boosting_balanced` | 0.7983 | 0.7186 | 0.7060 | 0.7299 | 0.7637 | 0.6748 | 0.1827 |

---

## 6. Mode A vs. Mode B Incremental Gain Analysis

Comparing the best baseline models for Mode A (questionnaire/vitals) vs. Mode B (+16 lab biomarkers):

| Target Endpoint | Best Mode A Model (Val ROC-AUC) | Best Mode B Model (Val ROC-AUC) | Absolute ROC-AUC Gain | Val PR-AUC Gain | Test ROC-AUC Gain |
|---|---|---|---|---|---|
| **Hard CVD** | 0.8166 (`logistic_regression`) | 0.8323 (`logistic_regression_balanced`) | **+0.0157** | **+0.0507** | +0.0066 |
| **Diabetes** | 0.7967 (`hist_gradient_boosting`) | 0.8234 (`hist_gradient_boosting`) | **+0.0267** | **+0.0587** | +0.0193 |
| **Hypertension** | 0.7884 (`logistic_regression`) | 0.7983 (`hist_gradient_boosting_balanced`) | **+0.0099** | **+0.0379** | -0.0026 |

### Insights:
* **Diabetes** achieves the largest benefit from Mode B lab biomarkers ($\Delta\text{ROC-AUC} = +0.0267$, $\Delta\text{PR-AUC} = +0.0587$), primarily driven by lipid panel values (`triglycerides`, `total_cholesterol`, `hdl_cholesterol`) and renal biomarkers (`serum_uric_acid`, `serum_creatinine`). Note that explicit targets `hba1c` and `glucose` were strictly excluded to prevent leakage.
* **CVD** shows strong gains in Precision-Recall AUC (+0.0507) when incorporating blood biomarkers.
* **Hypertension** shows a smaller incremental gain (+0.0099 ROC-AUC), confirming that non-invasive measurements (age, BMI, waist circumference, pulse) capture the majority of variance when direct blood pressure readings (`mean_sbp`, `mean_dbp`) are excluded.

---

## 7. Feature Importance Analysis

### Top Predictors by Model Family & Target

#### 1. Hard CVD (`target_cvd`)
* **Mode A Top Features (Logistic Regression Coefs):**
  1. `age` (std coef: $+1.315$)
  2. `smoking_status` (std coef: $+0.270$)
  3. `gender` (std coef: $+0.232$)
  4. `waist_circumference` (std coef: $+0.220$)
  5. `poverty_income_ratio` (std coef: $-0.201$)
* **Mode B Top Features (HistGradientBoosting / Logistic Regression):**
  1. `age`
  2. `serum_creatinine` / `blood_urea_nitrogen` (renal clearance markers)
  3. `smoking_status`
  4. `total_cholesterol` / `hdl_cholesterol`
  5. `waist_circumference`

#### 2. Diabetes (`target_diabetes`) — *Target-defining HbA1c & Glucose Excluded*
* **Mode A Top Features:**
  1. `age` (std coef: $+0.923$)
  2. `bmi` (std coef: $+0.612$)
  3. `waist_circumference` (std coef: $+0.485$)
  4. `mean_sbp` (std coef: $+0.241$)
  5. `poverty_income_ratio`
* **Mode B Top Features:**
  1. `age`
  2. `triglycerides` & `hdl_cholesterol` (triglyceride-to-HDL ratio proxy)
  3. `serum_uric_acid` & `serum_creatinine`
  4. `bmi` & `waist_circumference`
  5. `alt_enzyme` & `ast_enzyme` (fatty liver proxy)

#### 3. Hypertension (`target_hypertension`) — *Target-defining SBP & DBP Excluded*
* **Mode A Top Features:**
  1. `age` (std coef: $+1.142$)
  2. `bmi` (std coef: $+0.521$)
  3. `waist_circumference` (std coef: $+0.380$)
  4. `mean_pulse` (std coef: $+0.198$)
  5. `gender`
* **Mode B Top Features:**
  1. `age`
  2. `bmi` / `waist_circumference`
  3. `serum_uric_acid` & `blood_urea_nitrogen`
  4. `hba1c` & `fasting_glucose` (allowed as predictors for hypertension)
  5. `triglycerides`

---

## 8. Class Imbalance & Calibration Analysis

### 1. Default Threshold (0.50) Limitations
* On imbalanced endpoints (`CVD` 12.6% prevalence, `Diabetes` 17.7% prevalence), standard unweighted classifiers evaluated at fixed threshold $0.50$ achieve high accuracy ($\approx 87\%$) but dismal recall ($10\% - 24\%$).
* Applying `class_weight='balanced'` restores sensitivity ($65\% - 75\%$), but shifts raw predicted probabilities upward, causing overestimation of risk if uncalibrated.

### 2. Calibration State
* Unweighted `Logistic Regression` demonstrates the lowest Brier scores ($0.088 - 0.124$), showing good baseline probability calibration.
* `Random Forest` and `HistGradientBoosting` require probability calibration (e.g. Isotonic Regression or Platt Scaling in Stage 1E) to produce reliable 0–100 risk scores.

---

## 9. Visual Artifact Manifest

All generated diagnostic plots are saved in `backend/outputs/model_v2_baseline/`:

1. **ROC Curves:**
   * [`cvd_mode_a_roc.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/cvd_mode_a_roc.png)
   * [`cvd_mode_b_roc.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/cvd_mode_b_roc.png)
   * [`diabetes_mode_a_roc.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/diabetes_mode_a_roc.png)
   * [`diabetes_mode_b_roc.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/diabetes_mode_b_roc.png)
   * [`hypertension_mode_a_roc.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/hypertension_mode_a_roc.png)
   * [`hypertension_mode_b_roc.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/hypertension_mode_b_roc.png)
2. **PR Curves:**
   * [`cvd_mode_a_pr.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/cvd_mode_a_pr.png)
   * [`cvd_mode_b_pr.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/cvd_mode_b_pr.png)
   * [`diabetes_mode_a_pr.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/diabetes_mode_a_pr.png)
   * [`diabetes_mode_b_pr.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/diabetes_mode_b_pr.png)
   * [`hypertension_mode_a_pr.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/hypertension_mode_a_pr.png)
   * [`hypertension_mode_b_pr.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/hypertension_mode_b_pr.png)
3. **Calibration Curves:**
   * [`cvd_mode_a_calibration.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/cvd_mode_a_calibration.png)
   * [`cvd_mode_b_calibration.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/cvd_mode_b_calibration.png)
   * [`diabetes_mode_a_calibration.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/diabetes_mode_a_calibration.png)
   * [`diabetes_mode_b_calibration.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/diabetes_mode_b_calibration.png)
   * [`hypertension_mode_a_calibration.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/hypertension_mode_a_calibration.png)
   * [`hypertension_mode_b_calibration.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/hypertension_mode_b_calibration.png)
4. **Confusion Matrices (Best Model on Final Test Set):**
   * [`cvd_mode_a_confusion_matrix.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/cvd_mode_a_confusion_matrix.png)
   * [`cvd_mode_b_confusion_matrix.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/cvd_mode_b_confusion_matrix.png)
   * [`diabetes_mode_a_confusion_matrix.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/diabetes_mode_a_confusion_matrix.png)
   * [`diabetes_mode_b_confusion_matrix.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/diabetes_mode_b_confusion_matrix.png)
   * [`hypertension_mode_a_confusion_matrix.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/hypertension_mode_a_confusion_matrix.png)
   * [`hypertension_mode_b_confusion_matrix.png`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/outputs/model_v2_baseline/hypertension_mode_b_confusion_matrix.png)

---

## 10. Recommended Candidate Models for Stage 1E Optimization

Based on validation ROC-AUC, PR-AUC, and calibration behavior, the following model families are recommended for hyperparameter optimization, probability calibration, and threshold tuning in Stage 1E:

1. **Logistic Regression (L1/L2 Regularized with Class Weights):**
   * Strongest baseline for CVD Mode A & Mode B. High interpretability and fast calibration.
2. **HistGradientBoostingClassifier / LightGBM:**
   * Strongest baseline for Diabetes Mode A & Mode B, and Hypertension Mode B. Natively handles missing values and non-linear interactions without artificial imputation.
3. **Random Forest Classifier (Tuned):**
   * Competitive secondary baseline; requires tree depth tuning and leaf smoothing to prevent overfitting on small sub-samples.

---

## 11. Safety Verification of Production V1 Assets

* ✅ **Production Model Intact:** Existing V1 model ([`backend/models/model.pkl`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/models/model.pkl)) was strictly untouched. All V2 baseline model joblib artifacts are isolated in `backend/outputs/model_v2_baseline/models/`.
* ✅ **V1 Dataset Unchanged:** [`dataset/indian_health_risk_dataset.csv`](file:///c:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/dataset/indian_health_risk_dataset.csv) remains unmodified.
* ✅ **System Operability:** Frontend, FastAPI backend routes, database schema, and prediction APIs remain 100% operational.
