# Stage 1G-3: V2 SHAP Interpretability Validation & Research Figures

**Project**: AI Health Risk Scoring System  
**Stage**: 1G-3 (Interpretability Validation & Research Figures)  
**Evaluation Environment**: Python 3.12, scikit-learn 1.9.0, XGBoost 3.1.1, SHAP 0.49.1  
**Dataset**: NHANES 2021–2023 Locked 15% Held-Out Test Partitions (Participant-Level Splitting)  
**Status**: **PASS**  
**Date**: October 2026  

---

## 1. Executive Summary

This report establishes the interpretability audit and research-grade explanation layer for the six locked V2 disease-mode pipelines of the **AI Health Risk Scoring System**. Following the successful completion of Stage 1G-1 (Model Integrity Audit) and Stage 1G-2 (Consolidated Performance & Mode A/B Analysis), this stage validates the methodological rigor, feature-space fidelity, mathematical consistency, and scientific communication of SHAP (SHapley Additive exPlanations) across all six production-candidate pipelines.

### Key Conclusions:
1. **Mathematical Fidelity of Explanation Space**: The explainability audit identified that each locked model artifact consists of a base predictive estimator wrapped in a post-hoc Platt/sigmoid calibration object (`CalibratedClassifierCV` via `FrozenEstimator` in scikit-learn 1.9.0). SHAP explainers operate directly on the underlying predictive estimators (log-odds decision margins for XGBoost, HistGradientBoosting, and Logistic Regression; tree vote margins for Random Forest). SHAP values represent exact local additive decompositions in the base predictor margin space $f(x)$ prior to the monotonic sigmoidal transformation $P = \frac{1}{1 + \exp(A \cdot f(x) + B)}$. They are **not** direct linear additive contributions to calibrated probabilities, preserving rigorous theoretical consistency.
2. **Feature Mapping & Integrity**: Preprocessing pipelines (median imputation and standard scaling) operate strictly in a 1-to-1 feature manner without one-hot encoding or dimensionality alterations. Raw feature names in the processed NHANES Parquet files map directly and identically to model input features and SHAP explanation attributes.
3. **Zero V1 Contamination**: Exhaustive scanning across all feature vectors, explanation outputs, configuration files, and figures confirmed zero presence of obsolete V1 prototype features (`SDNN`, `RMSSD`, `HRV`, `resting_heart_rate`, `peak_heart_rate`, or V1 `family_history`). All calculations stem strictly from locked V2 NHANES 2021–2023 cohorts.
4. **Research Figure Generation**: Pre-existing IEEE figure references (`shap_summary_plot.png` and `shap_waterfall_plot.png`) were found to be unavailable in the repository. Genuine, publication-quality replacements have been generated directly from the held-out test partition alongside the locked six-panel global feature importance figure (`fig3_global_shap_feature_importance.png`, `.pdf`, `.svg`).
5. **Stage Assessment**: All verification criteria have been satisfied with zero alterations to the locked models, calibration parameters, decision thresholds, or evaluation datasets. Stage 1G-3 is designated **PASS**.

---

## 2. Scope

The scope of Stage 1G-3 is strictly analytical and evaluative:
- **No Refitting or Retraining**: Locked model artifacts from Stage 1F remain entirely untouched.
- **No Hyperparameter or Preprocessing Alterations**: Imputation methods, scaling, and estimator architectures remain locked.
- **No Threshold Modifications**: Validated operating thresholds ($0.13$, $0.15$, $0.15$, $0.17$, $0.41$, $0.52$) remain intact.
- **Strict Data Partitioning**: All SHAP values are computed exclusively on the 15% participant-level held-out test partitions. The training (70%) and validation (15%) partitions were not evaluated.

---

## 3. Locked Models Used

All six locked model artifacts were loaded and verified from [`backend/ml/evaluation/stage_1f/models/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/models/):

| Pipeline | Target Disease | Mode | Artifact Path | Calibration Wrapper | Base Predictive Estimator |
|---|---|---|---|---|---|
| **CVD Mode A** | Cardiovascular Disease | Non-Invasive (A) | [`cvd_mode_a_calibrated.joblib`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/models/cvd_mode_a_calibrated.joblib) | `CalibratedClassifierCV` (Sigmoid, prefit) | `Pipeline(SimpleImputer -> RandomForestClassifier)` |
| **CVD Mode B** | Cardiovascular Disease | Biomarker-Enhanced (B) | [`cvd_mode_b_calibrated.joblib`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/models/cvd_mode_b_calibrated.joblib) | `CalibratedClassifierCV` (Sigmoid, prefit) | `Pipeline(SimpleImputer -> RandomForestClassifier)` |
| **Diabetes Mode A** | Diabetes Mellitus | Non-Invasive (A) | [`diabetes_mode_a_calibrated.joblib`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/models/diabetes_mode_a_calibrated.joblib) | `CalibratedClassifierCV` (Sigmoid, prefit) | Standalone `HistGradientBoostingClassifier` |
| **Diabetes Mode B** | Diabetes Mellitus | Biomarker-Enhanced (B) | [`diabetes_mode_b_calibrated.joblib`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/models/diabetes_mode_b_calibrated.joblib) | `CalibratedClassifierCV` (Sigmoid, prefit) | Standalone `XGBClassifier` |
| **Hypertension Mode A**| Hypertension | Non-Invasive (A) | [`hypertension_mode_a_calibrated.joblib`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/models/hypertension_mode_a_calibrated.joblib) | `CalibratedClassifierCV` (Sigmoid, prefit) | `Pipeline(SimpleImputer -> StandardScaler -> LogisticRegression)` |
| **Hypertension Mode B**| Hypertension | Biomarker-Enhanced (B) | [`hypertension_mode_b_calibrated.joblib`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1f/models/hypertension_mode_b_calibrated.joblib) | `CalibratedClassifierCV` (Sigmoid, prefit) | `Pipeline(SimpleImputer -> RandomForestClassifier)` |

---

## 4. Data Used for SHAP

All SHAP computations were executed strictly against the 15% participant-level held-out test partitions established during Stage 1C and locked in [`backend/ml/data/processed/nhanes_2021_2023/splits/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/splits/).

| Disease Cohort | Processed Parquet Dataset | Split Index File | Total Participants | Held-Out Test Rows ($N$) | Predictor Features ($P$) | Excluded Survey Metadata |
|---|---|---|---|---|---|---|
| **CVD (Modes A & B)** | [`nhanes_cvd_mode_a.parquet`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/nhanes_cvd_mode_a.parquet)<br>[`nhanes_cvd_mode_b.parquet`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/nhanes_cvd_mode_b.parquet) | [`cvd_splits.csv`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/splits/cvd_splits.csv) | 7,816 | 1,172 | Mode A: 13<br>Mode B: 29 | `SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU`, `target_cvd` |
| **Diabetes (Modes A & B)** | [`nhanes_diabetes_mode_a.parquet`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/nhanes_diabetes_mode_a.parquet)<br>[`nhanes_diabetes_mode_b.parquet`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/nhanes_diabetes_mode_b.parquet) | [`diabetes_splits.csv`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/splits/diabetes_splits.csv) | 7,810 | 1,171 | Mode A: 13<br>Mode B: 27 | `SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU`, `target_diabetes` |
| **Hypertension (Modes A & B)** | [`nhanes_hypertension_mode_a.parquet`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/nhanes_hypertension_mode_a.parquet)<br>[`nhanes_hypertension_mode_b.parquet`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/nhanes_hypertension_mode_b.parquet) | [`hypertension_splits.csv`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/data/processed/nhanes_2021_2023/splits/hypertension_splits.csv) | 7,804 | 1,170 | Mode A: 11<br>Mode B: 27 | `SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU`, `target_hypertension` |

---

## 5. SHAP Explainer Method by Model

Rather than forcing an incompatible single explainer across disparate model architectures, each estimator family utilizes its mathematically designated explainer:

```
                  ┌───────────────────────────────────────────────┐
                  │       Locked Pipeline Model Artifact          │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                      Unwrap FrozenEstimator / CalibratedCV
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  │                                               │
                  ▼                                               ▼
     Tree-Based Ensembles                              Linear Classifiers
 (Random Forest, XGBoost, HistGB)                     (Logistic Regression)
                  │                                               │
                  ▼                                               ▼
          shap.TreeExplainer                              shap.LinearExplainer
  (Exact tree path game theory)                   (Analytical feature-weight attribution
                                                   with N=200 empirical background masker)
```

1. **Random Forest (CVD Mode A, CVD Mode B, Hypertension Mode B)**:
   - **Explainer**: `shap.TreeExplainer`
   - **Mechanism**: Tree traversal algorithm computing exact Shapley values over positive class vote margins.
   - **Background Data**: None required (native conditional expectation over tree structures).
2. **HistGradientBoosting (Diabetes Mode A)**:
   - **Explainer**: `shap.TreeExplainer`
   - **Mechanism**: Tree traversal over ensemble decision trees, operating in raw log-odds margin space.
   - **Background Data**: None required.
3. **XGBoost (Diabetes Mode B)**:
   - **Explainer**: `shap.TreeExplainer`
   - **Mechanism**: Native tree ensemble evaluation directly computing Shapley values in margin space ($f(x) \in (-\infty, \infty)$).
   - **Background Data**: None required.
4. **Logistic Regression (Hypertension Mode A)**:
   - **Explainer**: `shap.LinearExplainer`
   - **Mechanism**: Analytical linear attribution computing $\phi_i = w_i \cdot (x_i - \mathbb{E}[x_i])$.
   - **Background Data**: Deterministic background sample of $N=200$ held-out test observations (random seed = 42).

---

## 6. Output Space / Explanation Target

### Mathematical Clarification of Calibrated Pipelines
The final production inference pipeline evaluates participants through a two-stage process:
$$\mathbf{x} \xrightarrow{\text{Preprocessor}} \mathbf{z} \xrightarrow{\text{Base Classifier}} f(\mathbf{z}) \xrightarrow{\text{Platt Sigmoid}} P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + \exp(A \cdot f(\mathbf{z}) + B)}$$

Because the post-hoc Platt sigmoid wrapper (`CalibratedClassifierCV`) performs a nonlinear 1-dimensional scaling, attempting to treat SHAP values as direct additive components of $P(Y=1 \mid \mathbf{x})$ is mathematically invalid. 

**Explicit Methodological Declaration**:
- For **XGBoost, HistGradientBoosting, and Logistic Regression**, SHAP values explain the underlying **decision margin / log-odds space** $f(\mathbf{z})$:
  $$f(\mathbf{z}) = \mathbb{E}[f(\mathbf{z})] + \sum_{i=1}^{M} \phi_i(\mathbf{z})$$
- For **Random Forest**, SHAP values explain the underlying **uncalibrated ensemble vote proportion**:
  $$\text{Score}(\mathbf{z}) = \mathbb{E}[\text{Score}] + \sum_{i=1}^{M} \phi_i(\mathbf{z})$$
- In all six pipelines, post-hoc calibration is a strictly monotonic transformation ($A < 0$). Therefore, any feature that positively increases $f(\mathbf{z})$ ($\phi_i > 0$) strictly increases the final calibrated probability $P(Y=1 \mid \mathbf{x})$.
- Feature attributions are reported as contributions to the base predictor's output space, **not** as direct percentage point additions to the calibrated probability.

---

## 7. Preprocessing and Feature Mapping

A rigorous audit of the feature transformation pipelines was performed to verify whether any multi-column expansions (such as one-hot encoding) could introduce feature-name divergence:

1. **Preprocessing Architecture**:
   - `cvd_mode_a`: `SimpleImputer(strategy='median')`
   - `cvd_mode_b`: `SimpleImputer(strategy='median')`
   - `diabetes_mode_a`: Native missingness routing within `HistGradientBoostingClassifier`
   - `diabetes_mode_b`: Native missingness routing within `XGBClassifier`
   - `hypertension_mode_a`: `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`
   - `hypertension_mode_b`: `SimpleImputer(strategy='median')`
2. **Column Correspondence**:
   - No categorical variables underwent one-hot encoding in V2 (categorical features like `gender`, `smoking_status`, and `physical_activity_level` were mapped to integer-encoded ordinal representations during Stage 1C).
   - The number of features entering preprocessing exactly equals the number of features entering the base estimators ($P_{\text{in}} = P_{\text{out}}$).
   - Raw Parquet column names $\leftrightarrow$ Preprocessed feature matrix columns $\leftrightarrow$ SHAP explanation vector indices are strictly 1-to-1.

---

## 8. V1 Contamination Audit

An automated audit searched for legacy V1 prototype features across all code files, serialized models, dataset schemas, generated CSV/JSON artifacts, and figures:
- `SDNN` $\rightarrow$ **NOT DETECTED** (0 occurrences)
- `RMSSD` $\rightarrow$ **NOT DETECTED** (0 occurrences)
- `HRV` $\rightarrow$ **NOT DETECTED** (0 occurrences)
- `resting_heart_rate` $\rightarrow$ **NOT DETECTED** (0 occurrences)
- `peak_heart_rate` $\rightarrow$ **NOT DETECTED** (0 occurrences)
- `family_history` (V1 binary flag) $\rightarrow$ **NOT DETECTED** (0 occurrences)

**Audit Finding**: Zero V1 artifact contamination exists in the V2 interpretability pipeline.

---

## 9. Global SHAP Results

Global feature importance was computed on the full 15% held-out test partitions ($N_{\text{test}} \in [1170, 1172]$). Feature rankings are ordered by mean absolute SHAP value ($\frac{1}{N} \sum |\phi_i|$), representing average attribution magnitude on model decision output.

### 9.1 Cardiovascular Disease (CVD)

#### CVD Mode A (Non-Invasive)
- **Estimator**: Random Forest ($N=1,172$)
- **Explanation Space**: Ensemble Positive Vote Margin

| Rank | Feature Name | Mean Absolute SHAP Value | Relative Importance (% of Top 10) |
|:---:|---|:---:|:---:|
| 1 | `age` | 0.06419 | 42.6% |
| 2 | `smoking_status` | 0.01828 | 12.1% |
| 3 | `physical_activity_level` | 0.01461 | 9.7% |
| 4 | `waist_circumference` | 0.01357 | 9.0% |
| 5 | `education_level` | 0.01053 | 7.0% |
| 6 | `mean_sbp` | 0.00937 | 6.2% |
| 7 | `poverty_income_ratio` | 0.00877 | 5.8% |
| 8 | `mean_dbp` | 0.00835 | 5.5% |
| 9 | `gender` | 0.00769 | 5.1% |
| 10 | `sedentary_minutes` | 0.00528 | 3.5% |

#### CVD Mode B (Biomarker-Enhanced)
- **Estimator**: Random Forest ($N=1,172$)
- **Explanation Space**: Ensemble Positive Vote Margin

| Rank | Feature Name | Mean Absolute SHAP Value | Relative Importance (% of Top 10) |
|:---:|---|:---:|:---:|
| 1 | `age` | 0.07436 | 50.7% |
| 2 | `total_cholesterol` | 0.02464 | 16.8% |
| 3 | `hba1c` | 0.00947 | 6.4% |
| 4 | `physical_activity_level` | 0.00858 | 5.8% |
| 5 | `smoking_status` | 0.00773 | 5.3% |
| 6 | `serum_creatinine` | 0.00765 | 5.2% |
| 7 | `rdw` | 0.00677 | 4.6% |
| 8 | `poverty_income_ratio` | 0.00677 | 4.6% |
| 9 | `waist_circumference` | 0.00430 | 2.9% |
| 10 | `education_level` | 0.00429 | 2.9% |

---

### 9.2 Diabetes Mellitus

#### Diabetes Mode A (Non-Invasive)
- **Estimator**: HistGradientBoosting ($N=1,171$)
- **Explanation Space**: Log-Odds Decision Margin

| Rank | Feature Name | Mean Absolute SHAP Value | Relative Importance (% of Top 10) |
|:---:|---|:---:|:---:|
| 1 | `age` | 0.77397 | 29.0% |
| 2 | `waist_circumference` | 0.36944 | 13.9% |
| 3 | `mean_dbp` | 0.29587 | 11.1% |
| 4 | `mean_pulse` | 0.27599 | 10.4% |
| 5 | `bmi` | 0.24743 | 9.3% |
| 6 | `mean_sbp` | 0.22679 | 8.5% |
| 7 | `poverty_income_ratio` | 0.15004 | 5.6% |
| 8 | `education_level` | 0.14817 | 5.6% |
| 9 | `physical_activity_level` | 0.09391 | 3.5% |
| 10 | `sedentary_minutes` | 0.08378 | 3.1% |

#### Diabetes Mode B (Biomarker-Enhanced)
- **Estimator**: XGBoost ($N=1,171$)
- **Explanation Space**: Log-Odds Decision Margin

| Rank | Feature Name | Mean Absolute SHAP Value | Relative Importance (% of Top 10) |
|:---:|---|:---:|:---:|
| 1 | `age` | 0.63306 | 32.6% |
| 2 | `waist_circumference` | 0.24208 | 12.5% |
| 3 | `total_cholesterol` | 0.18920 | 9.7% |
| 4 | `bmi` | 0.18433 | 9.5% |
| 5 | `mean_pulse` | 0.16581 | 8.5% |
| 6 | `hdl_cholesterol` | 0.13980 | 7.2% |
| 7 | `mean_dbp` | 0.13813 | 7.1% |
| 8 | `triglycerides` | 0.12011 | 6.2% |
| 9 | `mean_sbp` | 0.11144 | 5.7% |
| 10 | `poverty_income_ratio` | 0.09923 | 5.1% |

---

### 9.3 Hypertension

#### Hypertension Mode A (Non-Invasive)
- **Estimator**: Logistic Regression ($N=1,170$)
- **Explanation Space**: Linear Decision Margin (Log-Odds)

| Rank | Feature Name | Mean Absolute SHAP Value | Relative Importance (% of Top 10) |
|:---:|---|:---:|:---:|
| 1 | `age` | 0.96613 | 54.0% |
| 2 | `bmi` | 0.20565 | 11.5% |
| 3 | `mean_pulse` | 0.09274 | 5.2% |
| 4 | `sedentary_minutes` | 0.08969 | 5.0% |
| 5 | `waist_circumference` | 0.08916 | 5.0% |
| 6 | `poverty_income_ratio` | 0.08821 | 4.9% |
| 7 | `education_level` | 0.07497 | 4.2% |
| 8 | `gender` | 0.05035 | 2.8% |
| 9 | `smoking_status` | 0.04413 | 2.5% |
| 10 | `physical_activity_level` | 0.03707 | 2.1% |

#### Hypertension Mode B (Biomarker-Enhanced)
- **Estimator**: Random Forest ($N=1,170$)
- **Explanation Space**: Ensemble Positive Vote Margin

| Rank | Feature Name | Mean Absolute SHAP Value | Relative Importance (% of Top 10) |
|:---:|---|:---:|:---:|
| 1 | `age` | 0.17117 | 55.4% |
| 2 | `hba1c` | 0.04020 | 13.0% |
| 3 | `waist_circumference` | 0.02708 | 8.8% |
| 4 | `bmi` | 0.02394 | 7.7% |
| 5 | `serum_uric_acid` | 0.00962 | 3.1% |
| 6 | `poverty_income_ratio` | 0.00942 | 3.0% |
| 7 | `mean_pulse` | 0.00924 | 3.0% |
| 8 | `fasting_glucose` | 0.00771 | 2.5% |
| 9 | `sedentary_minutes` | 0.00743 | 2.4% |
| 10 | `ldl_cholesterol` | 0.00712 | 2.3% |

---

## 10. Mode A vs Mode B Interpretability Analysis

A comparative synthesis between Non-Invasive (Mode A) and Biomarker-Enhanced (Mode B) configurations reveals distinct, disease-specific shifts in feature reliance:

### 10.1 Cardiovascular Disease: Strong Biomarker Reorganization
- In **Mode A**, model outputs depend primarily on `age` (0.0642), lifestyle indicators (`smoking_status` = 0.0183, `physical_activity_level` = 0.0146), and anthropometric proxy `waist_circumference` (0.0136). Blood pressures (`mean_sbp`, `mean_dbp`) contribute moderately (0.0094 and 0.0083).
- In **Mode B**, introducing circulating blood chemistry fundamentally reallocates model attributions. `total_cholesterol` immediately ascends to the #2 most influential predictor (0.0246), exceeding all lifestyle and vital features. Glycemic biomarker `hba1c` captures rank #3 (0.0095), while renal markers `serum_creatinine` (0.0076) and hematologic marker `rdw` (0.0068) enter the top 8. Consequently, the relative influence of `smoking_status` and `waist_circumference` is attenuated as biochemical markers provide additional informative predictive signals in the fitted model.

### 10.2 Diabetes Mellitus: Multi-Biomarker Dispersion
- In **Mode A**, the non-invasive HistGradientBoosting estimator relies heavily on adiposity indicators (`waist_circumference` = 0.3694, `bmi` = 0.2474) and vascular hemodynamics (`mean_dbp` = 0.2959, `mean_pulse` = 0.2760) alongside `age` (0.7740).
- In **Mode B**, target-defining glycemic biomarkers (`fasting_glucose`, `hba1c`) were purposefully excluded during Stage 1C to prevent circular classification. Under XGBoost, lipid profile markers absorb substantial predictive attribution: `total_cholesterol` emerges at rank #3 (0.1892), `hdl_cholesterol` at rank #6 (0.1398), and `triglycerides` at rank #8 (0.1201). Adiposity (`waist_circumference` = 0.2421, `bmi` = 0.1843) remains highly influential, aligning with the empirical co-occurrence of dyslipidemia and central adiposity in individuals identified with elevated glycemic risk.

### 10.3 Hypertension: Glycemic & Uric Acid Integration
- In **Mode A**, where blood pressure measurements are excluded from predictors (because systolic $\ge 130$ or diastolic $\ge 80$ defines the target), the Logistic Regression model attributes over 54% of its decision magnitude to `age` (0.9661) and 11.5% to `bmi` (0.2056).
- In **Mode B**, the Random Forest estimator leverages circulating biomarkers: `hba1c` emerges as the second most influential predictor (0.0402, 13.0% relative importance), followed by renal/inflammatory marker `serum_uric_acid` (0.0096) and `fasting_glucose` (0.0077). This indicates that circulating metabolic and inflammatory biomarkers provide informative secondary features for the fitted classifier in the absence of direct cuff readings.

---

## 11. Individual-Level Explanation

To validate genuine participant-level interpretability on held-out test data, a representative test subject was audited from the Diabetes Mode B cohort:

### Participant Audit: SEQN 130670
- **Cohort Dataset**: `nhanes_diabetes_mode_b.parquet` (Held-Out Test Partition)
- **Target Disease**: Diabetes Mellitus (Mode B, Biomarker-Enhanced)
- **Ground Truth Label ($Y$)**: **1** (Diagnosed Diabetes / Elevated Glycemic Criteria)
- **Base Estimator**: `xgboost.sklearn.XGBClassifier`
- **Explainer**: `shap.TreeExplainer`
- **Decision Space Output $f(\mathbf{x})$**: $+0.0327$
- **Expected Value $\mathbb{E}[f(X)]$**: $-1.5177$
- **Calibrated Probability $P(Y=1 \mid \mathbf{x})$**: **0.5389** ($53.89\%$)
- **Operating Decision Threshold**: **0.17**
- **Predicted Binary Class**: **1** (Correct Positive Screening Decision)

```
Base Value E[f(X)] = -1.518
    ├── +0.5521  age = 70.0 yr
    ├── +0.4097  bmi = 41.5 kg/m²
    ├── +0.3898  waist_circumference = 115.7 cm
    ├── +0.2616  total_cholesterol = 162.0 mg/dL
    ├── +0.2426  hemoglobin = 12.8 g/dL
    ├── +0.1361  rdw = 14.7%
    ├── +0.1037  triglycerides = 124.0 mg/dL
    ├── +0.1003  poverty_income_ratio = 1.61
    ├── -0.1113  blood_urea_nitrogen = 13.0 mg/dL
    └── -0.0799  ast_enzyme = 22.0 U/L
    ───────────────────────────────────────────
Model Margin f(x) = +0.033 ──[Platt Sigmoid]──> Calibrated P = 0.539 (Threshold = 0.17)
```

### Local Attribution Interpretation
1. **Risk-Increasing Factors (Positive $\phi$)**: Advanced age ($70$ years, $\phi = +0.5521$), severe obesity ($\text{BMI} = 41.5\text{ kg/m}^2$, $\phi = +0.4097$), and central adiposity ($\text{waist} = 115.7\text{ cm}$, $\phi = +0.3898$) served as the primary positive drivers elevating model margin $f(x)$ above the population baseline. Biochemical indicators including total cholesterol ($162\text{ mg/dL}$, $\phi = +0.2616$), hemoglobin ($12.8\text{ g/dL}$, $\phi = +0.2426$), and elevated red cell distribution width ($\text{RDW} = 14.7\%$, $\phi = +0.1361$) provided secondary positive attributions.
2. **Risk-Decreasing Factors (Negative $\phi$)**: Unremarkable blood urea nitrogen ($13\text{ mg/dL}$, $\phi = -0.1113$) and normal aspartate aminotransferase ($22\text{ U/L}$, $\phi = -0.0799$) exerted mild downward pressure.
3. **Decision Outcome**: The net positive attribution ($\sum \phi_i = +1.5504$) shifted the raw score from $-1.5177$ to $+0.0327$. When mapped through Platt scaling, the calibrated risk reached $53.89\%$, comfortably surpassing the validated screening threshold ($0.17$) and successfully identifying a true diabetic participant.

---

## 12. Research Interpretation

In accordance with rigorous academic and scientific publication standards:

1. **Non-Causal Interpretability**: SHAP attributions reflect the internal feature dependencies and decision surfaces learned by the fitted estimators on the NHANES 2021–2023 training partition. A high mean absolute SHAP value indicates that a feature strongly influenced model predictions within the evaluated sample; it does **not** prove an etiologic or causal mechanism.
2. **Directional Associations**:
   - In tree models, higher values of `age`, `waist_circumference`, and `bmi` were systematically associated with increased risk scores across all three diseases.
   - In Mode B configurations, elevated `total_cholesterol` and `hba1c` contributed positively to predicted risk, aligning with known epidemiological associations.
   - Lower socioeconomic status (indexed by `poverty_income_ratio`) consistently contributed positively to risk scores across models, reflecting known health disparity gradients in survey populations.
3. **No Unwarranted Clinical Claims**: These explanations elucidate computational model behavior for research auditing. They do not constitute diagnostic justifications, clinical treatment recommendations, or replacements for formal medical evaluations.

---

## 13. Limitations

1. **Monotonicity vs Additivity in Calibrated Outputs**: Because Platt scaling is a nonlinear monotonic mapping applied post-hoc to base estimator outputs, local SHAP attributions are strictly additive in the uncalibrated margin space $f(x)$, rather than directly additive on the final calibrated probability scale. While the rank-order direction of feature influence is preserved, individual feature contributions cannot be directly summed to yield the calibrated percentage.
2. **Correlated Feature Disentanglement**: Variables such as `bmi` and `waist_circumference`, or systolic and diastolic blood pressure, exhibit substantial collinearity. Tree-based SHAP algorithms distribute credit among correlated features based on split frequency and tree traversal order; individual feature importance must therefore be interpreted alongside related covariates.
3. **Background Sampling for Linear Explainer**: `hypertension_mode_a` utilized a background sample of $N=200$ test cases to estimate reference expectations. While empirical checks showed stability across random seeds, minor sampling variations ($\pm 0.002$ in mean SHAP) are inherent to sampling-based linear explainers.

---

## 14. Figure Inventory

All figures are verified, publication-grade, and located in [`backend/ml/evaluation/stage_1g/figures/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1g/figures/) and the workspace root:

| Figure Identifier | Primary File Path | Alternate Formats | Status | Description |
|---|---|---|---|---|
| **Figure 3: Global Feature Importance** | [`fig3_global_shap_feature_importance.png`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/fig3_global_shap_feature_importance.png) | [`.pdf`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/fig3_global_shap_feature_importance.pdf), [`.svg`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/fig3_global_shap_feature_importance.svg) | **PASS** | Six-panel publication figure displaying top-10 mean(\|SHAP value\|) for all six locked pipelines. |
| **Individual Waterfall Explanation** | [`shap_waterfall_plot.png`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/shap_waterfall_plot.png) | Mirror in [`stage_1g/figures/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1g/figures/shap_waterfall_plot.png) | **PASS** | Authentic participant-level waterfall decomposition for SEQN 130670 ($E[f(X)] = -1.52 \rightarrow f(x) = +0.03 \rightarrow P = 0.539$). |
| **Directional Summary Beeswarm** | [`shap_summary_plot.png`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/shap_summary_plot.png) | Mirror in [`stage_1g/figures/`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1g/figures/shap_summary_plot.png) | **PASS** | Beeswarm summary distribution across 300 held-out test participants showing directional feature impacts. |

---

## 15. Reproducibility Information

To reproduce all numerical SHAP values and research figures from scratch using the locked repository environment:

1. **Global SHAP Computation**:
   ```bash
   python shap_compute.py
   ```
   - Input: Locked models in `backend/ml/evaluation/stage_1f/models/`, test partitions in `backend/ml/data/processed/nhanes_2021_2023/splits/`.
   - Output: `shap_global_values.csv`, `shap_global_values.json` (60 verified records).
2. **Figure 3 Generation**:
   ```bash
   python shap_plot_figure3.py
   ```
   - Reads exact values from `shap_global_values.csv`.
   - Generates PNG, PDF, and SVG vectors.
3. **Structured Verification Data**:
   - Stored in [`backend/ml/evaluation/stage_1g/stage_1g_shap_analysis.json`](file:///C:/Users/sunny/Desktop/AI-Health-Risk-Scoring-System/backend/ml/evaluation/stage_1g/stage_1g_shap_analysis.json).

---

## 16. PASS / FAIL Assessment

| Audit Criteria | Verification Check | Result |
|---|---|:---:|
| **Locked Model Verification** | All six final model artifacts loaded and underlying estimator classes identified | **PASS** |
| **Zero Model Modification** | No model was retrained, refit, re-tuned, or re-calibrated | **PASS** |
| **Data Partition Integrity** | All explanations computed strictly on 15% held-out test sets | **PASS** |
| **Explainer Appropriateness** | Mathematical match between model families and SHAP explainer types | **PASS** |
| **Output Space Transparency** | Underlying base decision space vs. monotonic Platt calibration explicitly documented | **PASS** |
| **Feature Mapping Fidelity** | Preprocessing preserves 1-to-1 feature identity (0 one-hot encoding divergence) | **PASS** |
| **V1 Contamination Elimination** | Zero obsolete V1 features detected across datasets, models, code, and figures | **PASS** |
| **Global Attribution Completeness** | Exactly 60 real SHAP values computed across all 6 models | **PASS** |
| **Individual-Level Validation** | Authentic participant SEQN 130670 waterfall explanation generated and verified | **PASS** |
| **Research Figures** | Missing IEEE figures generated with authentic V2 values; publication quality confirmed | **PASS** |
| **Scientific Language Compliance** | Rigorous non-causal terminology adhered to throughout report | **PASS** |

### Final Determination: **STAGE 1G-3 PASS**
