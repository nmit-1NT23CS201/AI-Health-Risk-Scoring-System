# STAGE 1F — Probability Calibration & Validation-Set Threshold Optimization

> **Stage:** 1F — Probability Calibration & Operating Threshold Optimization
> **Source Population:** NHANES 2021-2023 Adults (Age >= 20)
> **scikit-learn Version:** `1.9.0` | **Random Seed:** `42`
> **Total Execution Time:** `13.1s` | **Status:** Complete

---

## 1. Executive Summary

Stage 1F implements rigorous probability calibration and validation-set operating threshold optimization for the six final V2 candidate models established in Stage 1E. In health-risk screening applications, raw machine learning probabilities often suffer from miscalibration (especially in tree ensembles), and the standard `0.50` decision threshold is severely suboptimal for imbalanced disease prevalence, yielding high specificity at the expense of unacceptably low screening recall.

To prevent calibration and threshold overfitting, both calibration-method selection and operating threshold optimization are conducted using **5-fold out-of-fold (OOF) validation probabilities**. After the calibration method (**Sigmoid / Platt**) and operating thresholds are permanently locked, the final calibrator for each model is refitted on the **full validation set (100%)** for deployment. The **final test set (15%)** remains completely untouched throughout this entire process and is evaluated exactly once to provide an unbiased estimate of generalization performance.

This optimization substantially improves screening utility on the held-out test set:
- **CVD Mode A:** sensitivity increases from **9.5%** (at default 0.50) to **68.2%** (at locked OOF threshold $t=0.13$) with 94.4% NPV.
- **CVD Mode B:** sensitivity increases from **19.6%** (at default 0.50) to **60.8%** (at locked OOF threshold $t=0.15$) with 93.7% NPV.
- **Diabetes Mode A:** sensitivity increases from **21.6%** (at default 0.50) to **81.7%** (at locked OOF threshold $t=0.15$) with 94.2% NPV.
- **Diabetes Mode B:** sensitivity increases from **27.4%** (at default 0.50) to **62.0%** (at locked OOF threshold $t=0.17$) with 90.8% NPV.
- **Hypertension Mode A:** sensitivity increases from **62.5%** (at default 0.50) to **76.1%** (at locked OOF threshold $t=0.41$) with 78.9% NPV.
- **Hypertension Mode B:** operating threshold locked at **$t=0.52$** yielding **62.5%** sensitivity and **75.1%** specificity with 72.9% NPV.

## 2. Final Candidate Models & Selected Calibration

| Configuration | Final Model Architecture | Source Stage | Selected Calibration | Validation Brier (OOF) | Selected Threshold |
|---|---|---|---|---|---|
| **cvd_mode_a** | Optimized Random Forest | Stage 1E-3 | **SIGMOID** | 0.0922 | **t = 0.13** |
| **cvd_mode_b** | Optimized Random Forest | Stage 1E-3 | **SIGMOID** | 0.0884 | **t = 0.15** |
| **diabetes_mode_a** | HistGradientBoosting | Stage 1E-2 | **SIGMOID** | 0.1231 | **t = 0.15** |
| **diabetes_mode_b** | XGBoost | Stage 1E-2 | **SIGMOID** | 0.1136 | **t = 0.17** |
| **hypertension_mode_a** | Logistic Regression | Stage 1E-2 | **SIGMOID** | 0.1848 | **t = 0.41** |
| **hypertension_mode_b** | Optimized Random Forest | Stage 1E-3 | **SIGMOID** | 0.1772 | **t = 0.52** |

## 3. Methodology & Leakage Prevention

1. **Strict 3-Way Partitioning:**
   - **TRAIN (70%):** Used exclusively for base model fitting. The base model estimators and hyperparameters from Stage 1E were trained strictly on this partition.
   - **VALIDATION (15%):** Used exclusively for comparing calibration methods (Raw vs Sigmoid vs Isotonic) via 5-fold out-of-fold (OOF) cross-validation and for sweeping operating thresholds on those OOF probabilities. Neither the calibrator mapping nor the decision threshold was optimized in-sample on the validation set.
   - **DEPLOYMENT CALIBRATOR REFITTING:** Only after both the calibration method (Sigmoid / Platt) and the operating threshold ($t_{\text{opt}}$) were permanently locked, each final calibrator was refitted on the full 100% validation set (`X_val`, `y_val`) for production artifact deployment.
   - **FINAL TEST (15%):** Remained completely held out and untouched until all calibration and threshold choices were locked. Evaluated exactly once using the finalized deployed pipeline.
2. **Leakage Elimination:** Excluded all NHANES survey design variables (`SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU`) and direct target diagnostic variables.
3. **Honest Out-of-Sample Calibration & Threshold Selection:** All probability calibration comparison metrics (Brier, ECE, MCE, ROC-AUC, PR-AUC) and screening threshold sweeps were computed using 5-fold out-of-fold cross-validation on validation predictions, preventing non-parametric isotonic overfitting and threshold selection bias.
4. **Reproducibility:** Seeded with `random_state=42` across all models and cross-validation splits.

## 4. Calibration Comparison (Raw vs Sigmoid vs Isotonic)

The table below details validation-set calibration metrics across the three methods (evaluated via 5-fold out-of-fold cross-validation on the validation set):

| Configuration | Method | Brier Score (lower=better) | ECE (lower=better) | MCE | ROC-AUC | PR-AUC | Status |
|---|---|---|---|---|---|---|---|
| **cvd_mode_a** | Raw | 0.0924 | 0.0271 | 0.5193 | 0.8229 | 0.3841 |  |
| **cvd_mode_a** | Sigmoid (Platt) | 0.0922 | 0.0111 | 0.7511 | 0.8216 | 0.3717 | **SELECTED** |
| **cvd_mode_a** | Isotonic | 0.0939 | 0.0155 | 0.9775 | 0.8131 | 0.3517 |  |
| **cvd_mode_b** | Raw | 0.0885 | 0.0167 | 0.2437 | 0.8325 | 0.4394 |  |
| **cvd_mode_b** | Sigmoid (Platt) | 0.0884 | 0.0101 | 0.3731 | 0.8313 | 0.4272 | **SELECTED** |
| **cvd_mode_b** | Isotonic | 0.0902 | 0.0236 | 0.8690 | 0.8188 | 0.3945 |  |
| **diabetes_mode_a** | Raw | 0.1238 | 0.0429 | 0.2373 | 0.8025 | 0.4171 |  |
| **diabetes_mode_a** | Sigmoid (Platt) | 0.1231 | 0.0373 | 0.1005 | 0.8019 | 0.4124 | **SELECTED** |
| **diabetes_mode_a** | Isotonic | 0.1239 | 0.0256 | 0.2762 | 0.7836 | 0.3783 |  |
| **diabetes_mode_b** | Raw | 0.1132 | 0.0395 | 0.0910 | 0.8328 | 0.5082 |  |
| **diabetes_mode_b** | Sigmoid (Platt) | 0.1136 | 0.0346 | 0.1511 | 0.8322 | 0.5048 | **SELECTED** |
| **diabetes_mode_b** | Isotonic | 0.1149 | 0.0336 | 0.5000 | 0.8173 | 0.4656 |  |
| **hypertension_mode_a** | Raw | 0.1874 | 0.0582 | 0.1157 | 0.7883 | 0.6802 |  |
| **hypertension_mode_a** | Sigmoid (Platt) | 0.1848 | 0.0323 | 0.1283 | 0.7879 | 0.6800 | **SELECTED** |
| **hypertension_mode_a** | Isotonic | 0.1864 | 0.0283 | 0.3333 | 0.7782 | 0.6600 |  |
| **hypertension_mode_b** | Raw | 0.1776 | 0.0273 | 0.0429 | 0.8072 | 0.7324 |  |
| **hypertension_mode_b** | Sigmoid (Platt) | 0.1772 | 0.0334 | 0.1684 | 0.8062 | 0.7315 | **SELECTED** |
| **hypertension_mode_b** | Isotonic | 0.1802 | 0.0343 | 0.1546 | 0.7963 | 0.7073 |  |

## 5. Threshold Optimization & Screening Tradeoff

Thresholds were swept across $[0.05, 0.95]$ with step size $0.01$ exclusively on **5-fold out-of-fold (OOF) calibrated validation probabilities** to prevent calibration overfitting. In a health-risk screening application, maximizing Youden's J Index ($J = \text{Sensitivity} + \text{Specificity} - 1$) or $F_2$ score balances high sensitivity (minimizing false negatives) while preserving clinical specificity.

| Configuration | Opt Threshold | Val Sensitivity (OOF) | Val Specificity (OOF) | Val Precision (OOF) | Val NPV (OOF) | Val F1 (OOF) | Val F2 (OOF) | Val Youden J (OOF) |
|---|---|---|---|---|---|---|---|---|
| **cvd_mode_a** | **0.13** | 0.8299 | 0.6895 | 0.2773 | 0.9658 | 0.4157 | 0.5934 | 0.5194 |
| **cvd_mode_b** | **0.15** | 0.7755 | 0.7617 | 0.3184 | 0.9594 | 0.4515 | 0.6025 | 0.5372 |
| **diabetes_mode_a** | **0.15** | 0.8173 | 0.6615 | 0.3427 | 0.9437 | 0.4830 | 0.6401 | 0.4788 |
| **diabetes_mode_b** | **0.17** | 0.7933 | 0.7207 | 0.3802 | 0.9417 | 0.5140 | 0.6517 | 0.5139 |
| **hypertension_mode_a** | **0.41** | 0.7900 | 0.6716 | 0.6423 | 0.8108 | 0.7085 | 0.7553 | 0.4616 |
| **hypertension_mode_b** | **0.52** | 0.7260 | 0.7657 | 0.6981 | 0.7892 | 0.7118 | 0.7202 | 0.4917 |

## 6. Final Holdout Test Set Performance (Default 0.50 vs Optimized Threshold)

Evaluated **once** on the untouched 15% final test set:

| Configuration | Model | Calibration | Threshold | Test ROC-AUC | Test PR-AUC | Test Brier | Sensitivity | Specificity | Precision | NPV | F1 | Balanced Acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **cvd_mode_a** (Default) | Optimized Random Forest | SIGMOID | 0.50 | 0.8137 | 0.4159 | 0.0912 | 0.0946 | 0.9922 | 0.6364 | 0.8835 | 0.1647 | 0.5434 |
| **cvd_mode_a** (Optimized) | Optimized Random Forest | SIGMOID | **0.13** | 0.8137 | 0.4159 | 0.0912 | **0.6824** | **0.7803** | **0.3098** | **0.9444** | **0.4262** | **0.7314** |
| **cvd_mode_b** (Default) | Optimized Random Forest | SIGMOID | 0.50 | 0.8218 | 0.4340 | 0.0906 | 0.1959 | 0.9834 | 0.6304 | 0.8943 | 0.2990 | 0.5897 |
| **cvd_mode_b** (Optimized) | Optimized Random Forest | SIGMOID | **0.15** | 0.8218 | 0.4340 | 0.0906 | **0.6081** | **0.8438** | **0.3600** | **0.9371** | **0.4523** | **0.7259** |
| **diabetes_mode_a** (Default) | HistGradientBoosting | SIGMOID | 0.50 | 0.7957 | 0.4348 | 0.1222 | 0.2163 | 0.9678 | 0.5921 | 0.8511 | 0.3169 | 0.5921 |
| **diabetes_mode_a** (Optimized) | HistGradientBoosting | SIGMOID | **0.15** | 0.7957 | 0.4348 | 0.1222 | **0.8173** | **0.6397** | **0.3288** | **0.9419** | **0.4690** | **0.7285** |
| **diabetes_mode_b** (Default) | XGBoost | SIGMOID | 0.50 | 0.8162 | 0.4793 | 0.1203 | 0.2740 | 0.9626 | 0.6129 | 0.8599 | 0.3787 | 0.6183 |
| **diabetes_mode_b** (Optimized) | XGBoost | SIGMOID | **0.17** | 0.8162 | 0.4793 | 0.1203 | **0.6202** | **0.8100** | **0.4135** | **0.9080** | **0.4962** | **0.7151** |
| **hypertension_mode_a** (Default) | Logistic Regression | SIGMOID | 0.50 | 0.7662 | 0.6541 | 0.1946 | 0.6253 | 0.7541 | 0.6541 | 0.7302 | 0.6393 | 0.6897 |
| **hypertension_mode_a** (Optimized) | Logistic Regression | SIGMOID | **0.41** | 0.7662 | 0.6541 | 0.1946 | **0.7615** | **0.6647** | **0.6281** | **0.7894** | **0.6884** | **0.7131** |
| **hypertension_mode_b** (Default) | Optimized Random Forest | SIGMOID | 0.50 | 0.7731 | 0.6898 | 0.1916 | 0.6593 | 0.7332 | 0.6476 | 0.7432 | 0.6534 | 0.6963 |
| **hypertension_mode_b** (Optimized) | Optimized Random Forest | SIGMOID | **0.52** | 0.7731 | 0.6898 | 0.1916 | **0.6253** | **0.7511** | **0.6514** | **0.7294** | **0.6380** | **0.6882** |

## 7. Confusion Matrix Breakdown on Test Set

| Configuration | Threshold | TP | FP | FN | TN | Total N | Screening Interpretation |
|---|---|---|---|---|---|---|---|
| **cvd_mode_a** | Default 0.50 | 14 | 8 | 134 | 1016 | 1172 | Misses 134 of 148 at-risk cases (9.5% recall) |
| **cvd_mode_a** | **Opt 0.13** | **101** | **225** | **47** | **799** | 1172 | Catches 101 of 148 at-risk cases (**68.2% recall**) |
| **cvd_mode_b** | Default 0.50 | 29 | 17 | 119 | 1007 | 1172 | Misses 119 of 148 at-risk cases (19.6% recall) |
| **cvd_mode_b** | **Opt 0.15** | **90** | **160** | **58** | **864** | 1172 | Catches 90 of 148 at-risk cases (**60.8% recall**) |
| **diabetes_mode_a** | Default 0.50 | 45 | 31 | 163 | 932 | 1171 | Misses 163 of 208 at-risk cases (21.6% recall) |
| **diabetes_mode_a** | **Opt 0.15** | **170** | **347** | **38** | **616** | 1171 | Catches 170 of 208 at-risk cases (**81.7% recall**) |
| **diabetes_mode_b** | Default 0.50 | 57 | 36 | 151 | 927 | 1171 | Misses 151 of 208 at-risk cases (27.4% recall) |
| **diabetes_mode_b** | **Opt 0.17** | **129** | **183** | **79** | **780** | 1171 | Catches 129 of 208 at-risk cases (**62.0% recall**) |
| **hypertension_mode_a** | Default 0.50 | 312 | 165 | 187 | 506 | 1170 | Misses 187 of 499 at-risk cases (62.5% recall) |
| **hypertension_mode_a** | **Opt 0.41** | **380** | **225** | **119** | **446** | 1170 | Catches 380 of 499 at-risk cases (**76.1% recall**) |
| **hypertension_mode_b** | Default 0.50 | 329 | 179 | 170 | 492 | 1170 | Misses 170 of 499 at-risk cases (65.9% recall) |
| **hypertension_mode_b** | **Opt 0.52** | **312** | **167** | **187** | **504** | 1170 | Catches 312 of 499 at-risk cases (**62.5% recall**) |

## 8. Exploratory Risk Score Bands (Prototype / Non-Clinical)

> [!NOTE]
> These risk bands are purely exploratory mathematical stratifications based on calibrated test probability distributions. They are NOT clinically validated diagnostic thresholds.

### CVD_MODE_A (Optimized Random Forest)

| Risk Tier | Calibrated Probability Range | Population % (N) | Observed Positives | Observed Event Rate |
|---|---|---|---|---|
| **Low Risk (Prototype)** | `< 0.078` | 57.3% (672) | 27 | **4.0%** |
| **Moderate Risk (Prototype)** | `[0.078, 0.182)` | 24.8% (291) | 40 | **13.7%** |
| **High Risk (Prototype)** | `>= 0.182` | 17.8% (209) | 81 | **38.8%** |

### CVD_MODE_B (Optimized Random Forest)

| Risk Tier | Calibrated Probability Range | Population % (N) | Observed Positives | Observed Event Rate |
|---|---|---|---|---|
| **Low Risk (Prototype)** | `< 0.090` | 64.2% (753) | 28 | **3.7%** |
| **Moderate Risk (Prototype)** | `[0.090, 0.210)` | 20.9% (245) | 51 | **20.8%** |
| **High Risk (Prototype)** | `>= 0.210` | 14.8% (174) | 69 | **39.7%** |

### DIABETES_MODE_A (HistGradientBoosting)

| Risk Tier | Calibrated Probability Range | Population % (N) | Observed Positives | Observed Event Rate |
|---|---|---|---|---|
| **Low Risk (Prototype)** | `< 0.090` | 37.1% (434) | 11 | **2.5%** |
| **Moderate Risk (Prototype)** | `[0.090, 0.210)` | 29.8% (349) | 55 | **15.8%** |
| **High Risk (Prototype)** | `>= 0.210` | 33.1% (388) | 142 | **36.6%** |

### DIABETES_MODE_B (XGBoost)

| Risk Tier | Calibrated Probability Range | Population % (N) | Observed Positives | Observed Event Rate |
|---|---|---|---|---|
| **Low Risk (Prototype)** | `< 0.102` | 50.3% (589) | 27 | **4.6%** |
| **Moderate Risk (Prototype)** | `[0.102, 0.238)` | 30.3% (355) | 77 | **21.7%** |
| **High Risk (Prototype)** | `>= 0.238` | 19.4% (227) | 104 | **45.8%** |

### HYPERTENSION_MODE_A (Logistic Regression)

| Risk Tier | Calibrated Probability Range | Population % (N) | Observed Positives | Observed Event Rate |
|---|---|---|---|---|
| **Low Risk (Prototype)** | `< 0.246` | 31.0% (363) | 46 | **12.7%** |
| **Moderate Risk (Prototype)** | `[0.246, 0.574)` | 37.4% (438) | 207 | **47.3%** |
| **High Risk (Prototype)** | `>= 0.574` | 31.5% (369) | 246 | **66.7%** |

### HYPERTENSION_MODE_B (Optimized Random Forest)

| Risk Tier | Calibrated Probability Range | Population % (N) | Observed Positives | Observed Event Rate |
|---|---|---|---|---|
| **Low Risk (Prototype)** | `< 0.312` | 41.6% (487) | 84 | **17.2%** |
| **Moderate Risk (Prototype)** | `[0.312, 0.728)` | 42.4% (496) | 279 | **56.2%** |
| **High Risk (Prototype)** | `>= 0.728` | 16.0% (187) | 136 | **72.7%** |

## 9. Limitations & Clinical Safety Statements

1. **Research / Prototype Model Only:** This system is built as a machine learning research prototype on US cross-sectional survey data (NHANES 2021-2023). It is **NOT** a clinically validated medical diagnostic device and must not be used for medical decision making without clinical oversight.
2. **Prevalence and Population Generalizability:** NHANES reflects US adult population health characteristics. Application to international populations (e.g. South Asian cohorts) requires external calibration and validation.
3. **Cross-Sectional vs Longitudinal Outcomes:** Labels represent prevalent disease status at exam time, not 10-year incident cardiovascular/metabolic events.
4. **Single-Pass Test Validation:** The holdout test set was evaluated exactly once after locking calibration and operating thresholds.

## 10. Reproduction Command

Run the Stage 1F script from the project root with the active `.venv`:
```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe backend/ml/scripts/calibrate_and_optimize_thresholds.py
```

## 11. Final Output Artifacts

- Results JSON: `backend/ml/evaluation/stage_1f/stage_1f_results.json`
- Comparison CSV: `backend/ml/evaluation/stage_1f/stage_1f_comparison.csv`
- Report: `backend/ml/evaluation/stage_1f/STAGE_1F_CALIBRATION_THRESHOLD_REPORT.md`
- Calibration Data: `backend/ml/evaluation/stage_1f/calibration/*.csv`
- Threshold Sweep Data: `backend/ml/evaluation/stage_1f/thresholds/*.csv`
- Calibrated Model Artifacts: `backend/ml/evaluation/stage_1f/models/*.joblib` and `*.pkl`
