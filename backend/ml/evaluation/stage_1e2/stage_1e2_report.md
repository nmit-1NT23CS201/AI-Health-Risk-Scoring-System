
# STAGE 1E-2 — Targeted Model Optimization Report

> **Stage:** 1E-2 — Controlled Hyperparameter Optimization

> **Source Population:** NHANES 2021–2023 Adults (Age ≥ 20)

> **scikit-learn Version:** `1.9.0`

> **XGBoost Version:** `3.4.1`

> **Random Seed:** 42

> **Total Wall-Clock Time:** 250.1s

> **Status:** Complete


## 1. Methodology

Stage 1E-2 performs controlled, cross-validated hyperparameter optimization for the strongest model candidates identified from Stage 1D and Stage 1E-1 baselines. The objective is a fair, CV-driven comparison without any test-set leakage.

**Key design decisions:**

- Hyperparameter search uses `RandomizedSearchCV` with `n_iter=40` and `scoring='roc_auc'`.

- Cross-validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

- The CV pool is the **development set** = Stage 1D training split + validation split (70% + 15% = 85% of participants).

- The **held-out test set** (15%) is **never accessed** during hyperparameter search. It is used only once for the final performance report.

- All preprocessing (imputation, scaling) is fitted **inside** CV fold training sets — no global fit on the full dev set before CV.

- `random_state=42` is used everywhere applicable.


## 2. CV Design & Data Partitioning

| Split | Participants | Used For |

|---|---|---|

| Train (70%) | 5,460–5,464 | Part of CV pool (dev set) |

| Validation (15%) | 1,170–1,171 | Part of CV pool (dev set) |

| **Dev set (85%)** | **6,630–6,635** | **Hyperparameter search via 5-fold stratified CV** |

| Test (15%) | 1,170–1,172 | Final evaluation only — **never touched during tuning** |


## 3. Models & Hyperparameter Search Spaces


### Logistic Regression (CVD Mode A/B, Hypertension Mode A/B)

Implemented as `Pipeline([SimpleImputer(median), StandardScaler, LogisticRegression])`.

Preprocessing is entirely inside the pipeline, ensuring correct CV behavior.

| Parameter | Values Explored |

|---|---|

| `C` | 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0 |

| `solver` | lbfgs, liblinear |

| `class_weight` | None, 'balanced' |

| `max_iter` | 2000 |


### HistGradientBoostingClassifier (Diabetes Mode A/B, Hypertension Mode B)

Native NaN handling — no imputation step required.

| Parameter | Values Explored |

|---|---|

| `learning_rate` | 0.01, 0.05, 0.1, 0.15, 0.2, 0.3 |

| `max_iter` | 100, 200, 300, 500 |

| `max_leaf_nodes` | 15, 31, 63, 127 |

| `min_samples_leaf` | 10, 20, 30, 50 |

| `l2_regularization` | 0.0, 0.01, 0.1, 1.0, 10.0 |

| `class_weight` | None, 'balanced' |


### XGBClassifier (Diabetes Mode B only)

| Parameter | Values Explored |

|---|---|

| `n_estimators` | 100, 200, 300, 500 |

| `max_depth` | 3, 4, 5, 6, 7 |

| `learning_rate` | 0.01, 0.05, 0.1, 0.2, 0.3 |

| `subsample` | 0.6, 0.7, 0.8, 0.9, 1.0 |

| `colsample_bytree` | 0.6, 0.7, 0.8, 0.9, 1.0 |

| `min_child_weight` | 1, 3, 5, 10 |

| `reg_lambda` | 0.1, 1.0, 5.0, 10.0 |

| `scale_pos_weight` | 1.0 (default), computed N_neg/N_pos (balanced) |


## 4. Best Parameters & CV Performance


### cvd_mode_a — logistic_regression

**Mean CV ROC-AUC:** 0.8122 ± 0.0094

**Best hyperparameters found:**

```

  clf__solver: lbfgs

  clf__max_iter: 2000

  clf__class_weight: balanced

  clf__C: 0.01

```


### cvd_mode_b — logistic_regression

**Mean CV ROC-AUC:** 0.8261 ± 0.0103

**Best hyperparameters found:**

```

  clf__solver: lbfgs

  clf__max_iter: 2000

  clf__class_weight: balanced

  clf__C: 0.01

```


### diabetes_mode_a — hist_gradient_boosting

**Mean CV ROC-AUC:** 0.8004 ± 0.0118

**Best hyperparameters found:**

```

  min_samples_leaf: 30

  max_leaf_nodes: 15

  max_iter: 200

  learning_rate: 0.05

  l2_regularization: 10.0

  class_weight: None

```


### diabetes_mode_b — hist_gradient_boosting

**Mean CV ROC-AUC:** 0.8347 ± 0.0139

**Best hyperparameters found:**

```

  min_samples_leaf: 30

  max_leaf_nodes: 15

  max_iter: 200

  learning_rate: 0.05

  l2_regularization: 10.0

  class_weight: None

```


### diabetes_mode_b — xgboost

**Mean CV ROC-AUC:** 0.8370 ± 0.0164

**Best hyperparameters found:**

```

  subsample: 0.6

  scale_pos_weight: 1.0

  reg_lambda: 0.1

  n_estimators: 500

  min_child_weight: 5

  max_depth: 5

  learning_rate: 0.01

  colsample_bytree: 0.6

```


### hypertension_mode_a — logistic_regression

**Mean CV ROC-AUC:** 0.7824 ± 0.0050

**Best hyperparameters found:**

```

  clf__solver: lbfgs

  clf__max_iter: 2000

  clf__class_weight: balanced

  clf__C: 10.0

```


### hypertension_mode_b — hist_gradient_boosting

**Mean CV ROC-AUC:** 0.7945 ± 0.0057

**Best hyperparameters found:**

```

  min_samples_leaf: 50

  max_leaf_nodes: 15

  max_iter: 100

  learning_rate: 0.05

  l2_regularization: 0.1

  class_weight: balanced

```


### hypertension_mode_b — logistic_regression

**Mean CV ROC-AUC:** 0.7916 ± 0.0092

**Best hyperparameters found:**

```

  clf__solver: liblinear

  clf__max_iter: 2000

  clf__class_weight: balanced

  clf__C: 0.01

```


## 5. Final Test-Set Performance (Optimized Models)

*Threshold = 0.50 throughout. Test set was accessed exactly once, after CV was complete.*



| Config | Candidate | Test ROC-AUC | Test PR-AUC | Test Recall | Test Spec | Test Brier |

|---|---|---|---|---|---|---|

| cvd_mode_a | `logistic_regression` | 0.8093 | 0.3699 | 0.7500 | 0.7070 | 0.1751 |

| cvd_mode_b | `logistic_regression` | 0.8220 | 0.4349 | 0.7365 | 0.7354 | 0.1637 |

| diabetes_mode_a | `hist_gradient_boosting` | 0.7892 | 0.4136 | 0.2308 | 0.9553 | 0.1245 |

| diabetes_mode_b | `hist_gradient_boosting` | 0.8120 | 0.4621 | 0.2692 | 0.9543 | 0.1192 |

| diabetes_mode_b | `xgboost` | 0.8153 | 0.4695 | 0.2260 | 0.9688 | 0.1175 |

| hypertension_mode_a | `logistic_regression` | 0.7655 | 0.6529 | 0.7375 | 0.6766 | 0.1969 |

| hypertension_mode_b | `hist_gradient_boosting` | 0.7678 | 0.6754 | 0.7275 | 0.6900 | 0.1952 |

| hypertension_mode_b | `logistic_regression` | 0.7719 | 0.6734 | 0.7595 | 0.6900 | 0.1938 |


## 6. Comparison Against Stage 1E-1 Baseline

Comparing the best Stage 1E-2 optimized model against Stage 1D winner and Stage 1E-1 XGBoost baseline:



| Target | Mode | Stage 1D ROC | 1E-1 XGB ROC | 1E-2 Opt ROC | Delta vs 1D | Delta vs 1E-1 | Recommendation |

|---|---|---|---|---|---|---|---|

| **CVD** | **mode_a** | 0.8125 | 0.7766 | 0.8093 | -0.0032 | +0.0326 | **KEEP STAGE 1D MODEL** |

| **CVD** | **mode_b** | 0.8191 | 0.7824 | 0.8220 | +0.0028 | +0.0396 | **OPTIMIZED MODEL IS BETTER** |

| **DIABETES** | **mode_a** | 0.7810 | 0.7551 | 0.7892 | +0.0082 | +0.0341 | **OPTIMIZED MODEL IS BETTER** |

| **DIABETES** | **mode_b** | 0.8003 | 0.8029 | 0.8153 | +0.0150 | +0.0124 | **OPTIMIZED MODEL IS BETTER** |

| **HYPERTENSION** | **mode_a** | 0.7663 | 0.7150 | 0.7655 | -0.0008 | +0.0505 | **ROUGHLY EQUIVALENT — PREFER SIMPLER** |

| **HYPERTENSION** | **mode_b** | 0.7637 | 0.7427 | 0.7678 | +0.0041 | +0.0251 | **OPTIMIZED MODEL IS BETTER** |


## 7. Winner for Each Target/Mode


### CVD MODE_A

- **Optimized candidate:** `logistic_regression`

- **CV ROC-AUC:** 0.8122 ± 0.0094

- **Test ROC-AUC:** 0.8093 (Stage 1D: 0.8125, Delta: -0.0032)

- **Test PR-AUC:** 0.3699 (Stage 1D: 0.3792, Delta: -0.0093)

- **Recommendation:** **KEEP STAGE 1D MODEL**


### CVD MODE_B

- **Optimized candidate:** `logistic_regression`

- **CV ROC-AUC:** 0.8261 ± 0.0103

- **Test ROC-AUC:** 0.8220 (Stage 1D: 0.8191, Delta: +0.0028)

- **Test PR-AUC:** 0.4349 (Stage 1D: 0.4219, Delta: +0.0129)

- **Recommendation:** **OPTIMIZED MODEL IS BETTER**


### DIABETES MODE_A

- **Optimized candidate:** `hist_gradient_boosting`

- **CV ROC-AUC:** 0.8004 ± 0.0118

- **Test ROC-AUC:** 0.7892 (Stage 1D: 0.7810, Delta: +0.0082)

- **Test PR-AUC:** 0.4136 (Stage 1D: 0.4145, Delta: -0.0009)

- **Recommendation:** **OPTIMIZED MODEL IS BETTER**


### DIABETES MODE_B

- **Optimized candidate:** `xgboost`

- **CV ROC-AUC:** 0.8370 ± 0.0164

- **Test ROC-AUC:** 0.8153 (Stage 1D: 0.8003, Delta: +0.0150)

- **Test PR-AUC:** 0.4695 (Stage 1D: 0.4594, Delta: +0.0101)

- **Recommendation:** **OPTIMIZED MODEL IS BETTER**


### HYPERTENSION MODE_A

- **Optimized candidate:** `logistic_regression`

- **CV ROC-AUC:** 0.7824 ± 0.0050

- **Test ROC-AUC:** 0.7655 (Stage 1D: 0.7663, Delta: -0.0008)

- **Test PR-AUC:** 0.6529 (Stage 1D: 0.6550, Delta: -0.0021)

- **Recommendation:** **ROUGHLY EQUIVALENT — PREFER SIMPLER**


### HYPERTENSION MODE_B

- **Optimized candidate:** `hist_gradient_boosting`

- **CV ROC-AUC:** 0.7945 ± 0.0057

- **Test ROC-AUC:** 0.7678 (Stage 1D: 0.7637, Delta: +0.0041)

- **Test PR-AUC:** 0.6754 (Stage 1D: 0.6748, Delta: +0.0006)

- **Recommendation:** **OPTIMIZED MODEL IS BETTER**


## 8. Warnings & Limitations

1. **Small dataset size:** With ~7,800 participants total (6,600 in dev set), small CV AUC differences (< 0.005) should be interpreted cautiously — they may be within noise.

2. **No probability calibration:** Reported probabilities are raw model outputs. If these are to be used as risk scores, Platt scaling or isotonic regression calibration should be applied in a subsequent stage.

3. **Threshold = 0.50 fixed:** All reported precision/recall/F1/specificity are at threshold 0.50. Optimal thresholds for clinical use may differ significantly.

4. **Test set used once:** The final test set was accessed exactly once after all CV-based model selection was complete. No further iterations or threshold tuning were performed against it.

5. **XGBoost vs HGB:** For Diabetes Mode B, both were optimized and compared; the CV-best was saved as the config winner. Check `stage_1e2_results.json` for both candidate metrics.


## 9. Output Artifact Paths

- `backend/ml/evaluation/stage_1e2/stage_1e2_results.json` — Full experiment JSON

- `backend/ml/evaluation/stage_1e2/stage_1e2_comparison.csv` — Comparison CSV

- `backend/ml/evaluation/stage_1e2/stage_1e2_report.md` — This report

- `backend/ml/evaluation/stage_1e2/models/` — All optimized model `.joblib` files

- `backend/ml/evaluation/stage_1e2/models/{target}_{mode}_v2.pkl` — Named v2 model files
