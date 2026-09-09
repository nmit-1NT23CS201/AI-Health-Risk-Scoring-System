
# STAGE 1E-3 -- Random Forest Optimization & Final Model Comparison

> **Stage:** 1E-3 -- Controlled Random Forest Hyperparameter Optimization

> **Source Population:** NHANES 2021-2023 Adults (Age >= 20)

> **scikit-learn Version:** `1.9.0`

> **Random Seed:** 42

> **CV Folds:** 5 (StratifiedKFold, shuffle=True)

> **RandomizedSearchCV n_iter:** 40

> **Total Wall-Clock Time:** 798.9s

> **Status:** Complete


## 1. Methodology

Stage 1E-3 performs a focused Random Forest hyperparameter optimization using **the identical train/test methodology as Stage 1E-2**, allowing direct performance comparison.



**Data partitioning (same as Stage 1E-2):**

- **Dev set** = Stage 1D `train` + `validation` SEQNs (70% + 15% = 85%)

- **Test set** (15%) is **never accessed during hyperparameter search**

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the dev set

- `RandomizedSearchCV(n_iter=40, scoring='roc_auc')` selects the best parameter set

- Best estimator (refitted on full dev set by scikit-learn) is evaluated **once** on the held-out test set



**Preprocessing pipeline:**

- `SimpleImputer(strategy='median')` + `RandomForestClassifier`

- Imputer fitted inside each CV fold (no data leakage across folds)

- `class_weight` treated as a hyperparameter (`None`, `'balanced'`, `'balanced_subsample'`)


## 2. Hyperparameter Search Space

| Parameter | Values Explored |

|---|---|

| `n_estimators` | 300, 500, 800 |

| `max_depth` | None, 8, 12, 16, 20 |

| `min_samples_split` | 2, 5, 10 |

| `min_samples_leaf` | 1, 2, 5, 10 |

| `max_features` | 'sqrt', 'log2', 0.5, 0.8 |

| `class_weight` | None, 'balanced', 'balanced_subsample' |



Search is conducted as `RandomizedSearchCV(n_iter=40)` across all three class-weight variants, with `random_state=42` ensuring full reproducibility.


## 3. Per-Configuration Results


### cvd_mode_a

**Dev set:** 6635 participants (834 positive)  |  **Test set:** 1172  |  **Features:** 13



**CV ROC-AUC (5-fold):** 0.8110 +/- 0.0103



**Best hyperparameters:**

```

  clf__n_estimators: 500

  clf__min_samples_split: 5

  clf__min_samples_leaf: 1

  clf__max_features: log2

  clf__max_depth: 8

  clf__class_weight: None

```



**Test-set metrics (threshold = 0.50, accessed once):**



| Metric | Value |

|---|---|

| ROC-AUC | **0.8140** |

| PR-AUC | 0.4086 |

| Accuracy | 0.8737 |

| Precision | 0.5000 |

| Recall | 0.0068 |

| F1 | 0.0133 |

| Specificity | 0.9990 |

| Brier Score | 0.0930 |

| TP / FP / FN / TN | 1 / 1 / 147 / 1023 |




### cvd_mode_b

**Dev set:** 6635 participants (834 positive)  |  **Test set:** 1172  |  **Features:** 29



**CV ROC-AUC (5-fold):** 0.8306 +/- 0.0073



**Best hyperparameters:**

```

  clf__n_estimators: 800

  clf__min_samples_split: 10

  clf__min_samples_leaf: 10

  clf__max_features: 0.5

  clf__max_depth: 8

  clf__class_weight: None

```



**Test-set metrics (threshold = 0.50, accessed once):**



| Metric | Value |

|---|---|

| ROC-AUC | **0.8244** |

| PR-AUC | 0.4304 |

| Accuracy | 0.8814 |

| Precision | 0.7368 |

| Recall | 0.0946 |

| F1 | 0.1677 |

| Specificity | 0.9951 |

| Brier Score | 0.0900 |

| TP / FP / FN / TN | 14 / 5 / 134 / 1019 |




### diabetes_mode_a

**Dev set:** 6635 participants (1177 positive)  |  **Test set:** 1171  |  **Features:** 13



**CV ROC-AUC (5-fold):** 0.7985 +/- 0.0165



**Best hyperparameters:**

```

  clf__n_estimators: 800

  clf__min_samples_split: 10

  clf__min_samples_leaf: 5

  clf__max_features: log2

  clf__max_depth: 12

  clf__class_weight: None

```



**Test-set metrics (threshold = 0.50, accessed once):**



| Metric | Value |

|---|---|

| ROC-AUC | **0.7791** |

| PR-AUC | 0.4053 |

| Accuracy | 0.8207 |

| Precision | 0.4750 |

| Recall | 0.0913 |

| F1 | 0.1532 |

| Specificity | 0.9782 |

| Brier Score | 0.1248 |

| TP / FP / FN / TN | 19 / 21 / 189 / 942 |




### diabetes_mode_b

**Dev set:** 6635 participants (1177 positive)  |  **Test set:** 1171  |  **Features:** 27



**CV ROC-AUC (5-fold):** 0.8291 +/- 0.0188



**Best hyperparameters:**

```

  clf__n_estimators: 800

  clf__min_samples_split: 10

  clf__min_samples_leaf: 5

  clf__max_features: log2

  clf__max_depth: 12

  clf__class_weight: None

```



**Test-set metrics (threshold = 0.50, accessed once):**



| Metric | Value |

|---|---|

| ROC-AUC | **0.8020** |

| PR-AUC | 0.4510 |

| Accuracy | 0.8309 |

| Precision | 0.6250 |

| Recall | 0.1202 |

| F1 | 0.2016 |

| Specificity | 0.9844 |

| Brier Score | 0.1204 |

| TP / FP / FN / TN | 25 / 15 / 183 / 948 |




### hypertension_mode_a

**Dev set:** 6630 participants (2832 positive)  |  **Test set:** 1170  |  **Features:** 11



**CV ROC-AUC (5-fold):** 0.7850 +/- 0.0040



**Best hyperparameters:**

```

  clf__n_estimators: 800

  clf__min_samples_split: 10

  clf__min_samples_leaf: 10

  clf__max_features: 0.5

  clf__max_depth: 8

  clf__class_weight: None

```



**Test-set metrics (threshold = 0.50, accessed once):**



| Metric | Value |

|---|---|

| ROC-AUC | **0.7611** |

| PR-AUC | 0.6659 |

| Accuracy | 0.6915 |

| Precision | 0.6312 |

| Recall | 0.6653 |

| F1 | 0.6478 |

| Specificity | 0.7109 |

| Brier Score | 0.1951 |

| TP / FP / FN / TN | 332 / 194 / 167 / 477 |




### hypertension_mode_b

**Dev set:** 6630 participants (2832 positive)  |  **Test set:** 1170  |  **Features:** 27



**CV ROC-AUC (5-fold):** 0.7953 +/- 0.0085



**Best hyperparameters:**

```

  clf__n_estimators: 800

  clf__min_samples_split: 10

  clf__min_samples_leaf: 10

  clf__max_features: 0.5

  clf__max_depth: 8

  clf__class_weight: None

```



**Test-set metrics (threshold = 0.50, accessed once):**



| Metric | Value |

|---|---|

| ROC-AUC | **0.7717** |

| PR-AUC | 0.6814 |

| Accuracy | 0.7026 |

| Precision | 0.6477 |

| Recall | 0.6633 |

| F1 | 0.6554 |

| Specificity | 0.7317 |

| Brier Score | 0.1904 |

| TP / FP / FN / TN | 331 / 180 / 168 / 491 |




## 4. Final Comparison: V1 RF vs Optimized RF vs Stage 1E-2 Winner

*All ROC-AUC values are from the same held-out test set (15% of participants). Lower test Brier = better calibration at threshold 0.50.*



| Configuration | V1 RF (best) | Opt RF (1E-3) | Stage 1E-2 Winner | RF vs V1 | RF vs 1E-2 | Recommended |

|---|---|---|---|---|---|---|

| **cvd_mode_a** | 0.7821 (`random_forest`) | **0.8140** | 0.8093 (`logistic_regression`) | +0.0318 | +0.0047 | **OPTIMIZED RF IS BEST** |

| **cvd_mode_b** | 0.7996 (`random_forest`) | **0.8244** | 0.8220 (`logistic_regression`) | +0.0248 | +0.0024 | **OPTIMIZED RF IS BEST** |

| **diabetes_mode_a** | 0.7673 (`random_forest_balanced`) | **0.7791** | 0.7892 (`hist_gradient_boosting`) | +0.0118 | -0.0101 | **KEEP STAGE 1E-2 (hist_gradient_boosting)** |

| **diabetes_mode_b** | 0.7801 (`random_forest`) | **0.8020** | 0.8153 (`xgboost`) | +0.0219 | -0.0133 | **KEEP STAGE 1E-2 (xgboost)** |

| **hypertension_mode_a** | 0.7437 (`random_forest`) | **0.7611** | 0.7655 (`logistic_regression`) | +0.0174 | -0.0044 | **KEEP STAGE 1E-2 (logistic_regression)** |

| **hypertension_mode_b** | 0.7592 (`random_forest`) | **0.7717** | 0.7678 (`hist_gradient_boosting`) | +0.0125 | +0.0039 | **OPTIMIZED RF IS BEST** |


## 5. Per-Configuration Recommendation


### CVD MODE_A

- **V1 RF best test ROC-AUC:** 0.7821 (`random_forest`)

- **Optimized RF test ROC-AUC:** 0.8140 (CV: 0.8110 +/- 0.0103)

- **Stage 1E-2 winner test ROC-AUC:** 0.8093 (`logistic_regression`)

- **Best RF params:** n_estimators=500, max_depth=8, max_features=log2, class_weight=None

- **Recommendation:** **OPTIMIZED RF IS BEST**




### CVD MODE_B

- **V1 RF best test ROC-AUC:** 0.7996 (`random_forest`)

- **Optimized RF test ROC-AUC:** 0.8244 (CV: 0.8306 +/- 0.0073)

- **Stage 1E-2 winner test ROC-AUC:** 0.8220 (`logistic_regression`)

- **Best RF params:** n_estimators=800, max_depth=8, max_features=0.5, class_weight=None

- **Recommendation:** **OPTIMIZED RF IS BEST**




### DIABETES MODE_A

- **V1 RF best test ROC-AUC:** 0.7673 (`random_forest_balanced`)

- **Optimized RF test ROC-AUC:** 0.7791 (CV: 0.7985 +/- 0.0165)

- **Stage 1E-2 winner test ROC-AUC:** 0.7892 (`hist_gradient_boosting`)

- **Best RF params:** n_estimators=800, max_depth=12, max_features=log2, class_weight=None

- **Recommendation:** **KEEP STAGE 1E-2 (hist_gradient_boosting)**




### DIABETES MODE_B

- **V1 RF best test ROC-AUC:** 0.7801 (`random_forest`)

- **Optimized RF test ROC-AUC:** 0.8020 (CV: 0.8291 +/- 0.0188)

- **Stage 1E-2 winner test ROC-AUC:** 0.8153 (`xgboost`)

- **Best RF params:** n_estimators=800, max_depth=12, max_features=log2, class_weight=None

- **Recommendation:** **KEEP STAGE 1E-2 (xgboost)**




### HYPERTENSION MODE_A

- **V1 RF best test ROC-AUC:** 0.7437 (`random_forest`)

- **Optimized RF test ROC-AUC:** 0.7611 (CV: 0.7850 +/- 0.0040)

- **Stage 1E-2 winner test ROC-AUC:** 0.7655 (`logistic_regression`)

- **Best RF params:** n_estimators=800, max_depth=8, max_features=0.5, class_weight=None

- **Recommendation:** **KEEP STAGE 1E-2 (logistic_regression)**




### HYPERTENSION MODE_B

- **V1 RF best test ROC-AUC:** 0.7592 (`random_forest`)

- **Optimized RF test ROC-AUC:** 0.7717 (CV: 0.7953 +/- 0.0085)

- **Stage 1E-2 winner test ROC-AUC:** 0.7678 (`hist_gradient_boosting`)

- **Best RF params:** n_estimators=800, max_depth=8, max_features=0.5, class_weight=None

- **Recommendation:** **OPTIMIZED RF IS BEST**




## 6. Warnings & Limitations

1. **Random Forest and tree depth:** `max_depth=None` allows fully-grown trees; verify that it does not indicate overfitting by comparing CV vs test ROC-AUC gaps.

2. **No probability calibration:** RF probabilities from `predict_proba` are not calibrated. Platt scaling or isotonic regression may be beneficial before clinical use.

3. **Threshold fixed at 0.50:** All recall/precision/F1/specificity are at 0.50. Clinical deployments may require threshold adjustment.

4. **Test set accessed once:** The final test set was touched exactly once per config, after all CV-based model selection was complete. No further iterations were performed.

5. **Small dataset:** ~7,800 NHANES participants is modest for RF. Differences less than ~0.003 ROC-AUC points should not be treated as meaningful improvements.


## 7. Reproduction Command

Run from the project root with the `.venv` activated:

```powershell

$env:PYTHONIOENCODING="utf-8"

.venv\Scripts\python.exe backend/ml/scripts/optimize_random_forest.py

```


## 8. Output Artifact Paths

- `backend/ml/evaluation/stage_1e3/random_forest_optimization_results.json`

- `backend/ml/evaluation/stage_1e3/random_forest_comparison.csv`

- `backend/ml/evaluation/stage_1e3/RANDOM_FOREST_STAGE_1E3_REPORT.md`

- `backend/ml/evaluation/stage_1e3/models/{config}_rf_v2_optimized.joblib`

- `backend/ml/evaluation/stage_1e3/models/{config}_rf_v2.pkl`
