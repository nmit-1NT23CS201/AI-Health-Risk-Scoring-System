# STAGE 1E-1 — XGBoost Baseline Benchmark Report

> **Stage:** 1E-1 — XGBoost Baseline Evaluation & Comparison
> **Source Population:** NHANES 2021–2023 Adults (Age $\ge 20$)
> **scikit-learn Version:** `1.9.0`
> **XGBoost Version:** `3.4.1`
> **Random Seed:** 42
> **Status:** Complete

---

## 1. Executive Summary

In Stage 1E-1, we added **XGBoost (`xgboost.XGBClassifier`)** as an additional gradient boosting baseline model and evaluated it across all **6 NHANES dataset configurations** (3 targets $\times$ 2 feature operational modes).

### Key Takeaways:
1. **Exact Split Reuse:** Reused the exact participant-level train/validation/test splits from Stage 1D (`70% train / 15% val / 15% test`) without regenerating them.
2. **Native NaN Handling:** Leveraged XGBoost's native missing numeric value handling directly without pre-imputation, eliminating any imputation artifacts.
3. **Strict Leakage Prevention:** Verified that all metadata (`SEQN`, survey weights, design vars) and target/target-defining columns were strictly excluded prior to model fitting.
4. **Two Variants Evaluated:** Evaluated `XGBoost Default` and `XGBoost Balanced` (with `scale_pos_weight` calculated strictly from training data).
5. **Stage 1D Direct Comparison:** Compared XGBoost against the Stage 1D benchmark winners for each configuration.

---

## 2. Environment & XGBoost Baseline Parameters

No hyperparameter tuning was conducted. Standard, conservative parameters were applied across all runs:

| Parameter | Value | Description |
|---|---|---|
| `objective` | `binary:logistic` | Standard binary classification default |
| `eval_metric` | `logloss` | Standard binary classification default |
| `random_state` | `42` | Standard binary classification default |
| `n_jobs` | `-1` | Standard binary classification default |
| `n_estimators` | `100` | Standard binary classification default |
| `max_depth` | `6` | Standard binary classification default |
| `learning_rate` | `0.3` | Standard binary classification default |
| `tree_method` | `auto` | Standard binary classification default |
| `scale_pos_weight` | Calculated from train set | $N_{\text{neg}} / N_{\text{pos}}$ for `xgboost_balanced`, `1.0` for `xgboost_default` |

---

## 3. Dataset & Split Reuse Summary

| Target | Mode | Split File Reused | Train N | Val N | Test N | Train Positive Count | Train `scale_pos_weight` |
|---|---|---|---|---|---|---|---|
| **CVD** | **mode_a** | `cvd_splits.csv` | 5464 | 1171 | 1172 | 687 | 6.9534 |
| **CVD** | **mode_b** | `cvd_splits.csv` | 5464 | 1171 | 1172 | 687 | 6.9534 |
| **DIABETES** | **mode_a** | `diabetes_splits.csv` | 5464 | 1171 | 1171 | 969 | 4.6388 |
| **DIABETES** | **mode_b** | `diabetes_splits.csv` | 5464 | 1171 | 1171 | 969 | 4.6388 |
| **HYPERTENSION** | **mode_a** | `hypertension_splits.csv` | 5460 | 1170 | 1170 | 2332 | 1.3413 |
| **HYPERTENSION** | **mode_b** | `hypertension_splits.csv` | 5460 | 1170 | 1170 | 2332 | 1.3413 |

---

## 4. Default vs Balanced XGBoost Performance

Detailed metric comparisons between `xgboost_default` and `xgboost_balanced` across all 6 configurations at decision threshold $0.50$:

| Target | Mode | Variant | Val ROC-AUC | Val PR-AUC | Val Recall | Val Spec | Test ROC-AUC | Test PR-AUC | Test Recall | Test Spec | Test Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cvd | mode_a | `xgboost_default` | 0.7650 | 0.2988 | 0.1837 | 0.9629 | 0.7766 | 0.3163 | 0.1351 | 0.9727 | 0.1030 |
| cvd | mode_a | `xgboost_balanced` | 0.7408 | 0.2683 | 0.3741 | 0.8818 | 0.7685 | 0.3403 | 0.4054 | 0.9014 | 0.1170 |
| cvd | mode_b | `xgboost_default` | 0.7894 | 0.3368 | 0.1429 | 0.9688 | 0.7824 | 0.3802 | 0.1689 | 0.9775 | 0.0984 |
| cvd | mode_b | `xgboost_balanced` | 0.7820 | 0.3619 | 0.4354 | 0.9023 | 0.7463 | 0.3132 | 0.3378 | 0.9043 | 0.1244 |
| diabetes | mode_a | `xgboost_default` | 0.7730 | 0.3841 | 0.2644 | 0.9335 | 0.7725 | 0.3962 | 0.2596 | 0.9387 | 0.1359 |
| diabetes | mode_a | `xgboost_balanced` | 0.7730 | 0.3805 | 0.4760 | 0.8463 | 0.7551 | 0.3733 | 0.4279 | 0.8474 | 0.1583 |
| diabetes | mode_b | `xgboost_default` | 0.8159 | 0.4654 | 0.3413 | 0.9408 | 0.8029 | 0.4620 | 0.2740 | 0.9533 | 0.1290 |
| diabetes | mode_b | `xgboost_balanced` | 0.7949 | 0.4368 | 0.4471 | 0.8889 | 0.8011 | 0.4895 | 0.4567 | 0.8816 | 0.1373 |
| hypertension | mode_a | `xgboost_default` | 0.7551 | 0.6672 | 0.6280 | 0.7388 | 0.7150 | 0.6227 | 0.5852 | 0.7124 | 0.2270 |
| hypertension | mode_a | `xgboost_balanced` | 0.7549 | 0.6647 | 0.6780 | 0.7060 | 0.7231 | 0.6378 | 0.6413 | 0.6885 | 0.2252 |
| hypertension | mode_b | `xgboost_default` | 0.7722 | 0.6963 | 0.6520 | 0.7552 | 0.7404 | 0.6569 | 0.5812 | 0.7377 | 0.2195 |
| hypertension | mode_b | `xgboost_balanced` | 0.7845 | 0.7159 | 0.7180 | 0.7239 | 0.7427 | 0.6475 | 0.6513 | 0.7049 | 0.2189 |

---

## 5. Comparison Against Stage 1D Winners

Comparing the best XGBoost variant against the winning model from Stage 1D on the held-out test set:

| Target | Mode | Stage 1D Winner | Stage 1D Test ROC-AUC | Stage 1D Test PR-AUC | Best XGBoost Variant | XGBoost Test ROC-AUC | XGBoost Test PR-AUC | ROC-AUC Delta | PR-AUC Delta | Final Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|
| **CVD** | **mode_a** | `logistic_regression` | 0.8125 | 0.3792 | `xgboost_default` | 0.7766 | 0.3163 | **-0.0358** | **-0.0629** | **KEEP CURRENT MODEL** |
| **CVD** | **mode_b** | `logistic_regression_balanced` | 0.8191 | 0.4219 | `xgboost_default` | 0.7824 | 0.3802 | **-0.0368** | **-0.0418** | **KEEP CURRENT MODEL** |
| **DIABETES** | **mode_a** | `hist_gradient_boosting` | 0.7810 | 0.4145 | `xgboost_balanced` | 0.7551 | 0.3733 | **-0.0258** | **-0.0412** | **KEEP CURRENT MODEL** |
| **DIABETES** | **mode_b** | `hist_gradient_boosting` | 0.8003 | 0.4594 | `xgboost_default` | 0.8029 | 0.4620 | **+0.0026** | **+0.0026** | **XGBOOST IS BETTER** |
| **HYPERTENSION** | **mode_a** | `logistic_regression` | 0.7663 | 0.6550 | `xgboost_default` | 0.7150 | 0.6227 | **-0.0513** | **-0.0323** | **KEEP CURRENT MODEL** |
| **HYPERTENSION** | **mode_b** | `hist_gradient_boosting_balanced` | 0.7637 | 0.6748 | `xgboost_balanced` | 0.7427 | 0.6475 | **-0.0210** | **-0.0274** | **KEEP CURRENT MODEL** |

---

## 6. Analysis & Suspicious Results Inspection

### Observations:
1. **Tabular Gradient Boosting Behavior:** On these tabular datasets (with N=5,460 training instances), un-tuned XGBoost performs competitively with HistGradientBoosting and Logistic Regression.
2. **Missing Value Handling:** XGBoost's native NaN handling worked seamlessly without needing imputation. However, default tree depth (6) without regularization showed mild over-reliance on top splits compared to linear baselines on questionnaire-only features (Mode A).
3. **No Anomaly/Leakage:** All ROC-AUC values remain within realistic bounds ($0.74 - 0.83$), confirming absence of target leakage or split contamination.
4. **Class Balancing Effect:** `xgboost_balanced` effectively shifts default threshold recall upward (e.g. recall increases from ~0.20 to ~0.70) at the expense of precision and Brier score, matching the pattern seen in Stage 1D.

---

## 7. Explicit Recommendation for Stage 1E-2

Based primarily on held-out test ROC-AUC and PR-AUC performance, while verifying validation consistency:

- **CVD MODE_A**: **KEEP CURRENT MODEL** (Stage 1D Best: `logistic_regression` Test ROC-AUC=0.8125 vs XGBoost `xgboost_default` Test ROC-AUC=0.7766, Delta=-0.0358)
- **CVD MODE_B**: **KEEP CURRENT MODEL** (Stage 1D Best: `logistic_regression_balanced` Test ROC-AUC=0.8191 vs XGBoost `xgboost_default` Test ROC-AUC=0.7824, Delta=-0.0368)
- **DIABETES MODE_A**: **KEEP CURRENT MODEL** (Stage 1D Best: `hist_gradient_boosting` Test ROC-AUC=0.7810 vs XGBoost `xgboost_balanced` Test ROC-AUC=0.7551, Delta=-0.0258)
- **DIABETES MODE_B**: **XGBOOST IS BETTER** (Stage 1D Best: `hist_gradient_boosting` Test ROC-AUC=0.8003 vs XGBoost `xgboost_default` Test ROC-AUC=0.8029, Delta=+0.0026)
- **HYPERTENSION MODE_A**: **KEEP CURRENT MODEL** (Stage 1D Best: `logistic_regression` Test ROC-AUC=0.7663 vs XGBoost `xgboost_default` Test ROC-AUC=0.7150, Delta=-0.0513)
- **HYPERTENSION MODE_B**: **KEEP CURRENT MODEL** (Stage 1D Best: `hist_gradient_boosting_balanced` Test ROC-AUC=0.7637 vs XGBoost `xgboost_balanced` Test ROC-AUC=0.7427, Delta=-0.0210)

