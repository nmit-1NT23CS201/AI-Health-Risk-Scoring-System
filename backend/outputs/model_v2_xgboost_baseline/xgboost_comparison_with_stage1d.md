# XGBoost Baseline vs Stage 1D Winners Comparison Report

> Stage 1E-1 — Comparison of XGBoost Baselines against Stage 1D Benchmark Winners

## Held-Out Test Set Performance Comparison

| Target | Mode | Stage 1D Best Model | Stage 1D Test ROC-AUC | Stage 1D Test PR-AUC | Best XGBoost Variant | XGBoost Test ROC-AUC | XGBoost Test PR-AUC | ROC-AUC Delta | PR-AUC Delta | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|
| cvd | mode_a | `logistic_regression` | 0.8125 | 0.3792 | `xgboost_default` | 0.7766 | 0.3163 | -0.0358 | -0.0629 | **KEEP CURRENT MODEL** |
| cvd | mode_b | `logistic_regression_balanced` | 0.8191 | 0.4219 | `xgboost_default` | 0.7824 | 0.3802 | -0.0368 | -0.0418 | **KEEP CURRENT MODEL** |
| diabetes | mode_a | `hist_gradient_boosting` | 0.7810 | 0.4145 | `xgboost_balanced` | 0.7551 | 0.3733 | -0.0258 | -0.0412 | **KEEP CURRENT MODEL** |
| diabetes | mode_b | `hist_gradient_boosting` | 0.8003 | 0.4594 | `xgboost_default` | 0.8029 | 0.4620 | +0.0026 | +0.0026 | **XGBOOST IS BETTER** |
| hypertension | mode_a | `logistic_regression` | 0.7663 | 0.6550 | `xgboost_default` | 0.7150 | 0.6227 | -0.0513 | -0.0323 | **KEEP CURRENT MODEL** |
| hypertension | mode_b | `hist_gradient_boosting_balanced` | 0.7637 | 0.6748 | `xgboost_balanced` | 0.7427 | 0.6475 | -0.0210 | -0.0274 | **KEEP CURRENT MODEL** |

## Validation Set Performance Comparison

| Target | Mode | Stage 1D Best Model | Stage 1D Val ROC-AUC | Stage 1D Val PR-AUC | Best XGBoost Variant | XGBoost Val ROC-AUC | XGBoost Val PR-AUC | Val ROC-AUC Delta | Val PR-AUC Delta |
|---|---|---|---|---|---|---|---|---|---|
| cvd | mode_a | `logistic_regression` | 0.8166 | 0.3714 | `xgboost_default` | 0.7650 | 0.2988 | -0.0516 | -0.0726 |
| cvd | mode_b | `logistic_regression_balanced` | 0.8323 | 0.4221 | `xgboost_default` | 0.7894 | 0.3368 | -0.0429 | -0.0853 |
| diabetes | mode_a | `hist_gradient_boosting` | 0.7967 | 0.4057 | `xgboost_balanced` | 0.7730 | 0.3805 | -0.0237 | -0.0252 |
| diabetes | mode_b | `hist_gradient_boosting` | 0.8234 | 0.4644 | `xgboost_default` | 0.8159 | 0.4654 | -0.0075 | +0.0010 |
| hypertension | mode_a | `logistic_regression` | 0.7884 | 0.6807 | `xgboost_default` | 0.7551 | 0.6672 | -0.0334 | -0.0135 |
| hypertension | mode_b | `hist_gradient_boosting_balanced` | 0.7983 | 0.7186 | `xgboost_balanced` | 0.7845 | 0.7159 | -0.0138 | -0.0027 |
