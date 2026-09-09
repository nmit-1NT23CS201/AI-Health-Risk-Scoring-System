# Model V2 Baseline Benchmark Summary

> Stage 1D — Model Benchmarking across 6 configurations (NHANES 2021–2023)

## CVD_MODE_A

| Rank | Model | Val ROC-AUC | Val PR-AUC | Val Recall | Val Specificity | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|---|
| 1 | logistic_regression | 0.8166 | 0.3714 | 0.1088 | 0.9883 | 0.8125 | 0.3792 |
| 2 | logistic_regression_balanced | 0.8159 | 0.3746 | 0.7959 | 0.6826 | 0.8105 | 0.3740 |
| 3 | random_forest_balanced | 0.7947 | 0.3225 | 0.3265 | 0.9189 | 0.7755 | 0.3091 |
| 4 | hist_gradient_boosting | 0.7863 | 0.3123 | 0.1293 | 0.9736 | 0.7916 | 0.3551 |
| 5 | random_forest | 0.7821 | 0.3140 | 0.0612 | 0.9873 | 0.7821 | 0.3512 |
| 6 | hist_gradient_boosting_balanced | 0.7750 | 0.3114 | 0.5374 | 0.8086 | 0.7920 | 0.3662 |

## CVD_MODE_B

| Rank | Model | Val ROC-AUC | Val PR-AUC | Val Recall | Val Specificity | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|---|
| 1 | logistic_regression_balanced | 0.8323 | 0.4221 | 0.8231 | 0.7148 | 0.8191 | 0.4219 |
| 2 | logistic_regression | 0.8319 | 0.4208 | 0.1565 | 0.9873 | 0.8200 | 0.4289 |
| 3 | random_forest_balanced | 0.8250 | 0.3923 | 0.4286 | 0.9277 | 0.7973 | 0.3388 |
| 4 | random_forest | 0.8135 | 0.3587 | 0.0680 | 0.9873 | 0.7996 | 0.3763 |
| 5 | hist_gradient_boosting_balanced | 0.8127 | 0.4178 | 0.5646 | 0.8643 | 0.7909 | 0.3761 |
| 6 | hist_gradient_boosting | 0.8111 | 0.3675 | 0.1633 | 0.9746 | 0.8024 | 0.3980 |

## DIABETES_MODE_A

| Rank | Model | Val ROC-AUC | Val PR-AUC | Val Recall | Val Specificity | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|---|
| 1 | hist_gradient_boosting | 0.7967 | 0.4057 | 0.2212 | 0.9481 | 0.7810 | 0.4145 |
| 2 | logistic_regression_balanced | 0.7928 | 0.4128 | 0.7308 | 0.7009 | 0.7841 | 0.4183 |
| 3 | hist_gradient_boosting_balanced | 0.7926 | 0.4091 | 0.6106 | 0.7913 | 0.7918 | 0.4541 |
| 4 | logistic_regression | 0.7925 | 0.4120 | 0.1827 | 0.9657 | 0.7831 | 0.4178 |
| 5 | random_forest_balanced | 0.7882 | 0.3729 | 0.4183 | 0.8660 | 0.7673 | 0.3852 |
| 6 | random_forest | 0.7875 | 0.3933 | 0.1490 | 0.9564 | 0.7586 | 0.3982 |

## DIABETES_MODE_B

| Rank | Model | Val ROC-AUC | Val PR-AUC | Val Recall | Val Specificity | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|---|
| 1 | hist_gradient_boosting | 0.8234 | 0.4644 | 0.3365 | 0.9439 | 0.8003 | 0.4594 |
| 2 | logistic_regression_balanced | 0.8167 | 0.4860 | 0.7067 | 0.7300 | 0.7966 | 0.4307 |
| 3 | logistic_regression | 0.8157 | 0.4821 | 0.3029 | 0.9502 | 0.7938 | 0.4285 |
| 4 | hist_gradient_boosting_balanced | 0.8140 | 0.4638 | 0.5625 | 0.8390 | 0.8037 | 0.4503 |
| 5 | random_forest_balanced | 0.8063 | 0.4400 | 0.4423 | 0.8816 | 0.7741 | 0.3966 |
| 6 | random_forest | 0.8033 | 0.4580 | 0.2212 | 0.9626 | 0.7801 | 0.4363 |

## HYPERTENSION_MODE_A

| Rank | Model | Val ROC-AUC | Val PR-AUC | Val Recall | Val Specificity | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|---|
| 1 | logistic_regression | 0.7884 | 0.6807 | 0.6820 | 0.7567 | 0.7663 | 0.6550 |
| 2 | logistic_regression_balanced | 0.7883 | 0.6803 | 0.7760 | 0.6836 | 0.7661 | 0.6541 |
| 3 | random_forest | 0.7788 | 0.6746 | 0.6560 | 0.7403 | 0.7437 | 0.6446 |
| 4 | random_forest_balanced | 0.7771 | 0.6679 | 0.7260 | 0.7045 | 0.7394 | 0.6366 |
| 5 | hist_gradient_boosting_balanced | 0.7731 | 0.6703 | 0.7300 | 0.6925 | 0.7470 | 0.6618 |
| 6 | hist_gradient_boosting | 0.7709 | 0.6709 | 0.6640 | 0.7522 | 0.7455 | 0.6630 |

## HYPERTENSION_MODE_B

| Rank | Model | Val ROC-AUC | Val PR-AUC | Val Recall | Val Specificity | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|---|
| 1 | hist_gradient_boosting_balanced | 0.7983 | 0.7186 | 0.7400 | 0.7119 | 0.7637 | 0.6748 |
| 2 | hist_gradient_boosting | 0.7971 | 0.7149 | 0.6760 | 0.7597 | 0.7633 | 0.6755 |
| 3 | logistic_regression | 0.7938 | 0.7002 | 0.6940 | 0.7507 | 0.7727 | 0.6743 |
| 4 | logistic_regression_balanced | 0.7936 | 0.6996 | 0.7720 | 0.6910 | 0.7725 | 0.6732 |
| 5 | random_forest_balanced | 0.7918 | 0.7007 | 0.7380 | 0.7284 | 0.7584 | 0.6566 |
| 6 | random_forest | 0.7891 | 0.6943 | 0.6860 | 0.7493 | 0.7592 | 0.6594 |

