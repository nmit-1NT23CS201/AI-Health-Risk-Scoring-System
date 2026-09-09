# NHANES 2021–2023 Processed Dataset QC Report
> **Stage:** 1C — Data Preparation  
> **Total Adults (age ≥ 20):** 7,809  
> **Duplicate SEQN:** 0  

---

## 1. Participant Counts

| Population | N |
|---|---|
| Total NHANES 2021–2023 interviewed | 11,933 |
| Adults aged ≥ 20 (analysis population) | 7,809 |
| MEC examined adults (BMX + BPXO coverage) | 5,970 |
| Routine lab panel adults (HbA1c, TC, HDL, etc.) | 5,767 |
| Fasting subsample adults (Glucose, TG, LDL) | 3,210 |

## 2. Target Prevalence

| Target | Positive | Negative | Missing/Undeterminable | Prevalence (%) |
|---|---|---|---|---|
| Hard CVD (MCQ160B-F) | 982 | 6,825 | 2 | 12.58% |
| Diabetes (ADA criteria) | 1,385 | 6,421 | 3 | 17.74% |
| Hypertension (JNC7) | 3,331 | 4,469 | 9 | 42.71% |

## 3. Mode A and Mode B Sample Sizes (per Model)

| Model | Mode | N with target | N complete cases | N features |
|---|---|---|---|---|
| CVD_MODE | A | 7,807 | 4,383 | 13 |
| CVD_MODE | B | 7,807 | 2,161 | 29 |
| DIABETES_MODE | A | 7,806 | 4,383 | 13 |
| DIABETES_MODE | B | 7,806 | 2,161 | 27 |
| HYPERTENSION_MODE | A | 7,800 | 4,383 | 11 |
| HYPERTENSION_MODE | B | 7,800 | 2,161 | 27 |

## 4. Feature Missingness Summary (Mode B, CVD Model)

| Feature | N present | Missing N | Missing % | Note |
|---|---|---|---|---|
| ldl_cholesterol | 3,042 | 4,767 | 61.04% | NHANES fasting subsample (~41% of adults) |
| triglycerides | 3,079 | 4,730 | 60.57% | NHANES fasting subsample (~41% of adults) |
| fasting_glucose | 3,210 | 4,599 | 58.89% | NHANES fasting subsample (~41% of adults) |
| alcohol_frequency | 5,230 | 2,579 | 33.03% |  |
| ast_enzyme | 5,426 | 2,383 | 30.52% |  |
| alt_enzyme | 5,438 | 2,371 | 30.36% |  |
| serum_creatinine | 5,443 | 2,366 | 30.3% |  |
| blood_urea_nitrogen | 5,444 | 2,365 | 30.29% |  |
| serum_uric_acid | 5,446 | 2,363 | 30.26% |  |
| serum_albumin | 5,478 | 2,331 | 29.85% |  |
| total_cholesterol | 5,498 | 2,311 | 29.59% |  |
| hdl_cholesterol | 5,498 | 2,311 | 29.59% |  |
| waist_circumference | 5,762 | 2,047 | 26.21% |  |
| hba1c | 5,767 | 2,042 | 26.15% |  |
| hemoglobin | 5,768 | 2,041 | 26.14% |  |
| platelet_count | 5,768 | 2,041 | 26.14% |  |
| wbc_count | 5,768 | 2,041 | 26.14% |  |
| rdw | 5,768 | 2,041 | 26.14% |  |
| mean_pulse | 5,863 | 1,946 | 24.92% |  |
| mean_dbp | 5,863 | 1,946 | 24.92% |  |
| mean_sbp | 5,863 | 1,946 | 24.92% |  |
| bmi | 5,970 | 1,839 | 23.55% |  |
| poverty_income_ratio | 6,489 | 1,320 | 16.9% |  |
| sedentary_minutes | 7,725 | 84 | 1.08% |  |
| smoking_status | 7,780 | 29 | 0.37% |  |
| education_level | 7,783 | 26 | 0.33% |  |
| physical_activity_level | 7,787 | 22 | 0.28% |  |
| age | 7,809 | 0 | 0.0% |  |
| gender | 7,809 | 0 | 0.0% |  |

## 5. Unit & Plausibility Checks

| Feature | N | Min | Max | Median | Mean | Flagged OOR | Plausible Range |
|---|---|---|---|---|---|---|---|
| bmi | 5,970 | 11.10 | 74.80 | 28.50 | 29.83 | 0 | [10.0, 80.0] |
| waist_circumference | 5,762 | 60.00 | 187.00 | 99.50 | 101.09 | 0 | [40.0, 200.0] |
| mean_sbp | 5,863 | 70.00 | 232.33 | 120.33 | 122.93 | 0 | [60.0, 260.0] |
| mean_dbp | 5,863 | 34.00 | 139.00 | 74.00 | 74.74 | 0 | [30.0, 160.0] |
| mean_pulse | 5,863 | 34.00 | 129.33 | 70.33 | 71.50 | 0 | [25.0, 200.0] |
| hba1c | 5,767 | 3.20 | 17.10 | 5.50 | 5.79 | 0 | [2.0, 20.0] |
| fasting_glucose | 3,210 | 59.00 | 561.00 | 101.00 | 109.77 | 1 | [40.0, 500.0] |
| total_cholesterol | 5,498 | 62.00 | 438.00 | 185.00 | 188.08 | 0 | [50.0, 600.0] |
| hdl_cholesterol | 5,498 | 22.00 | 159.00 | 52.00 | 54.52 | 0 | [10.0, 200.0] |
| triglycerides | 3,079 | 19.00 | 1745.00 | 99.00 | 120.65 | 0 | [10.0, 2000.0] |
| ldl_cholesterol | 3,042 | 3.00 | 314.00 | 106.00 | 108.61 | 5 | [20.0, 400.0] |
| serum_creatinine | 5,443 | 0.35 | 15.17 | 0.85 | 0.90 | 0 | [0.2, 20.0] |
| blood_urea_nitrogen | 5,444 | 4.00 | 74.00 | 14.00 | 15.18 | 0 | [2.0, 150.0] |
| serum_uric_acid | 5,446 | 1.10 | 13.20 | 5.10 | 5.16 | 0 | [1.0, 20.0] |
| alt_enzyme | 5,438 | 3.00 | 350.00 | 18.00 | 21.78 | 0 | [3.0, 2000.0] |
| ast_enzyme | 5,426 | 6.00 | 363.00 | 20.00 | 22.49 | 0 | [3.0, 2000.0] |
| serum_albumin | 5,478 | 2.20 | 5.50 | 4.10 | 4.07 | 0 | [1.0, 7.0] |
| hemoglobin | 5,768 | 6.30 | 18.60 | 13.90 | 13.88 | 0 | [3.0, 22.0] |
| wbc_count | 5,768 | 2.10 | 18.40 | 6.60 | 6.91 | 0 | [0.5, 50.0] |
| platelet_count | 5,768 | 46.00 | 787.00 | 248.00 | 255.45 | 0 | [10.0, 1000.0] |
| rdw | 5,768 | 11.50 | 37.50 | 13.60 | 13.87 | 2 | [10.0, 30.0] |

## 6. Target Leakage Verification

> [!NOTE]
> **All leakage checks passed.** No prohibited variable appears in any model's predictor set.

## 7. Special Missing-Value Handling

| Code | Meaning | Variables Affected | Action |
|---|---|---|---|
| 7 / 77 / 777 / 7777 | Refused | SMQ020/040, DIQ010/050/070, MCQ160B-F, BPQ020, ALQ111, PAD790Q, PAD810Q, PAD680 | Recoded to NaN |
| 9 / 99 / 999 / 9999 | Don't Know | Same set as above | Recoded to NaN |
| 5.3976e-79 (SAS epsilon) | True zero (SAS floating-point underflow) | Continuous numeric columns in XPT | Clamped to 0.0 |

## 8. Survey-Weight Variables (Preserved, NOT Applied)

| Variable | Scope | Population | Status |
|---|---|---|---|
| WTINT2YR | 2-year interview weight | All interviewed (N=11,933) | Preserved in merged dataset |
| WTMEC2YR | 2-year MEC examination weight | All examined | Preserved; to be used in calibration (Stage 2) |
| WTSAF2YR | Fasting subsample weight | Fasting subsample (N~3996 total; ~3210 adults) | Present in GLU_L.xpt; preserved |
| SDMVSTRA | Masked pseudo-stratum | Survey design variable | Preserved for variance estimation |
| SDMVPSU | Masked pseudo-PSU | Survey design variable | Preserved for variance estimation |

## 9. Duplicate SEQN Check

Duplicate SEQN in merged adult dataset: **0**

> [!NOTE]
> SEQN uniquely identifies each NHANES participant. Zero duplicates is the expected and observed result.

## 10. Flags & Unexpected Issues

The following features contain values outside the defined plausible range. These are flagged for review — they are NOT dropped or clipped at this stage.

- `fasting_glucose`: 1 values outside [40.0, 500.0]
- `ldl_cholesterol`: 5 values outside [20.0, 400.0]
- `rdw`: 2 values outside [10.0, 30.0]

---

*Generated by `nhanes_quality_checks.py` — Stage 1C data preparation.*  
*Feature statistics: `nhanes_feature_statistics.csv`*
