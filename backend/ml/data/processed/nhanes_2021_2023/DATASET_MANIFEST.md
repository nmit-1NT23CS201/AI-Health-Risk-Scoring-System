# NHANES 2021–2023 Processed Dataset Manifest

> **Stage:** 1C — Data Preparation Pipeline  
> **Source Dataset:** CDC NHANES August 2021–August 2023  
> **Analysis Population:** Adults aged 20+, N = 7,809  
> **Status:** Preparation Complete — Awaiting Stage 1D (Train/Val/Test Split)  

---

## 1. Source Files

| File | Domain | Rows | Key Variables Used |
|---|---|---|---|
| `DEMO_L.xpt` | Demographics, Survey Design | 11,933 | RIDAGEYR, RIAGENDR, DMDEDUC2, INDFMPIR, WTINT2YR, WTMEC2YR, SDMVSTRA, SDMVPSU |
| `BMX_L.xpt` | Anthropometrics | 8,860 | BMXBMI, BMXWAIST |
| `BPXO_L.xpt` | Blood Pressure (Oscillometric) | 7,801 | BPXOSY1-3, BPXODI1-3, BPXOPLS1-3 |
| `BPQ_L.xpt` | BP Questionnaire | 8,501 | BPQ020 (HTN target component) |
| `SMQ_L.xpt` | Smoking | 9,015 | SMQ020, SMQ040 |
| `ALQ_L.xpt` | Alcohol | 6,337 | ALQ111, ALQ121 |
| `PAQ_L.xpt` | Physical Activity | 8,153 | PAD790Q, PAD810Q, PAD680 |
| `DIQ_L.xpt` | Diabetes Questionnaire | 11,744 | DIQ010, DIQ050, DIQ070 (DM target) |
| `MCQ_L.xpt` | Medical Conditions | 11,744 | MCQ160B-F (CVD target) |
| `GHB_L.xpt` | HbA1c Laboratory | 7,199 | LBXGH |
| `GLU_L.xpt` | Fasting Glucose (Subsample) | 3,996 | LBXGLU, WTSAF2YR |
| `TCHOL_L.xpt` | Total Cholesterol | 8,068 | LBXTC |
| `HDL_L.xpt` | HDL Cholesterol | 8,068 | LBDHDD |
| `TRIGLY_L.xpt` | Triglycerides + LDL (Fasting) | 3,996 | LBXTLG, LBDLDL |
| `BIOPRO_L.xpt` | Biochemistry Profile | 7,199 | LBXSCR, LBXSBU, LBXSUA, LBXSATSI, LBXSASSI, LBXSAL |
| `CBC_L.xpt` | Complete Blood Count | 8,727 | LBXHGB, LBXWBCSI, LBXPLTSI, LBXRDW |

## 2. Merge Strategy

- **Base:** `DEMO_L.xpt` (all 11,933 participants)
- **Adult filter:** Applied first — `RIDAGEYR >= 20` → N = 7,809
- **Join type:** Left join on `SEQN` for all subsequent components
- **Rationale:** Preserves Mode A participants without Mode B lab data. Missing lab values appear as NaN, not as participant exclusion.

## 3. Target Definitions

### Hard CVD (target_cvd)
- **Formula:** `MCQ160B=1 OR MCQ160C=1 OR MCQ160D=1 OR MCQ160E=1 OR MCQ160F=1`
- **Criterion:** Self-reported physician diagnosis of CHF, CHD, Angina, MI, or Stroke
- **Result:** Positive=982 (12.58%), Negative=6,825, Missing=2

### Diabetes (target_diabetes) — ADA Multi-Criteria
- **Formula:** `HbA1c>=6.5 OR Fasting Glucose>=126 OR DIQ010=1 OR DIQ050=1 OR DIQ070=1`
- **Prohibited predictors in DM model:** hba1c, fasting_glucose, DIQ010, DIQ050, DIQ070
- **Result:** Positive=1,385 (17.74%), Negative=6,421, Missing=3

### Hypertension — JNC7 (target_hypertension)
- **Formula:** `mean_sbp>=140 OR mean_dbp>=90 OR BPQ020=1`
- **Definition:** JNC7 hypertension threshold (NOT ACC/AHA 2017, NOT Stage-2 only)
- **Prohibited predictors in HTN model:** mean_sbp, mean_dbp, BPQ020
- **Result:** Positive=3,331 (42.71%), Negative=4,469, Missing=9

## 4. Cleaning Rules

| Rule | Variables | Action |
|---|---|---|
| Refused (7/77/777/7777) | SMQ020/040, DIQ010/050/070, MCQ160B-F, BPQ020, ALQ111, PAD790Q/810Q/680 | Recoded to NaN |
| Don't Know (9/99/999/9999) | Same set | Recoded to NaN |
| SAS epsilon (5.4e-79) | All continuous XPT columns | Clamped to 0.0 |
| Smoking recode | SMQ020 + SMQ040 | 0=Never, 1=Former, 2=Current → smoking_status |
| Alcohol recode | ALQ111 + ALQ121 | 0=Never, 1=Former, 2=Current → alcohol_frequency |
| Physical activity recode | PAD790Q + PAD810Q | 1=High, 2=Moderate, 3=Low → physical_activity_level |
| BP means | BPXOSY1-3, BPXODI1-3, BPXOPLS1-3 | nanmean of valid readings → mean_sbp, mean_dbp, mean_pulse |

## 5. Predictor Exclusions (Leakage Matrix)

| Variable | CVD Model | DM Model | HTN Model |
|---|---|---|---|
| MCQ160B-F | PROHIBITED (defines target) | Not used | Not used |
| hba1c, fasting_glucose | ALLOWED | PROHIBITED (defines target) | ALLOWED |
| DIQ010, DIQ050, DIQ070 | Not used | PROHIBITED (defines target) | Not used |
| mean_sbp, mean_dbp | ALLOWED | ALLOWED | PROHIBITED (defines target) |
| BPQ020 | Not used | Not used | PROHIBITED (defines target) |
| BPQ101D (chol meds) | PROHIBITED (treatment proxy) | PROHIBITED | PROHIBITED |

## 6. Mode A Feature Set (13 Features)

age, gender, education_level, poverty_income_ratio, bmi, waist_circumference, mean_sbp\*, mean_dbp\*, mean_pulse, smoking_status, alcohol_frequency, physical_activity_level, sedentary_minutes

\* mean_sbp and mean_dbp are excluded from the HTN model predictor set.

## 7. Mode B Feature Set (Mode A + 16 Lab Biomarkers = 29 Features)

Mode A features + hba1c\*, fasting_glucose\*, total_cholesterol, hdl_cholesterol, triglycerides, ldl_cholesterol, serum_creatinine, blood_urea_nitrogen, serum_uric_acid, alt_enzyme, ast_enzyme, hemoglobin, wbc_count, platelet_count, rdw, serum_albumin

\* hba1c and fasting_glucose excluded from the DM model predictor set.

## 8. Missing-Value Strategy (Current Stage)

No imputation has been applied. All missing values remain as NaN.
- Mode A has larger sample sizes because it does not require laboratory measurements.
- Mode B has smaller effective sample sizes due to lab coverage (70.4% for routine panel, 41.1% for fasting subsample).
- Imputation strategy (IterativeImputer for Tier 1 labs, Median+indicator for fasting subsample) will be applied in Stage 1D during train/val/test splitting.

## 9. Survey-Weight Variables (Preserved, Not Applied)

| Variable | Location in raw data | Purpose |
|---|---|---|
| WTINT2YR | DEMO_L.xpt | 2-year interview weight (all participants) |
| WTMEC2YR | DEMO_L.xpt | 2-year MEC examination weight |
| WTSAF2YR | GLU_L.xpt | Fasting subsample weight (confirmed present) |
| SDMVSTRA | DEMO_L.xpt | Masked variance pseudo-stratum |
| SDMVPSU  | DEMO_L.xpt | Masked variance pseudo-PSU |

Survey weights will be applied during probability calibration (Stage 2), not during tree model training.

## 10. Output Files

| File | Rows | Features | Target |
|---|---|---|---|
| `nhanes_cvd_mode_a.parquet` | 7,809 | 13 | target_cvd |
| `nhanes_cvd_mode_b.parquet` | 7,809 | 29 | target_cvd |
| `nhanes_diabetes_mode_a.parquet` | 7,809 | 13 | target_diabetes |
| `nhanes_diabetes_mode_b.parquet` | 7,809 | 27 | target_diabetes |
| `nhanes_hypertension_mode_a.parquet` | 7,809 | 11 | target_hypertension |
| `nhanes_hypertension_mode_b.parquet` | 7,809 | 27 | target_hypertension |

## 11. Reproducibility

Run the full pipeline from the workspace root:

```bash
python -m backend.ml.scripts.build_nhanes_dataset
```

All outputs are regenerated from raw XPT files with no manual edits required.

---

*Generated by `build_nhanes_dataset.py` — Stage 1C.*
