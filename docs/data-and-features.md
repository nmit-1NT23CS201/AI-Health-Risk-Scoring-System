# Data and Features (NHANES 2021–2023)

## 1. Raw sources

`backend/ml/data/raw/nhanes_2021_2023/` holds 16 NHANES August 2021–August 2023 (`_L`) `.xpt` files:

| Domain | Files |
|---|---|
| Demographics | DEMO |
| Examination | BMX, BPXO |
| Questionnaire | BPQ, SMQ, ALQ, PAQ, DIQ, MCQ |
| Laboratory | GHB, GLU, TCHOL, HDL, TRIGLY, BIOPRO, CBC |

`backend/ml/data/raw/icmr_indiab/` (`meta.pdf`, `sample.dta`) is present but **not used** by V2.

## 2. Pipeline scripts (`backend/ml/scripts/`)

| Script | Role |
|---|---|
| `nhanes_loader.py` | Reads `.xpt` files |
| `nhanes_harmonizer.py` | Age filter, SEQN merge, recoding, feature construction |
| `nhanes_targets.py` | Target construction |
| `nhanes_quality_checks.py` | Feature lists, per-target exclusions, QC |
| `build_nhanes_dataset.py` | Orchestration; writes interim and processed outputs |
| `audit_datasets.py`, `generate_feature_matrix.py` | Dataset audit / feature matrix |

## 3. Extraction, merge, inclusion

- **Age:** `RIDAGEYR >= 20` (`nhanes_harmonizer.py:263`), applied to DEMO before any join: 11,933 → **7,809** participants.
- **Merge:** left-join each component on `SEQN` with an assertion that row count is unchanged (`:271-293`).
- **Recoding:** NHANES refused/don't-know codes (7/9, 77/99, 777/999, 7777/9999 as applicable) → NaN (`:52-73`, `:188-206`). Smoking, alcohol and activity recoded to ordinal features (`:80-158`).
- **Blood pressure:** `mean_sbp`, `mean_dbp`, `mean_pulse` are skip-NaN means of three oscillometric readings (`:161-185`).
- **Per-target exclusion:** rows with a missing target are dropped at modelling time: CVD 7,807, Diabetes 7,806, Hypertension 7,800.

## 4. Missing values (actual behaviour)

- No dataset-level imputation; processed parquet files keep NaN.
- Logistic Regression and Random Forest: `SimpleImputer(strategy="median")` (plus `StandardScaler` for LR) inside the sklearn Pipeline, fitted on training folds only (`optimize_models.py:251-256`, `optimize_random_forest.py:180-185`).
- HistGradientBoosting and XGBoost handle NaN natively.
- `DATASET_MANIFEST.md` lines 94-96 describe a planned `IterativeImputer` that was never implemented (stale).

## 5. Outputs

| Location | Contents |
|---|---|
| `backend/ml/data/interim/nhanes_2021_2023_merged.parquet` | Merged harmonized cohort |
| `backend/ml/data/interim/audit/` | Cleaning log, target construction JSON, QC files |
| `backend/ml/data/processed/nhanes_2021_2023/` | `nhanes_{cvd,diabetes,hypertension}_mode_{a,b}.parquet`, `DATASET_MANIFEST.md` |
| `.../splits/` | `cvd_splits.csv`, `diabetes_splits.csv`, `hypertension_splits.csv` |

All six parquet files contain **7,809** rows per `DATASET_MANIFEST.md` §10 and the Stage 1G-4 audit §3. The parquet files were not re-read independently (pandas was unavailable); split CSV totals were counted directly and are consistent.

## 6. Targets (`nhanes_targets.py`)

| Target | Positive if any holds | Missing rule | Positives |
|---|---|---|---|
| **CVD** (`:89-109`) | Any of `MCQ160B` (CHF), `C` (CHD), `D` (angina), `E` (heart attack), `F` (stroke) == 1 | 0 if at least one answered and none positive; NaN if all missing (2) | 982 (12.58%) |
| **Diabetes** (`:138-167`) | `hba1c >= 6.5`, `fasting_glucose (LBXGLU) >= 126`, `DIQ010 == 1`, `DIQ050 == 1` (insulin), `DIQ070 == 1` (pills) | 3 missing | 1,385 (17.74%) |
| **Hypertension** (JNC7, `:192-205`) | `mean_sbp >= 140`, `mean_dbp >= 90`, or `BPQ020 == 1` | 9 missing | 3,331 (42.71%) |

CVD is a **self-reported prior diagnosis**: the model estimates probability of having a CVD history, not future incidence.

## 7. Leakage prevention (`nhanes_quality_checks.py:80-122`)

| Target | Excluded from predictors |
|---|---|
| CVD | `MCQ160B–F`, `DIQ050`, `DIQ070`, `BPQ101D` |
| Diabetes | `hba1c`, `fasting_glucose`, `DIQ010`, `DIQ050`, `DIQ070`, `BPQ101D` |
| Hypertension | `mean_sbp`, `mean_dbp`, `BPQ020`, `BPQ101D` |

**Survey metadata:** `SEQN`, `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR`, `SDMVSTRA`, `SDMVPSU` are kept in the parquet files (`build_nhanes_dataset.py:118-123`) but removed from predictors via `EXCLUDE_COLS` (`benchmark_models.py:78`, `optimize_models.py:92`, `calibrate_and_optimize_thresholds.py:81-83`). **Survey weights are not used**, so metrics are unweighted sample estimates.

**Known weakness:** the secondary exclusion list in `calibrate_and_optimize_thresholds.py:85-99` uses raw-style names (`lbxglu`, `bpa_mean_sbp`) that do not match harmonized column names. Not an active leak (earlier stages already exclude these columns and final feature lists contain none), but that guard would not catch a regression.

## 8. Mode A and Mode B features

**Mode A base (13):** `age`, `gender`, `education_level`, `poverty_income_ratio`, `bmi`, `waist_circumference`, `mean_sbp`, `mean_dbp`, `mean_pulse`, `smoking_status`, `alcohol_frequency`, `physical_activity_level`, `sedentary_minutes`.

**Mode B lab additions (16):** `hba1c`, `fasting_glucose`, `total_cholesterol`, `hdl_cholesterol`, `triglycerides`, `ldl_cholesterol`, `serum_creatinine`, `blood_urea_nitrogen`, `serum_uric_acid`, `alt_enzyme`, `ast_enzyme`, `hemoglobin`, `wbc_count`, `platelet_count`, `rdw`, `serum_albumin`.

| Configuration | # Features | Composition |
|---|---:|---|
| CVD A | 13 | Mode A base |
| CVD B | 29 | Base + 16 labs |
| Diabetes A | 13 | Mode A base |
| Diabetes B | 27 | Base + 16 labs − `hba1c`, `fasting_glucose` |
| Hypertension A | 11 | Base − `mean_sbp`, `mean_dbp` |
| Hypertension B | 27 | 11 + 16 labs |

**Authoritative feature order:** no separate manifest exists. The ordered list per configuration is the `features` key of each record in `backend/ml/evaluation/stage_1f/stage_1f_results.json` (written at `calibrate_and_optimize_thresholds.py:563`).

V2 features differ entirely from V1's (no HRV, SpO₂ or family history); V1 form data cannot be fed to V2 models.

## 9. Splits

- Generated in `benchmark_models.py:254-281` (Stage 1D): Mode A dataset minus missing targets → `train_test_split(test_size=0.30, random_state=42, stratify=target)`, then the remaining 30% split 50/50 (same seed, stratified) = **70/15/15**.
- One split file per disease, **shared by Mode A and B**.
- Columns: `SEQN`, `target_<disease>`, `split` (`train` / `validation` / `test`).

| Disease | Train (pos) | Validation (pos) | Test (pos) | Total |
|---|---|---|---|---|
| CVD | 5,464 (687) | 1,171 (147) | 1,172 (148) | 7,807 |
| Diabetes | 5,464 (969) | 1,171 (208) | 1,171 (208) | 7,806 |
| Hypertension | 5,460 (2,332) | 1,170 (500) | 1,170 (499) | 7,800 |

Stage 1G-1 confirmed zero participant overlap between splits. Seed 42 is used throughout. Reproducibility limits: unpinned dependencies; no recorded library versions for serialized models; report hyperlinks point to another machine (`C:/Users/sunny/...`).
