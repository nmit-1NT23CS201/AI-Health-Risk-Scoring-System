# Phase 1 — Model Performance Investigation (Frozen V2, NHANES 2021–2023)

> **Status:** diagnostic audit and Phase 2 experiment plan. No model, dataset, script, API or frontend file was changed.
> **Test-set quarantine:** every number computed for this report uses only the `train` and `validation` splits. The `test` split rows are dropped right after loading (asserted in code). Test metrics shown in §1 are the **static published Stage 1F values** copied from `stage_1f_results.json`; they were not recomputed.
> **Reproduce:** `.\.venv\Scripts\python.exe backend/ml/experiments/scripts/diagnostics_phase1.py` → `backend/ml/experiments/results/phase1_diagnostics.json` (all raw numbers below come from that file plus one lipid check, §3.3).

Conventions: "base" is the Stage 1F estimator unwrapped from `CalibratedClassifierCV → FrozenEstimator`, refit by Stage 1F on `train` only. "Calibrated" is the full artifact; its sigmoid was fit on `validation`, so **calibrated-on-validation metrics are in-sample and are not used as evidence of calibration quality**. Validation sizes are 1,171 / 1,171 / 1,170 rows, with 147 / 208 / 500 positives (CVD / Diabetes / Hypertension).

---

## 1. Baseline summary (published Stage 1F, held-out test — static)

| Config | Model (base) | # feat | Threshold | ROC-AUC | PR-AUC | Brier | ECE | Sens | Spec | Prec | NPV | F1 | Bal.Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CVD A | RF (500, depth 8, log2) | 13 | 0.13 | 0.8137 | 0.4159 | 0.0912 | 0.0125 | 0.6824 | 0.7803 | 0.3098 | 0.9444 | 0.4262 | 0.7314 |
| CVD B | RF (800, depth 8, 0.5, leaf 10) | 29 | 0.15 | 0.8218 | 0.4340 | 0.0906 | 0.0368 | 0.6081 | 0.8438 | 0.3600 | 0.9371 | 0.4523 | 0.7259 |
| Diabetes A | HGB (lr 0.05, 200 it, 15 leaves, l2 10) | 13 | 0.15 | 0.7957 | 0.4348 | 0.1222 | 0.0250 | 0.8173 | 0.6397 | 0.3288 | 0.9419 | 0.4690 | 0.7285 |
| Diabetes B | XGB (500, depth 5, lr 0.01, λ 0.1) | 27 | 0.17 | 0.8162 | 0.4793 | 0.1203 | 0.0470 | 0.6202 | 0.8100 | 0.4135 | 0.9080 | 0.4962 | 0.7151 |
| HTN A | LR (C 10, balanced) | 11 | 0.41 | 0.7662 | 0.6541 | 0.1946 | 0.0331 | 0.7615 | 0.6647 | 0.6281 | 0.7894 | 0.6884 | 0.7131 |
| HTN B | RF (800, depth 8, 0.5, leaf 10) | 27 | 0.52 | 0.7731 | 0.6898 | 0.1916 | 0.0465 | 0.6253 | 0.7511 | 0.6514 | 0.7294 | 0.6380 | 0.6882 |

Hyperparameters shown were read from the loaded artifacts. Known caveat (unchanged): the test split also informed model-family choice in Stages 1D–1E.

---

## 2. Overfitting vs underfitting (base estimators, train vs validation)

| Config | Train ROC-AUC | Val ROC-AUC | **Gap** | Train PR-AUC | Val PR-AUC | Train Brier | Val Brier | Val calib. slope (base) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CVD A (RF-500) | 0.9307 | 0.8229 | **+0.108** | 0.757 | 0.384 | 0.0725 | 0.0924 | 1.36 |
| CVD B (RF-800) | 0.9199 | 0.8325 | **+0.087** | 0.707 | 0.439 | 0.0723 | 0.0885 | 1.15 |
| Diabetes A (HGB) | 0.8984 | 0.8025 | **+0.096** | 0.705 | 0.417 | 0.0916 | 0.1238 | 0.88 |
| Diabetes B (XGB) | 0.9223 | 0.8328 | **+0.090** | 0.775 | 0.508 | 0.0824 | 0.1132 | 1.06 |
| HTN A (LR) | 0.7825 | 0.7883 | **−0.006** | 0.689 | 0.680 | 0.1901 | 0.1874 | 1.02 |
| HTN B (RF-800) | 0.8760 | 0.8072 | **+0.069** | 0.855 | 0.732 | 0.1489 | 0.1776 | 1.14 |

Findings:

- **All five tree models clearly overfit.** The AUC gap is 0.07–0.11, and the PR-AUC roughly halves from train to validation for CVD A (0.757 → 0.384) and Diabetes A/B. CVD A is worst even though it has the fewest features. It uses `min_samples_leaf=1` with `max_depth=8` on 687 training positives.
- **HTN A (LR) is underfitting, not overfitting.** There is no gap, and its validation AUC (0.788) is ~0.02 below HTN B. With only 11 features and linear terms it cannot represent age×BMI curvature. Age alone carries 0.236 of the AUC under permutation.
- **The tree models are under-confident before calibration.** RF slopes of 1.14–1.36 are the usual compressed RF probabilities, and sigmoid calibration corrects them. HTN A LR with `class_weight="balanced"` over-predicts (val mean predicted 0.486 vs prevalence 0.427, intercept −0.31). Calibration also fixes that, but it is avoidable.
- **Train/validation gaps are big relative to sampling noise.** The bootstrap SD of a validation AUC is ~0.006–0.010 (§6), so stronger regularisation is a real lever, not noise-chasing.

---

## 3. Feature signal, redundancy and weak predictors

### 3.1 Signal concentration (validation permutation importance, ΔROC-AUC, base models)

| Config | Top features (Δ AUC) | Features with Δ ≤ 0.001 |
|---|---|---|
| CVD A | age 0.188 · physical_activity 0.012 · waist 0.012 · smoking 0.008 · bmi 0.007 | sedentary_minutes |
| CVD B | age 0.180 · **total_cholesterol 0.036** · activity 0.008 · creatinine 0.006 | **16 of 29**: mean_dbp, mean_pulse, alcohol, gender, wbc, albumin, TG, fasting_glucose, AST, ALT, HDL, sedentary, BUN, LDL, Hb, uric acid |
| Diabetes A | age 0.130 · waist 0.041 · pulse 0.028 · SBP 0.025 · DBP 0.014 | smoking_status |
| Diabetes B | age 0.119 · TC 0.011 · pulse 0.010 · TG 0.010 · waist 0.010 | **12 of 27**: ALT, activity, DBP, creatinine, smoking, gender, wbc, alcohol, bmi, LDL, albumin, rdw |
| HTN A | age 0.236 · bmi 0.026 · waist 0.006 | education, activity, smoking, alcohol |
| HTN B | age 0.207 · **hba1c 0.014** · bmi 0.013 · waist 0.008 | **18 of 27** (incl. most CBC, liver, renal markers) |

**Age dominates every model** (permutation Δ 0.12–0.24 vs ≤ 0.04 for anything else). Train mutual information agrees: age is the top or joint-top feature in all six. Permutation importance is a single-model, marginal measure and understates correlated features (e.g. bmi in Diabetes B, where waist carries the signal). Pruning decisions must therefore come from retraining ablations (Track A), not from this table alone.

`shap_global_values.csv` (Stage 1G-3) agrees qualitatively with this table. It is **computed on the test split**, however, so it is cited only as an existing artifact and was not used for any decision here.

### 3.2 Collinearity (train)

| Pair (Spearman ρ) | Configs |
|---|---|
| total_cholesterol – ldl_cholesterol **0.93–0.94** | all Mode B |
| bmi – waist_circumference **0.89–0.90** (VIF ≈ 6–7) | all six |
| alt_enzyme – ast_enzyme **0.73–0.74** | all Mode B |

### 3.3 Structural redundancy: LDL is a derived column, not a measurement

Lipid VIFs on complete-case train rows are 1,800–22,000. A direct check across all 3,042 rows with all four lipids shows the cause: `ldl_cholesterol = TC − HDL − TG/5` to within 0.4 mg/dL (linear fit R² = 0.99994, coefficients 1.000 / −1.000 / −0.200). **LDL is the Friedewald estimate**, so Mode B contains an exact linear combination of three other features. It also inherits the fasting-subsample missingness of TG (61%).

### 3.4 Reverse-causation signal in total cholesterol (treatment proxy)

The second-strongest Mode B feature for CVD and Diabetes is total cholesterol, but with an **inverted** relationship (train):

| Target | TC tertile low / mid / high target rate | Median TC (positive vs negative) |
|---|---|---|
| CVD | **0.212 / 0.095 / 0.066** | 160 vs 187 mg/dL |
| Diabetes | **0.301 / 0.152 / 0.118** | 166 vs 188 mg/dL |
| Hypertension | 0.497 / 0.393 / 0.433 | 183 vs 187 mg/dL |

People with diagnosed CVD or diabetes are routinely treated with statins, which lowers cholesterol. The models are therefore partly learning **"low cholesterol ⇒ already treated ⇒ already has the condition"**. This is legitimate for predicting a *prevalent self-reported diagnosis*, which is what the CVD target is. It is not a preventive risk factor, though, and it will behave unexpectedly for untreated screening users. The leakage lists exclude diabetes medications (`DIQ050`, `DIQ070`) and `BPQ101D`, but no lipid-lowering-medication field exists in the dataset.

### 3.5 Encoding quirks (train target rates)

- `physical_activity_level` (1 high / 2 moderate / 3 low) is monotone and useful despite ignoring `PAD790U`/`PAD810U`. CVD: 9.3% / 10.6% / 21.8%; Diabetes: 11.5% / 16.8% / 28.3%. The unit bug mostly blurs the 1-vs-2 boundary (small difference), and level 3 (all zeros) carries the signal.
- `alcohol_frequency` is 32.7% missing on train. "Former" drinkers have the highest rates (CVD 21.1% vs 9.9% current; Diabetes 26.7% vs 15.5%), a classic sick-quitter pattern. Coding never/former/current as ordinal 0/1/2 forces a monotone relationship onto a non-monotone one. Linear HTN A cannot use it (permutation Δ ≤ 0.001), and the trees partly recover it.
- `smoking_status` has the same shape: former > current > never (CVD 19.4% / 15.7% / 8.5%). Ordinal coding is again non-monotone for linear models.

---

## 4. Missing data and preprocessing

| Group (train) | Missing | Target rate missing vs observed |
|---|---:|---|
| BP / pulse (exam not attended) | 24.6% | CVD 12.2 vs 12.7%; **Diabetes 13.8 vs 19.0%**; HTN (pulse) 36.3 vs 44.8% |
| bmi / waist | 23–26% | Diabetes 13.3 vs 19.1%; HTN 36.8 vs 44.5% |
| Lab panel (HbA1c, CBC, chemistry) | 26–30% | Diabetes ≈ 13–15 vs 19%; HTN ≈ 37–40 vs 44% |
| Fasting subsample (glucose, TG, LDL) | 59–61% | CVD 12.7 vs 12.3% (uninformative) |
| alcohol_frequency | 32.7% | Diabetes 16.0 vs 18.5% |

- **Missingness is informative for Diabetes and HTN, not for CVD.** A logistic model on missingness indicators alone, trained on train and scored on validation, gives AUC 0.551 / 0.563 (Diabetes A/B) and 0.532 / 0.536 (HTN A/B), versus 0.510 / 0.519 for CVD. Missingness clusters by NHANES component (exam vs lab vs fasting subsample), so it is effectively a "did not attend the exam/lab" flag. That flag correlates with being younger and healthier.
- **Median imputation without indicators leaves residual bias in the RF/LR models.** Mean validation residual (y − calibrated p) is +0.014 vs −0.012 for CVD A rows with any missing value vs complete rows, and +0.008 vs −0.019 for CVD B. HTN B shows the reverse sign (−0.008 vs +0.020). These offsets are small but systematic, and the median pipelines cannot represent "value absent".
- **HGB (Diabetes A) and XGB (Diabetes B) already route NaN natively.** They are the only configs where missingness is informative and usable. This is a plausible contributor to Diabetes B's +0.030 validation AUC gain (§6).
- **Serving implication:** the API sends absent fields as NaN. A screening user who omits labs looks like an NHANES non-attender (healthier on average) to native-NaN models. That is a distribution-shift risk that Track B must measure explicitly.

---

## 5. Calibration and operating points

### 5.1 Why Mode B shows worse test ECE

- On **validation**, the uncalibrated Mode B bases are *better* calibrated than Mode A (raw ECE 0.017 vs 0.027 CVD, 0.040 vs 0.043 Diabetes, 0.027 vs 0.058 HTN; from Stage 1F's own comparison). The higher Mode B test ECE (0.037 / 0.047 / 0.047) is not inherent miscalibration.
- Each sigmoid is fit on only 1,171 rows (147 / 208 positives for CVD / Diabetes), and Mode B probabilities are more spread out: the validation 99th percentile is 0.70 / 0.86 vs 0.66 / 0.66 for Mode A. A 2-parameter sigmoid fit to few high-probability positives, and evaluated with 10 bins on 1,170 test rows, gives an ECE whose sampling noise is comparable to the A/B differences. A 0.01–0.03 ECE difference should not be read as a real calibration difference without a bootstrap CI (Track D adds one).

### 5.2 Why Mode B has lower sensitivity

- The thresholds were chosen by max Youden's J subject to sensitivity and specificity ≥ 0.65. Mode B's better ranking lets Youden move to a **higher-specificity point**. Validation sensitivity at the locked thresholds is CVD 0.830 (A) vs 0.776 (B), Diabetes 0.817 vs 0.793, HTN 0.790 vs 0.726. The trade-off is partly a choice made by the selection rule, not only a model weakness.
- **The operating points sit on steep parts of the curve.** On validation, ±0.01 around the locked threshold moves sensitivity by 2.5–5.5 points. CVD A goes 0.776 → 0.701 → 0.646 at t = 0.11 / 0.13 / 0.14; Diabetes B goes 0.712 → 0.659 → 0.649 at 0.16 / 0.17 / 0.18. 13% (CVD A), 15% (Diabetes A) and ~6–10% (others) of validation participants lie within ±0.03 of the threshold.
- **Single-split threshold selection is noisy.** With 147 validation positives one CVD case equals 0.7 sensitivity points, and the binomial SD of sensitivity is ≈ 0.035. The published validation→test sensitivity drop for CVD B (0.776 → 0.608) is ~4.8 SDs. That is consistent with optimistic threshold selection on one small split plus test sampling noise. Thresholds need cross-validated selection and a sensitivity *floor* applied with a margin.

---

## 6. Mode A vs Mode B: where the gain comes from

**Paired bootstrap of validation ROC-AUC (B − A, same participants, base models, 1,000 resamples):**

| Disease | Δ AUC | 95% CI | Significant? |
|---|---:|---|---|
| CVD | +0.010 | [−0.010, +0.029] | **No** |
| Diabetes | +0.030 | [+0.012, +0.049] | Yes |
| Hypertension | +0.019 | [+0.008, +0.031] | Yes |

**Group permutation on validation (Mode B base models, ΔAUC when a whole lab group is shuffled):**

| Group | CVD B | Diabetes B | HTN B |
|---|---:|---:|---:|
| Lipids (TC, HDL, TG, LDL) | **0.047** | **0.036** | 0.007 |
| Glycaemic (HbA1c, glucose) | 0.003 | — (excluded) | **0.018** |
| Renal (creatinine, BUN, uric acid) | 0.005 | 0.009 | 0.002 |
| Liver (ALT, AST, albumin) | 0.000 | 0.002 | 0.001 |
| CBC (Hb, WBC, platelets, RDW) | 0.006 | 0.001 | −0.001 |
| All 16 labs | 0.077 | 0.067 | 0.034 |

Interpretation:

- **CVD B:** the lab gain is not statistically distinguishable from zero. The signal it does use is mostly the inverted total-cholesterol (statin) effect (§3.4). Liver, CBC and glycaemic groups add nothing, and 16 of 29 features are individually near zero.
- **Diabetes B:** this is the most real gain. It comes from lipids (TG and TC, again partly treatment-driven) and renal markers, plus native-NaN use of lab missingness. CBC and liver add nothing.
- **HTN B:** the gain comes from HbA1c (glycaemic) and lipids. CBC is slightly negative, and liver/renal add ≤ 0.002.
- **Across all three:** CBC (4 features) and liver (3 features) are noise for every disease. They are candidates for removal: 7 of 16 lab fields, which also lowers the data-entry burden for screening users.

---

## 7. Top 5 bottlenecks (ranked by expected impact)

| # | Bottleneck | Evidence |
|---|---|---|
| 1 | **Tree models overfit** (CVD A leaf 1 is the worst) | Train−val AUC gap 0.07–0.11; PR-AUC roughly halves (§2) |
| 2 | **Treatment-proxy signal in total cholesterol** inflates CVD/Diabetes Mode B and undermines preventive use | Inverted TC tertile rates 21%→7% (CVD), 30%→12% (Diabetes); lipids are the largest lab group (§3.4, §6) |
| 3 | **Noisy, redundant Mode B inputs** | LDL = Friedewald(TC, HDL, TG) with R² 0.99994; CBC + liver groups add ≈ 0; 12–18 features ~0 importance; CVD B gain not significant (§3.3, §6) |
| 4 | **Missing data handled two incompatible ways**; informative missingness unused by RF/LR | Indicator-only AUC up to 0.56; residual offsets ±0.01–0.02; native NaN only in HGB/XGB (§4) |
| 5 | **Fragile operating points and underfit HTN A** | ±0.01 threshold ⇒ 2.5–5.5 pt sensitivity swing; one-split threshold; CVD B val→test sensitivity −0.17; HTN A linear, no gap, non-monotone categoricals coded ordinally (§2, §3.5, §5) |

---

## 8. Phase 2 experiment plan (proposed — not run)

### 8.1 Protocol (applies to every track)

- **Isolation:** all code under `backend/ml/experiments/` (`scripts/`, `results/`, `models/`). The frozen `backend/ml/{data,scripts,evaluation}`, the API and the frontend are read-only.
- **Data:** the frozen processed parquet files and the frozen split files. Development set = `train ∪ validation`. **The test split stays sealed** until a single final evaluation of promoted candidates only.
- **Model selection:** repeated stratified CV on the development set (5-fold × 3 repeats, seed 42). Imputers, indicators, encoders, feature selectors and calibrators are fit **inside each training fold**. A candidate is compared against the frozen configuration **re-evaluated under the identical CV**, never against published test numbers.
- **Leakage guard:** each experiment asserts its feature list contains none of that disease's exclusions (CVD: MCQ160B–F, DIQ050/070, BPQ101D; Diabetes: hba1c, fasting_glucose, DIQ010/050/070, BPQ101D; HTN: mean_sbp, mean_dbp, BPQ020, BPQ101D). Derived features are checked against the **source** columns (e.g. no pulse-pressure feature for HTN, no glucose-based ratio for Diabetes).
- **Reporting:** mean ± SD over CV folds for ROC-AUC, PR-AUC, Brier, ECE (with bootstrap CI), calibration slope/intercept, and sensitivity/specificity at the CV-selected threshold. Paired fold-wise Δ vs the frozen baseline.

### 8.2 Tracks

**Track A — Feature engineering and selection (per disease, leakage-checked)**

| ID | Experiment | Applies to | Rationale |
|---|---|---|---|
| A1 | Drop `ldl_cholesterol` (or replace TC+LDL with **non-HDL = TC − HDL**) | Mode B | Exact linear redundancy (§3.3) |
| A2 | Lipid ratios: **TG/HDL**, **TC/HDL**, non-HDL | Mode B | Cleaner metabolic signal than raw TC |
| A3 | **Drop the CBC and liver groups** (7 features); keep AST/ALT ratio as an optional single feature | Mode B | ≈ 0 contribution (§6) |
| A4 | **"Preventive" variant without total cholesterol / LDL** for CVD and Diabetes; report the AUC cost explicitly | CVD B, Diabetes B | Treatment-proxy signal (§3.4); a product decision, not only a metric decision |
| A5 | **Pulse pressure** (SBP − DBP) and **MAP** | CVD, Diabetes only (never HTN) | BP shape beyond levels |
| A6 | **Waist-to-height proxy unavailable** (no height column) → waist × sex, BMI × age interaction; or keep one of BMI/waist | all | ρ 0.9 redundancy (§3.2) |
| A7 | **eGFR (CKD-EPI 2021)** from creatinine, age and sex | CVD B, HTN B, Diabetes B | Renal signal in one clinically meaningful feature |
| A8 | **One-hot** smoking and alcohol (never/former/current) instead of ordinal | LR (HTN A), all | Non-monotone categories (§3.5) |
| A9 | Missingness flags per NHANES component (exam / lab / fasting), not per feature | all | Informative missingness at component level (§4) |

**Track B — Missing data and imputation**

| ID | Experiment |
|---|---|
| B1 | Median + `MissingIndicator` (component-level flags) for RF/LR configs |
| B2 | `IterativeImputer` (BayesianRidge, ≤ 10 iterations) and `KNNImputer` (k = 15) fit inside folds |
| B3 | Native NaN (HGB/XGB) for **all** configs as a family comparison |
| B4 | **Shift stress test:** for each candidate, score development rows with labs artificially masked and report the change in mean predicted risk. Accept only candidates whose predictions do not jump when a user omits labs (guards the serving risk in §4) |

**Track C — Algorithms, regularisation, imbalance**

| ID | Experiment |
|---|---|
| C1 | RF with `min_samples_leaf ∈ {20, 40, 80}`, `max_depth ∈ {4, 6}`, `max_samples ∈ {0.5, 0.7}` (target: halve the train−val gap) |
| C2 | HGB/XGB with stronger regularisation (lr 0.02–0.05, depth 3–4, `min_child_weight`/`min_samples_leaf` 30–100, λ 1–20, early stopping on inner fold) |
| C3 | LightGBM / CatBoost **if installable without changing the pinned serving environment** (separate experiments venv or optional extra); otherwise skip |
| C4 | HTN A: **spline LR** (`SplineTransformer` on age, bmi, waist; 4 knots) + one-hot categoricals; ElasticNet (l1_ratio 0–0.5) |
| C5 | Class weighting: `None` vs `balanced` vs sqrt-balanced, evaluated **after** calibration (weighting should change ranking, not just shift probabilities) |
| C6 | Soft-voting / stacked ensemble of the best LR + best GBDT per config (stack fit on out-of-fold predictions only) |

**Track D — Calibration and thresholding**

| ID | Experiment |
|---|---|
| D1 | Sigmoid vs isotonic vs **beta calibration** vs temperature scaling, fit on out-of-fold development predictions (cross-fitted, not a single validation split) |
| D2 | ECE with **bootstrap 95% CI** and adaptive (equal-mass) bins; report calibration slope/intercept |
| D3 | **Cross-validated threshold selection:** choose the threshold on pooled OOF predictions with an explicit **sensitivity floor of 0.70 (CVD/Diabetes) / 0.70 (HTN)**, specificity ≥ 0.60, and a stability check (sensitivity SD across folds ≤ 0.05) |
| D4 | Report the sensitivity/specificity curve ±0.03 around the chosen threshold and pick the flattest point within 0.005 Youden J of the maximum |

### 8.3 Execution order and budget

1. **D2 + baseline re-evaluation under CV** (establishes the comparison point and noise level).
2. **C1/C2 + A1/A3** (largest expected gain, smallest risk).
3. **B1/B3 + A9 + B4**.
4. **A2/A5/A7/A8 + C4** (HTN A specific).
5. **A4** (preventive variant) as a parallel, explicitly reported product option.
6. **D1/D3/D4** on the shortlisted candidates.
7. **C3/C5/C6** only if steps 2–4 leave a gap.

### 8.4 V2.1 promotion criteria (all must hold, per configuration)

A candidate replaces a frozen configuration only if, under the shared repeated-CV protocol on the development set:

1. **Discrimination:** ΔROC-AUC ≥ **+0.010** with the fold-paired 95% CI excluding 0, **or** ΔPR-AUC ≥ **+0.015** with the CI excluding 0 and ΔROC-AUC ≥ 0. (+0.010 is ≈ 1–1.5 bootstrap SDs of a single-split AUC, §6.)
2. **Calibration not degraded:** ΔBrier ≤ +0.002; ECE upper 95% bound ≤ 0.05; calibration slope in [0.85, 1.15].
3. **Screening operating point:** CV-selected threshold achieves OOF sensitivity ≥ 0.70 and specificity ≥ 0.60, with fold-wise sensitivity SD ≤ 0.05; sensitivity not lower than the frozen config's under the same protocol by more than 0.02.
4. **Generalisation:** mean train−validation AUC gap ≤ 0.05 (trees) — or reduced by ≥ 30% vs frozen.
5. **Robustness:** B4 lab-masking shift ≤ 0.02 in mean predicted risk; passes the leakage-guard assertion; no feature requires data the API cannot collect.
6. **Simplicity tie-break:** if two candidates are within CI, prefer fewer features / simpler model.
7. **Single final test evaluation:** only promoted candidates are scored once on the sealed test split. Results are reported alongside (not replacing) the Stage 1F numbers, with the test-selection caveat repeated. Promotion to serving then requires the existing parity/SHAP test suite to pass against the new artifacts.

---

## 9. Limitations of this investigation

- Permutation importance and group permutation are computed on one model per config and on a 1,170-row validation split. They identify *where to look*, and Phase 2 ablations must confirm them with retraining.
- Calibration conclusions in §5 rely on Stage 1F's recorded validation comparison and on the published test numbers. No new test computation was done, by design.
- The statin interpretation in §3.4 is inferred from direction and magnitude. The dataset has no lipid-lowering-medication field to confirm it.
- Survey weights remain unused (unchanged from Stage 1F). All numbers are unweighted sample estimates.
