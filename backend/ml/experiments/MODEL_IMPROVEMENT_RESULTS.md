# Phase 2 — Controlled Experiments: Results and V2.1 Recommendation

> **Bottom line:** none of the 30+ experimental variants delivered a reliable discrimination gain over the frozen V2 models, and **no configuration meets the V2.1 promotion criteria**. The one consistent, actionable finding is **operating-point related, not model related**. For CVD B and Diabetes B, thresholds re-selected by cross-validation recover the screening sensitivity that the single-split Stage 1F thresholds lost. Even that reproduces on test only for CVD B. The recommendation is **No-Go for a V2.1 model swap** in all six configurations. A separately reviewed **threshold recalibration** is the only candidate change, and only for CVD B (§7).
>
> All code: `backend/ml/experiments/scripts/phase2_experiments.py`. Raw results: `backend/ml/experiments/results/stage1_*.json`, `stage2_*.json`, `final_test.json`, `phase2_run.log`. Refit candidate models: `backend/ml/experiments/models/` (research artifacts, not served). Nothing outside `backend/ml/experiments/` was modified.

---

## 1. Protocol and safeguards

| Item | Implementation |
|---|---|
| Development set | Frozen `train ∪ validation` rows (CVD 6,635 · Diabetes 6,635 · HTN 6,630) from the frozen split files; test rows never loaded before the final step |
| Resampling | `RepeatedStratifiedKFold(5 × 3, random_state=42)`, identical folds for every candidate (paired comparisons) |
| Fitting discipline | Imputers, scalers, models and calibrators are fit inside each training fold. Feature engineering is **row-local** (ratios, eGFR, flags, one-hots; no fitted state), so it is applied before CV without leakage |
| Stage 1 (screening) | Uncalibrated out-of-fold ROC-AUC / PR-AUC and train−val AUC gap for every Track A/B/C candidate |
| Stage 2 (full) | Per-fold calibrators fit on **inner 3-fold OOF** predictions of the training fold. Sigmoid, isotonic, beta and temperature calibration are compared on the outer fold. Threshold = max Youden J on pooled OOF calibrated predictions subject to **sensitivity ≥ 0.70 and specificity ≥ 0.60**. All 10 metrics per fold. B4 lab-masking shift |
| Uncertainty | Fold-paired Δ vs the frozen configuration re-run under the same protocol, with **Nadeau–Bengio corrected repeated-CV t intervals** (95%). `*` = interval excludes 0 |
| Frozen baseline under CV | `C0_frozen_clone` = `sklearn.clone` of each frozen Stage 1F base estimator (identical hyperparameters) on the frozen feature list, run through the same Stage 1/2 protocol |
| Final test | Run once, after all selection was finished, for one experimental candidate per config plus the two preventive variants. Stage 1F protocol: base refit on `train`, calibrator fit on `validation`, **threshold fixed from Stage 2 CV** (not re-tuned) |
| Environment | Existing pinned `.venv` only (scikit-learn 1.9.0, xgboost 3.4.1). No LightGBM/CatBoost (Track C3 skipped as approved); `requirements.txt` untouched |

**Leakage guard (verified).** `make_features()` calls `guard()` on every feature matrix. It asserts that no survey or target column is present and that no raw or derived feature draws on the disease's excluded columns (CVD: MCQ160B–F, DIQ050/070, BPQ101D; Diabetes: hba1c, fasting_glucose, DIQ010/050/070, BPQ101D; HTN: mean_sbp, mean_dbp, BPQ020, BPQ101D). Each derived feature is mapped to its sources: pulse pressure and MAP come from SBP/DBP and are therefore never built for HTN; the exam-missingness flag uses `mean_pulse`, not SBP, so it is valid for HTN. No glucose- or HbA1c-derived feature exists for Diabetes. The guard ran on every one of the ~2,500 fold fits without a failure.

**Reproduction check.** Refitting the frozen configuration on `train` with a validation-fit sigmoid reproduced the published Stage 1F test ROC-AUC and PR-AUC exactly (CVD A 0.8137 / 0.4159, CVD B 0.8218 / 0.4340) and Brier to 1e-4. The harness is therefore faithful to Stage 1F.

**ECE comparability caveat.** This harness computes ECE with 10 equal-width bins. Stage 1F's ECE uses a different binning, so the same CVD A model gives 0.0181 here vs 0.0125 published. **ECE is only compared within this report** (CV columns, and experimental-vs-refit-frozen). Cross-report calibration comparisons use Brier.

---

## 2. Track C — models and regularisation (Stage 1, frozen features)

Mean ± SD over 15 folds; Δ vs frozen clone with 95% CI.

| Config | Frozen clone AUC (gap) | Best regularised single model | ΔAUC [95% CI] | ΔPR-AUC [95% CI] | Its gap |
|---|---|---|---|---|---|
| CVD A | 0.8120 ± 0.017 (**0.119**) | RF leaf 20, depth 6, max_samples 0.7 | −0.0014 [−0.0054, +0.0027] | −0.0092 [−0.026, +0.007] | **0.053** |
| CVD B | 0.8308 ± 0.018 (0.090) | RF leaf 20, depth 6, 0.7 | −0.0012 [−0.0045, +0.0021] | −0.0000 [−0.009, +0.009] | **0.050** |
| Diabetes A | 0.8005 ± 0.009 (0.100) | XGB depth 4, mcw 30, λ 5 | +0.0016 [−0.0009, +0.0040] | +0.0073 [−0.007, +0.021] | 0.059 |
| Diabetes B | 0.8366 ± 0.012 (0.086) | XGB depth 4, mcw 30, λ 5 | −0.0013 [−0.0054, +0.0027] | −0.0006 [−0.011, +0.010] | 0.061 |
| HTN A | 0.7821 ± 0.010 (0.002) | XGB depth 3, mcw 50, λ 10 | **+0.0042 [+0.0001, +0.0084]\*** | **+0.0194 [+0.008, +0.031]\*** | 0.027 |
| HTN B | 0.7954 ± 0.012 (0.083) | XGB depth 3, mcw 50, λ 10 | +0.0015 [−0.0024, +0.0053] | +0.0019 [−0.006, +0.010] | 0.035 |

Further observations:

- **Regularisation reduced the overfitting gap by 40–80%** (CVD A 0.119 → 0.053; RF leaf 80/depth 4: 0.020) **without improving validation discrimination**. Heavier regularisation (leaf 80, depth 4) significantly *reduced* AUC for CVD B (−0.009\*), Diabetes A (−0.018\*), Diabetes B (−0.035\*) and HTN B (−0.007\*). The Phase 1 gaps were real, but they reflect memorising training rows; they were not costing out-of-sample performance.
- **HTN A (logistic regression) is the only config a different model family helps.** A shallow XGBoost captures non-linearity, with a small but significant gain. Spline LR (+0.0016, CI includes 0) and ElasticNet (+0.0010) did not reach significance.
- **Ensembles (C6, rank-average of best LR and best GBDT).** HTN A +0.0049\* AUC / +0.0179\* PR; Diabetes A +0.0046 [−0.0001, +0.0093]; others ≈ 0. These were not carried forward: the gain is no larger than the single XGBoost for HTN A, and an ensemble needs two models plus SHAP aggregation in serving.
- **Class weighting (C5, evaluated after calibration, Stage 2)** never helped. `balanced` reduced ROC-AUC in CVD A (−0.0094\*), CVD B (−0.0059\*) and Diabetes B (−0.0057\*) and worsened Brier. Elsewhere it was neutral. It only moves the operating point, which threshold selection already controls.

---

## 3. Track B — missing data (Stage 1, frozen features)

| Config | RF median (frozen style) | RF median + indicators | RF iterative | LR median | LR median + indicators | HGB native NaN |
|---|---|---|---|---|---|---|
| CVD A | 0.8084 | 0.8086 | 0.8082 (PR −0.024\*) | 0.8117 | 0.8094 | 0.8086 |
| CVD B | 0.8281 | 0.8284 | 0.8254 | 0.8261 | 0.8236 | 0.8287 |
| Diabetes A | 0.7917 | 0.7919 | 0.7937 | 0.7984 | 0.7976 | 0.8001 |
| Diabetes B | 0.8145 | 0.8145 | 0.8139 | 0.8265 | 0.8258 | 0.8313 |
| HTN A | 0.7842 | 0.7843 | 0.7829 | 0.7821 | 0.7831 | 0.7845 |
| HTN B | 0.7930 | 0.7927 | 0.7917 | 0.7911 | 0.7917 | 0.7956 |

- **Missingness indicators changed AUC by ≤ 0.0025 everywhere.** Iterative imputation was never better and was significantly worse in PR-AUC for CVD A. The informative missingness found in Phase 1 (indicator-only AUC ≤ 0.56) is apparently already absorbed by the existing features, especially age.
- **Native NaN handling (HGB) is the best imputation choice for Diabetes A/B and HTN B**, which supports the frozen choice of native-NaN GBDTs for Diabetes. Its advantage over median-imputed RF on Diabetes B (+0.017) is a model-family effect and is not reproducible with RF + indicators.

### B4 — lab-masking shift stress test (Mode B, Stage 2)

All 16 raw lab columns of each outer-fold row were set to missing (and derived lab features recomputed). The sigmoid-calibrated prediction was compared with the unmasked prediction from the same fold model. The §8.4 limit is |mean shift| ≤ 0.02.

| Config | Candidate | Mean shift (masked − actual) | Mean \|shift\| per person | Passes ≤ 0.02? |
|---|---|---:|---:|---|
| CVD B | frozen | −0.019 ± 0.003 | 0.038 | borderline pass |
| CVD B | preventive | −0.022 ± 0.003 | 0.027 | **fail** (marginal) |
| Diabetes B | frozen | −0.019 ± 0.008 | 0.065 | borderline pass |
| Diabetes B | best (A1) | −0.019 ± 0.008 | 0.065 | borderline pass |
| Diabetes B | preventive | −0.021 ± 0.007 | 0.056 | **fail** (marginal) |
| HTN B | frozen (RF) | −0.021 ± 0.003 | 0.046 | **fail** (marginal) |
| HTN B | best (XGB) | **+0.008** ± 0.006 | 0.052 | pass |

Every Mode B model **systematically lowers risk when labs are omitted**, by about 2 points on average and 3–7 points per individual. A user who skips labs looks like the healthier NHANES "did not attend the lab" group. This is a **serving risk for the current frozen models**, independent of any V2.1 decision. The UI should steer users without labs to Mode A, or the API should refuse Mode B when most lab fields are absent (see §7).

---

## 4. Track A — feature engineering and pruning (Stage 1)

Reference model per config = best Track C model (frozen clone for CVD A/B and Diabetes B; XGB for Diabetes A and HTN A/B). ΔAUC vs frozen clone with 95% CI.

| Variant | CVD A | CVD B | Diabetes A | Diabetes B | HTN A | HTN B |
|---|---|---|---|---|---|---|
| A0 frozen features (ref model) | 0 | 0 | +0.0016 | 0 | +0.0042\* | +0.0015 |
| A1 drop LDL, add non-HDL | — | −0.0001 | — | +0.0002 | — | +0.0013 |
| A2 + TG/HDL, TC/HDL | — | −0.0004 | — | +0.0002 | — | +0.0012 |
| A3 drop CBC + liver (7 labs) | — | −0.0020 | — | **−0.0041\*** | — | +0.0011 (PR +0.004) |
| A5 pulse pressure + MAP | −0.0016 | +0.0000 | +0.0025 | +0.0005 | n/a (leakage) | n/a |
| A7 CKD-EPI 2021 eGFR | — | +0.0002 | — | +0.0001 | — | +0.0014 |
| A8 one-hot smoking & alcohol | −0.0013 | −0.0001 | +0.0018 | +0.0000 | +0.0043\* | +0.0014 |
| A9 component missingness flags | −0.0001 | +0.0006 | +0.0012 | −0.0001 | +0.0045\* | +0.0016 |
| A combo | −0.0018 | −0.0015 | +0.0023 | −0.0020 | +0.0042 | +0.0009 |
| **A4 preventive (no TC, no LDL)** | n/a | **−0.0143\*** | n/a | **−0.0101\*** | n/a | n/a |
| A4 preventive combo | n/a | −0.0162\* | n/a | −0.0122\* | n/a | n/a |

(HTN A starred values are the XGB-vs-LR family effect from §2, not a feature effect. A8/A9 add ≤ 0.0003 on top.)

- **No engineered feature produced a significant gain anywhere.** Clinically motivated ratios, eGFR, pulse pressure, one-hot recoding and missingness flags all lie within ±0.003 of the frozen features.
- **The redundant LDL column (Phase 1 §3.3) can be dropped at no cost** (A1: ±0.0002). That is a simplification, not an improvement.
- **CBC + liver pruning is not free for Diabetes B** (−0.0041\*), even though Phase 1 permutation importance suggested ≈ 0 contribution. Retraining reveals small redundant contributions. CVD B and HTN B are unaffected (CI includes 0).
- **The preventive variant (A4) costs about 0.010–0.014 ROC-AUC and 0.024–0.025 PR-AUC in CV** (all significant). This quantifies the treatment-proxy signal in total cholesterol identified in Phase 1.

---

## 5. Track D — calibration and stable thresholds (Stage 2)

### 5.1 Calibration method comparison (CV mean Brier / ECE, sigmoid vs best alternative)

| Config (candidate) | Sigmoid | Isotonic | Beta | Temperature | Chosen |
|---|---|---|---|---|---|
| CVD A (frozen) | 0.0932 / 0.018 | 0.0936 / 0.018 | 0.0932 / 0.018 | 0.0937 / 0.021 | sigmoid |
| CVD B (frozen) | 0.0902 / 0.018 | 0.0904 / 0.019 | 0.0902 / 0.018 | 0.0903 / 0.020 | sigmoid |
| Diabetes A (best) | 0.1195 / 0.020 | 0.1196 / 0.018 | 0.1196 / 0.018 | 0.1197 / 0.022 | sigmoid |
| Diabetes B (best) | 0.1101 / 0.019 | 0.1104 / 0.021 | 0.1101 / 0.019 | 0.1102 / 0.020 | sigmoid |
| HTN A (frozen) | 0.1874 / 0.031 | 0.1871 / 0.025 | 0.1871 / 0.026 | 0.1901 / 0.056 | sigmoid\* |
| HTN B (frozen) | 0.1821 / 0.026 | 0.1827 / 0.027 | 0.1821 / 0.026 | 0.1821 / 0.026 | sigmoid |

\*Isotonic lowered HTN A ECE by 0.006 but Brier by only 0.0003, below the 0.001 Brier margin required.

**Sigmoid remains the right choice for all six** (selection rule: an alternative must win by > 0.001 Brier **and** > 0.005 ECE). Beta calibration ties sigmoid; temperature scaling is worse or equal; isotonic never wins on Brier. This independently re-confirms the Stage 1F calibration decision with a stronger, cross-fitted protocol. CV calibration slopes are all 1.01–1.07.

### 5.2 Cross-validated thresholds vs the locked Stage 1F thresholds

| Config | Locked (Stage 1F, one validation split) | CV-selected (pooled OOF) | CV sensitivity ± fold SD | CV specificity |
|---|---:|---:|---|---:|
| CVD A | 0.13 | 0.12 | 0.819 ± 0.042 | 0.665 |
| CVD B | 0.15 | **0.12** | 0.839 ± 0.039 | 0.685 |
| Diabetes A | 0.15 | 0.155 (frozen) / 0.165 (best) | 0.807 ± 0.029 | 0.641 |
| Diabetes B | 0.17 | **0.19** (frozen) / 0.17 (A1) | 0.751 ± 0.035 / 0.790 ± 0.029 | 0.753 / 0.719 |
| HTN A | 0.41 | 0.40 (frozen) / 0.43 (XGB) | 0.790 ± 0.018 | 0.658 |
| HTN B | 0.52 | **0.44** (frozen) / 0.445 (XGB) | 0.762 ± 0.017 | 0.689 |

- Fold-wise sensitivity SDs are 0.017–0.042, all within the §8.4 limit of ≤ 0.05.
- **The largest disagreements with the locked thresholds are CVD B (0.15 → 0.12) and HTN B (0.52 → 0.44).** In both, the locked threshold sits higher than the cross-validated optimum. That matches the CVD B and HTN B test sensitivities of 0.61 and 0.63, which fell below the 0.65 target.

---

## 6. Final held-out test comparison (evaluated once)

Frozen = published Stage 1F values. Experimental = the Stage 2 winner per config, refit with the Stage 1F protocol and using its CV-fixed threshold. ECE uses different binning in the two rows (§1), so compare Brier for calibration.

| Config | Model | Thr | ROC-AUC | PR-AUC | Brier | ECE | Sens | Spec | Prec | NPV | F1 | Bal.Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CVD A** | Frozen V2 (RF-500) | 0.13 | 0.8137 | 0.4159 | 0.0912 | 0.0125† | 0.6824 | 0.7803 | 0.3098 | 0.9444 | 0.4262 | 0.7314 |
| | Experimental = frozen model, CV threshold | 0.12 | 0.8137 | 0.4159 | 0.0913 | 0.0181 | **0.7568** | 0.6797 | 0.2545 | 0.9508 | 0.3810 | 0.7182 |
| **CVD B** | Frozen V2 (RF-800) | 0.15 | 0.8218 | 0.4340 | 0.0906 | 0.0368† | 0.6081 | 0.8438 | 0.3600 | 0.9371 | 0.4523 | 0.7259 |
| | Experimental = frozen model, CV threshold | 0.12 | 0.8218 | 0.4340 | 0.0896 | 0.0133 | **0.8243** | 0.6943 | 0.2805 | 0.9647 | 0.4185 | **0.7593** |
| | **Preventive** (RF-800, no TC/LDL) | 0.125 | 0.8268 | 0.4587 | 0.0888 | 0.0248 | 0.8378 | 0.6680 | 0.2672 | 0.9661 | 0.4052 | 0.7529 |
| **Diabetes A** | Frozen V2 (HGB) | 0.15 | 0.7957 | 0.4348 | 0.1222 | 0.0250† | 0.8173 | 0.6397 | 0.3288 | 0.9419 | 0.4690 | 0.7285 |
| | Experimental (XGB d4 + A combo) | 0.165 | 0.8016 | 0.4522 | 0.1201 | 0.0241 | 0.8029 | 0.6552 | 0.3347 | 0.9390 | 0.4724 | 0.7291 |
| **Diabetes B** | Frozen V2 (XGB) | 0.17 | 0.8162 | 0.4793 | 0.1203 | 0.0470† | 0.6202 | 0.8100 | 0.4135 | 0.9080 | 0.4962 | 0.7151 |
| | Experimental (frozen XGB, LDL→non-HDL) | 0.17 | 0.8138 | 0.4657 | 0.1185 | 0.0184 | 0.7596 | 0.7092 | 0.3607 | 0.9318 | 0.4892 | 0.7344 |
| | **Preventive** (XGB, no TC/LDL) | 0.165 | 0.8129 | 0.4750 | 0.1178 | 0.0244 | 0.7885 | 0.6864 | 0.3519 | 0.9376 | 0.4866 | 0.7374 |
| **HTN A** | Frozen V2 (LR) | 0.41 | 0.7662 | 0.6541 | 0.1946 | 0.0331† | 0.7615 | 0.6647 | 0.6281 | 0.7894 | 0.6884 | 0.7131 |
| | Experimental (XGB d3 + miss. flags) | 0.43 | 0.7588 | 0.6580 | 0.1973 | 0.0370 | 0.7395 | 0.6677 | 0.6233 | 0.7751 | 0.6764 | 0.7036 |
| **HTN B** | Frozen V2 (RF-800) | 0.52 | 0.7731 | 0.6898 | 0.1916 | 0.0465† | 0.6253 | 0.7511 | 0.6514 | 0.7294 | 0.6380 | 0.6882 |
| | Experimental (XGB d3) | 0.445 | 0.7679 | 0.6711 | 0.1933 | 0.0538 | 0.7355 | 0.7019 | 0.6473 | 0.7811 | 0.6886 | 0.7187 |

† Stage 1F ECE binning; not comparable to the other ECE values in this table.

Reading the table:

- **Discrimination:** the test confirms the CV conclusion. The only experimental model gains are Diabetes A (+0.006 AUC, +0.017 PR-AUC; inside CV noise, CV Δ +0.0023 with CI including 0). The CV-significant HTN A gain did **not** reproduce (test −0.007 AUC). HTN B and Diabetes B experimental models are slightly worse. With 1,170 test rows the AUC SD is about 0.012, so all of these are within noise.
- **CVD A/B "experimental" rows are the frozen models themselves** (identical ROC-AUC/PR-AUC, which is a reproduction check) evaluated at the CV threshold. For CVD B, the 0.12 threshold raises test sensitivity from 0.608 to **0.824** and balanced accuracy from 0.726 to 0.759, at the cost of specificity (0.844 → 0.694) and precision (0.360 → 0.281). For CVD A, sensitivity rises by +0.074 while balanced accuracy falls slightly.
- **Preventive variants on test:** CVD B preventive scored *higher* than the standard model (AUC 0.8268 vs 0.8218, PR 0.4587 vs 0.4340). Diabetes B preventive was −0.003 AUC / −0.004 PR vs the frozen model. This **contradicts the CV estimate** (−0.014 / −0.010 AUC, both significant). The CV estimate rests on 6,635 participants × 15 folds and is far more reliable than one 1,172-row test split (AUC SD ≈ 0.015). The correct reading is that the preventive variant **costs about 0.01 AUC in expectation**, and on this particular test split that cost was hidden by sampling noise.
- **The HTN A and HTN B experimental XGB models have poor test calibration slopes (0.83–0.84)** vs the frozen models. That is another sign that the CV-significant HTN gains were within the noise envelope.

---

## 7. V2.1 Go/No-Go (against §8.4 of the investigation)

Criteria: (1) ΔAUC ≥ +0.010, or ΔPR ≥ +0.015 with ΔAUC ≥ 0, CI excluding 0; (2) ΔBrier ≤ +0.002, ECE upper 95% bound ≤ 0.05, slope 0.85–1.15; (3) OOF sens ≥ 0.70, spec ≥ 0.60, sens SD ≤ 0.05, sensitivity not more than 0.02 below frozen; (4) gap ≤ 0.05 or −30%; (5) B4 shift ≤ 0.02, leakage guard; (6) simplicity tie-break; (7) single test confirmation.

| Config | Best candidate | (1) Discrim. | (2) Calib. | (3) Op. point | (4) Gap | (5) B4 / leak | Test (7) | **Decision** |
|---|---|---|---|---|---|---|---|---|
| CVD A | = frozen | ✗ (no better model) | — | — | — | — | — | **No-Go** (keep frozen) |
| CVD B | = frozen | ✗ | — | — | — | borderline | — | **No-Go for model change.** Threshold review recommended (below) |
| Diabetes A | XGB d4 + combo | ✗ (+0.0023, CI ∋ 0) | ✓ | ✗ (sens −0.027\*) | ✗ (0.060, −40% ✓) | n/a / ✓ | +0.006 AUC (noise) | **No-Go** |
| Diabetes B | frozen XGB, LDL→non-HDL | ✗ (+0.0002) | ✓ | ✓ (sens +0.040\*) | ✗ | borderline / ✓ | −0.002 AUC | **No-Go** (optional cosmetic simplification: drop LDL) |
| HTN A | XGB d3 + flags | **✓ PR +0.0195\*** (AUC +0.0045\*) | ✓ (Brier −0.0019\*) | ✗ (sens −0.025\* > 0.02) | ✓ (0.027) | n/a / ✓ | ✗ −0.007 AUC, slope 0.83 | **No-Go** (failed criterion 3 and test confirmation) |
| HTN B | XGB d3 | ✗ (+0.0015) | ✓ | ✗ (sens −0.018\*, within limit) | ✓ (0.035) | ✓ (+0.008) | −0.005 AUC, slope 0.84 | **No-Go** |
| CVD B preventive | RF-800 no TC/LDL | ✗ (−0.0143\* by design) | ✓ | ✓ | ✗ | ✗ (−0.022) | +0.005 AUC (noise) | **Not a V2.1 replacement**; viable product option (below) |
| Diabetes B preventive | XGB no TC/LDL | ✗ (−0.0101\* by design) | ✓ | ✓ | ✗ | ✗ (−0.021) | −0.003 AUC | Same as above |

### Recommendations

1. **Keep all six frozen V2 models. No V2.1 model promotion.** Eight experiment families, about 40 variants and 2,500+ fold fits found no model or feature change that beats the frozen pipeline by a margin outside the noise of this dataset size. The frozen Stage 1D–1F selection was close to the attainable ceiling for these features.
2. **Threshold review for CVD B (and secondarily HTN B), handled as its own controlled change.** Cross-validated thresholds are more stable than the single-split Stage 1F choice. For CVD B, moving 0.15 → 0.12 restores the target sensitivity on test (0.608 → 0.824; balanced accuracy +0.033) using the same frozen model. This is a change to a locked research value, so it needs explicit approval and a re-run of the Stage 1G-style audit. It must not be adopted silently.
3. **Mitigate the lab-omission shift in serving (any version).** Mode B predictions fall by about 2 points on average when labs are missing. Proposed (not implemented): warn or disallow Mode B when fewer than about 4 core labs are supplied, and direct the user to Mode A.
4. **Preventive Mode B (no total cholesterol / LDL) is an honest product option, not a performance upgrade.** It costs about 0.010–0.014 ROC-AUC and 0.024 PR-AUC in CV. In exchange it removes the statin-driven inverted cholesterol signal, so estimates for untreated screening users are not driven by "low cholesterol means already treated". If adopted, it should be versioned and labelled as a distinct configuration, not presented as an improvement.
5. **Optional simplification:** drop the Friedewald-derived `ldl_cholesterol` from Mode B (no measurable cost). Defer it until a model rebuild is needed for another reason.

---

## 8. Limitations

- One cohort (NHANES 2021–2023, unweighted). The "near the ceiling" conclusion applies to these features and this sample size, not to the diseases in general.
- The test split remains a single 1,170-row sample (AUC SD ≈ 0.012–0.015). The test table is a confirmation check, not a precise ranking. CV deltas are the primary evidence.
- Stage 1 screened on uncalibrated AUC, and only the top candidates were taken to Stage 2. A candidate with weaker ranking but better calibration could have been missed; Stage 2's calibration results make this unlikely.
- The Nadeau–Bengio correction is approximate. Some CIs with lower bounds near 0 (e.g. HTN A +0.0001) should be read as borderline.
- C3 (LightGBM/CatBoost) was skipped by decision. Given how flat the results are across RF/HGB/XGB/LR, a large gain from another GBDT implementation is unlikely, but it was not tested.
