# ML Pipeline and Evaluation (V2, frozen)

> The V2 ML research pipeline is **frozen**. Do not modify scripts, models or results without explicit instruction.

## 1. Training and model selection

| Stage | Script | What happened |
|---|---|---|
| 1D | `benchmark_models.py` | LR, balanced LR, RF (100 trees), balanced RF, HGB, balanced HGB; fit on train, ranked by validation ROC-AUC then PR-AUC (`:91-115`, `:394`). Outputs in `backend/outputs/model_v2_baseline/`. Also creates the split files. |
| 1E-1 | `benchmark_xgboost.py` | XGBoost baseline. Outputs in `backend/outputs/model_v2_xgboost_baseline/`. |
| 1E-2 | `optimize_models.py` | `RandomizedSearchCV(n_iter=40, scoring="roc_auc", random_state=42)`, `StratifiedKFold(5, shuffle=True, random_state=42)` on the development set (train + validation) for LR, HGB, XGB (spaces `:150-211`). Winner within a configuration by mean CV ROC-AUC (`:492`). |
| 1E-3 | `optimize_random_forest.py` | Same design for RF (spaces `:112-140`, incl. `balanced` / `balanced_subsample`). |
| 1F | `calibrate_and_optimize_thresholds.py` | Loads selected model per configuration (`:102-163`), refits, calibrates, sets thresholds. |

### Final selections

| Configuration | Model | Notable hyperparameters (verified) | Source artifact (relative to `backend/ml/`) |
|---|---|---|---|
| CVD A | Optimized Random Forest | 500 trees, max_depth 8, max_features `log2` | `evaluation/stage_1e3/models/cvd_mode_a_rf_v2_optimized.joblib` |
| CVD B | Optimized Random Forest | 800 trees, max_depth 8, max_features 0.5 | `evaluation/stage_1e3/models/cvd_mode_b_rf_v2_optimized.joblib` |
| Diabetes A | HistGradientBoosting | see `stage_1e2_results.json` | `evaluation/stage_1e2/models/diabetes_mode_a_hist_gradient_boosting_v2_optimized.joblib` |
| Diabetes B | XGBoost | see `stage_1e2_results.json` | `evaluation/stage_1e2/models/diabetes_mode_b_xgboost_v2_optimized.joblib` |
| Hypertension A | Logistic Regression | median imputer + scaler + LR | `evaluation/stage_1e2/models/hypertension_mode_a_logistic_regression_v2_optimized.joblib` |
| Hypertension B | Optimized Random Forest | 800 trees, max_depth 8, max_features 0.5 | `evaluation/stage_1e3/models/hypertension_mode_b_rf_v2_optimized.joblib` |

Full hyperparameters are in `stage_1e2_results.json` and `random_forest_optimization_results.json`.

### ⚠️ Caveat: test set used in cross-stage selection

Comparisons **between** stages (1D vs 1E-2, and 1E-2 vs optimized RF) use **test-set** ROC-AUC: `optimize_models.py` computes `delta_vs_1d` from `opt_test["roc_auc"]` and issues "OPTIMIZED MODEL IS BETTER / KEEP STAGE 1D MODEL" from it; `optimize_random_forest.py:392-402` and the 1E-3 report §4 do the same. The choice of model family was therefore informed by the test set, conflicting with the "test evaluated once" statements in the Stage 1F and 1G-4 reports. No Stage 1G audit flags it. Final test metrics should be described as slightly optimistic, and this disclosed in any publication.

## 2. Calibration and thresholds (Stage 1F)

Implemented in `backend/ml/scripts/calibrate_and_optimize_thresholds.py`:

1. Base model cloned and refit on training data only (`:402-404`).
2. Raw vs sigmoid vs isotonic compared via 5-fold out-of-fold CV on validation predictions (`:279-307`). Sigmoid = logistic regression on the logit with `C=1e5`.
3. Selection rule (`:427-447`): isotonic only if it beats sigmoid by > 0.001 Brier **and** > 0.005 ECE; otherwise sigmoid if not worse than raw on Brier or better on ECE; else raw. **Sigmoid selected for all six.**
4. Threshold sweep (`:476-500`) over OOF validation probabilities, 0.05–0.95 in 91 steps: among thresholds with sensitivity ≥ 0.65 **and** specificity ≥ 0.65, maximize Youden's J (ties → F2, then F1); if none qualify, global max Youden's J.
5. Final calibrator (`:514-520`): `CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(X_val, y_val)`.
6. Serialization (`:544-548`): full calibrated estimator saved with joblib and pickle.
7. Single final evaluation on the test split.

### Locked thresholds (verified in `stage_1f_comparison.csv`)

| Configuration | Model | Calibration | Threshold |
|---|---|---|---:|
| CVD A | Optimized RF | Sigmoid | **0.13** |
| CVD B | Optimized RF | Sigmoid | **0.15** |
| Diabetes A | HistGradientBoosting | Sigmoid | **0.15** |
| Diabetes B | XGBoost | Sigmoid | **0.17** |
| Hypertension A | Logistic Regression | Sigmoid | **0.41** |
| Hypertension B | Optimized RF | Sigmoid | **0.52** |

## 3. Final test performance

Source of truth: `backend/ml/evaluation/stage_1f/stage_1f_comparison.csv` and `stage_1f_results.json`. All 60 values cross-checked. Binary metrics at the locked threshold. Test n = 1,172 / 1,172 / 1,171 / 1,171 / 1,170 / 1,170.

| Configuration | ROC-AUC | PR-AUC | Brier | ECE | Sensitivity | Specificity | Precision | NPV | F1 | Bal. Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CVD A | 0.8137 | 0.4159 | 0.0912 | 0.0125 | 0.6824 | 0.7803 | 0.3098 | 0.9444 | 0.4262 | 0.7314 |
| CVD B | 0.8218 | 0.4340 | 0.0906 | 0.0368 | 0.6081 | 0.8438 | 0.3600 | 0.9371 | 0.4523 | 0.7259 |
| Diabetes A | 0.7957 | 0.4348 | 0.1222 | 0.0250 | 0.8173 | 0.6397 | 0.3288 | 0.9419 | 0.4690 | 0.7285 |
| Diabetes B | 0.8162 | 0.4793 | 0.1203 | 0.0470 | 0.6202 | 0.8100 | 0.4135 | 0.9080 | 0.4962 | 0.7151 |
| Hypertension A | 0.7662 | 0.6541 | 0.1946 | 0.0331 | 0.7615 | 0.6647 | 0.6281 | 0.7894 | 0.6884 | 0.7131 |
| Hypertension B | 0.7731 | 0.6898 | 0.1916 | 0.0465 | 0.6253 | 0.7511 | 0.6514 | 0.7294 | 0.6380 | 0.6882 |

### Mode A vs Mode B

- Mode B improves ROC-AUC (+0.008 CVD, +0.021 Diabetes, +0.007 Hypertension), PR-AUC and (slightly) Brier for all three diseases.
- Mode B is **not** universally better: ECE is worse for all three; sensitivity drops at the locked thresholds (CVD 0.682 → 0.608, Diabetes 0.817 → 0.620, Hypertension 0.762 → 0.625) with specificity rising; balanced accuracy is lower for all three. These are different operating points, not just better models.
- CVD B and Hypertension B sensitivities (0.608, 0.625) fall below the 0.65 selection target; the constraint was met on validation but did not hold on test.
- Low precision for CVD and Diabetes (0.31–0.41) reflects low prevalence, with high NPV (0.91–0.94).
- Mode A may be preferable when sensitivity matters most; Mode B gives better-ranked probabilities.

## 4. Stage 1G audits

Reports in `backend/ml/evaluation/stage_1g/`.

| Audit | Report | Verdict | Checked | Issues |
|---|---|---|---|---|
| 1G-1 Model Integrity | `STAGE_1G_INTEGRITY_AUDIT.md` | **PASS** | Artifacts load; architectures; feature parity; split disjointness; sigmoid refit on validation; thresholds vs sweeps; test metrics recomputed within 1e-4; leakage; V1 contamination | None |
| 1G-2 Performance & Mode A/B | `STAGE_1G_PERFORMANCE_ANALYSIS.md` | **PASS** | Consolidated metrics, calibration comparison, Mode A/B, figures | None blocking |
| 1G-3 SHAP | `STAGE_1G_SHAP_ANALYSIS.md` | **PASS / LOCKED** | Explainer choice, base-model extraction, global/local explanations | Missing IEEE SHAP figures, resolved (regenerated); limitations noted |
| 1G-4 Research Readiness | `STAGE_1G-4_FINAL_RESEARCH_READINESS_AUDIT.md` | **CONDITIONAL PASS** (line 8): zero methodological blockers; pipeline frozen & verified | Provenance, targets, leakage, splits, artifacts, calibration, thresholds, metric reproduction, Mode A/B, SHAP fidelity, paper claims, reproducibility | 4 non-blocking |

1G-4 non-blocking issues: ISSUE-1 requirements missing xgboost/pyarrow — **resolved**; ISSUE-2 stale V1 docs — **resolved** (V1 documents and code were removed in Phase 5; the root `README.md` now describes V2); ISSUE-3 UI mockup shows SHAP as "+14.2%" — **open** (must not carry into V2 UI); ISSUE-4 Figures 1, 2, 5 only ASCII — **open** (publication only).

A conditional pass applies to the **research pipeline**. It is not clinical validation: no external validation, no prospective study, no regulatory review, US-based cohort.

### Report inconsistencies (not flagged by the audits)

1. Test-set use in cross-stage selection (above).
2. 1G-3 §4 gives cohort totals 7,816 / 7,810 / 7,804; actual 7,807 / 7,806 / 7,800.
3. 1G-3 §3 calls the calibrator "prefit"; code uses `FrozenEstimator`.
4. 1G-4 §7 lists Hypertension A imputation as `SimpleImputer()`; code uses `strategy="median"`.
5. 1G-4 §8 says isotonic had "higher Brier" in every case; the actual rule is margin-based (per-config values not re-verified).
6. `DATASET_MANIFEST.md`: line 6 says "Awaiting Stage 1D"; lines 94-96 describe an unused IterativeImputer; line 108 says survey weights would be used in calibration, but calibration is unweighted.

None change the reported test metrics, which were reproduced exactly.

## 5. Model artifacts and inference rules

| Artifact (`backend/ml/evaluation/stage_1f/`) | Purpose |
|---|---|
| `models/{cvd,diabetes,hypertension}_mode_{a,b}_calibrated.joblib` | **Final calibrated estimators (use these)** |
| `models/*_calibrated.pkl` | Pickle duplicates |
| `stage_1f_results.json` | Ordered `features`, threshold, calibration method, metrics |
| `stage_1f_comparison.csv` | Summary metrics |
| `calibration/*_calibration_data.csv`, `thresholds/*_threshold_sweep.csv` | Supporting evidence |

Stage 1E `*_optimized.joblib` files and `backend/outputs/model_v2_*` are intermediate, not for serving.

**Inference (implemented in `backend/api/v2/`, verified by tests):**

1. Startup loads the six `*_calibrated.joblib` files plus `features`, thresholds and offline metrics from `stage_1f_results.json` (nothing hard-coded) and asserts structure, calibrator type and feature parity (see [api-and-database.md](api-and-database.md) §2.1).
2. The configuration is chosen from `(disease, mode)`.
3. Inputs are validated; required: `age`, `gender`, `bmi`; every other field is optional and sent as `NaN`.
4. A one-row float64 DataFrame is built in exactly the stored column order using the harmonizer-equivalent encodings.
5. `p = model.predict_proba(X)[0, 1]` is **already calibrated**; no sigmoid is applied again.
6. `predicted_positive = p >= locked_threshold`; `model.predict()` (0.5 cut-off) is never used and thresholds are never replaced.
7. SHAP is computed per request on the unwrapped base estimator (see [explainability-shap.md](explainability-shap.md)).
8. Offline evaluation metrics are served separately by `GET /v2/model-metrics`, never inside prediction responses.

## 6. Inference parity verification

| Check | Result |
|---|---|
| Golden parity: ≥ 30 API-eligible sampled test rows per config (seed 42, plus the sparsest rows at encoded level) vs an independent offline load of the Stage 1F artifact | max \|api − offline\| = 7.8e-16 (limit 1e-6) |
| Aggregate: whole test split per config through the service path vs `stage_1f_results.json` | ROC-AUC, PR-AUC, Brier within 5e-5; TP/FP/FN/TN exactly equal; sensitivity, specificity, precision, NPV, F1, balanced accuracy within 5e-5 |
| Threshold boundary | `p == threshold` is positive; `predict()` never called; probability returned unmodified |
| Calibration location | calibrated output differs from the unwrapped base estimator; the API returns it verbatim (applied exactly once) |

Test-environment note: the artifacts were serialized with scikit-learn 1.9.0 and load without warnings under the pinned environment. Rows whose `bmi` is missing in NHANES (≈ 23.5%) cannot be sent through the API because `bmi` is a required field; they are covered by encoded-level parity tests.