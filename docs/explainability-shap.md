# Explainability (SHAP)

## 1. Research methodology (Stage 1G-3, implemented & verified)

Scripts live at the **repo root**: `shap_compute.py`, `shap_plot_figure3.py`.

| Aspect | Implementation |
|---|---|
| Model explained | The **base estimator** extracted from `calibrated_classifiers_[0]` and unwrapped from `FrozenEstimator` (`shap_compute.py:149-187`). For Pipeline models, `est[:-1]` transforms X and `est[-1]` is explained (`:226-247`). |
| Explainers | `shap.TreeExplainer` for RF, HGB, XGB (`:267-274`); `shap.LinearExplainer` for Hypertension A (LR) with a 200-row background sampled from the test set, `random_state=42` (`:275-279`). |
| Data | Held-out **test split** only (`:190-223`). No retraining. |
| Artifacts (root) | `shap_global_values.csv/.json`, `fig3_global_shap_feature_importance.{png,pdf,svg}`, `shap_summary_plot.png`, `shap_waterfall_plot.png` |
| Artifacts (`backend/ml/evaluation/stage_1g/`) | `figures/`, `stage_1g_shap_analysis.json`, `STAGE_1G_SHAP_ANALYSIS.md` |
| Examples (per report) | Waterfall: SEQN 130670 (Diabetes B, P = 0.539); beeswarm: 300 test participants |

Audit verdict: Stage 1G-3 **PASS / LOCKED**. The missing IEEE figures were regenerated. Noted limitations: margin space, correlated features, LinearExplainer background sampling.

## 2. Interpretation rules

- SHAP values are additive in the **base model's output space** (log-odds/margin for LR, HGB, XGB; the RF's probability output before calibration). They are **not** additive contributions to the **calibrated** probability and must **not** be shown as "+X percentage points" or "+X%" (the "+14.2%" in the old UI mockup is wrong; Stage 1G-4 ISSUE-3, open).
- SHAP describes **model behaviour, not causal medical effects**. Correlated features (e.g. BMI and waist) share attribution arbitrarily.
- Precomputed SHAP files describe **NHANES test participants**. They must never be presented as explanations of an application user.

## 3. V1 application SHAP

`/predict` (V1) uses `shap.TreeExplainer` on the V1 RandomForestRegressor's `model` step (`backend/ml/shap_explainer.py`), returns the top 6 contributors, and renders summary/bar plots from a cached 200-row sample of the V1 CSV plus a per-request waterfall (server-side `pyplot`). This is unrelated to the V2 models and remains until V1 is retired.

## 4. V2 live per-request SHAP (implemented & verified, `backend/api/v2/explain.py`)

Explanations are computed for the **submitted input only**; the service never reads `shap_global_values.*` or any precomputed NHANES explanation, never uses `matplotlib`/`pyplot` (a test enforces both), and returns numbers rather than images.

**Unwrapping** (mirrors `shap_compute.py`): `CalibratedClassifierCV → calibrated_classifiers_[0] → .estimator → FrozenEstimator → .estimator` gives either a `Pipeline` (CVD A/B, Hypertension A/B: `est[:-1]` transforms the row, `est[-1]` is explained) or a bare estimator (Diabetes A HistGradientBoosting, Diabetes B XGBoost: the raw frame is explained). Probabilities always come from the full calibrated artifact, never the unwrapped base.

| Config | Base estimator | Explainer | Values add up in |
|---|---|---|---|
| CVD A/B, Hypertension B | RandomForestClassifier | `TreeExplainer` | uncalibrated RF probability of class 1 |
| Diabetes A | HistGradientBoostingClassifier | `TreeExplainer` | log-odds (raw margin) |
| Diabetes B | XGBClassifier | `TreeExplainer` | log-odds (raw margin) |
| Hypertension A | LogisticRegression (scaled inputs) | `LinearExplainer` | log-odds |

**Verification:** at startup each explainer must satisfy `base_value + Σ shap = base model output` within 1e-5 on an all-NaN row, otherwise explanations are disabled for that config (predictions stay available; `/v2/health` reports `degraded`). The test suite checks 25+ rows per config: maximum error 2.2e-6 (XGBoost float32), others ≤ 1e-14. Values are converted to float64 so reported sums are internally consistent.

**Implementation note:** for XGBoost, SHAP updates `expected_value` after the first `shap_values` call (observed −1.5345 at construction vs −1.5177 afterwards, the latter matching XGBoost's native bias). The engine therefore reads the base value after each computation.

**Hypertension A background.** The research script uses 200 *test* rows as the `LinearExplainer` background; production must not. Verification: the scaler is fit on the training split, so the scaled training mean is zero (4e-16) and a zero background equals the full-population mean; but it deviates from a real 200-row training-split background by up to 1.7e-2 on test rows (a 200-row sample's mean differs by up to 0.125 in scaled units). Since this exceeds the 1e-6 threshold, production uses a persisted asset, `backend/api/v2/assets/hypertension_mode_a_shap_background.parquet`: 200 rows drawn with seed 42 from training-split rows (non-missing target), SEQN included. Startup rejects it unless it has exactly 200 rows, matches the registry feature order, every SEQN is `train` in `hypertension_splits.csv`, and none is in the test split. It is rebuilt with `python -m backend.api.v2.tools.build_hypertension_a_background`. SHAP's masker is given `max_samples=200` (its default silently subsamples to 100).

**Response payload** (`Explanation`): `method`, `explained_estimator`, `output_space` (`uncalibrated_rf_probability` or `log_odds_margin`), `base_value`, `sum_of_attributions`, `base_model_output`, `additive_to_calibrated_probability` (always `false`), `calibrated_probability` (echoed for context), `top_features` (default 8: `feature`, `label`, `input_value`, `was_imputed`, `shap_value`, `direction`, `rank`), `other_features_sum`, `reference_population`, `display_hint` (`relative_magnitude_only`) and a `notice`. Leakage-excluded fields (e.g. `mean_sbp` for Hypertension, `hba1c` for Diabetes B) never appear because they are not model features.

**Rendering rules for UIs:** show relative magnitude and direction only. Random Forest values look like probability points but refer to the *uncalibrated* RF output, and sigmoid calibration is non-linear, so they are not changes in the reported probability. Never label them "%" or "percentage points"; never present them as causal. Features with `was_imputed=true` were not supplied by the user (median-imputed by RF/LR pipelines, native NaN handling for HGB/XGB) and should be shown as such.

**Performance (measured):** 3-disease Mode B request with SHAP: p50 222 ms, p95 261 ms; without SHAP p50 198 ms (SHAP adds ≈ 25 ms; forest prediction dominates). A per-config lock serializes SHAP calls; 8 concurrent requests match serial results. `include_explanation` stays `true` by default.