from __future__ import annotations

import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap

from backend.api.v2 import config, explain, service
from backend.api.v2.preprocessing import encode_inputs
from backend.api.v2.schemas import ModeBInputs, PredictRequestB
from conftest import PROCESSED, load_split_frame, sample_rows

pytestmark = pytest.mark.artifacts

CONFIGS = [f"{d}_mode_{m}" for d in ("cvd", "diabetes", "hypertension") for m in ("a", "b")]
V2_DIR = Path(config.__file__).parent
MODE_B_INPUTS = {
    "age": 58, "gender": "male", "bmi": 31.2, "waist_circumference": 104, "mean_sbp": 138, "mean_dbp": 86,
    "mean_pulse": 74, "education_level": 3, "poverty_income_ratio": 2.1, "smoking_status": "former",
    "alcohol_frequency": "current", "physical_activity_level": "low", "sedentary_minutes": 420,
    "hba1c": 6.1, "fasting_glucose": 112, "total_cholesterol": 214, "hdl_cholesterol": 41, "triglycerides": 180,
    "ldl_cholesterol": 135, "serum_creatinine": 0.95, "blood_urea_nitrogen": 15, "serum_uric_acid": 6.1,
    "alt_enzyme": 28, "ast_enzyme": 24, "hemoglobin": 14.2, "wbc_count": 7.1, "platelet_count": 250,
    "rdw": 13.5, "serum_albumin": 4.2,
}


def _bundle(registry, key):
    return next(b for b in registry.all() if b.config_key == key)


@pytest.mark.parametrize("key", CONFIGS)
def test_additivity_in_base_model_output_space(registry, key):
    """base_value + sum(shap) == the base model's own output, within 1e-5, on 25 test rows (+ sparse rows)."""
    b = _bundle(registry, key)
    rows = sample_rows(key, n=25)
    worst = 0.0
    for _, row in rows.iterrows():
        x = pd.DataFrame([row[list(b.features)].to_numpy(dtype=float)], columns=list(b.features))
        worst = max(worst, b.explainer.additivity_error(x))
    print(f"\nadditivity {key}: max error {worst:.2e}")
    assert worst <= 1e-5
    expected_space = "uncalibrated_rf_probability" if b.explainer.explained_estimator == "RandomForestClassifier" else "log_odds_margin"
    assert b.explainer.output_space == expected_space
    assert b.explainer.method == ("LinearExplainer" if key == "hypertension_mode_a" else "TreeExplainer")


def test_explanation_payload_semantics(registry):
    req = PredictRequestB(mode="B", inputs=MODE_B_INPUTS)
    resp = service.predict(registry, req)
    assert [r.disease.value for r in resp.results] == ["cvd", "diabetes", "hypertension"]
    for r in resp.results:
        e = r.explanation
        assert e is not None and r.explanation_unavailable_reason is None
        assert e.additive_to_calibrated_probability is False
        assert e.calibrated_probability == r.probability
        assert len(e.top_features) == config.TOP_K_FEATURES
        assert [f.rank for f in e.top_features] == list(range(1, config.TOP_K_FEATURES + 1))
        mags = [abs(f.shap_value) for f in e.top_features]
        assert mags == sorted(mags, reverse=True)
        total = sum(f.shap_value for f in e.top_features) + e.other_features_sum
        assert total == pytest.approx(e.sum_of_attributions, abs=1e-9)
        assert e.base_value + e.sum_of_attributions == pytest.approx(e.base_model_output, abs=1e-12)
        assert "percent" not in json.dumps(e.model_dump()).lower().replace("percentage points", "")
        for f in e.top_features:
            assert f.was_imputed == (f.input_value is None)
    # leakage exclusions never appear as attributions
    htn = next(r for r in resp.results if r.disease.value == "hypertension").explanation
    assert not {"mean_sbp", "mean_dbp"} & {f.feature for f in htn.top_features}
    dm = next(r for r in resp.results if r.disease.value == "diabetes").explanation
    assert not {"hba1c", "fasting_glucose"} & {f.feature for f in dm.top_features}


def test_imputed_inputs_are_flagged_and_missing_heavy_rows_explain(registry):
    req = PredictRequestB(mode="B", inputs={"age": 60, "gender": "female", "bmi": 27}, diseases=["cvd"])
    e = service.predict(registry, req).results[0].explanation
    assert e is not None
    assert all(f.was_imputed for f in e.top_features if f.feature not in ("age", "gender", "bmi"))


# --------------------------------------------------- Hypertension Mode A LinearExplainer
def _htn_a_parts(registry):
    b = _bundle(registry, "hypertension_mode_a")
    est = explain.unwrap_base_estimator(b.model)
    pre, clf = explain.split_pipeline(est)
    return b, pre, clf


def test_htn_a_background_asset_is_training_split_only(registry):
    b = _bundle(registry, "hypertension_mode_a")
    bg = pd.read_parquet(config.HTN_A_BACKGROUND_FILE)
    sp = pd.read_csv(config.SPLITS_DIR / "hypertension_splits.csv")
    assert len(bg) == config.HTN_A_BACKGROUND_SIZE and bg["SEQN"].is_unique
    assert set(sp.loc[sp["SEQN"].isin(bg["SEQN"]), "split"]) == {"train"}
    assert not set(bg["SEQN"]) & set(sp.loc[sp["split"] == "test", "SEQN"])
    assert list(bg.columns) == ["SEQN", *b.features]
    # reproducible from the frozen data with seed 42
    df = pd.read_parquet(PROCESSED / "nhanes_hypertension_mode_a.parquet").dropna(subset=["target_hypertension"])
    train = df[df["SEQN"].isin(set(sp.loc[sp["split"] == "train", "SEQN"]))]
    again = train.sample(n=config.HTN_A_BACKGROUND_SIZE, random_state=config.HTN_A_BACKGROUND_SEED)
    assert list(again["SEQN"]) == list(bg["SEQN"])


def test_htn_a_loader_rejects_test_rows(registry, monkeypatch, tmp_path):
    b = _bundle(registry, "hypertension_mode_a")
    test_df = load_split_frame("hypertension_mode_a", "test").head(config.HTN_A_BACKGROUND_SIZE)
    bad = test_df[["SEQN", *b.features]]
    p = tmp_path / "bg.parquet"
    bad.to_parquet(p)
    monkeypatch.setattr(config, "HTN_A_BACKGROUND_FILE", p)
    with pytest.raises(ValueError, match="training split"):
        explain.load_htn_a_background(list(b.features))


def test_htn_a_zero_background_premise_vs_training_background(registry, record_property):
    """Plan premise: scaler fit on train => scaled train mean 0 => zero background == population mean.
    Verified against the real 200-row training background: the premise holds for the *population*,
    but a 200-row sample's mean differs, so production uses the persisted sample (asset)."""
    b, pre, clf = _htn_a_parts(registry)
    train = load_split_frame("hypertension_mode_a", "train")
    x_train = pre.transform(train[list(b.features)])
    mean_abs = float(np.abs(x_train.mean(axis=0)).max())
    assert mean_abs < 1e-9  # premise: scaled training mean is zero

    bg_raw = explain.load_htn_a_background(list(b.features))
    bg = pre.transform(bg_raw)
    test_rows = sample_rows("hypertension_mode_a", n=50)
    x = pre.transform(test_rows[list(b.features)])

    zero_vals = np.asarray(shap.LinearExplainer(clf, shap.maskers.Independent(np.zeros((1, bg.shape[1])), max_samples=1)).shap_values(x))
    asset_vals = np.asarray(b.explainer._explainer.shap_values(x))
    deviation = float(np.abs(zero_vals - asset_vals).max())
    record_property("htn_a_zero_vs_asset_max_abs_diff", deviation)
    record_property("htn_a_bg_scaled_mean_max_abs", float(np.abs(bg.mean(axis=0)).max()))
    print(f"\nHTN-A zero-background vs 200-row training background: max |diff| = {deviation:.4e} "
          f"(1e-6 threshold {'MET' if deviation <= 1e-6 else 'NOT MET -> persisted asset used'}); "
          f"scaled train mean max = {mean_abs:.2e}; 200-row bg mean max = {np.abs(bg.mean(axis=0)).max():.3f}")
    # The persisted-asset path is what production uses; it must equal the closed form with the bg mean.
    closed = clf.coef_[0] * (x - bg.mean(axis=0))
    assert np.abs(closed - asset_vals).max() <= 1e-6
    assert deviation > 1e-6, "premise now holds within 1e-6: asset could be dropped"  # documents the finding


# ----------------------------------------------------------------------- engineering
def test_concurrent_requests_match_serial(registry):
    req = PredictRequestB(mode="B", inputs=MODE_B_INPUTS)

    def run(_):
        r = service.predict(registry, req)
        return [(x.probability, [(f.feature, f.shap_value) for f in x.explanation.top_features]) for x in r.results]

    def close(a, b):  # RF thread-summation order differs at ~1e-17, so compare with a tolerance
        return all(pa == pytest.approx(pb, abs=1e-9) and [f for f, _ in sa] == [f for f, _ in sb]
                   and [v for _, v in sa] == pytest.approx([v for _, v in sb], abs=1e-9)
                   for (pa, sa), (pb, sb) in zip(a, b))

    serial = run(0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(run, range(16)))
    assert all(close(o, serial) for o in outs)


def test_v2_never_uses_pyplot_or_precomputed_shap():
    import ast

    for py in V2_DIR.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                      and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any("matplotlib" in a.name for a in node.names), py.name
            elif isinstance(node, ast.ImportFrom):
                assert "matplotlib" not in (node.module or ""), py.name
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                assert not re.search(r"shap_global_values|shap_summary_plot|shap_waterfall", node.value), (py.name, node.value)


def test_three_disease_mode_b_latency(registry):
    req = PredictRequestB(mode="B", inputs=MODE_B_INPUTS)
    no_expl = PredictRequestB(mode="B", inputs=MODE_B_INPUTS, include_explanation=False)
    for _ in range(3):
        service.predict(registry, req)

    def timeit(r, n=30):
        out = []
        for _ in range(n):
            t = time.perf_counter()
            service.predict(registry, r)
            out.append((time.perf_counter() - t) * 1000)
        out.sort()
        return statistics.median(out), out[int(0.95 * (len(out) - 1))], max(out)

    with_e, without = timeit(req), timeit(no_expl)
    print(f"\nLATENCY 3-disease Mode B (ms) with SHAP   p50={with_e[0]:.1f} p95={with_e[1]:.1f} max={with_e[2]:.1f}")
    print(f"LATENCY 3-disease Mode B (ms) without SHAP p50={without[0]:.1f} p95={without[1]:.1f} max={without[2]:.1f}")
    assert with_e[1] < 1000  # plan budget: p95 < 1 s

