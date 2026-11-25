from __future__ import annotations

import dataclasses
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from backend.api.v2 import config, service
from backend.api.v2.explain import unwrap_base_estimator
from backend.api.v2.preprocessing import EncodedInputs
from backend.api.v2.preprocessing import encode_inputs
from backend.api.v2.schemas import ModeBInputs, PredictRequestA, PredictRequestB
from conftest import load_split_frame, row_to_inputs, sample_rows

pytestmark = pytest.mark.artifacts

CONFIGS = [f"{d}_mode_{m}" for d in ("cvd", "diabetes", "hypertension") for m in ("a", "b")]
_MAX_DIFF: dict[str, float] = {}


def _bundle(registry, key):
    return next(b for b in registry.all() if b.config_key == key)


def _request(bundle, inputs):
    cls = PredictRequestA if bundle.mode == "A" else PredictRequestB
    return cls(mode=bundle.mode, diseases=[bundle.disease], inputs=inputs, include_explanation=False)


@pytest.mark.parametrize("key", CONFIGS)
def test_golden_parity_sampled_seqns(registry, key):
    """Schema -> encode -> service vs an independent offline load of the Stage 1F artifact, 1e-6.
    Rows whose bmi is missing in NHANES cannot be sent through the API (bmi is required by policy),
    so they are exercised at the encoded level below."""
    b = _bundle(registry, key)
    offline_model = joblib.load(b.artifact_path)  # independent deserialization
    rows = sample_rows(key, n=50)
    feats = list(b.features)
    offline = offline_model.predict_proba(rows[feats])[:, 1]
    worst, n_api = 0.0, 0
    for i, (_, row) in enumerate(rows.iterrows()):
        if pd.isna(row["bmi"]):
            continue
        n_api += 1
        res = service.predict(registry, _request(b, row_to_inputs(row, feats))).results[0]
        diff = abs(res.probability - offline[i])
        worst = max(worst, diff)
        assert diff <= 1e-6, (key, int(row["SEQN"]), diff)
        assert res.predicted_positive == bool(offline[i] >= b.threshold)
        assert res.threshold == b.threshold
        assert res.ignored_inputs == []
    assert n_api >= 30, f"too few API-eligible rows ({n_api})"
    _MAX_DIFF[key] = worst


@pytest.mark.parametrize("key", CONFIGS)
def test_golden_parity_encoded_level_includes_missing_bmi(registry, key):
    """Same production code below schema validation, for ALL 50+ sampled rows incl. NaN bmi / sparse rows."""
    b = _bundle(registry, key)
    rows = sample_rows(key, n=50)
    feats = list(b.features)
    offline = joblib.load(b.artifact_path).predict_proba(rows[feats])[:, 1]
    for i, (_, row) in enumerate(rows.iterrows()):
        values = {f: float(row[f]) for f in feats}
        enc = EncodedInputs(values=values, provided={f for f, v in values.items() if not np.isnan(v)})
        res = service.disease_result(b, enc, include_explanation=False)
        assert abs(res.probability - offline[i]) <= 1e-6, (key, int(row["SEQN"]))
        assert res.predicted_positive == bool(offline[i] >= b.threshold)


def test_golden_parity_through_http(client, registry):
    for key in CONFIGS:
        b = _bundle(registry, key)
        feats = list(b.features)
        rows = sample_rows(key, n=8)
        rows = rows[rows["bmi"].notna()].head(5)  # bmi is a required API field
        offline = joblib.load(b.artifact_path).predict_proba(rows[feats])[:, 1]
        for i, (_, row) in enumerate(rows.iterrows()):
            body = {"mode": b.mode, "diseases": [b.disease], "include_explanation": False,
                    "inputs": row_to_inputs(row, feats)}
            r = client.post("/v2/predict", json=body)
            assert r.status_code == 200, r.text
            assert abs(r.json()["results"][0]["probability"] - offline[i]) <= 1e-6


@pytest.mark.parametrize("key", CONFIGS)
def test_aggregate_test_split_metrics_match_stage_1f(registry, key):
    """Whole test split through the service's prediction path == published Stage 1F numbers (4 d.p.)."""
    b = _bundle(registry, key)
    rec = b.record
    df = load_split_frame(key)
    assert len(df) == rec["n_test"]
    y = df[f"target_{b.disease}"].astype(int).to_numpy()
    p = service.predict_proba_frame(b, df[list(b.features)])
    tm, tb = rec["test_metrics"], rec["test_binary_at_optimized_threshold"]
    tol = 5e-5
    assert abs(roc_auc_score(y, p) - tm["roc_auc"]) <= tol
    assert abs(average_precision_score(y, p) - tm["pr_auc"]) <= tol
    assert abs(brier_score_loss(y, p) - tm["brier_score"]) <= tol
    pred = p >= b.threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    assert (tp, fp, fn, tn) == (tb["tp"], tb["fp"], tb["fn"], tb["tn"])
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    prec, npv = tp / (tp + fp), tn / (tn + fn)
    f1 = 2 * prec * sens / (prec + sens)
    for got, want in ((sens, tb["sensitivity"]), (spec, tb["specificity"]), (prec, tb["precision"]),
                      (npv, tb["npv"]), (f1, tb["f1"]), ((sens + spec) / 2, tb["balanced_accuracy"])):
        assert abs(got - want) <= tol
    assert tb["threshold"] == b.threshold


class _SpyModel:
    """Returns a fixed probability; fails the test if predict() (0.5 cut-off) is ever used."""

    def __init__(self, p):
        self.p, self.calls = p, 0

    def predict_proba(self, X):
        self.calls += 1
        return np.array([[1 - self.p, self.p]])

    def predict(self, X):
        raise AssertionError("model.predict() must never be used")


@pytest.mark.parametrize("key", ["cvd_mode_a", "hypertension_mode_b"])
def test_threshold_boundary_is_inclusive_and_probability_untouched(registry, key):
    b = _bundle(registry, key)
    inputs = {"age": 50, "gender": "male", "bmi": 28}
    for p, expected in ((b.threshold, True), (np.nextafter(b.threshold, 0), False), (np.nextafter(b.threshold, 1), True)):
        spy = _SpyModel(float(p))
        fake = dataclasses.replace(b, model=spy)
        res = service.disease_result(fake, encode_inputs(ModeBInputs(**inputs)), include_explanation=False)
        assert spy.calls == 1
        assert res.probability == float(p)  # no post-processing / second sigmoid
        assert res.predicted_positive is expected
        assert res.decision_label == ("screen_positive" if expected else "screen_negative")


@pytest.mark.parametrize("key", CONFIGS)
def test_sigmoid_is_inside_the_artifact_not_applied_again(registry, key):
    """Calibrated output differs from the uncalibrated base estimator => calibration lives in the
    artifact, and the API returns it verbatim (checked above), i.e. it is applied exactly once."""
    b = _bundle(registry, key)
    df = load_split_frame(key).head(200)
    X = df[list(b.features)]
    cal = b.model.predict_proba(X)[:, 1]
    base = unwrap_base_estimator(b.model).predict_proba(X)[:, 1]
    assert np.abs(cal - base).max() > 1e-3
    api = service.predict_proba_frame(b, X)
    assert np.allclose(api, cal, rtol=0, atol=1e-12)  # RF thread-summation order may differ at ~1e-17


def test_parity_summary():
    if _MAX_DIFF:
        print("\nGOLDEN PARITY max |api-offline| per config:", json.dumps(_MAX_DIFF), "overall", max(_MAX_DIFF.values()))


