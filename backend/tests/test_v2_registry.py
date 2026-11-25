from __future__ import annotations

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from backend.api.v2 import config
from backend.api.v2.errors import RegistryLoadError
from backend.api.v2.registry import EXPECTED_CONFIG_KEYS, load_registry

pytestmark = pytest.mark.artifacts


def _results():
    return json.loads(config.DEFAULT_RESULTS_JSON.read_text())


def test_six_configs_loaded_from_json(registry):
    recs = {r["config_key"]: r for r in _results()["results"]}
    assert set(recs) == EXPECTED_CONFIG_KEYS == {b.config_key for b in registry.all()}
    for b in registry.all():
        rec = recs[b.config_key]
        assert list(b.features) == rec["features"]  # order preserved
        assert len(b.features) == rec["n_features"]
        assert b.threshold == rec["selected_threshold"]
        assert len(b.sha256) == 64


def test_locked_thresholds_and_feature_counts_match_spec(registry):
    expected = {"cvd_mode_a": (0.13, 13), "cvd_mode_b": (0.15, 29), "diabetes_mode_a": (0.15, 13),
                "diabetes_mode_b": (0.17, 27), "hypertension_mode_a": (0.41, 11), "hypertension_mode_b": (0.52, 27)}
    for b in registry.all():
        assert (b.threshold, len(b.features)) == expected[b.config_key]


def test_calibrated_artifacts_are_sigmoid_cv_wrappers(registry):
    for b in registry.all():
        assert type(b.model).__name__ == "CalibratedClassifierCV"
        assert [int(c) for c in b.model.classes_] == [0, 1]
        assert b.model.calibrated_classifiers_[0].method == "sigmoid"


def test_all_explainers_ready(registry):
    for b in registry.all():
        assert b.explainer is not None, (b.config_key, b.explanation_error)


# ----------------------------------------------------------------- degraded startup
def _tampered(tmp_path, mutate):
    doc = _results()
    mutate(doc)
    p = tmp_path / "stage_1f_results.json"
    p.write_text(json.dumps(doc))
    return p


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda d: d["results"][0].update(n_features=99), "n_features"),
        (lambda d: d["results"][1].update(selected_threshold=1.5), "threshold"),
        (lambda d: d["results"][2].update(features=d["results"][2]["features"] + ["SEQN"]), "survey"),
        (lambda d: d["results"][3].update(selected_calibration_method="isotonic"), "sigmoid"),
        (lambda d: d["results"][4]["features"].append("not_a_real_feature") or d["results"][4].update(n_features=12), "encoder"),
        (lambda d: d["results"].pop(), "six configs"),
    ],
)
def test_tampered_results_json_fails_startup(tmp_path, mutate, needle):
    path = _tampered(tmp_path, mutate)
    with pytest.raises(RegistryLoadError) as ei:
        load_registry(path)
    assert needle in " ".join(ei.value.reasons)


def test_missing_artifact_fails_startup(tmp_path):
    shutil.copy(config.DEFAULT_RESULTS_JSON, tmp_path / "stage_1f_results.json")
    with pytest.raises(RegistryLoadError) as ei:
        load_registry(tmp_path / "stage_1f_results.json", backend_dir=tmp_path)  # no models under tmp_path
    assert len([r for r in ei.value.reasons if "artifact not found" in r]) == 6


def test_corrupt_artifact_fails_startup(tmp_path):
    for rec in _results()["results"]:
        dest = tmp_path.joinpath(*rec["artifact_paths"]["joblib"].replace("\\", "/").split("/"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"not a joblib file")
    shutil.copy(config.DEFAULT_RESULTS_JSON, tmp_path / "stage_1f_results.json")
    with pytest.raises(RegistryLoadError) as ei:
        load_registry(tmp_path / "stage_1f_results.json", backend_dir=tmp_path)
    assert any("failed to deserialize" in r for r in ei.value.reasons)


def test_unreadable_results_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{not json")
    with pytest.raises(RegistryLoadError):
        load_registry(p)
    with pytest.raises(RegistryLoadError):
        load_registry(tmp_path / "does_not_exist.json")


def test_app_stays_up_with_v2_503_when_registry_broken(monkeypatch, tmp_path, preserve_v2_state):
    monkeypatch.setenv("V2_RESULTS_JSON", str(tmp_path / "missing.json"))
    from backend.api.main import app

    with TestClient(app) as c:
        h = c.get("/v2/health")
        assert h.status_code == 503 and h.json()["status"] == "unavailable" and h.json()["errors"]
        body = {"mode": "A", "inputs": {"age": 40, "gender": "male", "bmi": 25}}
        r = c.post("/v2/predict", json=body)
        assert r.status_code == 503 and r.json()["error"] == "v2_unavailable"
        assert c.get("/v2/model-metrics").status_code == 503
        assert c.get("/health").json() == {"status": "ok"}  # V1 unaffected
    # registry recovers on the next normal startup
    monkeypatch.delenv("V2_RESULTS_JSON")
    with TestClient(app) as c:
        assert c.get("/v2/health").status_code == 200

