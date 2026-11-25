from __future__ import annotations

import pytest

pytestmark = pytest.mark.artifacts

A = {"age": 52, "gender": "female", "bmi": 29, "mean_sbp": 128, "mean_dbp": 82}


def test_predict_mode_a_all_diseases(client):
    r = client.post("/v2/predict", json={"mode": "A", "inputs": A})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "A" and body["history_saved"] is True and body["warnings"] == [] and "disclaimer" in body
    assert {"risk_score", "overall_score", "risk_level"}.isdisjoint(body)
    assert [x["disease"] for x in body["results"]] == ["cvd", "diabetes", "hypertension"]
    htn = body["results"][2]
    assert {"mean_sbp", "mean_dbp"} <= set(htn["ignored_inputs"])  # leakage exclusions reported, not silent
    assert "mean_sbp" not in htn["imputed_features"]
    for x in body["results"]:
        assert 0 <= x["probability"] <= 1 and x["decision_label"] in ("screen_positive", "screen_negative")
        assert x["predicted_positive"] == (x["probability"] >= x["threshold"])
        assert x["explanation"]["additive_to_calibrated_probability"] is False


def test_predict_mode_b_subset_and_options(client):
    body = {"mode": "B", "diseases": ["diabetes"], "include_explanation": False, "save_to_history": False,
            "inputs": {**A, "hba1c": 6.8, "fasting_glucose": 130, "total_cholesterol": 200}}
    r = client.post("/v2/predict", json=body)
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["explanation"] is None and res["config_key"] == "diabetes_mode_b"
    assert {"hba1c", "fasting_glucose"} <= set(res["ignored_inputs"])  # excluded for diabetes (target-defining)


def test_age_top_coding_reported(client):
    r = client.post("/v2/predict", json={"mode": "A", "inputs": {**A, "age": 90}, "include_explanation": False})
    assert r.status_code == 200 and r.json()["input_adjustments"]


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "A", "inputs": {**A, "hba1c": 5.5}},
        {"mode": "A", "inputs": {"age": 52, "gender": "female"}},
        {"mode": "A", "inputs": {**A, "bmi": 3}},
        {"mode": "C", "inputs": A},
        {"mode": "A", "inputs": A, "diseases": []},
    ],
)
def test_invalid_requests_are_422(client, body):
    assert client.post("/v2/predict", json=body).status_code == 422


def test_health(client):
    h = client.get("/v2/health")
    assert h.status_code == 200
    j = h.json()
    assert j["status"] == "ok" and len(j["configs"]) == 6 and j["sklearn_version_matches"] is True


def test_model_metrics_are_offline_only(client):
    j = client.get("/v2/model-metrics").json()
    assert j["evaluation"] == "offline_held_out_nhanes_test_split" and j["caveats"]
    assert len(j["configs"]) == 6
    c = j["configs"]["cvd_mode_a"]
    assert c["locked_threshold"] == 0.13 and c["test_metrics"]["roc_auc"] == 0.8137
    assert "exploratory_risk_bands" not in str(j)


def test_features_endpoint(client):
    a = client.get("/v2/features/A").json()
    b = client.get("/v2/features/b").json()
    assert len(a["fields"]) == 13 and len(b["fields"]) == 29
    req = {f["name"] for f in a["fields"] if f["required"]}
    assert req == {"age", "gender", "bmi"}
    bm = {f["name"]: f for f in b["fields"]}
    assert bm["mean_sbp"]["used_by"] == {"cvd": True, "diabetes": True, "hypertension": False}
    assert bm["hba1c"]["used_by"] == {"cvd": True, "diabetes": False, "hypertension": True}
    assert bm["gender"]["allowed_values"] == ["male", "female"]
    assert bm["bmi"]["hard_bounds"] == {"min": 10, "max": 80}
    assert client.get("/v2/features/C").status_code == 422
