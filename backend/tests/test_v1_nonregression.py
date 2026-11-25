"""V1 contract (what frontend/app/page.tsx depends on) must survive V2 wiring.
V1 files are untouched (checked via git when available); runs against a temp DB."""
from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

V1_PAYLOAD = {  # defaultValues from frontend/app/page.tsx:64-79
    "age": 40, "gender": "Male", "bmi": 25, "systolic_bp": 125, "diastolic_bp": 80, "cholesterol_mg_dl": 190,
    "smoking": "No", "alcohol_consumption": "No", "physical_activity": "Medium", "family_history": "No",
    "heart_rate_bpm": 76, "sdnn_hrv": 55, "rmssd_hrv": 50, "spo2": 97,
}
V1_ROUTES = {("GET", "/"), ("GET", "/health"), ("GET", "/history"), ("DELETE", "/history/{assessment_id}"),
             ("DELETE", "/history"), ("GET", "/model-metrics"), ("POST", "/predict")}
PREDICT_KEYS = {"risk_score", "risk_level", "positive_contributors", "negative_contributors", "insights",
                "recommendations", "summary_plot", "bar_plot", "waterfall_plot"}


def _contract(c: TestClient):
    assert c.get("/").json() == {"status": "ok"}
    assert c.get("/health").json() == {"status": "ok"}

    r = c.post("/predict", json=V1_PAYLOAD)
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j) == PREDICT_KEYS
    assert isinstance(j["risk_score"], float) and isinstance(j["risk_level"], str)
    assert all(set(x) == {"feature", "value"} for x in j["positive_contributors"] + j["negative_contributors"])
    assert all(isinstance(j[k], str) and len(j[k]) > 100 for k in ("summary_plot", "bar_plot", "waterfall_plot"))

    hist = c.get("/history").json()
    assert hist and set(hist[0]) == {"id", "age", "risk_score", "risk_level", "created_at"}
    assert c.delete(f"/history/{hist[0]['id']}").json() == {"success": True, "message": "Assessment deleted"}
    assert c.delete("/history/999999").status_code == 404
    assert c.delete("/history").json() == {"success": True, "message": "All assessments deleted"}

    m = c.get("/model-metrics")
    assert m.status_code == 200 and set(m.json()) == {"regression_metrics", "classification_metrics", "confusion_matrix"}
    assert c.post("/predict", json={**V1_PAYLOAD, "age": 5}).status_code == 422


def test_v1_contract_with_v2_running(client):
    _contract(client)


def test_v1_contract_when_v2_registry_is_broken(monkeypatch, tmp_path, preserve_v2_state):
    monkeypatch.setenv("V2_RESULTS_JSON", str(tmp_path / "missing.json"))
    with TestClient(app) as c:
        assert c.get("/v2/health").status_code == 503
        _contract(c)


def test_v1_routes_and_schemas_unchanged(client):
    spec = client.get("/openapi.json").json()
    got = {(m.upper(), p) for p, ops in spec["paths"].items() if not p.startswith("/v2") for m in ops}
    assert got == V1_ROUTES
    s = spec["components"]["schemas"]
    assert set(s["PredictionInput"]["properties"]) == set(V1_PAYLOAD)
    assert set(s["PredictionResponse"]["properties"]) == PREDICT_KEYS
    assert {p for p in spec["paths"] if p.startswith("/v2")} == {
        "/v2/predict", "/v2/health", "/v2/model-metrics", "/v2/features/{mode}", "/v2/history",
        "/v2/history/{submission_id}"}


def test_v1_source_files_untouched():
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--", "backend/api/api_server.py", "backend/api/app.py",
             "backend/ml", "shap_compute.py", "shap_plot_figure3.py"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
    except Exception:
        pytest.skip("git unavailable")
    assert out == "", f"frozen/V1 files modified: {out}"



def test_v1_database_code_is_only_extended():
    """Phase 2C appends V2 functions to database.py; the committed V1 text must remain a verbatim prefix."""
    from pathlib import Path

    try:
        head = subprocess.run(["git", "show", "HEAD:backend/database/database.py"], capture_output=True,
                              check=True, timeout=30).stdout.decode("utf-8")
    except Exception:
        pytest.skip("git unavailable")
    current = (Path(__file__).resolve().parents[2] / "backend" / "database" / "database.py").read_bytes().decode("utf-8")
    norm = lambda s: s.replace("\r\n", "\n")  # git may serve LF while the working tree is CRLF
    assert norm(current).startswith(norm(head).rstrip("\n")), "V1 database code was modified (only appending is allowed)"