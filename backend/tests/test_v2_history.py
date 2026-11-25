from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import backend.database.database as db

pytestmark = pytest.mark.artifacts

A = {"age": 52, "gender": "female", "bmi": 29, "mean_sbp": 128, "mean_dbp": 82}
B = {**A, "hba1c": 5.9, "total_cholesterol": 205}


def _rows(sql="SELECT * FROM v2_assessments ORDER BY id"):
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def _post(client, mode="A", inputs=A, **kw):
    r = client.post("/v2/predict", json={"mode": mode, "inputs": inputs, "include_explanation": False, **kw})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(autouse=True)
def clean_history(client):
    db.delete_all_v2_history()
    yield
    db.delete_all_v2_history()


# ------------------------------------------------------------------ schema / lifecycle
def test_tables_created_by_lifespan_not_import(monkeypatch, tmp_path, preserve_v2_state):
    from backend.api.main import app

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "fresh.db")
    assert not (tmp_path / "fresh.db").exists()  # nothing at import time
    with TestClient(app):
        pass
    cols = {r["name"]: r for r in _rows("PRAGMA table_info(v2_assessments)")}
    assert list(cols) == ["id", "submission_id", "created_at", "disease", "mode", "config_key", "inputs_json",
                          "model_input_json", "probability", "threshold", "predicted_positive", "imputed_json",
                          "model_version", "artifact_sha256", "schema_version"]
    idx = {r["name"] for r in _rows("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_v2_assessments_submission", "idx_v2_assessments_created"} <= idx


def test_create_v2_tables_is_idempotent_and_leaves_v1_table_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "v1.db")
    db.create_tables()
    before = _rows("SELECT sql FROM sqlite_master WHERE name='assessments'")
    db.create_v2_tables()
    db.create_v2_tables()
    assert _rows("SELECT sql FROM sqlite_master WHERE name='assessments'") == before


def test_check_constraints_reject_bad_rows():
    base = dict(submission_id="s", created_at="t", disease="cvd", mode="A", config_key="k", inputs_json="{}",
                model_input_json="{}", probability=0.1, threshold=0.2, predicted_positive=0, imputed_json="[]",
                model_version="v", artifact_sha256="x", schema_version=1)
    for bad in ({"disease": "flu"}, {"mode": "C"}, {"predicted_positive": 2}):
        with pytest.raises(sqlite3.IntegrityError):
            db.save_v2_submission([{**base, **bad}])


def test_submission_insert_is_atomic():
    base = dict(submission_id="atomic", created_at="t", disease="cvd", mode="A", config_key="k", inputs_json="{}",
                model_input_json="{}", probability=0.1, threshold=0.2, predicted_positive=0, imputed_json="[]",
                model_version="v", artifact_sha256="x", schema_version=1)
    rows = [base, {**base, "disease": "diabetes"}, {**base, "disease": "not-a-disease"}]
    with pytest.raises(sqlite3.IntegrityError):
        db.save_v2_submission(rows)
    assert _rows() == []  # nothing stored


# ------------------------------------------------------------------ persistence via /v2/predict
def test_three_disease_predict_creates_three_rows_sharing_submission_id(client):
    body = _post(client)
    rows = _rows()
    assert body["history_saved"] is True and body["warnings"] == []
    assert len(rows) == 3
    assert {r["submission_id"] for r in rows} == {body["submission_id"]}
    assert [r["disease"] for r in rows] == ["cvd", "diabetes", "hypertension"]
    for r, res in zip(rows, body["results"]):
        assert r["probability"] == res["probability"] and r["threshold"] == res["threshold"]
        assert r["predicted_positive"] == int(res["predicted_positive"])
        assert r["config_key"] == res["config_key"] and r["model_version"] == res["model_version"]
        assert r["artifact_sha256"] == res["artifact_sha256"] and r["schema_version"] == 1 and r["mode"] == "A"
        assert json.loads(r["imputed_json"]) == res["imputed_features"]
    # timestamps: UTC ISO-8601 with Z, identical across the submission
    assert len({r["created_at"] for r in rows}) == 1
    assert rows[0]["created_at"].endswith("Z")
    datetime.fromisoformat(rows[0]["created_at"].replace("Z", "+00:00"))
    # inputs as submitted (human enums), encoded vector in model order with NaN as null
    assert json.loads(rows[0]["inputs_json"]) == {**A}
    vec = json.loads(rows[0]["model_input_json"])
    assert vec["gender"] == 2.0 and vec["bmi"] == 29.0 and vec["smoking_status"] is None
    assert list(vec)[:3] == ["age", "gender", "education_level"]


def test_save_to_history_false_creates_no_rows(client):
    body = _post(client, save_to_history=False)
    assert body["history_saved"] is False and body["warnings"] == []
    assert _rows() == []


def test_subset_of_diseases_and_mode_b(client):
    body = _post(client, mode="B", inputs=B, diseases=["diabetes", "cvd"])
    rows = _rows()
    assert len(rows) == 2 and {r["mode"] for r in rows} == {"B"} and body["history_saved"] is True


def test_failed_validation_stores_nothing(client):
    r = client.post("/v2/predict", json={"mode": "A", "inputs": {"age": 52}})
    assert r.status_code == 422 and _rows() == []


def test_v1_assessments_table_untouched_by_v2_activity(client):
    before = _rows("SELECT * FROM assessments")
    sql_before = _rows("SELECT sql FROM sqlite_master WHERE name='assessments'")
    sid = _post(client)["submission_id"]
    client.get("/v2/history")
    client.delete(f"/v2/history/{sid}")
    _post(client)
    client.delete("/v2/history?confirm=true")
    assert _rows("SELECT * FROM assessments") == before
    assert _rows("SELECT sql FROM sqlite_master WHERE name='assessments'") == sql_before


# ------------------------------------------------------------------ failure policy
def test_db_failure_after_prediction_returns_200_with_warning(client, monkeypatch):
    def boom(rows):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(db, "save_v2_submission", boom)
    r = client.post("/v2/predict", json={"mode": "A", "inputs": A})
    assert r.status_code == 200
    body = r.json()
    assert body["history_saved"] is False
    assert len(body["warnings"]) == 1 and "could not be saved" in body["warnings"][0]
    assert "disk I/O" not in json.dumps(body)  # no internal detail leaked
    assert len(body["results"]) == 3 and all(x["explanation"] for x in body["results"])  # prediction intact


# ------------------------------------------------------------------ list / get / delete
def test_list_is_newest_first_grouped_and_paginated(client):
    ids = [_post(client)["submission_id"] for _ in range(3)]
    full = client.get("/v2/history").json()
    assert [i["submission_id"] for i in full["items"]] == ids[::-1]
    assert full["next_before_id"] is None
    first = full["items"][0]
    assert first["mode"] == "A" and len(first["results"]) == 3 and first["created_at"].endswith("Z")
    assert set(first["results"][0]) == {"disease", "config_key", "probability", "threshold", "predicted_positive",
                                        "model_version"}

    p1 = client.get("/v2/history?limit=2").json()
    assert [i["submission_id"] for i in p1["items"]] == ids[::-1][:2] and p1["next_before_id"] is not None
    p2 = client.get(f"/v2/history?limit=2&before_id={p1['next_before_id']}").json()
    assert [i["submission_id"] for i in p2["items"]] == ids[::-1][2:] and p2["next_before_id"] is None


@pytest.mark.parametrize("q", ["limit=0", "limit=101", "before_id=0", "limit=abc"])
def test_list_query_validation(client, q):
    assert client.get(f"/v2/history?{q}").status_code == 422


def test_get_submission_detail_and_404(client):
    sid = _post(client, mode="B", inputs=B)["submission_id"]
    d = client.get(f"/v2/history/{sid}")
    assert d.status_code == 200
    j = d.json()
    assert j["submission_id"] == sid and j["mode"] == "B" and j["inputs"] == B
    assert [r["disease"] for r in j["results"]] == ["cvd", "diabetes", "hypertension"]
    r0 = j["results"][0]
    assert r0["model_input"]["hba1c"] == 5.9 and r0["model_input"]["smoking_status"] is None
    assert "smoking_status" in r0["imputed_features"] and len(r0["artifact_sha256"]) == 64
    assert client.get("/v2/history/does-not-exist").status_code == 404


def test_delete_one_and_404(client):
    keep, drop = _post(client)["submission_id"], _post(client)["submission_id"]
    r = client.delete(f"/v2/history/{drop}")
    assert r.status_code == 200 and r.json() == {"deleted_rows": 3, "submission_id": drop}
    assert [i["submission_id"] for i in client.get("/v2/history").json()["items"]] == [keep]
    assert client.delete(f"/v2/history/{drop}").status_code == 404
    assert client.get(f"/v2/history/{drop}").status_code == 404


def test_delete_all_requires_confirm_true(client):
    _post(client)
    _post(client)
    for q in ("", "?confirm=false", "?confirm=0", "?confirm=maybe"):
        assert client.delete(f"/v2/history{q}").status_code == 422
    assert len(_rows()) == 6  # guard held
    r = client.delete("/v2/history?confirm=true")
    assert r.status_code == 200 and r.json()["deleted_rows"] == 6
    assert _rows() == [] and client.get("/v2/history").json() == {"items": [], "next_before_id": None}


def test_history_works_while_registry_is_unavailable(client, monkeypatch, tmp_path, preserve_v2_state):
    sid = _post(client)["submission_id"]
    monkeypatch.setenv("V2_RESULTS_JSON", str(tmp_path / "missing.json"))
    from backend.api.main import app

    with TestClient(app) as c:
        assert c.post("/v2/predict", json={"mode": "A", "inputs": A}).status_code == 503
        assert [i["submission_id"] for i in c.get("/v2/history").json()["items"]] == [sid]
        assert c.delete(f"/v2/history/{sid}").status_code == 200


def test_history_db_error_is_503_not_500(client, monkeypatch):
    def boom(*a, **k):
        raise sqlite3.OperationalError("no such table")

    monkeypatch.setattr(db, "list_v2_history", boom)
    r = client.get("/v2/history")
    assert r.status_code == 503 and r.json()["error"] == "v2_history_unavailable"
    assert "no such table" not in r.text