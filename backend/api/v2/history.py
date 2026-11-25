"""V2 history persistence: maps prediction results to ``v2_assessments`` rows and back.

SQL lives in ``backend/database/database.py``; this module only shapes data. Functions are
called through the ``database`` module object so tests can substitute them.
"""
from __future__ import annotations

import json
import math
from datetime import datetime

import pandas as pd

from backend.api.v2.schemas import (
    HistoryDetail,
    HistoryDetailResult,
    HistoryList,
    HistoryResult,
    HistorySubmission,
    PredictRequestA,
    PredictRequestB,
    PredictResponse,
)
from backend.database import database


def iso_utc(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _vector(frame: pd.DataFrame) -> dict[str, float | None]:
    return {c: (None if math.isnan(v) else float(v)) for c, v in frame.iloc[0].items()}


def save_submission(
    request: PredictRequestA | PredictRequestB, response: PredictResponse, frames: list[pd.DataFrame]
) -> int:
    """Store every disease row of one submission atomically. Raises on failure (caller degrades)."""
    inputs_json = json.dumps(request.inputs.model_dump(mode="json", exclude_none=True))
    created = iso_utc(response.created_at)
    rows = [
        {
            "submission_id": response.submission_id,
            "created_at": created,
            "disease": r.disease.value,
            "mode": r.mode,
            "config_key": r.config_key,
            "inputs_json": inputs_json,
            "model_input_json": json.dumps(_vector(frame)),
            "probability": r.probability,
            "threshold": r.threshold,
            "predicted_positive": int(r.predicted_positive),
            "imputed_json": json.dumps(r.imputed_features),
            "model_version": r.model_version,
            "artifact_sha256": r.artifact_sha256,
            "schema_version": database.V2_SCHEMA_VERSION,
        }
        for r, frame in zip(response.results, frames)
    ]
    return database.save_v2_submission(rows)


def _result(row: dict) -> HistoryResult:
    return HistoryResult(
        disease=row["disease"], config_key=row["config_key"], probability=row["probability"],
        threshold=row["threshold"], predicted_positive=bool(row["predicted_positive"]),
        model_version=row["model_version"],
    )


def list_history(limit: int, before_id: int | None) -> HistoryList:
    groups = database.list_v2_history(limit=limit, before_id=before_id)
    items = [
        HistorySubmission(
            submission_id=g["submission_id"], created_at=g["rows"][0]["created_at"], mode=g["rows"][0]["mode"],
            results=[_result(r) for r in g["rows"]],
        )
        for g in groups
    ]
    nxt = groups[-1]["last_id"] if len(groups) == limit else None
    return HistoryList(items=items, next_before_id=nxt)


def get_submission(submission_id: str) -> HistoryDetail | None:
    rows = database.get_v2_submission(submission_id)
    if not rows:
        return None
    first = rows[0]
    return HistoryDetail(
        submission_id=submission_id, created_at=first["created_at"], mode=first["mode"],
        inputs=json.loads(first["inputs_json"]),
        results=[
            HistoryDetailResult(
                **_result(r).model_dump(), model_input=json.loads(r["model_input_json"]),
                imputed_features=json.loads(r["imputed_json"] or "[]"), artifact_sha256=r["artifact_sha256"],
            )
            for r in rows
        ],
    )


def delete_submission(submission_id: str) -> int:
    return database.delete_v2_submission(submission_id)


def delete_all() -> int:
    return database.delete_all_v2_history()