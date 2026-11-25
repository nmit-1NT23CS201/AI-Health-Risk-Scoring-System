"""/v2 routes. Thin: validation + delegation to service/registry; no business logic."""
from __future__ import annotations

import logging
import uuid
from typing import Literal

import annotated_types
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from backend.api.v2 import config, history, service
from backend.api.v2.errors import HistoryUnavailableError, RegistryUnavailableError
from backend.api.v2.preprocessing import FEATURE_SPECS, REQUIRED_FIELDS, input_field_names
from backend.api.v2.registry import DISEASES, MODES, ModelRegistry
from backend.database import database
from backend.api.v2.schemas import (
    HistoryDeleteResponse,
    HistoryDetail,
    HistoryList,
    ModeAInputs,
    ModeBInputs,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger("backend.api.v2")
router = APIRouter(prefix="/v2", tags=["v2"])


MAX_CLIENT_ID_LENGTH = 128


def get_client_id(x_client_id: str | None = Header(default=None, alias="X-Client-ID")) -> str:
    """Per-browser history scope. Missing/blank falls back to "anonymous" (curl / Swagger keep working)."""
    value = (x_client_id or "").strip()
    if not value:
        return database.DEFAULT_CLIENT_ID
    if len(value) > MAX_CLIENT_ID_LENGTH:
        raise HTTPException(status_code=422, detail=f"X-Client-ID must be at most {MAX_CLIENT_ID_LENGTH} characters.")
    return value


def get_registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "v2_registry", None)
    if registry is None:
        raise RegistryUnavailableError(
            "V2 models are unavailable.", getattr(request.app.state, "v2_startup_errors", []) or []
        )
    return registry


@router.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    registry: ModelRegistry = Depends(get_registry),
    client_id: str = Depends(get_client_id),
):
    try:
        return service.predict(registry, body, client_id)
    except Exception:
        request_id = str(uuid.uuid4())
        log.exception("v2_predict_failed request_id=%s", request_id)
        return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": request_id})


@router.get("/health")
def health(request: Request):
    registry = getattr(request.app.state, "v2_registry", None)
    if registry is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "database_backend": database.backend_name(),
                "errors": getattr(request.app.state, "v2_startup_errors", []),
            },
        )
    configs = {
        b.config_key: {
            "loaded": True,
            "n_features": len(b.features),
            "threshold": b.threshold,
            "model_name": b.model_name,
            "artifact_sha256": b.sha256,
            "explanations_available": b.explainer is not None,
            "explanation_error": b.explanation_error,
        }
        for b in registry.all()
    }
    degraded = any(not c["explanations_available"] for c in configs.values())
    return {
        "status": "degraded" if degraded else "ok",
        "database_backend": database.backend_name(),
        "results_sha256": registry.results_sha256,
        "sklearn_expected": registry.sklearn_expected,
        "versions_runtime": registry.versions_runtime,
        "sklearn_version_matches": registry.versions_runtime["scikit-learn"] == registry.sklearn_expected,
        "configs": configs,
    }


@router.get("/model-metrics")
def model_metrics(registry: ModelRegistry = Depends(get_registry)):
    """Offline research evaluation only (from stage_1f_results.json). No per-user data."""
    out = {}
    for b in registry.all():
        r = b.record
        out[b.config_key] = {
            "disease": b.disease,
            "mode": b.mode,
            "model_name": b.model_name,
            "locked_threshold": b.threshold,
            "calibration": r["selected_calibration_method"],
            "n_train": r["n_train"],
            "n_val": r["n_val"],
            "n_test": r["n_test"],
            "test_metrics": r["test_metrics"],
            "test_binary_at_locked_threshold": r["test_binary_at_optimized_threshold"],
        }
    return {"evaluation": "offline_held_out_nhanes_test_split", "caveats": list(config.METRICS_CAVEATS), "configs": out}


def _bounds(model, name: str) -> dict:
    lo = hi = None
    for m in model.model_fields[name].metadata:
        if isinstance(m, annotated_types.Ge):
            lo = m.ge
        elif isinstance(m, annotated_types.Le):
            hi = m.le
    return {"min": lo, "max": hi}


@router.get("/features/{mode}")
def features(mode: Literal["A", "B", "a", "b"], registry: ModelRegistry = Depends(get_registry)):
    mode = mode.upper()
    model = ModeBInputs if mode == "B" else ModeAInputs
    used_by = {d: set(registry.get(d, mode).features) for d in DISEASES}
    fields = []
    for name in input_field_names(mode):
        spec = FEATURE_SPECS[name]
        info = model.model_fields[name]
        enum_values = None
        for arg in getattr(info.annotation, "__args__", (info.annotation,)):
            if hasattr(arg, "__members__"):
                enum_values = [m.value for m in arg]
        fields.append(
            {
                "name": name,
                "label": spec.label,
                "unit": spec.unit,
                "required": name in REQUIRED_FIELDS,
                "allowed_values": enum_values,
                "hard_bounds": _bounds(model, name),
                "training_range": list(spec.training_range) if spec.training_range else None,
                "description": info.description,
                "used_by": {d: name in used_by[d] for d in DISEASES},
            }
        )
    return {"mode": mode, "fields": fields}


# ----------------------------------------------------------------------------- history
# History only needs the database, so it keeps working when the model registry is unavailable.
def _db(fn, *args):
    try:
        return fn(*args)
    except database.DB_ERRORS as exc:
        log.error("v2_history_db_error error=%s", type(exc).__name__)  # never log driver messages (may hold hosts)
        raise HistoryUnavailableError("V2 history storage is unavailable.") from None


@router.get("/history", response_model=HistoryList)
def list_history(
    limit: int = Query(20, ge=1, le=100),
    before_id: int | None = Query(None, ge=1),
    client_id: str = Depends(get_client_id),
):
    return _db(history.list_history, client_id, limit, before_id)


@router.get("/history/{submission_id}", response_model=HistoryDetail)
def get_history(submission_id: str, client_id: str = Depends(get_client_id)):
    detail = _db(history.get_submission, client_id, submission_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Submission {submission_id} not found.")
    return detail


@router.delete("/history/{submission_id}", response_model=HistoryDeleteResponse)
def delete_history_item(submission_id: str, client_id: str = Depends(get_client_id)):
    n = _db(history.delete_submission, client_id, submission_id)
    if n == 0:
        raise HTTPException(status_code=404, detail=f"Submission {submission_id} not found.")
    return HistoryDeleteResponse(deleted_rows=n, submission_id=submission_id)


@router.delete("/history", response_model=HistoryDeleteResponse)
def delete_all_history(confirm: bool = Query(False), client_id: str = Depends(get_client_id)):
    if not confirm:
        raise HTTPException(status_code=422, detail="Deleting all V2 history requires ?confirm=true.")
    return HistoryDeleteResponse(deleted_rows=_db(history.delete_all, client_id))