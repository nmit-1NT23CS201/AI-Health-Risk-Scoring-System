"""Preventive Health Risk Assessment API (V2).

Run with:  uvicorn backend.api.main:app --reload

A failed model-registry load never takes the process down: errors are logged as structured JSON,
`/v2/health`, `/v2/predict`, `/v2/model-metrics` and `/v2/features/*` return 503, and history
endpoints (database only) keep working.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.v2.errors import RegistryLoadError, V2Error, v2_error_handler
from backend.api.v2.registry import load_registry, log_event
from backend.api.v2.router import router as v2_router
from backend.database.database import create_v2_tables

API_VERSION = "2.0"


def _load_v2(app_: FastAPI) -> None:
    try:  # history storage is independent of the model registry
        create_v2_tables()
    except Exception as exc:
        log_event("v2_history_init_failed", logging.ERROR, reason=f"{type(exc).__name__}: {exc}")
    app_.state.v2_registry = None
    app_.state.v2_startup_errors = []
    try:
        app_.state.v2_registry = load_registry()
        log_event("v2_ready", configs=len(app_.state.v2_registry.bundles))
    except RegistryLoadError as exc:
        app_.state.v2_startup_errors = exc.reasons
        log_event("v2_startup_failed", logging.ERROR, reasons=exc.reasons)
    except Exception as exc:  # unexpected failure: stay up, report 503
        app_.state.v2_startup_errors = [f"{type(exc).__name__}: {exc}"]
        log_event("v2_startup_failed", logging.ERROR, reasons=app_.state.v2_startup_errors)


@asynccontextmanager
async def v2_lifespan(app_: FastAPI):
    _load_v2(app_)
    yield


app = FastAPI(title="Preventive Health Risk Assessment API", version=API_VERSION, lifespan=v2_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_exception_handler(V2Error, v2_error_handler)
app.include_router(v2_router)  # router defines the /v2 prefix


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "version": API_VERSION}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": API_VERSION}