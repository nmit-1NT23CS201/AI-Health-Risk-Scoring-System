"""Composition root (Option A): V1 app unchanged + /v2 router + V2 startup lifespan.

Run with:  uvicorn backend.api.main:app
V1-only entry point remains:  uvicorn backend.api.api_server:app

A failed V2 startup never blocks V1: errors are logged, /v2/health and /v2/predict return 503.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from backend.api.api_server import app  # V1 app, untouched
from backend.api.v2.errors import RegistryLoadError, V2Error, v2_error_handler
from backend.api.v2.registry import load_registry, log_event
from backend.api.v2.router import router as v2_router
from backend.database.database import create_v2_tables

_v1_lifespan = app.router.lifespan_context


def _load_v2(app_) -> None:
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
    except Exception as exc:  # never let V2 take V1 down
        app_.state.v2_startup_errors = [f"{type(exc).__name__}: {exc}"]
        log_event("v2_startup_failed", logging.ERROR, reasons=app_.state.v2_startup_errors)


@asynccontextmanager
async def lifespan(app_):
    _load_v2(app_)
    async with _v1_lifespan(app_) as state:
        yield state


app.router.lifespan_context = lifespan
app.add_exception_handler(V2Error, v2_error_handler)
app.include_router(v2_router)
