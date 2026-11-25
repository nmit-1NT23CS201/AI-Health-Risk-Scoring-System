"""Typed V2 errors and their HTTP mapping (no ``str(exc)`` leaks)."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class V2Error(Exception):
    status_code = 500
    code = "v2_error"

    def __init__(self, message: str = "", details: list[str] | None = None):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = details or []


class RegistryUnavailableError(V2Error):
    status_code = 503
    code = "v2_unavailable"


class HistoryUnavailableError(V2Error):
    status_code = 503
    code = "v2_history_unavailable"


class RegistryLoadError(Exception):
    """Raised by ``load_registry`` with every hard startup-assertion failure."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


async def v2_error_handler(request: Request, exc: V2Error) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "message": exc.message, "details": exc.details},
    )
