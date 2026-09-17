"""Structured JSON error handling for the FastAPI app."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class SignBridgeError(Exception):
    """Base class for errors that should reach the client as clean JSON."""

    code = "internal_error"
    status_code = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ConfigurationError(SignBridgeError):
    code = "configuration_error"
    status_code = 503


class RecognitionError(SignBridgeError):
    code = "recognition_error"
    status_code = 422


class MotionNotFoundError(SignBridgeError):
    code = "motion_not_found"
    status_code = 404


class InvalidTokenError(SignBridgeError):
    code = "invalid_token"
    status_code = 400


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(SignBridgeError)
    async def handle_signbridge_error(request: Request, exc: SignBridgeError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "detail": exc.detail, "code": exc.code},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": f"{type(exc).__name__}: {exc}",
                "code": "internal_error",
            },
        )
