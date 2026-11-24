# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from flouds_onnx.exceptions import ServerConfigurationError, ServerNotInstalledError

log = logging.getLogger(__name__)


def create_app(api_token: Optional[str] = None, allowed_origins: Optional[Iterable[str]] = None) -> Any:
    """Build the FastAPI application.

    *api_token* and *allowed_origins* default to the ``FLOUDS_ONNX_API_TOKEN`` and
    ``FLOUDS_ONNX_ALLOWED_ORIGINS`` environment variables. Without a token the server
    only answers requests from this machine; see ``flouds_onnx.api.security``.
    """
    try:
        from fastapi import Depends, FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise ServerNotInstalledError() from exc
    from flouds_onnx import __version__
    from flouds_onnx.api import security
    from flouds_onnx.api.dependencies import require_access
    from flouds_onnx.api.routers import health, models
    from flouds_onnx.exceptions import InvalidModelReferenceError

    app = FastAPI(title="flouds-onnx", description="ONNX model store and runtime REST API", version=__version__)
    app.state.api_token = api_token.strip() if isinstance(api_token, str) and api_token.strip() else security.configured_token()
    app.state.allowed_origins = (
        frozenset(origin.strip().rstrip("/").lower() for origin in allowed_origins) if allowed_origins is not None else security.configured_origins()
    )
    if not app.state.api_token:
        log.warning("flouds-onnx server: no API token configured; only requests from this machine will be served.")

    @app.exception_handler(InvalidModelReferenceError)
    async def _invalid_model_reference(_request: Request, exc: InvalidModelReferenceError) -> JSONResponse:
        # Raised before any file is touched; report it as a bad request on every route.
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    app.include_router(health.router)
    app.include_router(models.router, prefix="/api/v1", dependencies=[Depends(require_access)])
    return app


def run_server(host: str = "127.0.0.1", port: int = 19720, reload: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise ServerNotInstalledError() from exc
    from flouds_onnx.api import security

    # The server can export, load and delete models. Listening beyond this machine
    # without a token would hand that to anyone who can reach the port.
    if not security.is_loopback_host(host) and not security.configured_token():
        raise ServerConfigurationError(f"Refusing to listen on '{host}' without an API token. Set {security.API_TOKEN_ENV}, or use --host 127.0.0.1.")
    uvicorn.run("flouds_onnx.api.server:create_app", factory=True, host=host, port=port, reload=reload)
