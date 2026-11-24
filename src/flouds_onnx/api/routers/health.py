# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

import os
from typing import Any


def store_status(models_root: str) -> tuple[bool, str]:
    """Whether the model store can be used, and why not if it cannot.

    A store directory that does not exist yet is fine: it is created by the first
    pull. One that exists but cannot be read or listed is not.
    """
    if not os.path.exists(models_root):
        return True, "not created yet"
    if not os.path.isdir(models_root):
        return False, "store path is not a directory"
    try:
        os.listdir(models_root)
    except OSError as exc:
        return False, f"store is not readable: {type(exc).__name__}"
    return True, "ok"


try:
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router: Any = APIRouter(tags=["health"])

    @router.get("/health")
    async def health() -> dict[str, str]:
        """Liveness: the process is up. Does not touch the model store."""
        from flouds_onnx import __version__

        return {"status": "ok", "service": "flouds-onnx", "version": __version__}

    @router.get("/health/ready")
    async def ready() -> Any:
        """Readiness: the model store is usable. Returns 503 when it is not."""
        from flouds_onnx.api.client import get_default_client

        ok, reason = store_status(str(get_default_client().settings.models_root))
        body = {"status": "ready" if ok else "not_ready", "store": reason}
        return body if ok else JSONResponse(status_code=503, content=body)

except ImportError:
    router = None
