# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.api.dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FastAPI dependencies. Imported only by the server, after FastAPI is known to be
installed; the rules themselves live in ``floudsonnx.api.security``, which has no
FastAPI import.
"""
from fastapi import HTTPException, Request

from floudsonnx.api.security import check_access


async def require_access(request: Request) -> None:
    """Applied to every route except /health. See ``floudsonnx.api.security``."""
    state = request.app.state
    status, reason = check_access(
        token=getattr(state, "api_token", None),
        allowed_origins=getattr(state, "allowed_origins", frozenset()),
        authorization=request.headers.get("authorization", ""),
        origin=request.headers.get("origin", ""),
        host_header=request.headers.get("host", ""),
        client_host=request.client.host if request.client else None,
    )
    if status != 200:
        headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
        raise HTTPException(status_code=status, detail=reason, headers=headers)
