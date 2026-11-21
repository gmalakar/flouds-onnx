# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

import logging
import os
from typing import Annotated, Any, Callable

log = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel as _BM
    from pydantic import ConfigDict
    from starlette.concurrency import run_in_threadpool

    from floudsonnx.api.client import get_default_client
    from floudsonnx.config.export_options import ExportOptions, SafeExportOptions
    from floudsonnx.exceptions import (
        ExporterNotInstalledError,
        FloudsOnnxError,
        InvalidModelReferenceError,
        ModelNotFoundError,
        OptimumNotInstalledError,
    )

    router = APIRouter(tags=["models"])

    def _http_error(exc: FloudsOnnxError) -> HTTPException:
        """Translate a library error into an HTTP response.

        Errors the caller can act on keep their message. Anything else is logged in
        full and reported without detail, because the underlying text contains
        filesystem paths and third-party error output.
        """
        if isinstance(exc, InvalidModelReferenceError):
            return HTTPException(status_code=400, detail=str(exc))
        if isinstance(exc, ModelNotFoundError):
            return HTTPException(status_code=404, detail=exc.args[0] if exc.args else "Model not found")
        if isinstance(exc, (ExporterNotInstalledError, OptimumNotInstalledError)):
            return HTTPException(status_code=501, detail=exc.args[0] if exc.args else "A required optional package is not installed")
        log.error("floudsonnx server: %s: %s", type(exc).__name__, exc)
        return HTTPException(status_code=500, detail=f"{type(exc).__name__}: the operation failed. See the server log for details.")

    async def _run(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run blocking store or runtime work in a worker thread.

        Exports take minutes and model loads take seconds. Run on the event loop they
        would stall every other request, including /health.
        """
        try:
            return await run_in_threadpool(func, *args, **kwargs)
        except FloudsOnnxError as exc:
            raise _http_error(exc) from exc

    class PullRequest(SafeExportOptions):
        """Request body for pull and create: the model, plus the export options that are
        safe to accept over HTTP.

        ``trust_remote_code`` and ``hf_token`` are deliberately not part of
        ``SafeExportOptions``. The first runs Python from the model repository on this
        server; the second would put a credential in request bodies and logs. Use the
        CLI or the Python API for models that need remote code; the server takes its
        Hugging Face token from HUGGINGFACE_HUB_TOKEN in its own environment.
        """

        # Unknown fields are rejected rather than ignored, so a caller who sends an
        # option this API does not offer is told so.
        model_config = ConfigDict(extra="forbid", protected_namespaces=())

        model_name: str
        model_for: str = "fe"
        force: bool = False

    class LoadRequest(_BM):
        model_config = ConfigDict(extra="forbid", protected_namespaces=())

        model_name: str
        model_for: str = "fe"

    def _export_options(req: PullRequest) -> ExportOptions:
        safe = req.model_dump(exclude={"model_name", "model_for", "force"})
        return ExportOptions(**safe, trust_remote_code=False, hf_token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or None)

    @router.get("/models")
    async def list_models() -> list[dict[str, Any]]:
        manifests = await _run(get_default_client().list)
        return [m.model_dump(mode="json") for m in manifests]

    @router.get("/models/{name:path}")
    async def get_model(name: str, model_for: Annotated[str, Query()] = "fe") -> dict[str, Any]:
        manifest = await _run(get_default_client().get_manifest, name, model_for)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        return manifest.model_dump(mode="json")

    @router.post("/models/pull")
    async def pull_model(req: PullRequest) -> dict[str, Any]:
        manifest = await _run(get_default_client().pull, req.model_name, req.model_for, force=req.force, **_export_options(req).as_kwargs())
        return manifest.model_dump(mode="json")

    @router.post("/models/create")
    async def create_model(req: PullRequest) -> dict[str, Any]:
        loaded = await _run(
            get_default_client().create_model, req.model_name, req.model_for, force_pull=req.force, **_export_options(req).as_kwargs()
        )
        return {
            "model_name": loaded.model_name,
            "model_for": loaded.model_for,
            "strategy": loaded.session_strategy.value,
            "is_seq2seq": loaded.is_seq2seq,
            "status": "created",
        }

    @router.post("/models/load")
    async def load_model(req: LoadRequest) -> dict[str, str]:
        loaded = await _run(get_default_client().load_model, req.model_name, req.model_for)
        return {"model_name": loaded.model_name, "model_for": loaded.model_for, "strategy": loaded.session_strategy.value, "status": "loaded"}

    @router.post("/models/reload")
    async def reload_model(req: LoadRequest) -> dict[str, str]:
        loaded = await _run(get_default_client().reload, req.model_name, req.model_for)
        return {"model_name": loaded.model_name, "strategy": loaded.session_strategy.value, "status": "reloaded"}

    @router.post("/models/unload")
    async def unload_model(req: LoadRequest) -> dict[str, Any]:
        evicted = await _run(get_default_client().unload, req.model_name, req.model_for)
        return {"model_name": req.model_name, "evicted": evicted}

    @router.delete("/models/{name:path}")
    async def remove_model(name: str, model_for: Annotated[str, Query()] = "fe") -> dict[str, Any]:
        removed = await _run(get_default_client().remove, name, model_for)
        return {"model_name": name, "removed": removed}

    @router.get("/stats")
    async def stats() -> dict[str, Any]:
        return await _run(get_default_client().cache_stats)

except ImportError:
    router = None
