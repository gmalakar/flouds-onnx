# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.store.exporter_bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thin lazy wrapper over flouds-model-exporter.

Mirrors the CLI invocation:
    flouds-export export --model-for fe \\
        --model-name sentence-transformers/all-MiniLM-L6-v2 \\
        --task feature-extraction \\
        --library transformers \\
        --normalize-embeddings

Key notes:
  - The options are defined in floudsonnx.config.export_options.ExportOptions; their
    defaults match the flouds-export CLI defaults.
  - task must not be None; the registry resolves it from model_for if omitted.
  - library="transformers" bypasses sentence_transformers auto-detection.
  - The output root is passed to the exporter as an argument; the process
    environment is not changed.
  - Exports take turns: during an in-process export the exporter redirects stderr and
    sets thread-count environment variables, both process-wide.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from floudsonnx.config.export_options import ExportOptions
from floudsonnx.exceptions import ExporterNotInstalledError, ExportError, ExportInProgressError, InvalidModelReferenceError, RemoteCodeRequiredError

# The exporter changes process-wide state while an in-process export runs (it
# redirects stderr and sets thread-count environment variables), so two exports in one
# process would interfere with each other. They take turns.
_EXPORT_LOCK = threading.Lock()


def _translate(model_name: str, exc: Exception) -> Exception:
    """The floudsonnx error for an exception raised by the exporter."""
    from model_exporter import errors as exporter_errors

    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, exporter_errors.RemoteCodeNotAllowedError):
        return RemoteCodeRequiredError(
            f"'{model_name}' needs trust_remote_code, which was not allowed",
            detail="Loading it runs code from the model repository. Pass trust_remote_code=True after reviewing that code.",
        )
    if isinstance(exc, exporter_errors.ExportInProgressError):
        return ExportInProgressError(f"Another process is already exporting '{model_name}'", detail=str(exc))
    if isinstance(exc, exporter_errors.InvalidModelReferenceError):
        return InvalidModelReferenceError(str(exc))
    failure_detail = str(getattr(exc, "detail", "") or "").strip()
    if isinstance(exc, exporter_errors.ExportFailedError) and failure_detail:
        detail = failure_detail.splitlines()[-1][:1000]
    return ExportError(f"Export failed for '{model_name}'", detail=detail)


class ExporterBridge:

    @staticmethod
    def onnx_root_for(model_dir: str) -> str:
        """The exporter root for *model_dir*: three levels up.

        The exporter writes to ``<root>/models/<model_for>/<folder>``.
        """
        return str(Path(model_dir).parent.parent.parent)

    @staticmethod
    def export(
        model_name: str,
        model_dir: str,
        model_for: str,
        task: Optional[str] = None,
        *,
        force: bool = False,
        options: Optional[ExportOptions] = None,
        **option_kwargs: Any,
    ) -> str:
        """
        Export *model_name* using flouds-model-exporter.

        Options come from *options*, from keyword arguments named after
        ``ExportOptions`` fields, or both (keywords win). Unknown keywords are passed
        to the exporter unchanged.

        Returns the output directory path on success.
        Raises ExporterNotInstalledError or ExportError.
        """
        try:
            from model_exporter.export.pipeline import export as _export
        except ImportError as exc:
            raise ExporterNotInstalledError() from exc

        resolved = ExportOptions.build(options, task=task, **option_kwargs)
        call_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "model_for": model_for,
            "force": force,
            **resolved.exporter_kwargs(),
            "onnx_path": ExporterBridge.onnx_root_for(model_dir),
        }

        with _EXPORT_LOCK:
            try:
                result = _export(**call_kwargs)
            except Exception as exc:
                raise _translate(model_name, exc) from exc

        return str(result)
