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
  - ONNX_PATH is set for the duration of the export and restored afterwards. It is a
    process-wide variable, so exports take turns.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

from floudsonnx.config.export_options import ExportOptions
from floudsonnx.exceptions import ExporterNotInstalledError, ExportError

# ONNX_PATH is a process-wide environment variable. Two exports into different stores
# at the same time would each see the other's value, so exports take turns.
_ENV_LOCK = threading.Lock()


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
        call_kwargs: dict[str, Any] = {"model_name": model_name, "model_for": model_for, "force": force, **resolved.exporter_kwargs()}

        with _ENV_LOCK:
            _orig = os.environ.get("ONNX_PATH")
            os.environ["ONNX_PATH"] = ExporterBridge.onnx_root_for(model_dir)
            try:
                result = _export(**call_kwargs)
            except Exception as exc:
                raise ExportError(
                    f"Export failed for '{model_name}'",
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc
            finally:
                if _orig is None:
                    os.environ.pop("ONNX_PATH", None)
                else:
                    os.environ["ONNX_PATH"] = _orig

        return str(result)
