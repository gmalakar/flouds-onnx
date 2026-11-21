# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.store.manifest
~~~~~~~~~~~~~~~~~~~~~~~~~~
ModelManifest schema and read/write helpers.
One manifest.json lives alongside the ONNX files in each model directory.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.exceptions import ManifestError

MANIFEST_FILENAME = "manifest.json"
# Increase when the manifest's meaning changes, so older stores can be recognised.
MANIFEST_SCHEMA_VERSION = 2


class ExportOptions(BaseModel):
    """The options the model was actually exported with.

    Fields are optional because a value of ``None`` means "the exporter's default was
    used", which is different from a value floudsonnx chose.
    """

    task: Optional[str] = None
    optimize: Optional[bool] = None
    optimization_level: Optional[int] = None
    opset_version: Optional[int] = None
    device: Optional[str] = None
    framework: Optional[str] = None
    library: Optional[str] = None
    normalize_embeddings: Optional[bool] = None
    use_external_data_format: Optional[bool] = None
    merge: Optional[bool] = None


class ModelManifest(BaseModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    model_name: str
    model_for: str
    folder: str
    pulled_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hf_revision: Optional[str] = None
    session_strategy: str = "ort_inference_session"
    onnx_files: List[str] = Field(default_factory=list)
    export_options: ExportOptions = Field(default_factory=ExportOptions)
    config: Optional[ModelConfig] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    def save(self, model_dir: str) -> str:
        """Write manifest.json into *model_dir*. Returns the file path.

        Written to a temporary file and renamed into place, so a crash or a full disk
        leaves either the previous manifest or none, never a truncated one.
        """
        path = os.path.join(model_dir, MANIFEST_FILENAME)
        tmp_path = ""
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=model_dir)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except OSError as exc:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise ManifestError(f"Failed to write manifest to {path}", detail=str(exc)) from exc
        return path

    @classmethod
    def load(cls, model_dir: str) -> "ModelManifest":
        """Read and validate manifest.json from *model_dir*."""
        path = os.path.join(model_dir, MANIFEST_FILENAME)
        if not os.path.isfile(path):
            raise ManifestError(f"manifest.json not found in {model_dir}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"Failed to read manifest from {path}", detail=str(exc)) from exc
        if isinstance(data, dict) and "schema_version" not in data:
            data["schema_version"] = 1  # written before the field existed
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise ManifestError(f"Invalid manifest schema in {path}", detail=str(exc)) from exc

    @classmethod
    def exists(cls, model_dir: str) -> bool:
        return os.path.isfile(os.path.join(model_dir, MANIFEST_FILENAME))

    def scan_onnx_files(self, model_dir: str) -> None:
        """Update onnx_files by scanning *model_dir* for *.onnx files."""
        try:
            self.onnx_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".onnx"))
        except OSError:
            self.onnx_files = []
