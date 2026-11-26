# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
flouds-onnx
~~~~~~~~~~
Ollama-style ONNX model store and runtime.

Quick start:
    from flouds_onnx import create_model

    model = create_model("sentence-transformers/all-MiniLM-L6-v2")
    outputs = model.run(None, {"input_ids": ids, "attention_mask": mask})
"""
from typing import Any, List

from flouds_onnx.api.client import FloudsOnnxClient, get_default_client
from flouds_onnx.config.export_options import ExportOptions
from flouds_onnx.config.model_config import ModelConfig
from flouds_onnx.config.settings import FloudsOnnxSettings
from flouds_onnx.exceptions import (
    ExporterNotInstalledError,
    ExportError,
    ExportInProgressError,
    FloudsOnnxError,
    InvalidModelReferenceError,
    ManifestError,
    ModelLoadError,
    ModelNameConflictError,
    ModelNotFoundError,
    OptimumNotInstalledError,
    RemoteCodeRequiredError,
    RevisionMismatchError,
    StrategyError,
    TokenizerError,
)
from flouds_onnx.runtime.loader import LoadedModel
from flouds_onnx.runtime.strategy import SessionStrategy
from flouds_onnx.store.manifest import ModelManifest

__version__ = "2.0.1"
__all__ = [
    # Primary API
    "create_model",
    "load_model",
    "pull",
    "list_models",
    "remove_model",
    # Client / settings
    "FloudsOnnxClient",
    "FloudsOnnxSettings",
    "get_default_client",
    # Types
    "ModelConfig",
    "ExportOptions",
    "LoadedModel",
    "SessionStrategy",
    # Exceptions
    "FloudsOnnxError",
    "ModelNotFoundError",
    "InvalidModelReferenceError",
    "ModelNameConflictError",
    "ModelLoadError",
    "TokenizerError",
    "ExportError",
    "ExporterNotInstalledError",
    "ExportInProgressError",
    "RemoteCodeRequiredError",
    "RevisionMismatchError",
    "OptimumNotInstalledError",
    "ManifestError",
    "StrategyError",
]


# ── Top-level convenience functions (delegate to default client singleton) ────


def create_model(model_name: str, model_for: str = "fe", **kwargs: Any) -> LoadedModel:
    """Pull (auto-export if missing) + load session. Primary entry-point."""
    return get_default_client().create_model(model_name, model_for, **kwargs)


def load_model(model_name: str, model_for: str = "fe") -> LoadedModel:
    """Load a model already on disk (no auto-pull)."""
    return get_default_client().load_model(model_name, model_for)


def pull(model_name: str, model_for: str = "fe", **kwargs: Any) -> ModelManifest:
    """Export model to disk only (no session load)."""
    return get_default_client().pull(model_name, model_for, **kwargs)


def list_models() -> List[ModelManifest]:
    """Return manifests for all locally stored models."""
    return get_default_client().list()


def remove_model(model_name: str, model_for: str = "fe") -> bool:
    """Delete model from disk and evict from session pool."""
    return get_default_client().remove(model_name, model_for)
