# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

import threading
from typing import Any, List, Optional

from floudsonnx.config.export_options import ExportOptions
from floudsonnx.config.model_config import ModelConfig
from floudsonnx.config.settings import FloudsOnnxSettings
from floudsonnx.runtime.loader import LoadedModel, ModelLoader
from floudsonnx.store.manifest import ModelManifest


class FloudsOnnxClient:
    """Friendly facade over ModelLoader."""

    def __init__(self, settings: Optional[FloudsOnnxSettings] = None) -> None:
        self._loader = ModelLoader(settings)

    def pull(
        self,
        model_name: str,
        model_for: str = "fe",
        task: Optional[str] = None,
        force: bool = False,
        config: Optional[ModelConfig] = None,
        options: Optional[ExportOptions] = None,
        **option_kwargs: Any,
    ) -> ModelManifest:
        """Export a model into the local store if it is not there already.

        Export options can be given as an ``ExportOptions`` object, as keyword
        arguments named after its fields (``optimize``, ``opset_version``, ``library``,
        ``normalize_embeddings``, ``trust_remote_code``, ``hf_token``, ...), or both.
        """
        return self._loader.pull(model_name, model_for, task=task, force=force, config=config, options=options, **option_kwargs)

    def list(self) -> List[ModelManifest]:
        return self._loader.list()

    def get_manifest(self, model_name: str, model_for: str = "fe") -> Optional[ModelManifest]:
        """Manifest of a model in the local store, or None if it is not there."""
        return self._loader.get_manifest(model_name, model_for)

    def remove(self, model_name: str, model_for: str = "fe") -> bool:
        return self._loader.remove(model_name, model_for)

    def create_model(
        self,
        model_name: str,
        model_for: str = "fe",
        task: Optional[str] = None,
        force_pull: bool = False,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> LoadedModel:
        return self._loader.create_model(model_name, model_for, task=task, force_pull=force_pull, config=config, **kwargs)

    def load_model(self, model_name: str, model_for: str = "fe") -> LoadedModel:
        return self._loader.load_model(model_name, model_for)

    def reload(self, model_name: str, model_for: str = "fe") -> LoadedModel:
        return self._loader.reload(model_name, model_for)

    def unload(self, model_name: str, model_for: str = "fe") -> bool:
        return self._loader.unload(model_name, model_for)

    def is_loaded(self, model_name: str, model_for: str = "fe") -> bool:
        return self._loader.is_loaded(model_name, model_for)

    def cache_stats(self) -> dict[str, Any]:
        return self._loader.cache_stats()

    @property
    def settings(self) -> FloudsOnnxSettings:
        return self._loader._settings


_default_client: Optional[FloudsOnnxClient] = None
_default_client_lock = threading.Lock()


def get_default_client() -> FloudsOnnxClient:
    global _default_client
    if _default_client is None:
        with _default_client_lock:
            if _default_client is None:
                _default_client = FloudsOnnxClient(FloudsOnnxSettings.from_env())
    return _default_client
