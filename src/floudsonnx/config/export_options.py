# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.config.export_options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The options that control an export, defined once.

Every entry point (Python API, CLI, HTTP server) builds one ``ExportOptions`` and the
store, the exporter bridge and the manifest all read from it. Adding an option means
adding one field here.

Two classes, because not every option may come from every caller:

``SafeExportOptions``
    Options that only change how a model is converted. The HTTP request model is
    built on this one.

``ExportOptions``
    Adds the options that must not be accepted from an untrusted caller:
    ``trust_remote_code`` (runs Python from the model repository) and ``hf_token``
    (a credential), plus ``extra`` for exporter arguments floudsonnx does not model.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SafeExportOptions(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    task: Optional[str] = None
    optimize: bool = False
    optimization_level: int = 99
    opset_version: Optional[int] = None
    device: str = "cpu"
    framework: Optional[str] = None
    library: Optional[str] = None
    normalize_embeddings: bool = False
    use_external_data_format: bool = False
    use_subprocess: bool = False
    use_fallback_if_failed: bool = False
    merge: bool = False
    skip_validator: bool = False

    @model_validator(mode="before")
    @classmethod
    def _none_means_default(cls, data: Any) -> Any:
        """A value of None (or JSON null) means "use the default", for every option."""
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value is not None}
        return data


class ExportOptions(SafeExportOptions):
    trust_remote_code: bool = False
    hf_token: Optional[str] = Field(default=None, repr=False)
    # Arguments passed to the exporter unchanged, for options not modelled above.
    extra: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(cls, options: Optional[SafeExportOptions] = None, **overrides: Any) -> "ExportOptions":
        """Combine an options object with keyword overrides.

        Keywords that are not fields of this class are kept in ``extra`` and reach the
        exporter as they are. A keyword set to None does not override anything.
        """
        values: Dict[str, Any] = options.model_dump() if options is not None else {}
        extra: Dict[str, Any] = dict(values.pop("extra", None) or {})
        for key, value in overrides.items():
            if key == "extra":
                extra.update(value or {})
            elif key in cls.model_fields:
                if value is not None:
                    values[key] = value
            else:
                extra[key] = value
        return cls(**values, extra=extra)

    def as_kwargs(self) -> Dict[str, Any]:
        """Every option as flat keyword arguments (``extra`` merged in)."""
        values = self.model_dump(exclude={"extra"})
        return {**self.extra, **values}

    def exporter_kwargs(self) -> Dict[str, Any]:
        """Arguments for ``model_exporter.export.pipeline.export``.

        Options left unset are omitted so the exporter applies its own default, and the
        token is passed under the exporter's name for it.
        """
        kwargs: Dict[str, Any] = {
            "task": self.task,
            "optimize": self.optimize,
            "optimization_level": self.optimization_level,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "trust_remote_code": self.trust_remote_code,
            "use_external_data_format": self.use_external_data_format,
            "use_subprocess": self.use_subprocess,
            "use_fallback_if_failed": self.use_fallback_if_failed,
            "merge": self.merge,
            "skip_validator": self.skip_validator,
        }
        if self.opset_version is not None:
            kwargs["opset_version"] = self.opset_version
        if self.framework is not None:
            kwargs["framework"] = self.framework
        if self.library is not None:
            kwargs["library"] = self.library
        if self.hf_token is not None:
            kwargs["huggingface_hub_token"] = self.hf_token
        kwargs.update(self.extra)
        return kwargs

    def manifest_record(self) -> Dict[str, Any]:
        """What the manifest stores about this export. Never includes the token."""
        return {
            "task": self.task,
            "optimize": self.optimize,
            # The level only means something when optimisation ran.
            "optimization_level": self.optimization_level if self.optimize else None,
            "opset_version": self.opset_version,
            "device": self.device,
            "framework": self.framework,
            "library": self.library,
            "normalize_embeddings": self.normalize_embeddings,
            "use_external_data_format": self.use_external_data_format,
            "merge": self.merge,
        }
