# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.exceptions
~~~~~~~~~~~~~~~~~~~~~
Full exception hierarchy. All public exceptions importable from the package root.
"""


class FloudsOnnxError(Exception):
    """Base exception for all floudsonnx errors."""

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.detail = detail

    def __str__(self) -> str:
        if self.detail:
            return f"{super().__str__()} — {self.detail}"
        return super().__str__()


# ── Store layer ───────────────────────────────────────────────────────────────


class InvalidModelReferenceError(FloudsOnnxError, ValueError):
    """A model name or model type cannot be used to address the local store.

    Also a ValueError, so existing ``except ValueError`` handlers keep working.
    """


class ModelNotFoundError(FloudsOnnxError):
    """Model does not exist in the local store and cannot be pulled."""


class ModelNameConflictError(FloudsOnnxError):
    """A different model with the same short name already occupies the store folder."""


class RevisionMismatchError(FloudsOnnxError):
    """The stored export was made from a different revision than the one requested."""


class ManifestError(FloudsOnnxError):
    """manifest.json is missing, corrupt, or fails schema validation."""


class ExportError(FloudsOnnxError):
    """flouds-model-exporter raised an error during ONNX export."""


class RemoteCodeRequiredError(ExportError):
    """The model needs ``trust_remote_code``, which was not allowed.

    Loading such a model runs Python from the model repository. Pass
    ``trust_remote_code=True`` after reviewing that code. The HTTP server never accepts
    it, so such models must be pulled through the Python API or the CLI.
    """


class ExportInProgressError(ExportError):
    """Another process is already exporting this model into the store."""


class ExporterNotInstalledError(ExportError):
    """flouds-model-exporter is not installed."""

    def __init__(self) -> None:
        super().__init__("flouds-model-exporter is required for auto-export. " "Install it with: pip install floudsonnx[export]")


# ── Runtime layer ─────────────────────────────────────────────────────────────


class ModelLoadError(FloudsOnnxError):
    """ORT InferenceSession or ORTModelForSeq2SeqLM failed to load."""


class TokenizerError(FloudsOnnxError):
    """Tokenizer could not be loaded from disk."""


class StrategyError(FloudsOnnxError):
    """Cannot determine a session strategy from the given ModelConfig."""


class OptimumNotInstalledError(ModelLoadError):
    """optimum[onnxruntime] is not installed; required for seq2seq models."""

    def __init__(self) -> None:
        super().__init__("optimum[onnxruntime] is required for seq2seq / LLM model loading. " "Install it with: pip install floudsonnx[seq2seq]")


# ── Server layer ──────────────────────────────────────────────────────────────


class ServerNotInstalledError(FloudsOnnxError):
    """fastapi / uvicorn are not installed; required for the HTTP server."""

    def __init__(self) -> None:
        super().__init__("fastapi and uvicorn are required for the HTTP server. " "Install them with: pip install floudsonnx[server]")


class ServerConfigurationError(FloudsOnnxError):
    """The HTTP server was asked to start in an unsafe or invalid configuration."""
