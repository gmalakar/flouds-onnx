# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import BaseModel, field_validator

# Default tasks per model_for — required by optimum (task must not be None)
MODEL_FOR_DEFAULT_TASKS: dict[str, str] = {
    "fe": "feature-extraction",
    "sc": "text-classification",
    "ranker": "text-classification",
    "s2s": "seq2seq-lm",
    "llm": "text-generation-with-past",
}

# Environment variables read by FloudsOnnxSettings.from_env(): variable -> field.
ENV_FIELDS: dict[str, str] = {
    "FLOUDS_ONNX_HOME_DIR": "home_dir",
    "FLOUDS_ONNX_ONNX_PATH": "onnx_path",
    "FLOUDS_ONNX_SESSION_PROVIDER": "session_provider",
    "FLOUDS_ONNX_STRICT_PROVIDER": "strict_provider",
    "FLOUDS_ONNX_INTRA_OP_THREADS": "intra_op_num_threads",
    "FLOUDS_ONNX_INTER_OP_THREADS": "inter_op_num_threads",
    "FLOUDS_ONNX_ENCODER_CACHE_MAX": "encoder_cache_max",
    "FLOUDS_ONNX_SEQ2SEQ_CACHE_MAX": "seq2seq_cache_max",
    "FLOUDS_ONNX_TOKENIZER_CACHE_MAX": "tokenizer_cache_max",
    "FLOUDS_ONNX_ALLOW_HUB_TOKENIZER_FALLBACK": "allow_hub_tokenizer_fallback",
}


class FloudsOnnxSettings(BaseModel):
    # ── Store root ────────────────────────────────────────────────────────────
    home_dir: Path = Path.home() / ".flouds"
    onnx_path: Optional[Path] = None

    # ── Session cache sizes ───────────────────────────────────────────────────
    encoder_cache_max: int = 5
    seq2seq_cache_max: int = 3
    # Tokenizers kept per thread (fast tokenizers are not safe to share between threads).
    tokenizer_cache_max: int = 16

    # ── ORT provider ─────────────────────────────────────────────────────────
    session_provider: str = "CPUExecutionProvider"
    # When the requested provider is unavailable: fail (True) or fall back to CPU (False).
    # A silent fall back turns a deployment mistake into a much slower service.
    strict_provider: bool = False

    # ── ORT threading ────────────────────────────────────────────────────────
    # None leaves ONNX Runtime's default, which is one thread per physical core for
    # every session. With several models loaded in one process, or several worker
    # processes on one machine, set these so the sessions do not oversubscribe the CPU.
    intra_op_num_threads: Optional[int] = None
    inter_op_num_threads: Optional[int] = None

    # ── Tokenizers ───────────────────────────────────────────────────────────
    # Tokenizers are loaded from the model directory only. Enabling this lets a failed
    # local load download a known replacement from the Hugging Face Hub instead.
    allow_hub_tokenizer_fallback: bool = False

    @field_validator("encoder_cache_max", "seq2seq_cache_max", "tokenizer_cache_max")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Cache max sizes must be >= 1")
        return v

    @field_validator("intra_op_num_threads", "inter_op_num_threads")
    @classmethod
    def _positive_or_none(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("Thread counts must be >= 1, or unset for the ONNX Runtime default")
        return v

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None, **overrides: Any) -> "FloudsOnnxSettings":
        """Build settings from ``FLOUDS_ONNX_*`` environment variables.

        Unset or empty variables keep the default. Keyword arguments win over the
        environment. Used for the process-wide default client and the HTTP server;
        pass a ``FloudsOnnxSettings`` to ``FloudsOnnxClient`` for explicit control.
        """
        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for variable, field in ENV_FIELDS.items():
            raw = env.get(variable)
            if raw is not None and raw.strip():
                values[field] = raw.strip()
        values.update(overrides)
        return cls(**values)

    @property
    def models_root(self) -> Path:
        """Directory holding ``<model_for>/<model folder>``. Not created by reading it."""
        return self.onnx_path or (self.home_dir / "models")
