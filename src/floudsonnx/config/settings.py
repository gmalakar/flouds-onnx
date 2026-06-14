# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator

# Default tasks per model_for — required by optimum (task must not be None)
MODEL_FOR_DEFAULT_TASKS: dict[str, str] = {
    "fe": "feature-extraction",
    "sc": "text-classification",
    "ranker": "text-classification",
    "s2s": "seq2seq-lm",
    "llm": "text-generation-with-past",
}


class FloudsOnnxSettings(BaseModel):
    # ── Store root ────────────────────────────────────────────────────────────
    home_dir: Path = Path.home() / ".flouds"
    onnx_path: Optional[Path] = None

    # ── Session cache sizes ───────────────────────────────────────────────────
    encoder_cache_max: int = 5
    decoder_cache_max: int = 5
    seq2seq_cache_max: int = 3

    # ── ORT provider ─────────────────────────────────────────────────────────
    session_provider: str = "CPUExecutionProvider"

    @field_validator("encoder_cache_max", "decoder_cache_max", "seq2seq_cache_max")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Cache max sizes must be >= 1")
        return v

    @property
    def models_root(self) -> Path:
        root = self.onnx_path or (self.home_dir / "models")
        root.mkdir(parents=True, exist_ok=True)
        return root
