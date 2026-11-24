# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
flouds_onnx.runtime.ort_session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loads a raw onnxruntime.InferenceSession.
Used for: fe, sc, ranker, decoder-only llm.

ONNX file selection priority:
  1. <model_dir>/<config.optimized_onnx_model>   (model_optimized.onnx)
  2. <model_dir>/<config.encoder_onnx_model>     (model.onnx)
  3. <model_dir>/model.onnx                      (canonical fallback)

Provider selection: if the requested provider is unavailable the loader falls back to
CPUExecutionProvider with a warning, or raises when ``strict_provider`` is set.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import onnxruntime as ort

from flouds_onnx.config.model_config import ModelConfig
from flouds_onnx.exceptions import ModelLoadError
from flouds_onnx.utils.path_guard import safe_join

log = logging.getLogger(__name__)

_FALLBACK_PROVIDER = "CPUExecutionProvider"
_CANDIDATE_NAMES = ["model_optimized.onnx", "model.onnx"]


def resolve_provider(requested: str, strict: bool = False) -> str:
    """The provider to use: *requested* if this ONNX Runtime build has it.

    Otherwise CPU, with a warning; or a ModelLoadError when *strict* is set, so that a
    GPU deployment missing its GPU build fails at start-up instead of running many
    times slower without anyone noticing.
    """
    available = ort.get_available_providers()
    if requested in available:
        return requested
    if strict:
        raise ModelLoadError(f"Execution provider '{requested}' is not available", detail=f"Available: {available}")
    log.warning("Provider '%s' not available (available: %s); falling back to %s", requested, available, _FALLBACK_PROVIDER)
    return _FALLBACK_PROVIDER


def build_session_options(intra_op_num_threads: Optional[int] = None, inter_op_num_threads: Optional[int] = None) -> Optional[ort.SessionOptions]:
    """SessionOptions with the given thread limits, or None to use ONNX Runtime's defaults."""
    if intra_op_num_threads is None and inter_op_num_threads is None:
        return None
    options = ort.SessionOptions()
    if intra_op_num_threads is not None:
        options.intra_op_num_threads = intra_op_num_threads
    if inter_op_num_threads is not None:
        options.inter_op_num_threads = inter_op_num_threads
    return options


class OrtInferenceSessionLoader:

    @staticmethod
    def load(
        model_dir: str,
        config: ModelConfig,
        provider: str = _FALLBACK_PROVIDER,
        *,
        strict_provider: bool = False,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ) -> ort.InferenceSession:
        onnx_path = OrtInferenceSessionLoader._resolve_onnx_file(model_dir, config)
        resolved_provider = OrtInferenceSessionLoader._resolve_provider(provider, strict_provider)
        session_options = build_session_options(intra_op_num_threads, inter_op_num_threads)
        log.debug("OrtInferenceSessionLoader: loading '%s' provider=%s", onnx_path, resolved_provider)
        try:
            if session_options is None:
                session = ort.InferenceSession(onnx_path, providers=[resolved_provider])
            else:
                session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=[resolved_provider])
        except Exception as exc:
            raise ModelLoadError(f"ort.InferenceSession failed for '{onnx_path}'", detail=str(exc)) from exc
        log.info("OrtInferenceSessionLoader: session ready — %s", os.path.basename(onnx_path))
        return session

    @staticmethod
    def _resolve_onnx_file(model_dir: str, config: ModelConfig) -> str:
        candidates = [config.optimized_onnx_model, config.encoder_onnx_model] + _CANDIDATE_NAMES
        seen: set[str] = set()
        ordered: List[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        for name in ordered:
            try:
                path = safe_join(model_dir, name)
            except ValueError:
                continue
            if os.path.isfile(path):
                return path
        raise ModelLoadError(f"No usable ONNX file found in '{model_dir}'", detail=f"Tried: {ordered}")

    @staticmethod
    def _resolve_provider(requested: str, strict: bool = False) -> str:
        return resolve_provider(requested, strict)
