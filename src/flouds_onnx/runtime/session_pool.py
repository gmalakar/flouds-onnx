# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
flouds_onnx.runtime.session_pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thread-safe LRU caches for loaded ONNX Runtime sessions.

Two caches:
  _encoder_cache  — ort.InferenceSession      (fe, sc, ranker, decoder-only llm)
  _seq2seq_cache  — ORTModelForSeq2SeqLM      (s2s, seq2seq llm)

Cache keys:
  encoder : "<abs_model_dir>#<provider>"
  seq2seq : "<abs_model_dir>#seq2seq#<provider>"

Double-checked locking per key prevents duplicate loads under concurrency.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

import onnxruntime as ort

from flouds_onnx.config.model_config import ModelConfig
from flouds_onnx.runtime.ort_seq2seq import OrtSeq2SeqLoader
from flouds_onnx.runtime.ort_session import OrtInferenceSessionLoader
from flouds_onnx.utils.concurrent_dict import ConcurrentDict

log = logging.getLogger(__name__)


class SessionPool:

    def __init__(
        self,
        encoder_max: int = 5,
        seq2seq_max: int = 3,
        *,
        strict_provider: bool = False,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ) -> None:
        self._strict_provider = strict_provider
        self._intra_op_num_threads = intra_op_num_threads
        self._inter_op_num_threads = inter_op_num_threads
        self._encoder_cache: ConcurrentDict[ort.InferenceSession] = ConcurrentDict("encoders", encoder_max)
        self._seq2seq_cache: ConcurrentDict[Any] = ConcurrentDict("seq2seq", seq2seq_max)
        self._load_locks: dict[str, threading.Lock] = {}
        self._lock_registry = threading.Lock()

    def get_or_load_encoder(self, model_dir: str, config: ModelConfig, provider: str = "CPUExecutionProvider") -> ort.InferenceSession:
        key = self._enc_key(model_dir, provider)
        cached = self._encoder_cache.get(key)
        if cached is not None:
            log.debug("session_pool: encoder HIT — %s", key)
            return cached
        with self._load_lock(key):
            cached = self._encoder_cache.get(key)
            if cached is not None:
                return cached
            log.debug("session_pool: encoder MISS — loading %s", key)
            session = OrtInferenceSessionLoader.load(
                model_dir,
                config,
                provider,
                strict_provider=self._strict_provider,
                intra_op_num_threads=self._intra_op_num_threads,
                inter_op_num_threads=self._inter_op_num_threads,
            )
            self._encoder_cache.put(key, session)
            return session

    def get_or_load_seq2seq(self, model_dir: str, config: ModelConfig, provider: str = "CPUExecutionProvider") -> Any:
        key = self._seq2seq_key(model_dir, provider)
        cached = self._seq2seq_cache.get(key)
        if cached is not None:
            log.debug("session_pool: seq2seq HIT — %s", key)
            return cached
        with self._load_lock(key):
            cached = self._seq2seq_cache.get(key)
            if cached is not None:
                return cached
            log.debug("session_pool: seq2seq MISS — loading %s", key)
            model = OrtSeq2SeqLoader.load(model_dir, config, provider, strict_provider=self._strict_provider)
            self._seq2seq_cache.put(key, model)
            return model

    def evict(self, model_dir: str) -> int:
        """Evict all cached sessions for *model_dir*. Returns count removed."""
        abs_dir = self._norm(model_dir)
        # Keys are "<dir>#<suffix>". Matching on "<dir>#" keeps "all-MiniLM-L6" from
        # also evicting "all-MiniLM-L6-v2".
        prefix = self._dir_prefix(model_dir)
        total = 0
        total += self._encoder_cache.remove_prefix(prefix)
        total += self._seq2seq_cache.remove_prefix(prefix)
        if total:
            log.info("session_pool: evicted %d session(s) for '%s'", total, abs_dir)
        return total

    def contains(self, model_dir: str) -> bool:
        """True if any session for exactly *model_dir* is cached."""
        prefix = self._dir_prefix(model_dir)
        return any(key.startswith(prefix) for cache in (self._encoder_cache, self._seq2seq_cache) for key in cache.keys())

    def stats(self) -> dict:
        return {
            "encoder": self._encoder_cache.stats(),
            "seq2seq": self._seq2seq_cache.stats(),
        }

    def _load_lock(self, key: str) -> threading.Lock:
        with self._lock_registry:
            if key not in self._load_locks:
                self._load_locks[key] = threading.Lock()
            return self._load_locks[key]

    @staticmethod
    def _norm(path: str) -> str:
        return os.path.realpath(os.path.abspath(path))

    @classmethod
    def _dir_prefix(cls, model_dir: str) -> str:
        return f"{cls._norm(model_dir)}#"

    @classmethod
    def _enc_key(cls, model_dir: str, provider: str) -> str:
        return f"{cls._norm(model_dir)}#{provider}"

    @classmethod
    def _seq2seq_key(cls, model_dir: str, provider: str = "CPUExecutionProvider") -> str:
        return f"{cls._norm(model_dir)}#seq2seq#{provider}"
