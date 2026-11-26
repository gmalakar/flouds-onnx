# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
flouds_onnx.runtime.tokenizer_cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-thread tokenizer cache.

Tokenizers are kept per thread because fast (Rust) tokenizers are not safe to use from
several threads at once when padding or truncation settings change between calls.

Two things are shared between threads:

* a generation number per model directory. ``evict()`` increases it, and a thread whose
  cached tokenizer carries an older generation reloads on next use. Without it, a
  reload or removal only reached the thread that asked for it and every other thread
  kept the old tokenizer.
* the per-thread size limit, so a long-lived worker thread does not accumulate one
  tokenizer for every model it has ever served.

Loading order: local files in the requested mode, then local files in legacy (slow)
mode. A download from the Hugging Face Hub is attempted only when the cache was created
with ``allow_hub_fallback=True``.

Cache key: "<abs_model_dir>#legacy=<0|1>"
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Tuple

from flouds_onnx.exceptions import TokenizerError

log = logging.getLogger(__name__)

# Models whose exported tokenizer files are known not to load; used only when the Hub
# fallback is explicitly enabled. Each maps to (repository, commit): the download is
# pinned to that commit, so a later push to the repository cannot change what loads.
_HF_FALLBACKS: Dict[str, Tuple[str, str]] = {
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "sentence-t5-base": ("sentence-transformers/sentence-t5-base", "fc5d4628481afbbaaacd7af6bb07cf9d3865f781"),
}


class TokenizerCache:

    def __init__(self, max_per_thread: int = 16, allow_hub_fallback: bool = False) -> None:
        if max_per_thread < 1:
            raise ValueError("max_per_thread must be >= 1")
        self._local = threading.local()
        self._max_per_thread = max_per_thread
        self._allow_hub_fallback = allow_hub_fallback
        self._generations: Dict[str, int] = {}
        self._generation_lock = threading.Lock()

    def get_or_load(self, model_dir: str, use_legacy: bool = False) -> Any:
        key = self._key(model_dir, use_legacy)
        generation = self._generation(model_dir)
        store = self._store()
        cached = store.get(key)
        if cached is not None and cached[0] == generation:
            store.move_to_end(key)
            return cached[1]
        tokenizer = self._load(model_dir, use_legacy, self._allow_hub_fallback)
        store[key] = (generation, tokenizer)
        store.move_to_end(key)
        while len(store) > self._max_per_thread:
            store.popitem(last=False)
        return tokenizer

    def evict(self, model_dir: str) -> None:
        """Drop every thread's tokenizers for *model_dir*.

        This thread's entries are removed now; other threads notice the new generation
        and reload the next time they ask.
        """
        prefix = self._dir_prefix(model_dir)
        with self._generation_lock:
            self._generations[prefix] = self._generations.get(prefix, 0) + 1
        store = self._store()
        # Keys are "<dir>#legacy=<n>"; match the whole directory, not a name prefix.
        for k in [k for k in list(store.keys()) if k.startswith(prefix)]:
            del store[k]

    def _store(self) -> "OrderedDict[str, Tuple[int, Any]]":
        if not hasattr(self._local, "tokenizers"):
            self._local.tokenizers = OrderedDict()
        return self._local.tokenizers

    def _generation(self, model_dir: str) -> int:
        with self._generation_lock:
            return self._generations.get(self._dir_prefix(model_dir), 0)

    @staticmethod
    def _dir_prefix(model_dir: str) -> str:
        return os.path.realpath(os.path.abspath(model_dir)) + "#"

    @classmethod
    def _key(cls, model_dir: str, use_legacy: bool) -> str:
        return f"{cls._dir_prefix(model_dir)}legacy={int(use_legacy)}"

    @staticmethod
    def _load(model_dir: str, use_legacy: bool, allow_hub_fallback: bool = False) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TokenizerError("transformers is required for tokenizer loading.") from exc

        errors = []

        # Attempt 1: local, requested mode
        try:
            log.debug("TokenizerCache: loading from '%s' (legacy=%s)", model_dir, use_legacy)
            tok = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True, use_fast=not use_legacy
            )  # nosec B615 - local files only, no download
            log.info("TokenizerCache: ready — %s", os.path.basename(model_dir))
            return tok
        except Exception as e1:
            errors.append(f"local: {type(e1).__name__}: {e1}")
            log.warning("TokenizerCache: attempt 1 failed (%s); trying legacy", e1)

        # Attempt 2: legacy fallback
        if not use_legacy:
            try:
                tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False)  # nosec B615 - local files only, no download
                log.info("TokenizerCache: ready (legacy) — %s", os.path.basename(model_dir))
                return tok
            except Exception as e2:
                errors.append(f"local legacy: {type(e2).__name__}: {e2}")
                log.warning("TokenizerCache: attempt 2 failed (%s)", e2)

        # Attempt 3: download a known replacement, only if the caller opted in. A local
        # load that quietly reaches the network is a surprise in an offline deployment,
        # and what it fetches is whatever the repository holds that day.
        folder_name = os.path.basename(os.path.normpath(model_dir))
        hf_name, hf_revision = _HF_FALLBACKS.get(folder_name, (None, None))
        if hf_name and allow_hub_fallback:
            try:
                log.warning(
                    "TokenizerCache: local tokenizer unusable; downloading '%s' (revision %s) from the Hugging Face Hub",
                    hf_name,
                    hf_revision,
                )
                return AutoTokenizer.from_pretrained(hf_name, revision=hf_revision, use_fast=False)
            except Exception as e3:
                errors.append(f"hub: {type(e3).__name__}: {e3}")
                log.error("TokenizerCache: HF Hub fallback failed for '%s': %s", hf_name, e3)

        hint = ""
        if hf_name and not allow_hub_fallback:
            hint = " A Hub fallback exists for this model; enable it with allow_hub_tokenizer_fallback=True."
        raise TokenizerError(f"Could not load tokenizer from '{model_dir}'.{hint}", detail="; ".join(errors))
