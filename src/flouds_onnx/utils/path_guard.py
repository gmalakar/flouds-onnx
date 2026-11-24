# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
Path safety helpers.

Model names and model types arrive from callers (and, through the optional HTTP
server, from the network) and are turned into directories that are created, read and
deleted. Everything that builds such a path goes through this module.
"""
from __future__ import annotations

import os
import re

from flouds_onnx.exceptions import InvalidModelReferenceError

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# A model type becomes one directory level: "fe", "s2s", "ranker", or a custom type.
_MODEL_FOR_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")


def safe_model_folder(model_name: str) -> str:
    """
    Return the last path segment of a HuggingFace model name as a safe folder name.

    Raises InvalidModelReferenceError (a ValueError) when no safe folder can be derived:
    an empty name, or one that is only dots ("." and ".." are directory references, and
    a caller able to pass them could reach, or delete, the store's parent).

    >>> safe_model_folder("sentence-transformers/all-MiniLM-L6-v2")
    'all-MiniLM-L6-v2'
    >>> safe_model_folder("t5-small")
    't5-small'
    """
    if not isinstance(model_name, str):
        raise InvalidModelReferenceError(f"model_name must be a string, got {type(model_name).__name__}")
    folder = model_name.strip().rstrip("/").split("/")[-1]
    folder = _UNSAFE_CHARS.sub("_", folder)
    if not folder:
        raise InvalidModelReferenceError(f"Cannot derive a safe folder name from model_name={model_name!r}")
    # "..", ".", "..." and friends. Windows also ignores trailing dots and spaces, so
    # "model." and "model" would be the same directory there.
    if not folder.strip(". ") or folder != folder.rstrip(". "):
        raise InvalidModelReferenceError(f"model_name={model_name!r} is not a valid model name")
    return folder


def safe_model_for(model_for: str) -> str:
    """
    Validate a model type before it is used as a directory name.

    Accepts the built-in types (fe, sc, ranker, s2s, llm) and custom ones made of
    letters, digits, '-' and '_'. Anything containing a path separator or dots is
    refused. Raises InvalidModelReferenceError (a ValueError).
    """
    if not isinstance(model_for, str) or not _MODEL_FOR_RE.fullmatch(model_for):
        raise InvalidModelReferenceError(f"model_for={model_for!r} is not a valid model type")
    return model_for


def safe_join(base: str, *parts: str) -> str:
    """
    Join *base* with *parts* and verify the result stays inside *base*.
    Raises ValueError on path-traversal attempts.
    """
    base_abs = os.path.realpath(os.path.abspath(base))
    joined = os.path.realpath(os.path.abspath(os.path.join(base, *parts)))
    if not joined.startswith(base_abs + os.sep) and joined != base_abs:
        raise ValueError(f"Path traversal detected: '{os.path.join(*parts)}' escapes base '{base}'")
    return joined
