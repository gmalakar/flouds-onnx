# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""flouds-onnx errors survive pickle and copy with their message and detail."""

import copy
import pickle

import pytest

from flouds_onnx.exceptions import ExporterNotInstalledError, FloudsOnnxError, InvalidModelReferenceError, ModelNotFoundError

CASES = [
    FloudsOnnxError("base failure", detail="more context"),
    ModelNotFoundError("model missing", detail="looked in /store"),
    InvalidModelReferenceError("bad name"),
    ExporterNotInstalledError(),
]


@pytest.mark.parametrize("error", CASES, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("clone", [lambda e: pickle.loads(pickle.dumps(e)), copy.copy], ids=["pickle", "copy"])
def test_round_trip_keeps_type_message_and_detail(error, clone):
    restored = clone(error)
    assert type(restored) is type(error)
    assert str(restored) == str(error)
    assert restored.detail == error.detail
    assert restored.args == error.args
