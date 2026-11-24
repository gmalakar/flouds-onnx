# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
Integration test: full pull → load → generate for a real S2S model.

Requires:
  pip install flouds-onnx[export,seq2seq]
  Internet access

Run with:  pytest -m integration
"""
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:Converting a tensor to a Python boolean might cause the trace to be incorrect.*"),
]


@pytest.fixture(scope="module")
def s2s_client(tmp_path_factory):
    from flouds_onnx import FloudsOnnxClient, FloudsOnnxSettings

    home = tmp_path_factory.mktemp("flouds_s2s")
    return FloudsOnnxClient(FloudsOnnxSettings(home_dir=home))


def test_s2s_pull(s2s_client):
    manifest = s2s_client.pull("t5-small", model_for="s2s", task="seq2seq-lm")
    assert manifest.session_strategy == "ort_seq2seq_lm"


def test_s2s_create_model(s2s_client):
    model = s2s_client.create_model("t5-small", model_for="s2s")
    assert model.is_seq2seq is True and model.seq2seq_model is not None


def test_s2s_generate(s2s_client):
    model = s2s_client.create_model("t5-small", model_for="s2s")
    enc = model.tokenizer(
        ["summarize: The quick brown fox jumps over the lazy dog."],
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )
    outputs = model.seq2seq_model.generate(input_ids=enc["input_ids"], max_new_tokens=32)
    decoded = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert isinstance(decoded, str) and len(decoded) > 0
