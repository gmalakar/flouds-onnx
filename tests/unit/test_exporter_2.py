# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""floudsonnx with flouds-model-exporter 2.0: the exporter's errors become floudsonnx
errors, the revision reaches the exporter and the manifest, and the HTTP server answers
each new error with its own status."""
import os
import sys
import types

import pytest

from floudsonnx.config.export_options import ExportOptions
from floudsonnx.exceptions import ExportError, ExportInProgressError, InvalidModelReferenceError, RemoteCodeRequiredError, RevisionMismatchError
from floudsonnx.store.exporter_bridge import ExporterBridge

exporter_errors = pytest.importorskip("model_exporter.errors")


def _install_fake_exporter(monkeypatch, exc=None, seen=None):
    pipeline = types.ModuleType("model_exporter.export.pipeline")

    def fake_export(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        if exc is not None:
            raise exc
        return "ok"

    pipeline.export = fake_export
    monkeypatch.setitem(sys.modules, "model_exporter.export.pipeline", pipeline)


@pytest.mark.parametrize(
    "make_error, expected",
    [
        (lambda: exporter_errors.RemoteCodeNotAllowedError("org/m"), RemoteCodeRequiredError),
        (lambda: exporter_errors.ExportInProgressError("/out", 42), ExportInProgressError),
        (lambda: exporter_errors.InvalidModelReferenceError("model_name must not start with '-'"), InvalidModelReferenceError),
        (lambda: exporter_errors.ExportFailedError("org/m", "line\nOSError: not a valid model identifier"), ExportError),
        (lambda: RuntimeError("anything else"), ExportError),
    ],
    ids=["remote-code", "in-progress", "invalid-reference", "export-failed", "other"],
)
def test_exporter_errors_become_floudsonnx_errors(tmp_path, monkeypatch, make_error, expected):
    _install_fake_exporter(monkeypatch, make_error())
    with pytest.raises(expected) as caught:
        ExporterBridge.export("org/m", str(tmp_path / "models" / "fe" / "m"), "fe", task="feature-extraction")
    assert type(caught.value) is expected


def test_failed_export_detail_is_the_exporters_last_reason(tmp_path, monkeypatch):
    failure = exporter_errors.ExportFailedError("org/m", "noise\nOSError: org/m is not a valid model identifier\n")
    _install_fake_exporter(monkeypatch, failure)
    with pytest.raises(ExportError) as caught:
        ExporterBridge.export("org/m", str(tmp_path / "models" / "fe" / "m"), "fe", task="feature-extraction")
    assert caught.value.detail == "OSError: org/m is not a valid model identifier"


def test_revision_is_passed_to_the_exporter_only_when_set(tmp_path, monkeypatch):
    seen: dict = {}
    _install_fake_exporter(monkeypatch, seen=seen)
    model_dir = str(tmp_path / "models" / "fe" / "m")

    ExporterBridge.export("org/m", model_dir, "fe", task="feature-extraction", revision="abc123")
    assert seen["revision"] == "abc123"

    seen.clear()
    ExporterBridge.export("org/m", model_dir, "fe", task="feature-extraction")
    assert "revision" not in seen


def test_revision_is_an_option_safe_to_accept_over_http():
    from floudsonnx.config.export_options import SafeExportOptions

    assert SafeExportOptions(revision="abc123").revision == "abc123"
    assert ExportOptions().revision is None


def _fake_store_export(model_name, model_dir, model_for, task=None, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.onnx"), "wb") as fh:
        fh.write(b"onnx")
    return model_dir


def test_revision_is_recorded_and_a_different_revision_is_refused(tmp_path, monkeypatch):
    from floudsonnx.config.settings import FloudsOnnxSettings
    from floudsonnx.store.registry import ModelRegistry

    registry = ModelRegistry(FloudsOnnxSettings(home_dir=tmp_path))
    monkeypatch.setattr("floudsonnx.store.registry.ExporterBridge.export", _fake_store_export)

    assert registry.pull("org/m", model_for="fe", revision="abc123").hf_revision == "abc123"
    assert registry.pull("org/m", model_for="fe", revision="abc123").hf_revision == "abc123"
    assert registry.pull("org/m", model_for="fe").hf_revision == "abc123", "no revision asked for: the stored one is fine"

    with pytest.raises(RevisionMismatchError) as caught:
        registry.pull("org/m", model_for="fe", revision="def456")
    assert "abc123" in str(caught.value) and "force=True" in str(caught.value)

    assert registry.pull("org/m", model_for="fe", revision="def456", force=True).hf_revision == "def456"


def test_pull_without_a_recorded_revision_refuses_a_specific_one(tmp_path, monkeypatch):
    from floudsonnx.config.settings import FloudsOnnxSettings
    from floudsonnx.store.registry import ModelRegistry

    registry = ModelRegistry(FloudsOnnxSettings(home_dir=tmp_path))
    monkeypatch.setattr("floudsonnx.store.registry.ExporterBridge.export", _fake_store_export)

    registry.pull("org/m", model_for="fe")
    with pytest.raises(RevisionMismatchError, match="no revision recorded"):
        registry.pull("org/m", model_for="fe", revision="abc123")


@pytest.mark.parametrize(
    "error, status, text",
    [
        (RemoteCodeRequiredError("'org/m' needs trust_remote_code, which was not allowed"), 422, "Python API or the CLI"),
        (ExportInProgressError("Another process is already exporting 'org/m'"), 409, "already exporting"),
        (RevisionMismatchError("'org/m' is stored from abc, not revision 'def'."), 409, "not revision"),
    ],
    ids=["remote-code", "in-progress", "revision"],
)
def test_http_status_for_the_new_errors(error, status, text):
    pytest.importorskip("fastapi")
    from floudsonnx.api.routers import models as models_router

    http = models_router._http_error(error)
    assert http.status_code == status
    assert text in http.detail
