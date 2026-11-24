# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""Export options are defined once and mean the same thing at every entry point."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from flouds_onnx.api.client import FloudsOnnxClient
from flouds_onnx.cli import main as cli
from flouds_onnx.config.export_options import ExportOptions, SafeExportOptions
from flouds_onnx.store.exporter_bridge import ExporterBridge
from flouds_onnx.store.registry import ModelRegistry

EXPORT = "flouds_onnx.store.registry.ExporterBridge.export"


class TestExportOptions:
    def test_defaults_match_the_exporters(self):
        o = ExportOptions()
        assert (o.optimize, o.optimization_level, o.device, o.opset_version) == (False, 99, "cpu", None)
        assert (o.trust_remote_code, o.use_subprocess, o.merge, o.skip_validator, o.hf_token) == (False, False, False, False, None)

    def test_none_means_use_the_default(self):
        o = ExportOptions(optimize=None, device=None, optimization_level=None)
        assert (o.optimize, o.device, o.optimization_level) == (False, "cpu", 99)

    def test_build_merges_an_object_with_keyword_overrides(self):
        base = ExportOptions(optimize=True, library="transformers")
        o = ExportOptions.build(base, library="sentence_transformers", opset_version=14, optimize=None)
        assert (o.optimize, o.library, o.opset_version) == (True, "sentence_transformers", 14)

    def test_unknown_keywords_are_kept_for_the_exporter(self):
        o = ExportOptions.build(task="feature-extraction", pack_single_file=True)
        assert o.extra == {"pack_single_file": True}
        assert o.exporter_kwargs()["pack_single_file"] is True

    def test_exporter_arguments_omit_unset_options_and_rename_the_token(self):
        kwargs = ExportOptions(task="t", hf_token="secret").exporter_kwargs()
        assert kwargs["huggingface_hub_token"] == "secret" and "hf_token" not in kwargs
        assert not {"opset_version", "framework", "library"} & set(kwargs)

    def test_the_manifest_record_never_contains_the_token(self):
        record = ExportOptions(task="t", hf_token="secret", trust_remote_code=True).manifest_record()
        assert "secret" not in str(record) and "hf_token" not in record and "trust_remote_code" not in record

    def test_the_token_is_not_shown_when_the_options_are_printed(self):
        assert "secret" not in repr(ExportOptions(hf_token="secret"))

    def test_options_safe_for_untrusted_callers_exclude_remote_code_and_the_token(self):
        assert not {"trust_remote_code", "hf_token", "extra"} & set(SafeExportOptions.model_fields)
        assert set(SafeExportOptions.model_fields) < set(ExportOptions.model_fields)


class TestEveryEntryPointUsesTheSameOptions:
    """The same request made three ways must reach the exporter identically."""

    WANT = {
        "task": "feature-extraction",
        "optimize": True,
        "opset_version": 14,
        "library": "transformers",
        "normalize_embeddings": True,
        "merge": True,
    }

    def _exported_with(self, settings, call):
        captured = {}

        def fake_export(**kw):
            captured.update(kw)
            raise RuntimeError("stop after capturing")

        with patch(EXPORT, side_effect=fake_export), pytest.raises(RuntimeError):
            call(FloudsOnnxClient(settings))
        captured.pop("model_dir")
        return captured

    def test_keywords_object_and_cli_agree(self, settings, monkeypatch):
        by_keywords = self._exported_with(settings, lambda c: c.pull("org/model", "fe", **self.WANT))
        by_object = self._exported_with(settings, lambda c: c.pull("org/model", "fe", options=ExportOptions(**self.WANT)))
        by_create = self._exported_with(settings, lambda c: c.create_model("org/model", "fe", **self.WANT))

        def run_cli(client):
            monkeypatch.setattr(cli, "_client", lambda: client)
            monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
            argv = ["flouds-onnx", "pull", "org/model", "--task", "feature-extraction", "--opset-version", "14", "--library", "transformers"]
            monkeypatch.setattr(sys, "argv", argv + ["--normalize-embeddings", "--merge"])
            with pytest.raises(SystemExit):
                cli.main()
            raise RuntimeError("stop after capturing")

        by_cli = self._exported_with(settings, run_cli)
        assert by_keywords == by_object == by_create == by_cli
        assert {key: by_keywords[key] for key in self.WANT} == self.WANT

    def test_default_task_is_filled_in_from_the_model_type(self, settings):
        seen = self._exported_with(settings, lambda c: c.pull("org/model", "s2s"))
        assert seen["task"] == "seq2seq-lm"

    def test_registry_and_bridge_accept_the_same_keywords(self, settings, monkeypatch):
        called = {}
        pipeline = MagicMock()
        pipeline.export.side_effect = lambda **kw: called.update(kw) or "ok"
        monkeypatch.setitem(sys.modules, "model_exporter", MagicMock())
        monkeypatch.setitem(sys.modules, "model_exporter.export", MagicMock())
        monkeypatch.setitem(sys.modules, "model_exporter.export.pipeline", pipeline)

        model_dir = str(settings.models_root / "fe" / "model")
        ExporterBridge.export("org/model", model_dir, "fe", "feature-extraction", optimize=True, hf_token="secret", force=True)
        assert (called["model_name"], called["model_for"], called["force"], called["task"]) == ("org/model", "fe", True, "feature-extraction")
        assert called["optimize"] is True and called["huggingface_hub_token"] == "secret"

        with patch(EXPORT, side_effect=RuntimeError("stop")) as export, pytest.raises(RuntimeError):
            ModelRegistry(settings).pull("org/model", "fe", optimize=True, hf_token="secret")
        assert export.call_args.kwargs["optimize"] is True and export.call_args.kwargs["hf_token"] == "secret"
