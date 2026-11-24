# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flouds_onnx.exceptions import ExporterNotInstalledError, ExportError
from flouds_onnx.store.exporter_bridge import ExporterBridge


class TestExporterBridge:
    def test_raises_when_not_installed(self):
        with patch.dict(
            sys.modules,
            {
                "model_exporter": None,
                "model_exporter.export": None,
                "model_exporter.export.pipeline": None,
            },
        ):
            with pytest.raises(ExporterNotInstalledError):
                ExporterBridge.export("x/y", "/tmp/fe/x/y", "fe", None)

    def test_wraps_export_error(self, tmp_path):
        model_dir = str(tmp_path / "models" / "fe" / "mymodel")
        os.makedirs(model_dir, exist_ok=True)
        with patch.object(ExporterBridge, "export", side_effect=ExportError("Export failed for 'x/y'", detail="boom")):
            with pytest.raises(ExportError, match="Export failed"):
                ExporterBridge.export("x/y", model_dir, "fe", None)

    def test_onnx_root_is_three_levels_above_model_dir(self, tmp_path):
        model_dir = tmp_path / "models" / "fe" / "mymodel"
        model_dir.mkdir(parents=True)
        assert str(Path(str(model_dir)).parent.parent.parent) == str(tmp_path)

    def test_passes_the_store_root_as_onnx_path(self, tmp_path, monkeypatch):
        """The exporter gets the store root as an argument; ONNX_PATH is never touched."""
        model_dir = str(tmp_path / "models" / "fe" / "mymodel")
        os.makedirs(model_dir, exist_ok=True)
        monkeypatch.setenv("ONNX_PATH", "/caller/own/value")
        captured = {}

        def fake_export(**kw):
            captured.update(kw)
            captured["env_during_export"] = os.environ.get("ONNX_PATH")
            return model_dir

        mock_pipeline = MagicMock()
        mock_pipeline.export = fake_export

        with patch.dict(
            sys.modules,
            {
                "model_exporter": MagicMock(),
                "model_exporter.export": MagicMock(),
                "model_exporter.export.pipeline": mock_pipeline,
            },
        ):
            ExporterBridge.export("org/model", model_dir, "fe", task="feature-extraction")

        assert captured["onnx_path"] == str(tmp_path)
        assert captured["env_during_export"] == "/caller/own/value"
        assert os.environ["ONNX_PATH"] == "/caller/own/value"

    def test_passes_task_and_optimize_to_exporter(self, tmp_path):
        """task and optimize are forwarded to export()."""
        model_dir = str(tmp_path / "models" / "fe" / "mymodel")
        os.makedirs(model_dir, exist_ok=True)
        captured = {}

        def fake_export(**kw):
            captured.update(kw)
            return model_dir

        mock_pipeline = MagicMock()
        mock_pipeline.export = fake_export

        with patch.dict(
            sys.modules,
            {
                "model_exporter": MagicMock(),
                "model_exporter.export": MagicMock(),
                "model_exporter.export.pipeline": mock_pipeline,
            },
        ):
            ExporterBridge.export("org/model", model_dir, "fe", task="feature-extraction", optimize=True)

        assert captured.get("task") == "feature-extraction"
        assert captured.get("optimize") is True
        assert captured.get("model_name") == "org/model"
        assert captured.get("model_for") == "fe"

    def test_passes_hf_token_as_huggingface_hub_token(self, tmp_path):
        """flouds-model-exporter 1.x expects huggingface_hub_token."""
        model_dir = str(tmp_path / "models" / "fe" / "mymodel")
        os.makedirs(model_dir, exist_ok=True)
        captured = {}

        def fake_export(**kw):
            captured.update(kw)
            return model_dir

        mock_pipeline = MagicMock()
        mock_pipeline.export = fake_export

        with patch.dict(
            sys.modules,
            {
                "model_exporter": MagicMock(),
                "model_exporter.export": MagicMock(),
                "model_exporter.export.pipeline": mock_pipeline,
            },
        ):
            ExporterBridge.export(
                "org/private-model",
                model_dir,
                "fe",
                task="feature-extraction",
                hf_token="hf_test",
            )

        assert captured.get("huggingface_hub_token") == "hf_test"
        assert "hf_token" not in captured
