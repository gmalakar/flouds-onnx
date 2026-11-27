# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""The ``allow_export`` switch: off means a missing model is an error, never an export."""
import asyncio
import copy
import pickle
import sys
from unittest.mock import MagicMock, patch

import pytest

from flouds_onnx import ExportDisabledError, ExporterNotInstalledError
from flouds_onnx.config.settings import FloudsOnnxSettings
from flouds_onnx.store.manifest import ModelManifest
from flouds_onnx.store.registry import ModelRegistry

EXPORT = "flouds_onnx.store.registry.ExporterBridge.export"


def _store_model(settings: FloudsOnnxSettings, model_name: str = "org/model", folder: str = "model") -> None:
    model_dir = settings.models_root / "fe" / folder
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"onnx")
    manifest = ModelManifest(model_name=model_name, model_for="fe", folder=folder)
    manifest.onnx_files = ["model.onnx"]
    manifest.save(str(model_dir))


class TestSetting:
    def test_export_is_allowed_by_default(self):
        assert FloudsOnnxSettings().allow_export is True

    @pytest.mark.parametrize(("raw", "expected"), [("false", False), ("0", False), ("true", True), ("1", True)])
    def test_read_from_the_environment(self, raw, expected):
        settings = FloudsOnnxSettings.from_env({"FLOUDS_ONNX_ALLOW_EXPORT": raw})
        assert settings.allow_export is expected

    def test_unset_variable_keeps_the_default(self):
        assert FloudsOnnxSettings.from_env({}).allow_export is True


class TestExportDisabled:
    @pytest.fixture()
    def off(self, tmp_path):
        return FloudsOnnxSettings(home_dir=tmp_path / "home", allow_export=False)

    def test_a_missing_model_is_an_error_and_nothing_is_exported_or_created(self, off):
        with patch(EXPORT) as export:
            with pytest.raises(ExportDisabledError, match="export is disabled"):
                ModelRegistry(off).pull("org/model")
        export.assert_not_called()
        assert not (off.models_root / "fe").exists()

    def test_a_model_in_the_store_still_loads(self, off):
        _store_model(off)
        with patch(EXPORT) as export:
            manifest = ModelRegistry(off).pull("org/model")
        export.assert_not_called()
        assert manifest.model_name == "org/model"

    def test_force_does_not_delete_the_stored_model(self, off):
        _store_model(off)
        with patch(EXPORT) as export:
            with pytest.raises(ExportDisabledError):
                ModelRegistry(off).pull("org/model", force=True)
        export.assert_not_called()
        assert (off.models_root / "fe" / "model" / "model.onnx").is_file()

    def test_a_corrupt_manifest_is_not_re_exported(self, off):
        model_dir = off.models_root / "fe" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "manifest.json").write_text("{not json")
        with patch(EXPORT) as export:
            with pytest.raises(ExportDisabledError):
                ModelRegistry(off).pull("org/model")
        export.assert_not_called()

    def test_the_error_survives_pickle_and_copy(self):
        error = ExportDisabledError("export is disabled")
        for restored in (pickle.loads(pickle.dumps(error)), copy.copy(error)):
            assert type(restored) is ExportDisabledError
            assert str(restored) == str(error)


class TestExportAllowedWithoutTheExporter:
    def test_says_how_to_install_the_exporter(self, tmp_path):
        settings = FloudsOnnxSettings(home_dir=tmp_path / "home", allow_export=True)
        missing = {"model_exporter": None, "model_exporter.export": None, "model_exporter.export.pipeline": None}
        with patch.dict(sys.modules, missing):
            with pytest.raises(ExporterNotInstalledError, match=r'pip install "flouds-onnx\[export\]"'):
                ModelRegistry(settings).pull("org/model")
        assert not (settings.models_root / "fe").exists()


class TestServer:
    def test_export_disabled_is_403(self):
        pytest.importorskip("fastapi")
        from fastapi import HTTPException

        from flouds_onnx.api.routers import models

        client = MagicMock()
        client.pull.side_effect = ExportDisabledError("export is disabled")
        req = models.PullRequest(model_name="org/model", model_for="fe")
        with patch.object(models, "get_default_client", return_value=client):
            with pytest.raises(HTTPException) as info:
                asyncio.run(models.pull_model(req))
        assert info.value.status_code == 403
        assert "export is disabled" in info.value.detail
