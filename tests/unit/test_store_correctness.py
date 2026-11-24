# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""The store records what really happened: names, export location, options, removal."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from flouds_onnx.config.settings import FloudsOnnxSettings
from flouds_onnx.exceptions import ExportError, ManifestError, ModelNameConflictError
from flouds_onnx.store.exporter_bridge import ExporterBridge
from flouds_onnx.store.manifest import MANIFEST_SCHEMA_VERSION, ModelManifest
from flouds_onnx.store.registry import ModelRegistry

EXPORT = "flouds_onnx.store.registry.ExporterBridge.export"


def _exporter_writing(files=("model.onnx", "tokenizer.json")):
    """A fake exporter that behaves like the real one: it writes to
    <ONNX root>/models/<model_for>/<folder>, wherever the store itself lives."""

    def fake_export(**kw):
        out = Path(ExporterBridge.onnx_root_for(kw["model_dir"])) / "models" / kw["model_for"] / Path(kw["model_dir"]).name
        out.mkdir(parents=True, exist_ok=True)
        for name in files:
            (out / name).write_text("x")
        return str(out)

    return fake_export


def _stored(settings, name, model_for="fe"):
    folder = name.split("/")[-1]
    model_dir = settings.models_root / model_for / folder
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_text("x")
    ModelManifest(model_name=name, model_for=model_for, folder=folder, onnx_files=["model.onnx"]).save(str(model_dir))
    return model_dir


class TestModelsFromDifferentOrganisations:
    def test_a_model_is_not_served_in_place_of_another_with_the_same_short_name(self, settings):
        _stored(settings, "org-a/bert-base")
        registry = ModelRegistry(settings)

        assert registry.exists("org-a/bert-base") is True
        assert registry.exists("org-b/bert-base") is False
        with pytest.raises(ModelNameConflictError, match="org-a/bert-base"):
            registry.get_manifest("org-b/bert-base")

    def test_pull_refuses_instead_of_returning_the_other_model(self, settings):
        _stored(settings, "org-a/bert-base")
        with patch(EXPORT) as export:
            with pytest.raises(ModelNameConflictError):
                ModelRegistry(settings).pull("org-b/bert-base")
        export.assert_not_called()

    def test_force_pull_replaces_the_stored_model(self, settings):
        _stored(settings, "org-a/bert-base")
        with patch(EXPORT, side_effect=_exporter_writing()):
            manifest = ModelRegistry(settings).pull("org-b/bert-base", force=True)
        assert manifest.model_name == "org-b/bert-base"

    def test_remove_does_not_delete_the_other_organisations_model(self, settings):
        model_dir = _stored(settings, "org-a/bert-base")
        with pytest.raises(ModelNameConflictError):
            ModelRegistry(settings).remove("org-b/bert-base")
        assert model_dir.is_dir()

    def test_short_names_and_case_still_address_the_stored_model(self, settings):
        _stored(settings, "Org-A/bert-base")
        registry = ModelRegistry(settings)
        assert registry.exists("bert-base") is True
        assert registry.get_manifest("org-a/bert-base").model_name == "Org-A/bert-base"
        assert registry.remove("bert-base") is True


class TestExportLandsInTheStore:
    def test_default_store_layout(self, settings):
        with patch(EXPORT, side_effect=_exporter_writing()):
            manifest = ModelRegistry(settings).pull("org/model")
        assert manifest.onnx_files == ["model.onnx"]
        assert (settings.models_root / "fe" / "model" / "manifest.json").is_file()

    def test_store_root_not_named_models(self, tmp_path):
        """onnx_path='/data/store': the exporter writes beside the store, in
        '/data/models/...'. The files must end up in the store with the manifest."""
        settings = FloudsOnnxSettings(home_dir=tmp_path / "home", onnx_path=tmp_path / "data" / "store")
        with patch(EXPORT, side_effect=_exporter_writing()):
            manifest = ModelRegistry(settings).pull("org/model", "s2s")

        model_dir = tmp_path / "data" / "store" / "s2s" / "model"
        assert manifest.onnx_files == ["model.onnx"]
        assert sorted(p.name for p in model_dir.iterdir()) == ["manifest.json", "model.onnx", "tokenizer.json"]
        assert not (tmp_path / "data" / "models").exists(), "the exporter's own folder should be cleaned up"

    def test_an_export_that_produces_no_model_is_an_error_and_leaves_no_manifest(self, settings):
        with patch(EXPORT, side_effect=_exporter_writing(files=("tokenizer.json",))):
            with pytest.raises(ExportError, match="no ONNX files"):
                ModelRegistry(settings).pull("org/model")
        assert not (settings.models_root / "fe" / "model" / "manifest.json").exists()
        assert ModelRegistry(settings).exists("org/model") is False

    def test_a_failed_export_leaves_no_empty_folders(self, tmp_path):
        from flouds_onnx.exceptions import ExporterNotInstalledError

        settings = FloudsOnnxSettings(home_dir=tmp_path / "home")
        with patch(EXPORT, side_effect=ExporterNotInstalledError()):
            with pytest.raises(ExporterNotInstalledError):
                ModelRegistry(settings).pull("org/model")
        assert not (settings.models_root / "fe").exists()

    def test_a_failed_export_does_not_remove_files_another_process_is_writing(self, settings):
        """The exporter exits when another process holds its lock on the same folder."""
        model_dir = settings.models_root / "fe" / "model"

        def other_process_is_exporting(**kw):
            (Path(kw["model_dir"]) / ".export.lock").write_text("held by another process")
            raise SystemExit(0)

        with patch(EXPORT, side_effect=other_process_is_exporting):
            with pytest.raises(SystemExit):
                ModelRegistry(settings).pull("org/model")
        assert (model_dir / ".export.lock").is_file()

    def test_reading_the_store_does_not_create_directories(self, tmp_path):
        settings = FloudsOnnxSettings(home_dir=tmp_path / "not-created")
        registry = ModelRegistry(settings)
        assert registry.list() == []
        assert registry.exists("org/model") is False
        assert registry.get_manifest("org/model") is None
        assert not (tmp_path / "not-created").exists()


class TestManifestIsTruthful:
    def test_export_options_are_the_ones_used(self, settings):
        with patch(EXPORT, side_effect=_exporter_writing()):
            manifest = ModelRegistry(settings).pull(
                "org/model",
                task="feature-extraction",
                optimize=True,
                optimization_level=2,
                opset_version=14,
                library="transformers",
                normalize_embeddings=True,
            )
        options = manifest.export_options
        assert (options.task, options.optimize, options.optimization_level, options.opset_version) == ("feature-extraction", True, 2, 14)
        assert (options.library, options.normalize_embeddings, options.device) == ("transformers", True, "cpu")

    def test_defaults_are_recorded_as_defaults_not_invented_values(self, settings):
        with patch(EXPORT, side_effect=_exporter_writing()):
            options = ModelRegistry(settings).pull("org/model").export_options
        assert options.optimize is False
        assert options.opset_version is None  # the exporter chose; flouds-onnx did not
        assert options.optimization_level is None  # meaningless when not optimising

    def test_manifest_has_a_schema_version_and_older_ones_load_as_version_1(self, tmp_path):
        manifest = ModelManifest(model_name="m", model_for="fe", folder="m")
        manifest.save(str(tmp_path))
        assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == MANIFEST_SCHEMA_VERSION

        legacy = {"model_name": "m", "model_for": "fe", "folder": "m", "export_options": {"optimize": True, "opset_version": 17, "device": "cpu"}}
        (tmp_path / "manifest.json").write_text(json.dumps(legacy))
        loaded = ModelManifest.load(str(tmp_path))
        assert loaded.schema_version == 1
        assert loaded.export_options.opset_version == 17

    def test_a_failed_write_keeps_the_previous_manifest(self, tmp_path):
        ModelManifest(model_name="first", model_for="fe", folder="m").save(str(tmp_path))
        with patch("flouds_onnx.store.manifest.os.replace", side_effect=OSError("disk full")):
            with pytest.raises(ManifestError):
                ModelManifest(model_name="second", model_for="fe", folder="m").save(str(tmp_path))
        assert ModelManifest.load(str(tmp_path)).model_name == "first"
        assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"], "no temporary file left behind"


class TestRemoveTellsTheTruth:
    def test_returns_false_when_the_directory_could_not_be_deleted(self, settings):
        model_dir = _stored(settings, "org/model")
        with patch("flouds_onnx.store.registry.shutil.rmtree"):  # as when Windows holds model.onnx open
            assert ModelRegistry(settings).remove("org/model") is False
        assert model_dir.is_dir()

    def test_returns_true_when_deleted_and_false_when_absent(self, settings):
        _stored(settings, "org/model")
        registry = ModelRegistry(settings)
        assert registry.remove("org/model") is True
        assert registry.remove("org/model") is False


class TestConcurrentExportsGetTheirOwnRootAndTakeTurns:
    def test_each_export_gets_its_own_root_and_exports_do_not_overlap(self, tmp_path, monkeypatch):
        seen = {}
        both_inside = threading.Barrier(2, timeout=1)

        def fake_export(**kw):
            try:
                both_inside.wait()  # succeeds only if two exports run at the same time
                overlapped = True
            except threading.BrokenBarrierError:
                overlapped = False
            seen[kw["model_name"]] = (kw["onnx_path"], overlapped)
            return "ok"

        import sys
        import types

        pipeline = types.ModuleType("model_exporter.export.pipeline")
        pipeline.export = fake_export
        for name, module in (
            ("model_exporter", types.ModuleType("model_exporter")),
            ("model_exporter.export", types.ModuleType("model_exporter.export")),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setitem(sys.modules, "model_exporter.export.pipeline", pipeline)
        monkeypatch.delenv("ONNX_PATH", raising=False)

        dirs = {"a": tmp_path / "storeA" / "models" / "fe" / "a", "b": tmp_path / "storeB" / "models" / "fe" / "b"}
        threads = [
            threading.Thread(target=ExporterBridge.export, kwargs={"model_name": n, "model_dir": str(d), "model_for": "fe", "task": "t"})
            for n, d in dirs.items()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for name, model_dir in dirs.items():
            root, overlapped = seen[name]
            assert root == str(model_dir.parents[2])
            # The exporter changes process-wide state during an export, so they take turns.
            assert overlapped is False
        assert "ONNX_PATH" not in os.environ
