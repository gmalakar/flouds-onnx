# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""A model name or model type must never address anything outside the model store."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from floudsonnx.exceptions import FloudsOnnxError, InvalidModelReferenceError
from floudsonnx.runtime.loader import ModelLoader
from floudsonnx.runtime.session_pool import SessionPool
from floudsonnx.runtime.tokenizer_cache import TokenizerCache
from floudsonnx.store.manifest import ModelManifest
from floudsonnx.store.registry import ModelRegistry
from floudsonnx.utils.path_guard import safe_model_folder, safe_model_for

BAD_NAMES = ["..", ".", "...", "org/..", "org/.", "../..", "", "   ", "/", "model.", "a/b/.."]
BAD_TYPES = ["..", ".", "../..", "..\\..", "fe/..", "fe\\x", "/abs", "C:", "", " fe", "fe ", "a" * 33, "fe.d"]


def _store_model(settings, model_for: str, folder: str) -> Path:
    model_dir = settings.models_root / model_for / folder
    model_dir.mkdir(parents=True)
    ModelManifest(model_name=folder, model_for=model_for, folder=folder).save(str(model_dir))
    return model_dir


class TestNameAndTypeValidation:
    @pytest.mark.parametrize("name", BAD_NAMES)
    def test_unsafe_model_names_are_refused(self, name):
        with pytest.raises(InvalidModelReferenceError):
            safe_model_folder(name)

    @pytest.mark.parametrize("model_for", BAD_TYPES)
    def test_unsafe_model_types_are_refused(self, model_for):
        with pytest.raises(InvalidModelReferenceError):
            safe_model_for(model_for)

    @pytest.mark.parametrize("model_for", ["fe", "sc", "ranker", "s2s", "llm", "custom_type", "Custom-2"])
    def test_built_in_and_custom_types_are_accepted(self, model_for):
        assert safe_model_for(model_for) == model_for

    @pytest.mark.parametrize("name", ["t5-small", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "org/model.v2", ".hidden"])
    def test_real_model_names_are_accepted(self, name):
        assert safe_model_folder(name) == name.split("/")[-1]

    def test_error_is_both_a_library_error_and_a_value_error(self):
        with pytest.raises(ValueError):
            safe_model_folder("..")
        with pytest.raises(FloudsOnnxError):
            safe_model_for("../..")


class TestRegistryStaysInsideTheStore:
    def test_remove_cannot_delete_the_store_through_the_model_name(self, settings):
        """remove('..') used to delete <models_root> itself."""
        kept = _store_model(settings, "fe", "model-a")
        registry = ModelRegistry(settings)
        for name in ("..", ".", "org/.."):
            with pytest.raises(InvalidModelReferenceError):
                registry.remove(name, "fe")
        assert kept.is_dir() and settings.models_root.is_dir()

    def test_remove_cannot_reach_outside_the_store_through_the_model_type(self, settings, tmp_path):
        """model_for='../..' used to delete a sibling of the store."""
        victim = tmp_path / "victim"
        victim.mkdir()
        (victim / "important.txt").write_text("data")
        (victim / "manifest.json").write_text("{}")
        escape = os.path.relpath(tmp_path, settings.models_root)
        registry = ModelRegistry(settings)
        with pytest.raises(InvalidModelReferenceError):
            registry.remove("victim", escape)
        assert (victim / "important.txt").read_text() == "data"

    @pytest.mark.parametrize("call", ["exists", "get_manifest", "model_dir_path", "remove"])
    def test_every_lookup_refuses_a_traversing_type(self, settings, call):
        registry = ModelRegistry(settings)
        with pytest.raises(InvalidModelReferenceError):
            getattr(registry, call)("some-model", "../..")

    def test_pull_refuses_before_exporting_or_deleting(self, settings, tmp_path, mocker):
        export = mocker.patch("floudsonnx.store.registry.ExporterBridge.export")
        rmtree = mocker.patch("floudsonnx.store.registry.shutil.rmtree")
        registry = ModelRegistry(settings)
        for name, model_for in (("..", "fe"), ("some-model", "../.."), ("some-model", "fe/../../x")):
            with pytest.raises(InvalidModelReferenceError):
                registry.pull(name, model_for, force=True)
        export.assert_not_called()
        rmtree.assert_not_called()

    def test_loader_refuses_too(self, settings):
        loader = ModelLoader(settings)
        for method in (loader.remove, loader.unload, loader.is_loaded, loader.load_model, loader.reload):
            with pytest.raises(InvalidModelReferenceError):
                method("..", "fe")

    def test_normal_models_still_resolve_inside_the_store(self, settings):
        model_dir = _store_model(settings, "fe", "all-MiniLM-L6-v2")
        registry = ModelRegistry(settings)
        assert Path(registry.model_dir_path("sentence-transformers/all-MiniLM-L6-v2", "fe")) == model_dir
        assert registry.exists("sentence-transformers/all-MiniLM-L6-v2", "fe")
        assert registry.remove("sentence-transformers/all-MiniLM-L6-v2", "fe") is True
        assert not model_dir.exists()


class TestEvictionMatchesOneModelOnly:
    """Folder names that share a prefix: all-MiniLM-L6 and all-MiniLM-L6-v2."""

    def _dirs(self, tmp_path):
        return str(tmp_path / "fe" / "all-MiniLM-L6"), str(tmp_path / "fe" / "all-MiniLM-L6-v2")

    def test_session_eviction_leaves_the_longer_named_model_alone(self, tmp_path):
        short, long = self._dirs(tmp_path)
        pool = SessionPool()
        pool._encoder_cache.put(pool._enc_key(short, "CPUExecutionProvider"), object())
        pool._encoder_cache.put(pool._enc_key(long, "CPUExecutionProvider"), object())
        pool._seq2seq_cache.put(pool._seq2seq_key(long), object())

        assert pool.evict(short) == 1
        assert pool.contains(long) is True
        assert pool.contains(short) is False
        assert pool.evict(long) == 2

    def test_is_loaded_does_not_report_a_model_because_a_longer_named_one_is_loaded(self, settings):
        loader = ModelLoader(settings)
        long_dir = loader._registry.model_dir_path("all-MiniLM-L6-v2", "fe")
        loader._pool._encoder_cache.put(loader._pool._enc_key(long_dir, "CPUExecutionProvider"), object())
        assert loader.is_loaded("all-MiniLM-L6-v2", "fe") is True
        assert loader.is_loaded("all-MiniLM-L6", "fe") is False

    def test_tokenizer_eviction_leaves_the_longer_named_model_alone(self, tmp_path):
        short, long = self._dirs(tmp_path)
        cache = TokenizerCache()
        cache._store()[cache._key(short, False)] = "short"
        cache._store()[cache._key(long, False)] = "long"
        cache.evict(short)
        assert list(cache._store().values()) == ["long"]


class TestServerReportsBadReferencesAsBadRequests:
    def test_traversal_attempts_get_400_and_delete_nothing(self, settings, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from floudsonnx.api import client as client_module
        from floudsonnx.api.client import FloudsOnnxClient
        from floudsonnx.api.server import create_app

        kept = _store_model(settings, "fe", "model-a")
        monkeypatch.setattr(client_module, "_default_client", FloudsOnnxClient(settings))
        http = TestClient(create_app(), base_url="http://127.0.0.1:19720", client=("127.0.0.1", 50000))

        assert http.delete("/api/v1/models/model-a", params={"model_for": "../.."}).status_code == 400
        assert http.get("/api/v1/models/model-a", params={"model_for": "..\\.."}).status_code == 400
        assert http.post("/api/v1/models/unload", json={"model_name": "..", "model_for": "fe"}).status_code == 400
        assert http.post("/api/v1/models/load", json={"model_name": "x", "model_for": "../.."}).status_code == 400
        assert kept.is_dir()
