# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""Settings from the environment, ONNX Runtime options, provider rules, tokenizer cache."""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.config.settings import FloudsOnnxSettings
from floudsonnx.exceptions import ModelLoadError, TokenizerError
from floudsonnx.runtime.loader import ModelLoader
from floudsonnx.runtime.ort_session import OrtInferenceSessionLoader, build_session_options, resolve_provider
from floudsonnx.runtime.session_pool import SessionPool
from floudsonnx.runtime.tokenizer_cache import TokenizerCache


class TestSettingsFromEnvironment:
    def test_reads_every_supported_variable(self, tmp_path):
        env = {
            "FLOUDSONNX_HOME_DIR": str(tmp_path / "home"),
            "FLOUDSONNX_ONNX_PATH": str(tmp_path / "store"),
            "FLOUDSONNX_SESSION_PROVIDER": "CUDAExecutionProvider",
            "FLOUDSONNX_STRICT_PROVIDER": "true",
            "FLOUDSONNX_INTRA_OP_THREADS": "4",
            "FLOUDSONNX_INTER_OP_THREADS": "1",
            "FLOUDSONNX_ENCODER_CACHE_MAX": "9",
            "FLOUDSONNX_SEQ2SEQ_CACHE_MAX": "2",
            "FLOUDSONNX_TOKENIZER_CACHE_MAX": "7",
            "FLOUDSONNX_ALLOW_HUB_TOKENIZER_FALLBACK": "1",
        }
        s = FloudsOnnxSettings.from_env(env)
        assert s.models_root == tmp_path / "store"
        assert (s.session_provider, s.strict_provider) == ("CUDAExecutionProvider", True)
        assert (s.intra_op_num_threads, s.inter_op_num_threads) == (4, 1)
        assert (s.encoder_cache_max, s.seq2seq_cache_max, s.tokenizer_cache_max) == (9, 2, 7)
        assert s.allow_hub_tokenizer_fallback is True

    def test_unset_and_empty_variables_keep_defaults_and_overrides_win(self):
        defaults = FloudsOnnxSettings()
        s = FloudsOnnxSettings.from_env({"FLOUDSONNX_SESSION_PROVIDER": "  ", "FLOUDSONNX_ENCODER_CACHE_MAX": "9"}, encoder_cache_max=2)
        assert s.session_provider == defaults.session_provider
        assert s.encoder_cache_max == 2
        assert s.strict_provider is False and s.allow_hub_tokenizer_fallback is False
        assert s.intra_op_num_threads is None

    @pytest.mark.parametrize(
        "env", [{"FLOUDSONNX_INTRA_OP_THREADS": "0"}, {"FLOUDSONNX_ENCODER_CACHE_MAX": "0"}, {"FLOUDSONNX_INTER_OP_THREADS": "many"}]
    )
    def test_invalid_values_are_rejected(self, env):
        with pytest.raises(ValueError):
            FloudsOnnxSettings.from_env(env)

    def test_default_client_uses_the_environment(self, tmp_path, monkeypatch):
        from floudsonnx.api import client as client_module

        monkeypatch.setenv("FLOUDSONNX_HOME_DIR", str(tmp_path / "from-env"))
        monkeypatch.setattr(client_module, "_default_client", None)
        assert client_module.get_default_client().settings.home_dir == tmp_path / "from-env"
        monkeypatch.setattr(client_module, "_default_client", None)


class TestSessionOptionsAndProviders:
    def test_no_thread_settings_means_onnx_runtime_defaults(self):
        assert build_session_options() is None

    def test_thread_limits_are_applied(self):
        options = build_session_options(intra_op_num_threads=2, inter_op_num_threads=1)
        assert (options.intra_op_num_threads, options.inter_op_num_threads) == (2, 1)

    def test_loader_passes_thread_limits_to_the_session(self, tmp_path):
        (tmp_path / "model.onnx").write_text("x")
        with patch("floudsonnx.runtime.ort_session.ort.InferenceSession") as session:
            OrtInferenceSessionLoader.load(str(tmp_path), ModelConfig(model_name="m"), intra_op_num_threads=3)
        assert session.call_args.kwargs["sess_options"].intra_op_num_threads == 3

    def test_settings_reach_the_session_loader(self, tmp_path):
        settings = FloudsOnnxSettings(home_dir=tmp_path, intra_op_num_threads=2, inter_op_num_threads=1, strict_provider=True)
        loader = ModelLoader(settings)
        with patch("floudsonnx.runtime.session_pool.OrtInferenceSessionLoader.load", return_value=MagicMock()) as load:
            loader._pool.get_or_load_encoder(str(tmp_path / "m"), ModelConfig(model_name="m"))
        assert load.call_args.kwargs == {"strict_provider": True, "intra_op_num_threads": 2, "inter_op_num_threads": 1}

    def test_unavailable_provider_falls_back_to_cpu_by_default(self):
        with patch("floudsonnx.runtime.ort_session.ort.get_available_providers", return_value=["CPUExecutionProvider"]):
            assert resolve_provider("CUDAExecutionProvider") == "CPUExecutionProvider"

    def test_unavailable_provider_is_an_error_when_strict(self):
        with patch("floudsonnx.runtime.ort_session.ort.get_available_providers", return_value=["CPUExecutionProvider"]):
            with pytest.raises(ModelLoadError, match="CUDAExecutionProvider"):
                resolve_provider("CUDAExecutionProvider", strict=True)

    def test_seq2seq_models_use_the_configured_provider(self, tmp_path):
        optimum = pytest.importorskip("optimum.onnxruntime")
        with (
            patch.object(optimum.ORTModelForSeq2SeqLM, "from_pretrained", return_value=MagicMock()) as from_pretrained,
            patch("floudsonnx.runtime.ort_session.ort.get_available_providers", return_value=["CPUExecutionProvider", "CUDAExecutionProvider"]),
        ):
            pool = SessionPool()
            pool.get_or_load_seq2seq(str(tmp_path), ModelConfig(model_name="t5", model_for="s2s"), "CUDAExecutionProvider")
            pool.get_or_load_seq2seq(str(tmp_path), ModelConfig(model_name="t5", model_for="s2s"), "CPUExecutionProvider")
        gpu_call, cpu_call = from_pretrained.call_args_list
        assert gpu_call.kwargs["provider"] == "CUDAExecutionProvider"
        assert "provider" not in cpu_call.kwargs
        assert pool.evict(str(tmp_path)) == 2  # one cached model per provider


class TestTokenizerCache:
    def _cache(self, **kwargs):
        cache = TokenizerCache(**kwargs)
        loads = []

        def fake_load(model_dir, use_legacy, allow_hub_fallback=False):
            loads.append(Path(model_dir).name)
            return object()

        return cache, loads, patch.object(TokenizerCache, "_load", side_effect=fake_load)

    def test_evict_makes_every_thread_reload(self, tmp_path):
        """A reload or removal used to reach only the thread that requested it."""
        model_dir = str(tmp_path / "m")
        cache, loads, patched = self._cache()
        seen = {}
        loaded, evicted = threading.Event(), threading.Event()

        def worker():
            seen["before"] = cache.get_or_load(model_dir)
            loaded.set()
            evicted.wait(timeout=5)
            seen["after"] = cache.get_or_load(model_dir)

        with patched:
            thread = threading.Thread(target=worker)
            thread.start()
            assert loaded.wait(timeout=5)
            cache.evict(model_dir)  # from the main thread
            evicted.set()
            thread.join(timeout=5)

        assert seen["after"] is not seen["before"]
        assert loads == ["m", "m"]

    def test_evicting_one_model_does_not_reload_others(self, tmp_path):
        cache, loads, patched = self._cache()
        with patched:
            kept = cache.get_or_load(str(tmp_path / "all-MiniLM-L6-v2"))
            cache.get_or_load(str(tmp_path / "all-MiniLM-L6"))
            cache.evict(str(tmp_path / "all-MiniLM-L6"))
            assert cache.get_or_load(str(tmp_path / "all-MiniLM-L6-v2")) is kept

    def test_each_thread_keeps_at_most_the_configured_number(self, tmp_path):
        cache, loads, patched = self._cache(max_per_thread=2)
        with patched:
            for name in ("a", "b", "c"):
                cache.get_or_load(str(tmp_path / name))
            assert len(cache._store()) == 2
            cache.get_or_load(str(tmp_path / "c"))  # still cached
            cache.get_or_load(str(tmp_path / "a"))  # evicted as least recently used
        assert loads == ["a", "b", "c", "a"]

    def test_no_download_unless_enabled(self, tmp_path):
        model_dir = tmp_path / "all-MiniLM-L6-v2"
        model_dir.mkdir()
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("no tokenizer files")) as from_pretrained:
            with pytest.raises(TokenizerError, match="allow_hub_tokenizer_fallback"):
                TokenizerCache().get_or_load(str(model_dir))
        assert all(call.kwargs.get("local_files_only") is True for call in from_pretrained.call_args_list)

    def test_download_happens_when_enabled(self, tmp_path):
        model_dir = tmp_path / "all-MiniLM-L6-v2"
        model_dir.mkdir()
        hub_tokenizer = object()

        def from_pretrained(name, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("no tokenizer files")
            assert name == "sentence-transformers/all-MiniLM-L6-v2"
            return hub_tokenizer

        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=from_pretrained):
            assert TokenizerCache(allow_hub_fallback=True).get_or_load(str(model_dir)) is hub_tokenizer

    def test_failure_reports_why_each_attempt_failed(self, tmp_path):
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=PermissionError("access denied")):
            with pytest.raises(TokenizerError) as raised:
                TokenizerCache().get_or_load(str(tmp_path))
        assert "PermissionError" in str(raised.value) and "access denied" in str(raised.value)
