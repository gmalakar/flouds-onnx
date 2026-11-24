# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""Command-line exit codes and the server's health and readiness checks."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import flouds_onnx
from flouds_onnx.api.routers.health import store_status
from flouds_onnx.cli import main as cli
from flouds_onnx.exceptions import InvalidModelReferenceError


def _run(monkeypatch, argv, client):
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(sys, "argv", ["flouds-onnx", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    return exit_info.value.code


class TestCli:
    def test_version_flag(self, monkeypatch, capsys):
        assert _run(monkeypatch, ["--version"], MagicMock()) == 0
        assert flouds_onnx.__version__ in capsys.readouterr().out

    def test_remove_succeeds_with_exit_code_0(self, monkeypatch):
        client = MagicMock()
        client.remove.return_value = True
        assert _run(monkeypatch, ["remove", "org/model"], client) == 0

    def test_remove_of_a_missing_model_is_a_failure(self, monkeypatch, capsys):
        client = MagicMock()
        client.get_manifest.return_value = None
        client.remove.return_value = False
        assert _run(monkeypatch, ["remove", "org/model"], client) == 1
        assert "not in the local store" in capsys.readouterr().err

    def test_remove_that_could_not_delete_says_so(self, monkeypatch, capsys):
        client = MagicMock()
        client.get_manifest.return_value = MagicMock()
        client.remove.return_value = False
        assert _run(monkeypatch, ["remove", "org/model"], client) == 1
        assert "could not be fully deleted" in capsys.readouterr().err

    def test_invalid_model_name_is_reported_without_a_traceback(self, monkeypatch, capsys):
        client = MagicMock()
        client.get_manifest.side_effect = InvalidModelReferenceError("model_name='..' is not a valid model name")
        assert _run(monkeypatch, ["info", ".."], client) == 2
        err = capsys.readouterr().err
        assert "not a valid model name" in err and "Traceback" not in err

    def test_pull_takes_the_hub_token_from_the_environment(self, monkeypatch):
        client = MagicMock()
        monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "from-environment")
        _run(monkeypatch, ["pull", "org/model"], client)
        assert client.pull.call_args.kwargs["hf_token"] == "from-environment"

    def test_pull_without_any_token_passes_none(self, monkeypatch):
        client = MagicMock()
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        _run(monkeypatch, ["pull", "org/model"], client)
        assert client.pull.call_args.kwargs["hf_token"] is None


class TestStoreStatus:
    def test_missing_store_is_usable(self, tmp_path):
        assert store_status(str(tmp_path / "not-there")) == (True, "not created yet")

    def test_existing_store_is_usable(self, tmp_path):
        assert store_status(str(tmp_path)) == (True, "ok")

    def test_a_file_where_the_store_should_be_is_not_usable(self, tmp_path):
        path = tmp_path / "models"
        path.write_text("not a directory")
        ok, reason = store_status(str(path))
        assert ok is False and "not a directory" in reason


class TestHealthRoutes:
    def _http(self, settings, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from flouds_onnx.api import client as client_module
        from flouds_onnx.api.client import FloudsOnnxClient
        from flouds_onnx.api.server import create_app

        monkeypatch.setattr(client_module, "_default_client", FloudsOnnxClient(settings))
        return TestClient(create_app(api_token="token"))

    def test_health_and_readiness_need_no_token(self, settings, monkeypatch):
        http = self._http(settings, monkeypatch)
        assert http.get("/health").json() == {"status": "ok", "service": "flouds-onnx", "version": flouds_onnx.__version__}
        assert http.get("/health/ready").json()["status"] == "ready"

    def test_readiness_is_503_when_the_store_is_unusable(self, tmp_path, monkeypatch):
        from flouds_onnx.config.settings import FloudsOnnxSettings

        blocked = tmp_path / "models"
        blocked.write_text("not a directory")
        http = self._http(FloudsOnnxSettings(home_dir=tmp_path), monkeypatch)
        response = http.get("/health/ready")
        assert response.status_code == 503 and response.json()["status"] == "not_ready"
        assert http.get("/health").status_code == 200, "liveness must not depend on the store"
