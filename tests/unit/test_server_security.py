# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""Access control, request limits and error handling of the optional HTTP server."""
from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from flouds_onnx.api import client as client_module  # noqa: E402
from flouds_onnx.api import security  # noqa: E402
from flouds_onnx.api.routers import models as models_router  # noqa: E402
from flouds_onnx.api.server import create_app, run_server  # noqa: E402
from flouds_onnx.exceptions import ExportError, ModelLoadError, ModelNotFoundError, ServerConfigurationError  # noqa: E402

TOKEN = "unit-test-token"
PROTECTED = [
    ("get", "/api/v1/models", None),
    ("get", "/api/v1/models/some-model", None),
    ("get", "/api/v1/stats", None),
    ("post", "/api/v1/models/pull", {"model_name": "m"}),
    ("post", "/api/v1/models/create", {"model_name": "m"}),
    ("post", "/api/v1/models/load", {"model_name": "m"}),
    ("post", "/api/v1/models/reload", {"model_name": "m"}),
    ("post", "/api/v1/models/unload", {"model_name": "m"}),
    ("delete", "/api/v1/models/some-model", None),
]


@pytest.fixture(autouse=True)
def _no_ambient_configuration(monkeypatch):
    monkeypatch.delenv(security.API_TOKEN_ENV, raising=False)
    monkeypatch.delenv(security.ALLOWED_ORIGINS_ENV, raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)


@pytest.fixture()
def fake_client(monkeypatch):
    client = MagicMock()
    client.list.return_value = []
    client.get_manifest.return_value = None
    client.cache_stats.return_value = {}
    client.unload.return_value = False
    client.remove.return_value = False
    monkeypatch.setattr(models_router, "get_default_client", lambda: client)
    monkeypatch.setattr(client_module, "_default_client", client)
    return client


def _call(http, method, path, body, **kwargs):
    return getattr(http, method)(path, **({"json": body} if body is not None else {}), **kwargs)


class TestTokenMode:
    @pytest.mark.parametrize("method, path, body", PROTECTED)
    def test_every_model_route_needs_the_token(self, fake_client, method, path, body):
        http = TestClient(create_app(api_token=TOKEN))
        assert _call(http, method, path, body).status_code == 401
        assert _call(http, method, path, body, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert _call(http, method, path, body, headers={"Authorization": TOKEN}).status_code == 401
        fake_client.pull.assert_not_called()
        fake_client.remove.assert_not_called()

    def test_valid_token_is_accepted(self, fake_client):
        http = TestClient(create_app(api_token=TOKEN))  # TestClient is not a loopback client: the token alone admits it
        assert http.get("/api/v1/models", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200

    def test_token_is_read_from_the_environment(self, fake_client, monkeypatch):
        monkeypatch.setenv(security.API_TOKEN_ENV, TOKEN)
        http = TestClient(create_app())
        assert http.get("/api/v1/models").status_code == 401
        assert http.get("/api/v1/models", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200

    def test_health_needs_no_token(self, fake_client):
        assert TestClient(create_app(api_token=TOKEN)).get("/health").status_code == 200

    def test_remote_clients_are_served_when_they_have_the_token(self, fake_client):
        http = TestClient(create_app(api_token=TOKEN), client=("203.0.113.9", 50000))
        response = http.get("/api/v1/models", headers={"Authorization": f"Bearer {TOKEN}", "Host": "models.example.com"})
        assert response.status_code == 200


class TestLocalModeWithoutAToken:
    def test_local_requests_are_served(self, fake_client):
        http = TestClient(create_app(), base_url="http://127.0.0.1:19720", client=("127.0.0.1", 50000))
        assert http.get("/api/v1/models").status_code == 200

    def test_requests_from_another_machine_are_refused(self, fake_client):
        http = TestClient(create_app(), base_url="http://127.0.0.1:19720", client=("203.0.113.9", 50000))
        response = http.delete("/api/v1/models/some-model")
        assert response.status_code == 403
        assert security.API_TOKEN_ENV in response.json()["detail"]
        fake_client.remove.assert_not_called()

    def test_dns_rebinding_is_refused(self, fake_client):
        """The connection is local, but the browser was sent here under another name."""
        http = TestClient(create_app(), base_url="http://attacker.example:19720", client=("127.0.0.1", 50000))
        assert http.post("/api/v1/models/pull", json={"model_name": "m"}).status_code == 403
        fake_client.pull.assert_not_called()


class TestWebOrigins:
    def test_a_web_page_on_another_origin_cannot_drive_the_server(self, fake_client):
        http = TestClient(create_app(), base_url="http://127.0.0.1:19720", client=("127.0.0.1", 50000))
        response = http.delete("/api/v1/models/some-model", headers={"Origin": "https://evil.example"})
        assert response.status_code == 403
        fake_client.remove.assert_not_called()

    def test_another_origin_is_refused_even_with_the_token(self, fake_client):
        http = TestClient(create_app(api_token=TOKEN))
        headers = {"Authorization": f"Bearer {TOKEN}", "Origin": "https://evil.example"}
        assert http.get("/api/v1/models", headers=headers).status_code == 403

    def test_the_servers_own_origin_and_listed_origins_are_allowed(self, fake_client):
        http = TestClient(create_app(allowed_origins=["https://console.example/"]), base_url="http://127.0.0.1:19720", client=("127.0.0.1", 50000))
        assert http.get("/api/v1/models", headers={"Origin": "http://127.0.0.1:19720"}).status_code == 200
        assert http.get("/api/v1/models", headers={"Origin": "https://console.example"}).status_code == 200


class TestStartup:
    def test_refuses_to_listen_beyond_this_machine_without_a_token(self, monkeypatch):
        started = MagicMock()
        monkeypatch.setitem(sys.modules, "uvicorn", MagicMock(run=started))
        for host in ("0.0.0.0", "192.168.1.10", "::"):  # nosec B104
            with pytest.raises(ServerConfigurationError, match=security.API_TOKEN_ENV):
                run_server(host=host)
        started.assert_not_called()

    def test_listens_beyond_this_machine_with_a_token_and_locally_without(self, monkeypatch):
        started = MagicMock()
        monkeypatch.setitem(sys.modules, "uvicorn", MagicMock(run=started))
        run_server(host="127.0.0.1")
        run_server(host="localhost")
        monkeypatch.setenv(security.API_TOKEN_ENV, TOKEN)
        run_server(host="0.0.0.0")  # nosec B104
        assert started.call_count == 3

    @pytest.mark.parametrize(
        "host, expected", [("127.0.0.1", True), ("127.0.0.1:19720", True), ("localhost:80", True), ("[::1]:19720", True), ("::1", True)]
    )
    def test_loopback_hosts(self, host, expected):
        assert security.is_loopback_host(host) is expected

    @pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "example.com", "127.0.0.1.evil.example", "", None])  # nosec B104
    def test_non_loopback_hosts(self, host):
        assert security.is_loopback_host(host) is False


class TestRequestsCannotAskForDangerousOptions:
    def _local(self):
        return TestClient(create_app(), base_url="http://127.0.0.1:19720", client=("127.0.0.1", 50000))

    @pytest.mark.parametrize("field, value", [("trust_remote_code", True), ("hf_token", "hf_secret"), ("anything_else", 1)])
    def test_fields_not_offered_over_http_are_rejected(self, fake_client, field, value):
        response = self._local().post("/api/v1/models/pull", json={"model_name": "m", field: value})
        assert response.status_code == 422
        fake_client.pull.assert_not_called()

    def test_remote_code_is_always_off_and_the_token_comes_from_the_server(self, fake_client, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "server-side-token")
        manifest = MagicMock()
        manifest.model_dump.return_value = {"model_name": "m"}
        fake_client.pull.return_value = manifest
        assert self._local().post("/api/v1/models/pull", json={"model_name": "m", "library": "transformers"}).status_code == 200
        kwargs = fake_client.pull.call_args.kwargs
        assert kwargs["trust_remote_code"] is False
        assert kwargs["hf_token"] == "server-side-token"
        assert kwargs["library"] == "transformers"

    def test_create_does_not_reveal_the_servers_filesystem_path(self, fake_client):
        loaded = MagicMock(model_name="m", model_for="fe", model_dir="/srv/secret/path/fe/m", is_seq2seq=False)
        loaded.session_strategy.value = "ort_inference_session"
        fake_client.create_model.return_value = loaded
        body = self._local().post("/api/v1/models/create", json={"model_name": "m"}).json()
        assert body["status"] == "created"
        assert "model_dir" not in body and "/srv/secret" not in str(body)


class TestErrorsAndResponsiveness:
    def _local(self):
        return TestClient(create_app(), base_url="http://127.0.0.1:19720", client=("127.0.0.1", 50000))

    def test_missing_model_is_404_not_500(self, fake_client):
        fake_client.load_model.side_effect = ModelNotFoundError("Model 'm' (model_for=fe) not in local store.")
        response = self._local().post("/api/v1/models/load", json={"model_name": "m"})
        assert response.status_code == 404

    def test_internal_errors_do_not_leak_paths_or_third_party_output(self, fake_client):
        fake_client.pull.side_effect = ExportError(
            "Export failed for 'm'", detail="FileNotFoundError: C:\\Users\\svc\\.flouds\\models\\fe\\m\\config.json"
        )
        fake_client.load_model.side_effect = ModelLoadError("ort.InferenceSession failed for '/srv/models/fe/m/model.onnx'", detail="stack")
        for path in ("/api/v1/models/pull", "/api/v1/models/load"):
            response = self._local().post(path, json={"model_name": "m"})
            assert response.status_code == 500
            text = response.text
            assert "Users" not in text and "/srv/models" not in text and "config.json" not in text

    def test_a_slow_export_does_not_block_other_requests(self, fake_client):
        """Exports run in a worker thread, so /health still answers while one is in flight."""
        started, release = threading.Event(), threading.Event()

        def slow_pull(*_args, **_kwargs):
            started.set()
            release.wait(timeout=10)
            manifest = MagicMock()
            manifest.model_dump.return_value = {"model_name": "m"}
            return manifest

        fake_client.pull.side_effect = slow_pull
        http = self._local()
        result = {}
        worker = threading.Thread(target=lambda: result.update(pull=http.post("/api/v1/models/pull", json={"model_name": "m"}).status_code))
        worker.start()
        try:
            assert started.wait(timeout=5)
            assert http.get("/health").status_code == 200
            assert not release.is_set()
        finally:
            release.set()
            worker.join(timeout=10)
        assert result["pull"] == 200
