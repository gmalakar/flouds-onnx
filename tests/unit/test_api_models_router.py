# =============================================================================
# File: test_api_models_router.py
# Date Created: 2026-11-21
# Date Updated: 2026-11-21
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from flouds_onnx.runtime.strategy import SessionStrategy

pytest.importorskip("fastapi")

from flouds_onnx.api.routers import models  # noqa: E402


class TestModelsRouter:
    def test_pull_request_exposes_export_options(self):
        req = models.PullRequest(
            model_name="org/model",
            model_for="fe",
            task="feature-extraction",
            library="transformers",
            normalize_embeddings=True,
            device="cpu",
            use_subprocess=True,
            merge=True,
            skip_validator=True,
        )

        assert req.library == "transformers"
        assert req.normalize_embeddings is True
        assert req.use_subprocess is True
        assert req.merge is True
        assert req.skip_validator is True

    def test_create_model_forwards_request_options(self):
        loaded = MagicMock()
        loaded.model_name = "org/model"
        loaded.model_for = "fe"
        loaded.model_dir = "/models/fe/model"
        loaded.session_strategy = SessionStrategy.ORT_INFERENCE_SESSION
        loaded.is_seq2seq = False

        client = MagicMock()
        client.create_model.return_value = loaded

        req = models.PullRequest(
            model_name="org/model",
            model_for="fe",
            task="feature-extraction",
            library="transformers",
            normalize_embeddings=True,
        )

        with patch.object(models, "get_default_client", return_value=client):
            result = asyncio.run(models.create_model(req))

        assert result["status"] == "created"
        assert result["strategy"] == "ort_inference_session"
        client.create_model.assert_called_once()
        kwargs = client.create_model.call_args.kwargs
        assert kwargs["library"] == "transformers"
        assert kwargs["normalize_embeddings"] is True
