"""Tests for the validated Ollama HTTP boundary."""

import json

import httpx
import pytest

from local_file_agent.config import Settings
from local_file_agent.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)


def test_client_validates_successful_api_responses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.32.6"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "embeddinggemma"}, {"name": "qwen3.5:4b"}]},
            )
        if request.url.path == "/api/embed":
            body = json.loads(request.content)
            assert body["truncate"] is False
            return httpx.Response(
                200,
                json={
                    "model": "embeddinggemma",
                    "embeddings": [[0.1, -0.2, 0.3]],
                    "total_duration": 2_500_000,
                    "load_duration": 1_000_000,
                },
            )
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            assert body["stream"] is False
            assert body["options"]["temperature"] == 0
            return httpx.Response(
                200,
                json={
                    "model": "qwen3.5:4b",
                    "message": {"role": "assistant", "content": '{"status":"ok"}'},
                    "total_duration": 4_000_000,
                    "load_duration": 2_000_000,
                },
            )
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "qwen3.5:4b", "size_vram": 1024},
                        {"name": "embeddinggemma", "size_vram": 512},
                    ]
                },
            )
        return httpx.Response(404)

    with OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client:
        assert client.version() == "0.32.6"
        assert client.model_names() == {"embeddinggemma", "qwen3.5:4b"}
        embedding = client.embedding_probe("embeddinggemma")
        generation = client.generation_probe("qwen3.5:4b")
        runtime = client.runtime_probe()

    assert embedding.dimension == 3
    assert embedding.total_duration_ms == 2.5
    assert embedding.load_duration_ms == 1.0
    assert generation.total_duration_ms == 4.0
    assert runtime.reported_vram_bytes == 1536


@pytest.mark.parametrize("embeddings", [[], [[]], [[0.1], [0.2]]])
def test_embedding_probe_rejects_invalid_vector_shape(embeddings: list[list[float]]) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "embeddinggemma", "embeddings": embeddings},
        )

    with (
        OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(OllamaResponseError),
    ):
        client.embedding_probe("embeddinggemma")


def test_generation_probe_rejects_invalid_structured_output() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3.5:4b",
                "message": {"role": "assistant", "content": '{"status":"wrong"}'},
            },
        )

    with (
        OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(OllamaResponseError, match="invalid structured output"),
    ):
        client.generation_probe("qwen3.5:4b")


def test_http_error_does_not_expose_response_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="sensitive server details")

    with (
        OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(OllamaResponseError) as captured,
    ):
        client.version()

    assert "500" in str(captured.value)
    assert "sensitive server details" not in str(captured.value)


def test_invalid_response_schema_becomes_safe_application_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    with (
        OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(OllamaResponseError, match="invalid response"),
    ):
        client.version()


def test_connection_error_becomes_safe_application_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("low-level connection details", request=request)

    with (
        OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(OllamaConnectionError, match="Could not connect to local Ollama"),
    ):
        client.version()


def test_embedding_probe_rejects_non_finite_values() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"model":"embeddinggemma","embeddings":[[NaN]]}',
            headers={"content-type": "application/json"},
        )

    with (
        OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(OllamaResponseError, match="non-finite"),
    ):
        client.embedding_probe("embeddinggemma")


def test_timeout_becomes_safe_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("low-level timeout details", request=request)

    with (
        OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(OllamaConnectionError, match="Timed out while contacting local Ollama"),
    ):
        client.version()
