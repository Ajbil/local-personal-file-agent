"""Tests for the validated Ollama HTTP boundary."""

import json

import httpx
import pytest

from local_file_agent.config import Settings
from local_file_agent.ollama import (
    ChatMessage,
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
    model_names_equivalent,
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
            assert body["input"] == ["Local retrieval systems map meaning into vectors."]
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
            assert body["think"] is False
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


def test_structured_chat_preserves_roles_schema_and_metrics() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {
            "model": "qwen3.5:4b",
            "messages": [
                {"role": "system", "content": "Use evidence only."},
                {"role": "user", "content": "Question and evidence"},
            ],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0},
        }
        return httpx.Response(
            200,
            json={
                "model": "qwen3.5:4b",
                "message": {"role": "assistant", "content": '{"answer":"supported"}'},
                "total_duration": 8_000_000,
                "load_duration": 2_000_000,
                "prompt_eval_count": 40,
                "eval_count": 8,
            },
        )

    with OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client:
        result = client.chat_structured(
            "qwen3.5:4b",
            [
                ChatMessage(role="system", content="Use evidence only."),
                ChatMessage(role="user", content="Question and evidence"),
            ],
            schema,
        )

    assert result.model == "qwen3.5:4b"
    assert result.content == '{"answer":"supported"}'
    assert result.total_duration_ms == 8.0
    assert result.load_duration_ms == 2.0
    assert result.prompt_eval_count == 40
    assert result.eval_count == 8


@pytest.mark.parametrize(
    ("messages", "schema", "message"),
    [
        ([], {"type": "object"}, "At least one"),
        ([ChatMessage(role="user", content=" ")], {"type": "object"}, "must not be empty"),
        ([ChatMessage(role="user", content="question")], {}, "schema is required"),
    ],
)
def test_structured_chat_rejects_invalid_requests_before_transport(
    messages: list[ChatMessage], schema: dict[str, object], message: str
) -> None:
    with OllamaClient(Settings()) as client, pytest.raises(ValueError, match=message):
        client.chat_structured("qwen3.5:4b", messages, schema)


@pytest.mark.parametrize(
    ("returned_model", "role", "content", "message"),
    [
        ("other", "assistant", "{}", "unexpected answer model"),
        ("qwen3.5:4b", "user", "{}", "invalid answer message"),
        ("qwen3.5:4b", "assistant", " ", "invalid answer message"),
    ],
)
def test_structured_chat_rejects_untrusted_response_identity_or_message(
    returned_model: str, role: str, content: str, message: str
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": returned_model, "message": {"role": role, "content": content}},
        )

    with (
        OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(OllamaResponseError, match=message),
    ):
        client.chat_structured(
            "qwen3.5:4b",
            [ChatMessage(role="user", content="question")],
            {"type": "object"},
        )


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


def test_embed_preserves_batch_order_and_transport_metrics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {
            "model": "embeddinggemma",
            "input": ["first", "second"],
            "truncate": False,
        }
        return httpx.Response(
            200,
            json={
                "model": "embeddinggemma:latest",
                "embeddings": [[1.0, 0.0], [0.0, 1.0]],
                "total_duration": 3_000_000,
                "load_duration": 1_000_000,
                "prompt_eval_count": 2,
            },
        )

    with OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client:
        response = client.embed("embeddinggemma", ["first", "second"])

    assert response.model == "embeddinggemma:latest"
    assert response.vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert response.total_duration_ms == 3.0
    assert response.load_duration_ms == 1.0
    assert response.prompt_eval_count == 2


def test_embed_rejects_empty_input_before_transport() -> None:
    with OllamaClient(Settings()) as client, pytest.raises(ValueError, match="At least one"):
        client.embed("embeddinggemma", [])


@pytest.mark.parametrize(
    ("returned_model", "vectors", "message"),
    [
        ("different-model", [[1.0]], "unexpected embedding model"),
        ("embeddinggemma", [[1.0]], "unexpected embedding count"),
    ],
)
def test_embed_rejects_response_identity_or_count_mismatch(
    returned_model: str,
    vectors: list[list[float]],
    message: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": returned_model, "embeddings": vectors},
        )

    with (
        OllamaClient(Settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(OllamaResponseError, match=message),
    ):
        client.embed("embeddinggemma", ["first", "second"])


def test_model_name_equivalence_only_allows_implicit_latest_tag() -> None:
    assert model_names_equivalent("embeddinggemma", "embeddinggemma")
    assert model_names_equivalent("embeddinggemma", "embeddinggemma:latest")
    assert not model_names_equivalent("embeddinggemma:latest", "embeddinggemma")
    assert not model_names_equivalent("embeddinggemma", "other:latest")
