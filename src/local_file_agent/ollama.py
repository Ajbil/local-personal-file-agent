"""A narrow, validated boundary around the local Ollama HTTP API."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, Protocol, cast

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from local_file_agent.config import Settings


class OllamaError(RuntimeError):
    """Base class for safe, user-facing Ollama failures."""


class OllamaConnectionError(OllamaError):
    """Ollama could not be reached within the configured boundary."""


class OllamaResponseError(OllamaError):
    """Ollama returned an unsuccessful or structurally invalid response."""


class EmbeddingProbe(BaseModel):
    """Safe metadata from an embedding smoke test; vector values are discarded."""

    dimension: int
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None


class GenerationProbe(BaseModel):
    """Safe metadata from a structured-generation smoke test."""

    total_duration_ms: float | None = None
    load_duration_ms: float | None = None


class RuntimeProbe(BaseModel):
    """Non-sensitive model residency information reported by Ollama."""

    loaded_models: list[str]
    reported_vram_bytes: int


class OllamaGateway(Protocol):
    """The capabilities used by diagnostics, independent of HTTPX."""

    def version(self) -> str: ...

    def model_names(self) -> set[str]: ...

    def embedding_probe(self, model: str) -> EmbeddingProbe: ...

    def generation_probe(self, model: str) -> GenerationProbe: ...

    def runtime_probe(self) -> RuntimeProbe: ...


class _VersionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: str


class _ModelSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str


class _TagsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models: list[_ModelSummary]


class _EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    embeddings: list[list[float]]
    total_duration: int | None = None
    load_duration: int | None = None


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class _ChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    message: _Message
    total_duration: int | None = None
    load_duration: int | None = None


class _SmokePayload(BaseModel):
    status: Literal["ok"]


class _RunningModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    size_vram: int = 0


class _ProcessResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models: list[_RunningModel]


def _validate_response[ResponseModel: BaseModel](
    model: type[ResponseModel], payload: object
) -> ResponseModel:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise OllamaResponseError("Ollama returned an invalid response") from exc


def _nanoseconds_to_milliseconds(value: int | None) -> float | None:
    return None if value is None else round(value / 1_000_000, 3)


class OllamaClient:
    """Synchronous Ollama client restricted by validated application settings."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        timeout = httpx.Timeout(
            timeout=settings.model_timeout_seconds,
            connect=settings.connect_timeout_seconds,
        )
        self._client = httpx.Client(
            base_url=settings.ollama_api_url,
            timeout=timeout,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def version(self) -> str:
        response = _validate_response(
            _VersionResponse,
            self._request_json("GET", "/api/version"),
        )
        return response.version

    def model_names(self) -> set[str]:
        response = _validate_response(
            _TagsResponse,
            self._request_json("GET", "/api/tags"),
        )
        return {model.name for model in response.models}

    def embedding_probe(self, model: str) -> EmbeddingProbe:
        payload = self._request_json(
            "POST",
            "/api/embed",
            json_body={
                "model": model,
                "input": "Local retrieval systems map meaning into vectors.",
                "truncate": False,
            },
        )
        response = _validate_response(_EmbeddingResponse, payload)
        if response.model != model:
            raise OllamaResponseError("Ollama returned an unexpected embedding model")
        if len(response.embeddings) != 1:
            raise OllamaResponseError("Ollama did not return exactly one embedding")

        vector = response.embeddings[0]
        if not vector:
            raise OllamaResponseError("Ollama returned an empty embedding")
        if any(not math.isfinite(value) for value in vector):
            raise OllamaResponseError("Ollama returned a non-finite embedding value")

        return EmbeddingProbe(
            dimension=len(vector),
            total_duration_ms=_nanoseconds_to_milliseconds(response.total_duration),
            load_duration_ms=_nanoseconds_to_milliseconds(response.load_duration),
        )

    def generation_probe(self, model: str) -> GenerationProbe:
        schema = _SmokePayload.model_json_schema()
        payload = self._request_json(
            "POST",
            "/api/chat",
            json_body={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": 'Return JSON with exactly one field: {"status": "ok"}.',
                    }
                ],
                "stream": False,
                "format": schema,
                "options": {"temperature": 0},
            },
        )
        response = _validate_response(_ChatResponse, payload)
        if response.model != model:
            raise OllamaResponseError("Ollama returned an unexpected answer model")
        try:
            _SmokePayload.model_validate_json(response.message.content)
        except ValidationError as exc:
            raise OllamaResponseError("Ollama returned invalid structured output") from exc

        return GenerationProbe(
            total_duration_ms=_nanoseconds_to_milliseconds(response.total_duration),
            load_duration_ms=_nanoseconds_to_milliseconds(response.load_duration),
        )

    def runtime_probe(self) -> RuntimeProbe:
        response = _validate_response(
            _ProcessResponse,
            self._request_json("GET", "/api/ps"),
        )
        return RuntimeProbe(
            loaded_models=[model.name for model in response.models],
            reported_vram_bytes=sum(model.size_vram for model in response.models),
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, object] | None = None,
    ) -> object:
        try:
            response = self._client.request(method, path, json=json_body)
            response.raise_for_status()
            return cast(object, response.json())
        except httpx.TimeoutException as exc:
            raise OllamaConnectionError("Timed out while contacting local Ollama") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError("Could not connect to local Ollama") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaResponseError(
                f"Ollama returned HTTP status {exc.response.status_code}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaResponseError("Ollama returned an invalid response") from exc
