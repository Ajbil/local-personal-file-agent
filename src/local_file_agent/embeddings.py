"""Prompt-aware local embedding, validation, batching, and similarity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from local_file_agent.chunking import Chunk
from local_file_agent.ollama import EmbeddingBatch, model_names_equivalent

DEFAULT_EMBEDDING_BATCH_SIZE = 8
DEFAULT_TOP_K = 5
PROMPT_STRATEGY = "embeddinggemma-question-answering-v1"


class EmbeddingError(RuntimeError):
    """Embedding data violated a safe application invariant."""


class EmbeddingGateway(Protocol):
    """The model operation required by the embedding service."""

    def embed(
        self,
        model: str,
        inputs: Sequence[str],
        *,
        truncate: bool = False,
    ) -> EmbeddingBatch: ...


@dataclass(frozen=True, slots=True)
class EmbeddingBatchMetrics:
    batch_index: int
    input_count: int
    wall_duration_ms: float
    total_duration_ms: float | None
    load_duration_ms: float | None
    prompt_eval_count: int | None


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: Chunk
    vector: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class DocumentEmbeddingRun:
    requested_model: str
    returned_model: str | None
    prompt_strategy: str
    dimension: int | None
    embedded_chunks: tuple[EmbeddedChunk, ...]
    batch_size: int
    batches: tuple[EmbeddingBatchMetrics, ...]
    wall_duration_ms: float
    total_duration_ms: float | None


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    requested_model: str
    returned_model: str
    prompt_strategy: str
    dimension: int
    vector: NDArray[np.float32]
    wall_duration_ms: float
    total_duration_ms: float | None
    load_duration_ms: float | None
    prompt_eval_count: int | None


class BatchMetricsReport(BaseModel):
    batch_index: int = Field(ge=0)
    input_count: int = Field(gt=0)
    wall_duration_ms: float = Field(ge=0)
    total_duration_ms: float | None = Field(default=None, ge=0)
    load_duration_ms: float | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)


class SimilarityResult(BaseModel):
    rank: int = Field(gt=0)
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    similarity: float = Field(ge=-1, le=1)
    vector_norm: float = Field(gt=0)
    content_sha256: str
    text: str | None = None


class EmbeddingInspectionReport(BaseModel):
    status: str = "completed"
    requested_model: str
    returned_model: str
    prompt_strategy: str
    dimension: int = Field(gt=0)
    document_id: str
    relative_path: str
    document_characters: int = Field(ge=0)
    chunk_count: int = Field(gt=0)
    vector_count: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    batch_count: int = Field(gt=0)
    document_wall_duration_ms: float = Field(ge=0)
    document_total_duration_ms: float | None = Field(default=None, ge=0)
    query_characters: int = Field(gt=0)
    query_wall_duration_ms: float = Field(ge=0)
    query_total_duration_ms: float | None = Field(default=None, ge=0)
    query_vector_norm: float = Field(gt=0)
    content_included: bool
    query_text: str | None = None
    batches: list[BatchMetricsReport]
    results: list[SimilarityResult]


def format_document_input(text: str) -> str:
    """Apply EmbeddingGemma's retrieval-document prompt."""

    return f"title: none | text: {text}"


def format_query_input(query: str) -> str:
    """Apply EmbeddingGemma's question-answering query prompt."""

    return f"task: question answering | query: {query}"


def _to_valid_float32_vector(
    values: Sequence[float],
    *,
    expected_dimension: int | None,
) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise EmbeddingError("Embedding vector must be one-dimensional and non-empty.")
    if expected_dimension is not None and vector.size != expected_dimension:
        raise EmbeddingError("Embedding vector dimension does not match the expected dimension.")
    if not np.isfinite(vector).all():
        raise EmbeddingError("Embedding vector contains a non-finite Float32 value.")
    if float(np.linalg.norm(vector.astype(np.float64))) == 0:
        raise EmbeddingError("Embedding vector must have a non-zero norm.")
    vector.setflags(write=False)
    return vector


def _optional_sum(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(sum(present), 3) if present else None


class EmbeddingService:
    """Apply prompt semantics and validate all vectors returned by a gateway."""

    def __init__(self, gateway: EmbeddingGateway, model: str) -> None:
        self._gateway = gateway
        self._model = model

    def _validated_vectors(
        self,
        response: EmbeddingBatch,
        *,
        input_count: int,
        expected_dimension: int | None,
    ) -> tuple[tuple[NDArray[np.float32], ...], int]:
        if not model_names_equivalent(self._model, response.model):
            raise EmbeddingError("Embedding response used an unexpected model.")
        if len(response.vectors) != input_count:
            raise EmbeddingError("Embedding response count does not match input count.")

        dimension = expected_dimension
        vectors: list[NDArray[np.float32]] = []
        for values in response.vectors:
            vector = _to_valid_float32_vector(values, expected_dimension=dimension)
            if dimension is None:
                dimension = int(vector.size)
            vectors.append(vector)
        if dimension is None:
            raise EmbeddingError("Embedding response did not contain vectors.")
        return tuple(vectors), dimension

    def embed_documents(
        self,
        chunks: Sequence[Chunk],
        *,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> DocumentEmbeddingRun:
        if batch_size < 1:
            raise ValueError("Embedding batch size must be at least one.")
        if not chunks:
            return DocumentEmbeddingRun(
                requested_model=self._model,
                returned_model=None,
                prompt_strategy=PROMPT_STRATEGY,
                dimension=None,
                embedded_chunks=(),
                batch_size=batch_size,
                batches=(),
                wall_duration_ms=0,
                total_duration_ms=None,
            )

        run_started = perf_counter()
        embedded: list[EmbeddedChunk] = []
        metrics: list[EmbeddingBatchMetrics] = []
        dimension: int | None = None
        returned_model: str | None = None

        for batch_index, offset in enumerate(range(0, len(chunks), batch_size)):
            batch_chunks = chunks[offset : offset + batch_size]
            inputs = [format_document_input(chunk.text) for chunk in batch_chunks]
            batch_started = perf_counter()
            response = self._gateway.embed(self._model, inputs, truncate=False)
            wall_duration_ms = round((perf_counter() - batch_started) * 1_000, 3)
            vectors, dimension = self._validated_vectors(
                response,
                input_count=len(batch_chunks),
                expected_dimension=dimension,
            )
            returned_model = response.model
            metrics.append(
                EmbeddingBatchMetrics(
                    batch_index=batch_index,
                    input_count=len(batch_chunks),
                    wall_duration_ms=wall_duration_ms,
                    total_duration_ms=response.total_duration_ms,
                    load_duration_ms=response.load_duration_ms,
                    prompt_eval_count=response.prompt_eval_count,
                )
            )
            embedded.extend(
                EmbeddedChunk(chunk=chunk, vector=vector)
                for chunk, vector in zip(batch_chunks, vectors, strict=True)
            )

        return DocumentEmbeddingRun(
            requested_model=self._model,
            returned_model=returned_model,
            prompt_strategy=PROMPT_STRATEGY,
            dimension=dimension,
            embedded_chunks=tuple(embedded),
            batch_size=batch_size,
            batches=tuple(metrics),
            wall_duration_ms=round((perf_counter() - run_started) * 1_000, 3),
            total_duration_ms=_optional_sum([item.total_duration_ms for item in metrics]),
        )

    def embed_query(
        self,
        query: str,
        *,
        expected_dimension: int | None = None,
    ) -> QueryEmbedding:
        normalized = query.strip()
        if not normalized:
            raise ValueError("Embedding query must not be empty.")

        started = perf_counter()
        response = self._gateway.embed(
            self._model,
            [format_query_input(normalized)],
            truncate=False,
        )
        wall_duration_ms = round((perf_counter() - started) * 1_000, 3)
        vectors, dimension = self._validated_vectors(
            response,
            input_count=1,
            expected_dimension=expected_dimension,
        )
        return QueryEmbedding(
            requested_model=self._model,
            returned_model=response.model,
            prompt_strategy=PROMPT_STRATEGY,
            dimension=dimension,
            vector=vectors[0],
            wall_duration_ms=wall_duration_ms,
            total_duration_ms=response.total_duration_ms,
            load_duration_ms=response.load_duration_ms,
            prompt_eval_count=response.prompt_eval_count,
        )


def cosine_similarity(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
) -> float:
    """Return validated cosine similarity without mutating either vector."""

    if first.ndim != 1 or second.ndim != 1 or first.size == 0 or second.size == 0:
        raise EmbeddingError("Cosine similarity requires non-empty one-dimensional vectors.")
    if first.shape != second.shape:
        raise EmbeddingError("Cosine similarity requires matching vector dimensions.")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise EmbeddingError("Cosine similarity received a non-finite vector.")

    first64 = first.astype(np.float64)
    second64 = second.astype(np.float64)
    denominator = float(np.linalg.norm(first64) * np.linalg.norm(second64))
    if denominator == 0:
        raise EmbeddingError("Cosine similarity requires non-zero vectors.")
    score = float(np.dot(first64, second64) / denominator)
    return max(-1.0, min(1.0, score))


def build_embedding_inspection_report(
    document_characters: int,
    run: DocumentEmbeddingRun,
    query: str,
    query_embedding: QueryEmbedding,
    *,
    chunk_size: int,
    overlap: int,
    top_k: int,
    include_text: bool,
) -> EmbeddingInspectionReport:
    if top_k < 1:
        raise ValueError("Top-k must be at least one.")
    if not run.embedded_chunks or run.dimension is None or run.returned_model is None:
        raise EmbeddingError("At least one embedded chunk is required for inspection.")
    if query_embedding.dimension != run.dimension:
        raise EmbeddingError("Query and document embedding dimensions do not match.")

    ranked = sorted(
        (
            (embedded, cosine_similarity(query_embedding.vector, embedded.vector))
            for embedded in run.embedded_chunks
        ),
        key=lambda item: (-item[1], item[0].chunk.chunk_index),
    )[:top_k]

    first_chunk = run.embedded_chunks[0].chunk
    results = [
        SimilarityResult(
            rank=rank,
            chunk_index=embedded.chunk.chunk_index,
            start_char=embedded.chunk.start_char,
            end_char=embedded.chunk.end_char,
            similarity=round(score, 6),
            vector_norm=round(float(np.linalg.norm(embedded.vector.astype(np.float64))), 6),
            content_sha256=embedded.chunk.content_sha256,
            text=embedded.chunk.text if include_text else None,
        )
        for rank, (embedded, score) in enumerate(ranked, start=1)
    ]
    return EmbeddingInspectionReport(
        requested_model=run.requested_model,
        returned_model=query_embedding.returned_model,
        prompt_strategy=run.prompt_strategy,
        dimension=run.dimension,
        document_id=first_chunk.document_id,
        relative_path=first_chunk.relative_path,
        document_characters=document_characters,
        chunk_count=len(run.embedded_chunks),
        vector_count=len(run.embedded_chunks),
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=run.batch_size,
        batch_count=len(run.batches),
        document_wall_duration_ms=run.wall_duration_ms,
        document_total_duration_ms=run.total_duration_ms,
        query_characters=len(query.strip()),
        query_wall_duration_ms=query_embedding.wall_duration_ms,
        query_total_duration_ms=query_embedding.total_duration_ms,
        query_vector_norm=round(
            float(np.linalg.norm(query_embedding.vector.astype(np.float64))), 6
        ),
        content_included=include_text,
        query_text=query.strip() if include_text else None,
        batches=[
            BatchMetricsReport(
                batch_index=item.batch_index,
                input_count=item.input_count,
                wall_duration_ms=item.wall_duration_ms,
                total_duration_ms=item.total_duration_ms,
                load_duration_ms=item.load_duration_ms,
                prompt_eval_count=item.prompt_eval_count,
            )
            for item in run.batches
        ],
        results=results,
    )
