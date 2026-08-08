"""Tests for prompt-aware local embedding and similarity invariants."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from local_file_agent.chunking import Chunk
from local_file_agent.embeddings import (
    PROMPT_STRATEGY,
    DocumentEmbeddingRun,
    EmbeddedChunk,
    EmbeddingBatchMetrics,
    EmbeddingError,
    EmbeddingService,
    QueryEmbedding,
    build_embedding_inspection_report,
    cosine_similarity,
    format_document_input,
    format_query_input,
)
from local_file_agent.ollama import EmbeddingBatch


def make_chunk(index: int, text: str | None = None) -> Chunk:
    content = text or f"chunk-{index}"
    start = index * 10
    return Chunk(
        document_id="document-id",
        relative_path="notes.md",
        chunk_index=index,
        start_char=start,
        end_char=start + len(content),
        text=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


class RecordingGateway:
    def __init__(self, responses: Sequence[EmbeddingBatch] | None = None) -> None:
        self.calls: list[tuple[str, list[str], bool]] = []
        self._responses = list(responses or [])

    def embed(
        self,
        model: str,
        inputs: Sequence[str],
        *,
        truncate: bool = False,
    ) -> EmbeddingBatch:
        copied = list(inputs)
        self.calls.append((model, copied, truncate))
        if self._responses:
            return self._responses.pop(0)
        return EmbeddingBatch(
            model="embeddinggemma:latest",
            vectors=[[float(index + 1), 1.0] for index in range(len(copied))],
            total_duration_ms=2.5,
            load_duration_ms=0.5,
            prompt_eval_count=len(copied),
        )


def test_embeddinggemma_document_and_query_prompts_are_intentionally_different() -> None:
    assert format_document_input("RAG notes") == "title: none | text: RAG notes"
    assert format_query_input("What is RAG?") == ("task: question answering | query: What is RAG?")


def test_document_embedding_batches_preserve_order_and_use_float32() -> None:
    chunks = tuple(make_chunk(index) for index in range(17))
    gateway = RecordingGateway()

    run = EmbeddingService(gateway, "embeddinggemma").embed_documents(chunks, batch_size=8)

    assert [len(inputs) for _, inputs, _ in gateway.calls] == [8, 8, 1]
    assert all(truncate is False for _, _, truncate in gateway.calls)
    assert gateway.calls[0][1][0] == "title: none | text: chunk-0"
    assert [item.chunk for item in run.embedded_chunks] == list(chunks)
    assert all(item.vector.dtype == np.float32 for item in run.embedded_chunks)
    assert all(not item.vector.flags.writeable for item in run.embedded_chunks)
    assert run.dimension == 2
    assert run.returned_model == "embeddinggemma:latest"
    assert run.prompt_strategy == PROMPT_STRATEGY
    assert [metric.input_count for metric in run.batches] == [8, 8, 1]
    assert run.total_duration_ms == 7.5


def test_empty_document_run_does_not_call_model() -> None:
    gateway = RecordingGateway()

    run = EmbeddingService(gateway, "embeddinggemma").embed_documents([])

    assert gateway.calls == []
    assert run.dimension is None
    assert run.embedded_chunks == ()


def test_query_is_trimmed_prompted_and_checked_against_document_dimension() -> None:
    gateway = RecordingGateway([EmbeddingBatch(model="embeddinggemma", vectors=[[3.0, 4.0]])])

    result = EmbeddingService(gateway, "embeddinggemma").embed_query(
        "  What is retrieval?  ", expected_dimension=2
    )

    assert gateway.calls[0][1] == ["task: question answering | query: What is retrieval?"]
    assert result.dimension == 2
    assert result.vector.dtype == np.float32
    assert not result.vector.flags.writeable


@pytest.mark.parametrize("query", ["", "  \n\t"])
def test_empty_query_is_rejected_without_model_call(query: str) -> None:
    gateway = RecordingGateway()
    with pytest.raises(ValueError, match="must not be empty"):
        EmbeddingService(gateway, "embeddinggemma").embed_query(query)
    assert gateway.calls == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (EmbeddingBatch(model="wrong", vectors=[[1.0, 2.0]]), "unexpected model"),
        (EmbeddingBatch(model="embeddinggemma", vectors=[]), "count"),
        (EmbeddingBatch(model="embeddinggemma", vectors=[[]]), "non-empty"),
        (EmbeddingBatch(model="embeddinggemma", vectors=[[0.0, 0.0]]), "non-zero"),
        (EmbeddingBatch(model="embeddinggemma", vectors=[[float("nan")]]), "non-finite"),
        (EmbeddingBatch(model="embeddinggemma", vectors=[[1e100]]), "non-finite"),
    ],
)
def test_invalid_model_vectors_become_domain_errors(
    response: EmbeddingBatch,
    message: str,
) -> None:
    gateway = RecordingGateway([response])
    with pytest.raises(EmbeddingError, match=message):
        EmbeddingService(gateway, "embeddinggemma").embed_query("question")


def test_dimension_drift_across_document_batches_is_rejected() -> None:
    gateway = RecordingGateway(
        [
            EmbeddingBatch(model="embeddinggemma", vectors=[[1.0, 2.0]]),
            EmbeddingBatch(model="embeddinggemma", vectors=[[1.0, 2.0, 3.0]]),
        ]
    )
    with pytest.raises(EmbeddingError, match="dimension"):
        EmbeddingService(gateway, "embeddinggemma").embed_documents(
            [make_chunk(0), make_chunk(1)], batch_size=1
        )


def test_cosine_similarity_has_expected_geometry() -> None:
    horizontal = np.asarray([1.0, 0.0], dtype=np.float32)
    vertical = np.asarray([0.0, 1.0], dtype=np.float32)
    opposite = np.asarray([-1.0, 0.0], dtype=np.float32)

    assert cosine_similarity(horizontal, horizontal) == pytest.approx(1.0)
    assert cosine_similarity(horizontal, vertical) == pytest.approx(0.0)
    assert cosine_similarity(horizontal, opposite) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ([1.0], [1.0, 2.0], "matching"),
        ([0.0], [1.0], "non-zero"),
        ([float("inf")], [1.0], "non-finite"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(
    first: list[float], second: list[float], message: str
) -> None:
    with pytest.raises(EmbeddingError, match=message):
        cosine_similarity(
            np.asarray(first, dtype=np.float32),
            np.asarray(second, dtype=np.float32),
        )


def test_inspection_report_ranks_stably_and_hides_content_by_default() -> None:
    secret_query = "PRIVATE QUERY"
    chunks = [make_chunk(0, "PRIVATE ZERO"), make_chunk(1, "PRIVATE ONE")]
    embedded = (
        EmbeddedChunk(chunks[0], np.asarray([1.0, 0.0], dtype=np.float32)),
        EmbeddedChunk(chunks[1], np.asarray([0.8, 0.2], dtype=np.float32)),
    )
    batch = EmbeddingBatchMetrics(0, 2, 1.0, 0.5, 0.1, 2)
    run = DocumentEmbeddingRun(
        "embeddinggemma",
        "embeddinggemma:latest",
        PROMPT_STRATEGY,
        2,
        embedded,
        8,
        (batch,),
        1.0,
        0.5,
    )
    query = QueryEmbedding(
        "embeddinggemma",
        "embeddinggemma:latest",
        PROMPT_STRATEGY,
        2,
        np.asarray([1.0, 0.0], dtype=np.float32),
        1.0,
        0.5,
        0.1,
        1,
    )

    private = build_embedding_inspection_report(
        20, run, secret_query, query, chunk_size=10, overlap=2, top_k=1, include_text=False
    )
    visible = build_embedding_inspection_report(
        20, run, secret_query, query, chunk_size=10, overlap=2, top_k=2, include_text=True
    )

    assert private.results[0].chunk_index == 0
    assert private.results[0].similarity == 1.0
    assert secret_query not in private.model_dump_json(exclude_none=True)
    assert "PRIVATE ZERO" not in private.model_dump_json(exclude_none=True)
    assert private.results[0].text is None
    assert visible.query_text == secret_query
    assert visible.results[0].text == "PRIVATE ZERO"
