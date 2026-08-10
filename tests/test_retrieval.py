"""Tests for deterministic read-only vector retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import local_file_agent.retrieval as retrieval
from local_file_agent.chunking import Chunk
from local_file_agent.embeddings import (
    PROMPT_STRATEGY,
    EmbeddedChunk,
    QueryEmbedding,
)
from local_file_agent.ollama import EmbeddingBatch
from local_file_agent.retrieval import (
    RetrievalError,
    SearchOptions,
    build_search_report,
    chunk_overlap_ratio,
    retrieve,
    search_database,
)
from local_file_agent.storage import (
    SCHEMA_VERSION,
    VECTOR_FORMAT,
    IndexMetadata,
    IndexStorageError,
    StoredDocument,
    StoredIndex,
)


def make_chunk(
    path: str,
    index: int,
    start: int,
    end: int,
    *,
    document_id: str | None = None,
) -> Chunk:
    identity = document_id or hashlib.sha256(path.encode()).hexdigest()
    text = chr(65 + index) * (end - start)
    return Chunk(
        document_id=identity,
        relative_path=path,
        chunk_index=index,
        start_char=start,
        end_char=end,
        text=text,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def make_query(
    vector: list[float],
    *,
    model: str = "embeddinggemma",
    prompt_strategy: str = PROMPT_STRATEGY,
) -> QueryEmbedding:
    array = np.asarray(vector, dtype=np.float32)
    return QueryEmbedding(
        requested_model="embeddinggemma",
        returned_model=model,
        prompt_strategy=prompt_strategy,
        dimension=len(vector),
        vector=array,
        wall_duration_ms=1.0,
        total_duration_ms=0.5,
        load_duration_ms=0.1,
        prompt_eval_count=1,
    )


def make_index(items: list[tuple[Chunk, list[float]]]) -> StoredIndex:
    paths: dict[str, StoredDocument] = {}
    for chunk, _vector in items:
        paths.setdefault(
            chunk.document_id,
            StoredDocument(
                document_id=chunk.document_id,
                relative_path=chunk.relative_path,
                size_bytes=chunk.end_char,
                character_count=chunk.end_char,
                content_sha256="0" * 64,
            ),
        )
    metadata = IndexMetadata(
        schema_version=SCHEMA_VERSION,
        requested_embedding_model="embeddinggemma",
        embedding_model="embeddinggemma",
        prompt_strategy=PROMPT_STRATEGY,
        embedding_dimension=2,
        vector_format=VECTOR_FORMAT,
        chunk_size=100,
        overlap=20,
        corpus_fingerprint="f" * 64,
        document_count=len(paths),
        chunk_count=len(items),
        embedding_count=len(items),
        built_at_utc="2026-08-10T00:00:00Z",
    )
    return StoredIndex(
        metadata=metadata,
        documents=tuple(paths.values()),
        embedded_chunks=tuple(
            EmbeddedChunk(chunk, np.asarray(vector, dtype=np.float32)) for chunk, vector in items
        ),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "between 1 and 100"),
        ({"top_k": 101}, "between 1 and 100"),
        ({"min_score": -1.1}, "between -1.0 and 1.0"),
        ({"min_score": 1.1}, "between -1.0 and 1.0"),
        ({"max_overlap_ratio": 1.1}, "between 0.0 and 1.0"),
    ],
)
def test_search_options_reject_invalid_policy(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SearchOptions(**kwargs)  # type: ignore[arg-type]


def test_overlap_ratio_uses_shorter_range_and_same_document_only() -> None:
    first = make_chunk("a.md", 0, 0, 100)
    same_document = make_chunk("a.md", 1, 20, 120, document_id=first.document_id)
    different_document = make_chunk("b.md", 0, 20, 120)

    assert chunk_overlap_ratio(first, same_document) == pytest.approx(0.8)
    assert chunk_overlap_ratio(first, different_document) == 0.0


def test_retrieval_filters_suppresses_then_applies_top_k_deterministically() -> None:
    first = make_chunk("a.md", 0, 0, 100)
    overlapping = make_chunk("a.md", 1, 20, 120, document_id=first.document_id)
    later = make_chunk("a.md", 2, 100, 200, document_id=first.document_id)
    other = make_chunk("b.md", 0, 0, 100)
    unrelated = make_chunk("z.md", 0, 0, 100)
    index = make_index(
        [
            (first, [1.0, 0.0]),
            (overlapping, [0.99, 0.01]),
            (later, [0.8, 0.2]),
            (other, [1.0, 0.0]),
            (unrelated, [-1.0, 0.0]),
        ]
    )

    run = retrieve(index, make_query([1.0, 0.0]), SearchOptions(top_k=3, min_score=0.0))

    assert run.indexed_chunk_count == 5
    assert run.above_threshold_count == 4
    assert run.suppressed_count == 1
    assert [(item.chunk.relative_path, item.chunk.chunk_index) for item in run.results] == [
        ("a.md", 0),
        ("b.md", 0),
        ("a.md", 2),
    ]


def test_minimum_score_is_inclusive_and_zero_results_are_valid() -> None:
    orthogonal = make_chunk("orthogonal.md", 0, 0, 50)
    index = make_index([(orthogonal, [0.0, 1.0])])

    included = retrieve(index, make_query([1.0, 0.0]), SearchOptions(min_score=0.0))
    excluded = retrieve(index, make_query([1.0, 0.0]), SearchOptions(min_score=0.1))

    assert len(included.results) == 1
    assert included.results[0].similarity == pytest.approx(0.0)
    assert excluded.results == ()
    assert excluded.above_threshold_count == 0


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (make_query([1.0, 0.0], model="other"), "model"),
        (make_query([1.0, 0.0], prompt_strategy="other"), "prompt strategy"),
        (make_query([1.0, 0.0, 0.0]), "dimension"),
    ],
)
def test_query_and_index_compatibility_is_mandatory(query: QueryEmbedding, message: str) -> None:
    index = make_index([(make_chunk("a.md", 0, 0, 10), [1.0, 0.0])])

    with pytest.raises(RetrievalError, match=message):
        retrieve(index, query)


def test_report_hides_query_and_text_unless_explicitly_requested() -> None:
    secret_query = "PRIVATE QUERY"
    secret_text = "P" * 20
    chunk = make_chunk("note.md", 0, 0, 20)
    chunk = Chunk(
        document_id=chunk.document_id,
        relative_path=chunk.relative_path,
        chunk_index=chunk.chunk_index,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        text=secret_text,
        content_sha256=hashlib.sha256(secret_text.encode()).hexdigest(),
    )
    index = make_index([(chunk, [1.0, 0.0])])
    run = retrieve(index, make_query([1.0, 0.0]))

    private = build_search_report(
        index, secret_query, run, index_load_duration_ms=1.0, include_text=False
    )
    visible = build_search_report(
        index, secret_query, run, index_load_duration_ms=1.0, include_text=True
    )

    assert secret_query not in private.model_dump_json(exclude_none=True)
    assert secret_text not in private.model_dump_json(exclude_none=True)
    assert private.results[0].text is None
    assert private.results[0].citation == "note.md#chunk-0[0:20)"
    assert visible.results[0].text == secret_text


class RecordingGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def embed(
        self,
        model: str,
        inputs: Sequence[str],
        *,
        truncate: bool = False,
    ) -> EmbeddingBatch:
        assert truncate is False
        self.calls.append((model, list(inputs)))
        return EmbeddingBatch(model="embeddinggemma", vectors=[[1.0, 0.0]])


def test_search_database_loads_index_before_using_its_recorded_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = make_index([(make_chunk("a.md", 0, 0, 10), [1.0, 0.0])])
    gateway = RecordingGateway()
    monkeypatch.setattr(retrieval, "load_index", lambda _path: index)

    report = search_database(Path("ignored.sqlite"), "question", gateway)

    assert gateway.calls[0][0] == index.metadata.embedding_model
    assert gateway.calls[0][1] == ["task: question answering | query: question"]
    assert report.result_count == 1


def test_invalid_index_fails_before_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = RecordingGateway()

    def fail_load(_path: Path) -> StoredIndex:
        raise IndexStorageError("synthetic corrupt index")

    monkeypatch.setattr(retrieval, "load_index", fail_load)

    with pytest.raises(IndexStorageError, match="corrupt"):
        search_database(Path("ignored.sqlite"), "question", gateway)

    assert gateway.calls == []
