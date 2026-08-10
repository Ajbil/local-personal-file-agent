"""Read-only brute-force vector retrieval with deterministic ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field

from local_file_agent.chunking import Chunk
from local_file_agent.embeddings import (
    EmbeddingGateway,
    EmbeddingService,
    QueryEmbedding,
    cosine_similarity,
)
from local_file_agent.ollama import model_names_equivalent
from local_file_agent.storage import StoredIndex, load_index

DEFAULT_MIN_SCORE = 0.30
DEFAULT_TOP_K = 5
MAX_TOP_K = 100
MAX_OVERLAP_RATIO = 0.80


class RetrievalError(RuntimeError):
    """Stored and query embeddings cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class SearchOptions:
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    max_overlap_ratio: float = MAX_OVERLAP_RATIO

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= MAX_TOP_K:
            raise ValueError(f"Top-k must be between 1 and {MAX_TOP_K}.")
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError("Minimum score must be between -1.0 and 1.0.")
        if not 0.0 <= self.max_overlap_ratio <= 1.0:
            raise ValueError("Maximum overlap ratio must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    similarity: float


@dataclass(frozen=True, slots=True)
class SearchRun:
    query_embedding: QueryEmbedding
    options: SearchOptions
    indexed_chunk_count: int
    above_threshold_count: int
    suppressed_count: int
    results: tuple[SearchResult, ...]
    retrieval_wall_duration_ms: float


@dataclass(frozen=True, slots=True)
class DatabaseSearchRun:
    """Validated index and retrieval result shared by search and answer flows."""

    index: StoredIndex
    question: str
    search: SearchRun
    index_load_duration_ms: float


class SearchResultReport(BaseModel):
    rank: int = Field(gt=0)
    citation: str
    relative_path: str
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    similarity: float = Field(ge=-1, le=1)
    content_sha256: str
    text: str | None = None


class SearchReport(BaseModel):
    status: str = "completed"
    query_characters: int = Field(gt=0)
    stored_embedding_model: str
    returned_embedding_model: str
    prompt_strategy: str
    embedding_dimension: int = Field(gt=0)
    corpus_fingerprint: str
    top_k: int = Field(gt=0)
    min_score: float = Field(ge=-1, le=1)
    max_overlap_ratio: float = Field(ge=0, le=1)
    indexed_chunk_count: int = Field(gt=0)
    above_threshold_count: int = Field(ge=0)
    suppressed_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    index_load_duration_ms: float = Field(ge=0)
    query_embedding_wall_duration_ms: float = Field(ge=0)
    query_embedding_total_duration_ms: float | None = Field(default=None, ge=0)
    retrieval_wall_duration_ms: float = Field(ge=0)
    content_included: bool
    results: list[SearchResultReport]


def chunk_overlap_ratio(first: Chunk, second: Chunk) -> float:
    """Return intersection divided by the shorter chunk for one source document."""

    if first.document_id != second.document_id:
        return 0.0
    intersection = max(
        0,
        min(first.end_char, second.end_char) - max(first.start_char, second.start_char),
    )
    shorter = min(first.character_count, second.character_count)
    if shorter <= 0:
        raise RetrievalError("Overlap comparison requires non-empty chunks.")
    return intersection / shorter


def retrieve(
    index: StoredIndex,
    query_embedding: QueryEmbedding,
    options: SearchOptions | None = None,
) -> SearchRun:
    """Score all validated vectors, filter, de-duplicate, and select top-K."""

    policy = options or SearchOptions()
    metadata = index.metadata
    if not model_names_equivalent(metadata.embedding_model, query_embedding.returned_model):
        raise RetrievalError("Query embedding model does not match the stored index model.")
    if query_embedding.prompt_strategy != metadata.prompt_strategy:
        raise RetrievalError("Query prompt strategy does not match the stored index.")
    if query_embedding.dimension != metadata.embedding_dimension:
        raise RetrievalError("Query embedding dimension does not match the stored index.")

    started = perf_counter()
    scored = [
        SearchResult(
            chunk=item.chunk,
            similarity=cosine_similarity(query_embedding.vector, item.vector),
        )
        for item in index.embedded_chunks
    ]
    above_threshold = [result for result in scored if result.similarity >= policy.min_score]
    ranked = sorted(
        above_threshold,
        key=lambda result: (
            -result.similarity,
            result.chunk.relative_path.casefold(),
            result.chunk.relative_path,
            result.chunk.chunk_index,
        ),
    )

    retained: list[SearchResult] = []
    suppressed_count = 0
    for candidate in ranked:
        if any(
            chunk_overlap_ratio(candidate.chunk, selected.chunk) >= policy.max_overlap_ratio
            for selected in retained
        ):
            suppressed_count += 1
            continue
        retained.append(candidate)

    return SearchRun(
        query_embedding=query_embedding,
        options=policy,
        indexed_chunk_count=len(index.embedded_chunks),
        above_threshold_count=len(above_threshold),
        suppressed_count=suppressed_count,
        results=tuple(retained[: policy.top_k]),
        retrieval_wall_duration_ms=round((perf_counter() - started) * 1_000, 3),
    )


def citation_for_chunk(chunk: Chunk) -> str:
    """Build a trusted, stable citation label from application-owned metadata."""

    return f"{chunk.relative_path}#chunk-{chunk.chunk_index}[{chunk.start_char}:{chunk.end_char})"


def build_search_report(
    index: StoredIndex,
    query: str,
    run: SearchRun,
    *,
    index_load_duration_ms: float,
    include_text: bool,
) -> SearchReport:
    """Create a serializable view without exposing query or passage text by default."""

    results = [
        SearchResultReport(
            rank=rank,
            citation=citation_for_chunk(result.chunk),
            relative_path=result.chunk.relative_path,
            chunk_index=result.chunk.chunk_index,
            start_char=result.chunk.start_char,
            end_char=result.chunk.end_char,
            similarity=round(result.similarity, 6),
            content_sha256=result.chunk.content_sha256,
            text=result.chunk.text if include_text else None,
        )
        for rank, result in enumerate(run.results, start=1)
    ]
    query_embedding = run.query_embedding
    return SearchReport(
        query_characters=len(query.strip()),
        stored_embedding_model=index.metadata.embedding_model,
        returned_embedding_model=query_embedding.returned_model,
        prompt_strategy=query_embedding.prompt_strategy,
        embedding_dimension=query_embedding.dimension,
        corpus_fingerprint=index.metadata.corpus_fingerprint,
        top_k=run.options.top_k,
        min_score=run.options.min_score,
        max_overlap_ratio=run.options.max_overlap_ratio,
        indexed_chunk_count=run.indexed_chunk_count,
        above_threshold_count=run.above_threshold_count,
        suppressed_count=run.suppressed_count,
        result_count=len(results),
        index_load_duration_ms=index_load_duration_ms,
        query_embedding_wall_duration_ms=query_embedding.wall_duration_ms,
        query_embedding_total_duration_ms=query_embedding.total_duration_ms,
        retrieval_wall_duration_ms=run.retrieval_wall_duration_ms,
        content_included=include_text,
        results=results,
    )


def search_database(
    database: Path,
    query: str,
    gateway: EmbeddingGateway,
    *,
    options: SearchOptions | None = None,
    include_text: bool = False,
) -> SearchReport:
    """Validate an index first, then embed and retrieve using its exact model contract."""

    database_run = run_database_search(database, query, gateway, options=options)
    return build_search_report(
        database_run.index,
        database_run.question,
        database_run.search,
        index_load_duration_ms=database_run.index_load_duration_ms,
        include_text=include_text,
    )


def run_database_search(
    database: Path,
    query: str,
    gateway: EmbeddingGateway,
    *,
    options: SearchOptions | None = None,
) -> DatabaseSearchRun:
    """Return trusted internal retrieval state without converting it to display data."""

    normalized = query.strip()
    if not normalized:
        raise ValueError("Search question must not be empty.")
    started = perf_counter()
    index = load_index(database)
    index_load_duration_ms = round((perf_counter() - started) * 1_000, 3)
    query_embedding = EmbeddingService(
        gateway,
        index.metadata.embedding_model,
    ).embed_query(
        normalized,
        expected_dimension=index.metadata.embedding_dimension,
    )
    run = retrieve(index, query_embedding, options)
    return DatabaseSearchRun(
        index=index,
        question=normalized,
        search=run,
        index_load_duration_ms=index_load_duration_ms,
    )
