"""End-to-end orchestration for safe, atomic local index builds."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from local_file_agent.chunking import ChunkingOptions, chunk_document
from local_file_agent.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EmbeddingGateway,
    EmbeddingService,
)
from local_file_agent.ingestion import scan_source
from local_file_agent.storage import (
    DATABASE_SUFFIXES,
    SCHEMA_VERSION,
    VECTOR_FORMAT,
    IndexMetadata,
    IndexStorageError,
    StoredIndex,
    corpus_fingerprint,
    create_index_database,
    load_index,
)

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class IndexBuildError(RuntimeError):
    """An index target or build invariant is unsafe or invalid."""


class IndexBuildReport(BaseModel):
    status: str = "completed"
    schema_version: int = Field(gt=0)
    database_created: bool = True
    replaced_existing: bool
    requested_embedding_model: str
    embedding_model: str
    prompt_strategy: str
    embedding_dimension: int = Field(gt=0)
    vector_format: str
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    batch_count: int = Field(gt=0)
    accepted_documents: int = Field(gt=0)
    skipped_entries: int = Field(ge=0)
    chunk_count: int = Field(gt=0)
    embedding_count: int = Field(gt=0)
    corpus_fingerprint: str
    built_at_utc: str
    embedding_wall_duration_ms: float = Field(ge=0)
    embedding_total_duration_ms: float | None = Field(default=None, ge=0)
    content_included: bool = False


class IndexInspectionReport(BaseModel):
    status: str = "valid"
    opened_read_only: bool = True
    schema_version: int = Field(gt=0)
    requested_embedding_model: str
    embedding_model: str
    prompt_strategy: str
    embedding_dimension: int = Field(gt=0)
    vector_format: str
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    document_count: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    embedding_count: int = Field(gt=0)
    corpus_fingerprint: str
    built_at_utc: str
    content_included: bool = False
    integrity_check: str = "ok"
    foreign_keys_valid: bool = True


def _is_link_or_reparse(path: Path) -> bool:
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return path.is_symlink() or bool(attributes & _WINDOWS_REPARSE_POINT)


def prepare_database_target(source: Path, database: Path, *, force: bool) -> tuple[Path, bool]:
    """Resolve a safe output path and validate replacement authorization."""

    if database.suffix.lower() not in DATABASE_SUFFIXES:
        raise IndexBuildError("Database must use a .db, .sqlite, or .sqlite3 extension.")
    try:
        source_root = source.expanduser().resolve(strict=True)
        parent = database.expanduser().parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IndexBuildError("Source and database parent must already exist.") from exc
    if not source_root.is_dir():
        raise IndexBuildError("Source must be a directory.")
    if not parent.is_dir() or _is_link_or_reparse(parent):
        raise IndexBuildError("Database parent must be a regular directory.")

    target = parent / database.name
    unresolved_target = target.resolve(strict=False)
    if unresolved_target.is_relative_to(source_root):
        raise IndexBuildError("Database must be stored outside the approved source folder.")

    exists = target.exists()
    if target.is_symlink() or _is_link_or_reparse(target):
        raise IndexBuildError("Database target must not be a link or reparse point.")
    if exists:
        if not target.is_file():
            raise IndexBuildError("Database target must be a regular file.")
        if not force:
            raise IndexBuildError("Database already exists; use --force to replace it safely.")
        try:
            load_index(target)
        except IndexStorageError as exc:
            raise IndexBuildError(
                "Existing target is not a valid app-owned index and will not be overwritten."
            ) from exc
    return target, exists


def _inspection_report(index: StoredIndex) -> IndexInspectionReport:
    metadata = index.metadata
    return IndexInspectionReport(
        schema_version=metadata.schema_version,
        requested_embedding_model=metadata.requested_embedding_model,
        embedding_model=metadata.embedding_model,
        prompt_strategy=metadata.prompt_strategy,
        embedding_dimension=metadata.embedding_dimension,
        vector_format=metadata.vector_format,
        chunk_size=metadata.chunk_size,
        overlap=metadata.overlap,
        document_count=metadata.document_count,
        chunk_count=metadata.chunk_count,
        embedding_count=metadata.embedding_count,
        corpus_fingerprint=metadata.corpus_fingerprint,
        built_at_utc=metadata.built_at_utc,
    )


def inspect_index(database: Path) -> IndexInspectionReport:
    """Return a metadata-only report after complete read-only validation."""

    return _inspection_report(load_index(database))


def build_index(
    source: Path,
    database: Path,
    gateway: EmbeddingGateway,
    embedding_model: str,
    *,
    chunking: ChunkingOptions,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    force: bool = False,
) -> IndexBuildReport:
    """Build, validate, and atomically publish one complete local index."""

    if batch_size < 1:
        raise ValueError("Embedding batch size must be at least one.")
    target, replaced_existing = prepare_database_target(source, database, force=force)
    outcome = scan_source(source)
    chunks = tuple(
        chunk for document in outcome.documents for chunk in chunk_document(document, chunking)
    )
    if not chunks:
        raise IndexBuildError("Approved source did not produce any indexable chunks.")

    run = EmbeddingService(gateway, embedding_model).embed_documents(
        chunks,
        batch_size=batch_size,
    )
    if (
        run.returned_model is None
        or run.dimension is None
        or len(run.embedded_chunks) != len(chunks)
    ):
        raise IndexBuildError("Embedding run did not produce one compatible vector per chunk.")

    built_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    fingerprint = corpus_fingerprint(
        outcome.documents,
        chunks,
        embedding_model=run.returned_model,
        prompt_strategy=run.prompt_strategy,
        embedding_dimension=run.dimension,
        chunk_size=chunking.chunk_size,
        overlap=chunking.overlap,
    )
    metadata = IndexMetadata(
        schema_version=SCHEMA_VERSION,
        requested_embedding_model=run.requested_model,
        embedding_model=run.returned_model,
        prompt_strategy=run.prompt_strategy,
        embedding_dimension=run.dimension,
        vector_format=VECTOR_FORMAT,
        chunk_size=chunking.chunk_size,
        overlap=chunking.overlap,
        corpus_fingerprint=fingerprint,
        document_count=len(outcome.documents),
        chunk_count=len(chunks),
        embedding_count=len(run.embedded_chunks),
        built_at_utc=built_at,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        create_index_database(
            temporary,
            metadata,
            outcome.documents,
            run.embedded_chunks,
        )
        validated = load_index(temporary)
        if validated.metadata != metadata:
            raise IndexBuildError("Validated database metadata changed during persistence.")
        os.replace(temporary, target)
    except IndexBuildError:
        raise
    except (OSError, IndexStorageError) as exc:
        raise IndexBuildError("Could not safely publish the completed SQLite index.") from exc
    finally:
        if temporary.exists():
            with suppress(OSError):
                temporary.unlink()

    return IndexBuildReport(
        schema_version=metadata.schema_version,
        replaced_existing=replaced_existing,
        requested_embedding_model=metadata.requested_embedding_model,
        embedding_model=metadata.embedding_model,
        prompt_strategy=metadata.prompt_strategy,
        embedding_dimension=metadata.embedding_dimension,
        vector_format=metadata.vector_format,
        chunk_size=metadata.chunk_size,
        overlap=metadata.overlap,
        batch_size=run.batch_size,
        batch_count=len(run.batches),
        accepted_documents=len(outcome.documents),
        skipped_entries=outcome.report.summary.skipped_entries,
        chunk_count=len(chunks),
        embedding_count=len(run.embedded_chunks),
        corpus_fingerprint=fingerprint,
        built_at_utc=built_at,
        embedding_wall_duration_ms=run.wall_duration_ms,
        embedding_total_duration_ms=run.total_duration_ms,
    )
