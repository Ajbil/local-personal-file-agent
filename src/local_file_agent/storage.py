"""Versioned SQLite persistence and validation for local RAG indexes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from numpy.typing import NDArray

from local_file_agent.chunking import Chunk
from local_file_agent.embeddings import PROMPT_STRATEGY, EmbeddedChunk, EmbeddingError
from local_file_agent.ingestion import Document
from local_file_agent.ollama import model_names_equivalent

SCHEMA_VERSION = 1
APPLICATION_ID = 0x4C464147  # ASCII "LFAG"
VECTOR_FORMAT = "float32-le"
DATABASE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})


class IndexStorageError(RuntimeError):
    """A database is unsafe, corrupt, incompatible, or cannot be persisted."""


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    schema_version: int
    requested_embedding_model: str
    embedding_model: str
    prompt_strategy: str
    embedding_dimension: int
    vector_format: str
    chunk_size: int
    overlap: int
    corpus_fingerprint: str
    document_count: int
    chunk_count: int
    embedding_count: int
    built_at_utc: str


@dataclass(frozen=True, slots=True)
class StoredDocument:
    document_id: str
    relative_path: str
    size_bytes: int
    character_count: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class StoredIndex:
    metadata: IndexMetadata
    documents: tuple[StoredDocument, ...]
    embedded_chunks: tuple[EmbeddedChunk, ...]


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE index_metadata (
        singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
        schema_version INTEGER NOT NULL CHECK (schema_version > 0),
        requested_embedding_model TEXT NOT NULL CHECK (length(requested_embedding_model) > 0),
        embedding_model TEXT NOT NULL CHECK (length(embedding_model) > 0),
        prompt_strategy TEXT NOT NULL CHECK (length(prompt_strategy) > 0),
        embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
        vector_format TEXT NOT NULL,
        chunk_size INTEGER NOT NULL CHECK (chunk_size > 0),
        overlap INTEGER NOT NULL CHECK (overlap >= 0 AND overlap < chunk_size),
        corpus_fingerprint TEXT NOT NULL CHECK (length(corpus_fingerprint) = 64),
        document_count INTEGER NOT NULL CHECK (document_count > 0),
        chunk_count INTEGER NOT NULL CHECK (chunk_count > 0),
        embedding_count INTEGER NOT NULL CHECK (embedding_count > 0),
        built_at_utc TEXT NOT NULL CHECK (length(built_at_utc) > 0)
    ) STRICT
    """,
    """
    CREATE TABLE documents (
        document_id TEXT PRIMARY KEY CHECK (length(document_id) = 64),
        relative_path TEXT NOT NULL UNIQUE CHECK (length(relative_path) > 0),
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        character_count INTEGER NOT NULL CHECK (character_count >= 0),
        content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64)
    ) STRICT
    """,
    """
    CREATE TABLE chunks (
        document_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
        start_char INTEGER NOT NULL CHECK (start_char >= 0),
        end_char INTEGER NOT NULL CHECK (end_char > start_char),
        text TEXT NOT NULL CHECK (length(text) = end_char - start_char),
        content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
        PRIMARY KEY (document_id, chunk_index),
        FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
    ) STRICT
    """,
    """
    CREATE TABLE embeddings (
        document_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
        dimension INTEGER NOT NULL CHECK (dimension > 0),
        vector BLOB NOT NULL CHECK (length(vector) > 0),
        PRIMARY KEY (document_id, chunk_index),
        FOREIGN KEY (document_id, chunk_index)
            REFERENCES chunks(document_id, chunk_index) ON DELETE CASCADE
    ) STRICT
    """,
)

_EXPECTED_COLUMNS = {
    "index_metadata": (
        "singleton_id",
        "schema_version",
        "requested_embedding_model",
        "embedding_model",
        "prompt_strategy",
        "embedding_dimension",
        "vector_format",
        "chunk_size",
        "overlap",
        "corpus_fingerprint",
        "document_count",
        "chunk_count",
        "embedding_count",
        "built_at_utc",
    ),
    "documents": (
        "document_id",
        "relative_path",
        "size_bytes",
        "character_count",
        "content_sha256",
    ),
    "chunks": (
        "document_id",
        "chunk_index",
        "start_char",
        "end_char",
        "text",
        "content_sha256",
    ),
    "embeddings": ("document_id", "chunk_index", "dimension", "vector"),
}


def serialize_vector(vector: NDArray[np.float32]) -> bytes:
    """Serialize one validated vector as contiguous little-endian Float32 bytes."""

    if vector.ndim != 1 or vector.size == 0:
        raise IndexStorageError("Vector must be one-dimensional and non-empty.")
    little_endian = np.ascontiguousarray(vector, dtype=np.dtype("<f4"))
    if not np.isfinite(little_endian).all():
        raise IndexStorageError("Vector contains a non-finite Float32 value.")
    if float(np.linalg.norm(little_endian.astype(np.float64))) == 0:
        raise IndexStorageError("Vector must have a non-zero norm.")
    return little_endian.tobytes(order="C")


def deserialize_vector(blob: bytes, dimension: int) -> NDArray[np.float32]:
    """Decode and validate the portable vector representation."""

    if dimension < 1:
        raise IndexStorageError("Stored vector dimension must be positive.")
    if len(blob) != dimension * np.dtype("<f4").itemsize:
        raise IndexStorageError("Stored vector byte length does not match its dimension.")
    vector = np.frombuffer(blob, dtype=np.dtype("<f4")).astype(np.float32, copy=True)
    if not np.isfinite(vector).all():
        raise IndexStorageError("Stored vector contains a non-finite value.")
    if float(np.linalg.norm(vector.astype(np.float64))) == 0:
        raise IndexStorageError("Stored vector must have a non-zero norm.")
    vector.setflags(write=False)
    return vector


def _fingerprint_payload(
    documents: Sequence[Document | StoredDocument],
    chunks: Sequence[Chunk],
    *,
    schema_version: int,
    embedding_model: str,
    prompt_strategy: str,
    embedding_dimension: int,
    chunk_size: int,
    overlap: int,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "embedding_model": embedding_model,
        "prompt_strategy": prompt_strategy,
        "embedding_dimension": embedding_dimension,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "documents": [
            {
                "document_id": document.document_id,
                "relative_path": document.relative_path,
                "size_bytes": document.size_bytes,
                "character_count": document.character_count,
                "content_sha256": document.content_sha256,
            }
            for document in documents
        ],
        "chunks": [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "content_sha256": chunk.content_sha256,
            }
            for chunk in chunks
        ],
    }


def corpus_fingerprint(
    documents: Sequence[Document | StoredDocument],
    chunks: Sequence[Chunk],
    *,
    embedding_model: str,
    prompt_strategy: str,
    embedding_dimension: int,
    chunk_size: int,
    overlap: int,
) -> str:
    """Hash canonical provenance while excluding timestamps and absolute paths."""

    payload = _fingerprint_payload(
        documents,
        chunks,
        schema_version=SCHEMA_VERSION,
        embedding_model=embedding_model,
        prompt_strategy=prompt_strategy,
        embedding_dimension=embedding_dimension,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_index_database(
    path: Path,
    metadata: IndexMetadata,
    documents: Sequence[Document],
    embedded_chunks: Sequence[EmbeddedChunk],
) -> None:
    """Create a complete index in one explicit SQLite transaction."""

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.execute("BEGIN IMMEDIATE")
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)

        connection.execute(
            """
            INSERT INTO index_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                metadata.schema_version,
                metadata.requested_embedding_model,
                metadata.embedding_model,
                metadata.prompt_strategy,
                metadata.embedding_dimension,
                metadata.vector_format,
                metadata.chunk_size,
                metadata.overlap,
                metadata.corpus_fingerprint,
                metadata.document_count,
                metadata.chunk_count,
                metadata.embedding_count,
                metadata.built_at_utc,
            ),
        )
        connection.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
            [
                (
                    document.document_id,
                    document.relative_path,
                    document.size_bytes,
                    document.character_count,
                    document.content_sha256,
                )
                for document in documents
            ],
        )
        connection.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    item.chunk.document_id,
                    item.chunk.chunk_index,
                    item.chunk.start_char,
                    item.chunk.end_char,
                    item.chunk.text,
                    item.chunk.content_sha256,
                )
                for item in embedded_chunks
            ],
        )
        connection.executemany(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?)",
            [
                (
                    item.chunk.document_id,
                    item.chunk.chunk_index,
                    metadata.embedding_dimension,
                    serialize_vector(item.vector),
                )
                for item in embedded_chunks
            ],
        )
        connection.commit()
    except (sqlite3.Error, IndexStorageError) as exc:
        if connection is not None:
            connection.rollback()
        raise IndexStorageError("Could not create a complete SQLite index.") from exc
    finally:
        if connection is not None:
            connection.close()


def _open_read_only(path: Path) -> sqlite3.Connection:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise IndexStorageError("Index path must be an existing regular file.")
        connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
    except (OSError, sqlite3.Error) as exc:
        raise IndexStorageError("Index does not exist or cannot be opened read-only.") from exc


def _validate_relative_path(value: str) -> None:
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if (
        windows.is_absolute()
        or posix.is_absolute()
        or posix == PurePosixPath(".")
        or ".." in posix.parts
        or posix.as_posix() != value
    ):
        raise IndexStorageError("Stored document path is not a safe relative path.")


def _metadata_from_row(row: sqlite3.Row) -> IndexMetadata:
    return IndexMetadata(
        schema_version=int(row["schema_version"]),
        requested_embedding_model=str(row["requested_embedding_model"]),
        embedding_model=str(row["embedding_model"]),
        prompt_strategy=str(row["prompt_strategy"]),
        embedding_dimension=int(row["embedding_dimension"]),
        vector_format=str(row["vector_format"]),
        chunk_size=int(row["chunk_size"]),
        overlap=int(row["overlap"]),
        corpus_fingerprint=str(row["corpus_fingerprint"]),
        document_count=int(row["document_count"]),
        chunk_count=int(row["chunk_count"]),
        embedding_count=int(row["embedding_count"]),
        built_at_utc=str(row["built_at_utc"]),
    )


def _validate_schema(connection: sqlite3.Connection) -> None:
    application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if application_id != APPLICATION_ID:
        raise IndexStorageError("Database is not a Local Personal File Agent index.")
    if user_version != SCHEMA_VERSION:
        raise IndexStorageError("Index schema version is not supported.")

    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if tables != set(_EXPECTED_COLUMNS):
        raise IndexStorageError("Index contains an unexpected table layout.")
    for table, expected in _EXPECTED_COLUMNS.items():
        columns = tuple(str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})"))
        if columns != expected:
            raise IndexStorageError("Index contains an unexpected column layout.")


def load_index(path: Path) -> StoredIndex:
    """Open an index read-only and validate every persisted record."""

    connection = _open_read_only(path)
    try:
        _validate_schema(connection)
        integrity_rows = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
        if integrity_rows != ["ok"]:
            raise IndexStorageError("SQLite integrity validation failed.")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise IndexStorageError("SQLite foreign-key validation failed.")

        metadata_rows = connection.execute("SELECT * FROM index_metadata").fetchall()
        if len(metadata_rows) != 1:
            raise IndexStorageError("Index must contain exactly one metadata row.")
        metadata = _metadata_from_row(metadata_rows[0])
        if metadata.schema_version != SCHEMA_VERSION:
            raise IndexStorageError("Index metadata schema version is not supported.")
        if metadata.vector_format != VECTOR_FORMAT:
            raise IndexStorageError("Index vector format is not supported.")
        if not metadata.embedding_model or not metadata.requested_embedding_model:
            raise IndexStorageError("Index embedding model metadata is invalid.")
        if not model_names_equivalent(metadata.requested_embedding_model, metadata.embedding_model):
            raise IndexStorageError("Index embedding model metadata is incompatible.")
        if metadata.prompt_strategy != PROMPT_STRATEGY:
            raise IndexStorageError("Index prompt strategy is not supported.")
        if metadata.embedding_dimension < 1:
            raise IndexStorageError("Index embedding dimension is invalid.")
        if metadata.chunk_size < 1 or not 0 <= metadata.overlap < metadata.chunk_size:
            raise IndexStorageError("Index chunking metadata is invalid.")
        try:
            built_at = datetime.fromisoformat(metadata.built_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise IndexStorageError("Index build timestamp is invalid.") from exc
        if built_at.tzinfo is None or built_at.utcoffset() != UTC.utcoffset(built_at):
            raise IndexStorageError("Index build timestamp must use UTC.")

        documents = tuple(
            StoredDocument(
                document_id=str(row["document_id"]),
                relative_path=str(row["relative_path"]),
                size_bytes=int(row["size_bytes"]),
                character_count=int(row["character_count"]),
                content_sha256=str(row["content_sha256"]),
            )
            for row in connection.execute(
                "SELECT * FROM documents ORDER BY relative_path COLLATE NOCASE, relative_path"
            )
        )
        document_by_id = {document.document_id: document for document in documents}
        for document in documents:
            _validate_relative_path(document.relative_path)
            if len(document.document_id) != 64 or len(document.content_sha256) != 64:
                raise IndexStorageError("Stored document hash metadata is invalid.")

        embedded: list[EmbeddedChunk] = []
        rows = connection.execute(
            """
            SELECT c.document_id, c.chunk_index, c.start_char, c.end_char,
                   c.text, c.content_sha256, e.dimension, e.vector
            FROM chunks AS c
            JOIN embeddings AS e
              ON e.document_id = c.document_id AND e.chunk_index = c.chunk_index
            JOIN documents AS d ON d.document_id = c.document_id
            ORDER BY d.relative_path COLLATE NOCASE, d.relative_path, c.chunk_index
            """
        )
        for row in rows:
            document_id = str(row["document_id"])
            stored_document = document_by_id.get(document_id)
            if stored_document is None:
                raise IndexStorageError("Stored chunk references an unknown document.")
            text = str(row["text"])
            start = int(row["start_char"])
            end = int(row["end_char"])
            content_hash = str(row["content_sha256"])
            if start < 0 or end <= start or end > stored_document.character_count:
                raise IndexStorageError("Stored chunk offsets are invalid.")
            if len(text) != end - start:
                raise IndexStorageError("Stored chunk text does not match its offsets.")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != content_hash:
                raise IndexStorageError("Stored chunk content hash is invalid.")
            dimension = int(row["dimension"])
            if dimension != metadata.embedding_dimension:
                raise IndexStorageError("Stored vector dimension is incompatible with the index.")
            blob = row["vector"]
            if not isinstance(blob, bytes):
                raise IndexStorageError("Stored vector is not a binary value.")
            try:
                vector = deserialize_vector(blob, dimension)
            except IndexStorageError as exc:
                raise IndexStorageError("Stored vector failed validation.") from exc
            embedded.append(
                EmbeddedChunk(
                    chunk=Chunk(
                        document_id=document_id,
                        relative_path=stored_document.relative_path,
                        chunk_index=int(row["chunk_index"]),
                        start_char=start,
                        end_char=end,
                        text=text,
                        content_sha256=content_hash,
                    ),
                    vector=vector,
                )
            )

        document_count = int(connection.execute("SELECT count(*) FROM documents").fetchone()[0])
        chunk_count = int(connection.execute("SELECT count(*) FROM chunks").fetchone()[0])
        embedding_count = int(connection.execute("SELECT count(*) FROM embeddings").fetchone()[0])
        if (
            document_count != metadata.document_count
            or chunk_count != metadata.chunk_count
            or embedding_count != metadata.embedding_count
            or chunk_count != embedding_count
            or len(embedded) != embedding_count
        ):
            raise IndexStorageError("Stored row counts do not match index metadata.")

        calculated_fingerprint = corpus_fingerprint(
            documents,
            [item.chunk for item in embedded],
            embedding_model=metadata.embedding_model,
            prompt_strategy=metadata.prompt_strategy,
            embedding_dimension=metadata.embedding_dimension,
            chunk_size=metadata.chunk_size,
            overlap=metadata.overlap,
        )
        if calculated_fingerprint != metadata.corpus_fingerprint:
            raise IndexStorageError("Index corpus fingerprint does not match stored records.")
        return StoredIndex(metadata=metadata, documents=documents, embedded_chunks=tuple(embedded))
    except (sqlite3.Error, EmbeddingError) as exc:
        raise IndexStorageError("Index could not be validated.") from exc
    finally:
        connection.close()
