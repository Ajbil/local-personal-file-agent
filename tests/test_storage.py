"""Tests for the versioned SQLite schema and portable vector codec."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from local_file_agent.chunking import Chunk
from local_file_agent.embeddings import PROMPT_STRATEGY, EmbeddedChunk
from local_file_agent.ingestion import Document, document_id_for
from local_file_agent.storage import (
    APPLICATION_ID,
    SCHEMA_VERSION,
    VECTOR_FORMAT,
    IndexMetadata,
    IndexStorageError,
    corpus_fingerprint,
    create_index_database,
    deserialize_vector,
    load_index,
    serialize_vector,
)


def make_document(text: str = "A synthetic stored passage.") -> Document:
    return Document(
        document_id=document_id_for("note.md"),
        relative_path="note.md",
        text=text,
        size_bytes=len(text.encode()),
        character_count=len(text),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def make_records() -> tuple[Document, EmbeddedChunk]:
    document = make_document()
    chunk = Chunk(
        document_id=document.document_id,
        relative_path=document.relative_path,
        chunk_index=0,
        start_char=0,
        end_char=len(document.text),
        text=document.text,
        content_sha256=document.content_sha256,
    )
    return document, EmbeddedChunk(chunk, np.asarray([0.25, -0.5], dtype=np.float32))


def create_valid_index(path: Path) -> IndexMetadata:
    document, embedded = make_records()
    fingerprint = corpus_fingerprint(
        [document],
        [embedded.chunk],
        embedding_model="embeddinggemma",
        prompt_strategy=PROMPT_STRATEGY,
        embedding_dimension=2,
        chunk_size=1200,
        overlap=200,
    )
    metadata = IndexMetadata(
        schema_version=SCHEMA_VERSION,
        requested_embedding_model="embeddinggemma",
        embedding_model="embeddinggemma",
        prompt_strategy=PROMPT_STRATEGY,
        embedding_dimension=2,
        vector_format=VECTOR_FORMAT,
        chunk_size=1200,
        overlap=200,
        corpus_fingerprint=fingerprint,
        document_count=1,
        chunk_count=1,
        embedding_count=1,
        built_at_utc="2026-08-09T00:00:00Z",
    )
    create_index_database(path, metadata, [document], [embedded])
    return metadata


def test_vector_codec_is_little_endian_float32_and_read_only() -> None:
    original = np.asarray([1.0, -2.5, 0.125], dtype=np.float32)

    blob = serialize_vector(original)
    restored = deserialize_vector(blob, 3)

    assert blob == np.asarray(original, dtype=np.dtype("<f4")).tobytes()
    assert restored.dtype == np.float32
    assert restored.tolist() == pytest.approx(original.tolist())
    assert not restored.flags.writeable


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "non-empty"),
        ([0.0, 0.0], "non-zero"),
        ([float("nan")], "non-finite"),
        ([float("inf")], "non-finite"),
    ],
)
def test_vector_serialization_rejects_invalid_values(values: list[float], message: str) -> None:
    with pytest.raises(IndexStorageError, match=message):
        serialize_vector(np.asarray(values, dtype=np.float32))


def test_vector_deserialization_rejects_wrong_size_and_values() -> None:
    with pytest.raises(IndexStorageError, match="byte length"):
        deserialize_vector(b"\x00\x00\x80", 1)
    with pytest.raises(IndexStorageError, match="non-finite"):
        deserialize_vector(np.asarray([float("nan")], dtype="<f4").tobytes(), 1)
    with pytest.raises(IndexStorageError, match="non-zero"):
        deserialize_vector(np.asarray([0.0], dtype="<f4").tobytes(), 1)


def test_create_close_and_read_only_reopen_preserves_records(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite"
    expected = create_valid_index(database)

    stored = load_index(database)

    assert stored.metadata == expected
    assert len(stored.documents) == 1
    assert len(stored.embedded_chunks) == 1
    assert stored.embedded_chunks[0].chunk.text == "A synthetic stored passage."
    assert stored.embedded_chunks[0].vector.tolist() == pytest.approx([0.25, -0.5])
    with (
        sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as connection,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        connection.execute("DELETE FROM documents")


def test_schema_enables_ownership_version_strict_tables_and_foreign_keys(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.sqlite"
    create_valid_index(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        strict_flags = {
            row[1]: row[5]
            for row in connection.execute("PRAGMA table_list")
            if not str(row[1]).startswith("sqlite_")
        }
        assert strict_flags == {
            "chunks": 1,
            "documents": 1,
            "embeddings": 1,
            "index_metadata": 1,
        }
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO embeddings VALUES (?, ?, ?, ?)",
                ("0" * 64, 99, 2, b"12345678"),
            )


def test_fingerprint_is_stable_and_changes_with_provenance() -> None:
    document, embedded = make_records()
    first = corpus_fingerprint(
        [document],
        [embedded.chunk],
        embedding_model="embeddinggemma",
        prompt_strategy=PROMPT_STRATEGY,
        embedding_dimension=2,
        chunk_size=1200,
        overlap=200,
    )
    second = corpus_fingerprint(
        [document],
        [embedded.chunk],
        embedding_model="embeddinggemma",
        prompt_strategy=PROMPT_STRATEGY,
        embedding_dimension=2,
        chunk_size=1200,
        overlap=200,
    )
    changed = corpus_fingerprint(
        [document],
        [embedded.chunk],
        embedding_model="embeddinggemma",
        prompt_strategy=PROMPT_STRATEGY,
        embedding_dimension=2,
        chunk_size=1200,
        overlap=0,
    )

    assert first == second
    assert first != changed


def test_wrong_application_or_schema_version_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite"
    create_valid_index(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA application_id = 0")

    with pytest.raises(IndexStorageError, match="not a Local Personal"):
        load_index(database)


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "UPDATE embeddings SET vector = x'00000000'",
            "Stored vector failed validation",
        ),
        (
            "UPDATE embeddings SET dimension = 3",
            "dimension is incompatible",
        ),
        (
            "UPDATE chunks SET text = 'tampered text', end_char = 13",
            "content hash is invalid",
        ),
        (
            "UPDATE index_metadata SET chunk_count = 2",
            "row counts",
        ),
        (
            "UPDATE index_metadata SET corpus_fingerprint = '" + ("0" * 64) + "'",
            "fingerprint",
        ),
        (
            "UPDATE index_metadata SET embedding_model = 'different-model'",
            "model metadata is incompatible",
        ),
        (
            "UPDATE index_metadata SET prompt_strategy = 'unknown-strategy'",
            "prompt strategy is not supported",
        ),
        (
            "UPDATE index_metadata SET built_at_utc = '2026-08-09T00:00:00'",
            "must use UTC",
        ),
    ],
)
def test_corrupt_records_are_rejected(tmp_path: Path, statement: str, message: str) -> None:
    database = tmp_path / "index.sqlite"
    create_valid_index(database)
    with sqlite3.connect(database) as connection:
        connection.execute(statement)

    with pytest.raises(IndexStorageError, match=message):
        load_index(database)


def test_missing_or_extra_table_layout_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite"
    create_valid_index(database)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unexpected (value TEXT) STRICT")

    with pytest.raises(IndexStorageError, match="table layout"):
        load_index(database)
