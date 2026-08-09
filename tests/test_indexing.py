"""Tests for safe complete index orchestration and atomic replacement."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

import local_file_agent.indexing as indexing
from local_file_agent.chunking import ChunkingOptions
from local_file_agent.embeddings import PROMPT_STRATEGY
from local_file_agent.indexing import (
    IndexBuildError,
    IndexBuildReport,
    build_index,
    inspect_index,
)
from local_file_agent.ollama import EmbeddingBatch
from local_file_agent.storage import IndexStorageError, load_index


class RecordingGateway:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def embed(
        self,
        _model: str,
        inputs: Sequence[str],
        *,
        truncate: bool = False,
    ) -> EmbeddingBatch:
        assert truncate is False
        copied = list(inputs)
        self.calls.append(copied)
        if self.fail:
            raise IndexStorageError("synthetic embedding failure")
        return EmbeddingBatch(
            model="embeddinggemma",
            vectors=[[1.0, float(index + 1)] for index in range(len(copied))],
        )


def write_source(root: Path, text: str = "Synthetic source text. " * 20) -> None:
    root.mkdir()
    (root / "note.md").write_text(text, encoding="utf-8")


def build(
    database: Path,
    source: Path,
    gateway: RecordingGateway,
    *,
    force: bool = False,
) -> IndexBuildReport:
    return build_index(
        source,
        database,
        gateway,
        "embeddinggemma",
        chunking=ChunkingOptions(chunk_size=100, overlap=20),
        batch_size=2,
        force=force,
    )


def test_build_persists_every_document_chunk_and_vector(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source(source)
    database = tmp_path / "index.sqlite"
    gateway = RecordingGateway()

    report = build(database, source, gateway)
    reopened = load_index(database)
    inspection = inspect_index(database)

    assert report.accepted_documents == 1
    assert report.chunk_count == report.embedding_count > 1
    assert len(reopened.embedded_chunks) == report.chunk_count
    assert inspection.opened_read_only is True
    assert inspection.corpus_fingerprint == report.corpus_fingerprint
    assert sum(len(call) for call in gateway.calls) == report.chunk_count
    assert all(item.chunk.text for item in reopened.embedded_chunks)


def test_existing_index_requires_force_before_embedding(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source(source)
    database = tmp_path / "index.sqlite"
    build(database, source, RecordingGateway())
    gateway = RecordingGateway()

    with pytest.raises(IndexBuildError, match="--force"):
        build(database, source, gateway)

    assert gateway.calls == []


def test_force_replaces_valid_index_and_preserves_stable_fingerprint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source(source)
    database = tmp_path / "index.sqlite"
    first = build(database, source, RecordingGateway())

    second = build(database, source, RecordingGateway(), force=True)

    assert second.replaced_existing is True
    assert second.corpus_fingerprint == first.corpus_fingerprint


def test_force_never_overwrites_unrelated_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source(source)
    database = tmp_path / "important.db"
    original = b"not an application index"
    database.write_bytes(original)

    with pytest.raises(IndexBuildError, match="will not be overwritten"):
        build(database, source, RecordingGateway(), force=True)

    assert database.read_bytes() == original


def test_database_inside_source_is_rejected_without_embedding(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source(source)
    gateway = RecordingGateway()

    with pytest.raises(IndexBuildError, match="outside"):
        build(source / "index.sqlite", source, gateway)

    assert gateway.calls == []


@pytest.mark.parametrize("suffix", [".txt", "", ".json"])
def test_database_extension_is_restricted(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / "source"
    write_source(source)
    with pytest.raises(IndexBuildError, match="extension"):
        build(tmp_path / f"index{suffix}", source, RecordingGateway())


def test_empty_corpus_fails_before_embedding(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "empty.md").write_text("", encoding="utf-8")
    gateway = RecordingGateway()

    with pytest.raises(IndexBuildError, match="indexable chunks"):
        build(tmp_path / "index.sqlite", source, gateway)

    assert gateway.calls == []


def test_failed_rebuild_preserves_old_index(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source(source)
    database = tmp_path / "index.sqlite"
    build(database, source, RecordingGateway())
    old_hash = hashlib.sha256(database.read_bytes()).hexdigest()

    with pytest.raises(IndexStorageError, match="synthetic embedding"):
        build(database, source, RecordingGateway(fail=True), force=True)

    assert hashlib.sha256(database.read_bytes()).hexdigest() == old_hash
    assert load_index(database).metadata.prompt_strategy == PROMPT_STRATEGY


def test_persistence_failure_cleans_temp_and_preserves_old_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    write_source(source)
    database = tmp_path / "index.sqlite"
    build(database, source, RecordingGateway())
    old_hash = hashlib.sha256(database.read_bytes()).hexdigest()

    def fail_create(*_args: object, **_kwargs: object) -> None:
        raise IndexStorageError("synthetic write failure")

    monkeypatch.setattr(indexing, "create_index_database", fail_create)
    with pytest.raises(IndexBuildError, match="safely publish"):
        build(database, source, RecordingGateway(), force=True)

    assert hashlib.sha256(database.read_bytes()).hexdigest() == old_hash
    assert list(tmp_path.glob(".index.sqlite.*.tmp")) == []


def test_source_files_are_not_modified(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source(source)
    source_file = source / "note.md"
    before = (source_file.read_bytes(), source_file.stat().st_mtime_ns)

    build(tmp_path / "index.sqlite", source, RecordingGateway())

    assert (source_file.read_bytes(), source_file.stat().st_mtime_ns) == before
