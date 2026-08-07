"""Tests for the secure file-discovery and parsing boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import local_file_agent.ingestion as ingestion
from local_file_agent.ingestion import (
    SkipReason,
    SourceRootError,
    _parse_candidate,
    document_id_for,
    scan_source,
)


def test_valid_documents_are_normalized_hashed_and_sorted(tmp_path: Path) -> None:
    nested = tmp_path / "Nested"
    nested.mkdir()
    (tmp_path / "z.TXT").write_bytes(b"first\rsecond")
    (nested / "A.md").write_bytes(b"\xef\xbb\xbf# Title\r\n\r\nBody\n")

    outcome = scan_source(tmp_path)

    assert [document.relative_path for document in outcome.documents] == [
        "Nested/A.md",
        "z.TXT",
    ]
    first, second = outcome.documents
    assert first.text == "# Title\n\nBody\n"
    assert first.document_id == document_id_for("Nested/A.md")
    assert first.character_count == len(first.text)
    assert first.content_sha256 == hashlib.sha256(first.text.encode("utf-8")).hexdigest()
    assert second.text == "first\nsecond"
    assert outcome.report.summary.accepted_files == 2
    assert outcome.report.summary.skipped_entries == 0
    assert outcome.report.accepted[0].document_id == first.document_id


def test_same_normalized_text_has_same_hash_for_different_line_endings(tmp_path: Path) -> None:
    (tmp_path / "windows.txt").write_bytes(b"one\r\ntwo\r\n")
    (tmp_path / "unix.txt").write_bytes(b"one\ntwo\n")

    outcome = scan_source(tmp_path)

    assert outcome.documents[0].text == outcome.documents[1].text
    assert outcome.documents[0].content_sha256 == outcome.documents[1].content_sha256


def test_unsupported_binary_invalid_utf8_and_oversized_files_are_skipped(
    tmp_path: Path,
) -> None:
    (tmp_path / "archive.pdf").write_bytes(b"not inspected as text")
    (tmp_path / "binary.txt").write_bytes(b"text\x00binary")
    (tmp_path / "invalid.md").write_bytes(b"\xff\xfe")
    (tmp_path / "large.txt").write_bytes(b"123456")
    (tmp_path / "valid.txt").write_text("hello", encoding="utf-8")

    outcome = scan_source(tmp_path, max_file_size_bytes=5)

    assert [document.relative_path for document in outcome.documents] == ["valid.txt"]
    reasons = {item.relative_path: item.reason for item in outcome.report.skipped}
    assert reasons == {
        "archive.pdf": SkipReason.UNSUPPORTED_EXTENSION,
        "binary.txt": SkipReason.FILE_TOO_LARGE,
        "invalid.md": SkipReason.INVALID_UTF8,
        "large.txt": SkipReason.FILE_TOO_LARGE,
    }


def test_binary_content_gets_a_distinct_reason(tmp_path: Path) -> None:
    (tmp_path / "binary.txt").write_bytes(b"a\x00b")

    outcome = scan_source(tmp_path)

    assert outcome.report.skipped[0].reason is SkipReason.BINARY_CONTENT


def test_exact_size_limit_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "exact.txt").write_bytes(b"12345")

    outcome = scan_source(tmp_path, max_file_size_bytes=5)

    assert len(outcome.documents) == 1
    assert outcome.documents[0].size_bytes == 5


def test_hidden_and_generated_directories_are_not_traversed(tmp_path: Path) -> None:
    hidden = tmp_path / ".private"
    hidden.mkdir()
    (hidden / "secret.txt").write_text("must not be read", encoding="utf-8")
    generated = tmp_path / ".git"
    generated.mkdir()
    (generated / "config.txt").write_text("must not be read", encoding="utf-8")
    (tmp_path / ".env.txt").write_text("must not be read", encoding="utf-8")

    outcome = scan_source(tmp_path)

    assert outcome.documents == ()
    reasons = {item.relative_path: item.reason for item in outcome.report.skipped}
    assert reasons == {
        ".env.txt": SkipReason.HIDDEN_ENTRY,
        ".git": SkipReason.EXCLUDED_DIRECTORY,
        ".private": SkipReason.HIDDEN_ENTRY,
    }
    assert all("secret.txt" not in item.relative_path for item in outcome.report.skipped)


def test_report_is_metadata_only_and_uses_relative_paths(tmp_path: Path) -> None:
    secret_text = "private sentence that must not reach reports"
    (tmp_path / "note.md").write_text(secret_text, encoding="utf-8")

    outcome = scan_source(tmp_path)
    serialized = outcome.report.model_dump_json()

    assert secret_text not in serialized
    assert str(tmp_path) not in serialized
    assert outcome.report.accepted[0].relative_path == "note.md"
    assert outcome.report.source_name == tmp_path.name
    assert json.loads(serialized)["summary"]["accepted_files"] == 1


@pytest.mark.parametrize("source_kind", ["missing", "file"])
def test_invalid_source_root_is_rejected(tmp_path: Path, source_kind: str) -> None:
    source = tmp_path / source_kind
    if source_kind == "file":
        source.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SourceRootError):
        scan_source(source)


def test_non_positive_size_limit_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        scan_source(tmp_path, max_file_size_bytes=0)


def test_file_that_changes_during_read_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "changing.txt").write_text("changing", encoding="utf-8")
    monkeypatch.setattr(ingestion, "_same_file_state", lambda _first, _second: False)

    outcome = scan_source(tmp_path)

    assert outcome.documents == ()
    assert outcome.report.skipped[0].reason is SkipReason.FILE_CHANGED_DURING_SCAN


def test_candidate_outside_root_is_rejected_without_leaking_absolute_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    outside = tmp_path / "notes-secret.md"
    outside.write_text("outside", encoding="utf-8")

    result = _parse_candidate(outside, root.resolve(), max_file_size_bytes=100)

    assert result is SkipReason.OUTSIDE_SOURCE_ROOT


def test_symlink_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside content", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    link = source / "linked.txt"
    try:
        os.symlink(outside, link)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable on this Windows setup: {exc}")

    outcome = scan_source(source)

    assert outcome.documents == ()
    assert outcome.report.skipped[0].reason is SkipReason.SYMLINK_OR_REPARSE_POINT
    assert "outside content" not in outcome.report.model_dump_json()
