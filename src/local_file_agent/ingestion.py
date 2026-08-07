"""Secure, deterministic discovery and parsing of approved text files."""

from __future__ import annotations

import hashlib
import os
import stat
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
SUPPORTED_EXTENSIONS = frozenset({".md", ".txt"})
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".data",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "venv",
    }
)

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_WINDOWS_HIDDEN = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0x2)


class SourceRootError(ValueError):
    """Raised when an explicitly approved source root cannot be scanned safely."""


class SkipReason(StrEnum):
    """Stable, content-free reasons why an entry was not accepted."""

    UNSUPPORTED_EXTENSION = "unsupported_extension"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    EXCLUDED_DIRECTORY = "excluded_directory"
    HIDDEN_ENTRY = "hidden_entry"
    SYMLINK_OR_REPARSE_POINT = "symlink_or_reparse_point"
    OUTSIDE_SOURCE_ROOT = "outside_source_root"
    FILE_TOO_LARGE = "file_too_large"
    BINARY_CONTENT = "binary_content"
    INVALID_UTF8 = "invalid_utf8"
    UNREADABLE_ENTRY = "unreadable_entry"
    FILE_CHANGED_DURING_SCAN = "file_changed_during_scan"


@dataclass(frozen=True, slots=True)
class Document:
    """Trusted normalized content passed to later RAG stages.

    Character offsets in later checkpoints refer to ``text`` after newline
    normalization, never to byte offsets in the source file.
    """

    document_id: str
    relative_path: str
    text: str
    size_bytes: int
    character_count: int
    content_sha256: str


class AcceptedFile(BaseModel):
    """Privacy-safe metadata for one accepted document."""

    document_id: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    character_count: int = Field(ge=0)
    content_sha256: str


class SkippedEntry(BaseModel):
    """Privacy-safe metadata for a rejected filesystem entry."""

    relative_path: str
    reason: SkipReason


class ScanSummary(BaseModel):
    """Aggregate scan counts suitable for terminal or JSON output."""

    inspected_entries: int = Field(ge=0)
    accepted_files: int = Field(ge=0)
    skipped_entries: int = Field(ge=0)
    skipped_by_reason: dict[str, int]


class ScanReport(BaseModel):
    """Serializable report that intentionally contains no document text."""

    status: str = "completed"
    source_name: str
    supported_extensions: list[str]
    max_file_size_bytes: int
    accepted: list[AcceptedFile]
    skipped: list[SkippedEntry]
    summary: ScanSummary


@dataclass(frozen=True, slots=True)
class ScanOutcome:
    """Internal documents plus the public, privacy-safe scan report."""

    documents: tuple[Document, ...]
    report: ScanReport


def _relative_sort_key(relative_path: str) -> tuple[str, str]:
    return relative_path.casefold(), relative_path


def _safe_relative_path(path: Path, root: Path) -> str:
    """Return a relative display path without leaking an outside absolute path."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name or "outside-source-entry"


def _file_attributes(path: Path) -> int:
    try:
        return int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return 0


def _is_link_or_reparse_point(path: Path) -> bool:
    return path.is_symlink() or bool(_file_attributes(path) & _WINDOWS_REPARSE_POINT)


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".") or bool(_file_attributes(path) & _WINDOWS_HIDDEN)


def _same_file_state(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_size,
        first.st_mtime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_size,
        second.st_mtime_ns,
    )


def document_id_for(relative_path: str) -> str:
    """Create a path-stable identity for a document within one approved source."""

    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


def _parse_candidate(
    path: Path,
    root: Path,
    *,
    max_file_size_bytes: int,
) -> Document | SkipReason:
    relative_path = _safe_relative_path(path, root)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return SkipReason.UNSUPPORTED_EXTENSION
    if _is_link_or_reparse_point(path):
        return SkipReason.SYMLINK_OR_REPARSE_POINT

    try:
        resolved_before = path.resolve(strict=True)
        if not resolved_before.is_relative_to(root):
            return SkipReason.OUTSIDE_SOURCE_ROOT
        before = path.stat(follow_symlinks=False)
        if before.st_size > max_file_size_bytes:
            return SkipReason.FILE_TOO_LARGE

        with path.open("rb") as source_file:
            opened = os.fstat(source_file.fileno())
            raw = source_file.read(max_file_size_bytes + 1)

        after = path.stat(follow_symlinks=False)
        resolved_after = path.resolve(strict=True)
    except OSError:
        return SkipReason.UNREADABLE_ENTRY

    if not resolved_after.is_relative_to(root) or _is_link_or_reparse_point(path):
        return SkipReason.OUTSIDE_SOURCE_ROOT
    if len(raw) > max_file_size_bytes:
        return SkipReason.FILE_TOO_LARGE
    if not _same_file_state(before, opened) or not _same_file_state(opened, after):
        return SkipReason.FILE_CHANGED_DURING_SCAN
    if b"\x00" in raw:
        return SkipReason.BINARY_CONTENT

    try:
        decoded = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        return SkipReason.INVALID_UTF8

    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return Document(
        document_id=document_id_for(relative_path),
        relative_path=relative_path,
        text=normalized,
        size_bytes=len(raw),
        character_count=len(normalized),
        content_sha256=content_hash,
    )


def scan_source(
    source: Path,
    *,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> ScanOutcome:
    """Discover and parse supported files under one explicitly approved root."""

    if max_file_size_bytes <= 0:
        raise ValueError("max_file_size_bytes must be positive")

    try:
        supplied_source = source.expanduser()
        root = supplied_source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SourceRootError("Source folder does not exist or cannot be resolved.") from exc

    if not root.is_dir():
        raise SourceRootError("Source must be a directory.")

    documents: list[Document] = []
    skipped: list[SkippedEntry] = []

    def record_skip(path: Path, reason: SkipReason) -> None:
        skipped.append(SkippedEntry(relative_path=_safe_relative_path(path, root), reason=reason))

    def visit(directory: Path, *, is_root: bool = False) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(
                    iterator,
                    key=lambda entry: (entry.name.casefold(), entry.name),
                )
        except OSError as exc:
            if is_root:
                raise SourceRootError("Source folder cannot be read.") from exc
            record_skip(directory, SkipReason.UNREADABLE_ENTRY)
            return

        for entry in entries:
            path = Path(entry.path)
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                record_skip(path, SkipReason.UNREADABLE_ENTRY)
                continue

            if path.name in EXCLUDED_DIRECTORY_NAMES and is_directory:
                record_skip(path, SkipReason.EXCLUDED_DIRECTORY)
                continue
            if _is_hidden(path):
                record_skip(path, SkipReason.HIDDEN_ENTRY)
                continue
            if _is_link_or_reparse_point(path):
                record_skip(path, SkipReason.SYMLINK_OR_REPARSE_POINT)
                continue

            if is_directory:
                visit(path)
                continue
            if not is_file:
                record_skip(path, SkipReason.UNSUPPORTED_FILE_TYPE)
                continue

            parsed = _parse_candidate(path, root, max_file_size_bytes=max_file_size_bytes)
            if isinstance(parsed, SkipReason):
                record_skip(path, parsed)
            else:
                documents.append(parsed)

    visit(root, is_root=True)
    documents.sort(key=lambda document: _relative_sort_key(document.relative_path))
    skipped.sort(key=lambda item: _relative_sort_key(item.relative_path))

    accepted = [
        AcceptedFile(
            document_id=document.document_id,
            relative_path=document.relative_path,
            size_bytes=document.size_bytes,
            character_count=document.character_count,
            content_sha256=document.content_sha256,
        )
        for document in documents
    ]
    skip_counts = Counter(item.reason.value for item in skipped)
    summary = ScanSummary(
        inspected_entries=len(accepted) + len(skipped),
        accepted_files=len(accepted),
        skipped_entries=len(skipped),
        skipped_by_reason=dict(sorted(skip_counts.items())),
    )
    report = ScanReport(
        source_name=root.name or "approved-source",
        supported_extensions=sorted(SUPPORTED_EXTENSIONS),
        max_file_size_bytes=max_file_size_bytes,
        accepted=accepted,
        skipped=skipped,
        summary=summary,
    )
    return ScanOutcome(documents=tuple(documents), report=report)
