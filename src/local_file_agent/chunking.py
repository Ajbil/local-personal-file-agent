"""Deterministically divide trusted documents into source-mapped chunks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import BaseModel, Field

from local_file_agent.ingestion import Document

DEFAULT_CHUNK_SIZE = 1_200
DEFAULT_OVERLAP = 200


class ChunkingError(ValueError):
    """Raised when chunking configuration cannot guarantee valid progress."""


@dataclass(frozen=True, slots=True)
class ChunkingOptions:
    """Character-based chunking policy with safe defaults."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ChunkingError("Chunk size must be greater than zero.")
        if self.overlap < 0:
            raise ChunkingError("Overlap must not be negative.")
        if self.overlap >= self.chunk_size:
            raise ChunkingError("Overlap must be smaller than chunk size.")

    @property
    def boundary_search_window(self) -> int:
        """Search for natural boundaries in the final quarter of a target chunk."""

        return max(1, self.chunk_size // 4)


@dataclass(frozen=True, slots=True)
class Chunk:
    """An exact, source-mapped substring of one trusted document."""

    document_id: str
    relative_path: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str
    content_sha256: str

    @property
    def character_count(self) -> int:
        return self.end_char - self.start_char


class ChunkMetadata(BaseModel):
    """Inspection-safe chunk metadata; text is present only after explicit opt-in."""

    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    character_count: int = Field(gt=0)
    overlap_with_previous: int = Field(ge=0)
    content_sha256: str
    text: str | None = None


class ChunkInspectionReport(BaseModel):
    """Serializable output for one document's chunking inspection."""

    status: str = "completed"
    document_id: str
    relative_path: str
    document_characters: int = Field(ge=0)
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    boundary_search_window: int = Field(gt=0)
    chunk_count: int = Field(ge=0)
    content_included: bool
    chunks: list[ChunkMetadata]


def _rightmost_sentence_boundary(text: str, lower: int, upper: int) -> int | None:
    for position in range(upper - 1, lower - 1, -1):
        if text[position] not in ".!?":
            continue
        next_position = position + 1
        if next_position == len(text) or text[next_position].isspace():
            return next_position
    return None


def _rightmost_whitespace_boundary(text: str, lower: int, upper: int) -> int | None:
    for position in range(upper - 1, lower - 1, -1):
        if text[position].isspace():
            return position + 1
    return None


def _preferred_end(text: str, start: int, desired_end: int, options: ChunkingOptions) -> int:
    lower = max(
        start + options.overlap + 1,
        desired_end - options.boundary_search_window,
    )

    paragraph = text.rfind("\n\n", lower, desired_end)
    if paragraph != -1:
        return paragraph + 2

    newline = text.rfind("\n", lower, desired_end)
    if newline != -1:
        return newline + 1

    sentence = _rightmost_sentence_boundary(text, lower, desired_end)
    if sentence is not None:
        return sentence

    whitespace = _rightmost_whitespace_boundary(text, lower, desired_end)
    if whitespace is not None:
        return whitespace

    return desired_end


def chunk_document(
    document: Document,
    options: ChunkingOptions | None = None,
) -> tuple[Chunk, ...]:
    """Return deterministic chunks whose offsets map exactly to normalized source text."""

    policy = options or ChunkingOptions()
    if not document.text:
        return ()

    chunks: list[Chunk] = []
    start = 0
    text_length = len(document.text)

    while start < text_length:
        desired_end = min(start + policy.chunk_size, text_length)
        end = (
            text_length
            if desired_end == text_length
            else _preferred_end(document.text, start, desired_end, policy)
        )
        if end <= start:
            raise RuntimeError("Chunker failed to make forward progress.")

        chunk_text = document.text[start:end]
        chunks.append(
            Chunk(
                document_id=document.document_id,
                relative_path=document.relative_path,
                chunk_index=len(chunks),
                start_char=start,
                end_char=end,
                text=chunk_text,
                content_sha256=hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
            )
        )

        if end == text_length:
            break
        next_start = end - policy.overlap
        if next_start <= start:
            raise RuntimeError("Chunker failed to advance after applying overlap.")
        start = next_start

    return tuple(chunks)


def build_inspection_report(
    document: Document,
    chunks: tuple[Chunk, ...],
    options: ChunkingOptions,
    *,
    include_text: bool,
) -> ChunkInspectionReport:
    """Build a privacy-aware view without changing the core chunk representation."""

    metadata: list[ChunkMetadata] = []
    previous_end = 0
    for chunk in chunks:
        overlap = max(0, previous_end - chunk.start_char) if metadata else 0
        metadata.append(
            ChunkMetadata(
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                character_count=chunk.character_count,
                overlap_with_previous=overlap,
                content_sha256=chunk.content_sha256,
                text=chunk.text if include_text else None,
            )
        )
        previous_end = chunk.end_char

    return ChunkInspectionReport(
        document_id=document.document_id,
        relative_path=document.relative_path,
        document_characters=document.character_count,
        chunk_size=options.chunk_size,
        overlap=options.overlap,
        boundary_search_window=options.boundary_search_window,
        chunk_count=len(chunks),
        content_included=include_text,
        chunks=metadata,
    )
