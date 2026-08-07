"""Tests for deterministic, exact-offset document chunking."""

from __future__ import annotations

import hashlib
from itertools import pairwise

import pytest

from local_file_agent.chunking import (
    ChunkingError,
    ChunkingOptions,
    build_inspection_report,
    chunk_document,
)
from local_file_agent.ingestion import Document, document_id_for


def make_document(text: str, relative_path: str = "notes.md") -> Document:
    return Document(
        document_id=document_id_for(relative_path),
        relative_path=relative_path,
        text=text,
        size_bytes=len(text.encode("utf-8")),
        character_count=len(text),
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "greater than zero"),
        (-1, 0, "greater than zero"),
        (10, -1, "must not be negative"),
        (10, 10, "smaller than chunk size"),
        (10, 11, "smaller than chunk size"),
    ],
)
def test_invalid_options_are_rejected(chunk_size: int, overlap: int, message: str) -> None:
    with pytest.raises(ChunkingError, match=message):
        ChunkingOptions(chunk_size=chunk_size, overlap=overlap)


def test_empty_document_produces_no_chunks() -> None:
    assert chunk_document(make_document("")) == ()


@pytest.mark.parametrize("text", ["tiny", "   \n\t", "x" * 10])
def test_content_at_or_below_target_produces_one_exact_chunk(text: str) -> None:
    document = make_document(text)

    chunks = chunk_document(document, ChunkingOptions(chunk_size=10, overlap=2))

    assert len(chunks) == 1
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)
    assert chunks[0].text == text


def test_one_character_over_target_uses_exact_overlap() -> None:
    document = make_document("abcdefghijk")

    chunks = chunk_document(document, ChunkingOptions(chunk_size=10, overlap=2))

    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [(0, 10), (8, 11)]
    assert chunks[0].text[-2:] == chunks[1].text[:2]


@pytest.mark.parametrize(
    ("text", "expected_end"),
    [
        ("x" * 15 + "\n\n" + "y" * 25, 17),
        ("x" * 17 + "\n" + "y" * 25, 18),
        ("x" * 16 + ". " + "y" * 25, 17),
        ("x" * 17 + " " + "y" * 25, 18),
        ("x" * 45, 20),
    ],
    ids=["paragraph", "newline", "sentence", "whitespace", "hard-cut"],
)
def test_boundary_preference_selects_exact_expected_end(text: str, expected_end: int) -> None:
    chunks = chunk_document(make_document(text), ChunkingOptions(chunk_size=20, overlap=3))

    assert chunks[0].end_char == expected_end


def test_paragraph_is_preferred_over_a_later_sentence_boundary() -> None:
    text = "x" * 15 + "\n\n" + "y. " + "z" * 30

    chunks = chunk_document(make_document(text), ChunkingOptions(chunk_size=20, overlap=3))

    assert chunks[0].end_char == 17


def test_offsets_hashes_overlap_and_indexes_hold_for_every_chunk() -> None:
    text = ("Alpha paragraph with words.\n\n" * 20) + "Final section 🚀 समाप्त।"
    document = make_document(text, "unicode.md")
    options = ChunkingOptions(chunk_size=80, overlap=15)

    chunks = chunk_document(document, options)

    assert chunks
    assert chunks[-1].end_char == len(text)
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.document_id == document.document_id
        assert chunk.relative_path == document.relative_path
        assert 0 <= chunk.start_char < chunk.end_char <= len(text)
        assert chunk.text == text[chunk.start_char : chunk.end_char]
        assert chunk.character_count == len(chunk.text)
        assert chunk.content_sha256 == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        if index:
            previous = chunks[index - 1]
            assert chunk.start_char > previous.start_char
            assert previous.end_char - chunk.start_char == options.overlap
            assert previous.text[-options.overlap :] == chunk.text[: options.overlap]


def test_zero_overlap_has_no_gaps_or_repeated_characters() -> None:
    text = "z" * 31

    chunks = chunk_document(make_document(text), ChunkingOptions(chunk_size=10, overlap=0))

    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 31),
    ]


def test_excessive_but_valid_overlap_still_makes_forward_progress() -> None:
    text = "q" * 25

    chunks = chunk_document(make_document(text), ChunkingOptions(chunk_size=10, overlap=9))

    assert len(chunks) == 16
    assert all(current.start_char > previous.start_char for previous, current in pairwise(chunks))
    assert chunks[-1].end_char == len(text)


def test_same_input_produces_identical_chunks() -> None:
    document = make_document("Sentence one. Sentence two. Sentence three. " * 10)
    options = ChunkingOptions(chunk_size=60, overlap=10)

    assert chunk_document(document, options) == chunk_document(document, options)


def test_document_identity_is_path_stable_and_separate_from_content_hash() -> None:
    before = make_document("version one", "folder/note.md")
    after = make_document("version two", "folder/note.md")
    renamed = make_document("version two", "folder/renamed.md")

    assert before.document_id == after.document_id
    assert before.content_sha256 != after.content_sha256
    assert after.document_id != renamed.document_id


def test_inspection_report_excludes_text_by_default_and_includes_it_on_request() -> None:
    secret = "private-learning-text-12345"
    document = make_document(secret)
    options = ChunkingOptions(chunk_size=10, overlap=2)
    chunks = chunk_document(document, options)

    private_report = build_inspection_report(document, chunks, options, include_text=False)
    visible_report = build_inspection_report(document, chunks, options, include_text=True)

    assert secret not in private_report.model_dump_json(exclude_none=True)
    assert all(item.text is None for item in private_report.chunks)
    assert visible_report.content_included is True
    assert "".join(item.text or "" for item in visible_report.chunks)
    assert visible_report.chunks[0].text == chunks[0].text


def test_default_options_are_applied_when_omitted() -> None:
    document = make_document("r" * 1_201)

    chunks = chunk_document(document)

    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [(0, 1_200), (1_000, 1_201)]
