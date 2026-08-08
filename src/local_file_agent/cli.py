"""Command-line interface for the Local Personal File Agent."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated

import typer
from pydantic import ValidationError

from local_file_agent.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    ChunkingError,
    ChunkingOptions,
    ChunkInspectionReport,
    build_inspection_report,
    chunk_document,
)
from local_file_agent.config import Settings
from local_file_agent.doctor import DoctorReport, run_doctor
from local_file_agent.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_TOP_K,
    EmbeddingError,
    EmbeddingInspectionReport,
    EmbeddingService,
    build_embedding_inspection_report,
)
from local_file_agent.ingestion import ScanReport, SourceRootError, scan_source
from local_file_agent.ollama import OllamaClient, OllamaError

app = typer.Typer(
    name="file-agent",
    help="Learn and run a local personal-file RAG agent.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run local-file RAG commands."""


def _render_human(report: DoctorReport) -> None:
    for check in report.checks:
        typer.echo(f"[{check.status.value}] {check.name}: {check.summary}")
        for key, value in check.details.items():
            if value is not None:
                typer.echo(f"       {key}={value}")
    readiness = "READY" if report.full_readiness else "NOT FULLY READY"
    typer.echo(f"\nCheckpoint 0 runtime status: {readiness}")


def _safe_validation_messages(exc: ValidationError) -> list[str]:
    messages: list[str] = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        messages.append(f"{location}: {error['msg']}")
    return messages


def _render_scan_report(report: ScanReport) -> None:
    typer.echo(f"Scan completed for approved source: {report.source_name}")
    typer.echo(f"Accepted files: {report.summary.accepted_files}")
    typer.echo(f"Skipped entries: {report.summary.skipped_entries}")

    if report.accepted:
        typer.echo("\nAccepted metadata:")
        for accepted in report.accepted:
            typer.echo(
                f"- {accepted.relative_path} "
                f"(bytes={accepted.size_bytes}, chars={accepted.character_count}, "
                f"document_id={accepted.document_id[:12]}..., "
                f"sha256={accepted.content_sha256[:12]}...)"
            )

    if report.skipped:
        typer.echo("\nSkipped entries:")
        for skipped in report.skipped:
            typer.echo(f"- {skipped.relative_path} [{skipped.reason.value}]")

    typer.echo("\nNo document content was printed or persisted.")


def _normalize_document_selector(value: str) -> str:
    candidate = value.strip()
    windows_path = PureWindowsPath(candidate)
    normalized_path = PurePosixPath(candidate.replace("\\", "/"))
    if (
        not candidate
        or windows_path.is_absolute()
        or normalized_path.is_absolute()
        or normalized_path == PurePosixPath(".")
        or ".." in normalized_path.parts
    ):
        raise ValueError("Document must be a safe relative path returned by scan.")
    return normalized_path.as_posix()


def _render_chunk_report(report: ChunkInspectionReport) -> None:
    typer.echo(f"Chunk inspection completed for: {report.relative_path}")
    typer.echo(f"Document ID: {report.document_id[:12]}...")
    typer.echo(f"Document characters: {report.document_characters}")
    typer.echo(
        f"Configuration: chunk_size={report.chunk_size}, overlap={report.overlap}, "
        f"boundary_window={report.boundary_search_window}"
    )
    typer.echo(f"Chunks: {report.chunk_count}")

    if report.content_included:
        typer.echo("\nWARNING: Exact document text follows because --show-text was supplied.")

    for chunk in report.chunks:
        typer.echo(
            f"\nChunk {chunk.chunk_index} [{chunk.start_char}:{chunk.end_char}) "
            f"chars={chunk.character_count} "
            f"overlap={chunk.overlap_with_previous} "
            f"sha256={chunk.content_sha256[:12]}..."
        )
        if chunk.text is not None:
            overlap = chunk.overlap_with_previous
            if overlap:
                typer.echo("--- overlap with previous chunk ---")
                typer.echo(chunk.text[:overlap])
            typer.echo("--- new content ---")
            typer.echo(chunk.text[overlap:])

    if not report.content_included:
        typer.echo("\nNo document content was printed. Use --show-text only for approved files.")


def _render_embedding_report(report: EmbeddingInspectionReport) -> None:
    typer.echo(f"Embedding inspection completed for: {report.relative_path}")
    typer.echo(f"Document ID: {report.document_id[:12]}...")
    typer.echo(f"Model: requested={report.requested_model}, returned={report.returned_model}")
    typer.echo(
        f"Vectors: dimension={report.dimension}, chunks={report.chunk_count}, "
        f"batches={report.batch_count}, batch_size={report.batch_size}"
    )
    typer.echo(f"Prompt strategy: {report.prompt_strategy}")
    typer.echo(
        f"Document timing: wall_ms={report.document_wall_duration_ms}, "
        f"model_total_ms={report.document_total_duration_ms}"
    )
    typer.echo(
        f"Query: characters={report.query_characters}, "
        f"vector_norm={report.query_vector_norm}, "
        f"wall_ms={report.query_wall_duration_ms}"
    )

    typer.echo("\nBatch metrics:")
    for batch in report.batches:
        typer.echo(
            f"- batch={batch.batch_index} inputs={batch.input_count} "
            f"wall_ms={batch.wall_duration_ms} "
            f"model_total_ms={batch.total_duration_ms}"
        )

    if report.content_included:
        typer.echo("\nWARNING: Exact query and chunk text follow because --show-text was supplied.")
        typer.echo(f"Query text: {report.query_text}")

    typer.echo("\nCosine-similarity ranking (learning aid; not a persisted index):")
    for result in report.results:
        typer.echo(
            f"\nRank {result.rank}: chunk={result.chunk_index} "
            f"score={result.similarity} range=[{result.start_char}:{result.end_char}) "
            f"norm={result.vector_norm} sha256={result.content_sha256[:12]}..."
        )
        if result.text is not None:
            typer.echo(result.text)

    if not report.content_included:
        typer.echo(
            "\nNo query or document content was printed. Use --show-text only for approved files."
        )


@app.command()
def doctor(
    skip_generation: Annotated[
        bool,
        typer.Option(help="Skip Qwen inference; full readiness will not be claimed."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable diagnostic report."),
    ] = False,
) -> None:
    """Validate the local Python and Ollama runtime."""

    try:
        settings = Settings()
    except ValidationError as exc:
        typer.echo("Invalid FILE_AGENT configuration:", err=True)
        for message in _safe_validation_messages(exc):
            typer.echo(f"- {message}", err=True)
        raise typer.Exit(code=2) from exc

    with OllamaClient(settings) as client:
        report = run_doctor(settings, client, skip_generation=skip_generation)

    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        _render_human(report)

    if not report.success:
        raise typer.Exit(code=1)


@app.command()
def scan(
    source: Annotated[
        Path,
        typer.Option("--source", help="Explicitly approved folder to scan recursively."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable metadata-only report."),
    ] = False,
) -> None:
    """Securely discover and parse approved Markdown and text files."""

    try:
        outcome = scan_source(source)
    except SourceRootError as exc:
        typer.echo(f"Source folder rejected: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(outcome.report.model_dump_json(indent=2))
    else:
        _render_scan_report(outcome.report)


@app.command()
def inspect_chunks(
    source: Annotated[
        Path,
        typer.Option("--source", help="Explicitly approved folder to scan recursively."),
    ],
    document: Annotated[
        str,
        typer.Option("--document", help="Relative document path returned by scan."),
    ],
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Target number of characters per chunk."),
    ] = DEFAULT_CHUNK_SIZE,
    overlap: Annotated[
        int,
        typer.Option(help="Characters repeated from the preceding chunk."),
    ] = DEFAULT_OVERLAP,
    show_text: Annotated[
        bool,
        typer.Option(help="Explicitly include exact document text in the output."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable inspection report."),
    ] = False,
) -> None:
    """Inspect deterministic chunks for one trusted document."""

    try:
        selector = _normalize_document_selector(document)
        options = ChunkingOptions(chunk_size=chunk_size, overlap=overlap)
    except (ValueError, ChunkingError) as exc:
        typer.echo(f"Invalid chunk inspection options: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        outcome = scan_source(source)
    except SourceRootError as exc:
        typer.echo(f"Source folder rejected: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    selected = next(
        (item for item in outcome.documents if item.relative_path == selector),
        None,
    )
    if selected is None:
        typer.echo("Selected document was not accepted or found.", err=True)
        raise typer.Exit(code=1)

    chunks = chunk_document(selected, options)
    report = build_inspection_report(selected, chunks, options, include_text=show_text)
    if json_output:
        typer.echo(report.model_dump_json(indent=2, exclude_none=True))
    else:
        _render_chunk_report(report)


@app.command()
def inspect_embeddings(
    source: Annotated[
        Path,
        typer.Option("--source", help="Explicitly approved folder to scan recursively."),
    ],
    document: Annotated[
        str,
        typer.Option("--document", help="Relative document path returned by scan."),
    ],
    query: Annotated[
        str,
        typer.Option("--query", help="Question to embed and compare with document chunks."),
    ],
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Target number of characters per chunk."),
    ] = DEFAULT_CHUNK_SIZE,
    overlap: Annotated[
        int,
        typer.Option(help="Characters repeated from the preceding chunk."),
    ] = DEFAULT_OVERLAP,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Document prompts sent per Ollama request."),
    ] = DEFAULT_EMBEDDING_BATCH_SIZE,
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Highest-scoring chunks to show."),
    ] = DEFAULT_TOP_K,
    show_text: Annotated[
        bool,
        typer.Option(help="Explicitly include exact query and chunk text in the output."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable inspection report."),
    ] = False,
) -> None:
    """Embed one trusted document and inspect query-to-chunk similarity."""

    try:
        selector = _normalize_document_selector(document)
        options = ChunkingOptions(chunk_size=chunk_size, overlap=overlap)
        if batch_size < 1:
            raise ValueError("Embedding batch size must be at least one.")
        if top_k < 1:
            raise ValueError("Top-k must be at least one.")
        if not query.strip():
            raise ValueError("Embedding query must not be empty.")
        settings = Settings()
    except ValidationError as exc:
        typer.echo("Invalid FILE_AGENT configuration:", err=True)
        for message in _safe_validation_messages(exc):
            typer.echo(f"- {message}", err=True)
        raise typer.Exit(code=2) from exc
    except (ValueError, ChunkingError) as exc:
        typer.echo(f"Invalid embedding inspection options: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        outcome = scan_source(source)
    except SourceRootError as exc:
        typer.echo(f"Source folder rejected: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    selected = next(
        (item for item in outcome.documents if item.relative_path == selector),
        None,
    )
    if selected is None:
        typer.echo("Selected document was not accepted or found.", err=True)
        raise typer.Exit(code=1)

    chunks = chunk_document(selected, options)
    if not chunks:
        typer.echo("Selected document did not produce any chunks.", err=True)
        raise typer.Exit(code=1)

    try:
        with OllamaClient(settings) as client:
            service = EmbeddingService(client, settings.embedding_model)
            document_run = service.embed_documents(chunks, batch_size=batch_size)
            query_embedding = service.embed_query(
                query,
                expected_dimension=document_run.dimension,
            )
        report = build_embedding_inspection_report(
            selected.character_count,
            document_run,
            query,
            query_embedding,
            chunk_size=options.chunk_size,
            overlap=options.overlap,
            top_k=top_k,
            include_text=show_text,
        )
    except (OllamaError, EmbeddingError) as exc:
        typer.echo(f"Embedding inspection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(report.model_dump_json(indent=2, exclude_none=True))
    else:
        _render_embedding_report(report)
