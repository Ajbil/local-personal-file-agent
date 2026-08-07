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
from local_file_agent.ingestion import ScanReport, SourceRootError, scan_source
from local_file_agent.ollama import OllamaClient

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
