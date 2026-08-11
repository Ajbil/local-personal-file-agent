"""Command-line interface for the Local Personal File Agent."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated

import typer
from pydantic import ValidationError

from local_file_agent.answering import AnswerGenerationError, AnswerReport, answer_database
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
from local_file_agent.evaluation import (
    DEFAULT_MANIFEST,
    EvaluationError,
    EvaluationInputError,
    EvaluationMode,
    EvaluationReport,
    run_evaluation,
)
from local_file_agent.indexing import (
    IndexBuildError,
    IndexBuildReport,
    IndexInspectionReport,
    build_index,
)
from local_file_agent.indexing import (
    inspect_index as inspect_stored_index,
)
from local_file_agent.ingestion import ScanReport, SourceRootError, scan_source
from local_file_agent.ollama import OllamaClient, OllamaError
from local_file_agent.retrieval import (
    DEFAULT_MIN_SCORE,
    RetrievalError,
    SearchOptions,
    SearchReport,
    search_database,
)
from local_file_agent.retrieval import (
    DEFAULT_TOP_K as DEFAULT_SEARCH_TOP_K,
)
from local_file_agent.storage import IndexStorageError

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


def _render_evaluation_report(report: EvaluationReport) -> None:
    typer.echo(
        f"Evaluation {report.status.upper()}: {report.passed_cases}/{report.case_count} cases"
    )
    typer.echo(f"Mode: {report.mode.value}")
    typer.echo(
        f"Retrieval: hit@{report.top_k}={report.metrics.hit_at_k:.3f}, "
        f"MRR={report.metrics.mean_reciprocal_rank:.3f}"
    )
    typer.echo(
        f"Answers: facts={report.metrics.answer_fact_accuracy:.3f}, "
        f"refusals={report.metrics.refusal_accuracy:.3f}"
    )
    typer.echo(
        f"Citations: valid={report.metrics.citation_validity:.3f}, "
        f"precise={report.metrics.citation_precision:.3f}"
    )
    typer.echo(f"Security leakage count: {report.metrics.security_leakage_count}")
    typer.echo("\nCases (content intentionally omitted):")
    for case in report.cases:
        state = "PASS" if case.passed else "FAIL"
        rank = case.first_relevant_rank if case.first_relevant_rank is not None else "n/a"
        suffix = f", failure_stage={case.failure_stage}" if case.failure_stage else ""
        typer.echo(
            f"[{state}] {case.case_id}: rank={rank}, decision={case.decision_reason}{suffix}"
        )


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


def _render_index_build_report(report: IndexBuildReport) -> None:
    typer.echo("SQLite vector index created and validated.")
    typer.echo(f"Schema: version={report.schema_version}, vector_format={report.vector_format}")
    typer.echo(
        f"Model: requested={report.requested_embedding_model}, "
        f"stored={report.embedding_model}, dimension={report.embedding_dimension}"
    )
    typer.echo(f"Prompt strategy: {report.prompt_strategy}")
    typer.echo(
        f"Chunking: chunk_size={report.chunk_size}, overlap={report.overlap}, "
        f"batch_size={report.batch_size}, batches={report.batch_count}"
    )
    typer.echo(
        f"Rows: documents={report.accepted_documents}, chunks={report.chunk_count}, "
        f"embeddings={report.embedding_count}, skipped_entries={report.skipped_entries}"
    )
    typer.echo(f"Corpus fingerprint: {report.corpus_fingerprint}")
    typer.echo(f"Replaced existing index: {report.replaced_existing}")
    typer.echo(
        f"Embedding timing: wall_ms={report.embedding_wall_duration_ms}, "
        f"model_total_ms={report.embedding_total_duration_ms}"
    )
    typer.echo("No document text or vector coordinates were printed.")


def _render_index_inspection_report(report: IndexInspectionReport) -> None:
    typer.echo("SQLite vector index is valid and was opened read-only.")
    typer.echo(f"Schema: version={report.schema_version}, vector_format={report.vector_format}")
    typer.echo(
        f"Model: requested={report.requested_embedding_model}, "
        f"stored={report.embedding_model}, dimension={report.embedding_dimension}"
    )
    typer.echo(f"Prompt strategy: {report.prompt_strategy}")
    typer.echo(f"Chunking: chunk_size={report.chunk_size}, overlap={report.overlap}")
    typer.echo(
        f"Rows: documents={report.document_count}, chunks={report.chunk_count}, "
        f"embeddings={report.embedding_count}"
    )
    typer.echo(f"Corpus fingerprint: {report.corpus_fingerprint}")
    typer.echo(
        f"Integrity: sqlite={report.integrity_check}, "
        f"foreign_keys_valid={report.foreign_keys_valid}"
    )
    typer.echo("No document text or vector coordinates were printed.")


def _render_search_report(report: SearchReport) -> None:
    typer.echo("Read-only vector search completed.")
    typer.echo(
        f"Model: stored={report.stored_embedding_model}, "
        f"returned={report.returned_embedding_model}, "
        f"dimension={report.embedding_dimension}"
    )
    typer.echo(
        f"Policy: top_k={report.top_k}, min_score={report.min_score}, "
        f"max_overlap_ratio={report.max_overlap_ratio}"
    )
    typer.echo(
        f"Candidates: indexed={report.indexed_chunk_count}, "
        f"above_threshold={report.above_threshold_count}, "
        f"suppressed={report.suppressed_count}, selected={report.result_count}"
    )
    typer.echo(
        f"Timing: index_load_ms={report.index_load_duration_ms}, "
        f"query_embedding_ms={report.query_embedding_wall_duration_ms}, "
        f"retrieval_ms={report.retrieval_wall_duration_ms}"
    )

    if not report.results:
        typer.echo(f"\nNo chunks met min_score={report.min_score}.")
        return
    if report.content_included:
        typer.echo("\nWARNING: Exact retrieved text follows because --show-text was supplied.")

    for result in report.results:
        typer.echo(
            f"\n[{result.rank}] {result.citation} "
            f"score={result.similarity} sha256={result.content_sha256[:12]}..."
        )
        if result.text is not None:
            typer.echo(result.text)

    if not report.content_included:
        typer.echo("\nNo retrieved text was printed. Use --show-text only for approved indexes.")


def _render_answer_report(report: AnswerReport) -> None:
    heading = (
        "Grounded answer generated." if report.status == "answered" else "Answer refused safely."
    )
    typer.echo(heading)
    typer.echo(f"Decision: {report.decision_reason.value}")
    typer.echo(
        f"Models: embedding={report.returned_embedding_model}, "
        f"answer={report.answer_model_returned or 'not_called'}"
    )
    typer.echo(
        f"Evidence: retrieved={report.retrieved_count}, context={report.context_count}, "
        f"characters={report.context_characters}, cited={len(report.citations)}, "
        f"budget_truncated={report.context_truncated}"
    )
    typer.echo(
        f"Timing: index_load_ms={report.index_load_duration_ms}, "
        f"query_embedding_ms={report.query_embedding_wall_duration_ms}, "
        f"retrieval_ms={report.retrieval_wall_duration_ms}, "
        f"generation_ms={report.generation_wall_duration_ms}, "
        f"generation_attempts={report.generation_attempts}"
    )
    typer.echo(f"\nAnswer:\n{report.answer}")

    if report.citations:
        typer.echo("\nSources:")
        for citation in report.citations:
            typer.echo(
                f"[{citation.citation_id}] {citation.citation} "
                f"score={citation.similarity} sha256={citation.content_sha256[:12]}..."
            )
    else:
        typer.echo("\nSources: none")

    if report.context_included:
        typer.echo(
            "\nWARNING: Exact context sent to Qwen follows because --show-context was supplied."
        )
        for passage in report.context or []:
            typer.echo(
                f"\n[{passage.citation_id}] {passage.citation} "
                f"score={passage.similarity} sha256={passage.content_sha256[:12]}..."
            )
            typer.echo(passage.text)
    elif report.context_count:
        typer.echo(
            "\nRetrieved context was not printed. Use --show-context only for approved indexes."
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


@app.command("index")
def index_documents(
    source: Annotated[
        Path,
        typer.Option("--source", help="Explicitly approved folder to index recursively."),
    ],
    database: Annotated[
        Path,
        typer.Option("--db", help="SQLite index path outside the approved source folder."),
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
    force: Annotated[
        bool,
        typer.Option(help="Replace an existing valid app-owned index."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable metadata-only report."),
    ] = False,
) -> None:
    """Build and atomically publish a validated local SQLite vector index."""

    try:
        settings = Settings()
        chunking = ChunkingOptions(chunk_size=chunk_size, overlap=overlap)
        if batch_size < 1:
            raise ValueError("Embedding batch size must be at least one.")
    except ValidationError as exc:
        typer.echo("Invalid FILE_AGENT configuration:", err=True)
        for message in _safe_validation_messages(exc):
            typer.echo(f"- {message}", err=True)
        raise typer.Exit(code=2) from exc
    except (ValueError, ChunkingError) as exc:
        typer.echo(f"Invalid index options: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        with OllamaClient(settings) as client:
            report = build_index(
                source,
                database,
                client,
                settings.embedding_model,
                chunking=chunking,
                batch_size=batch_size,
                force=force,
            )
    except (SourceRootError, OllamaError, EmbeddingError, IndexBuildError) as exc:
        typer.echo(f"Index build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(report.model_dump_json(indent=2, exclude_none=True))
    else:
        _render_index_build_report(report)


@app.command()
def inspect_index(
    database: Annotated[
        Path,
        typer.Option("--db", help="Existing Local Personal File Agent SQLite index."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable metadata-only report."),
    ] = False,
) -> None:
    """Open an existing index read-only and validate all stored records."""

    try:
        report = inspect_stored_index(database)
    except IndexStorageError as exc:
        typer.echo(f"Index inspection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        _render_index_inspection_report(report)


@app.command()
def search(
    question: Annotated[
        str,
        typer.Argument(help="Question to compare with the persisted chunk vectors."),
    ],
    database: Annotated[
        Path,
        typer.Option("--db", help="Existing Local Personal File Agent SQLite index."),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Maximum number of diverse results to return."),
    ] = DEFAULT_SEARCH_TOP_K,
    min_score: Annotated[
        float,
        typer.Option("--min-score", help="Inclusive cosine-similarity threshold."),
    ] = DEFAULT_MIN_SCORE,
    show_text: Annotated[
        bool,
        typer.Option(help="Explicitly include exact selected chunk text in the output."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable search report."),
    ] = False,
) -> None:
    """Retrieve relevant passages from a validated read-only SQLite index."""

    try:
        settings = Settings()
        options = SearchOptions(top_k=top_k, min_score=min_score)
        if not question.strip():
            raise ValueError("Search question must not be empty.")
    except ValidationError as exc:
        typer.echo("Invalid FILE_AGENT configuration:", err=True)
        for message in _safe_validation_messages(exc):
            typer.echo(f"- {message}", err=True)
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        typer.echo(f"Invalid search options: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        with OllamaClient(settings) as client:
            report = search_database(
                database,
                question,
                client,
                options=options,
                include_text=show_text,
            )
    except (IndexStorageError, OllamaError, EmbeddingError, RetrievalError) as exc:
        typer.echo(f"Search failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(report.model_dump_json(indent=2, exclude_none=True))
    else:
        _render_search_report(report)


@app.command()
def evaluate(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Strict synthetic evaluation manifest."),
    ] = DEFAULT_MANIFEST,
    mode: Annotated[
        EvaluationMode,
        typer.Option("--mode", help="Offline deterministic or real local-model evaluation."),
    ] = EvaluationMode.DETERMINISTIC,
    chunk_size: Annotated[
        int | None,
        typer.Option("--chunk-size", help="Override the manifest chunk size."),
    ] = None,
    overlap: Annotated[
        int | None,
        typer.Option("--overlap", help="Override the manifest chunk overlap."),
    ] = None,
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", help="Override the manifest retrieval depth."),
    ] = None,
    min_score: Annotated[
        float | None,
        typer.Option("--min-score", help="Override the mode-specific similarity threshold."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a privacy-safe machine-readable report."),
    ] = False,
) -> None:
    """Run the disposable synthetic quality and security regression suite."""

    try:
        settings = Settings() if mode is EvaluationMode.LIVE else None
    except ValidationError as exc:
        typer.echo("Invalid FILE_AGENT configuration:", err=True)
        for message in _safe_validation_messages(exc):
            typer.echo(f"- {message}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        if mode is EvaluationMode.LIVE:
            assert settings is not None
            with OllamaClient(settings) as client:
                report = run_evaluation(
                    manifest,
                    mode=mode,
                    live_gateway=client,
                    live_embedding_model=settings.embedding_model,
                    live_answer_model=settings.answer_model,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    top_k=top_k,
                    min_score=min_score,
                )
        else:
            report = run_evaluation(
                manifest,
                mode=mode,
                chunk_size=chunk_size,
                overlap=overlap,
                top_k=top_k,
                min_score=min_score,
            )
    except EvaluationInputError as exc:
        typer.echo(f"Invalid evaluation options: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (
        AnswerGenerationError,
        EmbeddingError,
        EvaluationError,
        IndexBuildError,
        IndexStorageError,
        OllamaError,
        RetrievalError,
        SourceRootError,
    ) as exc:
        typer.echo(f"Evaluation failed to run: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(report.model_dump_json(indent=2, exclude_none=True))
    else:
        _render_evaluation_report(report)
    if report.status == "failed":
        raise typer.Exit(code=1)


@app.command()
def ask(
    question: Annotated[
        str,
        typer.Argument(help="Question to answer from retrieved local evidence."),
    ],
    database: Annotated[
        Path,
        typer.Option("--db", help="Existing Local Personal File Agent SQLite index."),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Maximum number of diverse passages to retrieve."),
    ] = DEFAULT_SEARCH_TOP_K,
    min_score: Annotated[
        float,
        typer.Option("--min-score", help="Inclusive cosine-similarity threshold."),
    ] = DEFAULT_MIN_SCORE,
    show_context: Annotated[
        bool,
        typer.Option(help="Explicitly include the exact passages sent to Qwen."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable grounded-answer report."),
    ] = False,
) -> None:
    """Retrieve evidence and generate a locally grounded, cited answer."""

    try:
        settings = Settings()
        options = SearchOptions(top_k=top_k, min_score=min_score)
        if not question.strip():
            raise ValueError("Answer question must not be empty.")
    except ValidationError as exc:
        typer.echo("Invalid FILE_AGENT configuration:", err=True)
        for message in _safe_validation_messages(exc):
            typer.echo(f"- {message}", err=True)
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        typer.echo(f"Invalid answer options: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        with OllamaClient(settings) as client:
            report = answer_database(
                database,
                question,
                client,
                settings.answer_model,
                options=options,
                include_context=show_context,
            )
    except (
        AnswerGenerationError,
        IndexStorageError,
        OllamaError,
        EmbeddingError,
        RetrievalError,
    ) as exc:
        typer.echo(f"Answer generation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(report.model_dump_json(indent=2, exclude_none=True))
    else:
        _render_answer_report(report)
