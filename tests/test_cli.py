"""Tests for stable command-line behavior and exit codes."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

import local_file_agent.cli as cli
from local_file_agent.chunking import Chunk
from local_file_agent.config import Settings
from local_file_agent.doctor import CheckResult, CheckStatus, DoctorReport
from local_file_agent.embeddings import (
    PROMPT_STRATEGY,
    DocumentEmbeddingRun,
    EmbeddedChunk,
    EmbeddingBatchMetrics,
    EmbeddingError,
    QueryEmbedding,
)
from local_file_agent.ollama import ChatMessage, EmbeddingBatch, StructuredChatResult

runner = CliRunner()


class DummyClient:
    def __init__(self, _settings: Settings) -> None:
        pass

    def __enter__(self) -> "DummyClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        pass


def test_help_exposes_doctor_command() -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "inspect-chunks" in result.stdout
    assert "inspect-embeddings" in result.stdout
    assert "inspect-index" in result.stdout
    assert "index" in result.stdout
    assert "scan" in result.stdout
    assert "search" in result.stdout
    assert "ask" in result.stdout
    assert "evaluate" in result.stdout
    assert "--log-level" in result.stdout


def test_invalid_remote_configuration_returns_exit_code_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FILE_AGENT_OLLAMA_BASE_URL", "https://example.com")

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 2
    assert "Invalid FILE_AGENT configuration" in result.output
    assert "https://example.com" not in result.output


def test_json_report_is_machine_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    report = DoctorReport(
        success=True,
        full_readiness=True,
        checks=[
            CheckResult(
                name="test",
                status=CheckStatus.PASS,
                summary="diagnostics passed",
            )
        ],
    )

    def fake_run_doctor(
        _settings: Settings,
        _gateway: object,
        *,
        skip_generation: bool = False,
    ) -> DoctorReport:
        assert skip_generation is False
        return report

    monkeypatch.setattr(cli, "OllamaClient", DummyClient)
    monkeypatch.setattr(cli, "run_doctor", fake_run_doctor)

    result = runner.invoke(cli.app, ["doctor", "--json"])

    assert result.exit_code == 0
    assert '"full_readiness": true' in result.stdout
    assert '"status": "PASS"' in result.stdout


def test_human_report_returns_exit_one_on_failed_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = DoctorReport(
        success=False,
        full_readiness=False,
        checks=[
            CheckResult(
                name="ollama_api",
                status=CheckStatus.FAIL,
                summary="Could not connect to local Ollama",
                details={"remediation": "Start Ollama"},
            )
        ],
    )

    def fake_run_doctor(
        _settings: Settings,
        _gateway: object,
        *,
        skip_generation: bool = False,
    ) -> DoctorReport:
        assert skip_generation is False
        return report

    monkeypatch.setattr(cli, "OllamaClient", DummyClient)
    monkeypatch.setattr(cli, "run_doctor", fake_run_doctor)

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "[FAIL] ollama_api" in result.stdout
    assert "remediation=Start Ollama" in result.stdout
    assert "NOT FULLY READY" in result.stdout


def test_scan_requires_an_explicit_source() -> None:
    result = runner.invoke(cli.app, ["scan"])

    assert result.exit_code == 2
    assert "--source" in result.output


def test_scan_human_output_is_metadata_only(tmp_path: Path) -> None:
    secret_text = "a private sentence"
    (tmp_path / "note.md").write_text(secret_text, encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"image")

    result = runner.invoke(cli.app, ["scan", "--source", str(tmp_path)])

    assert result.exit_code == 0
    assert "Accepted files: 1" in result.output
    assert "Skipped entries: 1" in result.output
    assert "note.md" in result.output
    assert "unsupported_extension" in result.output
    assert secret_text not in result.output
    assert str(tmp_path) not in result.output


def test_scan_json_output_is_machine_readable_and_content_free(tmp_path: Path) -> None:
    secret_text = "another private sentence"
    (tmp_path / "note.txt").write_text(secret_text, encoding="utf-8")

    result = runner.invoke(cli.app, ["scan", "--source", str(tmp_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["summary"]["accepted_files"] == 1
    assert payload["accepted"][0]["relative_path"] == "note.txt"
    assert secret_text not in result.stdout
    assert str(tmp_path) not in result.stdout


def test_scan_invalid_root_returns_exit_one_without_echoing_path(tmp_path: Path) -> None:
    missing = tmp_path / "private-folder-name"

    result = runner.invoke(cli.app, ["scan", "--source", str(missing)])

    assert result.exit_code == 1
    assert "Source folder rejected" in result.output
    assert str(missing) not in result.output


def test_inspect_chunks_requires_source_and_document() -> None:
    missing_source = runner.invoke(cli.app, ["inspect-chunks"])
    missing_document = runner.invoke(
        cli.app,
        ["inspect-chunks", "--source", "."],
    )

    assert missing_source.exit_code == 2
    assert "--source" in missing_source.output
    assert missing_document.exit_code == 2
    assert "--document" in missing_document.output


def test_inspect_chunks_metadata_is_private_by_default(tmp_path: Path) -> None:
    secret = "private-chunk-content-" * 4
    (tmp_path / "note.md").write_text(secret, encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "inspect-chunks",
            "--source",
            str(tmp_path),
            "--document",
            "note.md",
            "--chunk-size",
            "30",
            "--overlap",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert "Chunks:" in result.output
    assert "[0:30)" in result.output
    assert "overlap=5" in result.output
    assert secret not in result.output
    assert str(tmp_path) not in result.output
    assert "No document content was printed" in result.output


def test_inspect_chunks_show_text_is_explicit_and_marks_overlap(tmp_path: Path) -> None:
    content = "abcdefghijABCDEFGHIJklmnopqrst"
    (tmp_path / "note.txt").write_text(content, encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "inspect-chunks",
            "--source",
            str(tmp_path),
            "--document",
            "note.txt",
            "--chunk-size",
            "20",
            "--overlap",
            "5",
            "--show-text",
        ],
    )

    assert result.exit_code == 0
    assert "WARNING: Exact document text follows" in result.output
    assert "--- overlap with previous chunk ---" in result.output
    assert "abcdefghijABCDEFGHIJ" in result.output


def test_inspect_chunks_json_omits_or_includes_text_by_policy(tmp_path: Path) -> None:
    content = "json-private-content"
    (tmp_path / "note.md").write_text(content, encoding="utf-8")
    base_args = [
        "inspect-chunks",
        "--source",
        str(tmp_path),
        "--document",
        "note.md",
        "--json",
    ]

    private_result = runner.invoke(cli.app, base_args)
    visible_result = runner.invoke(cli.app, [*base_args, "--show-text"])

    assert private_result.exit_code == 0
    private_payload = json.loads(private_result.stdout)
    assert private_payload["content_included"] is False
    assert "text" not in private_payload["chunks"][0]
    assert content not in private_result.stdout
    assert visible_result.exit_code == 0
    visible_payload = json.loads(visible_result.stdout)
    assert visible_payload["content_included"] is True
    assert visible_payload["chunks"][0]["text"] == content


def test_inspect_chunks_accepts_windows_relative_separator(tmp_path: Path) -> None:
    nested = tmp_path / "folder"
    nested.mkdir()
    (nested / "note.md").write_text("nested", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "inspect-chunks",
            "--source",
            str(tmp_path),
            "--document",
            "folder\\note.md",
        ],
    )

    assert result.exit_code == 0
    assert "folder/note.md" in result.output


@pytest.mark.parametrize("selector", ["../outside.md", "C:\\private\\note.md", "/note.md", "."])
def test_inspect_chunks_rejects_unsafe_document_selector(
    tmp_path: Path,
    selector: str,
) -> None:
    result = runner.invoke(
        cli.app,
        ["inspect-chunks", "--source", str(tmp_path), "--document", selector],
    )

    assert result.exit_code == 2
    assert "safe relative path" in result.output


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [("0", "0"), ("10", "-1"), ("10", "10")],
)
def test_inspect_chunks_rejects_invalid_chunking_options(
    tmp_path: Path,
    chunk_size: str,
    overlap: str,
) -> None:
    result = runner.invoke(
        cli.app,
        [
            "inspect-chunks",
            "--source",
            str(tmp_path),
            "--document",
            "note.md",
            "--chunk-size",
            chunk_size,
            "--overlap",
            overlap,
        ],
    )

    assert result.exit_code == 2
    assert "Invalid chunk inspection options" in result.output


def test_inspect_chunks_rejects_missing_or_unsupported_document(tmp_path: Path) -> None:
    private_content = "UNIQUE-UNSUPPORTED-DOCUMENT-CONTENT"
    (tmp_path / "unsupported.csv").write_text(private_content, encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "inspect-chunks",
            "--source",
            str(tmp_path),
            "--document",
            "unsupported.csv",
        ],
    )

    assert result.exit_code == 1
    assert "not accepted or found" in result.output
    assert private_content not in result.output


def test_inspect_chunks_rejects_invalid_source_without_leaking_path(tmp_path: Path) -> None:
    missing = tmp_path / "sensitive-source-name"

    result = runner.invoke(
        cli.app,
        ["inspect-chunks", "--source", str(missing), "--document", "note.md"],
    )

    assert result.exit_code == 1
    assert "Source folder rejected" in result.output
    assert str(missing) not in result.output


class FakeEmbeddingService:
    def __init__(self, _gateway: object, model: str) -> None:
        self.model = model

    def embed_documents(self, chunks: Sequence[Chunk], *, batch_size: int) -> DocumentEmbeddingRun:
        embedded = tuple(
            EmbeddedChunk(
                chunk,
                np.asarray([1.0, float(index)], dtype=np.float32),
            )
            for index, chunk in enumerate(chunks)
        )
        return DocumentEmbeddingRun(
            requested_model=self.model,
            returned_model=f"{self.model}:latest",
            prompt_strategy=PROMPT_STRATEGY,
            dimension=2,
            embedded_chunks=embedded,
            batch_size=batch_size,
            batches=(EmbeddingBatchMetrics(0, len(chunks), 1.0, 0.5, 0.1, len(chunks)),),
            wall_duration_ms=1.0,
            total_duration_ms=0.5,
        )

    def embed_query(self, _query: str, *, expected_dimension: int | None) -> QueryEmbedding:
        assert expected_dimension == 2
        return QueryEmbedding(
            requested_model=self.model,
            returned_model=f"{self.model}:latest",
            prompt_strategy=PROMPT_STRATEGY,
            dimension=2,
            vector=np.asarray([1.0, 0.0], dtype=np.float32),
            wall_duration_ms=1.0,
            total_duration_ms=0.5,
            load_duration_ms=0.1,
            prompt_eval_count=1,
        )


def _install_fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "OllamaClient", DummyClient)
    monkeypatch.setattr(cli, "EmbeddingService", FakeEmbeddingService)


def test_inspect_embeddings_is_private_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "PRIVATE-EMBEDDING-CONTENT-" * 3
    secret_query = "PRIVATE QUERY"
    (tmp_path / "note.md").write_text(secret, encoding="utf-8")
    _install_fake_embeddings(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "inspect-embeddings",
            "--source",
            str(tmp_path),
            "--document",
            "note.md",
            "--query",
            secret_query,
            "--chunk-size",
            "30",
            "--overlap",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert "dimension=2" in result.output
    assert "Cosine-similarity ranking" in result.output
    assert "score=" in result.output
    assert secret not in result.output
    assert secret_query not in result.output
    assert str(tmp_path) not in result.output
    assert "No query or document content was printed" in result.output
    assert "[1.0, 0.0]" not in result.output


def test_inspect_embeddings_json_privacy_and_show_text_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = "A synthetic explanation of retrieval augmented generation."
    query = "What is retrieval?"
    (tmp_path / "note.txt").write_text(content, encoding="utf-8")
    _install_fake_embeddings(monkeypatch)
    arguments = [
        "inspect-embeddings",
        "--source",
        str(tmp_path),
        "--document",
        "note.txt",
        "--query",
        query,
        "--json",
    ]

    private_result = runner.invoke(cli.app, arguments)
    visible_result = runner.invoke(cli.app, [*arguments, "--show-text"])

    assert private_result.exit_code == 0
    private = json.loads(private_result.stdout)
    assert private["content_included"] is False
    assert "query_text" not in private
    assert "text" not in private["results"][0]
    assert '"vector":' not in private_result.stdout
    assert query not in private_result.stdout
    assert visible_result.exit_code == 0
    visible = json.loads(visible_result.stdout)
    assert visible["query_text"] == query
    assert visible["results"][0]["text"] == content


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--batch-size", "0"], "batch size"),
        (["--top-k", "0"], "Top-k"),
        (["--query", "   "], "query must not be empty"),
    ],
)
def test_inspect_embeddings_rejects_invalid_options_before_ollama(
    tmp_path: Path,
    extra: list[str],
    message: str,
) -> None:
    (tmp_path / "note.md").write_text("content", encoding="utf-8")
    arguments = [
        "inspect-embeddings",
        "--source",
        str(tmp_path),
        "--document",
        "note.md",
        "--query",
        "question",
    ]
    if "--query" in extra:
        arguments = arguments[:-2]

    result = runner.invoke(cli.app, [*arguments, *extra])

    assert result.exit_code == 2
    assert message in result.output


def test_inspect_embeddings_rejects_empty_document_without_model_call(
    tmp_path: Path,
) -> None:
    (tmp_path / "empty.md").write_text("", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "inspect-embeddings",
            "--source",
            str(tmp_path),
            "--document",
            "empty.md",
            "--query",
            "question",
        ],
    )

    assert result.exit_code == 1
    assert "did not produce any chunks" in result.output


def test_inspect_embeddings_converts_domain_failure_to_safe_cli_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_content = "DO-NOT-LEAK-THIS-CONTENT"
    (tmp_path / "note.md").write_text(private_content, encoding="utf-8")

    class FailingService(FakeEmbeddingService):
        def embed_documents(
            self, chunks: Sequence[Chunk], *, batch_size: int
        ) -> DocumentEmbeddingRun:
            raise EmbeddingError("Embedding vector dimension does not match.")

    monkeypatch.setattr(cli, "OllamaClient", DummyClient)
    monkeypatch.setattr(cli, "EmbeddingService", FailingService)
    result = runner.invoke(
        cli.app,
        [
            "inspect-embeddings",
            "--source",
            str(tmp_path),
            "--document",
            "note.md",
            "--query",
            "question",
        ],
    )

    assert result.exit_code == 1
    assert "Embedding inspection failed" in result.output
    assert "dimension does not match" in result.output
    assert private_content not in result.output


class IndexClient(DummyClient):
    def embed(
        self,
        _model: str,
        inputs: Sequence[str],
        *,
        truncate: bool = False,
    ) -> EmbeddingBatch:
        assert truncate is False
        return EmbeddingBatch(
            model="embeddinggemma",
            vectors=[[1.0, float(index + 1)] for index, _ in enumerate(inputs)],
        )


def test_index_and_inspect_index_outputs_are_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "private-source-name"
    source.mkdir()
    secret = "PRIVATE-PERSISTED-CONTENT-" * 15
    (source / "note.md").write_text(secret, encoding="utf-8")
    database = tmp_path / "learning.sqlite"
    monkeypatch.setattr(cli, "OllamaClient", IndexClient)

    build_result = runner.invoke(
        cli.app,
        [
            "index",
            "--source",
            str(source),
            "--db",
            str(database),
            "--chunk-size",
            "100",
            "--overlap",
            "20",
            "--batch-size",
            "2",
        ],
    )
    inspect_result = runner.invoke(
        cli.app,
        ["inspect-index", "--db", str(database)],
    )

    assert build_result.exit_code == 0
    assert "SQLite vector index created and validated" in build_result.output
    assert "documents=1" in build_result.output
    assert "dimension=2" in build_result.output
    assert secret not in build_result.output
    assert str(source) not in build_result.output
    assert "[1.0," not in build_result.output
    assert inspect_result.exit_code == 0
    assert "opened read-only" in inspect_result.output
    assert secret not in inspect_result.output
    assert str(database) not in inspect_result.output


def test_index_and_inspect_index_json_reports_are_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    secret = "SYNTHETIC SECRET TEXT " * 10
    (source / "note.txt").write_text(secret, encoding="utf-8")
    database = tmp_path / "index.sqlite"
    monkeypatch.setattr(cli, "OllamaClient", IndexClient)

    build_result = runner.invoke(
        cli.app,
        [
            "index",
            "--source",
            str(source),
            "--db",
            str(database),
            "--json",
        ],
    )
    inspect_result = runner.invoke(
        cli.app,
        ["inspect-index", "--db", str(database), "--json"],
    )

    assert build_result.exit_code == 0
    build_payload = json.loads(build_result.stdout)
    assert build_payload["content_included"] is False
    assert build_payload["chunk_count"] == build_payload["embedding_count"]
    assert "text" not in build_result.stdout
    assert "vector" not in build_payload
    assert secret not in build_result.stdout
    assert inspect_result.exit_code == 0
    inspection = json.loads(inspect_result.stdout)
    assert inspection["opened_read_only"] is True
    assert inspection["integrity_check"] == "ok"
    assert inspection["content_included"] is False


def test_index_existing_target_requires_force_without_leaking_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    secret = "DO-NOT-PRINT-ME"
    (source / "note.md").write_text(secret, encoding="utf-8")
    database = tmp_path / "index.sqlite"
    monkeypatch.setattr(cli, "OllamaClient", IndexClient)
    arguments = ["index", "--source", str(source), "--db", str(database)]
    assert runner.invoke(cli.app, arguments).exit_code == 0

    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 1
    assert "--force" in result.output
    assert secret not in result.output


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--batch-size", "0"], "batch size"),
        (["--chunk-size", "10", "--overlap", "10"], "smaller than chunk size"),
    ],
)
def test_index_rejects_invalid_options(tmp_path: Path, extra: list[str], message: str) -> None:
    source = tmp_path / "source"
    source.mkdir()

    result = runner.invoke(
        cli.app,
        [
            "index",
            "--source",
            str(source),
            "--db",
            str(tmp_path / "index.sqlite"),
            *extra,
        ],
    )

    assert result.exit_code == 2
    assert "Invalid index options" in result.output
    assert message in result.output


def test_inspect_index_rejects_unrelated_database(tmp_path: Path) -> None:
    database = tmp_path / "unrelated.sqlite"
    database.write_bytes(b"not sqlite")

    result = runner.invoke(cli.app, ["inspect-index", "--db", str(database)])

    assert result.exit_code == 1
    assert "Index inspection failed" in result.output
    assert str(database) not in result.output


class SearchClient(DummyClient):
    requested_models: ClassVar[list[str]] = []

    def embed(
        self,
        model: str,
        inputs: Sequence[str],
        *,
        truncate: bool = False,
    ) -> EmbeddingBatch:
        assert truncate is False
        self.requested_models.append(model)
        return EmbeddingBatch(model="embeddinggemma", vectors=[[1.0, 1.0]])


class NoMatchSearchClient(SearchClient):
    def embed(
        self,
        model: str,
        inputs: Sequence[str],
        *,
        truncate: bool = False,
    ) -> EmbeddingBatch:
        assert truncate is False
        self.requested_models.append(model)
        return EmbeddingBatch(model="embeddinggemma", vectors=[[-1.0, -1.0]])


def _build_search_test_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "note.md").write_text(content, encoding="utf-8")
    database = tmp_path / "index.sqlite"
    monkeypatch.setattr(cli, "OllamaClient", IndexClient)
    result = runner.invoke(
        cli.app,
        ["index", "--source", str(source), "--db", str(database)],
    )
    assert result.exit_code == 0
    return database


def test_search_is_private_read_only_and_uses_index_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "PRIVATE RETRIEVED PASSAGE"
    database = _build_search_test_index(tmp_path, monkeypatch, secret)
    before = (database.read_bytes(), database.stat().st_mtime_ns)
    SearchClient.requested_models.clear()
    monkeypatch.setattr(cli, "OllamaClient", SearchClient)
    monkeypatch.setenv("FILE_AGENT_EMBEDDING_MODEL", "different-configured-model")

    result = runner.invoke(
        cli.app,
        ["search", "PRIVATE QUERY", "--db", str(database)],
    )

    assert result.exit_code == 0
    assert "Read-only vector search completed" in result.output
    assert "note.md#chunk-0" in result.output
    assert "score=" in result.output
    assert secret not in result.output
    assert "PRIVATE QUERY" not in result.output
    assert str(database) not in result.output
    assert "No retrieved text was printed" in result.output
    assert SearchClient.requested_models == ["embeddinggemma"]
    assert (database.read_bytes(), database.stat().st_mtime_ns) == before


def test_search_show_text_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = "A synthetic passage safe to display."
    database = _build_search_test_index(tmp_path, monkeypatch, content)
    monkeypatch.setattr(cli, "OllamaClient", SearchClient)

    result = runner.invoke(
        cli.app,
        ["search", "question", "--db", str(database), "--show-text"],
    )

    assert result.exit_code == 0
    assert "WARNING: Exact retrieved text follows" in result.output
    assert content in result.output


def test_search_json_is_metadata_only_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = "JSON-PRIVATE-PASSAGE"
    database = _build_search_test_index(tmp_path, monkeypatch, content)
    monkeypatch.setattr(cli, "OllamaClient", SearchClient)

    result = runner.invoke(
        cli.app,
        ["search", "private query", "--db", str(database), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["content_included"] is False
    assert payload["result_count"] == 1
    assert "text" not in payload["results"][0]
    assert "query" not in payload
    assert content not in result.stdout


def test_search_zero_matches_is_successful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = _build_search_test_index(tmp_path, monkeypatch, "unrelated")
    monkeypatch.setattr(cli, "OllamaClient", NoMatchSearchClient)

    result = runner.invoke(
        cli.app,
        ["search", "question", "--db", str(database), "--min-score", "0.3"],
    )

    assert result.exit_code == 0
    assert "No chunks met min_score=0.3" in result.output


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["question", "--top-k", "0"], "Top-k"),
        (["question", "--top-k", "101"], "Top-k"),
        (["question", "--min-score", "1.1"], "Minimum score"),
        (["   "], "must not be empty"),
    ],
)
def test_search_rejects_invalid_options(tmp_path: Path, arguments: list[str], message: str) -> None:
    result = runner.invoke(
        cli.app,
        ["search", *arguments, "--db", str(tmp_path / "index.sqlite")],
    )

    assert result.exit_code == 2
    assert "Invalid search options" in result.output
    assert message in result.output


class AnswerClient(SearchClient):
    chat_calls: ClassVar[list[list[ChatMessage]]] = []

    def chat_structured(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        schema: Mapping[str, object],
    ) -> StructuredChatResult:
        assert model == "qwen3.5:4b"
        assert schema["type"] == "object"
        self.chat_calls.append(list(messages))
        return StructuredChatResult(
            model=model,
            content=json.dumps(
                {
                    "answer": "The synthetic target is ten minutes.",
                    "citation_ids": [1],
                    "insufficient_evidence": False,
                }
            ),
            total_duration_ms=5.0,
        )


class NoMatchAnswerClient(NoMatchSearchClient):
    def chat_structured(
        self,
        _model: str,
        _messages: Sequence[ChatMessage],
        _schema: Mapping[str, object],
    ) -> StructuredChatResult:
        raise AssertionError("Qwen must not be called when retrieval has no results")


class MalformedAnswerClient(SearchClient):
    calls: ClassVar[int] = 0

    def chat_structured(
        self,
        model: str,
        _messages: Sequence[ChatMessage],
        _schema: Mapping[str, object],
    ) -> StructuredChatResult:
        type(self).calls += 1
        return StructuredChatResult(model=model, content="SENSITIVE MALFORMED OUTPUT")


def test_ask_generates_trusted_answer_without_printing_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = "Critical alerts have a synthetic ten-minute response target."
    database = _build_search_test_index(tmp_path, monkeypatch, context)
    before = (database.read_bytes(), database.stat().st_mtime_ns)
    AnswerClient.chat_calls.clear()
    monkeypatch.setattr(cli, "OllamaClient", AnswerClient)

    result = runner.invoke(cli.app, ["ask", "What is the response target?", "--db", str(database)])

    assert result.exit_code == 0
    assert "Grounded answer generated" in result.output
    assert "The synthetic target is ten minutes" in result.output
    assert "note.md#chunk-0" in result.output
    assert context not in result.output
    assert str(database) not in result.output
    assert "Retrieved context was not printed" in result.output
    assert len(AnswerClient.chat_calls) == 1
    assert context in AnswerClient.chat_calls[0][1].content
    assert "note.md" not in AnswerClient.chat_calls[0][1].content
    assert (database.read_bytes(), database.stat().st_mtime_ns) == before


def test_ask_show_context_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = "Synthetic context safe to reveal."
    database = _build_search_test_index(tmp_path, monkeypatch, context)
    monkeypatch.setattr(cli, "OllamaClient", AnswerClient)

    result = runner.invoke(
        cli.app,
        ["ask", "question", "--db", str(database), "--show-context"],
    )

    assert result.exit_code == 0
    assert "WARNING: Exact context sent to Qwen follows" in result.output
    assert context in result.output


def test_ask_json_omits_context_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = "JSON PRIVATE CONTEXT"
    database = _build_search_test_index(tmp_path, monkeypatch, context)
    monkeypatch.setattr(cli, "OllamaClient", AnswerClient)

    result = runner.invoke(
        cli.app,
        ["ask", "question", "--db", str(database), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "answered"
    assert payload["context_included"] is False
    assert "context" not in payload
    assert payload["citations"][0]["relative_path"] == "note.md"
    assert context not in result.output


def test_ask_zero_matches_is_successful_refusal_without_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _build_search_test_index(tmp_path, monkeypatch, "unrelated")
    monkeypatch.setattr(cli, "OllamaClient", NoMatchAnswerClient)

    result = runner.invoke(
        cli.app,
        [
            "--log-level",
            "info",
            "ask",
            "unsupported",
            "--db",
            str(database),
            "--min-score",
            "0.3",
        ],
    )

    assert result.exit_code == 0
    assert "Answer refused safely" in result.output
    assert "Sources: none" in result.output
    assert "generation_attempts=0" in result.output
    event = json.loads(result.stderr.splitlines()[-1])
    assert event["outcome"] == "refused"
    assert event["fields"]["citation_count"] == 0


def test_ask_malformed_output_fails_without_leaking_raw_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _build_search_test_index(tmp_path, monkeypatch, "relevant")
    MalformedAnswerClient.calls = 0
    monkeypatch.setattr(cli, "OllamaClient", MalformedAnswerClient)

    result = runner.invoke(cli.app, ["ask", "question", "--db", str(database)])

    assert result.exit_code == 1
    assert "invalid structured output after one retry" in result.output
    assert "SENSITIVE MALFORMED OUTPUT" not in result.output
    assert MalformedAnswerClient.calls == 2


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["question", "--top-k", "0"], "Top-k"),
        (["question", "--min-score", "1.1"], "Minimum score"),
        (["   "], "must not be empty"),
    ],
)
def test_ask_rejects_invalid_options(tmp_path: Path, arguments: list[str], message: str) -> None:
    result = runner.invoke(
        cli.app,
        ["ask", *arguments, "--db", str(tmp_path / "index.sqlite")],
    )

    assert result.exit_code == 2
    assert "Invalid answer options" in result.output
    assert message in result.output
