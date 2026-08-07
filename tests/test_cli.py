"""Tests for stable command-line behavior and exit codes."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import local_file_agent.cli as cli
from local_file_agent.config import Settings
from local_file_agent.doctor import CheckResult, CheckStatus, DoctorReport

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
    assert "scan" in result.stdout


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
