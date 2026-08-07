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
