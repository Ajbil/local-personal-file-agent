"""Tests for stable command-line behavior and exit codes."""

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
