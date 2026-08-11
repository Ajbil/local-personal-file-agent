"""Tests for opt-in structured observability and its privacy boundary."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

import local_file_agent.cli as cli
from local_file_agent.observability import (
    LogLevel,
    SafeLogFields,
    configure_observability,
    observed_command,
    record_observation,
)
from local_file_agent.ollama import OllamaConnectionError

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY / "examples" / "checkpoint-7" / "manifest.json"
runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_observability() -> Iterator[None]:
    configure_observability(LogLevel.OFF)
    yield
    configure_observability(LogLevel.OFF)


def _deterministic_runtime(stream: io.StringIO, *times: float) -> None:
    moments = iter(times)
    configure_observability(
        LogLevel.INFO,
        stream=stream,
        utc_now=lambda: datetime(2026, 8, 11, 8, 0, tzinfo=UTC),
        operation_id=lambda: "a" * 32,
        monotonic=lambda: next(moments),
    )


def test_info_events_are_versioned_correlated_and_deterministic() -> None:
    stream = io.StringIO()
    _deterministic_runtime(stream, 10.0, 10.025)

    @observed_command("scan")
    def command() -> str:
        record_observation(SafeLogFields(accepted_count=2, skipped_count=1))
        return "ok"

    assert command() == "ok"
    events = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert [event["event"] for event in events] == [
        "command.started",
        "command.completed",
    ]
    assert {event["operation_id"] for event in events} == {"a" * 32}
    assert events[1]["duration_ms"] == 25.0
    assert events[1]["fields"] == {"accepted_count": 2, "skipped_count": 1}
    assert all(event["schema_version"] == 1 for event in events)
    assert all(event["timestamp_utc"] == "2026-08-11T08:00:00.000Z" for event in events)


def test_off_emits_nothing() -> None:
    stream = io.StringIO()
    configure_observability(LogLevel.OFF, stream=stream)

    @observed_command("scan")
    def command() -> None:
        record_observation(SafeLogFields(accepted_count=1))

    command()

    assert stream.getvalue() == ""


def test_error_level_omits_info_and_records_safe_failure_category() -> None:
    stream = io.StringIO()
    moments = iter([3.0, 3.1])
    configure_observability(
        LogLevel.ERROR,
        stream=stream,
        operation_id=lambda: "b" * 32,
        monotonic=lambda: next(moments),
    )

    @observed_command("search")
    def command() -> None:
        raise typer.Exit(code=2)

    with pytest.raises(typer.Exit):
        command()

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "command.failed"
    assert events[0]["failure_category"] == "invalid_input"


def test_safe_field_schema_rejects_content_and_paths() -> None:
    for forbidden in ("question", "answer", "text", "path", "vector", "prompt", "canary"):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SafeLogFields.model_validate({forbidden: "PRIVATE-SENTINEL-8"})


def test_broken_log_sink_cannot_change_command_result() -> None:
    class BrokenStream(io.StringIO):
        def write(self, value: str) -> int:
            raise OSError("PRIVATE-SENTINEL-8")

    configure_observability(LogLevel.INFO, stream=BrokenStream())

    @observed_command("scan")
    def command() -> int:
        return 42

    assert command() == 42


def test_broken_logging_clock_and_id_cannot_change_command_result() -> None:
    def broken() -> float:
        raise RuntimeError("PRIVATE-SENTINEL-8")

    def broken_id() -> str:
        raise RuntimeError("PRIVATE-SENTINEL-8")

    def broken_now() -> datetime:
        raise RuntimeError("PRIVATE-SENTINEL-8")

    configure_observability(
        LogLevel.INFO,
        stream=io.StringIO(),
        operation_id=broken_id,
        monotonic=broken,
        utc_now=broken_now,
    )

    @observed_command("scan")
    def command() -> int:
        return 42

    assert command() == 42


def test_subclassed_operational_error_keeps_its_safe_category() -> None:
    stream = io.StringIO()
    configure_observability(LogLevel.ERROR, stream=stream)

    @observed_command("doctor")
    def command() -> None:
        try:
            raise OllamaConnectionError("PRIVATE-SENTINEL-8")
        except OllamaConnectionError as exc:
            raise typer.Exit(code=1) from exc

    with pytest.raises(typer.Exit):
        command()

    event = json.loads(stream.getvalue())
    assert event["failure_category"] == "ollama"
    assert "PRIVATE-SENTINEL-8" not in stream.getvalue()


def test_cli_logging_keeps_machine_json_on_stdout() -> None:
    result = runner.invoke(
        cli.app,
        [
            "--log-level",
            "info",
            "evaluate",
            "--manifest",
            str(MANIFEST),
            "--mode",
            "deterministic",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "passed"
    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert [event["event"] for event in events] == [
        "command.started",
        "command.completed",
    ]
    assert events[1]["outcome"] == "evaluation_passed"
    assert events[1]["fields"]["security_leakage_count"] == 0


def test_cli_logging_is_disabled_by_default() -> None:
    result = runner.invoke(
        cli.app,
        ["evaluate", "--manifest", str(MANIFEST), "--mode", "deterministic", "--json"],
    )

    assert result.exit_code == 0
    assert result.stderr == ""


def test_cli_rejects_unknown_log_level() -> None:
    result = runner.invoke(cli.app, ["--log-level", "debug", "evaluate"])

    assert result.exit_code == 2
    assert "Invalid value" in result.output


def test_cli_distinguishes_quality_failure_from_invalid_input() -> None:
    quality = runner.invoke(
        cli.app,
        [
            "--log-level",
            "error",
            "evaluate",
            "--manifest",
            str(MANIFEST),
            "--min-score",
            "1",
        ],
    )
    invalid = runner.invoke(
        cli.app,
        ["--log-level", "error", "evaluate", "--manifest", str(MANIFEST), "--top-k", "0"],
    )

    assert quality.exit_code == 1
    assert invalid.exit_code == 2
    assert json.loads(quality.stderr.splitlines()[-1])["failure_category"] == "quality_gate"
    assert json.loads(invalid.stderr.splitlines()[-1])["failure_category"] == "invalid_input"


def test_question_and_database_path_do_not_enter_structured_events(tmp_path: Path) -> None:
    sentinel = "PRIVATE-SENTINEL-8"
    result = runner.invoke(
        cli.app,
        [
            "--log-level",
            "error",
            "ask",
            sentinel,
            "--db",
            str(tmp_path / f"{sentinel}.sqlite"),
        ],
    )

    assert result.exit_code == 1
    structured_lines = [line for line in result.stderr.splitlines() if line.startswith("{")]
    assert structured_lines
    assert sentinel not in "\n".join(structured_lines)


def test_every_cli_command_uses_the_observability_wrapper() -> None:
    functions = (
        cli.doctor,
        cli.scan,
        cli.inspect_chunks,
        cli.inspect_embeddings,
        cli.index_documents,
        cli.inspect_index,
        cli.search,
        cli.evaluate,
        cli.ask,
    )

    assert all(hasattr(function, "__wrapped__") for function in functions)
