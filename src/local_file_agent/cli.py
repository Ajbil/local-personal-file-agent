"""Command-line interface for the Local Personal File Agent."""

from __future__ import annotations

from typing import Annotated

import typer
from pydantic import ValidationError

from local_file_agent.config import Settings
from local_file_agent.doctor import DoctorReport, run_doctor
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
