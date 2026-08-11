"""Tests for repeatable RAG quality gates and privacy-safe diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import local_file_agent.cli as cli
from local_file_agent.evaluation import (
    EvaluationInputError,
    EvaluationManifest,
    EvaluationMode,
    load_evaluation_manifest,
    run_evaluation,
)

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY / "examples" / "checkpoint-7" / "manifest.json"
runner = CliRunner()


def test_deterministic_suite_passes_all_required_gates(tmp_path: Path) -> None:
    report = run_evaluation(MANIFEST, work_root=tmp_path)

    assert report.status == "passed"
    assert report.mode is EvaluationMode.DETERMINISTIC
    assert report.case_count == 7
    assert report.failed_cases == 0
    assert report.metrics.hit_at_k == 1.0
    assert report.metrics.mean_reciprocal_rank == 1.0
    assert report.metrics.answer_fact_accuracy == 1.0
    assert report.metrics.citation_validity == 1.0
    assert report.metrics.citation_precision == 1.0
    assert report.metrics.refusal_accuracy == 1.0
    assert report.metrics.security_leakage_count == 0
    assert all(case.passed for case in report.cases)
    assert list(tmp_path.iterdir()) == []


def test_report_omits_questions_answers_passages_vectors_and_canaries(tmp_path: Path) -> None:
    manifest, _, _ = load_evaluation_manifest(MANIFEST)
    report = run_evaluation(MANIFEST, work_root=tmp_path)
    rendered = report.model_dump_json()

    sensitive_values = [
        *(case.question for case in manifest.cases),
        *(case.scripted_answer for case in manifest.cases if case.scripted_answer),
        *(value for case in manifest.cases for value in case.forbidden_output_substrings),
        "Reliability Engineering owns Project Atlas",
        "silver lighthouse",
    ]
    assert all(value not in rendered for value in sensitive_values)
    assert "vector" not in rendered.casefold()


def test_repeated_deterministic_runs_have_same_quality_results(tmp_path: Path) -> None:
    first = run_evaluation(MANIFEST, work_root=tmp_path)
    second = run_evaluation(MANIFEST, work_root=tmp_path)

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.corpus_fingerprint == second.corpus_fingerprint
    assert first.metrics == second.metrics
    assert [
        case.model_dump(exclude={"retrieval_duration_ms", "generation_duration_ms"})
        for case in first.cases
    ] == [
        case.model_dump(exclude={"retrieval_duration_ms", "generation_duration_ms"})
        for case in second.cases
    ]


def test_manifest_rejects_duplicate_cases() -> None:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["cases"].append(raw["cases"][0])

    with pytest.raises(ValidationError, match="case IDs must be unique"):
        EvaluationManifest.model_validate(raw)


def test_manifest_source_cannot_escape_its_directory(tmp_path: Path) -> None:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["source"] = "../outside"
    bad_manifest = tmp_path / "manifest.json"
    bad_manifest.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(EvaluationInputError, match="parent traversal"):
        load_evaluation_manifest(bad_manifest)


def test_cli_returns_one_when_quality_gate_fails() -> None:
    result = runner.invoke(
        cli.app,
        ["evaluate", "--manifest", str(MANIFEST), "--min-score", "1", "--json"],
    )

    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["status"] == "failed"
    assert report["failed_cases"] > 0


def test_cli_returns_two_for_invalid_manifest(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = runner.invoke(cli.app, ["evaluate", "--manifest", str(missing)])

    assert result.exit_code == 2
    assert "Invalid evaluation options" in result.output
    assert str(missing) not in result.output


@pytest.mark.parametrize("arguments", [["--top-k", "0"], ["--chunk-size", "0"]])
def test_cli_returns_two_for_zero_numeric_overrides(arguments: list[str]) -> None:
    result = runner.invoke(
        cli.app,
        ["evaluate", "--manifest", str(MANIFEST), *arguments],
    )

    assert result.exit_code == 2
    assert "Invalid evaluation options" in result.output
