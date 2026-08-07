"""Checkpoint 0 diagnostics for the local runtime."""

from __future__ import annotations

import platform
from enum import StrEnum

from pydantic import BaseModel, Field

from local_file_agent.config import Settings
from local_file_agent.ollama import OllamaError, OllamaGateway


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


DetailValue = str | int | float | bool | None


class CheckResult(BaseModel):
    name: str
    status: CheckStatus
    summary: str
    details: dict[str, DetailValue] = Field(default_factory=dict)


class DoctorReport(BaseModel):
    success: bool
    full_readiness: bool
    checks: list[CheckResult]


def _result(
    name: str,
    status: CheckStatus,
    summary: str,
    **details: DetailValue,
) -> CheckResult:
    return CheckResult(name=name, status=status, summary=summary, details=details)


def _finalize(checks: list[CheckResult], *, full_readiness: bool) -> DoctorReport:
    success = all(check.status is not CheckStatus.FAIL for check in checks)
    return DoctorReport(success=success, full_readiness=success and full_readiness, checks=checks)


def _model_is_installed(requested_model: str, installed_models: set[str]) -> bool:
    """Treat Ollama's implicit ``:latest`` tag as equivalent to an untagged request."""

    return requested_model in installed_models or (
        ":" not in requested_model and f"{requested_model}:latest" in installed_models
    )


def run_doctor(
    settings: Settings,
    gateway: OllamaGateway,
    *,
    skip_generation: bool = False,
) -> DoctorReport:
    """Run diagnostics in dependency order and return a safe structured report."""

    checks: list[CheckResult] = []

    python_version = platform.python_version()
    python_supported = platform.python_version_tuple()[:2] == ("3", "12")
    checks.append(
        _result(
            "python",
            CheckStatus.PASS if python_supported else CheckStatus.FAIL,
            "Supported Python runtime" if python_supported else "Python 3.12 is required",
            version=python_version,
            implementation=platform.python_implementation(),
        )
    )
    checks.append(
        _result(
            "configuration",
            CheckStatus.PASS,
            "Ollama endpoint is restricted to loopback",
            base_url=settings.ollama_api_url,
            answer_model=settings.answer_model,
            embedding_model=settings.embedding_model,
        )
    )

    try:
        version = gateway.version()
    except OllamaError as exc:
        checks.append(_result("ollama_api", CheckStatus.FAIL, str(exc)))
        return _finalize(checks, full_readiness=False)
    checks.append(
        _result("ollama_api", CheckStatus.PASS, "Local Ollama API is reachable", version=version)
    )

    try:
        installed_models = gateway.model_names()
    except OllamaError as exc:
        checks.append(_result("model_inventory", CheckStatus.FAIL, str(exc)))
        return _finalize(checks, full_readiness=False)

    required_models = {settings.answer_model, settings.embedding_model}
    missing_models = sorted(
        model for model in required_models if not _model_is_installed(model, installed_models)
    )
    if missing_models:
        checks.append(
            _result(
                "model_inventory",
                CheckStatus.FAIL,
                "Required Ollama models are missing",
                missing=", ".join(missing_models),
                remediation="Run ollama pull for each missing model",
            )
        )
    else:
        checks.append(
            _result(
                "model_inventory",
                CheckStatus.PASS,
                "Required Ollama models are installed",
                installed_count=len(installed_models),
            )
        )

    if _model_is_installed(settings.embedding_model, installed_models):
        try:
            embedding_probe = gateway.embedding_probe(settings.embedding_model)
            checks.append(
                _result(
                    "embedding_smoke",
                    CheckStatus.PASS,
                    "Embedding model returned a valid vector",
                    dimension=embedding_probe.dimension,
                    total_duration_ms=embedding_probe.total_duration_ms,
                    load_duration_ms=embedding_probe.load_duration_ms,
                )
            )
        except OllamaError as exc:
            checks.append(_result("embedding_smoke", CheckStatus.FAIL, str(exc)))
    else:
        checks.append(
            _result(
                "embedding_smoke",
                CheckStatus.FAIL,
                "Embedding smoke test cannot run because the model is missing",
            )
        )

    generation_completed = False
    if skip_generation:
        checks.append(
            _result(
                "generation_smoke",
                CheckStatus.WARN,
                "Generation smoke test was skipped; full readiness is not proven",
            )
        )
    elif _model_is_installed(settings.answer_model, installed_models):
        try:
            generation_probe = gateway.generation_probe(settings.answer_model)
            generation_completed = True
            checks.append(
                _result(
                    "generation_smoke",
                    CheckStatus.PASS,
                    "Answer model returned schema-valid JSON",
                    total_duration_ms=generation_probe.total_duration_ms,
                    load_duration_ms=generation_probe.load_duration_ms,
                )
            )
        except OllamaError as exc:
            checks.append(_result("generation_smoke", CheckStatus.FAIL, str(exc)))
    else:
        checks.append(
            _result(
                "generation_smoke",
                CheckStatus.FAIL,
                "Generation smoke test cannot run because the model is missing",
            )
        )

    try:
        runtime = gateway.runtime_probe()
        if runtime.reported_vram_bytes > 0:
            checks.append(
                _result(
                    "runtime",
                    CheckStatus.PASS,
                    "Ollama reports model data allocated in VRAM",
                    loaded_models=", ".join(runtime.loaded_models),
                    reported_vram_bytes=runtime.reported_vram_bytes,
                )
            )
        else:
            checks.append(
                _result(
                    "runtime",
                    CheckStatus.WARN,
                    "Ollama reports no VRAM allocation; CPU execution remains supported",
                    loaded_models=", ".join(runtime.loaded_models),
                    reported_vram_bytes=0,
                )
            )
    except OllamaError as exc:
        checks.append(_result("runtime", CheckStatus.WARN, f"Runtime details unavailable: {exc}"))

    return _finalize(checks, full_readiness=generation_completed)
