"""Tests for diagnostic orchestration independent of a live model runtime."""

import pytest

from local_file_agent.config import Settings
from local_file_agent.doctor import CheckStatus, run_doctor
from local_file_agent.ollama import (
    EmbeddingProbe,
    GenerationProbe,
    OllamaConnectionError,
    RuntimeProbe,
)


class HealthyGateway:
    def __init__(self) -> None:
        self.generation_calls = 0

    def version(self) -> str:
        return "0.32.6"

    def model_names(self) -> set[str]:
        return {"embeddinggemma", "qwen3.5:4b"}

    def embedding_probe(self, model: str) -> EmbeddingProbe:
        assert model == "embeddinggemma"
        return EmbeddingProbe(dimension=768, total_duration_ms=15.0)

    def generation_probe(self, model: str) -> GenerationProbe:
        assert model == "qwen3.5:4b"
        self.generation_calls += 1
        return GenerationProbe(total_duration_ms=250.0)

    def runtime_probe(self) -> RuntimeProbe:
        return RuntimeProbe(loaded_models=["qwen3.5:4b"], reported_vram_bytes=0)


class OfflineGateway(HealthyGateway):
    def version(self) -> str:
        raise OllamaConnectionError("Could not connect to local Ollama")


class MissingModelsGateway(HealthyGateway):
    def model_names(self) -> set[str]:
        return set()


class InventoryErrorGateway(HealthyGateway):
    def model_names(self) -> set[str]:
        raise OllamaConnectionError("Could not load model inventory")


class EmbeddingErrorGateway(HealthyGateway):
    def embedding_probe(self, model: str) -> EmbeddingProbe:
        raise OllamaConnectionError("Embedding probe failed")


class GenerationErrorGateway(HealthyGateway):
    def generation_probe(self, model: str) -> GenerationProbe:
        raise OllamaConnectionError("Generation probe failed")


class VramGateway(HealthyGateway):
    def runtime_probe(self) -> RuntimeProbe:
        return RuntimeProbe(loaded_models=["qwen3.5:4b"], reported_vram_bytes=2048)


class RuntimeErrorGateway(HealthyGateway):
    def runtime_probe(self) -> RuntimeProbe:
        raise OllamaConnectionError("Runtime details failed")


def test_healthy_cpu_runtime_is_ready_with_warning() -> None:
    report = run_doctor(Settings(), HealthyGateway())

    assert report.success is True
    assert report.full_readiness is True
    assert any(check.status is CheckStatus.WARN for check in report.checks)
    assert not any(check.status is CheckStatus.FAIL for check in report.checks)


def test_offline_runtime_fails_without_running_dependent_checks() -> None:
    report = run_doctor(Settings(), OfflineGateway())

    assert report.success is False
    assert report.full_readiness is False
    assert [check.name for check in report.checks] == [
        "python",
        "configuration",
        "ollama_api",
    ]


def test_missing_models_produce_actionable_failures() -> None:
    report = run_doctor(Settings(), MissingModelsGateway())

    failures = {check.name: check for check in report.checks if check.status is CheckStatus.FAIL}
    assert report.success is False
    assert failures["model_inventory"].details["missing"] == "embeddinggemma, qwen3.5:4b"
    assert "embedding_smoke" in failures
    assert "generation_smoke" in failures


def test_skip_generation_does_not_claim_full_readiness() -> None:
    gateway = HealthyGateway()

    report = run_doctor(Settings(), gateway, skip_generation=True)

    assert report.success is True
    assert report.full_readiness is False
    assert gateway.generation_calls == 0
    generation = next(check for check in report.checks if check.name == "generation_smoke")
    assert generation.status is CheckStatus.WARN


def test_inventory_failure_stops_dependent_checks() -> None:
    report = run_doctor(Settings(), InventoryErrorGateway())

    assert report.success is False
    assert report.checks[-1].name == "model_inventory"


@pytest.mark.parametrize(
    ("gateway", "failed_check"),
    [
        (EmbeddingErrorGateway(), "embedding_smoke"),
        (GenerationErrorGateway(), "generation_smoke"),
    ],
)
def test_model_probe_failure_is_reported(
    gateway: HealthyGateway,
    failed_check: str,
) -> None:
    report = run_doctor(Settings(), gateway)

    failed = next(check for check in report.checks if check.name == failed_check)
    assert failed.status is CheckStatus.FAIL


def test_vram_allocation_is_reported_as_pass() -> None:
    report = run_doctor(Settings(), VramGateway())

    runtime = next(check for check in report.checks if check.name == "runtime")
    assert runtime.status is CheckStatus.PASS
    assert runtime.details["reported_vram_bytes"] == 2048


def test_runtime_detail_failure_is_only_a_warning() -> None:
    report = run_doctor(Settings(), RuntimeErrorGateway())

    runtime = next(check for check in report.checks if check.name == "runtime")
    assert report.success is True
    assert runtime.status is CheckStatus.WARN
