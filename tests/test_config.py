"""Tests for secure configuration defaults and overrides."""

import pytest
from pydantic import ValidationError

from local_file_agent.config import Settings


def test_settings_use_local_baseline_defaults() -> None:
    settings = Settings()

    assert settings.ollama_api_url == "http://127.0.0.1:11434"
    assert settings.answer_model == "qwen3.5:4b"
    assert settings.embedding_model == "embeddinggemma"


def test_settings_accept_loopback_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FILE_AGENT_OLLAMA_BASE_URL", "http://localhost:12345")

    settings = Settings()

    assert settings.ollama_api_url == "http://localhost:12345"


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.10:11434",
        "https://example.com:11434",
        "https://127.0.0.1:11434",
        "http://user:password@127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?token=value",
    ],
)
def test_settings_reject_unsafe_ollama_urls(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    monkeypatch.setenv("FILE_AGENT_OLLAMA_BASE_URL", url)

    with pytest.raises(ValidationError):
        Settings()


def test_settings_strip_model_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FILE_AGENT_ANSWER_MODEL", "  qwen3.5:4b  ")

    assert Settings().answer_model == "qwen3.5:4b"
