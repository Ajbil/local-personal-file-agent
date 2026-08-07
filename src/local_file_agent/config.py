"""Typed application configuration with secure local defaults."""

from ipaddress import ip_address

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from ``FILE_AGENT_`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="FILE_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: AnyHttpUrl = "http://127.0.0.1:11434"  # type: ignore[assignment]
    answer_model: str = "qwen3.5:4b"
    embedding_model: str = "embeddinggemma"
    connect_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    model_timeout_seconds: float = Field(default=300.0, gt=0, le=1800)

    @field_validator("ollama_base_url")
    @classmethod
    def require_loopback_ollama(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        """Prevent accidental exposure of personal-document traffic to a remote host."""

        if value.scheme != "http":
            raise ValueError("Ollama URL must use HTTP on the local loopback interface")
        if value.username is not None or value.password is not None:
            raise ValueError("Ollama URL must not contain credentials")
        if value.query is not None or value.fragment is not None:
            raise ValueError("Ollama URL must not contain a query or fragment")
        if value.path not in {None, "/"}:
            raise ValueError("Ollama URL must not contain a path")

        if value.host is None:
            raise ValueError("Ollama URL must contain a host")
        host = value.host.lower()
        if host == "localhost":
            return value
        try:
            is_loopback = ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError("Ollama URL must use localhost or a loopback IP address")
        return value

    @field_validator("answer_model", "embedding_model")
    @classmethod
    def require_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model name must not be empty")
        return normalized

    @property
    def ollama_api_url(self) -> str:
        """Return the normalized base URL without a trailing slash."""

        return str(self.ollama_base_url).rstrip("/")
