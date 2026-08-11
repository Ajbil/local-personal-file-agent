"""Opt-in, privacy-safe JSON Lines observability for CLI orchestration."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import wraps
from time import perf_counter
from typing import IO, ParamSpec, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

LOG_SCHEMA_VERSION = 1

P = ParamSpec("P")
R = TypeVar("R")


class LogLevel(StrEnum):
    """Deliberately small logging policy for a personal-document application."""

    OFF = "off"
    ERROR = "error"
    INFO = "info"


class SafeLogFields(BaseModel):
    """The complete allowlist of values permitted in structured logs."""

    model_config = ConfigDict(extra="forbid", strict=True)

    endpoint_scope: str | None = Field(default=None, pattern=r"^loopback$")
    mode: str | None = None
    embedding_model: str | None = None
    answer_model: str | None = None
    decision: str | None = None
    full_readiness: bool | None = None
    document_count: int | None = Field(default=None, ge=0)
    accepted_count: int | None = Field(default=None, ge=0)
    skipped_count: int | None = Field(default=None, ge=0)
    chunk_count: int | None = Field(default=None, ge=0)
    embedding_count: int | None = Field(default=None, ge=0)
    batch_count: int | None = Field(default=None, ge=0)
    result_count: int | None = Field(default=None, ge=0)
    citation_count: int | None = Field(default=None, ge=0)
    case_count: int | None = Field(default=None, ge=0)
    passed_count: int | None = Field(default=None, ge=0)
    failed_count: int | None = Field(default=None, ge=0)
    security_leakage_count: int | None = Field(default=None, ge=0)
    dimension: int | None = Field(default=None, gt=0)
    chunk_size: int | None = Field(default=None, gt=0)
    overlap: int | None = Field(default=None, ge=0)
    batch_size: int | None = Field(default=None, gt=0)
    top_k: int | None = Field(default=None, gt=0)
    min_score: float | None = Field(default=None, ge=-1, le=1)
    top_score: float | None = Field(default=None, ge=-1, le=1)
    lowest_selected_score: float | None = Field(default=None, ge=-1, le=1)


class SafeLogEvent(BaseModel):
    """One versioned event whose schema excludes document and model content."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = Field(default=LOG_SCHEMA_VERSION, ge=1)
    timestamp_utc: str
    level: str = Field(pattern=r"^(INFO|ERROR)$")
    event: str = Field(pattern=r"^command\.(started|completed|failed)$")
    command: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")
    operation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    outcome: str = Field(min_length=1, max_length=64)
    duration_ms: float | None = Field(default=None, ge=0)
    failure_category: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{1,63}$")
    fields: SafeLogFields = Field(default_factory=SafeLogFields)


@dataclass(slots=True)
class _ObservabilityRuntime:
    level: LogLevel = LogLevel.OFF
    stream: IO[str] | None = None
    utc_now: Callable[[], datetime] = lambda: datetime.now(UTC)
    operation_id: Callable[[], str] = lambda: uuid4().hex
    monotonic: Callable[[], float] = perf_counter


_runtime = _ObservabilityRuntime()
_current: ContextVar[CommandObservation | None] = ContextVar(
    "local_file_agent_command_observation", default=None
)


def configure_observability(
    level: LogLevel,
    *,
    stream: IO[str] | None = None,
    utc_now: Callable[[], datetime] | None = None,
    operation_id: Callable[[], str] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> None:
    """Configure one CLI invocation; injectable boundaries keep tests deterministic."""

    _runtime.level = level
    _runtime.stream = stream
    _runtime.utc_now = utc_now or (lambda: datetime.now(UTC))
    _runtime.operation_id = operation_id or (lambda: uuid4().hex)
    _runtime.monotonic = monotonic or perf_counter


def _enabled(level: str) -> bool:
    if _runtime.level is LogLevel.OFF:
        return False
    return _runtime.level is LogLevel.INFO or level == "ERROR"


def _emit(event: SafeLogEvent) -> None:
    if not _enabled(event.level):
        return
    try:
        target = _runtime.stream or sys.stderr
        target.write(event.model_dump_json(exclude_none=True) + "\n")
        target.flush()
    except Exception:
        # Observability must never change application output or behavior.
        return


def _timestamp() -> str:
    try:
        value = _runtime.utc_now().astimezone(UTC)
    except Exception:
        value = datetime.now(UTC)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class CommandObservation:
    """State shared between a command wrapper and explicit safe metric recording."""

    def __init__(self, command: str) -> None:
        self.command = command
        try:
            self.operation_id = _runtime.operation_id()
        except Exception:
            self.operation_id = uuid4().hex
        try:
            self.started = _runtime.monotonic()
        except Exception:
            self.started = perf_counter()
        self.fields = SafeLogFields()
        self.outcome = "completed"
        self.failure_category: str | None = None

    def emit_started(self) -> None:
        try:
            event = SafeLogEvent(
                timestamp_utc=_timestamp(),
                level="INFO",
                event="command.started",
                command=self.command,
                operation_id=self.operation_id,
                outcome="started",
            )
        except Exception:
            return
        _emit(event)

    def update(self, fields: SafeLogFields, *, outcome: str | None = None) -> None:
        current = self.fields.model_dump(exclude_none=True)
        current.update(fields.model_dump(exclude_none=True))
        self.fields = SafeLogFields.model_validate(current)
        if outcome is not None:
            self.outcome = outcome

    def fail(self, category: str) -> None:
        self.failure_category = category

    def emit_completed(self) -> None:
        try:
            event = SafeLogEvent(
                timestamp_utc=_timestamp(),
                level="INFO",
                event="command.completed",
                command=self.command,
                operation_id=self.operation_id,
                outcome=self.outcome,
                duration_ms=self._duration(),
                fields=self.fields,
            )
        except Exception:
            return
        _emit(event)

    def emit_failed(self, category: str) -> None:
        try:
            event = SafeLogEvent(
                timestamp_utc=_timestamp(),
                level="ERROR",
                event="command.failed",
                command=self.command,
                operation_id=self.operation_id,
                outcome="failed",
                duration_ms=self._duration(),
                failure_category=category,
                fields=self.fields,
            )
        except Exception:
            return
        _emit(event)

    def _duration(self) -> float:
        try:
            finished = _runtime.monotonic()
        except Exception:
            finished = perf_counter()
        return round(max(0.0, (finished - self.started) * 1_000), 3)


def record_observation(fields: SafeLogFields, *, outcome: str | None = None) -> None:
    """Attach only schema-approved metrics to the active command event."""

    observation = _current.get()
    if observation is not None:
        observation.update(fields, outcome=outcome)


def mark_observation_failure(category: str) -> None:
    """Override the safe failure category before a command raises its exit signal."""

    observation = _current.get()
    if observation is not None:
        observation.fail(category)


_CAUSE_CATEGORIES = {
    "AnswerGenerationError": "answer_generation",
    "EmbeddingError": "embedding",
    "EvaluationError": "evaluation",
    "IndexBuildError": "index_build",
    "IndexStorageError": "index_storage",
    "OllamaError": "ollama",
    "RetrievalError": "retrieval",
    "SourceRootError": "source_root",
}


def _failure_category(exc: BaseException, observation: CommandObservation) -> str:
    if observation.failure_category is not None:
        return observation.failure_category
    exit_code = getattr(exc, "exit_code", None)
    if exit_code == 2:
        return "invalid_input"
    cause = exc.__cause__
    while cause is not None:
        for cause_type in type(cause).__mro__:
            category = _CAUSE_CATEGORIES.get(cause_type.__name__)
            if category is not None:
                return category
        cause = cause.__cause__
    if exit_code == 1:
        return "command_failure"
    return "unexpected_error"


def observed_command(command: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a Typer command without changing its inspected function signature."""

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            observation = CommandObservation(command)
            token: Token[CommandObservation | None] = _current.set(observation)
            observation.emit_started()
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                observation.emit_failed(_failure_category(exc, observation))
                raise
            else:
                observation.emit_completed()
                return result
            finally:
                _current.reset(token)

        return wrapper

    return decorator
