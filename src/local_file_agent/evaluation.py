"""Repeatable, privacy-aware quality and security evaluation for the local RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from time import perf_counter
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from local_file_agent.answering import AnswerGateway, answer_from_search
from local_file_agent.chunking import ChunkingOptions
from local_file_agent.indexing import build_index
from local_file_agent.ingestion import SourceRootError, scan_source
from local_file_agent.ollama import ChatMessage, EmbeddingBatch, StructuredChatResult
from local_file_agent.retrieval import SearchOptions, run_database_search

DETERMINISTIC_EMBEDDING_MODEL = "deterministic-hashed-lexical-v1"
DETERMINISTIC_ANSWER_MODEL = "deterministic-scripted-answer-v1"
DETERMINISTIC_DIMENSION = 2_048
DEFAULT_MANIFEST = Path("examples/checkpoint-7/manifest.json")

_TOKEN = re.compile(r"[a-z0-9]+")
_QUESTION = re.compile(r"QUESTION:\n(.*?)\n\nREQUIRED_OUTPUT_SCHEMA:", re.DOTALL)
_EVIDENCE = re.compile(r"BEGIN_UNTRUSTED_EVIDENCE\n(.*?)\nEND_UNTRUSTED_EVIDENCE", re.DOTALL)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "can",
    "does",
    "every",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "what",
    "when",
    "who",
    "with",
}
_ALIASES = {
    "annual": "year",
    "annually": "year",
    "calendar": "year",
    "yearly": "year",
    "budget": "allowance",
    "funding": "allowance",
    "credits": "allowance",
    "development": "learning",
    "professional": "learning",
    "training": "learning",
    "worker": "employee",
    "workers": "employee",
    "staff": "employee",
    "severe": "critical",
    "coordinates": "coordinate",
    "coordinating": "coordinate",
    "manages": "coordinate",
    "manage": "coordinate",
    "approves": "approve",
    "approved": "approve",
    "approval": "approve",
    "begins": "begin",
    "starts": "begin",
}


class EvaluationError(RuntimeError):
    """Evaluation input or execution failed without exposing corpus contents."""


class EvaluationInputError(EvaluationError):
    """The manifest or command options are invalid (CLI exit code 2)."""


class EvaluationMode(StrEnum):
    """The repeatability/cost trade-off selected for an evaluation run."""

    DETERMINISTIC = "deterministic"
    LIVE = "live"


class ManifestChunking(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_size: int = Field(default=1_200, gt=0)
    overlap: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def overlap_is_smaller(self) -> ManifestChunking:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class ManifestRetrieval(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    top_k: int = Field(default=5, ge=1, le=100)
    deterministic_min_score: float = Field(default=0.05, ge=-1, le=1)
    live_min_score: float = Field(default=0.30, ge=-1, le=1)
    max_overlap_ratio: float = Field(default=0.80, ge=0, le=1)


class EvaluationCase(BaseModel):
    """One synthetic expectation; answer text remains internal to the evaluator."""

    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    category: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=500)
    expected_refusal: bool
    expected_sources: list[str] = Field(default_factory=list, max_length=10)
    required_answer_facts: list[list[str]] = Field(default_factory=list, max_length=10)
    forbidden_output_substrings: list[str] = Field(default_factory=list, max_length=20)
    scripted_answer: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def expectation_is_coherent(self) -> EvaluationCase:
        if self.expected_refusal:
            if self.expected_sources or self.required_answer_facts or self.scripted_answer:
                raise ValueError("refusal cases cannot define sources, answer facts, or an answer")
        elif (
            not self.expected_sources or not self.required_answer_facts or not self.scripted_answer
        ):
            raise ValueError(
                "answerable cases require sources, answer facts, and a scripted answer"
            )
        if any(
            not alternatives or any(not value.strip() for value in alternatives)
            for alternatives in self.required_answer_facts
        ):
            raise ValueError(
                "each declared answer fact requires one or more non-empty alternatives"
            )
        if any(
            len(alternatives) > 10 or any(len(value) > 200 for value in alternatives)
            for alternatives in self.required_answer_facts
        ):
            raise ValueError("answer fact alternatives exceed the manifest safety limits")
        return self


class EvaluationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    suite_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    source: str = Field(min_length=1, max_length=200)
    chunking: ManifestChunking = Field(default_factory=ManifestChunking)
    retrieval: ManifestRetrieval = Field(default_factory=ManifestRetrieval)
    cases: list[EvaluationCase] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def case_ids_and_questions_are_unique(self) -> EvaluationManifest:
        ids = [case.case_id for case in self.cases]
        questions = [case.question.casefold().strip() for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("case IDs must be unique")
        if len(set(questions)) != len(questions):
            raise ValueError("case questions must be unique")
        return self


class EvaluationCaseResult(BaseModel):
    """Content-free diagnostics for one case, safe to keep in CI logs."""

    case_id: str
    category: str
    passed: bool
    failure_stage: Literal["retrieval", "generation", "citation", "refusal", "security"] | None
    retrieval_passed: bool | None
    answer_passed: bool | None
    citation_passed: bool | None
    refusal_passed: bool
    security_passed: bool
    first_relevant_rank: int | None = Field(default=None, gt=0)
    decision_reason: str
    retrieved_count: int = Field(ge=0)
    cited_count: int = Field(ge=0)
    retrieval_duration_ms: float = Field(ge=0)
    generation_duration_ms: float = Field(ge=0)


class EvaluationMetrics(BaseModel):
    supported_cases: int = Field(ge=0)
    refusal_cases: int = Field(ge=0)
    hit_at_k: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    answer_fact_accuracy: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    citation_precision: float = Field(ge=0, le=1)
    refusal_accuracy: float = Field(ge=0, le=1)
    security_leakage_count: int = Field(ge=0)


class EvaluationReport(BaseModel):
    """Aggregate evaluation result that deliberately excludes prompts and document contents."""

    status: Literal["passed", "failed"]
    mode: EvaluationMode
    suite_id: str
    manifest_sha256: str
    corpus_fingerprint: str
    embedding_model: str
    answer_model: str
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    min_score: float = Field(ge=-1, le=1)
    case_count: int = Field(gt=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    index_build_duration_ms: float = Field(ge=0)
    total_duration_ms: float = Field(ge=0)
    metrics: EvaluationMetrics
    cases: list[EvaluationCaseResult]


def _safe_relative_path(value: str, *, label: str) -> Path:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or ".." in windows.parts:
        raise EvaluationInputError(f"{label} must be a relative path without parent traversal.")
    return Path(*posix.parts)


def load_evaluation_manifest(path: Path) -> tuple[EvaluationManifest, Path, str]:
    """Load a strict manifest and confine its corpus to the manifest directory."""

    try:
        resolved = path.expanduser().resolve(strict=True)
        raw = resolved.read_bytes()
        manifest = EvaluationManifest.model_validate_json(raw)
    except (OSError, RuntimeError, ValidationError) as exc:
        raise EvaluationInputError(
            "Evaluation manifest is missing, unreadable, or invalid."
        ) from exc

    relative_source = _safe_relative_path(manifest.source, label="Manifest source")
    try:
        source = (resolved.parent / relative_source).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvaluationInputError("Evaluation source folder does not exist.") from exc
    if not source.is_dir() or not source.is_relative_to(resolved.parent):
        raise EvaluationInputError(
            "Evaluation source must be a folder inside the manifest directory."
        )
    for case in manifest.cases:
        for expected_source in case.expected_sources:
            _safe_relative_path(expected_source, label=f"Expected source for {case.case_id}")
    try:
        documents = {
            document.relative_path: document.text for document in scan_source(source).documents
        }
    except SourceRootError as exc:
        raise EvaluationInputError("Evaluation source folder failed safe ingestion.") from exc
    corpus_text = "\n".join(documents.values()).casefold()
    for case in manifest.cases:
        missing_sources = set(case.expected_sources) - documents.keys()
        if missing_sources:
            raise EvaluationInputError(
                f"Evaluation case {case.case_id} names a source that safe ingestion did not accept."
            )
        expected_text = "\n".join(documents[path] for path in case.expected_sources).casefold()
        if any(
            not any(alternative.casefold() in expected_text for alternative in alternatives)
            for alternatives in case.required_answer_facts
        ):
            raise EvaluationInputError(
                f"Evaluation case {case.case_id} declares a fact absent from its expected sources."
            )
        if any(value.casefold() not in corpus_text for value in case.forbidden_output_substrings):
            raise EvaluationInputError(
                f"Evaluation case {case.case_id} declares a security canary absent from the corpus."
            )
    return manifest, source, hashlib.sha256(raw).hexdigest()


def _canonical_tokens(text: str) -> list[str]:
    if "| text: " in text:
        text = text.split("| text: ", 1)[1]
    elif "| query: " in text:
        text = text.split("| query: ", 1)[1]
    tokens: list[str] = []
    for token in _TOKEN.findall(text.casefold()):
        if token in _STOPWORDS:
            continue
        tokens.append(_ALIASES.get(token, token))
    return tokens


def _hashed_vector(text: str) -> list[float]:
    vector = np.zeros(DETERMINISTIC_DIMENSION, dtype=np.float32)
    for token in _canonical_tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % DETERMINISTIC_DIMENSION
        vector[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm == 0:
        vector[0] = 1.0
    else:
        vector /= norm
    return vector.tolist()


class DeterministicEvaluationGateway:
    """Offline lexical embeddings plus scripted generation through production validators."""

    def __init__(self, cases: Sequence[EvaluationCase]) -> None:
        self._cases = {case.question.strip(): case for case in cases}

    def embed(
        self,
        model: str,
        inputs: Sequence[str],
        *,
        truncate: bool = False,
    ) -> EmbeddingBatch:
        del truncate
        return EmbeddingBatch(model=model, vectors=[_hashed_vector(value) for value in inputs])

    def chat_structured(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        schema: Mapping[str, object],
    ) -> StructuredChatResult:
        del schema
        user = next(
            (message.content for message in reversed(messages) if message.role == "user"), ""
        )
        question_match = _QUESTION.search(user)
        evidence_match = _EVIDENCE.search(user)
        if question_match is None or evidence_match is None:
            raise EvaluationError("Deterministic answer input did not match the production prompt.")
        case = self._cases.get(question_match.group(1).strip())
        if case is None:
            raise EvaluationError("Deterministic answer received an unknown evaluation case.")
        try:
            evidence = json.loads(evidence_match.group(1))
        except json.JSONDecodeError as exc:
            raise EvaluationError("Deterministic answer received invalid evidence JSON.") from exc

        citation_ids: list[int] = []
        if not case.expected_refusal:
            for item in evidence:
                content = str(item.get("content", ""))
                if all(
                    any(
                        alternative.casefold() in content.casefold() for alternative in alternatives
                    )
                    for alternatives in case.required_answer_facts
                ):
                    citation_ids = [int(item["id"])]
                    break
        insufficient = case.expected_refusal or not citation_ids
        payload = {
            "answer": "" if insufficient else case.scripted_answer,
            "citation_ids": [] if insufficient else citation_ids,
            "insufficient_evidence": insufficient,
        }
        return StructuredChatResult(model=model, content=json.dumps(payload, separators=(",", ":")))


def _contains_declared_facts(text: str, facts: Sequence[Sequence[str]]) -> bool:
    normalized = " ".join(text.casefold().split())
    return all(
        any(" ".join(alternative.casefold().split()) in normalized for alternative in alternatives)
        for alternatives in facts
    )


def _evaluate_case(
    case: EvaluationCase,
    database: Path,
    gateway: AnswerGateway,
    answer_model: str,
    options: SearchOptions,
    global_forbidden: Sequence[str],
) -> tuple[EvaluationCaseResult, int, int, int, int]:
    database_run = run_database_search(database, case.question, gateway, options=options)
    answer = answer_from_search(database_run, gateway, answer_model)
    ranked_paths = [result.chunk.relative_path for result in database_run.search.results]
    expected = set(case.expected_sources)
    first_rank = next(
        (rank for rank, path in enumerate(ranked_paths, start=1) if path in expected), None
    )
    retrieval_passed = None if case.expected_refusal else first_rank is not None
    answer_passed = (
        None
        if case.expected_refusal
        else answer.status == "answered"
        and _contains_declared_facts(answer.answer, case.required_answer_facts)
    )

    retrieved_by_citation = {
        (
            result.chunk.relative_path,
            result.chunk.chunk_index,
            result.chunk.start_char,
            result.chunk.end_char,
            result.chunk.content_sha256,
        )
        for result in database_run.search.results
    }
    citation_valid_count = sum(
        (
            citation.relative_path,
            citation.chunk_index,
            citation.start_char,
            citation.end_char,
            citation.content_sha256,
        )
        in retrieved_by_citation
        for citation in answer.citations
    )
    expected_citation_count = sum(
        citation.relative_path in expected for citation in answer.citations
    )
    citation_passed = (
        None
        if case.expected_refusal
        else bool(answer.citations)
        and citation_valid_count == len(answer.citations)
        and expected_citation_count == len(answer.citations)
    )
    refusal_passed = (answer.status == "refused") == case.expected_refusal
    observable = " ".join(
        [
            answer.answer,
            *(citation.citation for citation in answer.citations),
            *(citation.relative_path for citation in answer.citations),
        ]
    ).casefold()
    security_passed = not any(forbidden.casefold() in observable for forbidden in global_forbidden)

    passed = (
        refusal_passed
        and security_passed
        and (case.expected_refusal or bool(retrieval_passed and answer_passed and citation_passed))
    )
    failure_stage: str | None = None
    if not passed:
        if not security_passed:
            failure_stage = "security"
        elif not refusal_passed:
            failure_stage = "refusal"
        elif not retrieval_passed:
            failure_stage = "retrieval"
        elif not answer_passed:
            failure_stage = "generation"
        else:
            failure_stage = "citation"

    result = EvaluationCaseResult(
        case_id=case.case_id,
        category=case.category,
        passed=passed,
        failure_stage=failure_stage,  # type: ignore[arg-type]
        retrieval_passed=retrieval_passed,
        answer_passed=answer_passed,
        citation_passed=citation_passed,
        refusal_passed=refusal_passed,
        security_passed=security_passed,
        first_relevant_rank=first_rank,
        decision_reason=answer.decision_reason,
        retrieved_count=len(database_run.search.results),
        cited_count=len(answer.citations),
        retrieval_duration_ms=round(
            database_run.index_load_duration_ms
            + database_run.search.query_embedding.wall_duration_ms
            + database_run.search.retrieval_wall_duration_ms,
            3,
        ),
        generation_duration_ms=answer.generation_wall_duration_ms,
    )
    return (
        result,
        citation_valid_count,
        expected_citation_count,
        len(answer.citations),
        0 if security_passed else 1,
    )


def run_evaluation(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    mode: EvaluationMode = EvaluationMode.DETERMINISTIC,
    live_gateway: AnswerGateway | None = None,
    live_embedding_model: str | None = None,
    live_answer_model: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    work_root: Path = Path(".data/evaluation"),
) -> EvaluationReport:
    """Build a disposable index, run all cases, and enforce quality/security gates."""

    started = perf_counter()
    manifest, source, manifest_sha256 = load_evaluation_manifest(manifest_path)
    selected_chunk_size = manifest.chunking.chunk_size if chunk_size is None else chunk_size
    selected_overlap = manifest.chunking.overlap if overlap is None else overlap
    selected_top_k = manifest.retrieval.top_k if top_k is None else top_k
    selected_min_score = (
        min_score
        if min_score is not None
        else (
            manifest.retrieval.deterministic_min_score
            if mode is EvaluationMode.DETERMINISTIC
            else manifest.retrieval.live_min_score
        )
    )
    try:
        chunking = ChunkingOptions(selected_chunk_size, selected_overlap)
        options = SearchOptions(
            top_k=selected_top_k,
            min_score=selected_min_score,
            max_overlap_ratio=manifest.retrieval.max_overlap_ratio,
        )
    except ValueError as exc:
        raise EvaluationInputError("Evaluation overrides are invalid.") from exc

    if mode is EvaluationMode.DETERMINISTIC:
        gateway: AnswerGateway = DeterministicEvaluationGateway(manifest.cases)
        embedding_model = DETERMINISTIC_EMBEDDING_MODEL
        answer_model = DETERMINISTIC_ANSWER_MODEL
    else:
        if live_gateway is None or not live_embedding_model or not live_answer_model:
            raise EvaluationInputError(
                "Live evaluation requires the configured local Ollama gateway."
            )
        gateway = live_gateway
        embedding_model = live_embedding_model
        answer_model = live_answer_model

    try:
        work_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvaluationError("Evaluation work directory could not be created.") from exc

    case_results: list[EvaluationCaseResult] = []
    valid_citations = expected_citations = total_citations = leakage_count = 0
    global_forbidden = tuple(
        dict.fromkeys(
            forbidden for case in manifest.cases for forbidden in case.forbidden_output_substrings
        )
    )
    with tempfile.TemporaryDirectory(prefix="checkpoint-7-", dir=work_root) as temporary:
        database = Path(temporary) / "evaluation.sqlite3"
        index_started = perf_counter()
        index_report = build_index(
            source,
            database,
            gateway,
            embedding_model,
            chunking=chunking,
        )
        index_duration = round((perf_counter() - index_started) * 1_000, 3)
        for case in manifest.cases:
            result, valid, precise, cited, leaked = _evaluate_case(
                case, database, gateway, answer_model, options, global_forbidden
            )
            case_results.append(result)
            valid_citations += valid
            expected_citations += precise
            total_citations += cited
            leakage_count += leaked

    supported = [
        result
        for case, result in zip(manifest.cases, case_results, strict=True)
        if not case.expected_refusal
    ]
    refusal = [
        result
        for case, result in zip(manifest.cases, case_results, strict=True)
        if case.expected_refusal
    ]
    supported_count = len(supported)
    refusal_count = len(refusal)
    metric = EvaluationMetrics(
        supported_cases=supported_count,
        refusal_cases=refusal_count,
        hit_at_k=(sum(result.retrieval_passed is True for result in supported) / supported_count),
        mean_reciprocal_rank=(
            sum(
                1 / result.first_relevant_rank for result in supported if result.first_relevant_rank
            )
            / supported_count
        ),
        answer_fact_accuracy=(
            sum(result.answer_passed is True for result in supported) / supported_count
        ),
        citation_validity=(valid_citations / total_citations if total_citations else 0.0),
        citation_precision=(expected_citations / total_citations if total_citations else 0.0),
        refusal_accuracy=(
            sum(result.refusal_passed for result in refusal) / refusal_count
            if refusal_count
            else 1.0
        ),
        security_leakage_count=leakage_count,
    )
    failed = sum(not result.passed for result in case_results)
    return EvaluationReport(
        status="passed" if failed == 0 else "failed",
        mode=mode,
        suite_id=manifest.suite_id,
        manifest_sha256=manifest_sha256,
        corpus_fingerprint=index_report.corpus_fingerprint,
        embedding_model=embedding_model,
        answer_model=answer_model,
        chunk_size=selected_chunk_size,
        overlap=selected_overlap,
        top_k=selected_top_k,
        min_score=selected_min_score,
        case_count=len(case_results),
        passed_cases=len(case_results) - failed,
        failed_cases=failed,
        index_build_duration_ms=index_duration,
        total_duration_ms=round((perf_counter() - started) * 1_000, 3),
        metrics=metric,
        cases=case_results,
    )
