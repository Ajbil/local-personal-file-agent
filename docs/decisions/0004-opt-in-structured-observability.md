# ADR 0004: Use Opt-In Privacy-Safe Structured Observability

- Status: Accepted
- Date: 2026-08-11

## Context

RAG failures span ingestion, embedding, storage, retrieval, generation, validation, and evaluation.
Operators need stage, count, configuration, score, latency, outcome, and error signals. Conventional
debug logging can expose personal paths, questions, passages, prompts, answers, vectors, or model
responses and can also corrupt machine-readable stdout.

## Decision

- Keep logging off by default.
- Enable it explicitly with global `--log-level error|info`.
- Emit versioned JSON Lines to stderr only.
- Accept only typed allowlisted metrics and safe categories.
- Exclude content, paths, citations, raw exceptions/responses, hashes, and canaries.
- Do not automatically create or retain log files.
- Ensure serialization/sink failure never changes command behavior.

## Consequences

Positive:

- Commands become diagnosable without changing stdout contracts.
- Privacy policy is enforced by a schema rather than developer convention alone.
- JSONL supports deliberate redirection and later machine processing.

Tradeoffs:

- Logging must be enabled before reproducing a problem.
- Automatic historical diagnostics are unavailable.
- The allowlist requires deliberate schema changes for new metrics.

## Reconsider when

A service deployment introduces multiple requests, centralized collection, retention, access
control, redaction, sampling, trace propagation, or compliance obligations. Those requirements need
a separate production logging design and threat-model review.
