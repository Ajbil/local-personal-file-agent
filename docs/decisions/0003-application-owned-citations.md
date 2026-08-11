# ADR 0003: Keep Citations Application-Owned and Fail Closed

- Status: Accepted
- Date: 2026-08-11

## Context

An answer model can invent plausible filenames, offsets, and citation markers. Prompt instructions
reduce this behavior but cannot make model-authored provenance trustworthy, especially when
retrieved documents contain hostile instructions.

## Decision

- Give Qwen temporary numeric evidence IDs and passage content, not source metadata.
- Require a strict structured payload containing answer text, selected IDs, and a sufficiency flag.
- Accept IDs only when they belong to the exact current retrieval context.
- Construct final source paths, chunk indexes, offsets, scores, and hashes in Python.
- Return one fixed refusal with zero citations for insufficient or semantically invalid output.

## Consequences

Positive:

- Model output cannot create trusted provenance outside retrieved evidence.
- Citation validation is deterministic, testable, and independent of prompt compliance.
- Unsupported and malformed semantic outcomes fail safely.

Tradeoffs:

- Valid citations prove provenance, not that the answer interpreted evidence correctly.
- Small models may produce false refusals.
- The application owns schema compatibility, mapping, retry, and refusal logic.

## Reconsider when

Never delegate trust in provenance to free-form model text. A future API or reranker may change the
evidence representation, but application-side validation and source ownership must remain.
