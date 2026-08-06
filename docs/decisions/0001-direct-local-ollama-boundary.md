# ADR 0001: Use a Direct Local Ollama HTTP Boundary

- Status: Accepted
- Date: 2026-08-06

## Context

The project must teach the mechanics and failure modes of a local RAG system. A high-level AI framework would make the first implementation faster, but it would hide request construction, timeouts, schema validation, model identity, and transport errors.

The application will eventually process personal files, so normal inference must remain local and accidental remote configuration must fail early.

## Decision

- Call Ollama's loopback HTTP API directly through HTTPX.
- Validate all configuration and responses with Pydantic.
- Reject non-loopback Ollama hosts and URLs containing credentials, paths, queries, or fragments.
- Translate transport failures into safe application errors without response bodies.
- Expose a small gateway protocol so deterministic tests can replace Ollama without changing application logic.
- Defer LangChain, LlamaIndex, and hosted-model adapters until the direct pipeline is understood and evaluated.

## Consequences

Positive:

- The external boundary is explicit, inspectable, and testable.
- Personal-document traffic cannot be redirected remotely by ordinary configuration.
- Later runtime or SDK changes are isolated behind one adapter.
- Developers learn the underlying API rather than only a framework abstraction.

Tradeoffs:

- We own request/response validation and error translation.
- New Ollama API features require deliberate adapter changes.
- This is more initial code than using an orchestration framework.

