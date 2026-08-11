# Production Evolution Map

## Purpose

The current system is an inspectable single-user local learning baseline. This map describes how it
could evolve when measured requirements justify additional complexity. It is not a backlog promise
and none of these capabilities should be adopted only because they are fashionable.

## Current baseline

- One explicitly approved local folder per operation.
- UTF-8 Markdown and plain text.
- Full deterministic character chunking and reindexing.
- EmbeddingGemma and Qwen through loopback Ollama.
- Versioned SQLite with brute-force NumPy cosine similarity.
- Single-user CLI, no network API or ACLs.
- Deterministic and opt-in live evaluation.
- Opt-in local privacy-safe JSONL observability.

## Stage 1: Incremental and reliable indexing

Trigger:

- Full rebuild time becomes materially disruptive or corpora change frequently.

Evolution:

- Track source identity, content hash, parser/chunker version, model, and deletion tombstones.
- Re-embed only new or changed chunks and remove deleted-document records transactionally.
- Add exclusive indexing locks, resumable jobs, cancellation, and recovery tests.

New risks and proof required:

- Stale/deleted content, mixed model versions, partial publication, and concurrent readers.
- Prove equivalence between clean rebuild and incremental state with property/integration tests.

## Stage 2: PDF/DOCX and richer parsers

Trigger:

- Representative user needs cannot be met with Markdown/text.

Evolution:

- Add format-specific parser adapters with page/section provenance and parser-version metadata.
- Run complex native parsers in a resource-limited sandbox or isolated worker.
- Detect encrypted, malformed, oversized, macro-enabled, and decompression-bomb inputs.

New risks and proof required:

- Parser exploits, hidden text, OCR errors, layout loss, tables, and inaccurate citations.
- Build licensed synthetic fixtures and verify exact page/section mappings.

## Stage 3: Hybrid retrieval and metadata filtering

Trigger:

- Evaluation shows vector recall misses exact identifiers, codes, names, or rare terms.

Evolution:

- Combine BM25/keyword and vector candidates using a documented fusion method.
- Add validated metadata filters and source scopes before retrieval.
- Keep each candidate channel and fusion score observable.

New risks and proof required:

- One channel dominates, filters leak metadata, or thresholds overfit the benchmark.
- Compare Hit@K, MRR, refusal, latency, and security across a larger labeled set.

## Stage 4: Reranking and model comparison

Trigger:

- Candidate recall is good but distractors reduce answer precision or context efficiency.

Evolution:

- Retrieve a wider candidate set and apply a local cross-encoder/reranker before generation.
- Version prompt, embedding, reranker, and answer-model identities in evaluation results.
- Use paired evaluation rather than anecdotal examples for model changes.

New risks and proof required:

- Added latency/RAM, model supply-chain risk, new truncation behavior, and benchmark overfitting.
- Require statistically meaningful quality improvement without refusal/security regression.

## Stage 5: Vector database adoption

Trigger:

- Measured SQLite search latency or memory exceeds the service objective, or concurrency and
  incremental mutation become primary requirements.

Evolution:

- Evaluate local or managed stores for filter correctness, persistence, backup, tenancy, encryption,
  and operational support—not only nearest-neighbor speed.
- Preserve document/chunk IDs, hashes, model versions, and application-owned citation mapping.

New risks and proof required:

- Network exposure, vendor coupling, eventual consistency, migration errors, and tenant leakage.
- Benchmark representative scale and validate dual-read/migration rollback before cutover.

## Stage 6: Authenticated service and background work

Trigger:

- More than one trusted process or user requires access.

Evolution:

- Add a narrow authenticated API, request size/rate limits, idempotency, job queue, cancellation, and
  resource quotas.
- Separate indexing workers from read-only query serving.
- Define availability objectives and backpressure behavior.

New risks and proof required:

- Network attacks, abusive workloads, confused-deputy access, replay, and queue data retention.
- Threat-model the service boundary and perform authentication/authorization/load testing.

## Stage 7: Authorization and tenant isolation

Trigger:

- Users have different permissions or data ownership.

Evolution:

- Propagate document ACLs into chunk metadata and enforce them before candidate scoring/context.
- Use deny-by-default RBAC/ABAC, tenant-scoped encryption/storage, and authorization audit events.
- Test every cache, background job, retrieval path, and citation for tenant isolation.

New risks and proof required:

- Cross-tenant search, stale ACLs, cache leakage, and privileged background operations.
- Require adversarial authorization tests and auditable policy-change propagation.

## Stage 8: Data protection and lifecycle

Trigger:

- Organizational, legal, contractual, backup, or recovery requirements apply.

Evolution:

- Encrypt storage/backups, manage keys/secrets, define retention/deletion, classify data, and support
  reproducible restore plus deletion verification.
- Treat embeddings, questions, answers, evaluations, and logs according to their source sensitivity.

New risks and proof required:

- Key loss, undeleted replicas, sensitive backups/logs, and incomplete subject deletion.
- Test backup/restore, key rotation, retention enforcement, and disaster recovery.

## Stage 9: Central observability and operations

Trigger:

- A service team needs cross-request diagnosis, alerting, capacity planning, or audit evidence.

Evolution:

- Add trace/request propagation, metrics, privacy-reviewed structured events, dashboards, alerts,
  sampling, redaction, retention, and access control.
- Publish runbooks for Ollama/model, indexing, corruption, latency, and quality incidents.

New risks and proof required:

- Telemetry becomes a sensitive secondary datastore or high-cardinality cost source.
- Run privacy tests against the collector and rehearse incident/retention procedures.

## Stage 10: Scheduled evaluation and controlled model releases

Trigger:

- Models, prompts, parsers, retrieval policy, or corpora change regularly.

Evolution:

- Expand labeled domain datasets, schedule deterministic/live evaluation, store safe trends, compare
  candidates, and block deployment on declared quality/security regressions.
- Add human review for ambiguous answer grading and a rollback path for each model/config release.

New risks and proof required:

- Dataset leakage, stale labels, test overfitting, nondeterminism, and unsafe metric storage.
- Version datasets and graders, monitor distributions, and audit overrides.

## Decision principle

For every evolution, begin with a measured failure or requirement, make the smallest architectural
change that addresses it, preserve provenance/security boundaries, and prove the result through
representative evaluation and operational tests.
