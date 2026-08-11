# ADR 0002: Use SQLite and Brute-Force NumPy Retrieval for the Learning Baseline

- Status: Accepted
- Date: 2026-08-11

## Context

The project must expose vector serialization, provenance, compatibility validation, cosine
similarity, ranking, and overlap suppression. A managed vector database would hide several of these
mechanics and add operational complexity before scale requires it.

## Decision

- Persist metadata, documents, chunks, and Float32 vectors in versioned SQLite.
- Validate the complete index before use and open it read-only for search and answering.
- Load vectors and calculate cosine similarity with NumPy.
- Rank deterministically and suppress heavily overlapping same-source chunks.
- Treat top-K and similarity thresholds as evaluated corpus-specific policies.

## Consequences

Positive:

- Storage and retrieval remain local, inspectable, portable, and transactional.
- The learner can explain every vector-search step and corruption check.
- Backup and disposal are understandable for a single local database file.

Tradeoffs:

- Search is linear in the number of chunks and loads vectors into process memory.
- Rebuilding is full-corpus rather than incremental.
- SQLite provides no approximate-nearest-neighbor index or distributed scaling.

## Reconsider when

Measured search latency, memory, corpus size, concurrent access, incremental-update requirements, or
availability goals exceed the documented local baseline. Adoption requires representative
benchmarks and a migration/provenance plan, not only an assumption that a vector database is newer.
