# Local Personal File Agent

A learning-first implementation of a local Retrieval-Augmented Generation (RAG) agent for approved Markdown and text files.

The project is developed checkpoint by checkpoint so document ingestion, chunking, embeddings, vector indexing, retrieval, grounded generation, citations, evaluation, and security remain visible and understandable.

## Current Status

**Checkpoint 8 - Hardening and Senior-Engineer Retrospective:** complete. The planned local learning
baseline now includes opt-in privacy-safe structured observability, a threat model, recorded
architectural decisions, explicit limitations, and an evidence-driven production evolution map.

**Project milestone:** all eight planned checkpoints are complete. The application remains an
inspectable single-user learning baseline, not an enterprise production service.

Implemented:

- Reproducible Python 3.12 environment managed by `uv`.
- Typed, loopback-only configuration.
- Direct and validated Ollama HTTP boundary.
- `file-agent doctor` human and JSON diagnostics.
- Deterministic tests that do not require installed models.
- Recursive, deterministic discovery under an explicitly approved source root.
- UTF-8 parsing, line-ending normalization, stable content hashes, and safe rejection reasons.
- Metadata-only `file-agent scan` human and JSON reports.
- Stable path-derived document identities.
- Deterministic character chunking with natural-boundary preference and exact source offsets.
- Privacy-aware `file-agent inspect-chunks` human and JSON reports.
- Prompt-aware document and query embedding through the direct local Ollama API.
- Ordered batching with model, vector-count, dimension, finite-value, and non-zero-norm validation.
- NumPy Float32 vectors kept in memory and protected from accidental mutation.
- Privacy-aware `file-agent inspect-embeddings` reports with timings, norms, and cosine rankings.
- Versioned SQLite schema for metadata, documents, chunks, and portable Float32 embedding blobs.
- Transactional temporary builds with read-only validation before atomic replacement.
- App-ownership, schema, foreign-key, hash, count, dimension, and vector corruption checks.
- Metadata-only `file-agent index` and read-only `file-agent inspect-index` reports.
- Index-first query embedding using the exact stored model, prompt strategy, and dimension.
- Brute-force NumPy cosine retrieval with stable tie-breakers and provisional score thresholds.
- Fixed 80% same-document overlap suppression before top-K selection.
- Privacy-aware `file-agent search` human and JSON reports with trusted source citations.
- Structured local Qwen generation with temperature zero and thinking disabled.
- Bounded, JSON-escaped, explicitly untrusted evidence construction.
- Strict answer schema, one malformed-output retry, and fail-closed semantic validation.
- Application-owned citation mapping and fixed unsupported-question refusal.
- Privacy-aware `file-agent ask` human and JSON reports with opt-in context inspection.
- Strict synthetic evaluation manifest and corpus with path, expectation, and canary validation.
- Fast offline hashed-lexical evaluation through production indexing and retrieval.
- Live EmbeddingGemma/Qwen evaluation with stage-specific diagnostics.
- Hit@K, MRR, fact, citation, refusal, leakage, and latency metrics without content logging.
- Disposable Git-ignored evaluation indexes and automation-friendly exit codes.
- Opt-in versioned JSONL command events on stderr with a strict privacy allowlist.
- Threat model separating deterministic controls, model behavior, assumptions, and residual risk.
- Architecture decisions for the local runtime, SQLite retrieval, trusted citations, and logging.
- Production evolution map with measurable adoption triggers and new-risk analysis.

Intentional limitations include text formats only, character chunking, full index rebuilds,
brute-force search, a single-user CLI, no ACLs/encryption, a small local model, and a limited
synthetic evaluation corpus. See the threat model and production evolution map before extending the
baseline.

## Mental Model

```text
Python application
    |
    +-- Settings: safe local configuration
    |
    +-- Ollama HTTP adapter: external runtime boundary
    |
    +-- Doctor: ordered readiness checks
            |
            +-- EmbeddingGemma: text -> vector smoke test
            +-- Qwen 3.5 4B: structured generation smoke test
```

Python runs our code. `uv` reproduces its environment. Ollama is a separate local service that loads and runs the models.

The Checkpoint 1 ingestion boundary runs before any model operation:

```text
Approved source folder
    |
    +-- secure discovery and validation
            |
            +-- trusted normalized Documents (internal)
            +-- privacy-safe metadata report (terminal / JSON)
```

Checkpoint 3 adds a model transformation while keeping the stages inspectable:

```text
Trusted Document -> deterministic Chunks -> document prompts -> EmbeddingGemma -> Float32 vectors
User question -----------------------------> query prompt -----> EmbeddingGemma -> query vector
                                                                            |
                                                      in-memory cosine ranking
```

Checkpoint 4 adds a durable trust boundary:

```text
Documents -> Chunks -> Embeddings -> temporary SQLite transaction
                                      |
                              read-only validation
                                      |
                               atomic publication
                                      v
                         durable versioned local index
```

Checkpoint 5 adds independently observable retrieval:

```text
Question -> index-recorded EmbeddingGemma -> query vector
                                              |
read-only SQLite vectors ----------------> cosine scores
                                              |
                           threshold -> overlap suppression -> top-K citations
```

Checkpoint 6 completes the first RAG loop while keeping provenance outside the model:

```text
Question -> Checkpoint 5 retrieval -> numbered untrusted passages -> local Qwen structured JSON
                                                                         |
                         fixed refusal <- semantic validation <- answer + temporary IDs
                                                                         |
                                   trusted SQLite metadata -> real source citations
```

Qwen acts as a writer, not a provenance authority. Prompt instructions guide the writer; strict
schema and citation validation enforce the application boundary.

Checkpoint 8 adds an operational boundary around every command:

```text
CLI command -> safe typed lifecycle metrics -> optional JSONL stderr
       |
       +-> unchanged human or machine-readable stdout
```

Logging has no access to arbitrary content fields, is disabled by default, and cannot change a
command result if its output sink fails.

## Prerequisites

- Windows 10 22H2 or newer.
- Python 3.12.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
- [Ollama for Windows](https://docs.ollama.com/windows).
- Enough local storage for Ollama and both model weights.

Ollama must remain bound to `127.0.0.1`/`localhost`. Do not expose port `11434` to a LAN, proxy, or tunnel for this personal-file project.

## Setup

Install `uv` when needed:

```powershell
winget install --id=astral-sh.uv -e
```

Open a new PowerShell window, clone the repository, and reproduce the locked environment:

```powershell
uv sync --locked
uv run python --version
uv run file-agent --help
```

Install Ollama using its Windows installer, then download the baseline models:

```powershell
ollama pull embeddinggemma
ollama pull qwen3.5:4b
ollama list
```

Run the complete readiness check:

```powershell
uv run file-agent doctor
```

Machine-readable output:

```powershell
uv run file-agent doctor --json
```

Skip the slower Qwen smoke test while diagnosing other prerequisites:

```powershell
uv run file-agent doctor --skip-generation
```

This may succeed without claiming `full_readiness=true`.

## Scan Approved Documents

Start with the committed synthetic sample rather than personal documents:

```powershell
uv run file-agent scan --source examples/checkpoint-1/source
```

Machine-readable metadata:

```powershell
uv run file-agent scan --source examples/checkpoint-1/source --json
```

The source option is mandatory. Scanning is recursive, accepts only UTF-8 `.md` and `.txt` files,
and never prints document content. Source files are read-only and nothing is persisted yet.

## Inspect Deterministic Chunks

Inspect chunk metadata for the synthetic learning document:

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md
```

Use custom settings for learning experiments:

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md `
  --chunk-size 500 `
  --overlap 100
```

The inspector prints offsets and hashes but no text by default. Add `--show-text` only for files
whose contents you deliberately want in terminal output. Add `--json` for machine-readable output.

## Inspect Local Embeddings

Run the semantic comparison lab using committed synthetic content:

```powershell
uv run file-agent inspect-embeddings `
  --source examples/checkpoint-3/source `
  --document semantic-lab.md `
  --query "How much funding can a worker use for career development?" `
  --chunk-size 450 `
  --overlap 0 `
  --top-k 3
```

The command uses only EmbeddingGemma, keeps vectors in memory, and shows relative cosine rankings.
It prints no query text, document text, or raw vector coordinates by default. Add `--show-text` only
for deliberately approved content. Exact scores and timings are model/runtime dependent.

## Build and Inspect a SQLite Vector Index

Create a disposable Git-ignored destination and index the committed synthetic corpus:

```powershell
New-Item -ItemType Directory -Force .data/manual-testing/checkpoint-4

uv run file-agent index `
  --source examples/checkpoint-4/source `
  --db .data/manual-testing/checkpoint-4/index.sqlite
```

Reopen and validate it without calling Ollama:

```powershell
uv run file-agent inspect-index `
  --db .data/manual-testing/checkpoint-4/index.sqlite
```

An existing valid index is not replaced unless `--force` is supplied. Even with `--force`, an
unrelated or unrecognizable file is never overwritten. Indexes contain chunk text and embeddings;
keep them under `.data/` or another private Git-ignored location.

## Search a Vector Index

Search uses the exact embedding model recorded inside the index:

```powershell
uv run file-agent search `
  "How frequently do staff passwords need to be changed?" `
  --db .data/manual-testing/checkpoint-5/index.sqlite
```

The default policy returns at most five results with cosine score at least `0.30`, after suppressing
same-document chunks that overlap by 80% or more. These values are provisional and must eventually
be evaluated against representative questions.

Search reports trusted relative paths, chunk numbers, offsets, and scores without printing passage
text. Add `--show-text` only for an approved index. Search never invokes Qwen and never modifies the
SQLite database.

## Generate a Grounded Answer

Use the same read-only index to retrieve evidence and ask local Qwen for a cited answer:

```powershell
uv run file-agent ask `
  "How quickly must a critical Copper Lantern alert be acknowledged?" `
  --db .data/manual-testing/checkpoint-6/index.sqlite
```

Qwen receives only temporary evidence IDs and passage text. Python validates the structured answer
and converts accepted IDs into trusted relative paths, chunk numbers, offsets, scores, and hashes.
If evidence is absent or the model returns invalid citations, the application returns a fixed
refusal with no sources.

The default output hides retrieved context. Add `--show-context` only for an approved index. The
answer itself is derived from document content and may still be sensitive. `ask`, like `search`,
opens the SQLite index read-only.

## Evaluate Quality and Security

Run the fast, offline regression gate:

```powershell
uv run file-agent evaluate --mode deterministic
```

Run the real local EmbeddingGemma/Qwen benchmark:

```powershell
uv run file-agent evaluate --mode live
```

Each run builds and deletes a disposable index. Reports contain metrics, case IDs, decisions, ranks,
counts, and timings but omit questions, passages, answers, vectors, canaries, and absolute paths.
Deterministic mode is suitable for normal development; live mode may take several minutes on CPU.

## Enable Privacy-Safe Operational Events

Enable INFO lifecycle events for one command:

```powershell
uv run file-agent --log-level info evaluate --mode deterministic
```

Emit only failure events:

```powershell
uv run file-agent --log-level error evaluate --mode deterministic
```

Events are versioned JSON Lines written to stderr. Normal output—including `--json`—remains on
stdout. Logs include allowlisted counts, model/configuration identifiers, scores, decisions,
latency, and safe error categories. They exclude questions, answers, passages, prompts, paths,
citations, hashes, vectors, raw model/HTTP output, and canaries.

No log file is created automatically. If you deliberately redirect stderr, keep it in `.data/` or
another private ignored directory and delete it when no longer needed.

## Configuration

Safe defaults are shown in [.env.example](.env.example). Supported variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FILE_AGENT_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama endpoint; non-loopback hosts are rejected. |
| `FILE_AGENT_ANSWER_MODEL` | `qwen3.5:4b` | Structured answer-generation model. |
| `FILE_AGENT_EMBEDDING_MODEL` | `embeddinggemma` | Semantic embedding model. |
| `FILE_AGENT_CONNECT_TIMEOUT_SECONDS` | `5` | Connection timeout. |
| `FILE_AGENT_MODEL_TIMEOUT_SECONDS` | `300` | CPU-compatible model-operation timeout. |

Local `.env` files are ignored by Git. Never commit personal paths, credentials, prompts, model output, or document content.

## Development Checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=local_file_agent --cov-report=term-missing
```

Normal tests use deterministic gateways, which keeps them fast and independent of Ollama. Live
model validation is performed explicitly through `file-agent doctor` and
`file-agent evaluate --mode live`.

## Manual Checkpoint Verification

Every completed checkpoint includes a repeatable PowerShell lab with expected output,
interpretation, safe failure experiments, privacy checks, and automated validation:

- [Checkpoint 0 — Environment and runtime readiness](docs/testing/checkpoint-0.md)
- [Checkpoint 1 — Secure file discovery and parsing](docs/testing/checkpoint-1.md)
- [Checkpoint 2 — Deterministic chunking](docs/testing/checkpoint-2.md)
- [Checkpoint 3 — Local embeddings](docs/testing/checkpoint-3.md)
- [Checkpoint 4 — SQLite vector index](docs/testing/checkpoint-4.md)
- [Checkpoint 5 — Read-only vector search](docs/testing/checkpoint-5.md)
- [Checkpoint 6 — Grounded answers and trusted citations](docs/testing/checkpoint-6.md)
- [Checkpoint 7 — Evaluation and security regression suite](docs/testing/checkpoint-7.md)
- [Checkpoint 8 — Hardening and senior-engineer retrospective](docs/testing/checkpoint-8.md)

See the [manual verification index](docs/testing/README.md) for the learning workflow. Synthetic
inputs live under `examples/`; disposable experiments belong under the Git-ignored `.data/` folder.

## Troubleshooting

- **`uv` not found after installation:** open a new PowerShell window so the updated user `PATH` is loaded.
- **Ollama connection fails:** start Ollama and check `http://127.0.0.1:11434/api/version`.
- **Required model is missing:** run `ollama pull <model-name>` using the name printed by `doctor`.
- **Qwen is slow:** CPU execution is supported. Record timings before deciding whether a smaller development fallback is justified.
- **Configuration is rejected:** confirm the Ollama URL is HTTP on `localhost`, `127.0.0.1`, or another loopback IP and contains no credentials/path/query.

## Project Documents

- [Learning-first implementation plan](docs/learning-first-implementation-plan.md)
- [Checkpoint 0 learning record](docs/learning/checkpoint-0.md)
- [Checkpoint 1 learning record](docs/learning/checkpoint-1.md)
- [Checkpoint 2 learning record](docs/learning/checkpoint-2.md)
- [Checkpoint 3 learning record](docs/learning/checkpoint-3.md)
- [Checkpoint 4 learning record](docs/learning/checkpoint-4.md)
- [Checkpoint 5 learning record](docs/learning/checkpoint-5.md)
- [Checkpoint 6 learning record](docs/learning/checkpoint-6.md)
- [Checkpoint 7 learning record](docs/learning/checkpoint-7.md)
- [Checkpoint 8 learning record](docs/learning/checkpoint-8.md)
- [Threat model](docs/threat-model.md)
- [Production evolution map](docs/production-evolution.md)
- [Architecture decision: direct local Ollama boundary](docs/decisions/0001-direct-local-ollama-boundary.md)
- [Architecture decision: SQLite brute-force retrieval](docs/decisions/0002-sqlite-brute-force-retrieval.md)
- [Architecture decision: application-owned citations](docs/decisions/0003-application-owned-citations.md)
- [Architecture decision: opt-in structured observability](docs/decisions/0004-opt-in-structured-observability.md)
- [Original Notion guide](Local%20Personal%20File%20Agent%20060c2786553b82208d268122f958b13d.md)
- [Persistent collaboration guidance](AGENTS.md)
