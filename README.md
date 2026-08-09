# Local Personal File Agent

A learning-first implementation of a local Retrieval-Augmented Generation (RAG) agent for approved Markdown and text files.

The project is developed checkpoint by checkpoint so document ingestion, chunking, embeddings, vector indexing, retrieval, grounded generation, citations, evaluation, and security remain visible and understandable.

## Current Status

**Checkpoint 4 - SQLite Vector Index:** complete. Approved documents, exact chunks, provenance, and validated Float32 embeddings can now be transactionally persisted, atomically published, and reopened read-only.

**Next:** begin Checkpoint 5 - Vector Search.

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

Not implemented yet: reusable vector search, overlap suppression, citations, or answer generation.

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

Normal tests mock the Ollama HTTP boundary, which keeps them fast and deterministic. Live model validation is performed through `file-agent doctor` and future tests marked `live`.

## Manual Checkpoint Verification

Every completed checkpoint includes a repeatable PowerShell lab with expected output,
interpretation, safe failure experiments, privacy checks, and automated validation:

- [Checkpoint 0 — Environment and runtime readiness](docs/testing/checkpoint-0.md)
- [Checkpoint 1 — Secure file discovery and parsing](docs/testing/checkpoint-1.md)
- [Checkpoint 2 — Deterministic chunking](docs/testing/checkpoint-2.md)
- [Checkpoint 3 — Local embeddings](docs/testing/checkpoint-3.md)
- [Checkpoint 4 — SQLite vector index](docs/testing/checkpoint-4.md)

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
- [Architecture decision: direct local Ollama boundary](docs/decisions/0001-direct-local-ollama-boundary.md)
- [Original Notion guide](Local%20Personal%20File%20Agent%20060c2786553b82208d268122f958b13d.md)
- [Persistent collaboration guidance](AGENTS.md)
