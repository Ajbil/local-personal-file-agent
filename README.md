# Local Personal File Agent

A learning-first implementation of a local Retrieval-Augmented Generation (RAG) agent for approved Markdown and text files.

The project is developed checkpoint by checkpoint so document ingestion, chunking, embeddings, vector indexing, retrieval, grounded generation, citations, evaluation, and security remain visible and understandable.

## Current Status

**Checkpoint 0 - Environment and Project Foundation:** complete. The locked Python environment, loopback Ollama runtime, EmbeddingGemma, and Qwen 3.5 4B have passed deterministic and live validation.

**Next:** review and merge the Checkpoint 0 pull request, then begin Checkpoint 1 - Secure File Discovery and Parsing.

Implemented:

- Reproducible Python 3.12 environment managed by `uv`.
- Typed, loopback-only configuration.
- Direct and validated Ollama HTTP boundary.
- `file-agent doctor` human and JSON diagnostics.
- Deterministic tests that do not require installed models.

Not implemented yet: file discovery, chunking, indexing, retrieval, or user-document processing.

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

## Troubleshooting

- **`uv` not found after installation:** open a new PowerShell window so the updated user `PATH` is loaded.
- **Ollama connection fails:** start Ollama and check `http://127.0.0.1:11434/api/version`.
- **Required model is missing:** run `ollama pull <model-name>` using the name printed by `doctor`.
- **Qwen is slow:** CPU execution is supported. Record timings before deciding whether a smaller development fallback is justified.
- **Configuration is rejected:** confirm the Ollama URL is HTTP on `localhost`, `127.0.0.1`, or another loopback IP and contains no credentials/path/query.

## Project Documents

- [Learning-first implementation plan](docs/learning-first-implementation-plan.md)
- [Checkpoint 0 learning record](docs/learning/checkpoint-0.md)
- [Architecture decision: direct local Ollama boundary](docs/decisions/0001-direct-local-ollama-boundary.md)
- [Original Notion guide](Local%20Personal%20File%20Agent%20060c2786553b82208d268122f958b13d.md)
- [Persistent collaboration guidance](AGENTS.md)
