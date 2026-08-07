# Checkpoint 0 Learning Record

## Objective

Create a reproducible Python foundation and prove whether the local Ollama runtime, embedding model, and answer model are ready before implementing RAG behavior.

## Environment Observed

| Component | Observation |
| --- | --- |
| Operating environment | 64-bit Windows |
| CPU | 11th Gen Intel Core i5-1135G7 |
| RAM | Approximately 15.7 GB |
| Python | CPython 3.12.10 |
| uv | 0.12.0 |
| NVIDIA runtime | Not detected |
| Ollama | 0.32.6; loopback API validated |
| EmbeddingGemma | Installed as `embeddinggemma:latest`; live smoke test passed |
| Qwen 3.5 4B | Installed; live schema-constrained generation passed |

CPU execution is the compatibility baseline. GPU acceleration will only be claimed if Ollama reports VRAM allocation.

## What Was Built

- `uv` project with a committed dependency lockfile.
- Installable `src`-layout Python package and `file-agent` command.
- Typed settings with loopback-only Ollama validation.
- Direct HTTP adapter for Ollama version, model inventory, embedding, structured chat, and runtime APIs.
- Ordered `doctor` diagnostics with human and JSON output.
- Safe failure translation that does not print response bodies or vector values.
- Deterministic tests using an in-memory HTTP transport and gateway fakes.

## Validation Evidence

```text
Ruff: passed
mypy strict mode: passed
pytest: 33 passed
coverage: 96.02%
```

Live values recorded:

- Ollama version: `0.32.6`.
- Installed embedding model: `embeddinggemma:latest`.
- Embedding dimension: `768`.
- Embedding load duration: `4,218.384 ms`.
- Embedding total duration: `4,418.684 ms`.
- Qwen load duration: `26,204.793 ms`.
- Qwen total duration: `154,348.501 ms`.
- Reported VRAM allocation: `0` bytes; CPU execution is active.
- Qwen structured output: schema-valid JSON accepted by Pydantic.

## Important Concepts

### Runtime versus model

Ollama is the program that loads and executes models. EmbeddingGemma and Qwen are model weights with different jobs. Installing Ollama does not install every model, and downloading a model does not create our Python application.

### Dependency declaration versus lockfile

`pyproject.toml` declares compatible dependency ranges and project tooling. `uv.lock` records the exact resolved dependency graph used to reproduce the environment.

### Deterministic versus live tests

Normal tests replace the external model runtime with controlled fakes. This proves our application decisions and failure handling. A live smoke test proves the actual runtime/models work on this machine. Both are required, but they answer different questions.

### Local does not automatically mean secure

The application rejects non-loopback Ollama URLs. This reduces the risk that personal text is accidentally sent across a LAN or to a remote host through configuration.

## Completion Gate

Checkpoint 0 completed successfully with:

```powershell
uv sync --locked
uv run file-agent doctor
```

The final doctor report passed Python, loopback configuration, Ollama connectivity, model inventory, 768-dimensional embedding generation, and schema-constrained Qwen generation. Ollama reported CPU execution as a non-blocking performance warning.

## Explain-Back Questions

1. Why are Python, `uv`, Ollama, EmbeddingGemma, and Qwen separate components?
2. Why does the application reject an Ollama server on a LAN IP even though it may be owned by you?
3. What does deterministic testing prove that a live model test does not?
4. What does a live model test prove that mocked tests cannot?
5. Why does the doctor report embedding dimensions but not vector values?
6. What evidence would justify using a smaller answer model during development?
