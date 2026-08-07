# Checkpoint 0 Manual Verification — Environment and Runtime Readiness

## Goal

Verify that the reproducible Python environment, local Ollama boundary, embedding model, answer
model, diagnostics, safe configuration, and automated checks work as designed.

Run every command from the repository root.

## 1. Confirm repository and tool state

```powershell
git status
uv --version
ollama --version
```

Expected:

- The working tree is clean before testing.
- `uv` and Ollama are recognized commands.
- Exact version numbers may be newer than the original Checkpoint 0 record.

## 2. Reproduce the locked Python environment

```powershell
uv sync --locked
uv run python --version
uv run file-agent --help
```

Expected:

- Python reports a supported `3.12.x` version.
- The environment synchronizes without changing `uv.lock`.
- The `file-agent` command displays its available subcommands.

`pyproject.toml` declares allowed dependency ranges. `uv.lock` records the exact resolved dependency
graph used by the project.

## 3. Confirm the required local models

```powershell
ollama list
```

Expected model names include:

```text
embeddinggemma:latest
qwen3.5:4b
```

If one is missing:

```powershell
ollama pull embeddinggemma
ollama pull qwen3.5:4b
```

These downloads require internet access. Normal project inference remains local afterward.

## 4. Run the fast diagnostic path

```powershell
uv run file-agent doctor --skip-generation
```

Expected checks:

- Python 3.12 passes.
- Configuration is restricted to loopback.
- The local Ollama API is reachable.
- Both required models are installed.
- EmbeddingGemma returns a valid vector.
- Generation is reported as intentionally skipped.

The final status says `NOT FULLY READY` because full readiness cannot be claimed without running
Qwen. This is expected, not a failure.

## 5. Run the complete live diagnostic

```powershell
uv run file-agent doctor
```

Expected final line:

```text
Checkpoint 0 runtime status: READY
```

On CPU-only hardware, Qwen may take several minutes. A VRAM warning is informational because CPU
execution is an explicitly supported baseline.

The embedding check should report a dimension such as `768`. The vector values are deliberately
not printed because they are large and could encode information about future personal text.

## 6. Inspect the JSON contract

```powershell
uv run file-agent doctor --json
```

Verify that the result contains:

- `success`
- `full_readiness`
- ordered checks
- safe status, summary, and detail fields

It must not contain embedding-vector values or raw model-response bodies.

## 7. Test the loopback security boundary

Set an invalid remote URL only for the current PowerShell process:

```powershell
$env:FILE_AGENT_OLLAMA_BASE_URL = "https://example.com"
uv run file-agent doctor
$LASTEXITCODE
```

Expected:

- Configuration is rejected before any network request.
- Exit code is `2`.
- The supplied URL is not echoed in the safe error output.

Restore the environment immediately:

```powershell
Remove-Item Env:FILE_AGENT_OLLAMA_BASE_URL
```

Confirm local diagnostics work again:

```powershell
uv run file-agent doctor --skip-generation
```

## 8. Run automated verification

```powershell
uv run pytest
uv run pytest --cov=local_file_agent --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

All commands must pass. The coverage total must remain at or above the configured 85% threshold.

Normal tests use controlled fakes and do not require Ollama. `doctor` is the live integration test
for the actual local runtime and models.

## What this checkpoint proves

- A fresh checkout can reproduce its Python environment.
- Ollama is treated as a separate local service.
- Model inventory and behavior are validated rather than assumed.
- Unsafe remote configuration is rejected at the boundary.
- Deterministic tests and live diagnostics answer different questions.
- Performance warnings do not get confused with correctness failures.
