# Checkpoint 8 Manual Verification — Hardening and Senior-Engineer Retrospective

## Goal

Verify that structured operational events are opt-in, useful, machine-readable, and unable to leak
representative sensitive values; then confirm the final architecture, threat model, decisions,
evaluation, and quality gates are reproducible.

Read the [Checkpoint 8 learning record](../learning/checkpoint-8.md) and
[threat model](../threat-model.md) first. Run commands from the repository root.

## 1. Prepare and confirm the interface

```powershell
uv sync --locked
uv run file-agent --help
```

Expected: the global option is `--log-level <off|error|info>` and defaults to `off`. It must appear
before the subcommand.

## 2. Prove logging is off by default

```powershell
$checkpoint8Root = Join-Path $PWD ".data\manual-testing\checkpoint-8"
New-Item -ItemType Directory -Force $checkpoint8Root

uv run file-agent evaluate --mode deterministic --json `
  1> (Join-Path $checkpoint8Root "report.json") `
  2> (Join-Path $checkpoint8Root "disabled.log")

(Get-Item (Join-Path $checkpoint8Root "disabled.log")).Length
```

Expected: length is `0`. The JSON report remains valid and the evaluation passes.

```powershell
Get-Content (Join-Path $checkpoint8Root "report.json") |
  ConvertFrom-Json |
  Select-Object status, case_count, passed_cases, failed_cases
```

## 3. Enable INFO JSONL on stderr

```powershell
uv run file-agent --log-level info evaluate --mode deterministic --json `
  1> (Join-Path $checkpoint8Root "report-with-logs.json") `
  2> (Join-Path $checkpoint8Root "events.jsonl")

Get-Content (Join-Path $checkpoint8Root "report-with-logs.json") | ConvertFrom-Json |
  Select-Object status, case_count

Get-Content (Join-Path $checkpoint8Root "events.jsonl") | ForEach-Object {
  $_ | ConvertFrom-Json
} | Select-Object schema_version, level, event, command, operation_id, outcome, duration_ms
```

Expected:

- stdout is still one valid evaluation report;
- stderr contains `command.started` and `command.completed` JSON objects;
- both events share one operation ID;
- completion contains non-negative duration and safe evaluation counts;
- no database or log is created automatically outside your explicit redirection.

## 4. Inspect the safe field allowlist

```powershell
Get-Content (Join-Path $checkpoint8Root "events.jsonl") | ForEach-Object {
  ($_ | ConvertFrom-Json).fields
}
```

Expected fields include mode, model identifiers, chunking/retrieval settings, case counts, and
security leakage count. They do not include questions, passages, answers, paths, prompts, vectors,
citations, hashes, or canary values.

## 5. Compare ERROR logging

```powershell
uv run file-agent --log-level error evaluate --mode deterministic --json `
  1> (Join-Path $checkpoint8Root "error-level-report.json") `
  2> (Join-Path $checkpoint8Root "error-level-success.jsonl")

(Get-Item (Join-Path $checkpoint8Root "error-level-success.jsonl")).Length
```

Expected: length `0`, because the command passed and ERROR omits lifecycle events.

Force a quality failure without changing the baseline:

```powershell
uv run file-agent --log-level error evaluate `
  --mode deterministic `
  --min-score 1 `
  --json `
  1> (Join-Path $checkpoint8Root "quality-failure-report.json") `
  2> (Join-Path $checkpoint8Root "quality-failure.jsonl")
$LASTEXITCODE

Get-Content (Join-Path $checkpoint8Root "quality-failure.jsonl") | ConvertFrom-Json |
  Select-Object event, outcome, failure_category
```

Expected: exit `1`, event `command.failed`, and category `quality_gate`.

Force invalid input:

```powershell
uv run file-agent --log-level error evaluate --mode deterministic --top-k 0 `
  2> (Join-Path $checkpoint8Root "invalid-input.jsonl")
$LASTEXITCODE

Get-Content (Join-Path $checkpoint8Root "invalid-input.jsonl") |
  Select-Object -Last 1 |
  ConvertFrom-Json |
  Select-Object failure_category
```

Expected: exit `2` and category `invalid_input`.

## 6. Run a privacy-sentinel experiment

```powershell
$privateSentinel = "PRIVATE-SENTINEL-CHECKPOINT-8"
$privateDatabase = Join-Path $checkpoint8Root "$privateSentinel.sqlite"

uv run file-agent --log-level error ask $privateSentinel --db $privateDatabase `
  2> (Join-Path $checkpoint8Root "privacy-error.txt")

$structuredEvents = Get-Content (Join-Path $checkpoint8Root "privacy-error.txt") |
  Where-Object { $_.StartsWith("{") }

($structuredEvents -join "`n") -match $privateSentinel
```

Expected: `False`. The structured event records only a safe error category. The ordinary human
error remains separate from the JSON event and must also avoid echoing the question.

## 7. Observe a safe refusal

Build a disposable Checkpoint 7 index if one is not already available:

```powershell
$checkpoint8Db = Join-Path $checkpoint8Root "evaluation-source.sqlite"

uv run file-agent index `
  --source examples/checkpoint-7/source `
  --db $checkpoint8Db

uv run file-agent --log-level info ask `
  "What is the office Wi-Fi password?" `
  --db $checkpoint8Db `
  --json `
  1> (Join-Path $checkpoint8Root "refusal.json") `
  2> (Join-Path $checkpoint8Root "refusal-events.jsonl")
```

Expected: command exit `0`, answer status `refused`, zero citations, and completion outcome
`refused`. A refusal is a safe product decision, not an infrastructure error.

## 8. Review the hardening documents

```powershell
Get-Content docs/threat-model.md
Get-Content docs/decisions/0002-sqlite-brute-force-retrieval.md
Get-Content docs/decisions/0003-application-owned-citations.md
Get-Content docs/decisions/0004-opt-in-structured-observability.md
Get-Content docs/production-evolution.md
```

For each document, identify the context, current control/decision, tradeoff, residual risk, and
measurable trigger for change.

## 9. Run final automated acceptance

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest --cov=local_file_agent --cov-report=term-missing
uv run file-agent evaluate --mode deterministic
```

Then deliberately run the slower real-model gate:

```powershell
uv run file-agent evaluate --mode live
```

Expected: all deterministic and baseline live cases pass, citations remain valid/precise, refusal
accuracy remains `1.0`, and security leakage remains zero. Live runtime varies by hardware.

## 10. Clean disposable artifacts

All files created by this lab are under Git-ignored `.data/`. After reviewing them, remove the
specific checkpoint directory using your normal recoverable cleanup process. Never delete a broad
workspace or home-directory path.

## Explain-back questions

1. Why does structured logging use an allowlist instead of filtering known secret fields?
2. Why must logs go to stderr when commands support `--json`?
3. Why is a refusal an INFO completion while an Ollama failure is ERROR?
4. What can application-owned citations guarantee, and what can they not guarantee?
5. Which trust boundary handles malicious document instructions?
6. What measured condition would justify hybrid retrieval, reranking, or a vector database?
7. What new threats appear when the CLI becomes a multi-user network service?
8. Why does completing this checkpoint not make the application enterprise-production-ready?

## What this checkpoint proves

- Every command has opt-in lifecycle/failure observability.
- Machine-readable stdout remains stable.
- The event schema cannot accept representative content/path fields.
- Safe refusal, quality failure, invalid input, and operational failure are distinguishable.
- Security assumptions, controls, residual risks, and architecture decisions are documented.
- Production evolution is evidence-driven rather than prematurely implemented.
- The complete local learning baseline is reproducible and explainable.
