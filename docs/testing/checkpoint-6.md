# Checkpoint 6 Manual Verification — Grounded Answers and Trusted Citations

## Goal

Verify the complete local RAG path: retrieve relevant passages, send only numbered untrusted
evidence to Qwen, validate its structured response, map temporary IDs to trusted source metadata,
and return either a cited answer or a fixed refusal.

Read the [Checkpoint 6 learning record](../learning/checkpoint-6.md) first for the mental models,
trust boundaries, tradeoffs, failure taxonomy, and known false-refusal limitation.

This lab uses committed synthetic documents. Do not use `--show-context` with personal indexes in a
terminal, screenshot, or recording you may share.

## 1. Prepare the local runtime and disposable index

Run from the repository root:

```powershell
uv sync --locked
ollama list
uv run file-agent doctor

$checkpoint6Lab = Join-Path $PWD ".data\manual-testing\checkpoint-6"
New-Item -ItemType Directory -Force $checkpoint6Lab
$checkpoint6Db = Join-Path $checkpoint6Lab "index.sqlite"

uv run file-agent index `
  --source examples/checkpoint-6/source `
  --db $checkpoint6Db
```

If the index already exists, choose a new filename or deliberately add `--force`.

## 2. Inspect the new command

```powershell
uv run file-agent ask --help
```

Important inputs:

- positional question;
- required `--db`;
- `--top-k`, default `5`;
- `--min-score`, default `0.30`;
- `--show-context`, default off;
- `--json`.

The answer model comes from `FILE_AGENT_ANSWER_MODEL`, whose default is `qwen3.5:4b`. Retrieval uses
the embedding model recorded inside the index, not whichever embedding model is currently configured.

## 3. Ask a directly supported question

```powershell
uv run file-agent ask `
  "How quickly must a critical Copper Lantern alert be acknowledged?" `
  --db $checkpoint6Db
```

Expected behavior:

- the answer says ten minutes;
- the status is `Grounded answer generated`;
- at least one source points to `incident-response.md`;
- the source includes an application-owned chunk number and character range;
- retrieved passage text is not printed;
- generation may take noticeably longer on CPU the first time the model loads.

Exact wording, similarity scores, and latency may vary because Qwen and EmbeddingGemma are model
inference, not hard-coded lookup tables.

## 4. Compare retrieval with generation

Run the same question through Checkpoint 5 search:

```powershell
uv run file-agent search `
  "How quickly must a critical Copper Lantern alert be acknowledged?" `
  --db $checkpoint6Db
```

Compare the two commands:

```text
search → ranked evidence and scores; Qwen is not used
ask    → same retrieval, then Qwen, validation, and trusted citation mapping
```

If `search` does not retrieve the expected source, answer generation cannot repair the missing
evidence. This is how a senior engineer separates a retrieval failure from a generation failure.

## 5. Test a semantic paraphrase

```powershell
uv run file-agent ask `
  "What annual budget can a worker spend on professional development?" `
  --db $checkpoint6Db
```

Expected: the answer mentions 900 learning credits and cites `learning-benefit.txt`, even though the
question does not copy the document's exact wording.

## 6. Reveal exactly what Qwen received

```powershell
uv run file-agent ask `
  "How quickly must a critical Copper Lantern alert be acknowledged?" `
  --db $checkpoint6Db `
  --show-context
```

Expected:

- a privacy warning appears;
- every passage sent to Qwen is numbered;
- exact synthetic passage text is shown;
- real source metadata is printed by the application, not copied from Qwen;
- vectors, raw prompts, raw model JSON, and absolute paths remain hidden.

The context can contain passages Qwen did not cite. Retrieval selects candidates; the answer model
selects temporary IDs; the application validates and maps those IDs.

## 7. Inspect machine-readable provenance

```powershell
$checkpoint6Report = uv run file-agent ask `
  "Who owns the Atlas service dashboard?" `
  --db $checkpoint6Db `
  --json |
  ConvertFrom-Json

$checkpoint6Report | Select-Object `
  status, insufficient_evidence, decision_reason, `
  retrieved_count, context_count, context_characters, `
  generation_attempts, answer_model_returned

$checkpoint6Report.citations | Select-Object `
  citation_id, relative_path, chunk_index, start_char, end_char, similarity
```

Expected:

- `status` is normally `answered`;
- `decision_reason` is `grounded`;
- a citation points to `untrusted-instructions.md`;
- there is no `context` field because `--show-context` was not supplied;
- there are no absolute paths or vector coordinates.

The answer itself is document-derived output and can be sensitive when a personal index is used.

## 8. Prove deterministic refusal without calling Qwen

Use an intentionally impossible threshold:

```powershell
uv run file-agent ask `
  "What is the submarine launch code?" `
  --db $checkpoint6Db `
  --min-score 1.0

$LASTEXITCODE
```

Expected:

- `Answer refused safely`;
- `decision=no_retrieval_results`;
- the fixed refusal message is printed;
- `Sources: none`;
- `generation_attempts=0`;
- exit code `0`.

No evidence means Qwen is skipped. A refusal is a successful safety decision, not an infrastructure
failure.

## 9. Observe model-decided insufficient evidence

Allow retrieval but ask an unsupported question:

```powershell
uv run file-agent ask `
  "What is the office Wi-Fi password?" `
  --db $checkpoint6Db `
  --min-score 0
```

The desired result is the same fixed refusal with zero citations, normally with
`decision=model_insufficient_evidence`. A small local model can still make mistakes; Checkpoint 7
will measure this behavior across a repeatable evaluation suite.

The important invariant is that missing, duplicate, or invented citation IDs can never become real
source metadata. Python rejects them and returns the fixed refusal.

## 10. Inspect the prompt-injection boundary

```powershell
uv run file-agent ask `
  "Who owns the Atlas service dashboard?" `
  --db $checkpoint6Db `
  --show-context
```

The selected passage contains a malicious-looking instruction. It is deliberately stored as
untrusted evidence.

Verify that:

- the legitimate fact says Reliability Engineering;
- the application-generated source points to a real retrieved chunk;
- `attacker-selected.txt#chunk-99[0:10)` never appears as an application citation;
- filenames and offsets still come from validated SQLite metadata.

Prompt wording reduces the chance that Qwen follows the document instruction, but it is not a hard
security boundary. Application-side schema and provenance validation provide the deterministic
guarantees. Checkpoint 7 adds a canary-based regression measurement for answer-content injection.

## 11. Prove the index remains read-only

```powershell
$checkpoint6HashBefore = (Get-FileHash $checkpoint6Db -Algorithm SHA256).Hash
$checkpoint6TimeBefore = (Get-Item $checkpoint6Db).LastWriteTimeUtc

uv run file-agent ask `
  "How many learning credits are available?" `
  --db $checkpoint6Db

$checkpoint6HashAfter = (Get-FileHash $checkpoint6Db -Algorithm SHA256).Hash
$checkpoint6TimeAfter = (Get-Item $checkpoint6Db).LastWriteTimeUtc

$checkpoint6HashBefore -ceq $checkpoint6HashAfter
$checkpoint6TimeBefore -eq $checkpoint6TimeAfter
```

Expected: both expressions return `True`.

## 12. Distinguish validation and operational failures

Invalid options fail before model inference:

```powershell
uv run file-agent ask "question" --db $checkpoint6Db --top-k 0
$LASTEXITCODE

uv run file-agent ask "question" --db $checkpoint6Db --min-score 1.1
$LASTEXITCODE

uv run file-agent ask " " --db $checkpoint6Db
$LASTEXITCODE
```

Expected exit code: `2` for each command.

Stop Ollama temporarily and run `ask` against the valid index. Expected exit code: `1` with a safe
local connection error. Restart Ollama afterward.

Malformed structured model output is retried exactly once and then becomes exit code `1`. Automated
tests simulate this deterministically without exposing the raw response.

### Ollama schema compatibility learned during implementation

The installed Ollama `0.32.6` grammar compiler accepts the strict object shape and `maxItems`, but
rejects Pydantic's `maxLength` keyword with HTTP 400. The application therefore removes only
`maxLength` from the schema sent to Ollama while retaining the 2,000-character check in Pydantic
after generation. This is an important production lesson: provider-side structured output narrows
model behavior, but application-side validation remains authoritative.

## 13. Run automated quality checks

```powershell
uv run pytest tests/test_answering.py tests/test_ollama.py tests/test_retrieval.py tests/test_cli.py
uv run pytest --cov=local_file_agent --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

Normal tests use scripted vectors and model responses, so they do not require Ollama and are stable
in CI. The manual lab provides the separate live-model observation.

## Explain-back questions

Try answering these without consulting the implementation:

1. Why can Qwen return temporary ID `1` but not a filename?
2. Why is prompt injection not solved merely by saying “ignore document instructions”?
3. Why does malformed JSON become an operational error while an invalid citation becomes refusal?
4. Why does `ask` skip Qwen when retrieval returns zero passages?
5. What is the difference between `retrieved_count`, `context_count`, and citation count?
6. Why are complete chunks used instead of silently truncating context?
7. Which parts of this design would change for millions of chunks, and which trust boundaries should
   remain unchanged?

## What this checkpoint proves

- Retrieval and generation are separate, observable stages.
- Qwen sees only a bounded set of numbered untrusted passages.
- Structured output improves reliability but still requires application validation.
- Unsupported or semantically invalid answers fail closed.
- Real citations come only from trusted retrieval metadata.
- Raw context is hidden unless explicitly requested.
- The SQLite index remains read-only throughout answering.
