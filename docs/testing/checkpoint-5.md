# Checkpoint 5 Manual Verification — Read-Only Vector Search

## Goal

Verify semantic paraphrase retrieval, score filtering, deterministic top-K ranking, overlap
suppression, trusted citations, index immutability, privacy, and controlled failure behavior.

Search uses EmbeddingGemma but never calls Qwen.

## 1. Prepare a disposable index

Run from the repository root:

```powershell
uv sync --locked
ollama list
uv run file-agent doctor --skip-generation

$checkpoint5Lab = Join-Path $PWD ".data\manual-testing\checkpoint-5"
New-Item -ItemType Directory -Force $checkpoint5Lab
$checkpoint5Db = Join-Path $checkpoint5Lab "index.sqlite"

uv run file-agent index `
  --source examples/checkpoint-5/source `
  --db $checkpoint5Db
```

If the index already exists, use a new filename or deliberately rebuild it with `--force`.

## 2. Confirm the search interface

```powershell
uv run file-agent search --help
```

Important inputs:

- positional question;
- required `--db`;
- `--top-k`, default `5`;
- `--min-score`, default `0.30`;
- `--show-text`;
- `--json`.

## 3. Run a direct wording search

```powershell
uv run file-agent search `
  "Employees rotate credentials every ninety days" `
  --db $checkpoint5Db
```

Expected:

- `credential-policy.md` normally ranks first;
- every displayed score is at least `0.30`;
- citations contain relative paths, chunk numbers, and character ranges;
- no question text, passage text, vector coordinates, or absolute path is printed.

Exact scores depend on the installed model/runtime version.

## 4. Test semantic paraphrasing

```powershell
uv run file-agent search `
  "How frequently do staff passwords need to be changed?" `
  --db $checkpoint5Db
```

The query does not copy the document wording, but `credential-policy.md` should normally rank near
the top because the meanings are related.

Compare with exact text search:

```powershell
Select-String `
  -Path examples/checkpoint-5/source/* `
  -Pattern "staff passwords" `
  -SimpleMatch
```

Expected: no exact keyword match. This illustrates the problem semantic embeddings solve.

## 5. Reveal selected synthetic evidence deliberately

```powershell
uv run file-agent search `
  "How frequently do staff passwords need to be changed?" `
  --db $checkpoint5Db `
  --show-text
```

Expected: a warning followed by only the selected synthetic chunks. Raw vectors, filtered chunks,
and the exact query are still not printed.

Use `--show-text` carefully with personal indexes because terminal history and screenshots can
retain retrieved content.

## 6. Inspect machine-readable results

```powershell
$checkpoint5Report = uv run file-agent search `
  "How much professional-development funding does a worker receive?" `
  --db $checkpoint5Db `
  --json |
  ConvertFrom-Json

$checkpoint5Report | Select-Object `
  stored_embedding_model, embedding_dimension, top_k, min_score, `
  indexed_chunk_count, above_threshold_count, suppressed_count, result_count

$checkpoint5Report.results | Select-Object `
  rank, citation, relative_path, chunk_index, start_char, end_char, similarity
```

Expected: `learning-benefit.txt` normally ranks first and no result contains a `text` field.

## 7. Compare top-K values

```powershell
foreach ($checkpoint5TopK in 1, 3, 5) {
    $report = uv run file-agent search `
      "What happens during a severe production incident?" `
      --db $checkpoint5Db `
      --top-k $checkpoint5TopK `
      --min-score 0 `
      --json |
      ConvertFrom-Json

    [PSCustomObject]@{
        TopK = $checkpoint5TopK
        Results = $report.result_count
        BestSource = $report.results[0].relative_path
    }
}
```

`top-k` limits result count; it does not guarantee that every returned result is useful.

## 8. Compare score thresholds

```powershell
foreach ($checkpoint5Threshold in 0.0, 0.3, 0.6) {
    $report = uv run file-agent search `
      "Where are the gardening tools stored?" `
      --db $checkpoint5Db `
      --top-k 5 `
      --min-score $checkpoint5Threshold `
      --json |
      ConvertFrom-Json

    [PSCustomObject]@{
        MinimumScore = $checkpoint5Threshold
        AboveThreshold = $report.above_threshold_count
        Results = $report.result_count
    }
}
```

Increasing the threshold normally reduces distractors but may also remove a valid paraphrase.

## 9. Observe heavy-overlap suppression

Build a second index with deliberately excessive overlap:

```powershell
$checkpoint5OverlapDb = Join-Path $checkpoint5Lab "overlap.sqlite"

uv run file-agent index `
  --source examples/checkpoint-5/source `
  --db $checkpoint5OverlapDb `
  --chunk-size 350 `
  --overlap 300
```

Search the long recovery document:

```powershell
uv run file-agent search `
  "What does the Silver Anchor rollback procedure do?" `
  --db $checkpoint5OverlapDb `
  --top-k 10 `
  --min-score 0
```

Adjacent full chunks repeat about `300 / 350 = 85.7%`, above the fixed `80%` suppression ratio.
Expected: `suppressed` is greater than zero, and near-identical adjacent ranges do not consume every
result position.

## 10. Verify a valid zero-result outcome

Use an intentionally strict threshold:

```powershell
uv run file-agent search `
  "What is the submarine launch code?" `
  --db $checkpoint5Db `
  --min-score 1.0

$LASTEXITCODE
```

Expected:

- `No chunks met min_score=1.0.`
- exit code `0`.

This means search completed but found insufficient evidence. It is not an infrastructure failure.

## 11. Prove the database remains unchanged

```powershell
$checkpoint5HashBefore = (Get-FileHash $checkpoint5Db -Algorithm SHA256).Hash
$checkpoint5TimeBefore = (Get-Item $checkpoint5Db).LastWriteTimeUtc

uv run file-agent search `
  "How many learning credits are available?" `
  --db $checkpoint5Db

$checkpoint5HashAfter = (Get-FileHash $checkpoint5Db -Algorithm SHA256).Hash
$checkpoint5TimeAfter = (Get-Item $checkpoint5Db).LastWriteTimeUtc

$checkpoint5HashBefore -ceq $checkpoint5HashAfter
$checkpoint5TimeBefore -eq $checkpoint5TimeAfter
```

Expected: both expressions return `True`.

## 12. Prove the index model overrides current model configuration

```powershell
$checkpoint5PreviousModel = $env:FILE_AGENT_EMBEDDING_MODEL
$env:FILE_AGENT_EMBEDDING_MODEL = "some-uninstalled-model"

uv run file-agent search `
  "How many learning credits are available?" `
  --db $checkpoint5Db

$env:FILE_AGENT_EMBEDDING_MODEL = $checkpoint5PreviousModel
```

Expected: search still uses and reports the model stored in the index. The uninstalled environment
model is ignored for retrieval compatibility.

## 13. Test invalid options

```powershell
uv run file-agent search "question" --db $checkpoint5Db --top-k 0
$LASTEXITCODE

uv run file-agent search "question" --db $checkpoint5Db --min-score 1.1
$LASTEXITCODE

uv run file-agent search " " --db $checkpoint5Db
$LASTEXITCODE
```

Expected: each command returns exit code `2` before model inference.

## 14. Distinguish database and Ollama failures

Create a corrupt disposable database:

```powershell
$checkpoint5CorruptDb = Join-Path $checkpoint5Lab "corrupt.sqlite"
[System.IO.File]::WriteAllText($checkpoint5CorruptDb, "not sqlite")

uv run file-agent search "question" --db $checkpoint5CorruptDb
$LASTEXITCODE
```

Expected exit code: `1`; validation fails before a query embedding is requested.

Stop Ollama and search the valid index. Expected exit code: `1` with a safe local connection error.
Restart Ollama afterward. These are operational failures, unlike a valid zero-match result.

## 15. Run automated quality checks

```powershell
uv run pytest tests/test_retrieval.py tests/test_storage.py tests/test_cli.py
uv run pytest --cov=local_file_agent --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

All checks must pass. Deterministic tests use synthetic vectors and do not require Ollama.

## What this checkpoint proves

- Questions use the exact model contract recorded by the index.
- Brute-force cosine search retrieves semantic paraphrases without exact wording.
- Thresholding and top-K are separate, observable policies.
- Stable tie-breakers make result order reproducible.
- Heavy same-document overlaps cannot dominate selected evidence.
- Citations come from validated application metadata.
- Zero evidence is distinct from infrastructure failure.
- Search cannot modify the SQLite index and hides passage text by default.
