# Checkpoint 3 Manual Verification — Local Embeddings

## Goal

Verify local document/query embedding, batching, detected dimensions, cosine ranking, privacy-safe
output, clear failures, and deterministic application behavior around a live Ollama model.

This guide uses only committed synthetic content. Ollama must be running with `embeddinggemma`
installed; Qwen is not used by `inspect-embeddings`.

## 1. Confirm prerequisites

```powershell
uv sync --locked
ollama list
uv run file-agent doctor --skip-generation
uv run file-agent inspect-embeddings --help
```

Expected:

- `embeddinggemma` appears in `ollama list`.
- The doctor embedding smoke check passes and normally reports dimension `768`.
- Help lists `--source`, `--document`, `--query`, `--batch-size`, `--top-k`, `--show-text`, and
  `--json`.

The dimension is detected, not hard-coded, so another compatible model may report a different
value.

## 2. Inspect the synthetic source safely

```powershell
uv run file-agent scan --source examples/checkpoint-3/source
```

Expected: one accepted file named `semantic-lab.md` and no skipped entries.

## 3. Run a paraphrased semantic query

```powershell
uv run file-agent inspect-embeddings `
  --source examples/checkpoint-3/source `
  --document semantic-lab.md `
  --query "How much funding can a worker use for career development?" `
  --chunk-size 450 `
  --overlap 0 `
  --top-k 3
```

Expected behavior:

- Model is `embeddinggemma` or the equivalent explicit tag `embeddinggemma:latest`.
- Dimension is normally `768`.
- Chunk and vector counts are equal.
- Each result has rank, chunk index, score, offsets, norm, and content hash.
- The learning-budget passage should normally rank ahead of the unrelated garden passage.
- Neither the query nor document text appears in output.
- No raw 768-number vector is printed.

Exact scores and timings may vary by model version, hardware, model residency, and Ollama runtime.
Judge the relative ordering, not a memorized score.

## 4. Reveal synthetic text deliberately

```powershell
uv run file-agent inspect-embeddings `
  --source examples/checkpoint-3/source `
  --document semantic-lab.md `
  --query "How much funding can a worker use for career development?" `
  --chunk-size 450 `
  --overlap 0 `
  --top-k 3 `
  --show-text
```

Expected: a privacy warning followed by the exact query and ranked chunk text. Raw vector
coordinates still do not appear. Use this option only with synthetic or explicitly approved data;
terminal history and screenshots may retain the text.

## 5. Inspect machine-readable metadata

```powershell
$checkpoint3Report = uv run file-agent inspect-embeddings `
  --source examples/checkpoint-3/source `
  --document semantic-lab.md `
  --query "How much funding can a worker use for career development?" `
  --chunk-size 450 `
  --overlap 0 `
  --top-k 3 `
  --json |
  ConvertFrom-Json

$checkpoint3Report | Select-Object requested_model, returned_model, dimension, chunk_count, vector_count, batch_count
$checkpoint3Report.results | Select-Object rank, chunk_index, similarity, start_char, end_char
```

Prove the cardinality invariant:

```powershell
$checkpoint3Report.chunk_count -eq $checkpoint3Report.vector_count
```

Expected: `True`.

Prove content is excluded:

```powershell
$null -eq $checkpoint3Report.query_text
($checkpoint3Report.results | Where-Object { $null -ne $_.text }).Count -eq 0
```

Expected: both expressions return `True`.

## 6. Compare direct wording, paraphrase, and unrelated queries

```powershell
$checkpoint3Queries = @(
    "The annual professional-development allowance is 1,200 credits",
    "How much funding can a worker use for career development?",
    "Where are the gardening tools stored?"
)

foreach ($checkpoint3Query in $checkpoint3Queries) {
    $report = uv run file-agent inspect-embeddings `
      --source examples/checkpoint-3/source `
      --document semantic-lab.md `
      --query $checkpoint3Query `
      --chunk-size 450 `
      --overlap 0 `
      --top-k 1 `
      --json |
      ConvertFrom-Json

    [PSCustomObject]@{
        QueryCharacters = $report.query_characters
        BestChunk = $report.results[0].chunk_index
        BestScore = $report.results[0].similarity
    }
}
```

Interpretation:

- Direct wording and its paraphrase should normally select the learning-budget chunk.
- The gardening query should select the garden chunk.
- A paraphrase can be retrieved without exact keyword equality; that is the main value of semantic
  embeddings.

## 7. Observe batching

Use one prompt per request:

```powershell
uv run file-agent inspect-embeddings `
  --source examples/checkpoint-3/source `
  --document semantic-lab.md `
  --query "What happens after a severe incident?" `
  --chunk-size 300 `
  --overlap 50 `
  --batch-size 1 `
  --top-k 2
```

Then batch several prompts:

```powershell
uv run file-agent inspect-embeddings `
  --source examples/checkpoint-3/source `
  --document semantic-lab.md `
  --query "What happens after a severe incident?" `
  --chunk-size 300 `
  --overlap 50 `
  --batch-size 8 `
  --top-k 2
```

Compare batch count and timings. Do not infer a general performance conclusion from one cold and
one warm run; reverse the order and repeat before drawing a conclusion.

## 8. Check repeatability correctly

Run the same JSON command twice and compare stable fields:

```powershell
function Get-Checkpoint3StableResult {
    $report = uv run file-agent inspect-embeddings `
      --source examples/checkpoint-3/source `
      --document semantic-lab.md `
      --query "How much funding can a worker use for career development?" `
      --chunk-size 450 `
      --overlap 0 `
      --top-k 3 `
      --json |
      ConvertFrom-Json

    $report.results | Select-Object rank, chunk_index, start_char, end_char, content_sha256, similarity
}

$checkpoint3First = Get-Checkpoint3StableResult | ConvertTo-Json -Compress
$checkpoint3Second = Get-Checkpoint3StableResult | ConvertTo-Json -Compress
$checkpoint3First -ceq $checkpoint3Second
```

Expected: normally `True` for the same model/runtime. Timings are deliberately excluded because
they are not deterministic. Minor numerical changes after a model or runtime upgrade are possible.

## 9. Test safe failures

Invalid batch size:

```powershell
uv run file-agent inspect-embeddings --source examples/checkpoint-3/source --document semantic-lab.md --query "test" --batch-size 0
$LASTEXITCODE
```

Expected exit code: `2`.

Empty query:

```powershell
uv run file-agent inspect-embeddings --source examples/checkpoint-3/source --document semantic-lab.md --query " "
$LASTEXITCODE
```

Expected exit code: `2`.

Unknown document:

```powershell
uv run file-agent inspect-embeddings --source examples/checkpoint-3/source --document missing.md --query "test"
$LASTEXITCODE
```

Expected exit code: `1`, without a model call.

Stop Ollama temporarily and rerun a valid command. Expected exit code: `1` with a safe local
connection error, not document text or an HTTP response body. Restart Ollama afterward.

## 10. Run automated quality checks

```powershell
uv run pytest tests/test_embeddings.py tests/test_ollama.py tests/test_cli.py
uv run pytest --cov=local_file_agent --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

All checks must pass. Ordinary tests mock Ollama and do not send test data to a model.

## What this checkpoint proves

- Trusted chunks and questions become validated local Float32 vectors.
- Prompt roles for passages and questions are explicit and versioned.
- Batching preserves one-vector-per-chunk order and records useful timings.
- Model, vector count, dimension, finiteness, and non-zero norms fail closed.
- Cosine similarity can rank a paraphrase near relevant evidence.
- Debugging output remains useful without exposing content or raw vectors by default.
- No embedding index exists yet; persistence is the purpose of Checkpoint 4.
