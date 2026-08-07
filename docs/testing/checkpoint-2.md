# Checkpoint 2 Manual Verification — Deterministic Chunking

## Goal

Verify deterministic boundary selection, exact source offsets, controlled overlap, stable document
identity, privacy-aware inspection, forward progress, and failure handling.

Checkpoint 2 uses pure Python string processing and does not require Ollama.

## 1. Confirm the command interface

```powershell
uv sync --locked
uv run file-agent inspect-chunks --help
```

Important options:

- `--source`: explicitly approved folder.
- `--document`: accepted relative document path.
- `--chunk-size`: target characters per chunk.
- `--overlap`: repeated characters between adjacent chunks.
- `--show-text`: explicit content-display opt-in.
- `--json`: machine-readable report.

## 2. Scan the synthetic Checkpoint 2 corpus

```powershell
uv run file-agent scan --source examples/checkpoint-2/source
```

Expected:

```text
Accepted files: 2
Skipped entries: 0
```

The accepted files are `long-rag-note.md` and `short-note.txt`.

## 3. Inspect default chunk metadata

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md
```

Expected configuration:

```text
chunk_size=1200, overlap=200, boundary_window=300
```

Expected boundaries for the committed sample:

```text
Chunk 0 [0:1016)
Chunk 1 [816:2012)    overlap=200
Chunk 2 [1812:2961)   overlap=200
Chunk 3 [2761:3815)   overlap=200
Chunk 4 [3615:4533)   overlap=200
```

The chunk size is a target, not a fixed cut. Natural boundaries may produce smaller chunks.

## 4. Inspect JSON without content

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md `
  --json
```

Each chunk contains its index, `[start_char:end_char)` range, character count, overlap, and hash.
There must be no `text` field while `content_included` is false.

## 5. Deliberately inspect chunk text

Use only synthetic or explicitly approved documents:

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md `
  --show-text
```

Expected:

- A warning that exact text follows.
- Overlapping text is marked separately from new text.
- Chunks after the first report 200 repeated characters with default settings.

Terminal history, screenshots, and logs may retain `--show-text` output.

## 6. Verify deterministic output

```powershell
$checkpoint2Run1 = (
    uv run file-agent inspect-chunks `
      --source examples/checkpoint-2/source `
      --document long-rag-note.md `
      --json
) -join "`n"

$checkpoint2Run2 = (
    uv run file-agent inspect-chunks `
      --source examples/checkpoint-2/source `
      --document long-rag-note.md `
      --json
) -join "`n"

$checkpoint2Run1 -ceq $checkpoint2Run2
```

Expected: `True`.

## 7. Prove every offset maps to the source

Capture chunk text explicitly:

```powershell
$checkpoint2Report = uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md `
  --json `
  --show-text |
  ConvertFrom-Json
```

Read the source using the same newline normalization contract:

```powershell
$checkpoint2SourcePath = Join-Path $PWD "examples\checkpoint-2\source\long-rag-note.md"
$checkpoint2SourceText = [System.IO.File]::ReadAllText($checkpoint2SourcePath)
$checkpoint2SourceText = $checkpoint2SourceText.Replace("`r`n", "`n").Replace("`r", "`n")
```

Verify every chunk:

```powershell
$checkpoint2AllSlicesMatch = $true

foreach ($chunk in $checkpoint2Report.chunks) {
    $length = [int]$chunk.end_char - [int]$chunk.start_char
    $sourceSlice = $checkpoint2SourceText.Substring([int]$chunk.start_char, $length)

    if ($sourceSlice -cne $chunk.text) {
        $checkpoint2AllSlicesMatch = $false
        Write-Host "Mismatch in chunk $($chunk.chunk_index)"
    }
}

$checkpoint2AllSlicesMatch
```

Expected: `True`.

This mechanically proves:

```text
document.text[start_char:end_char] == chunk.text
```

## 8. Prove adjacent overlap is exact

```powershell
$checkpoint2AllOverlapsMatch = $true

for ($index = 1; $index -lt $checkpoint2Report.chunks.Count; $index++) {
    $previous = $checkpoint2Report.chunks[$index - 1]
    $current = $checkpoint2Report.chunks[$index]
    $overlap = [int]$current.overlap_with_previous

    $previousEnding = $previous.text.Substring($previous.text.Length - $overlap)
    $currentBeginning = $current.text.Substring(0, $overlap)

    if ($previousEnding -cne $currentBeginning) {
        $checkpoint2AllOverlapsMatch = $false
        Write-Host "Overlap mismatch in chunk $index"
    }
}

$checkpoint2AllOverlapsMatch
```

Expected: `True`.

## 9. Compare overlap policies

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md `
  --chunk-size 500 `
  --overlap 0

uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md `
  --chunk-size 500 `
  --overlap 100

uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md `
  --chunk-size 500 `
  --overlap 400
```

Observe how increasing overlap increases repeated context and chunk count while reducing the amount
of new information in each later chunk.

## 10. Compare chunk sizes

Repeat inspection with these configurations:

```powershell
uv run file-agent inspect-chunks --source examples/checkpoint-2/source --document long-rag-note.md --chunk-size 250 --overlap 50
uv run file-agent inspect-chunks --source examples/checkpoint-2/source --document long-rag-note.md --chunk-size 500 --overlap 100
uv run file-agent inspect-chunks --source examples/checkpoint-2/source --document long-rag-note.md --chunk-size 1500 --overlap 200
```

- Small chunks isolate details but may lose context.
- Large chunks preserve context but mix topics.
- The best policy must eventually be chosen using retrieval evaluation.

## 11. Test hard-boundary fallback

Create an ignored disposable lab and a file with no natural boundaries:

```powershell
$checkpoint2Lab = Join-Path $PWD ".data\manual-testing\checkpoint-2"
New-Item -ItemType Directory -Force $checkpoint2Lab

$checkpoint2NoBoundaries = Join-Path $checkpoint2Lab "no-boundaries.txt"
$checkpoint2Utf8 = [System.Text.UTF8Encoding]::new($false)

[System.IO.File]::WriteAllText(
    $checkpoint2NoBoundaries,
    ("x" * 2500),
    $checkpoint2Utf8
)
```

Inspect it:

```powershell
uv run file-agent inspect-chunks `
  --source $checkpoint2Lab `
  --document no-boundaries.txt `
  --chunk-size 1000 `
  --overlap 100
```

Expected boundaries:

```text
[0:1000)
[900:1900)
[1800:2500)
```

This proves that missing natural boundaries cannot stop progress.

## 12. Test empty-document behavior

```powershell
$checkpoint2Empty = Join-Path $checkpoint2Lab "empty.txt"
[System.IO.File]::WriteAllText($checkpoint2Empty, "")

uv run file-agent inspect-chunks `
  --source $checkpoint2Lab `
  --document empty.txt
```

Expected: zero document characters and zero chunks.

## 13. Verify stable identity versus changing content

```powershell
$checkpoint2IdentityFile = Join-Path $checkpoint2Lab "identity.md"

[System.IO.File]::WriteAllText(
    $checkpoint2IdentityFile,
    "Version one",
    $checkpoint2Utf8
)

$checkpoint2Before = uv run file-agent scan --source $checkpoint2Lab --json |
    ConvertFrom-Json
$checkpoint2BeforeDocument = $checkpoint2Before.accepted |
    Where-Object relative_path -eq "identity.md"

[System.IO.File]::WriteAllText(
    $checkpoint2IdentityFile,
    "Version two",
    $checkpoint2Utf8
)

$checkpoint2After = uv run file-agent scan --source $checkpoint2Lab --json |
    ConvertFrom-Json
$checkpoint2AfterDocument = $checkpoint2After.accepted |
    Where-Object relative_path -eq "identity.md"
```

Identity remains stable:

```powershell
$checkpoint2BeforeDocument.document_id -eq $checkpoint2AfterDocument.document_id
```

Expected: `True`.

Content hash changes:

```powershell
$checkpoint2BeforeDocument.content_sha256 -ne $checkpoint2AfterDocument.content_sha256
```

Expected: `True`.

## 14. Test invalid inputs and exit codes

Overlap equal to chunk size:

```powershell
uv run file-agent inspect-chunks --source examples/checkpoint-2/source --document long-rag-note.md --chunk-size 500 --overlap 500
$LASTEXITCODE
```

Expected exit code: `2`.

Traversal-style document selector:

```powershell
uv run file-agent inspect-chunks --source examples/checkpoint-2/source --document "..\outside.md"
$LASTEXITCODE
```

Expected exit code: `2`.

Unknown accepted document:

```powershell
uv run file-agent inspect-chunks --source examples/checkpoint-2/source --document missing.md
$LASTEXITCODE
```

Expected exit code: `1`.

## 15. Run automated verification

```powershell
uv run pytest tests/test_chunking.py tests/test_cli.py tests/test_ingestion.py
uv run pytest --cov=local_file_agent --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

All checks must pass. The existing Windows symlink test may be skipped when symlink creation is not
permitted.

## What this checkpoint proves

- Chunks are exact source substrings, not rewritten summaries.
- Boundary preference is deterministic and inspectable.
- Overlap preserves controlled boundary context.
- Invalid overlap cannot create an infinite loop.
- Document identity is separate from mutable content identity.
- Debugging visibility is available without exposing text by default.
- Character-based chunking is a measurable baseline for future embedding evaluation.
