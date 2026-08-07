# Checkpoint 1 Manual Verification — Secure File Discovery and Parsing

## Goal

Verify that one explicitly approved folder is scanned deterministically, supported files become
trusted normalized documents, unsafe entries are rejected with safe reasons, document contents
stay private, and source files remain unchanged.

Run every command from the repository root.

## 1. Confirm the command interface

```powershell
uv sync --locked
uv run file-agent scan --help
```

Confirm that `--source` is required. There is no implicit current-directory or home-directory scan.

## 2. Scan the committed synthetic corpus

```powershell
uv run file-agent scan --source examples/checkpoint-1/source
```

Expected summary:

```text
Accepted files: 2
Skipped entries: 1
```

Expected accepted files:

```text
notes/meeting-notes.txt
project-overview.md
```

Expected rejection:

```text
unsupported.csv [unsupported_extension]
```

The output contains relative paths, sizes, character counts, document IDs, and hashes—but no
document text or normal absolute source path.

## 3. Inspect the JSON contract

```powershell
uv run file-agent scan --source examples/checkpoint-1/source --json
```

Inspect these sections:

- `accepted`: privacy-safe metadata for trusted documents.
- `skipped`: relative path and stable reason code.
- `summary`: accepted/skipped counts and counts grouped by reason.
- `max_file_size_bytes`: the 5 MiB policy expressed as bytes.

There must be no `text` or `content` field.

## 4. Verify deterministic output

```powershell
$checkpoint1Run1 = (
    uv run file-agent scan --source examples/checkpoint-1/source --json
) -join "`n"

$checkpoint1Run2 = (
    uv run file-agent scan --source examples/checkpoint-1/source --json
) -join "`n"

$checkpoint1Run1 -ceq $checkpoint1Run2
```

Expected:

```text
True
```

The same folder state produces the same ordering, counts, document IDs, and hashes.

## 5. Create a disposable security lab

All lab data is placed under `.data/`, which is ignored by Git.

```powershell
$checkpoint1Lab = Join-Path $PWD ".data\manual-testing\checkpoint-1"
$checkpoint1Nested = Join-Path $checkpoint1Lab "nested"
$checkpoint1Generated = Join-Path $checkpoint1Lab ".git"

New-Item -ItemType Directory -Force $checkpoint1Nested
New-Item -ItemType Directory -Force $checkpoint1Generated
```

Create two valid files:

```powershell
[System.IO.File]::WriteAllText(
    (Join-Path $checkpoint1Lab "valid.md"),
    "# Valid note`n`nThis is approved synthetic content."
)

[System.IO.File]::WriteAllText(
    (Join-Path $checkpoint1Nested "notes.txt"),
    "This nested text file is valid."
)
```

Create unsupported, hidden, generated, binary, invalid-UTF-8, and oversized inputs:

```powershell
[System.IO.File]::WriteAllText(
    (Join-Path $checkpoint1Lab "unsupported.csv"),
    "name,status"
)

[System.IO.File]::WriteAllText(
    (Join-Path $checkpoint1Lab ".secret.txt"),
    "THIS-PRIVATE-MARKER-MUST-NOT-APPEAR"
)

[System.IO.File]::WriteAllText(
    (Join-Path $checkpoint1Generated "internal.txt"),
    "This directory must not be traversed."
)

[System.IO.File]::WriteAllBytes(
    (Join-Path $checkpoint1Lab "binary.txt"),
    [byte[]](65, 0, 66)
)

[System.IO.File]::WriteAllBytes(
    (Join-Path $checkpoint1Lab "invalid.md"),
    [byte[]](0xFF, 0xFE)
)

[System.IO.File]::WriteAllBytes(
    (Join-Path $checkpoint1Lab "too-large.txt"),
    [byte[]]::new((5 * 1024 * 1024) + 1)
)
```

## 6. Scan the security lab

```powershell
uv run file-agent scan --source $checkpoint1Lab
```

Expected:

```text
Accepted files: 2
Skipped entries: 6
```

Expected reason codes include:

- `unsupported_extension`
- `hidden_entry`
- `excluded_directory`
- `binary_content`
- `invalid_utf8`
- `file_too_large`

The private marker written to `.secret.txt` must not appear in output.

## 7. Verify source files are read-only

```powershell
$checkpoint1ValidFile = Join-Path $checkpoint1Lab "valid.md"
$checkpoint1HashBefore = (Get-FileHash $checkpoint1ValidFile).Hash

uv run file-agent scan --source $checkpoint1Lab

$checkpoint1HashAfter = (Get-FileHash $checkpoint1ValidFile).Hash
$checkpoint1HashBefore -eq $checkpoint1HashAfter
```

Expected:

```text
True
```

The scan does not rewrite, rename, or persist changes to source files.

## 8. Test source-root failures

Nonexistent root:

```powershell
uv run file-agent scan --source ".data\folder-that-does-not-exist"
$LASTEXITCODE
```

Expected exit code: `1`.

A file supplied instead of a folder:

```powershell
uv run file-agent scan --source $checkpoint1ValidFile
$LASTEXITCODE
```

Expected exit code: `1`.

Missing the required option:

```powershell
uv run file-agent scan
$LASTEXITCODE
```

Expected exit code: `2`.

This distinguishes runtime rejection from incorrect CLI usage.

## 9. Run automated verification

```powershell
uv run pytest tests/test_ingestion.py tests/test_cli.py
uv run pytest --cov=local_file_agent --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

The symlink test may be skipped when Windows does not permit the current process to create symbolic
links. Platform-independent containment tests must still pass.

## What this checkpoint proves

- The filesystem is treated as an untrusted boundary.
- Explicit approval and path containment implement least privilege.
- Valid files continue processing when unrelated entries are rejected.
- Normalization and hashes create a canonical document representation.
- Operational reports remain useful without exposing document content.
- Downstream code receives trusted `Document` values instead of arbitrary paths.
