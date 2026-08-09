# Checkpoint 4 Manual Verification — SQLite Vector Index

## Goal

Verify durable multi-document persistence, schema ownership, portable Float32 blobs, read-only
reopening, deterministic provenance, safe replacement, privacy, and failure recovery.

Building requires local Ollama and `embeddinggemma`. Inspecting an existing index does not require
Ollama or Qwen.

## 1. Prepare the disposable database folder

Run from the repository root:

```powershell
uv sync --locked
ollama list
uv run file-agent doctor --skip-generation

$checkpoint4Lab = Join-Path $PWD ".data\manual-testing\checkpoint-4"
New-Item -ItemType Directory -Force $checkpoint4Lab
$checkpoint4Db = Join-Path $checkpoint4Lab "index.sqlite"
```

The database lives under Git-ignored `.data/`, outside the approved synthetic source folder.

## 2. Confirm the interfaces

```powershell
uv run file-agent index --help
uv run file-agent inspect-index --help
```

Important build options are `--source`, `--db`, `--chunk-size`, `--overlap`, `--batch-size`,
`--force`, and `--json`.

## 3. Build a live local index

```powershell
uv run file-agent index `
  --source examples/checkpoint-4/source `
  --db $checkpoint4Db
```

Expected behavior:

- three documents are accepted;
- chunk count equals embedding count;
- dimension is normally `768` for the baseline EmbeddingGemma model;
- vector format is `float32-le`;
- the report shows a 64-character corpus fingerprint;
- no document text, vector coordinates, or absolute path is printed.

Exact timings depend on hardware and whether the model was already loaded.

## 4. Reopen in another command without Ollama

```powershell
uv run file-agent inspect-index --db $checkpoint4Db
```

Expected:

```text
SQLite vector index is valid and was opened read-only.
Integrity: sqlite=ok, foreign_keys_valid=True
```

Stop Ollama and repeat `inspect-index`. It should still succeed because inspection reads only the
database. Restart Ollama before later rebuild experiments.

## 5. Inspect metadata as JSON

```powershell
$checkpoint4Report = uv run file-agent inspect-index `
  --db $checkpoint4Db `
  --json |
  ConvertFrom-Json

$checkpoint4Report | Select-Object `
  schema_version, embedding_model, embedding_dimension, vector_format, `
  document_count, chunk_count, embedding_count, corpus_fingerprint
```

Prove cardinality:

```powershell
$checkpoint4Report.chunk_count -eq $checkpoint4Report.embedding_count
```

Expected: `True`.

Prove privacy fields:

```powershell
$checkpoint4Report.content_included -eq $false
```

Expected: `True`.

## 6. Inspect the relational schema with Python

No external SQLite CLI is required:

```powershell
@'
import sqlite3
from pathlib import Path

database = Path(r"DB_PATH_PLACEHOLDER")
with sqlite3.connect(database) as connection:
    print("application_id:", connection.execute("PRAGMA application_id").fetchone()[0])
    print("user_version:", connection.execute("PRAGMA user_version").fetchone()[0])
    print("foreign_keys declared:", connection.execute("PRAGMA foreign_key_list(embeddings)").fetchall())
    for table in ("index_metadata", "documents", "chunks", "embeddings"):
        count = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(table, count)
'@.Replace("DB_PATH_PLACEHOLDER", $checkpoint4Db) | uv run python -
```

Expected: schema version `1`, one metadata row, three document rows, and equal chunk/embedding
counts.

## 7. Prove vectors use four bytes per dimension

```powershell
@'
import sqlite3
from pathlib import Path

database = Path(r"DB_PATH_PLACEHOLDER")
with sqlite3.connect(database) as connection:
    invalid = connection.execute(
        "SELECT count(*) FROM embeddings WHERE length(vector) != dimension * 4"
    ).fetchone()[0]
    print("invalid vector byte lengths:", invalid)
'@.Replace("DB_PATH_PLACEHOLDER", $checkpoint4Db) | uv run python -
```

Expected: `invalid vector byte lengths: 0`.

Do not print the vector BLOB. Embeddings are sensitive derived data and individual coordinates are
not useful for learning.

## 8. Prove inspection is read-only

```powershell
@'
import sqlite3
from pathlib import Path

database = Path(r"DB_PATH_PLACEHOLDER").resolve()
uri = f"{database.as_uri()}?mode=ro"
try:
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("DELETE FROM documents")
except sqlite3.OperationalError as error:
    print(type(error).__name__, str(error))
'@.Replace("DB_PATH_PLACEHOLDER", $checkpoint4Db) | uv run python -
```

Expected: an error containing `readonly`. Re-run `inspect-index`; it must remain valid.

## 9. Test deterministic provenance

```powershell
$checkpoint4Before = uv run file-agent inspect-index --db $checkpoint4Db --json |
  ConvertFrom-Json

uv run file-agent index `
  --source examples/checkpoint-4/source `
  --db $checkpoint4Db `
  --force

$checkpoint4After = uv run file-agent inspect-index --db $checkpoint4Db --json |
  ConvertFrom-Json

$checkpoint4Before.corpus_fingerprint -ceq $checkpoint4After.corpus_fingerprint
```

Expected: `True`. Build timestamps and database file hashes may differ.

## 10. Observe fingerprint changes safely

Copy the synthetic corpus into disposable storage:

```powershell
$checkpoint4DisposableSource = Join-Path $checkpoint4Lab "source"
Copy-Item examples/checkpoint-4/source $checkpoint4DisposableSource -Recurse -Force
$checkpoint4ChangedDb = Join-Path $checkpoint4Lab "changed.sqlite"

uv run file-agent index --source $checkpoint4DisposableSource --db $checkpoint4ChangedDb
$checkpoint4OriginalFingerprint = (
  uv run file-agent inspect-index --db $checkpoint4ChangedDb --json |
  ConvertFrom-Json
).corpus_fingerprint

Add-Content `
  (Join-Path $checkpoint4DisposableSource "learning-benefit.txt") `
  "`nA new synthetic learning rule."

uv run file-agent index `
  --source $checkpoint4DisposableSource `
  --db $checkpoint4ChangedDb `
  --force

$checkpoint4ChangedFingerprint = (
  uv run file-agent inspect-index --db $checkpoint4ChangedDb --json |
  ConvertFrom-Json
).corpus_fingerprint

$checkpoint4OriginalFingerprint -ne $checkpoint4ChangedFingerprint
```

Expected: `True`.

## 11. Test safe replacement failures

Without `--force`:

```powershell
uv run file-agent index --source examples/checkpoint-4/source --db $checkpoint4Db
$LASTEXITCODE
```

Expected exit code: `1`; the existing index remains valid.

Unrelated file protection:

```powershell
$checkpoint4Unrelated = Join-Path $checkpoint4Lab "important.db"
[System.IO.File]::WriteAllText($checkpoint4Unrelated, "not an app index")

uv run file-agent index `
  --source examples/checkpoint-4/source `
  --db $checkpoint4Unrelated `
  --force
$LASTEXITCODE

[System.IO.File]::ReadAllText($checkpoint4Unrelated)
```

Expected exit code: `1`; content remains `not an app index`.

Database inside the source boundary:

```powershell
uv run file-agent index `
  --source examples/checkpoint-4/source `
  --db examples/checkpoint-4/source/unsafe.sqlite
$LASTEXITCODE
```

Expected exit code: `1`, and no database is created there.

## 12. Test corruption detection on a disposable copy

```powershell
$checkpoint4Corrupt = Join-Path $checkpoint4Lab "corrupt.sqlite"
Copy-Item $checkpoint4Db $checkpoint4Corrupt -Force

@'
import sqlite3
from pathlib import Path

database = Path(r"DB_PATH_PLACEHOLDER")
with sqlite3.connect(database) as connection:
    connection.execute("UPDATE embeddings SET vector = x'00000000' WHERE rowid = 1")
'@.Replace("DB_PATH_PLACEHOLDER", $checkpoint4Corrupt) | uv run python -

uv run file-agent inspect-index --db $checkpoint4Corrupt
$LASTEXITCODE
```

Expected exit code: `1` with a controlled validation error. Never corrupt the primary learning
index or personal data for an experiment.

## 13. Run automated quality checks

```powershell
uv run pytest tests/test_storage.py tests/test_indexing.py tests/test_cli.py
uv run pytest --cov=local_file_agent --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

All checks must pass. Normal automated tests use synthetic vectors and do not require Ollama.

## What this checkpoint proves

- Documents, exact chunks, and compatible embeddings survive process boundaries.
- SQLite schema identity and provenance are explicit and versioned.
- Float32 vectors have a portable validated byte representation.
- New indexes are transactionally built and validated before atomic publication.
- Failed or unauthorized replacement cannot damage an existing valid index.
- Read-only inspection detects structural, relational, metadata, hash, count, and vector corruption.
- Normal output remains useful without leaking stored content or vector coordinates.
