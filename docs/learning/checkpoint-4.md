# Checkpoint 4 — SQLite Vector Index

## What we built

Checkpoint 4 turns the in-memory output from Checkpoint 3 into a durable local index. The index
contains document provenance, exact chunks, and portable Float32 embedding blobs. It can be closed,
reopened in another process, and validated without calling Ollama.

```powershell
uv run file-agent index `
  --source examples/checkpoint-4/source `
  --db .data/manual-testing/checkpoint-4/index.sqlite

uv run file-agent inspect-index `
  --db .data/manual-testing/checkpoint-4/index.sqlite
```

This checkpoint deliberately does not search the vectors. Persistence and retrieval are separate
responsibilities, which makes failures easier to locate.

## Mental model

```text
Approved documents
        |
deterministic chunks
        |
validated embeddings
        |
temporary SQLite database
        |
transaction + constraints
        |
close and reopen read-only
        |
integrity + record validation
        |
atomic replacement
        v
durable app-owned index
```

The critical engineering idea is that a newly created file is not automatically a valid index.
The application trusts it only after reopening and validating the same representation that future
search will read.

## Relational design

The schema has four `STRICT` tables:

- `index_metadata` records schema version, models, prompt strategy, dimension, vector format,
  chunking policy, counts, timestamp, and corpus fingerprint.
- `documents` records stable identity, safe relative path, sizes, and source-content hash.
- `chunks` records exact text, zero-based index, offsets, and chunk hash.
- `embeddings` stores exactly one dimension and Float32 blob for each chunk.

Documents and chunks are separated because one document owns many chunks. Embeddings are separated
from chunks because the numerical representation has its own format and compatibility lifecycle.
Composite primary and foreign keys enforce the one-vector-per-chunk relationship.

Complete document text is not duplicated. Future retrieval needs chunk text, while document-level
metadata is enough for provenance and staleness checks.

## Database ownership and schema versions

SQLite files can belong to any application. Before reading or replacing a file, this application
checks:

- SQLite `application_id` equals the File Agent identifier;
- SQLite `user_version` equals schema version `1`;
- exactly the expected tables and ordered columns exist;
- the metadata table contains exactly one compatible row.

This prevents `--force` from overwriting an unrelated SQLite database merely because it has a
`.sqlite` extension.

Schema versioning creates an explicit future migration boundary. If version 2 changes vector
representation or tables, version 1 readers must not silently guess how to interpret it.

## Transactions versus atomic replacement

A transaction protects consistency inside one SQLite database. If an insert fails, SQLite can roll
back the transaction. A transaction alone does not protect an existing good index from a broken
rebuild that writes directly into the same file.

The build therefore uses two levels of safety:

1. Build all schema and rows inside one transaction in a new temporary sibling file.
2. Close and reopen that file read-only, then validate it completely.
3. Use `os.replace` only after validation succeeds.

The temporary file is placed beside the final database so replacement stays on one filesystem. If
embedding, insertion, validation, or replacement fails, the exact temporary file is cleaned up and
the previous index remains unchanged.

Existing indexes require `--force`. Even with `--force`, the old target must first validate as an
app-owned index. Corrupt or unrelated targets must be moved or removed deliberately by the owner.

## Float32 BLOB representation

SQLite has no native vector type in this baseline. Each vector is stored as contiguous
little-endian 32-bit floating-point bytes:

```text
768 coordinates × 4 bytes = 3,072 bytes per EmbeddingGemma vector
```

The format is named `float32-le` in metadata. Explicit endianness makes the byte representation
portable instead of depending on the CPU that created it.

When loading, the application validates:

- blob byte length equals `dimension × 4`;
- decoded dimension matches index metadata;
- all values remain finite;
- norm is non-zero;
- the resulting NumPy array is read-only.

Embedding data should still be treated as sensitive derived data. Numerical values are not a form
of guaranteed anonymization.

## Provenance and deterministic fingerprints

The corpus fingerprint hashes a canonical representation of:

- schema version;
- stored embedding model and prompt strategy;
- embedding dimension and chunking policy;
- sorted document identities, paths, sizes, and hashes;
- sorted chunk indexes, offsets, and hashes.

Unchanged source and configuration produce the same fingerprint. Editing, renaming, rechunking, or
changing the model/prompt strategy changes it.

The build timestamp and raw SQLite file bytes are excluded from determinism. SQLite page layout and
timestamps may differ even when logical records are equivalent.

The fingerprint proves internal provenance consistency; `inspect-index` does not rescan the source,
so it cannot claim that source files have not changed since the index was built.

## Read-only validation

`inspect-index` opens SQLite using URI `mode=ro`. It then checks:

- SQLite structural integrity;
- foreign-key integrity;
- application and schema identity;
- metadata and physical row counts;
- safe relative paths;
- chunk offsets, lengths, and SHA-256 hashes;
- one compatible vector for every chunk;
- complete corpus-fingerprint agreement.

This same loader validates the temporary database before publication. Reusing one reader prevents
the writer and inspector from developing different definitions of validity.

## Privacy behavior

Normal build and inspection reports contain counts, model metadata, dimensions, configuration,
timings, and fingerprints. They do not contain:

- absolute source or database paths;
- document or chunk text;
- raw vector coordinates;
- embedding prompts or model response bodies.

The SQLite database itself contains chunk text and embeddings because future local retrieval needs
them. It must remain Git-ignored and protected like the original documents.

## Senior-engineer lessons

1. Durable storage needs a reader-defined validation contract, not only successful writes.
2. Database transactions and atomic file publication solve different failure scopes.
3. File ownership markers prevent destructive replacement of unrelated data.
4. Model, prompt, dimension, and serialization format are part of data compatibility.
5. Logical determinism should be defined separately from binary file identity.
6. Corruption tests are as important as schema creation tests.
7. Derived AI artifacts need privacy controls and lifecycle management.

In an interview, describe this as a versioned, transactionally built, atomically published local
vector index with provenance and fail-closed validation—not simply “vectors stored in SQLite.”

## Why SQLite and when it stops being enough

SQLite is appropriate here because it is local, serverless, transactional, inspectable, portable,
and included with Python. For a small single-user corpus, future brute-force NumPy search is easy to
understand and operate.

It becomes insufficient when requirements include concurrent writers, shared multi-user access,
large vector collections, document-level authorization, distributed replicas, approximate nearest
neighbor indexes, or operational high availability. Those requirements can justify PostgreSQL with
pgvector, Qdrant, OpenSearch, or another evaluated vector platform.

## Current limitations

- Every build recreates the complete index.
- All chunks and vectors are held in memory during a build.
- Search and similarity ranking are not implemented yet.
- No schema migration exists because only version 1 exists.
- `inspect-index` validates internal data but does not compare it with live source files.
- Atomic replacement assumes a normal local filesystem and can fail if another process locks the
  database on Windows.

## Check your understanding

1. Why is a SQLite transaction not sufficient to protect an existing index during rebuild?
2. Why must `--force` recognize an app-owned database before replacing it?
3. Why record model, prompt strategy, dimension, and vector format together?
4. Why use explicit little-endian Float32 bytes?
5. What does the corpus fingerprint prove, and what does it not prove?
6. Why does the read-only loader recalculate chunk hashes and validate vector norms?
7. Why are database byte equality and logical determinism different?
8. At what operational scale would you replace this design with a vector database?
