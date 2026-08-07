# Checkpoint 1 — Secure File Discovery and Parsing

## What we built

Checkpoint 1 adds a trusted ingestion boundary for one explicitly approved folder. The new
`file-agent scan` command recursively discovers files, accepts UTF-8 Markdown and plain text,
normalizes their line endings, and reports metadata without printing document contents.

No embedding model is involved yet. This separation is intentional: retrieval quality cannot be
trusted if the application cannot first establish exactly which documents it read and how it
represented them.

## Mental model: an airport security gate

The filesystem is the public side of the gate. Files may be unsupported, unexpectedly large,
binary, incorrectly encoded, hidden, linked outside the approved folder, or changing while they
are read.

The trusted `Document` object is the secure side of the gate. Downstream code can rely on these
invariants:

- Its path is relative to the approved root.
- Its text decoded successfully as UTF-8.
- Its line endings use `\n` consistently.
- It passed the configured size limit.
- It was not reached through a traversed symlink or reparse point.
- Its character count and SHA-256 hash describe the normalized text.

This is an example of **making illegal states difficult to represent**. Later chunking code will
accept `Document` values instead of reopening arbitrary filesystem paths.

## Data flow

```text
Explicit --source folder
        |
        v
Deterministic discovery
        |
        v
Path / link / hidden / generated checks
        |
        v
Size / binary / UTF-8 checks
        |
        v
Newline normalization + SHA-256
        |
        +----> trusted Document objects (internal)
        |
        +----> metadata-only ScanReport (CLI)
```

## Why the source must be explicit

The command has no default source directory. Requiring `--source` is a small form of
least-privilege design: the application reads only the directory approved for that invocation.
It will not silently scan the current directory, home directory, or repository.

The resolved source root becomes the containment boundary. Candidate paths are checked using
path components rather than string prefixes. For example, `D:\notes-secret` is a sibling of
`D:\notes`, not a child, despite sharing the same text prefix.

## Discovery versus parsing

These are separate responsibilities:

- **Discovery** decides which filesystem entries are candidates and establishes deterministic
  ordering.
- **Validation** decides whether a candidate is safe and supported.
- **Parsing** converts accepted bytes into normalized text and metadata.

Keeping these concepts distinct makes failures easier to test and lets later index commands reuse
the ingestion boundary without duplicating security rules.

## Why invalid files are skipped

A valid source folder can contain thousands of files. Failing the entire operation because of one
image or malformed text file would make ingestion fragile. Checkpoint 1 therefore uses two failure
levels:

- Invalid or unreadable source root: fail the command with exit code `1`.
- Invalid individual entry: skip it, record a safe reason, and continue.

This is a common batch-processing pattern. It combines availability with observability: useful
work continues, but rejected inputs never disappear silently.

## UTF-8, binary detection, and line endings

Text files are byte sequences until decoded. This checkpoint uses strict UTF-8 decoding and
accepts an optional UTF-8 byte-order mark. It never replaces invalid bytes silently because that
would corrupt content without making the problem visible.

A null byte is treated as a strong binary-file signal. This intentionally simple first-version
rule is inspectable and avoids pretending that filename extensions prove content type.

Windows commonly stores newlines as `\r\n`, while Unix-like systems use `\n`. Standalone `\r` is
also possible. All three forms are normalized to `\n`. Future citation offsets will refer to this
normalized string, not raw source bytes.

## Why hash normalized text

SHA-256 creates a stable fingerprint for the parsed content. Hashing the normalized UTF-8 text
means a file changed only from Windows to Unix line endings keeps the same semantic identity.

The hash is not an embedding:

- A cryptographic hash answers, “Are these normalized contents exactly the same?”
- An embedding helps answer, “Are these meanings similar?”

Checkpoint 1 uses the first. Embeddings arrive later.

## Privacy model

The scanner needs document text internally so later checkpoints can chunk it, but terminal and
JSON reports contain metadata only:

- relative path
- byte and character counts
- SHA-256
- accepted/skipped counts
- safe skip-reason codes

They do not contain text previews or normal absolute paths. This reduces accidental disclosure in
terminal history, screenshots, logs, CI output, and bug reports.

## Try it safely

Run the committed synthetic example:

```powershell
uv run file-agent scan --source examples/checkpoint-1/source
```

Inspect the machine-readable contract:

```powershell
uv run file-agent scan --source examples/checkpoint-1/source --json
```

Expected result:

- Two accepted documents.
- One skipped CSV file.
- Relative paths in deterministic order.
- Metadata and reason codes, but no source text.

Repeat the command and compare the order and hashes. With unchanged files, they should remain
stable.

## Senior-engineer lessons

1. Validate at the boundary, then give internal code stronger guarantees.
2. Use allowlists for sensitive inputs instead of trying to enumerate every dangerous case.
3. Separate domain data (`Document`) from operational reporting (`ScanReport`).
4. Design deterministic pipelines so tests and production incidents are reproducible.
5. Model partial failure explicitly in batch workflows.
6. Keep logs useful without leaking the data being processed.
7. Document canonical representations before downstream systems depend on their offsets or hashes.

In an interview, describe this checkpoint as a secure and deterministic ingestion boundary, not as
“a script that reads files.” The boundary, invariants, failure taxonomy, privacy controls, and tests
are the engineering design.

## Current limitations

- Only Markdown and plain-text files are supported.
- The 5 MiB limit is currently a fixed application policy.
- Binary detection deliberately uses a conservative null-byte rule rather than a MIME library.
- Filesystem state can still change at very small race windows; production hardening could use
  platform-specific no-follow file handles and operating-system sandboxing.
- Scanning is synchronous and holds accepted document text in memory.
- There is no incremental index or persistent storage yet.

These are appropriate constraints for the learning baseline. We should measure a real need before
adding file-type detection libraries, concurrency, watchers, or platform-specific filesystem APIs.

## Check your understanding

1. Why is a local filesystem still an untrusted input boundary?
2. Why is path-component containment safer than comparing path strings?
3. Why does one unsupported file not fail the entire scan?
4. Why are normalized character offsets different from raw byte offsets?
5. What information does SHA-256 provide that an embedding does not?
6. Why does the CLI omit document text even though the parser successfully read it?
7. What scalability bottleneck would appear first if this scanned one million files?
