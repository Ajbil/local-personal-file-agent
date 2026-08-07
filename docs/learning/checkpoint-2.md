# Checkpoint 2 — Deterministic Chunking

## What we built

Checkpoint 2 converts trusted normalized documents into smaller passages for future embeddings,
retrieval, and citations. The algorithm is direct Python code with no model, tokenizer, framework,
database, or network call.

The new command is:

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md
```

It displays offsets, sizes, overlap, and hashes without printing document text by default.

## Mental model

```text
Trusted normalized Document
        |
        v
Target window (1,200 characters)
        |
        v
Natural-boundary search near the end
        |
        v
Exact source slice + offsets + hash
        |
        v
Move start backward by 200 characters for overlap
        |
        v
Repeat until the document end
```

The chunker does not summarize, rewrite, or decode text. It only chooses boundaries in the
normalized string created by Checkpoint 1.

## The central invariant

For every chunk:

```python
document.text[chunk.start_char : chunk.end_char] == chunk.text
```

`start_char` is inclusive and `end_char` is exclusive. This convention is valuable because it is
identical to Python slicing and represents empty ranges without special arithmetic.

Future citation validation can use this invariant to prove that retrieved evidence exists at the
claimed location in the trusted source.

## Default policy

- Target size: 1,200 characters.
- Overlap: 200 characters.
- Boundary search: final 25% of the target window.
- Boundary priority: paragraph, newline, sentence, whitespace, hard cut.

These numbers are hypotheses, not universal truths. Retrieval evaluation later in the project will
tell us whether they are appropriate for the actual document collection and embedding model.

## Boundary selection

If the remaining document fits within the target size, the final chunk ends at the document end.
Otherwise, the chunker searches backward near the target:

1. Prefer a blank line because it usually separates topics or paragraphs.
2. Otherwise prefer a single newline.
3. Otherwise prefer `.`, `!`, or `?` followed by whitespace.
4. Otherwise prefer any whitespace.
5. Otherwise cut exactly at the target.

The rightmost boundary in the preferred category is selected. Category priority deliberately wins
over a slightly later weak boundary.

This is a deterministic heuristic. It is not a complete Markdown parser or linguistic sentence
segmenter. For example, a period in an abbreviation may be treated as a sentence boundary.

## Why overlap exists

An answer may cross a chosen boundary. Without overlap, one chunk might contain the subject while
the next contains the conclusion. Repeating part of the preceding passage preserves nearby
context.

Overlap also has costs:

- More chunks and embedding operations.
- More vector-storage space.
- More near-duplicate retrieval results.
- Less new information per chunk when overlap becomes excessive.

This is why the configuration requires:

```text
0 <= overlap < chunk_size
```

The next start is `previous_end - overlap`. The selected end must be more than `overlap` characters
beyond the current start, which proves that the loop advances.

## Stable identity versus content identity

Documents now have two distinct hashes:

- `document_id`: SHA-256 of the normalized relative path.
- `content_sha256`: SHA-256 of normalized document text.

Editing a file keeps its logical document ID but changes its content hash. This will let a future
index replace old chunks for the same path. Renaming a file changes its document ID.

Chunks also have a `content_sha256` computed from their exact text. With deterministic boundaries,
an unchanged chunk retains the same hash.

## Character offsets are not byte or token offsets

Python indexes Unicode strings by code points. A character such as `é`, `🚀`, or `न` may require
multiple UTF-8 bytes but still participates in Python slicing as one code point.

Tokens are different again. An embedding model's tokenizer may divide one word into multiple
tokens or combine common character sequences. Character chunking is a transparent baseline; a
later production version may become tokenizer-aware if evaluation shows truncation or inefficient
context usage.

## Privacy-aware inspection

Default inspection prints only metadata:

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md
```

To deliberately inspect exact synthetic content and marked overlaps:

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md `
  --show-text
```

`--show-text` should be used carefully with personal documents because terminal history,
screenshots, and logs may retain output.

JSON metadata is available with:

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md `
  --json
```

Text is omitted from JSON unless `--show-text` is also supplied.

## Learning experiments

### Compare overlap

```powershell
uv run file-agent inspect-chunks `
  --source examples/checkpoint-2/source `
  --document long-rag-note.md `
  --chunk-size 500 `
  --overlap 0
```

Repeat with `--overlap 100` and `--overlap 400`. Observe the chunk count and the amount of new text
introduced by each chunk.

### Compare chunk size

Try `--chunk-size 250`, `500`, and `1500`. Small chunks isolate details but lose surrounding
context. Large chunks preserve more context but mix topics.

### Inspect actual boundaries

Use `--show-text` only on the synthetic example. Verify that the start of each later chunk repeats
the number of characters shown as its overlap.

### Verify determinism

Run the same JSON command twice. Chunk indexes, offsets, hashes, and ordering should be identical.

## Senior-engineer lessons

1. Separate correctness invariants from tunable policy values.
2. Preserve source provenance at every transformation.
3. Make batch algorithms prove forward progress.
4. Prefer deterministic preprocessing for reproducible downstream systems.
5. Treat debugging visibility and data privacy as an explicit tradeoff.
6. Keep stable entity identity separate from mutable content versions.
7. Establish a simple measurable baseline before adopting semantic chunking frameworks.

In an interview, explain this as a deterministic source-mapping transformation with explicit
invariants and failure constraints—not merely splitting strings every 1,200 characters.

## Current limitations

- Chunk sizes are measured in Python characters rather than model tokens.
- Sentence detection is a small punctuation heuristic.
- Markdown headings and code blocks do not receive special treatment.
- The algorithm does not measure semantic topic changes.
- Long documents and all chunks remain in memory.
- The defaults have not yet been evaluated against retrieval questions.

These limitations are intentional. Checkpoint 3 will add local embeddings, and later evaluation
will provide evidence for changing the chunking policy.

## Check your understanding

1. Why must a chunk store offsets as well as text?
2. Why is the end offset exclusive?
3. What failure occurs when overlap equals chunk size?
4. Why might excessive overlap reduce retrieval quality?
5. How does a path-derived document ID help incremental indexing?
6. Why can character limits and token limits disagree?
7. Which chunking choices are correctness guarantees and which are tunable policy?
