# Designing a Local Knowledge Assistant

## Why ingestion comes first

A dependable knowledge assistant begins by establishing which documents it is allowed to read.
The filesystem boundary validates paths, formats, sizes, encodings, and links before any model sees
the text. This makes later components simpler because they receive trusted normalized documents
instead of arbitrary paths and byte streams.

Security and retrieval quality are connected. If an index silently reads the wrong directory, a
perfect similarity algorithm still returns the wrong evidence. If decoding changes characters
without reporting it, citations can point at offsets that no longer match their source.

## Why documents become chunks

Embedding an entire long document into one vector mixes many ideas into one representation. A
question about a small implementation detail may then compete with unrelated sections. Smaller
passages create more focused semantic representations and make retrieved evidence easier to show
and cite.

Very small passages create the opposite problem. They may contain a name without the sentence that
explains it, or a conclusion without the assumptions that support it. Chunk size is therefore a
retrieval-quality tradeoff rather than a universal constant.

## The overlap experiment

Imagine that one chunk ends with the sentence, "The deployment key is derived from the approved
environment," while the next sentence begins, "and it must never be copied into diagnostic logs."
If the boundary falls between those clauses, retrieving only one side can lose essential meaning.

Overlap repeats a controlled amount of the previous passage at the beginning of the next passage.
The repeated context improves the chance that a boundary-crossing fact remains understandable. It
also creates costs: more chunks to embed, more storage, and a higher chance of retrieving duplicate
evidence. The default two-hundred-character overlap is a starting hypothesis to evaluate.

## Exact offsets and citations

A chunk is not merely a copied string. It records its start and end character positions in the
normalized document. The end is exclusive, following Python slicing, so the invariant is
`document.text[start:end] == chunk.text`. This gives later citation code a mechanical way to prove
that quoted evidence came from the claimed source.

Python character offsets differ from UTF-8 byte offsets. The phrase "café 🚀 नमस्ते" contains
characters that occupy different numbers of bytes, but Python slicing still maps the stored chunk
back to the normalized string exactly.

## Determinism as an operational feature

The same document and configuration must always produce the same chunk boundaries, indexes, and
hashes. Determinism makes an index reproducible, test failures explainable, and production
incidents easier to investigate. A hidden random choice in chunking would create unnecessary
changes in every downstream embedding and retrieval result.

Stable document identity is separate from content identity. A path-derived identifier lets a
future index replace the chunks for the same logical file after an edit. A content hash shows that
the text changed. Renaming the file intentionally creates a new path-derived identity.

## A deliberately long token

The following artificial token demonstrates why every chunker needs a hard-boundary fallback when
no paragraph, sentence, newline, or whitespace exists near the target:

ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789

Human writing normally offers natural boundaries, but machine-generated identifiers, minified
data, and corrupted input may not. The algorithm must still move forward rather than loop forever.

## Preparing for embeddings

Checkpoint 2 stops after producing deterministic chunks. The next checkpoint will send those exact
passages to the local embedding model. Keeping these stages separate lets us inspect chunk quality
before numerical vectors make mistakes harder to see.

The senior-engineering lesson is broader than RAG: establish explicit contracts at subsystem
boundaries, make important invariants executable through tests, and introduce complexity only when
evidence shows that the baseline is insufficient.
