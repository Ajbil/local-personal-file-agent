# Checkpoint 3 — Local Embeddings

## What we built

Checkpoint 3 converts trusted chunks and a question into validated numerical vectors using the
local EmbeddingGemma model. It keeps those vectors only in memory and provides an inspection
command that compares the question with chunks using cosine similarity.

```powershell
uv run file-agent inspect-embeddings `
  --source examples/checkpoint-3/source `
  --document semantic-lab.md `
  --query "How much funding can a worker use for career development?" `
  --chunk-size 450 `
  --overlap 0 `
  --top-k 3
```

The command displays vector dimensions, batch counts, latency, vector norms, and ranked chunk
metadata. It never prints raw vector coordinates. Query and chunk text are also hidden unless
`--show-text` is explicitly supplied.

## Mental model

```text
Trusted chunks                     User question
      |                                  |
document prompt                     query prompt
      |                                  |
      +--------> EmbeddingGemma <---------+
                         |
                  Float32 vectors
                         |
             cosine similarity scores
                         |
                inspectable ranking
```

An embedding is a learned coordinate representation of meaning. EmbeddingGemma returns 768
numbers with the current baseline model. A single coordinate is not intended to have a human label
such as "incident response." Meaning is represented by the vector's overall direction and its
relationship with other vectors.

## Why documents and questions use different prompts

EmbeddingGemma is instruction-aware. Retrieval works best when a stored passage is formatted as a
document and the user's text is formatted as a question:

```text
title: none | text: <chunk>
task: question answering | query: <question>
```

These strings are deliberately centralized under the versioned strategy
`embeddinggemma-question-answering-v1`. Changing prompt formats later is an index compatibility
decision: previously stored vectors may no longer be comparable with newly generated ones.

## The boundaries in the implementation

The implementation separates three responsibilities:

1. The Ollama adapter sends ordered arrays to `/api/embed`, requests `truncate=false`, validates
   the HTTP response shape, and converts Ollama nanosecond timings to milliseconds.
2. The embedding service owns document/query prompts, batching, model identity, vector count,
   dimensions, finite-value checks, non-zero norms, and Float32 conversion.
3. The CLI composes scanning, chunking, embedding, and an in-memory learning-only ranking while
   applying privacy-safe output policy.

This separation makes model transport replaceable without mixing it with retrieval semantics.

## Correctness invariants

For a successful document embedding run:

- each chunk has exactly one vector in the same order;
- every vector is one-dimensional, finite, non-empty, non-zero, and the same size;
- query and document vectors use the configured model and equal dimensions;
- conversion to NumPy `float32` must not introduce infinity;
- Ollama may return an explicit `:latest` tag for an untagged configured model, but a different
  model is rejected;
- truncation is disabled, so an input exceeding model capacity fails visibly instead of silently
  producing an embedding for incomplete text.

The NumPy arrays are marked read-only. This does not create a security boundary, but it catches
accidental mutation inside the application.

## Batching and latency

Chunks are sent in batches of eight by default. Batching reduces HTTP overhead and lets the model
process several inputs in one request. A larger batch may improve throughput but increases peak
memory and makes one failed request affect more chunks.

The report distinguishes wall time measured by Python from model duration reported by Ollama.
Wall time includes HTTP and application overhead. The first request may also include model loading,
so warm and cold measurements should not be compared as though they are identical workloads.

## Cosine similarity

For vectors `a` and `b`:

```text
cosine(a, b) = dot(a, b) / (length(a) × length(b))
```

The score compares direction rather than raw magnitude:

- `1` means the same direction;
- `0` means geometrically unrelated directions;
- `-1` means opposite directions.

Real embedding scores are model- and corpus-dependent. A value is not a universal probability of
relevance. Ranking and thresholds must be evaluated on representative questions.

The cosine ranking in this checkpoint is intentionally an in-memory teaching aid. Persistent
indexing belongs to Checkpoint 4 and the dedicated search contract belongs to Checkpoint 5.

## Privacy and security choices

- Inference uses the already validated loopback-only Ollama endpoint.
- Source folders remain explicitly approved and read-only.
- No vectors, documents, queries, prompts, or model responses are persisted.
- Default human and JSON reports exclude query and chunk text.
- Raw vector coordinates are never part of the report, even with `--show-text`.
- Safe model failures do not include server response bodies or document contents.

Embeddings can still leak information in advanced attacks and should be treated as derived
sensitive data. "Numerical" does not mean "anonymous."

## Senior-engineer lessons

1. Model output must be validated like any external service response.
2. Prompt strategy is versioned data compatibility, not merely prompt wording.
3. Preserve ordering and cardinality across batch boundaries.
4. Reject silent truncation because partial evidence corrupts retrieval invisibly.
5. Separate throughput measurements from correctness and from cold-start latency.
6. Treat embeddings as sensitive production data with access control and lifecycle policies.
7. Do not choose similarity thresholds by intuition; measure retrieval quality.

In an interview, describe embeddings as a typed, versioned transformation with explicit model,
prompt, dimensionality, provenance, and validation contracts—not as "calling an AI API."

## Current limitations

- Vectors disappear when the command exits.
- Only one selected document is compared at a time.
- Cosine ranking has no threshold or overlap suppression.
- Character chunking is not guaranteed to remain within every model token limit.
- The command does not retry transient failures.
- There is not yet a representative retrieval evaluation corpus.

These are intentional checkpoint boundaries. Checkpoint 4 will persist the validated vectors and
their compatibility metadata in SQLite.

## Check your understanding

1. Why are document and query prompt formats different?
2. Why must vector count and chunk count match exactly?
3. Why is `truncate=false` safer for retrieval correctness?
4. Why do all vectors in one index need the same model, prompt strategy, and dimension?
5. What does cosine similarity measure, and why is it not a probability?
6. Why can Float32 conversion require another finite-value check?
7. Why should embeddings be treated as sensitive derived data?
8. What tradeoff changes when batch size increases?
