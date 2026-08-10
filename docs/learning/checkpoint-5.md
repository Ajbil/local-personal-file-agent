# Checkpoint 5 — Read-Only Vector Search

## What we built

Checkpoint 5 converts a question into an embedding, compares it with every validated vector from a
SQLite index, and returns relevant source passages with trusted citations.

```powershell
uv run file-agent search `
  "How frequently do staff passwords need to be changed?" `
  --db .data/manual-testing/checkpoint-5/index.sqlite
```

Search stops at evidence retrieval. It does not send passages to Qwen or generate an answer.

## Mental model

```text
Question
   |
question-specific EmbeddingGemma prompt
   |
query vector
   |
cosine score against every stored chunk
   |
minimum-score filter
   |
deterministic ordering
   |
heavy-overlap suppression
   |
top-K trusted citations
```

The searcher is like a librarian: it finds potentially useful pages. It does not act as the writer
that explains those pages.

## Index-first compatibility

Search validates and loads the database before contacting Ollama. It reads the embedding model,
prompt strategy, and vector dimension from the index and embeds the question with that exact
contract.

This prevents a changed `.env` model from silently generating query vectors in a different
mathematical space. Two models can both produce 768 values while assigning completely different
meaning to those coordinates.

The application rejects mismatched model identity, prompt strategy, or dimension before comparing
vectors.

## Cosine similarity is a ranking signal

For query vector `q` and chunk vector `c`:

```text
cosine(q, c) = dot(q, c) / (length(q) × length(c))
```

Cosine similarity compares direction rather than magnitude. It is useful because semantically
related text often points in a similar direction even when the exact words differ.

A score is not a probability. `0.60` does not mean “60% correct.” Scores depend on the embedding
model, prompt strategy, corpus, chunking, language, and question type. Thresholds must eventually be
selected with an evaluation dataset.

## Brute-force search

Every validated stored vector is compared with the query using NumPy. With `N` chunks and dimension
`D`, the baseline cost is:

```text
time   O(N × D)
memory O(N × D)
```

This is intentionally transparent and adequate for a small local corpus. Approximate nearest
neighbor indexes trade some exactness and simplicity for much better performance at large scale.

## Retrieval policy order

The order is part of the contract:

1. Calculate full-precision cosine scores.
2. Keep scores where `score >= min_score`.
3. Sort by descending score.
4. Break equal-score ties by case-insensitive path, exact path, then chunk index.
5. Greedily suppress heavy same-document overlaps.
6. Return at most `top_k` results.

Applying `top-k` before suppression could let five near-duplicate chunks occupy all five positions.
Applying suppression first leaves room for diverse evidence.

Scores are rounded to six decimals only when building the report. Filtering and ordering use the
unrounded values.

## Top-K and minimum score

`top-k` limits quantity. It does not establish relevance. If only two useful passages exist,
`top-k=10` should not force eight unrelated passages into future model context.

`min-score` sets a provisional relevance boundary:

- lower threshold usually improves recall but admits more distractors;
- higher threshold usually improves precision but may discard valid paraphrases.

Defaults are:

```text
top_k = 5
min_score = 0.30
```

They are hypotheses inherited from the learning roadmap, not production-quality constants.

## Recall versus precision

- Recall asks whether the relevant evidence was retrieved at all.
- Precision asks how much retrieved evidence is actually useful.

Low recall means Qwen never receives the needed fact. Low precision wastes context and increases
the chance that generation uses an irrelevant passage. Retrieval evaluation must therefore be
separate from answer-quality evaluation.

## Heavy-overlap suppression

For two chunks from the same document:

```text
overlap ratio = intersecting character count / shorter chunk length
```

If the ratio is at least `0.80`, the lower-ranked candidate is suppressed. Chunks from different
documents never suppress one another.

This is deterministic range-based duplicate control. It does not attempt semantic deduplication of
different passages that happen to express the same fact.

The fixed 80% policy gives us a measurable baseline. It should change only after evaluation shows
that it harms recall or allows excessive duplication.

## Trusted citations

The application creates citation labels from validated database metadata:

```text
relative/path.md#chunk-2[1800:2950)
```

The relative path, chunk number, and offsets come from the read-only index—not from an AI model.
Checkpoint 6 will give Qwen temporary numeric IDs, but application code will remain responsible for
mapping those IDs to trusted citations.

## Zero results are not an operational error

If no chunk reaches the threshold, search returns exit code `0` and an empty result set. That means
the retrieval operation worked but found insufficient evidence.

This differs from exit code `1`, which represents a corrupt index, incompatible vector, unavailable
model, or runtime failure. Keeping these states separate is important for APIs, monitoring, and
future refusal behavior.

## Privacy and immutability

By default, search prints citations, scores, counts, policies, fingerprints, and timings. It does
not print:

- the exact question;
- retrieved chunk text;
- filtered or suppressed text;
- vector coordinates;
- absolute paths.

`--show-text` explicitly reveals only selected validated passages. Search opens SQLite read-only,
never rescans source files, and never updates index bytes or modification time.

## Senior-engineer lessons

1. Retrieval policy ordering is a public behavior, not an implementation detail.
2. Embedding compatibility includes model, prompt strategy, and dimension.
3. Stable tie-breakers make tests and production behavior reproducible.
4. Empty evidence and infrastructure failure need different states.
5. Thresholds require offline evaluation rather than intuition.
6. Duplicate control preserves diversity and future context-window budget.
7. Retrieval must remain independently observable from generation.

In an interview, explain this as exact brute-force semantic retrieval over a validated read-only
index with deterministic ranking, thresholding, diversity control, and application-owned citations.

## Current limitations

- Search loads and compares every vector.
- The score threshold has not been evaluated on representative questions.
- Suppression detects only character-range overlap within one document.
- There is no lexical/BM25 component, reranker, or hybrid retrieval.
- Search does not detect whether original source files changed after indexing.
- Search returns evidence but does not decide whether it is enough to answer safely.

## Check your understanding

1. Why must search use the model stored in the index instead of `.env`?
2. Why is cosine similarity not a probability?
3. How do `top-k` and `min-score` affect recall and precision differently?
4. Why must suppression happen before `top-k`?
5. Why are deterministic tie-breakers useful?
6. Why is zero results a success rather than an exception?
7. Which metadata is trusted, and why must Qwen not create citations?
8. When would brute-force search need to be replaced?
