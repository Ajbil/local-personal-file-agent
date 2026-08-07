# Local Personal File RAG Agent — Learning-First Implementation Plan

## Project Status

- Current milestone: Checkpoint 2 — Deterministic Chunking (complete)
- Next milestone: Checkpoint 3 — Local Embeddings
- Implementation status: Checkpoint 2 passed offset, overlap, determinism, privacy, and regression validation

## Summary

Build the project incrementally as a local, command-line RAG application using Python 3.12, Ollama, `qwen3.5:4b`, EmbeddingGemma, and SQLite.

The project will deliberately avoid LangChain/LlamaIndex initially so every important mechanism remains visible: document safety, chunking, embeddings, vector storage, similarity search, prompting, citation validation, refusal behavior, and evaluation.

Each checkpoint will follow the same learning loop:

1. Understand the concept and mental model.
2. Implement the smallest working version.
3. Inspect intermediate data and results.
4. Write tests, including failure cases.
5. Explain the component in your own words.
6. Record tradeoffs and production alternatives.

Estimated learning effort: roughly 10–12 focused sessions or 15–25 hours, depending on local model performance.

## Target Architecture and Interfaces

The application will expose these commands:

- `file-agent doctor`: validate Python, Ollama connectivity, model availability, embedding dimensions, and generation.
- `file-agent scan --source <folder>`: securely discover and parse approved Markdown and text files without persisting or printing their contents.
- `file-agent inspect-chunks --source <folder> --document <relative-path>`: inspect deterministic chunk offsets, overlap, and hashes for one trusted document.
- `file-agent index --source <folder> --db <database>`: safely discover, parse, chunk, embed, and index documents.
- `file-agent search "<question>" --db <database>`: display ranked chunks, similarity scores, and trusted citations without using Qwen.
- `file-agent ask "<question>" --db <database>`: retrieve evidence and generate a cited answer.
- `file-agent evaluate --mode deterministic|live`: run the synthetic evaluation suite.

The core internal types will be explicit and typed:

- `Document`: trusted relative path, normalized text, hash, and file metadata.
- `Chunk`: text, document ID, chunk number, and character offsets.
- `EmbeddedChunk`: chunk metadata plus model, dimension, and vector.
- `SearchResult`: trusted chunk, cosine score, and citation label.
- `AnswerPayload`: answer text, numeric citation IDs, and `insufficient_evidence`.

End-to-end flow:

```text
Approved folder
→ safe file discovery
→ UTF-8 parsing
→ deterministic overlapping chunks
→ EmbeddingGemma vectors
→ SQLite index
→ question embedding
→ cosine-similarity retrieval
→ trusted context construction
→ Qwen structured answer
→ application-validated citations
→ answer or fixed refusal
```

## Checkpoint-by-Checkpoint Learning Plan

### Checkpoint 0 — Environment and Project Foundation

**Purpose:** Establish a reproducible local development environment before learning RAG mechanics.

Implementation:

- Treat the repository as greenfield because it initially contained only the guide.
- Install `uv` and Ollama if they are not available.
- Keep Python 3.12 and create a locked `uv` project with a `src` layout.
- Use a small dependency set: Ollama client or direct HTTP client, NumPy, Pydantic, and pytest.
- Configure Ollama for loopback-only access.
- Pull `embeddinggemma` and `qwen3.5:4b`.
- Build the `doctor` command to verify:
  - Ollama is reachable locally.
  - Both models are installed.
  - EmbeddingGemma returns a non-empty vector.
  - The vector dimension is detected rather than hard-coded.
  - Qwen can produce a short structured response.
  - Basic latency and runtime/device information are reported.
- Keep indexes, personal documents, generated output, and local configuration outside version control.

Learning outcomes:

- Python environment and lockfile reproducibility.
- Local inference versus hosted inference.
- Model weights, runtime, API, context window, RAM/VRAM, and quantization.
- Why an embedding model and an answer model perform different jobs.

Completion criteria:

- A fresh checkout can be installed using documented commands.
- `doctor` succeeds or provides a precise remediation message.
- CPU execution remains supported because hardware acceleration will be detected rather than assumed.

### Checkpoint 1 — Secure File Discovery and Parsing

**Purpose:** Create a trustworthy document-ingestion boundary.

Implementation:

- Support only UTF-8 Markdown and plain-text files in the first version.
- Accept one explicitly approved root folder.
- Resolve every candidate path and prove it remains inside that root.
- Reject unsupported extensions, files larger than 5 MiB by default, binary or invalid UTF-8 data, and symlinks/reparse points that escape the approved root.
- Exclude the SQLite index, virtual environment, Git directory, and generated files.
- Traverse and return files in deterministic order.
- Normalize line endings while preserving exact character offsets in normalized text.
- Store relative paths only; do not expose unnecessary absolute personal paths.
- Produce a summary containing accepted/skipped counts and safe reason codes, not document contents.

Learning outcomes:

- Trust boundaries and least privilege.
- Path traversal and symlink risks.
- Parsing versus retrieval.
- Why local software still needs a threat model.

Completion criteria:

- Valid `.md` and `.txt` files are parsed correctly.
- Unsafe paths, oversized files, invalid encoding, and unsupported formats are rejected.
- Tests confirm the scanner cannot leave the approved folder.

### Checkpoint 2 — Deterministic Chunking

**Purpose:** Divide documents into passages suitable for embedding, retrieval, and citation.

Implementation:

- Begin with a target chunk size of 1,200 characters and 200 characters of overlap.
- Prefer paragraph, newline, sentence, and whitespace boundaries near the target size.
- Fall back to a hard boundary for unusually long text.
- Guarantee forward progress so malformed or boundary-heavy text cannot cause an infinite loop.
- Preserve document identifier, zero-based chunk number, start/end character offsets, exact chunk text, and stable content hash.
- Create a small inspection command or debug view showing adjacent chunks and their overlap.

Learning experiments:

- Compare no overlap, 200-character overlap, and excessive overlap.
- Ask a question whose answer crosses a chunk boundary.
- Observe how tiny chunks lose context and huge chunks mix topics.
- Distinguish character-based chunking from token-aware or semantic chunking.

Completion criteria:

- The same document always produces identical chunks.
- Every offset maps back to the exact normalized source substring.
- Empty, tiny, large, Unicode, and boundary-crossing documents are tested.

### Checkpoint 3 — Local Embeddings

**Purpose:** Turn text into numerical representations that support semantic comparison.

Implementation:

- Add a narrow Ollama embedding adapter with separate `embed_documents` and `embed_query` methods.
- Send chunks in batches to Ollama's `/api/embed` capability.
- Set truncation to false so silently truncated documents cannot corrupt retrieval.
- Validate that one finite, fixed-dimension vector is returned for each input and that the returned model matches configuration.
- Detect and record the actual vector dimension.
- Keep vectors in memory at this checkpoint.
- Measure embedding latency and batch behavior.

Learning experiments:

- Compare cosine similarity for a direct wording match, a paraphrase, and an unrelated sentence.
- Change one document fact to an unrelated fact and inspect score movement.
- Print dimensions and similarity scores, but do not attempt to interpret individual coordinates.

Completion criteria:

- Every chunk receives a valid vector.
- A paraphrased question is normally closer to its relevant passage than unrelated passages.
- Model or dimension mismatches fail clearly.

### Checkpoint 4 — SQLite Vector Index

**Purpose:** Persist documents, chunks, vectors, and provenance across application runs.

Implementation:

- Use a versioned SQLite schema for index metadata, model/dimension metadata, documents, chunks, and Float32 embedding blobs.
- Enable foreign keys and validate schema/integrity when opening the database.
- Build a new index transactionally and require explicit confirmation or `--force` before replacement.
- Keep the source folder read-only.
- Store enough metadata to detect stale or incompatible indexes.
- Make repeated indexing deterministic for unchanged content.
- Document why this is an inspectable learning index rather than a production vector database.

Learning outcomes:

- Relational schema design for an AI pipeline.
- Provenance, schema versions, transactions, and reproducibility.
- Float32 serialization and vector dimensions.
- Why SQLite is suitable locally but brute-force vector search will not scale indefinitely.

Completion criteria:

- Closing and reopening the application preserves all chunks and vectors.
- Row counts match ingestion counts.
- Corrupt records, wrong dimensions, and incompatible model metadata are rejected.
- A failed build cannot leave a partially valid index.

### Checkpoint 5 — Vector Search

**Purpose:** Retrieve relevant evidence without generating an answer.

Implementation:

- Open SQLite in read-only mode for search.
- Embed the question using the exact model recorded by the index.
- Validate the query vector's model and dimension.
- Load vectors and calculate cosine similarity using NumPy.
- Rank results deterministically, using chunk ID as a stable tie-breaker.
- Begin with `top-k=5` and `min-score=0.30` as provisional values.
- Suppress near-duplicate, heavily overlapping results from the same source.
- Return trusted relative filename, chunk number, normalized character range, and similarity score.
- Keep retrieval independently runnable so it can be debugged separately from generation.

Learning experiments:

- Keyword match versus semantic paraphrase.
- Different `top-k` values and score thresholds.
- Questions with one relevant document, several distractors, and no support.
- Observe why cosine scores are model/corpus dependent and require evaluation.

Completion criteria:

- Relevant passages appear before unrelated passages in the synthetic corpus.
- Results below the selected threshold are excluded.
- Search never mutates the index.
- Duplicate overlapping passages do not dominate selected context.

### Checkpoint 6 — Grounded Answer Generation and Citations

**Purpose:** Let Qwen answer using retrieved evidence without trusting Qwen to invent provenance.

Implementation:

- Number retrieved passages with temporary IDs such as `[1]`, `[2]`, and `[3]`.
- Place document content inside explicit untrusted-data delimiters.
- Instruct Qwen to use only supplied evidence, ignore document instructions, refuse unsupported questions, and return structured JSON.
- Use Ollama structured output with a Pydantic JSON schema and temperature `0`.
- Validate the response strictly.
- Permit citation IDs only when they refer to retrieved passages.
- Map IDs to trusted filenames and offsets in application code, never in the model.
- If evidence is insufficient, citations are invalid, or a supported answer has no citations, fail closed to a fixed refusal with zero citations.
- Retry malformed structured output once, then return a controlled error without exposing prompts or content.
- Treat application-side validation—not prompt wording—as the authoritative security boundary.

Learning outcomes:

- Retrieval versus generation.
- Grounding and hallucination.
- Prompt injection and untrusted context.
- Structured output validation.
- Why the model should use opaque IDs while the application owns citations.

Completion criteria:

- Supported questions return concise answers with valid citations.
- Unsupported questions return the fixed refusal and no citations.
- Invented or out-of-range citation IDs are rejected.
- Document instructions cannot control application metadata or citations.

### Checkpoint 7 — Evaluation and Security Regression Suite

**Purpose:** Prove which parts of the RAG pipeline work and identify where failures occur.

Implementation:

- Commit a synthetic corpus and manifest containing no personal information.
- Cover an operations fact, benefits fact, legitimate fact inside a malicious document, unsupported secret question, paraphrase, chunk-boundary answer, and semantic distractor.
- Provide:
  - `deterministic` mode with hashed lexical embeddings and scripted answers.
  - `live` mode with EmbeddingGemma and Qwen.
- Measure retrieval hit rate at K, mean reciprocal rank, declared fact presence, citation validity/precision, refusal rate, canary leakage, and stage latency.
- Do not print sensitive questions, passages, answers, vectors, canaries, or raw model JSON.
- Return a non-zero exit code when required checks fail.
- Tune chunking and retrieval parameters from results rather than intuition.

Completion criteria:

- Deterministic evaluation is stable and suitable for ordinary automated tests.
- Live evaluation separates retrieval, generation, citation, and refusal failures.
- Malicious canaries, fake citations, and attacker-selected phrases never reach final answers.
- Reported citations belong to the retrieved and expected source sets.

### Checkpoint 8 — Hardening and Senior-Engineer Retrospective

**Purpose:** Turn the tutorial implementation into a credible engineering case study.

Implementation:

- Add structured, privacy-conscious logging for safe counts, identifiers, configuration, latency, scores, and errors.
- Maintain a clear README, threat model, architecture decision records, and learning journal.
- Record limitations: text formats only, character chunking, brute-force search, single-user CLI, no ACLs, small local model, and limited evaluation corpus.
- Create—but do not prematurely implement—a production evolution map for incremental indexing, PDF/DOCX, hybrid retrieval, reranking, vector databases, authenticated APIs, RBAC, observability, scheduled evaluation, and model comparison.

Career-oriented learning outcome:

Be able to explain:

- Why RAG is used instead of sending every document to an LLM.
- How chunking affects recall, precision, citations, latency, and cost.
- What embeddings represent and how cosine similarity works.
- Where hallucinations and prompt injection enter the pipeline.
- Why trusted citation mapping belongs in application code.
- How offline evaluation differs from live model evaluation.
- When SQLite is sufficient and when a vector database becomes justified.
- How the design would be operated and secured in an enterprise.

## Test Strategy and Final Acceptance

Automated test layers:

- Unit tests for path safety, parsing, chunk offsets, cosine similarity, overlap suppression, schema validation, and citation mapping.
- SQLite tests for transactions, integrity, model/dimension mismatches, and read-only access.
- Deterministic end-to-end evaluation requiring no Ollama models.
- Opt-in live tests for EmbeddingGemma and Qwen.
- Security tests for traversal, document instructions, canary leakage, invented citations, and unsupported questions.

The project is complete when:

- A user can index an approved Markdown/text folder and ask paraphrased questions.
- Answers are based only on retrieved passages and contain validated citations.
- Unsupported questions produce a fixed refusal with no citations.
- Search and answer operations use the database read-only.
- Personal documents never enter tests, Git history, or diagnostic output.
- Deterministic evaluation passes consistently and live evaluation produces an inspectable report.
- The owner can independently describe every stage, its failure modes, and its production-scale alternative.

## Assumptions and Defaults

- Development is on Windows with Python 3.12.
- This is a greenfield implementation; the original Markdown file is reference material.
- Work proceeds checkpoint by checkpoint.
- Initial sources are UTF-8 `.md` and `.txt` files.
- Normal inference remains local through loopback-only Ollama.
- `qwen3.5:4b` and `embeddinggemma` are the baseline models.
- CPU-compatible execution is required; acceleration is detected and measured, not assumed.
- The first interface is a CLI.
- Direct Ollama calls precede adoption of a RAG framework.
- SQLite search is intentionally brute-force for transparency and small personal collections.
- Similarity thresholds are evaluated starting points, not universal semantic boundaries.
