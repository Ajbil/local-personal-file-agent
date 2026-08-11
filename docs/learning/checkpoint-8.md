# Checkpoint 8 — Hardening and Senior-Engineer Retrospective

## What we built

Checkpoint 8 completes the planned local learning baseline. It adds opt-in privacy-safe structured
observability, a threat model, three additional architecture decision records, a production
evolution map, final verification guidance, and this retrospective.

It deliberately does not add a web UI, network API, orchestration framework, vector database,
multi-user access, or new document formats. Senior engineering is partly the discipline to avoid
complexity until evidence establishes a requirement.

## The complete architecture

```text
Approved local folder
  -> secure UTF-8 ingestion
  -> deterministic source-mapped chunks
  -> local document embeddings
  -> transactional versioned SQLite index
  -> read-only query embedding and cosine search
  -> threshold + overlap suppression + top-K
  -> bounded numbered untrusted evidence
  -> local Qwen structured answer
  -> application-owned validation and citations
  -> grounded answer or fixed refusal
  -> deterministic/live evaluation
  -> opt-in privacy-safe operational events
```

Each arrow is a contract and possible failure boundary. The design is understandable because these
boundaries are explicit rather than hidden inside an AI framework.

## Observability without surveillance

Enable safe JSONL lifecycle events with:

```powershell
uv run file-agent --log-level info evaluate --mode deterministic
```

Normal command output stays on stdout. Logs go to stderr so scripts can parse `--json` output without
mixing it with diagnostics.

The log schema accepts counts, model/configuration identifiers, ranking scores, decisions, latency,
and safe error categories. It has no field for a question, answer, path, passage, prompt, vector,
citation, hash, HTTP body, raw exception, or canary. This is an allowlist security boundary: a
developer cannot add arbitrary logging context without changing and reviewing the schema.

Logging is off by default because even safe metadata creates retention and correlation concerns.
Automatic log files would silently create another sensitive datastore. A user can deliberately
redirect stderr into Git-ignored `.data/` when diagnosing a known problem.

## Failure ownership by stage

| Stage | Representative failure | Correct response |
| --- | --- | --- |
| Ingestion | path escape, unsupported/invalid file | reject before model access |
| Chunking | invalid overlap or context-destroying policy | validate and evaluate settings |
| Embedding | wrong model/count/dimension/non-finite vector | stop; never compare incompatible vectors |
| Storage | corrupt schema/hash/vector/count | reject index read-only |
| Retrieval | expected source absent from top-K | tune using labeled evaluation, not one anecdote |
| Generation | correct evidence retrieved but fact absent/refused | inspect prompt/model behavior |
| Citation | unknown/duplicate/missing ID | fail closed to fixed refusal |
| Security | canary or hostile phrase reaches answer | block change and investigate trust boundary |
| Operations | timeout/runtime/storage failure | safe error category plus stage latency/counts |

This taxonomy is more valuable than a single “RAG failed” message because each stage has different
owners, evidence, and remediation.

## Security reasoning

Three ideas must remain separate:

1. **Privacy:** prevent unnecessary personal information from leaving its intended boundary.
2. **Safety:** refuse when the system cannot validate a grounded outcome.
3. **Security:** resist malicious input and preserve provenance/integrity.

Local inference improves privacy by avoiding a hosted model, but local software still parses
untrusted files, stores derived sensitive data, calls another process, and emits terminal output.
The threat model therefore covers filesystem, database, model, and observability boundaries.

Prompt injection is not “solved” by one system prompt. Deterministic controls prevent Qwen from
authoring trusted provenance; live canary evaluation measures whether hostile text influences
answer content. The former is an architectural guarantee. The latter remains empirical.

## Why RAG instead of sending every document to Qwen

RAG narrows a large private corpus into a small relevant evidence set before generation. This:

- fits model context limits;
- reduces latency and resource use;
- limits irrelevant and hostile context exposure;
- enables source-level citations;
- makes retrieval independently measurable;
- supports refusal when evidence is absent.

The cost is a new failure mode: retrieval can omit the necessary evidence. That is why search is a
first-class observable/evaluated subsystem rather than an invisible prompt-building detail.

## Chunking tradeoffs

Smaller chunks can improve topic precision and reduce context cost, but may split a fact from its
qualifier. Larger chunks preserve context but mix topics, reduce retrieval precision, and expose
more untrusted text. Overlap protects boundary facts but increases storage, embedding work, and
duplicate candidates.

The baseline uses 1,200 characters with 200-character overlap because it is inspectable and tested,
not because it is universally optimal. A production domain must evaluate tokenizer-aware or
semantic alternatives against recall, answer quality, citation precision, latency, and security.

## Embeddings and cosine similarity

An embedding is a learned numeric representation used for comparison, not a human-interpretable
database field. Cosine similarity measures vector direction, producing a ranking signal. It does
not prove that a passage answers the question or that `0.55` is universally “good.” Scores depend on
model, prompt strategy, corpus, and query distribution.

This is why the index stores model/dimension/prompt provenance and why thresholds are evaluated
rather than copied from another system.

## SQLite versus a vector database

SQLite plus NumPy is appropriate here because the corpus is local/small, writes occur during an
explicit build, search is single-user, and inspectability is a primary learning goal. It becomes
the wrong tool when measured vector-loading memory, linear-search latency, concurrency,
incremental-update, filtering, replication, or availability requirements exceed the baseline.

A vector database is then an evidence-driven architecture change—not a prerequisite for calling a
system RAG.

## Deterministic versus live evaluation

Deterministic evaluation verifies application wiring, ranking policy, citation enforcement,
privacy, security gates, and exit codes without Ollama. Live evaluation measures the installed
EmbeddingGemma/Qwen combination and exposes semantic/model variability. Both are needed:

```text
deterministic pass + live fail -> model/prompt/semantic regression
deterministic fail             -> application/fixture/contract regression
both pass                       -> declared cases pass, not universal correctness
```

## Operating this design in an enterprise

The current CLI is not an enterprise service. An enterprise evolution would require authenticated
service boundaries, document ACL propagation before retrieval, tenant isolation, encrypted storage,
key/secrets management, background indexing, resource quotas, centralized redacted observability,
retention/deletion, scheduled evaluation, release gates, incident runbooks, and rollback.

The production evolution map ties each addition to a measurable trigger and new proof obligation.

## Interview-ready explanation

> I built a local RAG system from first principles so every trust and quality boundary remained
> visible. It securely discovers approved text files, creates source-mapped overlapping chunks,
> embeds and transactionally persists them in a versioned SQLite index, performs read-only NumPy
> cosine retrieval, sends bounded opaque-ID evidence to a local Qwen model, and validates citations
> in application code. Unsupported or invalid output fails closed. A versioned deterministic/live
> evaluation suite measures retrieval, facts, citations, refusals, injection canaries, and latency,
> while opt-in typed JSONL events provide operational evidence without logging content or paths. I
> also documented when the design should evolve to hybrid search, reranking, vector storage, an
> authenticated service, ACLs, and centralized operations.

## Final limitations

- UTF-8 Markdown/text only; no layout-aware or sandboxed complex parsers.
- Character rather than token/semantic chunking.
- Full rebuilds rather than incremental indexing.
- Brute-force in-memory vector search.
- Single-user CLI without authentication, ACLs, or multi-tenancy.
- No encryption at rest or managed backup/deletion lifecycle.
- Small local answer model with possible false refusals/misinterpretation.
- Valid citations establish provenance, not universal factual correctness.
- Seven-case synthetic evaluation is regression coverage, not production statistics.
- Opt-in local logs provide no historical fleet monitoring or alerting.

These limitations are explicit architectural boundaries, not hidden defects.

## Check your understanding

1. Why is local inference a privacy property but not a complete threat model?
2. Which failures can application validation prevent deterministically?
3. Why can a valid citation still accompany a wrong answer?
4. How do chunk size and overlap affect quality, latency, cost, and attack surface?
5. Why is cosine score not answer confidence?
6. When does SQLite become the wrong retrieval store?
7. Why must document ACLs be enforced before context reaches the model?
8. Why are deterministic and live evaluation separate release signals?
9. Why are logs off by default, and why must stdout/stderr remain separate?
10. Which production evolution would you prioritize for a given measured failure, and what new risk
    would it introduce?
