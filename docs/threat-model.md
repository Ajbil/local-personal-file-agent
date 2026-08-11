# Threat Model

## Purpose and scope

This threat model covers the single-user Windows command-line baseline: explicitly approved UTF-8
Markdown/text folders, a local SQLite vector index, loopback-only Ollama, EmbeddingGemma, Qwen, and
the deterministic/live evaluation suite. It explains what the application protects, where trust
changes, which controls are deterministic, and which risks remain.

It does not claim protection from a compromised operating system, local administrator, malicious
Ollama installation, or physical access to an unlocked machine.

## Protected assets

- Original personal documents and their filenames, folders, and metadata.
- Normalized document text, chunks, embeddings, and SQLite indexes.
- User questions, retrieved context, prompts, generated answers, and citations.
- Model/runtime configuration and operational diagnostics.
- Evaluation canaries and failure evidence.

Embeddings and hashes are sensitive derived data. They are not a safe anonymized substitute for the
documents that produced them.

## Trust boundaries and data flow

```text
Untrusted filesystem entries
        |
        v
Approved-root ingestion boundary -- rejects escape, links, size, type, encoding
        |
        v
Trusted normalized Documents -- exact text remains sensitive
        |
        v
Chunks -> loopback EmbeddingGemma -> vectors -> private SQLite index
                                               |
Question -> loopback EmbeddingGemma ------------+-> ranked evidence
                                                      |
                                                      v
                                      explicitly untrusted evidence
                                                      |
                                                      v
                                          loopback Qwen output
                                                      |
                                                      v
                           strict schema + application-owned citation validation
                                                      |
                                           answer or fixed refusal
                                                      |
                              opt-in allowlisted metrics to stderr
```

Trust changes at five important boundaries:

1. Filesystem entries become trusted documents only after secure ingestion.
2. Document text becomes model input but remains untrusted instruction data.
3. Ollama responses cross an external-runtime boundary and require validation.
4. SQLite data is trusted only after schema, integrity, provenance, count, hash, and vector checks.
5. Operational data may leave the process through terminal/log capture and must be allowlisted.

## Threats, controls, and residual risk

| Threat | Implemented controls | Residual risk |
| --- | --- | --- |
| Path traversal or source escape | Explicit approved root, resolved containment, safe relative paths | Small filesystem race windows remain |
| Symlink/reparse-point escape | Links and Windows reparse points rejected | OS compromise can bypass process assumptions |
| Unsupported, huge, binary, or invalid files | Extension, size, UTF-8, binary, and regular-file validation | Complex formats are intentionally unsupported |
| Source modification | Source opened read-only; indexes stored outside source | Another local process can change files concurrently |
| Corrupt or incompatible index | Versioned schema, read-only open, integrity/foreign-key/hash/count/model/dimension checks | SQLite file is not encrypted or authenticated at rest |
| Accidental remote inference | Configuration accepts HTTP loopback hosts only; no URL credentials/path/query | A malicious local service on the port is trusted as Ollama |
| Silent embedding truncation | Ollama embedding requests disable truncation | Model/runtime bugs remain external dependencies |
| Hallucinated answer | Evidence-only prompt, strict payload, required citations, fixed refusal | A cited answer can still misinterpret evidence |
| Invented provenance | Qwen receives opaque IDs; Python maps IDs to validated retrieval records | Citation validity is not the same as factual correctness |
| Prompt injection in documents | Role separation, JSON encoding, untrusted delimiters, opaque IDs, canary evaluation | Model answer content can still be influenced by novel attacks |
| Unsupported secret disclosure | Retrieval threshold plus Qwen sufficiency decision and fail-closed refusal | Relevant-but-insufficient passages can cause model mistakes |
| Diagnostic data leakage | Logging off by default, explicit field schema, stderr separation, no arbitrary fields/raw exceptions | Users can explicitly reveal content with inspection flags |
| Log retention exposure | No automatic log file; user must deliberately redirect stderr | Redirected files require local access and deletion discipline |
| Denial of service | File-size, top-K, answer-size, timeout, context, retry, and batch bounds | Large approved corpora and CPU inference can still be slow |

## Deterministic guarantees versus model behavior

Application code can deterministically guarantee that:

- configured inference cannot target a non-loopback host;
- accepted source paths remain inside the approved root at validation time;
- unsupported file classes do not enter the pipeline;
- stored vectors match the index contract before search;
- Qwen cannot author trusted filenames, offsets, or hashes;
- invalid/unknown citation IDs cannot reach final output;
- structured logs cannot accept question, answer, path, prompt, text, vector, or canary fields.

Application code cannot prove that:

- semantic retrieval always finds the best passage;
- Qwen always understands sufficient evidence;
- a validly cited sentence is logically correct;
- every possible prompt injection is resisted;
- a local administrator cannot read documents, memory, indexes, or redirected logs.

Those model-quality behaviors are measured through evaluation, not represented as security proofs.

## Security assumptions

- The user deliberately chooses the source folder and database destination.
- Windows, Python, `uv`, Ollama, installed models, and dependencies are trusted.
- The machine is a single-user learning environment with appropriate OS account protection.
- `.data/`, personal documents, SQLite indexes, `.env`, and redirected logs remain outside Git.
- Inspection flags that reveal content are used only with approved synthetic or private terminals.

## Non-goals in this baseline

- Authentication, authorization, RBAC, document ACL propagation, or multi-tenancy.
- Encryption at rest, key management, backup policy, or secure deletion guarantees.
- Network service exposure, remote clients, or hosted inference.
- Malware scanning or sandboxed PDF/DOCX parsing.
- Central audit-log storage, security monitoring, or compliance certification.

These are production-evolution concerns, not missing switches that should be added casually.

## Safe operating guidance

- Keep Ollama bound to loopback and never proxy/tunnel port `11434`.
- Keep indexes and redirected logs in `.data/` or another access-controlled ignored directory.
- Do not use `--show-text` or `--show-context` in shared terminals, screenshots, or recordings.
- Treat model answers as potentially sensitive document-derived output.
- Re-run deterministic and live evaluation after model, prompt, chunking, or retrieval changes.
- Delete disposable indexes/logs when their diagnostic purpose is finished.

## Revisit triggers

Review this threat model before adding a network API, another user, non-text parsers, hosted models,
automatic background indexing, shared storage, centralized logging, or any compliance requirement.
