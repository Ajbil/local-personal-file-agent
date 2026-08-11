# Checkpoint 7 — Evaluation and Security Regression Suite

## What we built

Checkpoint 7 turns the RAG pipeline from a working demo into a measured system. A strict,
versioned manifest describes seven synthetic scenarios and their expected sources, answer facts,
refusal behavior, and security canaries. The evaluator builds a fresh disposable SQLite index, runs
the real retrieval and answer-validation path, computes content-free metrics, deletes the index,
and returns success only when every required case passes.

```powershell
uv run file-agent evaluate --mode deterministic
uv run file-agent evaluate --mode live
```

The committed corpus contains no personal information. It covers a direct fact, paraphrases, a fact
near a chunk boundary, a semantic distractor, hostile document instructions beside a legitimate
fact, and an unsupported secret question.

## Mental model: a pipeline test bench

```text
Versioned synthetic manifest + corpus
                    |
                    v
          fresh disposable index
                    |
          question-by-question run
                    |
     +--------------+--------------+
     |              |              |
 retrieval rank  answer facts   refusal/security
     |              |              |
     +--------------+--------------+
                    |
          privacy-safe scorecard
                    |
          pass/fail process exit
```

An evaluation suite is executable evidence about declared behavior. It detects known regressions;
it does not prove universal quality or security.

## Why there are two modes

Deterministic mode uses stable 2,048-dimensional hashed lexical embeddings and scripted answer
payloads. A small visible synonym map supports known paraphrases. Scripted payloads still pass
through production structured-answer and trusted-citation validation. This mode is offline, fast,
repeatable, and suitable for normal tests and CI. It validates orchestration and invariants, not the
semantic quality of real models.

Live mode uses loopback Ollama, EmbeddingGemma, and Qwen. It can expose semantic retrieval misses,
false refusals, unsupported answers, wording variation, and prompt-injection behavior. It is slower
and can vary with model/runtime versions and hardware, so it is a deliberate local or scheduled
gate rather than a required part of every fast feedback loop.

## Reusing the production path

The evaluator reuses secure ingestion, chunking, transactional indexing, read-only search, overlap
suppression, bounded untrusted evidence, structured generation, fail-closed validation, and
application-owned citation mapping.

`answer_from_search` accepts the already completed `DatabaseSearchRun`. Evaluation can inspect the
exact ranking and generate from that same result without performing a second search. This avoids a
subtle test defect where the evidence being scored could differ from what Qwen receives.

## Strict manifest and corpus validation

The manifest rejects unknown fields, invalid types, duplicate cases, incoherent refusal
expectations, excessive values, unsafe paths, and traversal. Expected files must pass the normal
secure scanner. Each declared fact must exist in an expected source, and each canary must exist in
the synthetic corpus.

Facts are groups of acceptable text alternatives. For example, a time-period fact can accept
“calendar year,” “per year,” or “annually.” Every fact group must be present, but any declared
wording in a group is valid. This reduces false failures caused only by harmless Qwen phrasing.

## Metrics

| Metric | Meaning |
| --- | --- |
| Hit@K | At least one expected source appeared in selected top K |
| MRR | How early the first expected source appeared |
| Answer fact accuracy | Every declared fact appeared in the accepted answer |
| Citation validity | Every citation belonged to this exact retrieval run |
| Citation precision | Every citation belonged to an expected source |
| Refusal accuracy | Unsupported cases refused and supported cases answered |
| Security leakage count | A canary or attacker phrase reached observable output |
| Stage latency | Time spent indexing, retrieving, and generating |

Separating these metrics prevents incorrect tuning. If an expected source is rank one but answer
facts fail, lowering the retrieval threshold cannot fix the generation-stage problem.

## What the live baseline taught us

The first live run ranked the expected source first for every supported case, returned valid and
precise citations, safely refused the unsupported secret, and leaked no canaries. One benefits case
was initially marked failed because the rubric required the literal phrase “calendar year,” while
Qwen used equivalent annual wording.

That was a measurement defect, not a RAG defect. The correct change was fact alternatives in the
evaluator—not forcing Qwen to copy one phrase. Tests are software: a red result can identify a
product failure, fixture failure, or rubric failure, and engineers must diagnose which one changed.

## Privacy and security boundaries

Reports contain case IDs, booleans, counts, safe decision reasons, ranks, model names,
fingerprints, settings, and timings. They omit questions, passages, answers, vectors, prompts, raw
model JSON, canaries, and absolute paths. Disposable databases live under ignored
`.data/evaluation/` and are removed after a run.

Zero leakage proves only that declared attacks did not reach output in that run. Stronger
deterministic protections remain architectural: Qwen receives opaque evidence IDs, never receives
source metadata, and Python constructs citations from validated retrieval records. Canary testing
adds empirical coverage for answer-content influence that schema validation cannot eliminate.

## Exit codes

```text
exit 0  all required cases passed
exit 1  a quality/security gate failed, or evaluation could not run
exit 2  manifest, override, or configuration input was invalid
```

## Senior-engineer lessons

1. Measure retrieval, generation, provenance, refusal, and security independently.
2. Reuse production paths so the evaluator cannot drift from the product.
3. Use fast deterministic and slower live-model gates for different feedback loops.
4. Version corpus, expectations, model identity, chunking, and retrieval settings together.
5. Make reports observable without turning logs into a content-exfiltration channel.
6. Treat latency as baseline evidence, not a universal cross-hardware threshold.
7. Diagnose a failed test before tuning prompts or retrieval.
8. A small synthetic suite prevents known regressions but is not production coverage.

Interview-ready summary:

> I built a versioned synthetic RAG evaluation harness with deterministic and live modes. It creates
> a disposable index, reuses production retrieval and validation, reports Hit@K, MRR, fact,
> citation, refusal, security, and latency metrics, and fails automation without logging sensitive
> model or document content.

## Current limitations

- Seven synthetic cases are a regression seed, not statistically representative data.
- Fact checking is declared text matching, not semantic or human grading.
- Hashed lexical vectors validate wiring, not real embedding quality.
- Live results depend on installed Ollama/model versions and hardware.
- Latency is reported without a hardware-specific service-level gate.
- Canary coverage cannot represent every prompt-injection strategy.

Checkpoint 8 will document hardening, operational observability, threat boundaries, and the
production evolution path.

## Check your understanding

1. Why can deterministic evaluation pass while live evaluation fails?
2. What does Hit@5 reveal that MRR does not?
3. Why is rank-one retrieval insufficient to claim answer correctness?
4. Why must citations be both valid and precise?
5. Why are fact alternatives less brittle than one exact answer string?
6. Why was the initial benefits result a rubric failure?
7. Which injection protections are deterministic, and which are empirical?
8. What cases would you add for a real product domain?
