# Manual Checkpoint Verification

These guides turn each completed checkpoint into a repeatable hands-on lab. They are intended for
new contributors, reviewers, and learners who want to understand behavior rather than only run the
automated test suite.

## How the documentation is organized

- `docs/learning/` explains concepts, mental models, tradeoffs, and limitations.
- `docs/testing/` provides commands, expected results, and interpretation.
- `examples/` contains committed synthetic inputs that are safe to inspect and print.
- `.data/` is ignored by Git and is used for disposable local experiments.

## Recommended workflow

For each checkpoint:

1. Read its learning record.
2. Run its manual verification guide from the repository root.
3. Compare the observed output with the expected behavior.
4. Perform the failure and privacy experiments.
5. Run the automated quality checks.
6. Answer the explain-back questions without consulting the guide.

## Available guides

- [Checkpoint 0 — Environment and runtime readiness](checkpoint-0.md)
- [Checkpoint 1 — Secure file discovery and parsing](checkpoint-1.md)
- [Checkpoint 2 — Deterministic chunking](checkpoint-2.md)
- [Checkpoint 3 — Local embeddings](checkpoint-3.md)
- [Checkpoint 4 — SQLite vector index](checkpoint-4.md)
- [Checkpoint 5 — Read-only vector search](checkpoint-5.md)
- [Checkpoint 6 — Grounded answers and trusted citations](checkpoint-6.md)
- [Checkpoint 7 — Evaluation and security regression suite](checkpoint-7.md)
- [Checkpoint 8 — Hardening and senior-engineer retrospective](checkpoint-8.md)

Commands target Windows PowerShell because Windows is the project's baseline platform.
