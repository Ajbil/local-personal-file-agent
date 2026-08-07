# Project Guidance

## Mission

Build a local personal-file Retrieval-Augmented Generation (RAG) agent while using every checkpoint as a deliberate learning exercise. The primary outcome is not merely a working demo: it is deeper AI engineering, backend engineering, security, testing, system-design, and production-readiness skill appropriate for a senior software engineer targeting strong startup and MNC roles.

## Learner Context

- The project owner is an experienced software developer but is new to RAG and several modern AI concepts.
- Do not assume prior AI/ML knowledge when introducing a concept.
- Do not blindly accept the owner's proposed approach. If it is incorrect, unsafe, over-engineered, or misses a useful learning opportunity, explain why and recommend the stronger approach.
- Connect implementation details to senior-engineer concerns: tradeoffs, failure modes, security, observability, testing, scalability, maintainability, and how to explain the design in an interview.

## Working Method

- Work checkpoint by checkpoint according to `docs/learning-first-implementation-plan.md`.
- Before implementation, explain the checkpoint's problem, mental model, dependencies, design decisions, and definition of done.
- Prefer small, inspectable implementations over framework magic. Do not introduce LangChain, LlamaIndex, a vector database, or a web UI before the baseline direct implementation is understood and evaluated.
- Make intermediate artifacts visible: discovered files, chunks and offsets, vector dimensions, retrieval scores, selected context, validated answer structures, and evaluation metrics. Never expose personal document contents unnecessarily.
- Add tests alongside each checkpoint, including important failure and security cases.
- Keep commits small and scoped to one understood milestone.
- At the end of a checkpoint, summarize what was built, what was learned, remaining limitations, and a short set of questions or experiments the owner can use to verify understanding.
- Add or update `docs/testing/checkpoint-N.md` at the end of every checkpoint. Include prerequisites, copy-pasteable PowerShell commands, expected behavior, interpretation, safe failure/privacy experiments, and automated quality checks so another learner can independently verify the milestone.
- Keep committed synthetic inputs under `examples/checkpoint-N/` and use the Git-ignored `.data/` directory for disposable manual-testing artifacts. Do not place testing instructions inside `examples/`.
- Preserve user-authored or unrelated changes and ask before destructive or externally visible actions.

## Baseline Decisions

- Platform: Windows and Python 3.12 managed through `uv`.
- Runtime: loopback-only Ollama.
- Answer model: `qwen3.5:4b`.
- Embedding model: `embeddinggemma`.
- Initial formats: UTF-8 Markdown and plain text.
- Initial interface: command line.
- Initial storage: SQLite with brute-force NumPy cosine similarity.
- Source folders are explicitly allowlisted and treated as read-only.
- Personal documents, indexes, secrets, raw prompts, and sensitive evaluation output must not be committed.
- Normal inference stays local. Internet access is only expected for initial tool, dependency, and model downloads.

## Sources of Truth

- Learning and implementation roadmap: `docs/learning-first-implementation-plan.md`
- Manual checkpoint verification index: `docs/testing/README.md`
- Original Notion export: `Local Personal File Agent 060c2786553b82208d268122f958b13d.md`
- Future usage and setup instructions: `README.md`
