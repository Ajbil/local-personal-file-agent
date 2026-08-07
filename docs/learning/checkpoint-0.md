# Checkpoint 0 Learning Record

## Objective

Create a reproducible Python foundation and prove whether the local Ollama runtime, embedding model, and answer model are ready before implementing RAG behavior.

## Environment Observed

| Component | Observation |
| --- | --- |
| Operating environment | 64-bit Windows |
| CPU | 11th Gen Intel Core i5-1135G7 |
| RAM | Approximately 15.7 GB |
| Python | CPython 3.12.10 |
| uv | 0.12.0 |
| NVIDIA runtime | Not detected |
| Ollama | 0.32.6; loopback API validated |
| EmbeddingGemma | Installed as `embeddinggemma:latest`; live smoke test passed |
| Qwen 3.5 4B | Installed; live schema-constrained generation passed |

CPU execution is the compatibility baseline. GPU acceleration will only be claimed if Ollama reports VRAM allocation.

## What Was Built

- `uv` project with a committed dependency lockfile.
- Installable `src`-layout Python package and `file-agent` command.
- Typed settings with loopback-only Ollama validation.
- Direct HTTP adapter for Ollama version, model inventory, embedding, structured chat, and runtime APIs.
- Ordered `doctor` diagnostics with human and JSON output.
- Safe failure translation that does not print response bodies or vector values.
- Deterministic tests using an in-memory HTTP transport and gateway fakes.

## Validation Evidence

```text
Ruff: passed
mypy strict mode: passed
pytest: 33 passed
coverage: 96.02%
```

Live values recorded:

- Ollama version: `0.32.6`.
- Installed embedding model: `embeddinggemma:latest`.
- Embedding dimension: `768`.
- Embedding load duration: `4,218.384 ms`.
- Embedding total duration: `4,418.684 ms`.
- Qwen load duration: `26,204.793 ms`.
- Qwen total duration: `154,348.501 ms`.
- Reported VRAM allocation: `0` bytes; CPU execution is active.
- Qwen structured output: schema-valid JSON accepted by Pydantic.

## Important Concepts

### Runtime versus model

Ollama is the program that loads and executes models. EmbeddingGemma and Qwen are model weights with different jobs. Installing Ollama does not install every model, and downloading a model does not create our Python application.

### Dependency declaration versus lockfile

`pyproject.toml` declares compatible dependency ranges and project tooling. `uv.lock` records the exact resolved dependency graph used to reproduce the environment.

### Deterministic versus live tests

Normal tests replace the external model runtime with controlled fakes. This proves our application decisions and failure handling. A live smoke test proves the actual runtime/models work on this machine. Both are required, but they answer different questions.

### Local does not automatically mean secure

The application rejects non-loopback Ollama URLs. This reduces the risk that personal text is accidentally sent across a LAN or to a remote host through configuration.

## Completion Gate

Checkpoint 0 completed successfully with:

```powershell
uv sync --locked
uv run file-agent doctor
```

The final doctor report passed Python, loopback configuration, Ollama connectivity, model inventory, 768-dimensional embedding generation, and schema-constrained Qwen generation. Ollama reported CPU execution as a non-blocking performance warning.

For a complete reproducible lab, follow the [Checkpoint 0 manual verification guide](../testing/checkpoint-0.md).

## Explain-Back Questions

1. Why are Python, `uv`, Ollama, EmbeddingGemma, and Qwen separate components?
2. Why does the application reject an Ollama server on a LAN IP even though it may be owned by you?
3. What does deterministic testing prove that a live model test does not?
4. What does a live model test prove that mocked tests cannot?
5. Why does the doctor report embedding dimensions but not vector values?
6. What evidence would justify using a smaller answer model during development?


## learning as a senior developer - from claude 
Stepping back from the file-by-file details, here's what's actually worth carrying forward as reusable senior-engineer patterns — this codebase is small but demonstrates a lot of "production-grade" habits that show up in any language.

1. Validate at the boundary, trust everywhere else
config.py validates env vars once at startup (loopback-only URL, non-empty model names). ollama.py validates every HTTP response against a schema (_validate_response). Once data passes through these gates, the rest of the code treats it as trustworthy — no defensive re-checking scattered everywhere. Lesson: identify your system's actual boundaries (user input, network responses, config) and put validation exactly there, not everywhere.

2. Translate external errors into your own vocabulary
httpx.TimeoutException / ConnectError / HTTPStatusError all get caught in one place (_request_json) and re-raised as your own OllamaConnectionError / OllamaResponseError. Nothing downstream ever needs to know about httpx. Lesson: don't let third-party library exceptions leak through your whole call stack — wrap them once at the integration point so callers depend on your error types, which you control and can test against.

3. Depend on interfaces, not concrete classes (Protocol / duck typing)
doctor.py's run_doctor() takes an OllamaGateway (a Protocol), never OllamaClient directly. This is what let tests substitute HealthyGateway, OfflineGateway, etc. — plain classes with no network code at all — with zero mocking-library magic. Lesson: when a function's real dependency is "something with these methods," say that explicitly (interface/protocol), and your code becomes trivially testable and swappable (e.g., swap Ollama for OpenAI later without touching doctor.py).

4. Fail fast, in dependency order, and stop cleanly
run_doctor runs checks in an order where later checks assume earlier ones succeeded (can't check models if Ollama's unreachable) and explicitly returns early on failure instead of blowing up or silently continuing with garbage state. Lesson: when checks/steps have real dependencies, encode that ordering explicitly and short-circuit — don't let step 5 run on top of step 2's failure and produce a confusing report.

5. Return structured results, not printed text
run_doctor() returns a DoctorReport object (data), and only cli.py decides how to render it (human text or JSON). The core logic never calls print. Lesson: separate "compute the result" from "display the result." This is why the same report could trivially support --json — the logic layer doesn't know or care how it'll be shown.

6. Make illegal states hard to represent (type system as a design tool)
CheckStatus is an enum, not a raw string — you can't accidentally check status == "Pass" (wrong case) and silently get a bug. full_readiness can only be True if success is also True (enforced in _finalize, not left as an assumption). Lesson: encode invariants (valid states, relationships between fields) in code/types so a caller can't construct a nonsensical result by mistake.

7. Security/privacy thinking baked into normal code, not bolted on
Loopback-only URL validation, discarding embedding vector values while keeping only their dimension, stripping response bodies out of error messages (test_http_error_does_not_expose_response_body), never echoing a possibly-sensitive URL back in error output. Lesson: in any tool handling sensitive data, ask at each boundary "what's the minimum I need to keep/expose here" — and write a test that actively asserts the sensitive thing is absent from output, not just that the safe thing is present.

8. Test each layer in isolation by faking its one dependency
test_config.py → no fakes needed, config has no dependencies besides env vars (monkeypatch.setenv)
test_ollama.py → fakes the network (httpx.MockTransport)
test_doctor.py → fakes the gateway (hand-written classes satisfying the Protocol)
test_cli.py → fakes both OllamaClient and run_doctor (monkeypatch.setattr)
Each test file mocks exactly one layer down, never more. Lesson: good architecture (boundaries + interfaces from points 1–3) is what makes clean, fast, isolated unit tests possible in the first place — testability isn't bolted on afterward, it falls out of good separation of concerns.

9. Tests double as executable specification
Reading test names alone (test_offline_runtime_fails_without_running_dependent_checks, test_skip_generation_does_not_claim_full_readiness) tells you the intended behavior without reading implementation. Lesson: name tests as behavior statements — a new developer (or future you) should understand what the system promises just by skimming test names.

The single biggest transferable habit: this codebase draws a hard line between "things that talk to the outside world" (env vars, HTTP) and "things that make decisions" (doctor logic), and everything else — testability, error handling, the ability to add --json for free — falls out of maintaining that line. When you look at any new codebase, the fastest way to understand it is to ask: where are the boundaries, and what crosses them?
