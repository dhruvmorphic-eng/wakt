# Contributing to Wakt

Thanks for your interest in contributing! This guide will help you get started quickly.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Scope

Wakt is a fast, local, on-device decision engine. Contributions should stay in that spirit: they
must run in the user's own process or on their own hardware, with no dependency on a hosted service.

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Git

### Development Setup

```bash
# Clone the repo
git clone https://github.com/dhruvmorphic-eng/wakt.git
cd wakt

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[mcp,serve]"
```

If you use [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[mcp,serve]"
```

### Optional Extras

Install only what your change needs:

| Extra | What it adds |
|---|---|
| `serve` | FastAPI HTTP server |
| `mcp` | MCP stdio server |
| `langchain` | LangChain / LangGraph integrations |
| `llamaindex` | LlamaIndex selectors |
| `crewai` | CrewAI routing |
| `onnx` | ONNX Runtime export & inference |
| `fast` | TileLang GPU fast path |
| `structured` | Pydantic model support |

## Running Tests

Most suites are plain scripts:

```bash
python tests/test_router.py
python tests/test_criteria.py
python tests/test_hooks.py
python tests/test_hooks_api.py
```

Some use pytest:

```bash
python -m pytest tests/test_serve.py      # needs [serve] extra
python -m pytest tests/test_onnx.py       # needs [onnx] extra
```

Before opening a PR, run lint and compile checks:

```bash
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
```

## Ways to Contribute

- **Report bugs** — include a minimal reproduction, your Python/OS/device info, and what you expected vs. what happened.
- **Improve docs** — the `docs/` directory and docstrings in `laya/` are always welcome targets.
- **Fix bugs or add features** — keep PRs focused on one logical change.
- **Share benchmarks** — measured performance data is treated as a first-class contribution.

## Style Guide

- Keep the public API stable. If a change moves it, update `tests/test_hooks_api.py` in the same PR.
- Prefer the standard library and existing dependencies over adding new ones.
- Leave a runnable check behind for non-trivial logic — an assert-based script is enough.
- Comment only where the code cannot say it (non-obvious reasons, hardware caveats).
- Match the surrounding style rather than a personal preference.

## Commits

Use conventional commit prefixes:

```
feat(agent): add batch abstention support
fix(router): handle empty state gracefully
perf(common): reduce allocation in option rendering
docs(hooks): add lifecycle diagram
test(batch): cover sort_by_length edge case
```

Keep one logical change per commit.

## Pull Requests

Your PR should answer four questions:

1. **What** changed
2. **Why** — the concrete use case
3. **How it was verified** — exact commands you ran
4. Any follow-ups you deliberately left out

Before opening:

- Rebase onto the latest `main`
- Keep it focused — split unrelated work into separate PRs
- Update docs or examples when the public API changes
- Do not commit secrets, tokens, or large binary files
- If your change moves numbers, report before and after

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).

## Attribution

This project is a derivative of [Laya](https://github.com/NandhaKishorM/laya) by
[Convai Innovations](https://huggingface.co/convaiinnovations). See [NOTICE](NOTICE).
