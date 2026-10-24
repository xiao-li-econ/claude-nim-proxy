# claude-nim-proxy

A planned thin local translator that lets Claude Code use NVIDIA NIM models through the Anthropic Messages API.

The project is currently in the planning/setup stage. See [`plan.md`](plan.md) for the architecture, protocol mappings, CLI design, credential handling, testing strategy, and implementation phases.

## Architecture

```text
Claude Code
  │ Anthropic Messages API
  │ POST /v1/messages
  ▼
Local proxy (127.0.0.1)
  │ translates Anthropic → OpenAI Chat Completions
  │ POST /v1/chat/completions
  ▼
NVIDIA NIM
  │
  ▼
Local proxy
  │ parses OpenAI JSON/SSE
  │ translates → Anthropic response/events
  ▼
Claude Code
```

The proxy translates requests, responses, tool calls, and streaming events between Anthropic's Messages API and NVIDIA NIM's OpenAI-compatible Chat Completions API.

## Development with uv

This project uses [uv](https://docs.astral.sh/uv/) for Python version management, dependency resolution, locking, and the development environment.

```bash
# Create/sync the environment and install all development dependencies
uv sync --all-groups

# Run the test suite
uv run pytest

# Lint and type-check
uv run ruff check .
uv run mypy
```

`uv.lock` is committed so dependency resolution remains reproducible across machines.
