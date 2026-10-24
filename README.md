# claude-nim-proxy

A thin local translator that lets Claude Code use NVIDIA NIM models through the Anthropic Messages API.

Phase 1 is implemented on the `feature/phase-1-translators` branch. It provides the
`claude-nim` command skeleton, strict typed configuration, pure Anthropic/OpenAI request
and response translators, and an incremental SSE translation state machine. The networked
gateway, model catalog, credential storage, and one-command runner are implemented in the
later phases described in [`plan.md`](plan.md).

## Current CLI

The command tree is available for inspection:

```bash
uv run claude-nim --help
uv run claude-nim run --help
```

The Phase 1 `run`, `serve`, `auth`, `models`, and `doctor` commands are skeletons and do
not yet make network requests or launch Claude Code.

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
