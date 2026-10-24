# claude-nim-proxy

A thin local translator that lets Claude Code use NVIDIA NIM models through the Anthropic Messages API.

The current implementation includes the Phase 1 translators and the Phase 2 local
gateway. It provides strict typed configuration, Anthropic/OpenAI request and response
translation, incremental SSE translation, an authenticated loopback HTTP gateway, an
NVIDIA upstream client, Anthropic-style error conversion, timeouts, and cancellation
handling. The model catalog, OS credential-store commands, and one-command Claude Code
runner are implemented in the later phases described in [`plan.md`](plan.md).

## Current CLI

The command tree is available for inspection:

```bash
uv run claude-nim --help
uv run claude-nim run --help
```

`serve` now starts the local gateway. It requires an explicitly selected NVIDIA model
and, for the normal hosted upstream, an `NVIDIA_API_KEY` environment variable:

```bash
NVIDIA_API_KEY=nvapi-... \
  uv run claude-nim serve --model nvidia/nemotron-3-super-120b-a12b
```

The generated local bearer token and base URL are printed to standard error as an
export-ready snippet. The token can instead be supplied through `--token-stdin` or a
mode-`0600` file passed to `--token-file`.

The `run`, `auth`, `models`, and `doctor` commands remain skeletons in this phase; they
do not yet launch Claude Code or manage persistent credentials and model metadata.

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
