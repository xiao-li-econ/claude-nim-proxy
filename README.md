# claude-nim-proxy

A thin local translator that lets Claude Code use NVIDIA NIM models through the Anthropic Messages API.

The current implementation includes the Phase 1 translators, the Phase 2 local gateway,
and Phase 3 model discovery. It provides strict typed configuration, Anthropic/OpenAI
request and response translation, incremental SSE translation, an authenticated loopback
HTTP gateway, an NVIDIA upstream client, Anthropic-style error conversion, timeouts,
cancellation handling, and a credential-free NVIDIA model catalog cache. The OS
credential-store commands and one-command Claude Code runner are implemented in the later
phases described in [`plan.md`](plan.md).

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

## NVIDIA model catalog

Refresh the public NVIDIA model catalog without storing or sending an NVIDIA
credential:

```bash
uv run claude-nim models update
```

The cache is written atomically to:

```text
~/.config/claude-nim/models.json
```

It contains model IDs and capability metadata, never credentials. The bundled curated
metadata covers known NVIDIA models; unlisted models remain selectable with conservative
unknown capabilities. Per-model exceptions can be supplied in `config.toml`:

```toml
[model_overrides."nvidia/nemotron-3-super-120b-a12b"]
display_name = "My NVIDIA model"
context_window = 131072
max_output_tokens = 32768
reasoning = true
input = ["text"]
tool_calling = true
```

List cached IDs, reversible Claude Code aliases, and resolved capabilities:

```bash
uv run claude-nim models list
uv run claude-nim models list --json
```

When the cache matches the configured upstream, `serve` advertises its models through
`GET /v1/models` using IDs such as
`anthropic-nim/nvidia/nemotron-3-super-120b-a12b`. The `run`, `auth`, and `doctor`
commands remain skeletons in this phase; they do not yet launch Claude Code or manage
persistent credentials.

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
