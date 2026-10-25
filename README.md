# claude-nim-proxy

A thin local translator that lets Claude Code use NVIDIA NIM models through the Anthropic Messages API.

The current implementation includes the Phase 1 translators, the Phase 2 local gateway,
Phase 3 model discovery, the Phase 4 credential manager and one-command runner, and
Phase 5 compatibility hardening with redacted diagnostics. It provides strict typed
configuration, Anthropic/OpenAI request and response translation, incremental SSE
translation, an authenticated loopback HTTP gateway, an NVIDIA upstream client,
Anthropic-style error conversion, timeouts, cancellation handling, a credential-free
NVIDIA model catalog cache, secure CLI-managed credentials, an isolated Claude Code
launcher, and metadata-only debug tracing.

## One-command usage

Refresh the public model catalog, configure a credential, select a model, and launch
Claude Code in the current project:

```bash
uv run claude-nim models update
uv run claude-nim auth login
uv run claude-nim run . --model nvidia/nemotron-3-super-120b-a12b
```

With `default_model` set in `~/.config/claude-nim/config.toml` (or `NVIDIA_MODEL` set),
the model flag can be omitted:

```bash
uv run claude-nim run .
```

Arguments after `--` are passed to Claude Code verbatim:

```bash
uv run claude-nim run . --model nvidia/nemotron-3-super-120b-a12b -- --resume
```

`run` starts the loopback proxy on an ephemeral port, waits for its health check, creates
a temporary mode-`0600` Claude Code settings file, and launches:

```text
claude --settings <absolute-settings-path> [user-supplied Claude Code arguments...]
```

The temporary settings file contains only the local gateway URL, a random local bearer
token, and model-routing variables. It is deleted and the proxy is stopped when Claude
Code exits or receives a termination signal.

## Credential management

The CLI owns the NVIDIA credential lifecycle:

```bash
uv run claude-nim auth login
uv run claude-nim auth status
uv run claude-nim auth verify
uv run claude-nim auth logout
```

- `auth login` securely prompts with `getpass` and stores the key in the OS credential
  store. Use `--replace` to replace an existing stored credential.
- `auth status` reports only the source (`environment`, `credential store`, or `file
  fallback`) and never prints the key.
- `auth verify` sends one minimal authenticated NVIDIA request. This may consume a small
  amount of API credit.
- `auth logout` removes CLI-owned persistent storage but intentionally leaves an
  `NVIDIA_API_KEY` environment variable unchanged.

`NVIDIA_API_KEY` takes precedence over persistent storage and is never written
automatically. If no OS credential backend is available, the CLI uses
`~/.config/claude-nim/credentials.json` with mode `0600` and prints an explicit warning
that the secret is unencrypted.

The NVIDIA credential is removed from the Claude Code child environment and is never
placed in the temporary settings file, command line, model cache, or debug output. The
only credential inherited by Claude Code is the random local bearer token used by the
loopback proxy.

## Standalone gateway

`serve` runs only the translator for users who launch Claude Code separately:

```bash
NVIDIA_API_KEY=nvapi-... \
  uv run claude-nim serve --model nvidia/nemotron-3-super-120b-a12b
```

The generated local bearer token and base URL are printed to standard error as an
export-ready snippet. The token can instead be supplied through `--token-stdin` or a
mode-`0600` file passed to `--token-file`. Tokens are not accepted as command-line
arguments.

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
metadata (`src/claude_nim_proxy/data/nvidia_models.json`) covers the hosted models that
work well with Claude Code — the Nemotron 3 family, MiniMax M3.1 Flash, Kimi K2.6,
DeepSeek V4.1 Flash, GLM-5.3, Step 5 Preview, GPT-OSS 20B, and a few other
tool-calling chat models. Each entry records the context window, maximum output,
reasoning support, image support, and tool-calling support published by NVIDIA's model
cards and API reference. Values that NVIDIA does not publish are left unset and fall
back to the global configuration; unlisted models remain selectable with conservative
unknown capabilities. Per-model exceptions can be supplied in `config.toml`:

```toml
[model_overrides."nvidia/nemotron-3-super-120b-a12b"]
display_name = "My NVIDIA model"
context_window = 1048576
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

When the cache matches the configured upstream, the gateway advertises the bundled
curated models that are present in NVIDIA's current catalog through `GET /v1/models`,
using IDs such as `anthropic-nim/nvidia/nemotron-3-super-120b-a12b`. The explicitly
selected model is always included, even when it is not curated. The complete catalog
remains in `models.json` and `models list`; non-curated models remain directly
selectable with `--model`, `NVIDIA_MODEL`, or `default_model`.

## Diagnostics

```bash
uv run claude-nim doctor
```

`doctor` checks for Claude Code, model selection, credential resolution, loopback port
availability, the public NVIDIA catalog, and account entitlement to the selected model.
The entitlement probe sends one minimal authenticated request and may consume a small
amount of API credit.

## Redacted diagnostics

Enable metadata-only debug tracing with `--debug` or `CLAUDE_NIM_DEBUG=1`:

```bash
uv run claude-nim serve --model nvidia/nemotron-3-super-120b-a12b --debug
```

Debug output records request method/path, selected model, stream mode, and upstream
status. It never records prompts, message bodies, tool schemas, or response content.
Authorization/API-key headers, the random local token, and the configured NVIDIA key
are redacted.

## Compatibility smoke tests

The default test suite is offline and uses a fake NVIDIA upstream. An opt-in suite
launches the installed Claude Code CLI against that fake upstream to verify text,
tools, images, foreground subagents, and background subagents without contacting
NVIDIA:

```bash
CLAUDE_NIM_RUN_CLAUDE_COMPAT=1 \
  uv run pytest -q tests/test_claude_code_live.py -vv
```

The complete automated and live hosted-NVIDIA procedures are documented in
[`docs/smoke-tests.md`](docs/smoke-tests.md). Live NVIDIA requests may consume API
credit and require explicit approval before they are run.

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

The proxy translates requests, responses, tool calls, and streaming events between
Anthropic's Messages API and NVIDIA NIM's OpenAI-compatible Chat Completions API.

## Privacy

Prompts, project content, tool definitions, tool results, and model responses are sent
to the configured NVIDIA upstream. The local gateway binds to `127.0.0.1` by default and
requires a per-run bearer token on Claude-facing endpoints.

## Development with uv

This project uses [uv](https://docs.astral.sh/uv/) for Python version management,
dependency resolution, locking, and the development environment.

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
