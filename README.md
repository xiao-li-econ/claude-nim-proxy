# claude-nim-proxy

A thin local translator that lets Claude Code use NVIDIA NIM models through the Anthropic Messages API.

The current implementation includes the Phase 1 translators, the Phase 2 local gateway,
Phase 3 model discovery, the Phase 4 credential manager and one-command runner, and
Phase 5 compatibility hardening with redacted diagnostics. It provides strict typed
configuration, Anthropic/OpenAI request and response translation, incremental SSE
translation, an authenticated loopback HTTP gateway, an NVIDIA upstream client,
Anthropic-style error conversion, timeouts, cancellation handling, a credential-free
active model JSON, secure CLI-managed credentials, an isolated Claude Code launcher,
and metadata-only debug tracing.

## Installation

The project is not published to PyPI yet, so install it directly from the repository.

Requirements: Python 3.11 or newer, the [Claude Code CLI](https://code.claude.com/docs/en/setup)
(`claude` on `PATH`), and an NVIDIA API key. The [Pi CLI](https://github.com/badlogic/pi-mono)
is optional and only used to synchronize the model list.

Install the `claude-nim` command with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/xiao-li-econ/claude-nim-proxy.git
claude-nim --version
```

This creates an isolated environment and places the executable in uv's bin directory
(`~/.local/bin` by default), which must be on `PATH`. Upgrade with
`uv tool upgrade claude-nim-proxy` and remove it with `uv tool uninstall claude-nim-proxy`.
Pin a release with `uv tool install "git+https://github.com/xiao-li-econ/claude-nim-proxy.git@v0.1.0"`.

Run a single command without installing anything:

```bash
uvx --from git+https://github.com/xiao-li-econ/claude-nim-proxy.git claude-nim run .
```

With pipx:

```bash
pipx install git+https://github.com/xiao-li-econ/claude-nim-proxy.git
```

Or in a plain virtual environment:

```bash
python3 -m venv ~/.venvs/claude-nim
source ~/.venvs/claude-nim/bin/activate
pip install git+https://github.com/xiao-li-econ/claude-nim-proxy.git
```

To work on the code itself, clone the repository:

```bash
git clone https://github.com/xiao-li-econ/claude-nim-proxy.git
cd claude-nim-proxy
uv sync --all-groups
uv run claude-nim run .
```

After installing, configure a credential and launch Claude Code:

```bash
claude-nim auth login
claude-nim models update   # optional; requires the Pi CLI
claude-nim run .
```

## One-command usage

Optionally synchronize the model list from Pi, configure a credential, and launch Claude
Code in the current project:

```bash
uv run claude-nim models update
uv run claude-nim auth login
uv run claude-nim run .
```

The model flag can be omitted. Model selection follows this precedence:

1. `--model`
2. `NVIDIA_MODEL`
3. `default_model` in `~/.config/claude-nim/config.toml`
4. the shipped default model, `minimaxai/minimax-m3.1-flash`

Set a different default with:

```bash
uv run claude-nim config set default_model nvidia/nemotron-3.5-lightning-30b-a3b
```

Arguments after `--` are passed to Claude Code verbatim:

```bash
uv run claude-nim run . --model nvidia/nemotron-3.5-lightning-30b-a3b -- --resume
```

`run` starts the loopback proxy on an ephemeral port, waits for its health check, creates
a temporary mode-`0600` Claude Code settings file, and launches:

```text
claude --settings <absolute-settings-path> [user-supplied Claude Code arguments...]
```

The temporary settings file contains only the local gateway URL, a random local bearer
token, and model-routing variables. It is deleted and the proxy is stopped when Claude
Code exits or receives a termination signal.

## Configuration

`config.toml` holds general proxy settings and is created only by an explicit command:

```bash
uv run claude-nim config show
uv run claude-nim config show --json
uv run claude-nim config set default_model nvidia/nemotron-3.5-lightning-30b-a3b
```

- `config show` prints the effective configuration and where each value comes from
  (`config.toml`, an environment variable, or the built-in default).
- `config set <key> <value>` validates the value, preserves the other settings, and
  rewrites `~/.config/claude-nim/config.toml` atomically with mode `0600`.
- Supported keys: `default_model`, `verify_model`, `upstream_base_url`, `upstream_auth`,
  `listen_host`, `listen_port`, `context_window`, `max_output_tokens`,
  `disable_experimental_betas`, `upstream_connect_timeout_seconds`,
  `upstream_read_timeout_seconds`, and `debug`.

The file can also be edited by hand; `config set` rewrites it and therefore drops any
comments it contained.

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
placed in the temporary settings file, command line, active model JSON, or debug output. The
only credential inherited by Claude Code is the random local bearer token used by the
loopback proxy.

## Standalone gateway

`serve` runs only the translator for users who launch Claude Code separately:

```bash
NVIDIA_API_KEY=nvapi-... \
  uv run claude-nim serve --model nvidia/nemotron-3.5-lightning-30b-a3b
```

The generated local bearer token and base URL are printed to standard error as an
export-ready snippet. The token can instead be supplied through `--token-stdin` or a
mode-`0600` file passed to `--token-file`. Tokens are not accepted as command-line
arguments.

## Active NVIDIA model list

The active model list is stored in a credential-free JSON file:

```text
~/.config/claude-nim/models.json
```

The shipped bundled list is used only to initialize or reset that file:

```text
src/claude_nim_proxy/data/nvidia_models.json
```

If the Pi CLI is installed, synchronize the active list from Pi's NVIDIA provider:

```bash
uv run claude-nim models update
```

This runs `pi --list-models nvidia`. A successful, non-empty result replaces the active
JSON completely; models no longer listed by Pi are removed. Pi reports context, output,
reasoning, and image capabilities but not tool-calling support, so tool-calling remains
unknown. If Pi is unavailable, fails, times out, or returns an empty/malformed list,
the active JSON is preserved for manual maintenance.

Users without Pi can edit `~/.config/claude-nim/models.json` directly and set its
top-level `"source"` to `"manual"`. A successful `models update` overwrites manual
changes. Restore the shipped list with:

```bash
uv run claude-nim models reset
```

List active IDs, reversible Claude Code aliases, and resolved capabilities:

```bash
uv run claude-nim models list
uv run claude-nim models list --json
```

The gateway advertises every model in the active JSON through `GET /v1/models`, using
IDs such as `anthropic-nim/nvidia/nemotron-3.5-lightning-30b-a3b`. The explicitly
selected model is always included, even when it is absent from the active list. Models
outside the list remain directly selectable with `--model`, `NVIDIA_MODEL`, or
`default_model`, using conservative capability fallbacks.

To refresh from Pi as part of a run:

```bash
uv run claude-nim run . --refresh-models
```

If the refresh fails, `run` warns and continues with the existing active JSON. The
`models` command help output shows the portable active and bundled JSON paths.

## Diagnostics

```bash
uv run claude-nim doctor
```

`doctor` checks for Claude Code, the effective model (including its source), credential
resolution, loopback port availability, and account entitlement to that model. The
entitlement probe sends one minimal authenticated request and may consume a small amount
of API credit.

## Redacted diagnostics

Enable metadata-only debug tracing with `--debug` or `CLAUDE_NIM_DEBUG=1`:

```bash
uv run claude-nim serve --model nvidia/nemotron-3.5-lightning-30b-a3b --debug
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

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
