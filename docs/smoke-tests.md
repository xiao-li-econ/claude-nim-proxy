# Compatibility and smoke tests

## Automated offline compatibility suite

The normal test suite uses a local fake NVIDIA upstream and never makes an external
billable request:

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy
uv build
```

An opt-in suite launches the installed Claude Code CLI against the same local fake
upstream. It covers text streaming, a `Read` tool round trip, an image tool result,
foreground subagents, and background subagents:

```bash
CLAUDE_NIM_RUN_CLAUDE_COMPAT=1 \
  uv run pytest -q tests/test_claude_code_live.py -vv
```

The opt-in suite requires `claude` on `PATH`. It isolates Claude's home/configuration
under pytest's temporary directory and disables nonessential Claude traffic and
auto-updates. It does not contact NVIDIA and cannot consume NVIDIA API credit.

Record the tested Claude Code version with the results:

```bash
claude --version
```

## Live hosted NVIDIA smoke test

The following procedure sends real requests to hosted NVIDIA NIM and may consume API
credit. Run it only after explicit approval for a billable task.

### 1. Refresh the public catalog

```bash
uv run claude-nim models update
uv run claude-nim models list
```

Choose a model from the cached tool-capable candidates. Catalog membership does not
prove that the current NVIDIA account is entitled to call a model.

### 2. Configure and verify the credential

```bash
uv run claude-nim auth login
uv run claude-nim auth status
uv run claude-nim auth verify --model nvidia/<model-id>
```

`auth verify` sends one minimal authenticated generation request. A 404 indicates a
model entitlement/availability problem rather than a missing local credential.

### 3. Run diagnostics

```bash
uv run claude-nim doctor --model nvidia/<model-id>
```

`doctor` checks Claude Code, configuration, credential resolution, listener readiness,
the public catalog, and account entitlement. Its entitlement probe also sends one
minimal authenticated request.

### 4. Exercise Claude Code

Use a disposable project directory:

```bash
mkdir -p /tmp/claude-nim-smoke
cd /tmp/claude-nim-smoke
uv run claude-nim run . --model nvidia/<model-id>
```

In Claude Code, verify all of the following:

1. `/status` reports the loopback `ANTHROPIC_BASE_URL`.
2. A one-shot text prompt produces incremental output.
3. The model picker shows curated IDs beginning with `anthropic-nim/`, plus the explicitly selected model.
4. A request to read a local file completes the `tool_use` → tool result → final answer
   round trip.
5. A subagent and a background agent both use the selected NVIDIA model rather than an
   Anthropic-hosted model.
6. An image-capable selected model accepts an image tool result.

### 5. Verify cleanup

Exit Claude Code normally, then repeat it while terminating the proxy with `SIGTERM`.
No proxy or Claude child should remain:

```bash
pgrep -af 'claude-nim|claude' || true
ss -ltnp | grep 127.0.0.1 || true
```

The exact listener port is ephemeral by default. A fixed-port smoke test may use
`--listen-port <port>` and verify that the port is released afterward.

## Redacted diagnostics

For a failed live run, enable metadata-only debug tracing:

```bash
uv run claude-nim run . --model nvidia/<model-id> --debug
```

or:

```bash
CLAUDE_NIM_DEBUG=1 uv run claude-nim serve --model nvidia/<model-id>
```

Debug logs include request method/path, selected model, stream mode, and upstream
status. They exclude prompts, message bodies, tool schemas, and response content.
Authorization, API-key headers, the local token, and the configured NVIDIA credential
are redacted before output.
