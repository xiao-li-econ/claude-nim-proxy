# Implementation Plan: Thin Claude Code ↔ NVIDIA NIM Proxy

- **Status:** The MVP is merged into `main` through PR #4; the curated model-picker refinement is merged into `main` through PR #5; expansion of the curated catalog to every NVIDIA model visible through Pi is complete in open PR #6.
- **Research snapshot:** 2026-10-24
- **Verified local versions:** Python 3.12.3, Claude Code 2.1.310, uv 0.9.2, Node.js 24.14.1
- **Repository:** Git repository; Phase 4 is implemented in commit `361e816` and Phase 5 in commit `f6ae342`, merged into `main` through PR #4 as merge commit `e377a50`; the curated model-picker refinement is commit `ef609ec`, merged into `main` through PR #5 as merge commit `8717d6b`; the Pi-visible curated catalog expansion is commit `dbf3cf8` in open PR #6.
- **Current implementation:** The CLI/configuration skeleton, pure Anthropic/OpenAI translators, SSE state machine, authenticated aiohttp gateway, NVIDIA upstream client, Anthropic error conversion, timeout handling, cancellation handling, NVIDIA catalog parsing, bundled capability metadata, an atomic credential-free model cache, model commands, OS credential-store integration, the warned mode-0600 fallback, auth commands, the one-command Claude Code runner, doctor diagnostics, redacted metadata-only tracing, current-Claude compatibility coverage, and documented smoke tests are implemented.
- **Next action:** Review and merge PR #6; restart Claude Code to load the expanded 27-model picker.

## Implementation tracker

This table is the durable handoff state between sessions. Update it whenever a phase starts, completes, becomes blocked, changes scope, or gains new validation evidence. Record the validation commands and Git commit for completed phases.

| Phase | Status | Validation / commit | Notes |
|---|---|---|---|
| Project scaffold | Complete | `uv sync --all-groups`, Ruff, mypy, and `uv build` passed; commit `da83fc5` | Package metadata, dependencies, lockfile, lint/type/test configuration, and minimal package created. |
| Phase 1 — Scaffold completion and pure translators | Complete | 51 tests passed; Ruff, mypy, and `uv build` passed; commit `3122cc8` | Added the CLI skeleton, strict typed configuration, pure request/response translators, incremental SSE state machine, and focused unit tests. |
| Phase 2 — Local gateway | Complete | 89 tests passed; Ruff, mypy, and `uv build` passed; commit `7d7c305` | Added the aiohttp server, local bearer authentication, NVIDIA upstream client, health/model endpoints, Anthropic error conversion, timeouts, cancellation handling, standalone `serve`, and fake-upstream integration tests. |
| Phase 3 — Models and discovery | Complete | 117 tests passed; Ruff, mypy, and `uv build` passed; commits `61723cf`, `33dab7a` | Added public NVIDIA catalog refresh, bundled capability metadata, capability precedence, atomic credential-free cache storage, reversible aliases, model list/update commands, cached gateway discovery, and model-specific image capability checks. The curated overlay was expanded to 19 hosted agentic/tool-calling models with NVIDIA-published context, output, reasoning, image, and tool-calling values. |
| Phase 4 — Credentials and one-command runner | Complete | 133 tests passed; Ruff, mypy, and `uv build` passed; commit `361e816` | Added `auth login/status/verify/logout`, environment/keyring/file-fallback credential resolution, secure prompts, a one-command Claude Code launcher, isolated mode-0600 settings, signal forwarding and cleanup, and doctor diagnostics. |
| Phase 5 — Compatibility hardening and documentation | Complete | 137 default tests passed (5 live-Claude tests skipped); all 5 opt-in Claude Code 2.1.310 compatibility tests passed; Ruff, mypy, and `uv build` passed; commit `f6ae342` | Added metadata-only redacted tracing, a 2,000-delta long-stream test, opt-in installed-Claude coverage for text/tools/images/subagents/background agents, final README guidance, and documented offline/live smoke procedures. Hosted NVIDIA inference was not run because it requires explicit billable-request approval. |
| Post-MVP refinement — Curated model picker | Complete | 138 default tests passed (5 live-Claude tests skipped); all 5 opt-in Claude Code compatibility tests passed; Ruff, mypy, and `uv build` passed; commit `ef609ec` | Claude Code discovery now advertises only bundled curated models present in the current catalog plus the explicitly selected model. The complete catalog remains cached and selectable through explicit configuration, and the stale Nemotron Nano alias was removed. |
| Post-MVP refinement — Pi-visible curated catalog | Complete | 139 default tests passed (5 live-Claude tests skipped); all 5 opt-in Claude Code compatibility tests passed; Ruff, mypy, and `uv build` passed; credential-free cache refreshed to 85 models; commit `dbf3cf8`; open PR #6 | Adds all 22 NVIDIA models exposed by `pi --list-models nvidia` to the bundled curated catalog while retaining the five existing curated models that Pi does not expose. |

**Current next action:** Review and merge PR #6; restart Claude Code to load the expanded 27-model picker. All implementation decisions in section 19 are confirmed and binding.

## 1. Goal

Build a small Python CLI that:

1. Runs a loopback-only translator exposing the Anthropic Messages API expected by Claude Code.
2. Translates Anthropic Messages requests, responses, tool calls, and SSE streams to/from NVIDIA NIM's OpenAI-compatible Chat Completions API.
3. Starts the translator and Claude Code together in a chosen project directory with one command.
4. Provides NVIDIA model refresh commands and secure CLI-managed credential storage.
5. Avoids LiteLLM and general-purpose provider/router abstractions.

Distribution name: package `claude-nim-proxy`, executable `claude-nim`.

## 2. Important research findings

### Claude Code gateway contract

The official gateway documentation establishes the following contract:

- Claude Code selects the Anthropic Messages format through `ANTHROPIC_BASE_URL`.
- For a bearer-token gateway, `ANTHROPIC_AUTH_TOKEN` is the appropriate credential variable.
- The required inference endpoint is `POST /v1/messages`; Claude Code may append `?beta=true`, so routing must match the path and ignore the query.
- `POST /v1/messages/count_tokens` is optional. Without it, Claude Code falls back to an estimate.
- `GET /v1/models` is optional and powers gateway model discovery when `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` is set.
- `HEAD /api/hello` is a best-effort startup probe and may be rejected without breaking Claude Code.
- Streaming must not be buffered. Claude Code expects the ordered Anthropic event sequence `message_start` → `content_block_start`/`content_block_delta`/`content_block_stop` → `message_delta` → `message_stop`.
- Response headers matter: streamed responses should use `text/event-stream`; `retry-after`, `x-should-retry`, and rate-limit headers should be preserved or sensibly synthesized.
- Claude Code sends current Anthropic beta fields and may add new ones over time. A gateway fronting a non-Anthropic upstream must bridge or deliberately drop unsupported fields rather than blindly forwarding them.

### Claude Code model discovery constraint

Claude Code's discovery endpoint accepts a response shaped like:

```json
{
  "data": [
    {
      "id": "anthropic-nim/nvidia/nemotron-3-super-120b-a12b",
      "display_name": "NVIDIA Nemotron 3 Super 120B A12B (NIM)",
      "description": "NVIDIA NIM model"
    }
  ]
}
```

Claude Code currently keeps only discovered IDs containing `claude` or `anthropic`, case-insensitively. Raw NVIDIA IDs such as `nvidia/nemotron-3-super-120b-a12b` are ignored by the picker. The proxy therefore uses this reversible gateway alias:

```text
anthropic-nim/<actual NVIDIA model ID>
```

The translator strips `anthropic-nim/` before calling NVIDIA. Explicit raw NVIDIA IDs will also be accepted, even though they will not appear through discovery.

### NVIDIA NIM contract

- Hosted NIM base URL: `https://integrate.api.nvidia.com/v1`.
- Hosted authentication: `Authorization: Bearer $NVIDIA_API_KEY`.
- Hosted inference: `POST /v1/chat/completions`.
- Model catalog: `GET /v1/models`, returning OpenAI-style `{ "object": "list", "data": [...] }`.
- Tool calling is model- and backend-dependent. Claude Code is not useful without reliable tool calling.
- The hosted NVIDIA endpoint was checked during planning and exposes Chat Completions but returns 404 for `/v1/messages`, so a translator is needed for that target.
- This project targets hosted NVIDIA NIM only. Self-hosted/local NIM deployment and the native Anthropic-compatible endpoint found in some local NIM releases are explicitly out of scope.

### Model catalog and credential ownership

NVIDIA's `/v1/models` endpoint is the authoritative source of model IDs and can be queried without a credential. It is a global catalog rather than a per-account entitlement list: it does not indicate which models are free, which are paid, or which a particular key may call. The proxy will present catalog discovery separately from account availability.

The CLI will own the complete NVIDIA credential lifecycle: secure prompt, validation, storage, status, replacement, and removal. No external agent or credential helper is required.

## 3. Scope

### MVP scope

- Anthropic Messages frontend on `127.0.0.1`.
- OpenAI Chat Completions upstream translation.
- Streaming and non-streaming responses.
- Text, system prompts, images where supported by the selected model, custom tools, parallel tool calls, and tool results.
- Claude model alias/fallback handling.
- Claude-compatible model discovery.
- One-command proxy plus Claude Code launcher.
- NVIDIA model cache/refresh commands and secure CLI-managed credential storage.
- Hosted NVIDIA NIM as the production upstream. The base URL remains configurable for automated fake-upstream tests, but alternate providers and local model deployment are not product goals.
- Clear diagnostics for missing Claude Code, missing credentials, unsupported models, and failed upstream calls.

### Explicit non-goals for the MVP

- Multi-provider routing, quotas, budgets, billing, or a dashboard.
- Starting or managing NIM containers/GPU deployments.
- Anthropic prompt-cache semantics or cache token accounting.
- Preserving Anthropic signed thinking blocks; OpenAI-compatible NIM responses cannot create valid Anthropic signatures.
- Anthropic server-side tools such as web search, web fetch, code execution, advisor, or MCP tool search.
- Reimplementing Claude Code's system prompt or otherwise making a non-Claude model behave exactly like Claude.
- Translating every future Anthropic beta feature. Unknown fields will be ignored safely and recorded in debug mode.

## 4. CLI

```text
claude-nim run [PROJECT_DIR] [OPTIONS] [-- CLAUDE_ARGS...]
claude-nim serve [OPTIONS]
claude-nim auth login [OPTIONS]
claude-nim auth status [OPTIONS]
claude-nim auth verify [OPTIONS]
claude-nim auth logout [OPTIONS]
claude-nim models list [OPTIONS]
claude-nim models update [OPTIONS]
claude-nim doctor [OPTIONS]
```

### Primary one-command usage

```bash
claude-nim run /path/to/project \
  --model nvidia/nemotron-3-super-120b-a12b
```

With a configured default model:

```bash
claude-nim run .
```

Additional Claude Code arguments are passed through verbatim:

```bash
claude-nim run . --model nvidia/nemotron-3-super-120b-a12b -- --resume
```

### Subcommand responsibilities

| Command | Behavior |
|---|---|
| `run` | Resolve configuration and key, start the local translator, launch Claude Code in `PROJECT_DIR`, then cleanly stop the translator. |
| `serve` | Run only the translator, for users who want to manage Claude Code separately. |
| `auth login` | Securely prompt for the NVIDIA API key, optionally replace an existing credential, and store it in the OS credential store. |
| `auth status` | Report whether a credential is configured and where it came from without revealing the key. |
| `auth verify` | Validate the credential with an explicit, potentially billable minimal NVIDIA request. |
| `auth logout` | Remove the credential from the OS credential store without changing an environment-provided key. |
| `models list` | Show cached/refreshed NVIDIA models and their local Claude aliases. |
| `models update` | Refresh model IDs directly from NVIDIA and merge them with curated capability metadata. |
| `doctor` | Check Claude Code, credential resolution, upstream reachability, model availability, and loopback port readiness. |

## 5. One-command startup flow

`claude-nim run` will perform these steps in order:

1. Resolve and validate the project directory; it must exist and be a directory.
2. Resolve the upstream model in this order:
   - `--model`;
   - `NVIDIA_MODEL` environment variable;
   - user config `default_model`.

   The proxy must not guess a model from the cached catalog. If none of these sources resolves a model, stop with a concise error listing cached tool-capable candidates and explain how to select one with `--model`, `NVIDIA_MODEL`, or `default_model`.
3. Resolve the upstream base URL:
   - `--upstream-base-url`;
   - `NVIDIA_BASE_URL` environment variable;
   - user config;
   - default `https://integrate.api.nvidia.com/v1`.

   Normalize the URL by removing trailing slashes and append `chat/completions` or `models` to the configured OpenAI-compatible base URL without duplicating `/v1`.
4. Resolve the upstream authentication mode (`auto`, `required`, or `none`) and, when authentication is required, resolve the NVIDIA credential:
   - `NVIDIA_API_KEY` from the environment, taking precedence so CI and ephemeral use remain possible;
   - otherwise the OS credential store;
   - otherwise, in an interactive terminal, securely prompt with `getpass`, store it, and continue;
   - otherwise fail with a concise instruction to run `claude-nim auth login`.

   In `auto` mode, a normal production upstream requires a credential. In `none` mode, no credential is resolved or sent upstream; this is intended for automated fake-upstream tests. In `required` mode, a missing credential is always an error.
5. Start the HTTP server on `127.0.0.1`, using an ephemeral port by default and recording the selected port before launching Claude Code.
6. Wait for the local health check to succeed.
7. Build a child environment and an ephemeral settings file for Claude Code, then start `claude` with `cwd=PROJECT_DIR`, attached to the parent's terminal.
8. Forward interrupt/termination signals to Claude Code, then shut down the translator, delete the ephemeral settings file, and return Claude Code's exit status. The proxy must not be left orphaned.

The translator should run in the CLI process for the `run` command unless testing demonstrates that a separate server process gives materially better secret isolation or signal handling.

## 6. Claude Code environment mapping

The child process receives a temporary, session-only environment; the tool will not modify `~/.claude/settings.json` or the target project's `.claude/` files. Because shell-exported gateway variables do not reliably reach Claude Code background agents hosted by its supervisor, `run` will also create an ephemeral external JSON settings file containing only the local gateway environment and pass it with `--settings`. That file will contain the random local token—not the NVIDIA credential—and will be deleted when the session ends. Managed settings that override these values will be detected and reported. The exact file shape and launch contract are specified in section 7.1.

| Variable | Value/purpose |
|---|---|
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:<selected-port>` with no trailing path or redirect. |
| `ANTHROPIC_AUTH_TOKEN` | Random per-run local bearer token; the translator validates it. |
| `ANTHROPIC_MODEL` | Selected gateway alias, e.g. `anthropic-nim/nvidia/...`. |
| `ANTHROPIC_CUSTOM_MODEL_OPTION` | Selected gateway alias, ensuring the active model remains selectable if discovery is unavailable. |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Same selected gateway alias, preventing built-in Opus routing. |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Same selected gateway alias, preventing built-in Sonnet routing. |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Same selected gateway alias, preventing built-in Haiku/background routing. |
| `ANTHROPIC_DEFAULT_FABLE_MODEL` | Same selected gateway alias when supported by the installed Claude Code version. |
| `CLAUDE_CODE_SUBAGENT_MODEL` | Same selected gateway alias, preventing subagents from requesting an Anthropic model. |
| `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY` | `1`, so the proxy's NVIDIA aliases appear in `/model`. |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | Model-specific context size when known; conservative fallback otherwise. |
| `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` | Default `1` for MVP compatibility with the OpenAI-shaped upstream; allow an explicit override. |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | Not set by default because it disables Claude Code auto-updates. |

Before launching Claude Code, inherited `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`, and `NVIDIA_API_KEY` values should be removed/replaced in the child environment. In particular, the upstream NVIDIA credential must not be inherited by Claude Code or its Bash subprocesses.

The translator also maps any remaining `claude-*`, `opus`, `sonnet`, `haiku`, or `fable` request to the configured default NVIDIA model as a defensive fallback. This covers background requests that Claude Code may send even when family variables are set.

## 7. Gateway endpoints

| Method/path | Purpose |
|---|---|
| `POST /v1/messages` | Core Anthropic-to-NVIDIA translation, streaming or non-streaming. |
| `GET /v1/models` | Claude Code discovery response containing `anthropic-nim/<NVIDIA ID>` aliases. |
| `HEAD /api/hello` | Return 200/204 as a harmless startup-warmup response. |
| `GET /healthz` | Local process/upstream readiness for CLI startup and `doctor`. |
| `POST /v1/messages/count_tokens` | Optional later endpoint; initially Claude Code's documented estimate fallback is acceptable. |

The server binds only to loopback by default. A fixed host/port override is available for advanced use, but non-loopback binding requires an explicit warning and local bearer authentication remains mandatory.

### 7.1 Wire-level contracts

The following contracts are normative for the MVP. Tests must assert these shapes rather than accepting several variants.

#### Local authentication

- `run` always uses `ANTHROPIC_AUTH_TOKEN`. Claude Code maps that variable to `Authorization: Bearer <token>`; the proxy therefore validates exactly that header form.
- `POST /v1/messages` and `POST /v1/messages/count_tokens` require the bearer token.
- `GET /v1/models` also requires the bearer token because Claude Code performs credentialed gateway discovery. The `limit` query parameter is accepted and ignored.
- `GET /healthz` and `HEAD /api/hello` are unauthenticated loopback probes and never expose upstream credentials or model data.
- A missing or invalid local token produces an Anthropic-style `401` response with error type `authentication_error`.
- Tokens are generated with `secrets.token_urlsafe(32)`, compared in constant time, accepted only in the header (never a query parameter), and never logged.

#### Non-streaming Messages response

A successful non-streaming response has this shape:

```json
{
  "id": "msg_<random>",
  "type": "message",
  "role": "assistant",
  "model": "nvidia/model-id",
  "content": [
    {"type": "text", "text": "Hello"}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 3
  }
}
```

A tool call is represented as:

```json
{"type": "tool_use", "id": "call_abc123", "name": "Read", "input": {"file_path": "/tmp/example"}}
```

The upstream OpenAI tool-call ID is preserved verbatim in `tool_use.id` so the next request's `tool_result.tool_use_id` can be translated back to the exact upstream `tool_call_id`. If the upstream omits an ID, the proxy generates a stable `toolu_<random>` ID for that response. The response `model` is the upstream model identifier when present, otherwise the resolved NVIDIA model ID. Missing usage values are represented as `0`.

#### Anthropic SSE response

Every SSE frame contains both an SSE event name and a matching JSON `type`. The canonical text-only sequence is:

```text
event: message_start
data: {"type":"message_start","message":{"id":"msg_<random>","type":"message","role":"assistant","content":[],"model":"nvidia/model-id","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}
```

A tool-use block starts with the call ID and name, then receives partial JSON fragments:

```text
event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call_abc123","name":"Read","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":\"/tmp/example\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}
```

The stream may contain keepalive events at any point:

```text
event: ping
data: {"type":"ping"}
```

The following invariants apply:

- Content-block indices are contiguous, start at `0`, and correspond to the final message content array.
- Every opened block emits exactly one `content_block_stop`. On a fatal mid-stream error, any open block is closed before the error event.
- `message_delta.usage.output_tokens` is the cumulative final output count. `message_start.usage.input_tokens` carries the input count when known, otherwise `0`.
- `stop_reason` and `stop_sequence` are always present in `message_delta`; `stop_sequence` is `null` unless an upstream stop sequence is known.
- Exactly one terminal `message_delta` is emitted on a successful stream, followed by exactly one `message_stop`.
- Unknown upstream SSE event types are ignored. A fatal upstream error is represented by the Anthropic error event described below rather than an OpenAI-shaped event.

#### Anthropic error response

HTTP errors use the Anthropic error envelope. The proxy generates its own non-secret request ID when one is not available from the upstream:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "The request was invalid."
  },
  "request_id": "req_<random>"
}
```

The MVP status mapping is:

| Condition | HTTP status | Anthropic error type |
|---|---:|---|
| Malformed local request | 400 | `invalid_request_error` |
| Missing/invalid local bearer token | 401 | `authentication_error` |
| Upstream 401 | 401 | `authentication_error` |
| Upstream 403 | 403 | `permission_error` |
| Upstream 400 or 422 | 400 | `invalid_request_error` |
| Upstream 404 | 404 | `not_found_error` |
| Upstream 429 | 429 | `rate_limit_error` |
| Upstream 408 or 504 | 504 | `timeout_error` |
| Upstream 500 or 502 | 500 | `api_error` |
| Upstream 503 | 529 | `overloaded_error` |
| Upstream connection failure | 502 | `api_error` |

Upstream messages are sanitized, truncated, and included only when useful. They must never contain credentials, raw request bodies, or unredacted exception text. `retry-after`, `x-should-retry`, and Anthropic rate-limit headers are preserved where safe.

If an error occurs after a streaming response has started with HTTP 200, emit:

```text
event: error
data: {"type":"error","error":{"type":"api_error","message":"NVIDIA NIM upstream error"}}
```

A fatal stream error is terminal; do not emit a successful `message_delta` or `message_stop` afterward.

#### Claude Code temporary settings and launch

`run` creates a regular JSON file with mode `0600`, referenced by an absolute path. Its initial shape is:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:54321",
    "ANTHROPIC_AUTH_TOKEN": "<random-local-token>",
    "ANTHROPIC_MODEL": "anthropic-nim/nvidia/model-id",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "anthropic-nim/nvidia/model-id",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "anthropic-nim/nvidia/model-id",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "anthropic-nim/nvidia/model-id",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "anthropic-nim/nvidia/model-id",
    "ANTHROPIC_DEFAULT_FABLE_MODEL": "anthropic-nim/nvidia/model-id",
    "CLAUDE_CODE_SUBAGENT_MODEL": "anthropic-nim/nvidia/model-id",
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "131072",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1"
  }
}
```

The launch command is:

```text
claude --settings <absolute-settings-path> [user-supplied Claude Code arguments...]
```

`claude` is resolved from `PATH`. The process starts with `cwd=PROJECT_DIR`, inherited standard streams, and a sanitized copy of the parent environment. The same local-only gateway values are placed in both the child environment and the settings file so normal sessions and supervisor-hosted background sessions receive them. Claude Code gives `--settings` values precedence over `settings.json` files and shell environment values for the same keys.

Before launch, the child environment must remove `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN`, and `NVIDIA_API_KEY`, and replace any inherited `ANTHROPIC_AUTH_TOKEN` with the new local token. The NVIDIA key must never appear in the settings file, child environment, command line, or Claude Code subprocess environment.

On POSIX, start Claude Code in a new process group. Forward `SIGINT` and `SIGTERM` to that group, wait for a bounded grace period, then terminate and kill it if necessary. In every exit path, stop the local server and delete the temporary settings file. Return Claude Code's exit status when it exits normally.

#### Standalone `serve` token contract

`serve` uses the same local authentication rules as `run`, but must give the user a way to supply the token to an independently launched Claude Code:

- `--token-stdin` reads one token line from standard input and is the preferred automation path.
- `--token-file PATH` reads one token line from a file; the CLI rejects a world/group-readable file unless the user explicitly accepts the warning.
- If neither option is supplied, `serve` generates a random token and prints the base URL and token once to standard error as an export-ready snippet.
- A token may not be accepted as a command-line argument because it would be visible in process listings and shell history.

The printed snippet has this form and must not be written to a persistent file by the proxy:

```bash
export ANTHROPIC_BASE_URL='http://127.0.0.1:54321'
export ANTHROPIC_AUTH_TOKEN='<random-local-token>'
```

## 8. Request translation

### Anthropic → OpenAI mapping

| Anthropic input | OpenAI/NIM output | Notes |
|---|---|---|
| `model` | `model` | Reverse the `anthropic-nim/` alias; accept raw NVIDIA IDs; fall back from Claude IDs to the selected model. |
| Top-level `system` string/blocks | System message | Preserve text order; remove `cache_control` markers. |
| Mid-conversation `role: "system"` | System-reminder user message or system message | Prefer a model-capability-aware conversion; never fail merely because the upstream rejects system roles. |
| User/assistant text blocks | Message text content | Preserve ordering around tool blocks. |
| Anthropic image block | `image_url` part | Base64 becomes a data URI; URL sources pass through; reject clearly when the selected model lacks image input. |
| Assistant `tool_use` block | Assistant `tool_calls[]` | JSON-encode `input`; preserve a stable call ID. |
| User `tool_result` block | `role: "tool"` message | One message per result; carry `tool_call_id`; flatten text/image content into the model's supported representation. |
| `tools[].input_schema` | `tools[].function.parameters` | Convert custom Anthropic tools to OpenAI function tools. |
| `tool_choice` | `tool_choice` plus `parallel_tool_calls` | `auto` → `auto`, `any` → `required`, named tool → function object, `none` → `none`. |
| `disable_parallel_tool_use` | `parallel_tool_calls: false` | Applied when the upstream supports it. |
| `max_tokens` | `max_tokens` | Preserve the limit. |
| `stop_sequences` | `stop` | Preserve order. |
| `temperature`, `top_p` | Same fields | Omit absent values. |
| `stream` | `stream` plus `stream_options.include_usage` | Request usage when supported. |
| `thinking`/`output_config.effort` | `reasoning_effort` or dropped | Best-effort and model-specific; the proxy must not pass Anthropic-only objects to NVIDIA. |
| `metadata`, `service_tier`, `cache_control`, Anthropic beta-only fields | Dropped | Unknown/unsupported fields are ignored rather than forwarded blindly. |

Tool names should be preserved when valid. If an upstream enforces a shorter tool-name limit, use a deterministic local name mapping and restore it in responses instead of truncating two different tools to the same name.

Built-in Anthropic server tools and MCP tool-search-only fields are outside the MVP. If encountered, they should be omitted with a debug warning rather than forwarded in an incompatible schema.

### OpenAI/NIM → Anthropic response mapping

| OpenAI response | Anthropic response |
|---|---|
| `choices[0].message.content` | `content: [{"type":"text","text":"..."}]` |
| `choices[0].message.tool_calls[]` | One `tool_use` block per call, with `id`, `name`, and parsed JSON `input`. |
| Legacy `function_call` | One `tool_use` block. |
| `finish_reason: stop` | `stop_reason: end_turn` |
| `finish_reason: length` | `stop_reason: max_tokens` |
| `finish_reason: tool_calls/function_call` | `stop_reason: tool_use` |
| `usage.prompt_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` | `usage.output_tokens` |
| Cached prompt-token details, when present | `cache_read_input_tokens` |
| Upstream response ID | Anthropic-style `msg_...` response ID, while retaining the upstream ID for diagnostics. |

Reasoning-only `reasoning_content` is not emitted as an Anthropic signed thinking block in the MVP. It may be dropped from the client-visible response while remaining available in redacted debug logs. A later opt-in mode can expose it with clearly documented signature limitations.

## 9. Streaming translation

A dedicated stateful stream translator will parse OpenAI SSE incrementally and emit Anthropic SSE without buffering the full response.

Required behavior:

1. Emit `message_start` with an Anthropic message envelope.
2. Open a text `content_block` on the first text delta and emit `text_delta` events.
3. Open one `content_block` per streamed tool call and emit `input_json_delta` fragments.
4. Close every opened content block exactly once.
5. Emit `message_delta` with the mapped stop reason and final output usage.
6. Emit `message_stop` even when the upstream ends without a normal finish chunk, provided no fatal error occurred.
7. Parse fragmented SSE frames, comments, multiple `data:` lines, `[DONE]`, usage-only chunks, and upstream error frames.
8. Assemble streamed tool-call arguments by tool-call index before finalizing them.
9. Send Anthropic `ping` events while waiting for slow upstream bytes so Claude Code's streaming idle watchdog does not treat a quiet generation as a dropped connection.
10. If Claude Code disconnects, cancel the upstream HTTP request rather than continuing to spend tokens.

The implementation will use a small explicit state machine rather than ad-hoc event transformations.

## 10. Error handling

Upstream errors will be converted to Anthropic-style errors, for example:

```json
{
  "type": "error",
  "error": {
    "type": "api_error",
    "message": "NVIDIA NIM upstream error"
  }
}
```

Planned mappings follow the normative table in section 7.1:

- 401 → `authentication_error`; 403 → `permission_error`.
- 400/422 → `invalid_request_error`, preserving a useful but sanitized upstream message.
- 404 → `not_found_error` (used in diagnostics to distinguish model availability from other missing resources).
- 408/504 → `timeout_error`; 500/502 → `api_error`; 503 → `overloaded_error`.
- 429 → `rate_limit_error`, preserving `retry-after` and `x-should-retry` when present.
- Upstream connection failure → HTTP 502 with Anthropic error type `api_error`.
- Client disconnect → cancel upstream without generating another response.

The proxy will not blindly retry billable generation requests. It will preserve retry headers so Claude Code can apply its documented retry behavior.

For known unsupported Anthropic fields, the proxy will proactively remove or translate the field. This is more reliable than relying on Claude Code to recognize an upstream error string that has already changed protocols.

## 11. Model refresh

### `claude-nim models update`

The command will:

1. Query the configured NVIDIA base URL's `GET /v1/models`. This endpoint is public, so catalog refresh does not require a stored credential.
2. Parse the OpenAI-style `data[].id` list and retain the raw NVIDIA IDs as the authoritative catalog.
3. Merge those IDs with a small bundled, curated capability catalog containing context size, maximum output, reasoning support, image support, and known tool-calling support. New or unknown NVIDIA models remain selectable with conservative defaults and an "unknown capabilities" marker.
4. Write the proxy's cache atomically, without storing credentials.
5. Report counts, additions/removals relative to the previous cache, and any IDs whose curated metadata is missing.

The NVIDIA endpoint exposes IDs but not capability metadata or per-account entitlements. Capability metadata will therefore be a bundled/generated data file maintained from NVIDIA's supported-model documentation and model cards, with user-config overrides for exceptions. The model catalog depends only on NVIDIA and local curated metadata.

The model cache lives outside the project at:

```text
~/.config/claude-nim/models.json
```

Cache schema:

```json
{
  "version": 1,
  "updated_at": "2026-10-24T16:18:00Z",
  "upstream_base_url": "https://integrate.api.nvidia.com/v1",
  "models": [
    {
      "upstream_id": "nvidia/nemotron-3-super-120b-a12b",
      "gateway_id": "anthropic-nim/nvidia/nemotron-3-super-120b-a12b",
      "display_name": "NVIDIA Nemotron 3 Super 120B A12B (NIM)",
      "context_window": 1048576,
      "max_output_tokens": 32768,
      "reasoning": true,
      "input": ["text"],
      "tool_calling": true,
      "source": ["nvidia"],
      "metadata_source": "bundled"
    }
  ]
}
```

The bundled file `src/claude_nim_proxy/data/nvidia_models.json` uses the same model-object shape and a top-level `version`/`models` structure, but does not contain `updated_at`, `upstream_base_url`, or user overrides. `gateway_id` is derived from `upstream_id` rather than duplicated in the bundled source data. The cache may store the derived value for convenient inspection.

The curated overlay contains 27 models: all 22 NVIDIA models exposed by `pi --list-models nvidia`, plus the five existing curated NVIDIA models that Pi does not expose (Mistral Large 2, Mixtral 8x22B, Nemotron 3 Super, Nemotron 3.5 Bolt, and Nemotron Nano 3). Capability values for the original curated entries are taken from NVIDIA's model cards and NIM API reference. Context, output, reasoning, and image capability values for the nine newly added Pi-visible models come from the same Pi model registry used by `pi --list-models nvidia`. Pi does not publish tool-calling capability, so `tool_calling` remains unset for those additions and is resolved as unknown rather than guessed.

`reasoning` and `tool_calling` are tri-state: `true`, `false`, or `null` for unknown. The `input` list contains supported modalities (`"text"` and optionally `"image"`). `metadata_source` is one of `bundled`, `user`, or `unknown`.

A model absent from the bundled capability catalog remains selectable and receives conservative runtime defaults: `context_window = 32768`, `max_output_tokens = 4096`, `reasoning = null`, `tool_calling = null`, and `input = ["text"]`. Unknown tool-calling support is allowed on a best-effort basis because rejecting it would make an otherwise functional NIM model unusable with Claude Code; unknown image support is rejected locally with a clear capability error. A user override can replace any of these values.

The credential-free cache retains the complete authoritative NVIDIA catalog. Claude Code's `/model` discovery endpoint is intentionally narrower: it advertises bundled curated models that are present in the current catalog, plus the explicitly selected model even when that model lacks curated metadata. Uncatalogued or non-curated models remain directly selectable through `--model`, `NVIDIA_MODEL`, or `default_model`.

A model's presence in `/v1/models` does not guarantee that every NVIDIA key is entitled to call it. `doctor` and `auth verify` must distinguish catalog discovery from actual account entitlement and report a 404 from NVIDIA as an entitlement/model-availability problem.

## 12. Credential management

The CLI owns NVIDIA credential acquisition and persistence.

### Resolution order

When the selected upstream authentication mode requires a credential, resolve it in this order:

1. `NVIDIA_API_KEY` from the process environment. This takes precedence and is never written to persistent storage automatically.
2. The OS credential store, using service name `claude-nim` and account name `nvidia`.
3. An interactive `getpass` prompt when an interactive terminal is available. The entered key is stored before the command continues.
4. In a non-interactive environment with no available credential, exit with an instruction to run `claude-nim auth login` or set `NVIDIA_API_KEY`.

The `auto` authentication mode requires a credential for normal production upstreams. The `none` mode does not resolve or transmit a credential and is intended for automated fake-upstream tests. The `required` mode always resolves one.

### Commands

- `claude-nim auth login`: securely prompt for the key, confirm the storage backend, and replace an existing stored credential when `--replace` is supplied.
- `claude-nim auth status`: print only readiness and source (`environment`, `credential store`, `file fallback`, or `not configured`); never print any part of the key.
- `claude-nim auth verify`: make an explicit minimal authenticated NVIDIA request and report success/failure. This may consume a small amount of API credit and will state that before running. It always resolves and sends a credential, independent of the `upstream_auth` mode used by `run` and `serve`.
- `claude-nim auth logout`: remove the credential-store entry. It will not alter an environment-provided `NVIDIA_API_KEY` because that value is outside the CLI's ownership.

There will be no recommended `--api-key` command-line option because command-line secrets leak through shell history and process listings. Automation should use `NVIDIA_API_KEY`.

### Storage

The preferred backend is the Python `keyring` package, which uses Secret Service/KWallet on Linux, Keychain on macOS, and Credential Manager on Windows. If no OS backend is available, the CLI falls back to `~/.config/claude-nim/credentials.json` with mode `0600`, clearly warns that the unencrypted secret is stored on disk, and offers to remove it. The API key will never be written to `config.toml`, the model cache, project files, logs, or Claude Code's child environment.

## 13. Configuration

### File format and location

The user config path is:

```text
~/.config/claude-nim/config.toml
```

A missing config file is valid and means that all values use their defaults. The CLI parses the file with Python's standard-library `tomllib` and validates it against the schema below. Unknown top-level keys, unknown per-model keys, invalid value types, and invalid enum values are configuration errors with the offending key named; silently ignoring them would hide configuration mistakes.

A complete example is:

```toml
default_model = "nvidia/nemotron-3-super-120b-a12b"
verify_model = "nvidia/nemotron-3-super-120b-a12b"
upstream_base_url = "https://integrate.api.nvidia.com/v1"
upstream_auth = "auto"
listen_host = "127.0.0.1"
listen_port = 0
context_window = 131072
max_output_tokens = 32768
disable_experimental_betas = true
upstream_connect_timeout_seconds = 10
upstream_read_timeout_seconds = 600
debug = false

[model_overrides."nvidia/nemotron-3-super-120b-a12b"]
display_name = "NVIDIA Nemotron 3 Super 120B A12B (NIM)"
context_window = 131072
max_output_tokens = 32768
reasoning = true
input = ["text", "image"]
tool_calling = true
```

The supported top-level keys are:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `default_model` | string | unset | Raw NVIDIA ID or reversible `anthropic-nim/...` gateway ID used when no CLI or environment model is supplied. |
| `verify_model` | string | `default_model` | Model used by `auth verify` for its explicit minimal authenticated request. |
| `upstream_base_url` | string | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible NVIDIA base URL, normally ending in `/v1`. |
| `upstream_auth` | string enum | `auto` | Upstream credential policy: `auto`, `required`, or `none` as defined below. |
| `listen_host` | string | `127.0.0.1` | Local bind address. A non-loopback value is rejected unless explicitly allowed. |
| `listen_port` | integer | `0` | `0` selects an ephemeral port. |
| `context_window` | integer | `32768` | Global fallback context size when a selected model has no curated or per-model override. |
| `max_output_tokens` | integer | `4096` | Global fallback output limit when a selected model has no curated or per-model override. |
| `disable_experimental_betas` | boolean | `true` | Default value for `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` in the Claude Code child. |
| `upstream_connect_timeout_seconds` | number | `10` | Upstream connection timeout. |
| `upstream_read_timeout_seconds` | number | `600` | Upstream response/read timeout, including streamed generation. |
| `debug` | boolean | `false` | Enable redacted diagnostic logging. |
| `model_overrides` | table | empty | Per-NVIDIA-model capability overrides keyed by the raw upstream model ID. |

A per-model override accepts exactly these keys:

| Key | Type | Meaning |
|---|---|---|
| `display_name` | string | Human-readable name shown by `models list` and Claude Code discovery. |
| `context_window` | positive integer | Context window in tokens. |
| `max_output_tokens` | positive integer | Maximum output in tokens. |
| `reasoning` | boolean or `null` | Whether the model supports reasoning; `null` means unknown. |
| `input` | array of strings | Supported input modalities; only `"text"` and `"image"` are valid. |
| `tool_calling` | boolean or `null` | Whether the model reliably supports tool calls; `null` means unknown/best-effort. |

Capability precedence is: user per-model override, bundled per-model metadata, global config fallback, then conservative unknown-model defaults. `models update` may refresh IDs and bundled metadata but must never overwrite user configuration.

`upstream_auth` has these exact meanings:

- `auto`: require and send a credential for a normal production upstream.
- `required`: always resolve and send a credential; fail if none is available.
- `none`: never resolve or send an NVIDIA credential.

### Environment and CLI precedence

Configuration precedence is strict:

1. Explicit CLI flag.
2. Environment variable.
3. `config.toml` value.
4. Built-in default.

The supported environment variables are:

| Variable | Equivalent setting |
|---|---|
| `NVIDIA_MODEL` | `default_model` |
| `NVIDIA_BASE_URL` | `upstream_base_url` |
| `CLAUDE_NIM_UPSTREAM_AUTH` | `upstream_auth` |
| `CLAUDE_NIM_VERIFY_MODEL` | `verify_model` |
| `CLAUDE_NIM_LISTEN_HOST` | `listen_host` |
| `CLAUDE_NIM_LISTEN_PORT` | `listen_port` |
| `CLAUDE_NIM_CONTEXT_WINDOW` | `context_window` |
| `CLAUDE_NIM_MAX_OUTPUT_TOKENS` | `max_output_tokens` |
| `CLAUDE_NIM_DISABLE_EXPERIMENTAL_BETAS` | `disable_experimental_betas` |
| `CLAUDE_NIM_UPSTREAM_CONNECT_TIMEOUT_SECONDS` | `upstream_connect_timeout_seconds` |
| `CLAUDE_NIM_UPSTREAM_READ_TIMEOUT_SECONDS` | `upstream_read_timeout_seconds` |
| `CLAUDE_NIM_DEBUG` | `debug` |

Boolean environment variables accept `1`/`true`/`yes` and `0`/`false`/`no`, case-insensitively. Integer and numeric values must parse exactly and must be positive where the setting requires a positive value.

The common CLI flags are `--model`, `--upstream-base-url`, `--upstream-auth`, `--listen-host`, `--listen-port`, `--context-window`, `--max-output-tokens`, `--debug`, `--allow-non-loopback`, and the mutually exclusive `--disable-experimental-betas`/`--enable-experimental-betas`. Subcommands may add command-specific flags, but they must follow the same precedence rule.

The NVIDIA API key is never stored in this configuration file; the `auth` commands manage it separately in the OS credential store or the explicitly warned file fallback. `NVIDIA_API_KEY` is a credential input rather than a normal configuration override.

If `--model`, `NVIDIA_MODEL`, and `default_model` are all absent, the CLI must fail with the actionable model-selection message defined in section 5 rather than choosing a model from the cache.

## 14. Security and privacy

- Bind to `127.0.0.1` by default.
- Generate a random local bearer token per run and validate `Authorization: Bearer <token>` on credentialed Claude-facing endpoints.
- Resolve the NVIDIA credential from the environment or OS credential store into proxy memory only. The persistent copy may exist only in the OS credential store, or in the mode-`0600` fallback file after an explicit warning; never place it in command-line arguments, logs, the model cache, project files, or Claude Code's child environment.
- Redact `Authorization`, `x-api-key`, and configured secrets from all debug output.
- Do not log prompts, tool schemas, message bodies, or responses by default.
- Never include the NVIDIA key in redirected upstream URLs or exception text.
- Propagate cancellation promptly when Claude Code exits or disconnects.
- Clearly document that project content, prompts, tool definitions, and tool results are sent to the configured NVIDIA upstream.

## 15. Package layout

```text
AGENTS.md
plan.md
pyproject.toml
README.md
src/claude_nim_proxy/
  __init__.py
  cli.py
  config.py
  credentials.py
  models.py
  runner.py
  server.py
  upstream.py
  data/
    nvidia_models.json
  translate/
    __init__.py
    anthropic_to_openai.py
    openai_to_anthropic.py
    stream.py
    errors.py
tests/
  fixtures/
  test_anthropic_to_openai.py
  test_openai_to_anthropic.py
  test_stream_translation.py
  test_gateway.py
  test_cli.py
  test_credentials.py
  test_model_sync.py
```

### Runtime dependencies

Keep this intentionally small:

- `aiohttp` for the local `aiohttp.web` HTTP server and asynchronous upstream client, including streaming.
- `keyring` for OS-native credential storage, with a small mode-`0600` file fallback when no OS backend is available.
- Python 3.11+ standard library `argparse`, `asyncio`, `getpass`, `json`, `tomllib`, `pathlib`, `subprocess`, and `signal`.

No LiteLLM, OpenAI SDK, Anthropic SDK, database, or model-routing framework is required.

Development dependencies will be limited to `pytest`, `pytest-asyncio`, `ruff`, and optionally `mypy`.

## 16. Testing plan

### Pure translation tests

- Text-only request and response.
- String and block-form system prompts.
- Mid-conversation system messages.
- Base64 and URL image conversion.
- Tool definitions and all tool-choice modes.
- Assistant tool use, user tool results, parallel calls, and round-trip IDs.
- Unsupported field dropping (`cache_control`, `strict`, `defer_loading`, Anthropic-only thinking/context fields).
- Stop-reason and usage mapping.
- Exact non-streaming Anthropic response envelope and tool-use block shape.
- Model alias, raw NVIDIA ID, and Claude fallback mapping.

### SSE state-machine tests

- Exact Anthropic event ordering.
- Text and tool-call streams.
- Fragmented SSE frames and multi-line data.
- `[DONE]`, comments, usage-only chunks, and missing finish chunks.
- Tool arguments split across chunks.
- Upstream error frames and disconnects.
- No buffered first token.
- Ping emission during a slow upstream.
- Exactly-once content-block close and final `message_stop`.
- Exact event names, JSON fields, contiguous content-block indices, and cumulative usage in the section 7.1 frames.
- Fatal error-frame conversion, including closing an open block before the Anthropic error event.

### Gateway integration tests

Use a small fake NVIDIA server to verify:

- Bearer key is attached upstream and never returned downstream.
- Correct upstream path and model ID.
- Non-streaming JSON shape.
- Streaming content type and incremental delivery.
- `GET /v1/models` alias format and Claude-compatible filtering.
- `HEAD /api/hello` and `/healthz`.
- 401, 403, 404, 422, 429, 500, 502, 503, 504, timeout, and malformed-upstream handling.
- Exact Anthropic error status/body mapping from section 7.1, including a generated non-secret `request_id`.
- Local bearer authentication on `/v1/messages` and `/v1/models`, with unauthenticated loopback probes.
- Client cancellation cancels the upstream request.

### CLI tests

Use a fake `claude` executable and a mocked credential-store backend to verify:

- Correct project directory, exact `claude --settings <path> ...` launch, and Claude arguments.
- Exact mode-`0600` temporary settings-file shape and deletion on every exit path.
- Correct child environment and Claude Code settings-file/environment precedence.
- NVIDIA key absent from the Claude child environment and temporary settings file.
- Claude exit code propagation.
- Ctrl-C/termination cleanup and no orphaned listener.
- Fixed and ephemeral port behavior.
- Environment, credential-store, interactive-prompt, and missing-credential resolution paths.
- `auto`, `required`, and `none` upstream authentication modes, including credential-free fake-upstream tests.
- `auth login`, `auth status`, `auth verify`, and `auth logout` behavior without revealing secrets.
- Model cache atomic update and no secret persistence.
- `serve` token sources (`--token-stdin`, `--token-file`, and generated-token output) without accepting a token in command-line arguments.
- No implicit model selection when `--model`, `NVIDIA_MODEL`, and `default_model` are absent.

### Configuration tests

- Valid TOML parsing and rejection of unknown keys, invalid types, invalid enum values, and invalid per-model override fields.
- Strict CLI → environment → config → default precedence for every supported setting.
- Capability precedence across per-model overrides, bundled metadata, global fallback, and conservative unknown-model defaults.
- Atomic writes for the model cache and user config when the CLI creates or updates them.

### Real smoke tests

Against a supported NVIDIA model:

1. Start with `claude-nim run <temp-project> --model <model>`.
2. Confirm `/status` reports the local `ANTHROPIC_BASE_URL`.
3. Run a one-shot text request and verify incremental output.
4. Run a request that requires Claude Code's `Read` tool and verify a complete `tool_use` → tool result → final answer round trip.
5. Verify `/model` shows NVIDIA aliases.
6. Exercise a subagent/background request and verify it does not fall back to an Anthropic model.
7. Confirm termination leaves no proxy process listening.

## 17. Implementation phases

Agents follow `AGENTS.md` and work in tracker order. A phase should normally fit one focused session; when it does not, stop at a tested boundary, update the implementation tracker, commit the completed portion, and resume from the tracker in the next session.

### Phase 1 — Scaffold completion and pure translators

- The uv package scaffold already exists in commit `da83fc5`; verify and preserve it rather than recreating it.
- Implement the CLI entry-point skeleton and typed configuration model without performing network I/O.
- Implement pure Anthropic ↔ OpenAI request/response functions.
- Implement the Anthropic SSE state machine.
- Add focused unit tests before adding I/O.
- Update the tracker with validation results and the Phase 1 commit.

### Phase 2 — Local gateway

- Implement `aiohttp` server, local authentication, upstream client, health/model endpoints, error conversion, cancellation, and timeout behavior.
- Add fake-upstream integration tests.

### Phase 3 — Model discovery and NVIDIA catalog refresh

- Implement NVIDIA model parsing, reversible gateway aliases, bundled capability metadata, cache storage, `models list`, and `models update`.
- Verify Claude Code discovery behavior with real `anthropic-nim/...` IDs.

### Phase 4 — Credential management and one-command runner

- Implement the `auth` commands, OS credential-store integration, secure interactive prompting, and the warned file fallback.
- Implement environment construction, model/credential resolution, Claude Code process launch, signal handling, and cleanup.
- Add CLI subprocess and credential-store tests.

### Phase 5 — Compatibility hardening and documentation

- Test current Claude Code releases against text, tools, images, subagents, background calls, and long streams.
- Add redacted debug tracing.
- Write README usage, hosted NVIDIA setup, model-selection guidance, security notes, and troubleshooting.

## 18. Acceptance criteria

The MVP is complete when:

- `claude-nim run <project> --model <NVIDIA ID>` starts the proxy and Claude Code in the project with one command.
- Claude Code sees the local gateway URL and does not request an Anthropic-hosted model.
- Text responses stream incrementally with a valid Anthropic event sequence.
- Claude Code can call a local tool and complete the tool-result round trip.
- Built-in model aliases, subagents, and background requests resolve to the selected NVIDIA model.
- Curated NVIDIA models present in the current catalog are selectable through `/model` using `anthropic-nim/...` discovery IDs; the explicitly selected model is always included.
- `claude-nim models update` refreshes model IDs directly from NVIDIA, works without a stored credential, and never persists one in the model cache.
- `claude-nim auth login` securely prompts for and stores the NVIDIA key; `auth status` never reveals it; `auth logout` removes the stored copy.
- The NVIDIA key is not present in the Claude Code child environment, logs, cache, or process arguments.
- The proxy shuts down cleanly when Claude Code exits or receives an interrupt.
- The test suite passes without a live NVIDIA account; live NVIDIA and Claude Code smoke tests are documented separately.
- The local gateway emits the exact Anthropic response, SSE, and error envelopes specified in section 7.1.
- The Claude Code child is launched with the exact temporary settings shape and command-line contract specified in section 7.1, with no NVIDIA credential inherited.
- Configuration precedence and capability overrides follow section 13 exactly, and an unspecified model is never selected implicitly.
- The implementation tracker records the final validation evidence and completed-phase commit.

## 19. Confirmed implementation decisions

All decisions below are confirmed by the user and are binding for implementation:

1. **CLI/package name:** use `claude-nim` / `claude-nim-proxy`.
2. **Upstream target:** hosted NVIDIA NIM only (`https://integrate.api.nvidia.com/v1`). Local/self-hosted NIM deployment and alternate providers are out of scope. The configurable base URL remains available for automated fake-upstream tests.
3. **Platform target:** Linux/macOS first; Windows path and signal behavior can follow after the core is stable.
4. **Images:** include basic image conversion in the MVP, but reject unsupported models with a clear message.
5. **Experimental Anthropic betas:** disable them by default in the Claude Code child, with an explicit override for testing.
6. **Native NIM Anthropic endpoint:** not applicable and out of scope. The project will not implement native Anthropic passthrough or a local-NIM bypass.
7. **Credential storage:** prefer the OS credential store through `keyring`; use a mode-`0600` file only as an explicitly warned, unencrypted fallback.
8. **Model picker scope:** Claude Code discovery advertises only bundled curated models present in the current NVIDIA catalog, plus the explicitly selected model. The complete catalog remains in the local cache, and non-curated models remain selectable through explicit model configuration.
9. **Pi-visible curated models:** The bundled curated catalog must include every NVIDIA model exposed by `pi --list-models nvidia`. Existing curated models not exposed by Pi remain included. Pi-provided capability metadata may fill context, output, reasoning, and image fields, while unprovided tool-calling support remains unknown.

## 20. Sources

### Official Claude Code / Anthropic documentation

- Connect Claude Code to an LLM gateway: <https://code.claude.com/docs/en/llm-gateway-connect>
- Claude Code gateway compatibility guide: <https://code.claude.com/docs/en/llm-gateway-protocol>
- Roll out an LLM gateway: <https://code.claude.com/docs/en/llm-gateway-rollout>
- Claude Code model configuration: <https://code.claude.com/docs/en/model-config>
- Claude Code CLI reference (including `--settings`): <https://code.claude.com/docs/en/cli-reference>
- Anthropic Messages API: <https://platform.claude.com/docs/en/api/messages>
- Anthropic streaming event contract: <https://platform.claude.com/docs/en/build-with-claude/streaming>
- Anthropic HTTP error contract: <https://platform.claude.com/docs/en/api/errors>
- Anthropic tool use: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>

### Official NVIDIA documentation

- NVIDIA NIM LLM/VLM API reference: <https://docs.nvidia.com/nim/large-language-models/latest/reference/api-reference.html>
- NVIDIA NIM tool calling and MCP integration: <https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/tool-calling-and-mcp.html>
- NVIDIA NIM quickstart/model discovery: <https://docs.nvidia.com/nim/large-language-models/latest/get-started/quickstart.html>
- NVIDIA supported models: <https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html>

### Developer projects reviewed for protocol-conversion patterns

These are reference implementations, not proposed runtime dependencies:

- `oai2ant` (pure Anthropic/OpenAI conversion and SSE state handling): <https://github.com/PeronGH/oai2ant>
- `openai-anthropic-converter` (Python request/response/stream conversion patterns): <https://github.com/wenmengzhou/openai-anthropic-converter>
- `chat-to-claude-code` (one-command Claude Code proxy and Anthropic SSE mapping): <https://github.com/chinfeng/chat-to-claude-code>
- `free-llm-proxy` (pure Python Anthropic translation module and tests): <https://github.com/promsoft/free-llm-proxy>
