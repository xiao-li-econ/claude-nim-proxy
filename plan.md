# Implementation Plan: Thin Claude Code ↔ NVIDIA NIM Proxy

**Status:** Planning only — no implementation has been started.  
**Research snapshot:** 2026-10-24  
**Verified local versions:** Python 3.12.3, Claude Code 2.1.310  
**Workspace:** Greenfield directory; it currently contains only this plan and is not a Git repository.

## 1. Goal

Build a small Python CLI that:

1. Runs a loopback-only translator exposing the Anthropic Messages API expected by Claude Code.
2. Translates Anthropic Messages requests, responses, tool calls, and SSE streams to/from NVIDIA NIM's OpenAI-compatible Chat Completions API.
3. Starts the translator and Claude Code together in a chosen project directory with one command.
4. Provides NVIDIA model refresh commands and secure CLI-managed credential storage.
5. Avoids LiteLLM and general-purpose provider/router abstractions.

Proposed distribution name: package `claude-nim-proxy`, executable `claude-nim`.

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

Claude Code currently keeps only discovered IDs containing `claude` or `anthropic`, case-insensitively. Raw NVIDIA IDs such as `nvidia/nemotron-3-super-120b-a12b` are ignored by the picker. The proxy therefore needs a reversible gateway alias, proposed as:

```text
anthropic-nim/<actual NVIDIA model ID>
```

The translator strips `anthropic-nim/` before calling NVIDIA. Explicit raw NVIDIA IDs will also be accepted, even though they will not appear through discovery.

### NVIDIA NIM contract

- Hosted NIM base URL: `https://integrate.api.nvidia.com/v1`.
- Hosted authentication: `Authorization: Bearer $NVIDIA_API_KEY`.
- Hosted inference: `POST /v1/chat/completions`.
- Model catalog: `GET /v1/models`, returning OpenAI-style `{ "object": "list", "data": [...] }`.
- A self-hosted NIM commonly uses `http://127.0.0.1:8000/v1` and the same Chat Completions contract.
- Tool calling is model- and backend-dependent. Local vLLM/SGLang NIM deployments may need the correct tool-call parser enabled. Claude Code is not useful without reliable tool calling.
- The hosted NVIDIA endpoint was checked during planning and exposes Chat Completions but returns 404 for `/v1/messages`, so a translator is needed for that target.
- Recent local NIM releases also expose a native Anthropic-compatible `/v1/messages` endpoint, and NVIDIA documents using Claude Code directly with them. That is a useful bypass/future mode, but the primary target of this project remains hosted NIM and NIM deployments that expose the OpenAI-compatible API.

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
- Local NIM as well as hosted NIM via a configurable base URL.
- Clear diagnostics for missing Claude Code, missing credentials, unsupported models, and failed upstream calls.

### Explicit non-goals for the MVP

- Multi-provider routing, quotas, budgets, billing, or a dashboard.
- Starting or managing NIM containers/GPU deployments.
- Anthropic prompt-cache semantics or cache token accounting.
- Preserving Anthropic signed thinking blocks; OpenAI-compatible NIM responses cannot create valid Anthropic signatures.
- Anthropic server-side tools such as web search, web fetch, code execution, advisor, or MCP tool search.
- Reimplementing Claude Code's system prompt or otherwise making a non-Claude model behave exactly like Claude.
- Translating every future Anthropic beta feature. Unknown fields will be ignored safely and recorded in debug mode.

## 4. Proposed CLI

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
   - user config `default_model`;
   - a configured/cached tool-capable model, if unambiguous.
3. Resolve the NVIDIA credential:
   - `NVIDIA_API_KEY` from the environment, taking precedence so CI and ephemeral use remain possible;
   - otherwise the OS credential store;
   - otherwise, in an interactive terminal, securely prompt with `getpass`, store it, and continue;
   - otherwise fail with a concise instruction to run `claude-nim auth login`.
4. Resolve the upstream base URL:
   - `--upstream-base-url`;
   - `NVIDIA_BASE_URL` environment variable;
   - user config;
   - default `https://integrate.api.nvidia.com/v1`.
5. Start the HTTP server on `127.0.0.1`, using an ephemeral port by default and recording the selected port before launching Claude Code.
6. Wait for the local health check to succeed.
7. Build a child environment and an ephemeral settings file for Claude Code, then start `claude` with `cwd=PROJECT_DIR`, attached to the parent's terminal.
8. Forward interrupt/termination signals to Claude Code, then shut down the translator, delete the ephemeral settings file, and return Claude Code's exit status. The proxy must not be left orphaned.

The translator should run in the CLI process for the `run` command unless testing demonstrates that a separate server process gives materially better secret isolation or signal handling.

## 6. Claude Code environment mapping

The child process receives a temporary, session-only environment; the tool will not modify `~/.claude/settings.json` or the target project's `.claude/` files. Because shell-exported gateway variables do not reliably reach Claude Code background agents hosted by its supervisor, `run` will also create an ephemeral external settings file containing only the local gateway environment and pass it with `--settings`. That file will contain the random local token—not the NVIDIA credential—and will be deleted when the session ends. Managed settings that override these values will be detected and reported.

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

Planned mappings:

- 401/403 → `authentication_error`.
- 400/404/422 → `invalid_request_error`, preserving a useful but sanitized upstream message.
- 408/502/503/504 → `api_error` or `overloaded_error` as appropriate.
- 429 → `rate_limit_error`, preserving `retry-after` and `x-should-retry` when present.
- Upstream connection failure → 502 Anthropic error.
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

The model cache will live outside the project, proposed at:

```text
~/.config/claude-nim/models.json
```

Suggested schema:

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
      "context_window": 131072,
      "max_output_tokens": 32768,
      "reasoning": true,
      "input": ["text", "image"],
      "tool_calling": true,
      "source": ["nvidia"],
      "metadata_source": "bundled"
    }
  ]
}
```

A model's presence in `/v1/models` does not guarantee that every NVIDIA key is entitled to call it. `doctor` and `auth verify` must distinguish catalog discovery from actual account entitlement and report a 404 from NVIDIA as an entitlement/model-availability problem.

## 12. Credential management

The CLI owns NVIDIA credential acquisition and persistence.

### Resolution order

1. `NVIDIA_API_KEY` from the process environment. This takes precedence and is never written to persistent storage automatically.
2. The OS credential store, using service name `claude-nim` and account name `nvidia`.
3. An interactive `getpass` prompt when an interactive terminal is available. The entered key is stored before the command continues.
4. In a non-interactive environment with no available credential, exit with an instruction to run `claude-nim auth login` or set `NVIDIA_API_KEY`.

### Commands

- `claude-nim auth login`: securely prompt for the key, confirm the storage backend, and replace an existing stored credential when `--replace` is supplied.
- `claude-nim auth status`: print only readiness and source (`environment`, `credential store`, `file fallback`, or `not configured`); never print any part of the key.
- `claude-nim auth verify`: make an explicit minimal authenticated NVIDIA request and report success/failure. This may consume a small amount of API credit and will state that before running.
- `claude-nim auth logout`: remove the credential-store entry. It will not alter an environment-provided `NVIDIA_API_KEY` because that value is outside the CLI's ownership.

There will be no recommended `--api-key` command-line option because command-line secrets leak through shell history and process listings. Automation should use `NVIDIA_API_KEY`.

### Storage

The preferred backend is the Python `keyring` package, which uses Secret Service/KWallet on Linux, Keychain on macOS, and Credential Manager on Windows. If no OS backend is available, the CLI may fall back to `~/.config/claude-nim/credentials.json` with mode `0600`, clearly warn that the secret is stored on disk, and offer to remove it. The API key will never be written to `config.toml`, the model cache, project files, logs, or Claude Code's child environment.

## 13. Configuration

Proposed user config path:

```text
~/.config/claude-nim/config.toml
```

Example:

```toml
default_model = "nvidia/nemotron-3-super-120b-a12b"
upstream_base_url = "https://integrate.api.nvidia.com/v1"
listen_host = "127.0.0.1"
listen_port = 0
context_window = 131072
disable_experimental_betas = true
```

Environment variables override config values, and CLI flags override both. The API key is never stored in this configuration file; the `auth` commands manage it separately in the OS credential store or the explicitly warned file fallback.

## 14. Security and privacy

- Bind to `127.0.0.1` by default.
- Generate a random local bearer token per run and validate `Authorization`/`x-api-key` on Claude-facing endpoints.
- Resolve the NVIDIA credential from the environment or OS credential store into proxy memory only. The persistent copy may exist only in the OS credential store, or in the mode-`0600` fallback file after an explicit warning; never place it in command-line arguments, logs, the model cache, project files, or Claude Code's child environment.
- Redact `Authorization`, `x-api-key`, and configured secrets from all debug output.
- Do not log prompts, tool schemas, message bodies, or responses by default.
- Never include the NVIDIA key in redirected upstream URLs or exception text.
- Propagate cancellation promptly when Claude Code exits or disconnects.
- Clearly document that project content, prompts, tool definitions, and tool results are sent to the configured NVIDIA upstream.

## 15. Proposed package layout

```text
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

### Gateway integration tests

Use a small fake NVIDIA server to verify:

- Bearer key is attached upstream and never returned downstream.
- Correct upstream path and model ID.
- Non-streaming JSON shape.
- Streaming content type and incremental delivery.
- `GET /v1/models` alias format and Claude-compatible filtering.
- `HEAD /api/hello` and `/healthz`.
- 401, 404, 429, 500, timeout, and malformed-upstream handling.
- Client cancellation cancels the upstream request.

### CLI tests

Use a fake `claude` executable and a mocked credential-store backend to verify:

- Correct project directory and Claude arguments.
- Correct child environment.
- NVIDIA key absent from the Claude child environment.
- Claude exit code propagation.
- Ctrl-C/termination cleanup and no orphaned listener.
- Fixed and ephemeral port behavior.
- Environment, credential-store, interactive-prompt, and missing-credential resolution paths.
- `auth login`, `auth status`, `auth verify`, and `auth logout` behavior without revealing secrets.
- Model cache atomic update and no secret persistence.

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

### Phase 1 — Project scaffold and pure translators

- Create package metadata and CLI skeleton.
- Implement pure Anthropic ↔ OpenAI request/response functions.
- Implement the Anthropic SSE state machine.
- Add focused unit tests before adding I/O.

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
- Write README usage, NVIDIA setup, local NIM setup, model-selection guidance, security notes, and troubleshooting.
- Add an optional native Anthropic-upstream passthrough mode for recent local NIM releases if it remains small and clearly separated from the OpenAI translator.

## 18. Acceptance criteria

The MVP is complete when:

- `claude-nim run <project> --model <NVIDIA ID>` starts the proxy and Claude Code in the project with one command.
- Claude Code sees the local gateway URL and does not request an Anthropic-hosted model.
- Text responses stream incrementally with a valid Anthropic event sequence.
- Claude Code can call a local tool and complete the tool-result round trip.
- Built-in model aliases, subagents, and background requests resolve to the selected NVIDIA model.
- Multiple NVIDIA models are selectable through `/model` using `anthropic-nim/...` discovery IDs.
- `claude-nim models update` refreshes model IDs directly from NVIDIA, works without a stored credential, and never persists one in the model cache.
- `claude-nim auth login` securely prompts for and stores the NVIDIA key; `auth status` never reveals it; `auth logout` removes the stored copy.
- The NVIDIA key is not present in the Claude Code child environment, logs, cache, or process arguments.
- The proxy shuts down cleanly when Claude Code exits or receives an interrupt.
- The test suite passes without a live NVIDIA account; live NVIDIA and Claude Code smoke tests are documented separately.

## 19. Decisions to confirm before implementation

The following defaults are recommended, but should be confirmed at implementation start:

1. **CLI/package name:** use `claude-nim` / `claude-nim-proxy`.
2. **Default upstream:** hosted NVIDIA NIM (`https://integrate.api.nvidia.com/v1`), with local NIM selected by flag/config.
3. **Platform target:** Linux/macOS first; Windows path and signal behavior can follow after the core is stable.
4. **Images:** include basic image conversion in the MVP, but reject unsupported models with a clear message.
5. **Experimental Anthropic betas:** disabled by default in the Claude Code child, with an override for testing.
6. **Native NIM Anthropic endpoint:** document the direct bypass immediately; implement optional passthrough only after the core translator is stable.
7. **Credential storage:** prefer the OS credential store through `keyring`; use a mode-`0600` file only as an explicitly warned fallback.

## 20. Sources

### Official Claude Code / Anthropic documentation

- Connect Claude Code to an LLM gateway: <https://code.claude.com/docs/en/llm-gateway-connect>
- Claude Code gateway compatibility guide: <https://code.claude.com/docs/en/llm-gateway-protocol>
- Roll out an LLM gateway: <https://code.claude.com/docs/en/llm-gateway-rollout>
- Claude Code model configuration: <https://code.claude.com/docs/en/model-config>
- Anthropic Messages API: <https://platform.claude.com/docs/en/api/messages>
- Anthropic tool use: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>

### Official NVIDIA documentation

- NVIDIA NIM LLM/VLM API reference: <https://docs.nvidia.com/nim/large-language-models/latest/reference/api-reference.html>
- Use Claude Code with NIM: <https://docs.nvidia.com/nim/large-language-models/latest/ai-assistant-integrations/claude-code.html>
- NVIDIA NIM tool calling and MCP integration: <https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/tool-calling-and-mcp.html>
- NVIDIA NIM quickstart/model discovery: <https://docs.nvidia.com/nim/large-language-models/latest/get-started/quickstart.html>
- NVIDIA supported models: <https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html>

### Developer projects reviewed for protocol-conversion patterns

These are reference implementations, not proposed runtime dependencies:

- `oai2ant` (pure Anthropic/OpenAI conversion and SSE state handling): <https://github.com/PeronGH/oai2ant>
- `openai-anthropic-converter` (Python request/response/stream conversion patterns): <https://github.com/wenmengzhou/openai-anthropic-converter>
- `chat-to-claude-code` (one-command Claude Code proxy and Anthropic SSE mapping): <https://github.com/chinfeng/chat-to-claude-code>
- `free-llm-proxy` (pure Python Anthropic translation module and tests): <https://github.com/promsoft/free-llm-proxy>
