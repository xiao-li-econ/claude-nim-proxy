# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below `1.0.0`, minor releases may contain breaking changes. User
state in `~/.config/claude-nim/` (the active model JSON and `config.toml`) is owned by the
user and is never modified by an upgrade.

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-10-25

### Added

- Local Anthropic Messages gateway on `127.0.0.1` with a per-run bearer token, translated
  to NVIDIA NIM's OpenAI-compatible Chat Completions API.
- Streaming and non-streaming translation, system prompts, images, custom tools, parallel
  tool calls, tool results, and reasoning passthrough.
- Claude-compatible model discovery through reversible `anthropic-nim/...` aliases.
- One-command runner (`claude-nim run`) that starts the proxy, launches Claude Code with an
  isolated mode-`0600` settings file, and cleans up on exit or signal.
- Credential lifecycle commands (`auth login`, `auth status`, `auth verify`, `auth logout`)
  with OS credential storage and an explicitly warned mode-`0600` file fallback.
- Pi-backed active model catalog: `models update` replaces the active JSON with
  `pi --list-models nvidia`, `models reset` restores the bundled template, and
  `run --refresh-models` refreshes before startup.
- Shipped default model `minimaxai/minimax-m3.1-flash`, plus `config show` and
  `config set` for inspecting and storing user defaults.
- Redacted metadata-only debug tracing and an opt-in Claude Code compatibility suite.
- Installation documentation, MIT license, and automated CI and release workflows.

[Unreleased]: https://github.com/xiao-li-econ/claude-nim-proxy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/xiao-li-econ/claude-nim-proxy/releases/tag/v0.1.0
