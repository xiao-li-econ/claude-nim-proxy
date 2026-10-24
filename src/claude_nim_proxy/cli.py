"""Command-line entry point for claude-nim.

Phase 1 provides the command and configuration skeleton. Networked commands intentionally
return a clear not-implemented status until their corresponding implementation phase.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import AppConfig, ConfigError, UpstreamAuth, load_config

_PROGRAM_NAME = "claude-nim"


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI parser."""

    parser = argparse.ArgumentParser(
        prog=_PROGRAM_NAME,
        description="Use NVIDIA NIM models from Claude Code through a local Anthropic proxy.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the proxy and Claude Code together")
    run.add_argument("project_dir", nargs="?", default=".", help="Claude Code project directory")
    _add_common_options(run)

    serve = commands.add_parser("serve", help="run only the local Anthropic proxy")
    token_source = serve.add_mutually_exclusive_group()
    token_source.add_argument(
        "--token-stdin", action="store_true", help="read the local token from stdin"
    )
    token_source.add_argument("--token-file", type=Path, help="read the local token from a file")
    _add_common_options(serve)

    auth = commands.add_parser("auth", help="manage the NVIDIA API credential")
    auth_commands = auth.add_subparsers(dest="auth_command", required=True)
    login = auth_commands.add_parser("login", help="store an NVIDIA API credential")
    login.add_argument(
        "--replace", action="store_true", help="replace an existing stored credential"
    )
    auth_commands.add_parser("status", help="show credential readiness without revealing the key")
    auth_commands.add_parser("verify", help="verify the credential with a billable request")
    auth_commands.add_parser("logout", help="remove the stored NVIDIA API credential")

    models = commands.add_parser("models", help="inspect and refresh NVIDIA models")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_commands.add_parser("list", help="list cached NVIDIA models and gateway aliases")
    model_commands.add_parser("update", help="refresh the NVIDIA model catalog")

    doctor = commands.add_parser("doctor", help="check proxy prerequisites and connectivity")
    _add_common_options(doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its process exit status."""

    arguments = sys.argv[1:] if argv is None else argv
    own_arguments, passthrough = _split_passthrough(arguments)
    parser = build_parser()
    namespace = parser.parse_args(own_arguments)

    try:
        config = _resolved_config(namespace)
    except ConfigError as exc:
        parser.error(str(exc))
        return 2

    if namespace.command == "run":
        return _run_skeleton(config, namespace.project_dir, passthrough)
    if namespace.command == "serve":
        return _serve_skeleton(config)
    if namespace.command == "auth":
        return _auth_skeleton(namespace.auth_command)
    if namespace.command == "models":
        return _models_skeleton(namespace.models_command)
    if namespace.command == "doctor":
        return _doctor_skeleton(config)
    parser.error(f"unknown command: {namespace.command}")
    return 2


def _split_passthrough(arguments: list[str]) -> tuple[list[str], list[str]]:
    try:
        separator = arguments.index("--")
    except ValueError:
        return arguments, []
    return arguments[:separator], arguments[separator + 1 :]


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="path to config.toml")
    parser.add_argument("--model", help="raw NVIDIA model ID or anthropic-nim/ gateway alias")
    parser.add_argument("--upstream-base-url", help="OpenAI-compatible NVIDIA base URL")
    parser.add_argument(
        "--upstream-auth",
        choices=[item.value for item in UpstreamAuth],
        help="upstream credential policy",
    )
    parser.add_argument("--listen-host", help="local bind address")
    parser.add_argument(
        "--listen-port", type=int, help="local bind port; 0 selects an ephemeral port"
    )
    parser.add_argument("--context-window", type=int, help="fallback model context size")
    parser.add_argument("--max-output-tokens", type=int, help="fallback model output limit")
    parser.add_argument(
        "--debug", action="store_true", default=None, help="enable redacted diagnostics"
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="allow a non-loopback bind address",
    )
    beta = parser.add_mutually_exclusive_group()
    beta.add_argument(
        "--disable-experimental-betas",
        action="store_true",
        default=None,
        help="disable Anthropic experimental beta features",
    )
    beta.add_argument(
        "--enable-experimental-betas",
        action="store_false",
        dest="disable_experimental_betas",
        help="enable Anthropic experimental beta features",
    )


def _resolved_config(namespace: argparse.Namespace) -> AppConfig:
    config_path = getattr(namespace, "config", None)
    config = load_config(config_path)
    updates: dict[str, Any] = {}

    explicit_values = {
        "model": "default_model",
        "upstream_base_url": "upstream_base_url",
        "listen_host": "listen_host",
        "context_window": "context_window",
        "max_output_tokens": "max_output_tokens",
    }
    for attribute, key in explicit_values.items():
        value = getattr(namespace, attribute, None)
        if value is not None:
            updates[key] = value

    upstream_auth = getattr(namespace, "upstream_auth", None)
    if upstream_auth is not None:
        updates["upstream_auth"] = UpstreamAuth(upstream_auth)
    listen_port = getattr(namespace, "listen_port", None)
    if listen_port is not None:
        updates["listen_port"] = listen_port
    disable_experimental_betas = getattr(namespace, "disable_experimental_betas", None)
    if disable_experimental_betas is not None:
        updates["disable_experimental_betas"] = disable_experimental_betas
    debug = getattr(namespace, "debug", None)
    if debug is not None:
        updates["debug"] = debug

    return replace(config, **updates)


def _run_skeleton(config: AppConfig, project_dir: str, passthrough: list[str]) -> int:
    del config, project_dir, passthrough
    print(
        "claude-nim run is not available yet; Phase 1 provides the CLI skeleton only.",
        file=sys.stderr,
    )
    return 1


def _serve_skeleton(config: AppConfig) -> int:
    del config
    print(
        "claude-nim serve is not available yet; Phase 1 provides the CLI skeleton only.",
        file=sys.stderr,
    )
    return 1


def _auth_skeleton(command: str) -> int:
    print(
        f"claude-nim auth {command} is not available yet; "
        "credential support is planned for Phase 4.",
        file=sys.stderr,
    )
    return 1


def _models_skeleton(command: str) -> int:
    print(
        f"claude-nim models {command} is not available yet; model support is planned for Phase 3.",
        file=sys.stderr,
    )
    return 1


def _doctor_skeleton(config: AppConfig) -> int:
    del config
    print(
        "claude-nim doctor is not available yet; connectivity checks are planned for Phase 4.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
