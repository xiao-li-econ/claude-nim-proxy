"""Command-line entry point for claude-nim.

The standalone gateway is implemented in Phase 2. Commands that depend on later model,
credential, or runner phases return a clear not-implemented status.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import signal
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import AppConfig, ConfigError, UpstreamAuth, load_config
from .server import GatewayServer, GatewaySettings

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
    serve.add_argument(
        "--allow-insecure-token-file",
        action="store_true",
        help="allow a group- or world-accessible token file",
    )
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
        return _serve_gateway(
            config,
            token_stdin=namespace.token_stdin,
            token_file=namespace.token_file,
            allow_insecure_token_file=namespace.allow_insecure_token_file,
            allow_non_loopback=namespace.allow_non_loopback,
        )
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
        "claude-nim run is not available yet; the one-command runner is planned for Phase 4.",
        file=sys.stderr,
    )
    return 1


def _serve_gateway(
    config: AppConfig,
    *,
    token_stdin: bool,
    token_file: Path | None,
    allow_insecure_token_file: bool,
    allow_non_loopback: bool,
) -> int:
    if not config.default_model:
        print(
            "No NVIDIA model is selected. Use --model, set NVIDIA_MODEL, "
            "or set default_model in config.toml.",
            file=sys.stderr,
        )
        return 2

    try:
        token, generated = _resolve_local_token(
            token_stdin=token_stdin,
            token_file=token_file,
            allow_insecure_token_file=allow_insecure_token_file,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    api_key = (
        None if config.upstream_auth is UpstreamAuth.NONE else os.environ.get("NVIDIA_API_KEY")
    )
    if config.upstream_auth is not UpstreamAuth.NONE and not api_key:
        print(
            "No NVIDIA API credential is configured. Set NVIDIA_API_KEY; "
            "OS credential-store support will be added with the auth commands.",
            file=sys.stderr,
        )
        return 2

    try:
        settings = GatewaySettings(
            local_token=token,
            default_model=config.default_model,
            upstream_base_url=config.upstream_base_url,
            api_key=api_key,
            upstream_auth=config.upstream_auth,
            listen_host=config.listen_host,
            listen_port=config.listen_port,
            upstream_connect_timeout_seconds=config.upstream_connect_timeout_seconds,
            upstream_read_timeout_seconds=config.upstream_read_timeout_seconds,
            allow_non_loopback=allow_non_loopback,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if settings.binds_non_loopback:
        print(
            "WARNING: the proxy is listening on a non-loopback address. "
            "Local bearer authentication remains enabled.",
            file=sys.stderr,
        )
    try:
        return asyncio.run(_serve_until_stopped(settings, token, generated))
    except KeyboardInterrupt:
        return 0


async def _serve_until_stopped(
    settings: GatewaySettings,
    token: str,
    generated: bool,
) -> int:
    server = GatewayServer(settings)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, stop_event.set)
        except (NotImplementedError, RuntimeError):
            continue
        installed_signals.append(signal_number)

    try:
        base_url = await server.start()
        if generated:
            print(
                f"export ANTHROPIC_BASE_URL='{base_url}'\nexport ANTHROPIC_AUTH_TOKEN='{token}'",
                file=sys.stderr,
            )
        else:
            print(f"claude-nim proxy listening at {base_url}", file=sys.stderr)
        await stop_event.wait()
    finally:
        for signal_number in installed_signals:
            loop.remove_signal_handler(signal_number)
        await server.stop()
    return 0


def _resolve_local_token(
    *,
    token_stdin: bool,
    token_file: Path | None,
    allow_insecure_token_file: bool,
) -> tuple[str, bool]:
    if token_stdin:
        token = sys.stdin.readline().rstrip("\r\n")
        if not token:
            raise ValueError("No local bearer token was provided on stdin.")
        return token, False

    if token_file is not None:
        try:
            mode = stat.S_IMODE(token_file.stat().st_mode)
        except OSError as exc:
            raise ValueError(f"Cannot read token file {token_file}: {exc.strerror}") from exc
        if mode & 0o077:
            if not allow_insecure_token_file:
                raise ValueError(
                    f"Token file {token_file} is group/world accessible. "
                    "Use chmod 600 or pass --allow-insecure-token-file."
                )
            print(
                f"WARNING: token file {token_file} is group/world accessible.",
                file=sys.stderr,
            )
        try:
            token = token_file.read_text(encoding="utf-8").rstrip("\r\n")
        except OSError as exc:
            raise ValueError(f"Cannot read token file {token_file}: {exc.strerror}") from exc
        if not token:
            raise ValueError(f"Token file {token_file} is empty.")
        return token, False

    return secrets.token_urlsafe(32), True


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
