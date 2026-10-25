"""Command-line entry point for claude-nim.

The command tree includes credential management, the standalone gateway, active model
catalog operations, diagnostics, and the one-command Claude Code runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import signal
import socket
import stat
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp

from .config import (
    DEFAULT_MODEL,
    SUPPORTED_CONFIG_KEYS,
    AppConfig,
    ConfigError,
    UpstreamAuth,
    config_sources,
    default_config_path,
    load_config,
    set_config_value,
)
from .credentials import (
    CredentialError,
    CredentialManager,
    CredentialSource,
    ResolvedCredential,
)
from .diagnostics import configure_debug, set_debug_secrets
from .models import (
    CatalogDiff,
    ModelCatalog,
    ModelCatalogError,
    ModelMetadata,
    compare_catalog_ids,
    default_model_cache_path,
    ensure_model_catalog,
    reset_model_catalog,
    resolve_model_catalog,
    resolve_model_metadata,
    save_model_cache,
)
from .pi_models import PiModelListError, run_pi_model_list
from .runner import (
    ClaudeLaunchError,
    create_temporary_settings,
    gateway_environment,
    launch_claude,
    managed_settings_conflicts,
    remove_temporary_settings,
    resolve_claude_executable,
    sanitized_child_environment,
    wait_for_health,
)
from .server import GatewayServer, GatewaySettings
from .translate.anthropic_to_openai import resolve_upstream_model
from .translate.errors import AnthropicError
from .upstream import NvidiaUpstreamClient

_PROGRAM_NAME = "claude-nim"
_MODEL_JSON_HELP = """Model JSON files:

  Active model JSON:
    ~/.config/claude-nim/models.json

  Bundled default JSON:
    src/claude_nim_proxy/data/nvidia_models.json

You can manually edit the active model JSON and set its top-level source to "manual".
`claude-nim models update` replaces it when Pi returns a valid non-empty list.
`claude-nim models reset` restores the bundled default."""


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI parser."""

    parser = argparse.ArgumentParser(
        prog=_PROGRAM_NAME,
        description="Use NVIDIA NIM models from Claude Code through a local Anthropic proxy.",
        epilog=(
            "Model selection: --model > NVIDIA_MODEL > config.toml default_model > "
            f"{DEFAULT_MODEL}\n"
            "Change the default with: claude-nim config set default_model <id>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the proxy and Claude Code together")
    run.add_argument("project_dir", nargs="?", default=".", help="Claude Code project directory")
    run.add_argument(
        "--refresh-models",
        action="store_true",
        help="refresh the active model JSON from Pi before starting",
    )
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
    verify = auth_commands.add_parser(
        "verify", help="verify the credential with a billable request"
    )
    verify.add_argument("--config", type=Path, help="path to config.toml")
    verify.add_argument("--model", help="raw NVIDIA model ID or anthropic-nim/ gateway alias")
    verify.add_argument("--upstream-base-url", help="OpenAI-compatible NVIDIA base URL")
    auth_commands.add_parser("logout", help="remove the stored NVIDIA API credential")

    models = commands.add_parser(
        "models",
        help="inspect and maintain the active NVIDIA model list",
        epilog=_MODEL_JSON_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_list = model_commands.add_parser(
        "list",
        help="list the active NVIDIA models and aliases",
        epilog=_MODEL_JSON_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_list.add_argument("--config", type=Path, help="path to config.toml")
    model_list.add_argument("--json", action="store_true", help="emit JSON output")
    model_update = model_commands.add_parser(
        "update",
        help="replace the active model list with Pi's NVIDIA model list",
        epilog=_MODEL_JSON_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_update.add_argument("--config", type=Path, help="path to config.toml")
    model_update.add_argument("--json", action="store_true", help="emit JSON output")
    model_reset = model_commands.add_parser(
        "reset",
        help="restore the bundled default model list",
        epilog=_MODEL_JSON_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_reset.add_argument("--config", type=Path, help="path to config.toml")
    model_reset.add_argument("--json", action="store_true", help="emit JSON output")

    config_parser = commands.add_parser(
        "config",
        help="inspect and update user configuration",
    )
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    config_set = config_commands.add_parser(
        "set",
        help="set a configuration value in config.toml",
    )
    config_set.add_argument("--config", type=Path, help="path to config.toml")
    config_set.add_argument("key", choices=SUPPORTED_CONFIG_KEYS, help="configuration key")
    config_set.add_argument("value", help="new value")
    config_show = config_commands.add_parser(
        "show",
        help="show the effective configuration and where each value comes from",
    )
    config_show.add_argument("--config", type=Path, help="path to config.toml")
    config_show.add_argument("--json", action="store_true", help="emit JSON output")

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
    configure_debug(config.debug)

    if namespace.command == "run":
        return _run_with_claude(
            config,
            namespace.project_dir,
            passthrough,
            allow_non_loopback=namespace.allow_non_loopback,
            refresh_models=namespace.refresh_models,
        )
    if namespace.command == "serve":
        return _serve_gateway(
            config,
            token_stdin=namespace.token_stdin,
            token_file=namespace.token_file,
            allow_insecure_token_file=namespace.allow_insecure_token_file,
            allow_non_loopback=namespace.allow_non_loopback,
        )
    if namespace.command == "auth":
        return _auth_command(config, namespace)
    if namespace.command == "models":
        return _models_command(
            config,
            namespace.models_command,
            json_output=namespace.json,
        )
    if namespace.command == "config":
        return _config_command(namespace)
    if namespace.command == "doctor":
        return _doctor_command(config, namespace.config or default_config_path())
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


def _run_with_claude(
    config: AppConfig,
    project_dir: str,
    claude_arguments: list[str],
    *,
    allow_non_loopback: bool,
    refresh_models: bool,
) -> int:
    try:
        project_path = Path(project_dir).expanduser().resolve()
    except OSError as exc:
        print(f"Cannot resolve project directory {project_dir}: {exc}", file=sys.stderr)
        return 2
    if not project_path.exists() or not project_path.is_dir():
        print(
            f"Project directory does not exist or is not a directory: {project_path}",
            file=sys.stderr,
        )
        return 2
    if refresh_models:
        try:
            catalog, _ = _refresh_active_models_from_pi()
        except PiModelListError as exc:
            print(
                f"WARNING: Pi model refresh failed; using the existing active model JSON: {exc}",
                file=sys.stderr,
            )
        except ModelCatalogError as exc:
            print(f"Model catalog error: {exc}", file=sys.stderr)
            return 2
        else:
            print(f"Refreshed active model JSON from Pi: {len(catalog.models)} models")
    selected_model_id = config.default_model
    if not selected_model_id:
        print(_missing_model_message(config), file=sys.stderr)
        return 2

    try:
        credential = _resolve_upstream_credential(config)
    except CredentialError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        resolve_claude_executable()
    except ClaudeLaunchError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        return asyncio.run(
            _run_gateway_and_claude(
                config,
                selected_model_id,
                project_path,
                claude_arguments,
                credential,
                allow_non_loopback=allow_non_loopback,
            )
        )
    except (ClaudeLaunchError, CredentialError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


async def _run_gateway_and_claude(
    config: AppConfig,
    selected_model_id: str,
    project_path: Path,
    claude_arguments: list[str],
    credential: ResolvedCredential | None,
    *,
    allow_non_loopback: bool,
) -> int:
    local_token = secrets.token_urlsafe(32)
    set_debug_secrets(
        secret
        for secret in (local_token, credential.api_key if credential is not None else None)
        if secret is not None
    )
    try:
        settings = _gateway_settings(
            config,
            api_key=credential.api_key if credential is not None else None,
            allow_non_loopback=allow_non_loopback,
            local_token=local_token,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    server = GatewayServer(settings)
    temporary_settings: Path | None = None
    try:
        base_url = await server.start()
        await wait_for_health(base_url)
        selected_model = resolve_model_metadata(selected_model_id, config)
        values = gateway_environment(
            config,
            base_url=base_url,
            local_token=local_token,
            model=selected_model,
        )
        conflicts = managed_settings_conflicts(values)
        if conflicts:
            print(
                "WARNING: system-managed Claude Code settings override: "
                + ", ".join(conflicts),
                file=sys.stderr,
            )
        temporary_settings = create_temporary_settings(values)
        child_environment = sanitized_child_environment(os.environ, values)
        return await launch_claude(
            project_dir=project_path,
            claude_arguments=claude_arguments,
            environment=child_environment,
            settings_path=temporary_settings,
        )
    finally:
        if temporary_settings is not None:
            remove_temporary_settings(temporary_settings)
        await server.stop()


def _missing_model_message(config: AppConfig) -> str:
    candidates: list[str] = []
    try:
        catalog = ensure_model_catalog(default_model_cache_path())
        candidates = [
            model.upstream_id
            for model in resolve_model_catalog(catalog, config)
            if model.tool_calling is not False
        ]
    except ModelCatalogError:
        candidates = []
    message = (
        "No NVIDIA model is selected. Use --model, set NVIDIA_MODEL, "
        "or set default_model in config.toml."
    )
    if candidates:
        message += " Active tool-capable candidates: " + ", ".join(candidates[:10]) + "."
    return message


def _resolve_upstream_credential(config: AppConfig) -> ResolvedCredential | None:
    if config.upstream_auth is UpstreamAuth.NONE:
        return None
    manager = CredentialManager()
    credential = manager.resolve(allow_prompt=True)
    if credential is None:
        raise CredentialError(
            "No NVIDIA API credential is configured. Set NVIDIA_API_KEY or run "
            "'claude-nim auth login'."
        )
    if credential.source is CredentialSource.FILE:
        _warn_file_fallback(manager, credential)
    return credential


def _warn_file_fallback(
    manager: CredentialManager,
    credential: ResolvedCredential,
) -> None:
    if credential.source is CredentialSource.FILE:
        print(
            "WARNING: the NVIDIA credential is stored unencrypted in "
            f"{manager.fallback_path} with mode 0600.",
            file=sys.stderr,
        )


def _serve_gateway(
    config: AppConfig,
    *,
    token_stdin: bool,
    token_file: Path | None,
    allow_insecure_token_file: bool,
    allow_non_loopback: bool,
) -> int:
    selected_model_id = config.default_model
    if not selected_model_id:
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

    try:
        credential = _resolve_upstream_credential(config)
    except CredentialError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    set_debug_secrets(
        secret
        for secret in (token, credential.api_key if credential is not None else None)
        if secret
    )

    try:
        settings = _gateway_settings(
            config,
            api_key=credential.api_key if credential is not None else None,
            allow_non_loopback=allow_non_loopback,
            local_token=token,
            selected_model_id=selected_model_id,
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


def _gateway_settings(
    config: AppConfig,
    *,
    api_key: str | None,
    allow_non_loopback: bool,
    local_token: str | None = None,
    selected_model_id: str | None = None,
) -> GatewaySettings:
    resolved_model_id = selected_model_id or config.default_model
    if not resolved_model_id:
        raise ValueError("No NVIDIA model is selected.")
    try:
        catalog = ensure_model_catalog(default_model_cache_path())
    except ModelCatalogError as exc:
        raise ValueError(f"Invalid active model catalog: {exc}") from exc
    catalog_models = resolve_model_catalog(catalog, config)
    selected_model = resolve_model_metadata(resolved_model_id, config, catalog)
    picker_model_ids = tuple(model.upstream_id for model in catalog_models)
    return GatewaySettings(
        local_token=local_token or secrets.token_urlsafe(32),
        default_model=resolved_model_id,
        upstream_base_url=config.upstream_base_url,
        api_key=api_key,
        upstream_auth=config.upstream_auth,
        listen_host=config.listen_host,
        listen_port=config.listen_port,
        upstream_connect_timeout_seconds=config.upstream_connect_timeout_seconds,
        upstream_read_timeout_seconds=config.upstream_read_timeout_seconds,
        supports_images=selected_model.supports_images,
        model_ids=picker_model_ids,
        model_display_names={
            model.upstream_id: model.display_name for model in (selected_model, *catalog_models)
        },
        model_image_support={
            model.upstream_id: model.supports_images
            for model in (selected_model, *catalog_models)
        },
        known_model_ids=tuple(model.upstream_id for model in (selected_model, *catalog_models)),
        context_window=selected_model.context_window,
        max_output_tokens=selected_model.max_output_tokens,
        allow_non_loopback=allow_non_loopback,
    )


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


def _auth_command(config: AppConfig, namespace: argparse.Namespace) -> int:
    command = namespace.auth_command
    manager = CredentialManager()
    try:
        if command == "login":
            return _auth_login(manager, replace=namespace.replace)
        if command == "status":
            return _auth_status(manager)
        if command == "verify":
            verify_model = namespace.model or config.verify_model or config.default_model
            return _auth_verify(config, manager, verify_model)
        if command == "logout":
            return _auth_logout(manager)
    except CredentialError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Unknown auth command: {command}", file=sys.stderr)
    return 2


def _auth_login(manager: CredentialManager, *, replace: bool) -> int:
    if manager.has_stored_credential() and not replace:
        print(
            "A stored NVIDIA credential already exists. Use 'claude-nim auth login --replace' "
            "to replace it.",
            file=sys.stderr,
        )
        return 2
    source = manager.prompt_and_store()
    if source is CredentialSource.FILE:
        print(
            f"WARNING: no OS credential store is available. The unencrypted credential was "
            f"stored in {manager.fallback_path} with mode 0600.",
            file=sys.stderr,
        )
        print("Stored NVIDIA credential in file fallback.", file=sys.stderr)
    else:
        print("Stored NVIDIA credential in the OS credential store.", file=sys.stderr)
    return 0


def _auth_status(manager: CredentialManager) -> int:
    credential = manager.resolve(allow_prompt=False)
    if credential is None:
        print("NVIDIA credential: not configured", file=sys.stderr)
        return 2
    print(f"NVIDIA credential: configured ({credential.source})", file=sys.stderr)
    if credential.source is CredentialSource.FILE:
        print(
            f"WARNING: the credential is stored unencrypted in {manager.fallback_path}.",
            file=sys.stderr,
        )
    return 0


def _auth_verify(
    config: AppConfig,
    manager: CredentialManager,
    verify_model: str | None,
) -> int:
    if not verify_model:
        print(
            "No verification model is selected. Use --model, set CLAUDE_NIM_VERIFY_MODEL, "
            "or set verify_model/default_model in config.toml.",
            file=sys.stderr,
        )
        return 2
    credential = manager.resolve(allow_prompt=True)
    if credential is None:
        print(
            "No NVIDIA API credential is configured. Set NVIDIA_API_KEY or run "
            "'claude-nim auth login'.",
            file=sys.stderr,
        )
        return 2
    if credential.source is CredentialSource.FILE:
        _warn_file_fallback(manager, credential)
    set_debug_secrets((credential.api_key,))
    print(
        "Sending one minimal authenticated NVIDIA request; this may consume a small amount "
        "of API credit.",
        file=sys.stderr,
    )
    raw_model = resolve_upstream_model(verify_model)
    payload = {
        "model": raw_model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        return asyncio.run(
            _auth_verify_request(
                config,
                credential.api_key,
                payload,
            )
        )
    except AnthropicError as exc:
        print(f"NVIDIA credential verification failed: {exc.message}", file=sys.stderr)
        return 2


async def _auth_verify_request(
    config: AppConfig,
    api_key: str,
    payload: dict[str, Any],
) -> int:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        client = NvidiaUpstreamClient(
            session,
            base_url=config.upstream_base_url,
            api_key=api_key,
            auth_mode=UpstreamAuth.REQUIRED,
            connect_timeout_seconds=config.upstream_connect_timeout_seconds,
            read_timeout_seconds=config.upstream_read_timeout_seconds,
        )
        response = await client.complete(payload)
    print("NVIDIA credential verified successfully.", file=sys.stderr)
    del response
    return 0


def _auth_logout(manager: CredentialManager) -> int:
    removed = manager.delete_stored()
    if removed:
        print("Removed the stored NVIDIA credential.", file=sys.stderr)
    else:
        print("No stored NVIDIA credential was found.", file=sys.stderr)
    if os.environ.get("NVIDIA_API_KEY"):
        print(
            "The NVIDIA_API_KEY environment variable was not changed.",
            file=sys.stderr,
        )
    return 0


def _models_command(config: AppConfig, command: str, *, json_output: bool) -> int:
    if command == "list":
        return _models_list(config, json_output=json_output)
    if command == "update":
        try:
            return _models_update(config, json_output=json_output)
        except PiModelListError as exc:
            print(f"Pi model refresh failed: {exc}", file=sys.stderr)
            print(
                "Maintain the active model JSON manually at "
                "~/.config/claude-nim/models.json.",
                file=sys.stderr,
            )
            return 2
        except ModelCatalogError as exc:
            print(f"Model catalog error: {exc}", file=sys.stderr)
            return 2
    if command == "reset":
        return _models_reset(config, json_output=json_output)
    print(f"Unknown models command: {command}", file=sys.stderr)
    return 2


def _config_command(namespace: argparse.Namespace) -> int:
    path = namespace.config or default_config_path()
    if namespace.config_command == "set":
        try:
            config = set_config_value(path, namespace.key, namespace.value)
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        values = _config_values(config)
        print(f"Set {namespace.key} = {_config_value_text(values[namespace.key])} in {path}")
        print(f"Effective {namespace.key}: {_config_value_text(values[namespace.key])}")
        return 0
    if namespace.config_command == "show":
        try:
            config = load_config(path)
            sources = config_sources(path)
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        values = _config_values(config)
        if namespace.json:
            print(
                json.dumps(
                    {
                        "path": str(path),
                        "exists": path.exists(),
                        "values": values,
                        "sources": sources,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        state = "exists" if path.exists() else "not created"
        print(f"Config file: {path} ({state})")
        print("Effective configuration:")
        for key, value in values.items():
            print(f"  {key} = {_config_value_text(value)}  [{_source_label(sources[key])}]")
        return 0
    print(f"Unknown config command: {namespace.config_command}", file=sys.stderr)
    return 2


def _config_values(config: AppConfig) -> dict[str, Any]:
    return {
        "default_model": config.default_model,
        "verify_model": config.verify_model,
        "upstream_base_url": config.upstream_base_url,
        "upstream_auth": config.upstream_auth.value,
        "listen_host": config.listen_host,
        "listen_port": config.listen_port,
        "context_window": config.context_window,
        "max_output_tokens": config.max_output_tokens,
        "disable_experimental_betas": config.disable_experimental_betas,
        "upstream_connect_timeout_seconds": config.upstream_connect_timeout_seconds,
        "upstream_read_timeout_seconds": config.upstream_read_timeout_seconds,
        "debug": config.debug,
    }


def _config_value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _source_label(source: str) -> str:
    if source == "default":
        return "built-in default"
    return source


def _models_list(config: AppConfig, *, json_output: bool) -> int:
    catalog_path = default_model_cache_path()
    try:
        catalog = ensure_model_catalog(catalog_path)
    except ModelCatalogError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    models = resolve_model_catalog(catalog, config)

    if json_output:
        print(
            json.dumps(
                {
                    "version": catalog.version,
                    "source": catalog.source,
                    "updated_at": catalog.updated_at,
                    "models": [model.to_dict() for model in models],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"NVIDIA models: {len(models)}")
    print(f"Catalog source: {catalog.source}")
    print(f"Catalog updated: {catalog.updated_at or 'not set (bundled initialization)'}")
    for model in models:
        tools = _unknown_text(model.tool_calling)
        reasoning = _unknown_text(model.reasoning)
        print(f"{model.upstream_id} -> {model.gateway_id}")
        print(
            f"  {model.display_name}; context={model.context_window}; "
            f"output={model.max_output_tokens}; input={','.join(model.input)}; "
            f"reasoning={reasoning}; tools={tools}; metadata={model.metadata_source}"
        )
    return 0


def _models_update(config: AppConfig, *, json_output: bool) -> int:
    del config
    catalog_path = default_model_cache_path()
    try:
        catalog, difference = _refresh_active_models_from_pi()
    except ModelCatalogError:
        raise
    if json_output:
        print(
            json.dumps(
                {
                    "version": catalog.version,
                    "source": catalog.source,
                    "updated_at": catalog.updated_at,
                    "model_count": len(catalog.models),
                    "added": list(difference.added),
                    "removed": list(difference.removed),
                    "unchanged": list(difference.unchanged),
                    "catalog_path": str(catalog_path),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"Updated active model JSON from Pi: {len(catalog.models)} models")
    print(f"Added: {len(difference.added)}; removed: {len(difference.removed)}")
    print(f"Active model JSON: {catalog_path}")
    return 0


def _models_reset(config: AppConfig, *, json_output: bool) -> int:
    del config
    catalog_path = default_model_cache_path()
    try:
        catalog = reset_model_catalog(catalog_path)
    except ModelCatalogError as exc:
        print(f"Model catalog error: {exc}", file=sys.stderr)
        return 2
    if json_output:
        print(
            json.dumps(
                {
                    "version": catalog.version,
                    "source": catalog.source,
                    "updated_at": catalog.updated_at,
                    "model_count": len(catalog.models),
                    "catalog_path": str(catalog_path),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    print(f"Restored bundled model list: {len(catalog.models)} models")
    print(f"Active model JSON: {catalog_path}")
    return 0


def _refresh_active_models_from_pi() -> tuple[ModelCatalog, CatalogDiff]:
    catalog_path = default_model_cache_path()
    previous = ensure_model_catalog(catalog_path)
    pi_models = run_pi_model_list()
    metadata = tuple(
        ModelMetadata(
            upstream_id=model.upstream_id,
            display_name=model.display_name,
            context_window=model.context_window,
            max_output_tokens=model.max_output_tokens,
            reasoning=model.reasoning,
            input=model.input,
            tool_calling=None,
            source=("nvidia",),
            metadata_source="pi",
        )
        for model in sorted(pi_models, key=lambda item: item.upstream_id)
    )
    updated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    catalog = ModelCatalog(
        source="pi",
        updated_at=updated_at,
        models=metadata,
    )
    difference = compare_catalog_ids(
        (model.upstream_id for model in previous.models),
        (model.upstream_id for model in metadata),
    )
    save_model_cache(catalog, catalog_path)
    return catalog, difference


def _unknown_text(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _doctor_command(config: AppConfig, config_path: Path) -> int:
    results: list[tuple[bool, str]] = []

    try:
        executable = resolve_claude_executable()
        results.append((True, f"Claude Code: found at {executable}"))
    except ClaudeLaunchError as exc:
        results.append((False, str(exc)))

    selected_model = config.default_model
    if selected_model:
        model_source = _source_label(config_sources(config_path).get("default_model", "default"))
        results.append(
            (True, f"Model: {resolve_upstream_model(selected_model)} ({model_source})")
        )
    else:
        results.append(
            (
                False,
                "Model: not selected; use --model, NVIDIA_MODEL, or default_model",
            )
        )

    credential: ResolvedCredential | None = None
    credential_error: CredentialError | None = None
    try:
        credential = CredentialManager().resolve(allow_prompt=False)
    except CredentialError as exc:
        credential_error = exc
        results.append((False, f"Credential: {exc}"))
    if credential is not None:
        results.append((True, f"Credential: configured ({credential.source})"))
        set_debug_secrets((credential.api_key,))
        if credential.source is CredentialSource.FILE:
            print(
                "WARNING: the NVIDIA credential is stored unencrypted in "
                f"{CredentialManager().fallback_path} with mode 0600.",
                file=sys.stderr,
            )
    elif credential_error is None:
        results.append(
            (
                False,
                "Credential: not configured; set NVIDIA_API_KEY or run 'claude-nim auth login'",
            )
        )

    port_ready, port_message = _doctor_port(config.listen_host, config.listen_port)
    results.append((port_ready, f"Listener: {port_message}"))

    if selected_model and credential is not None:
        print(
            "The model entitlement probe sends one minimal authenticated request and may "
            "consume a small amount of API credit.",
            file=sys.stderr,
        )
        upstream_ready, upstream_message = asyncio.run(
            _doctor_upstream(config, credential.api_key, selected_model)
        )
        results.append((upstream_ready, f"NVIDIA upstream: {upstream_message}"))
    else:
        results.append((False, "NVIDIA upstream: not checked because prerequisites failed"))

    for passed, message in results:
        print(f"{'PASS' if passed else 'FAIL'}: {message}")
    return 0 if all(passed for passed, _ in results) else 2


def _doctor_port(host: str, port: int) -> tuple[bool, str]:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as listener:
            listener.bind((host, port))
    except OSError as exc:
        return False, f"cannot bind {host}:{port} ({exc.strerror or exc})"
    return True, f"{host}:{port or 'ephemeral'} is available"


async def _doctor_upstream(
    config: AppConfig,
    api_key: str,
    selected_model: str,
) -> tuple[bool, str]:
    raw_model = resolve_upstream_model(selected_model)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            upstream_client = NvidiaUpstreamClient(
                session,
                base_url=config.upstream_base_url,
                api_key=api_key,
                auth_mode=UpstreamAuth.REQUIRED,
                connect_timeout_seconds=config.upstream_connect_timeout_seconds,
                read_timeout_seconds=config.upstream_read_timeout_seconds,
            )
            await upstream_client.complete(
                {
                    "model": raw_model,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "max_tokens": 1,
                    "stream": False,
                }
            )
    except AnthropicError as exc:
        if exc.status == 404:
            return (
                False,
                f"model {raw_model} is not available to this NVIDIA account or upstream",
            )
        return False, exc.message
    return True, f"model availability and account entitlement check passed for {raw_model}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
