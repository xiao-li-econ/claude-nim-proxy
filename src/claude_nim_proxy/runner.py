"""Launch Claude Code against the local proxy with isolated gateway settings."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import aiohttp

from .config import AppConfig
from .models import ModelInfo

_CLAUDE_ENVIRONMENT_KEYS = {
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_CUSTOM_MODEL_OPTION",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
}
_SECRET_ENVIRONMENT_KEYS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "NVIDIA_API_KEY",
}
_TERMINATION_GRACE_SECONDS = 5.0


class ClaudeLaunchError(RuntimeError):
    """Raised when Claude Code cannot be found or launched safely."""


def resolve_claude_executable() -> str:
    """Resolve the Claude Code executable from PATH."""

    executable = shutil.which("claude")
    if executable is None:
        raise ClaudeLaunchError(
            "Claude Code was not found in PATH. Install Claude Code and try again."
        )
    return executable


def gateway_environment(
    config: AppConfig,
    *,
    base_url: str,
    local_token: str,
    model: ModelInfo,
) -> dict[str, str]:
    """Build the gateway-only environment shared by the process and settings file."""

    gateway_id = model.gateway_id
    return {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": local_token,
        "ANTHROPIC_MODEL": gateway_id,
        "ANTHROPIC_CUSTOM_MODEL_OPTION": gateway_id,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": gateway_id,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": gateway_id,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": gateway_id,
        "ANTHROPIC_DEFAULT_FABLE_MODEL": gateway_id,
        "CLAUDE_CODE_SUBAGENT_MODEL": gateway_id,
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(model.context_window),
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": (
            "1" if config.disable_experimental_betas else "0"
        ),
    }


def sanitized_child_environment(
    parent_environment: Mapping[str, str],
    gateway_values: Mapping[str, str],
) -> dict[str, str]:
    """Return a child environment with inherited provider credentials removed."""

    child = dict(parent_environment)
    for key in _SECRET_ENVIRONMENT_KEYS:
        child.pop(key, None)
    for key in _CLAUDE_ENVIRONMENT_KEYS:
        child.pop(key, None)
    child.update(gateway_values)
    return child


def create_temporary_settings(gateway_values: Mapping[str, str]) -> Path:
    """Create a mode-0600 Claude Code settings file containing only gateway values."""

    payload = {"env": dict(gateway_values)}
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="claude-nim-settings-",
            suffix=".json",
            text=True,
        )
    except OSError as exc:
        raise ClaudeLaunchError(f"Cannot create temporary Claude settings: {exc.strerror}") from exc
    path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as settings_file:
            os.chmod(path, 0o600)
            json.dump(payload, settings_file, indent=2)
            settings_file.write("\n")
            settings_file.flush()
            os.fsync(settings_file.fileno())
        os.chmod(path, 0o600)
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise ClaudeLaunchError(f"Cannot write temporary Claude settings: {exc.strerror}") from exc
    return path


def remove_temporary_settings(path: Path) -> None:
    """Remove a temporary settings file, ignoring an already-removed file."""

    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def managed_settings_conflicts(gateway_values: Mapping[str, str]) -> tuple[str, ...]:
    """Return gateway keys overridden by system-managed Claude Code settings.

    Managed settings are policy-controlled and cannot be replaced by the temporary
    ``--settings`` file. Conflicting keys are reported so the user can diagnose why
    Claude Code may ignore the local gateway model.
    """

    paths = [Path("/etc/claude-code/managed-settings.json")]
    if os.name == "darwin":
        paths.append(Path("/Library/Application Support/ClaudeCode/managed-settings.json"))
    conflicts: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        managed_environment = payload.get("env")
        if not isinstance(managed_environment, dict):
            continue
        for key, expected in gateway_values.items():
            if key in managed_environment and managed_environment[key] != expected:
                conflicts.append(key)
    return tuple(sorted(set(conflicts)))


async def wait_for_health(base_url: str, *, timeout_seconds: float = 5.0) -> None:
    """Wait until the local gateway reports that it is ready."""

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_error: BaseException | None = None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1.0)) as session:
        while asyncio.get_running_loop().time() < deadline:
            try:
                async with session.get(f"{base_url}/healthz") as response:
                    if response.status == 200:
                        return
            except (TimeoutError, aiohttp.ClientError) as exc:
                last_error = exc
            await asyncio.sleep(0.05)
    raise ClaudeLaunchError(f"Local proxy health check failed: {last_error or 'timeout'}")


async def launch_claude(
    *,
    project_dir: Path,
    claude_arguments: Sequence[str],
    environment: Mapping[str, str],
    settings_path: Path,
    executable: str | None = None,
) -> int:
    """Launch Claude Code in its own process group and return its exit status.

    SIGINT and SIGTERM received by the proxy are forwarded to Claude Code. A bounded
    termination grace period prevents an orphaned child process.
    """

    command = executable or resolve_claude_executable()
    process_options: dict[str, Any] = {}
    if os.name == "posix":
        process_options["start_new_session"] = True
    try:
        process = await asyncio.create_subprocess_exec(
            command,
            "--settings",
            str(settings_path),
            *claude_arguments,
            cwd=project_dir,
            env=dict(environment),
            **process_options,
        )
    except OSError as exc:
        raise ClaudeLaunchError(f"Cannot launch Claude Code: {exc.strerror or exc}") from exc

    loop = asyncio.get_running_loop()
    signal_event = asyncio.Event()
    forwarded_signal: signal.Signals | None = None

    def forward(received: signal.Signals) -> None:
        nonlocal forwarded_signal
        if forwarded_signal is None:
            forwarded_signal = received
            _signal_process_group(process.pid, received, force=os.name == "posix")
        else:
            _signal_process_group(process.pid, signal.SIGKILL, force=os.name == "posix")
        signal_event.set()

    installed_signals: list[signal.Signals] = []
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, forward, signal_number)
        except (NotImplementedError, RuntimeError):
            continue
        installed_signals.append(signal_number)

    try:
        try:
            wait_task = asyncio.create_task(process.wait())
            signal_task = asyncio.create_task(signal_event.wait())
            done, pending = await asyncio.wait(
                (wait_task, signal_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait_task in done:
                signal_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await signal_task
                return wait_task.result()
            signal_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await signal_task
            try:
                await asyncio.wait_for(wait_task, timeout=_TERMINATION_GRACE_SECONDS)
            except TimeoutError:
                _signal_process_group(process.pid, signal.SIGTERM, force=os.name == "posix")
                try:
                    await asyncio.wait_for(wait_task, timeout=_TERMINATION_GRACE_SECONDS)
                except TimeoutError:
                    _signal_process_group(process.pid, signal.SIGKILL, force=os.name == "posix")
                    await wait_task
            return wait_task.result()
        except asyncio.CancelledError:
            _signal_process_group(process.pid, signal.SIGTERM, force=os.name == "posix")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=_TERMINATION_GRACE_SECONDS)
            if process.returncode is None:
                _signal_process_group(process.pid, signal.SIGKILL, force=os.name == "posix")
                await process.wait()
            raise
    finally:
        for signal_number in installed_signals:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(signal_number)


def _signal_process_group(pid: int, signal_number: signal.Signals, *, force: bool) -> None:
    if force:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal_number)
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal_number)
