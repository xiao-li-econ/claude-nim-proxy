"""Typed configuration loading for the Claude Code to NVIDIA NIM proxy."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_UPSTREAM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 0
DEFAULT_CONTEXT_WINDOW = 32_768
DEFAULT_MAX_OUTPUT_TOKENS = 4_096
DEFAULT_UPSTREAM_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_UPSTREAM_READ_TIMEOUT_SECONDS = 600.0


class ConfigError(ValueError):
    """Raised when user configuration is invalid."""


class UpstreamAuth(StrEnum):
    """Policy for sending credentials to the configured upstream."""

    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ModelOverride:
    """Capability metadata supplied by the user for one raw NVIDIA model ID."""

    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    reasoning: bool | None = None
    input: tuple[str, ...] | None = None
    tool_calling: bool | None = None


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Fully resolved proxy configuration.

    Values are resolved using the documented precedence order: explicit CLI values are
    applied by the caller after this loader returns environment-over-config-over-default
    values.
    """

    default_model: str | None = None
    verify_model: str | None = None
    upstream_base_url: str = DEFAULT_UPSTREAM_BASE_URL
    upstream_auth: UpstreamAuth = UpstreamAuth.AUTO
    listen_host: str = DEFAULT_LISTEN_HOST
    listen_port: int = DEFAULT_LISTEN_PORT
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    disable_experimental_betas: bool = True
    upstream_connect_timeout_seconds: float = DEFAULT_UPSTREAM_CONNECT_TIMEOUT_SECONDS
    upstream_read_timeout_seconds: float = DEFAULT_UPSTREAM_READ_TIMEOUT_SECONDS
    debug: bool = False
    model_overrides: Mapping[str, ModelOverride] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_upstream_base_url(self.upstream_base_url)


_TOP_LEVEL_KEYS = {
    "default_model",
    "verify_model",
    "upstream_base_url",
    "upstream_auth",
    "listen_host",
    "listen_port",
    "context_window",
    "max_output_tokens",
    "disable_experimental_betas",
    "upstream_connect_timeout_seconds",
    "upstream_read_timeout_seconds",
    "debug",
    "model_overrides",
}
_OVERRIDE_KEYS = {
    "display_name",
    "context_window",
    "max_output_tokens",
    "reasoning",
    "input",
    "tool_calling",
}
_ENVIRONMENT_KEYS = {
    "NVIDIA_MODEL": "default_model",
    "NVIDIA_BASE_URL": "upstream_base_url",
    "CLAUDE_NIM_UPSTREAM_AUTH": "upstream_auth",
    "CLAUDE_NIM_VERIFY_MODEL": "verify_model",
    "CLAUDE_NIM_LISTEN_HOST": "listen_host",
    "CLAUDE_NIM_LISTEN_PORT": "listen_port",
    "CLAUDE_NIM_CONTEXT_WINDOW": "context_window",
    "CLAUDE_NIM_MAX_OUTPUT_TOKENS": "max_output_tokens",
    "CLAUDE_NIM_DISABLE_EXPERIMENTAL_BETAS": "disable_experimental_betas",
    "CLAUDE_NIM_UPSTREAM_CONNECT_TIMEOUT_SECONDS": "upstream_connect_timeout_seconds",
    "CLAUDE_NIM_UPSTREAM_READ_TIMEOUT_SECONDS": "upstream_read_timeout_seconds",
    "CLAUDE_NIM_DEBUG": "debug",
}


def default_config_path() -> Path:
    """Return the per-user configuration path."""

    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "claude-nim" / "config.toml"


def load_config(
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load and validate config.toml, then apply environment overrides.

    A missing config file is valid. Unknown keys and invalid values are rejected rather
    than silently ignored.
    """

    environment = os.environ if environ is None else environ
    path = config_path if config_path is not None else default_config_path()
    file_values = _read_config_file(path)
    values: dict[str, Any] = {
        "default_model": None,
        "verify_model": None,
        "upstream_base_url": DEFAULT_UPSTREAM_BASE_URL,
        "upstream_auth": UpstreamAuth.AUTO.value,
        "listen_host": DEFAULT_LISTEN_HOST,
        "listen_port": DEFAULT_LISTEN_PORT,
        "context_window": DEFAULT_CONTEXT_WINDOW,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "disable_experimental_betas": True,
        "upstream_connect_timeout_seconds": DEFAULT_UPSTREAM_CONNECT_TIMEOUT_SECONDS,
        "upstream_read_timeout_seconds": DEFAULT_UPSTREAM_READ_TIMEOUT_SECONDS,
        "debug": False,
        "model_overrides": file_values.get("model_overrides", {}),
    }
    values.update({key: value for key, value in file_values.items() if key != "model_overrides"})
    _apply_environment(values, environment)

    default_model = _optional_string(values, "default_model")
    verify_model = _optional_string(values, "verify_model")
    upstream_base_url = _required_string(values, "upstream_base_url")
    listen_host = _required_string(values, "listen_host")

    try:
        upstream_auth = UpstreamAuth(_required_string(values, "upstream_auth"))
    except ValueError as exc:
        raise ConfigError("upstream_auth must be one of: auto, required, none") from exc

    return AppConfig(
        default_model=default_model,
        verify_model=verify_model if verify_model is not None else default_model,
        upstream_base_url=upstream_base_url,
        upstream_auth=upstream_auth,
        listen_host=listen_host,
        listen_port=_integer(values, "listen_port", minimum=0),
        context_window=_integer(values, "context_window", minimum=1),
        max_output_tokens=_integer(values, "max_output_tokens", minimum=1),
        disable_experimental_betas=_boolean(values, "disable_experimental_betas"),
        upstream_connect_timeout_seconds=_number(
            values, "upstream_connect_timeout_seconds", minimum=0.0, exclusive=True
        ),
        upstream_read_timeout_seconds=_number(
            values, "upstream_read_timeout_seconds", minimum=0.0, exclusive=True
        ),
        debug=_boolean(values, "debug"),
        model_overrides=_parse_model_overrides(values["model_overrides"]),
    )


def _validate_upstream_base_url(value: str) -> None:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise ConfigError("upstream_base_url must be a valid HTTP(S) URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError("upstream_base_url must be a valid HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError("upstream_base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ConfigError("upstream_base_url must not contain a query or fragment")


def _read_config_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {path}: {exc.strerror}") from exc

    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    unknown = sorted(set(parsed) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"unknown configuration key: {unknown[0]}")
    return parsed


def _apply_environment(values: dict[str, Any], environment: Mapping[str, str]) -> None:
    for variable, key in _ENVIRONMENT_KEYS.items():
        if variable not in environment:
            continue
        raw_value = environment[variable]
        if key in {"listen_port", "context_window", "max_output_tokens"}:
            try:
                values[key] = int(raw_value)
            except ValueError as exc:
                raise ConfigError(f"{key} must be an integer") from exc
        elif key in {
            "upstream_connect_timeout_seconds",
            "upstream_read_timeout_seconds",
        }:
            try:
                values[key] = float(raw_value)
            except ValueError as exc:
                raise ConfigError(f"{key} must be a number") from exc
        elif key in {"disable_experimental_betas", "debug"}:
            normalized = raw_value.strip().lower()
            if normalized in {"1", "true", "yes"}:
                values[key] = True
            elif normalized in {"0", "false", "no"}:
                values[key] = False
            else:
                raise ConfigError(f"{key} must be a boolean")
        else:
            values[key] = raw_value


def _parse_model_overrides(raw: Any) -> dict[str, ModelOverride]:
    if not isinstance(raw, dict):
        raise ConfigError("model_overrides must be a table")
    result: dict[str, ModelOverride] = {}
    for model_id, model_values in raw.items():
        if not isinstance(model_id, str) or not model_id:
            raise ConfigError("model_overrides keys must be non-empty model IDs")
        if not isinstance(model_values, dict):
            raise ConfigError(f"model_overrides.{model_id} must be a table")
        unknown = sorted(set(model_values) - _OVERRIDE_KEYS)
        if unknown:
            raise ConfigError(f"unknown configuration key: model_overrides.{model_id}.{unknown[0]}")
        result[model_id] = _parse_model_override(model_id, model_values)
    return result


def _parse_model_override(model_id: str, values: Mapping[str, Any]) -> ModelOverride:
    display_name = values.get("display_name")
    if display_name is not None and (not isinstance(display_name, str) or not display_name):
        raise ConfigError(f"model_overrides.{model_id}.display_name must be a non-empty string")

    context_window = values.get("context_window")
    if context_window is not None:
        context_window = _positive_int(context_window, f"model_overrides.{model_id}.context_window")
    max_output_tokens = values.get("max_output_tokens")
    if max_output_tokens is not None:
        max_output_tokens = _positive_int(
            max_output_tokens, f"model_overrides.{model_id}.max_output_tokens"
        )

    reasoning = values.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, bool):
        raise ConfigError(f"model_overrides.{model_id}.reasoning must be a boolean or null")

    tool_calling = values.get("tool_calling")
    if tool_calling is not None and not isinstance(tool_calling, bool):
        raise ConfigError(f"model_overrides.{model_id}.tool_calling must be a boolean or null")

    modalities = values.get("input")
    parsed_modalities: tuple[str, ...] | None = None
    if modalities is not None:
        if not isinstance(modalities, list) or not all(
            isinstance(item, str) for item in modalities
        ):
            raise ConfigError(f"model_overrides.{model_id}.input must be an array of strings")
        invalid = sorted(set(modalities) - {"text", "image"})
        if invalid:
            raise ConfigError(
                f"model_overrides.{model_id}.input contains invalid modality: {invalid[0]}"
            )
        parsed_modalities = tuple(modalities)

    return ModelOverride(
        display_name=display_name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        input=parsed_modalities,
        tool_calling=tool_calling,
    )


def _optional_string(values: Mapping[str, Any], key: str) -> str | None:
    value = values[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = _optional_string(values, key)
    if value is None:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _integer(values: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    if value < minimum:
        comparator = "non-negative" if minimum == 0 else "positive"
        raise ConfigError(f"{key} must be {comparator}")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _boolean(values: Mapping[str, Any], key: str) -> bool:
    value = values[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    raise ConfigError(f"{key} must be a boolean")


def _number(
    values: Mapping[str, Any], key: str, *, minimum: float, exclusive: bool = False
) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    converted = float(value)
    invalid = converted <= minimum if exclusive else converted < minimum
    if invalid:
        raise ConfigError(f"{key} must be greater than {minimum:g}")
    return converted
