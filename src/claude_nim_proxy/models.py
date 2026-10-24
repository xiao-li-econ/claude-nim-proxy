"""NVIDIA model catalog parsing, capability metadata, and cache management."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

from .config import AppConfig, ModelOverride
from .translate.anthropic_to_openai import GATEWAY_MODEL_PREFIX

MODEL_CACHE_VERSION = 1
UNKNOWN_CONTEXT_WINDOW = 32_768
UNKNOWN_MAX_OUTPUT_TOKENS = 4_096
_VALID_INPUT_MODALITIES = {"text", "image"}
_VALID_METADATA_SOURCES = {"bundled", "user", "unknown"}
_CACHE_MODEL_KEYS = {
    "upstream_id",
    "gateway_id",
    "display_name",
    "context_window",
    "max_output_tokens",
    "reasoning",
    "input",
    "tool_calling",
    "source",
    "metadata_source",
}


class ModelCatalogError(ValueError):
    """Raised when NVIDIA model catalog data is invalid."""


class ModelCacheError(ModelCatalogError):
    """Raised when the on-disk model cache is invalid."""


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Optional capability metadata for one raw NVIDIA model ID."""

    upstream_id: str
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    reasoning: bool | None = None
    input: tuple[str, ...] | None = None
    tool_calling: bool | None = None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Fully resolved model metadata stored in the model cache."""

    upstream_id: str
    display_name: str
    context_window: int
    max_output_tokens: int
    reasoning: bool | None
    input: tuple[str, ...]
    tool_calling: bool | None
    source: tuple[str, ...]
    metadata_source: str

    @property
    def gateway_id(self) -> str:
        """Return the reversible Claude Code discovery alias."""

        return f"{GATEWAY_MODEL_PREFIX}{self.upstream_id}"

    @property
    def supports_images(self) -> bool:
        """Return whether this model accepts image input."""

        return "image" in self.input

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible cache representation."""

        return {
            "upstream_id": self.upstream_id,
            "gateway_id": self.gateway_id,
            "display_name": self.display_name,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": self.reasoning,
            "input": list(self.input),
            "tool_calling": self.tool_calling,
            "source": list(self.source),
            "metadata_source": self.metadata_source,
        }


@dataclass(frozen=True, slots=True)
class ModelCache:
    """Versioned, credential-free cache of the NVIDIA model catalog."""

    updated_at: str
    upstream_base_url: str
    models: tuple[ModelInfo, ...]
    version: int = MODEL_CACHE_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible cache representation."""

        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "upstream_base_url": self.upstream_base_url,
            "models": [model.to_dict() for model in self.models],
        }


@dataclass(frozen=True, slots=True)
class CatalogDiff:
    """Changes between two authoritative NVIDIA model ID lists."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: tuple[str, ...]


def normalize_upstream_model_id(model_id: str) -> str:
    """Return a raw NVIDIA model ID from either a raw ID or gateway alias."""

    normalized = model_id.strip()
    if normalized.startswith(GATEWAY_MODEL_PREFIX):
        normalized = normalized[len(GATEWAY_MODEL_PREFIX) :]
    if not normalized:
        raise ModelCatalogError("model ID must not be empty")
    if any(character.isspace() for character in normalized):
        raise ModelCatalogError("model ID must not contain whitespace")
    return normalized


def gateway_alias(model_id: str) -> str:
    """Return the reversible ``anthropic-nim/`` discovery alias."""

    return f"{GATEWAY_MODEL_PREFIX}{normalize_upstream_model_id(model_id)}"


def parse_nvidia_model_catalog(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Parse the authoritative model IDs from an OpenAI-style catalog response."""

    data = payload.get("data")
    if not isinstance(data, list):
        raise ModelCatalogError("NVIDIA model catalog response must contain a data array")
    model_ids: set[str] = set()
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ModelCatalogError(f"NVIDIA model catalog entry {index} must be an object")
        raw_id = entry.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ModelCatalogError(
                f"NVIDIA model catalog entry {index} must contain a non-empty string id"
            )
        try:
            model_ids.add(normalize_upstream_model_id(raw_id))
        except ModelCatalogError as exc:
            raise ModelCatalogError(f"NVIDIA model catalog entry {index}: {exc}") from exc
    return tuple(sorted(model_ids))


def load_bundled_model_metadata() -> dict[str, ModelMetadata]:
    """Load and validate the bundled curated NVIDIA capability catalog."""

    resource = resources.files("claude_nim_proxy").joinpath("data/nvidia_models.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCatalogError("bundled NVIDIA model metadata is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ModelCatalogError("bundled NVIDIA model metadata has an unsupported version")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ModelCatalogError("bundled NVIDIA model metadata must contain a models array")

    result: dict[str, ModelMetadata] = {}
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ModelCatalogError(f"bundled model entry {index} must be an object")
        upstream_id = raw_model.get("upstream_id")
        if not isinstance(upstream_id, str) or not upstream_id.strip():
            raise ModelCatalogError(f"bundled model entry {index} requires an upstream_id")
        try:
            normalized_id = normalize_upstream_model_id(upstream_id)
        except ModelCatalogError as exc:
            raise ModelCatalogError(f"bundled model entry {index}: {exc}") from exc
        if normalized_id in result:
            raise ModelCatalogError(f"bundled model metadata contains duplicate ID {normalized_id}")
        result[normalized_id] = _parse_optional_metadata(normalized_id, raw_model)
    return result


def merge_model_metadata(
    upstream_ids: Iterable[str],
    config: AppConfig,
) -> tuple[ModelInfo, ...]:
    """Merge NVIDIA IDs with user, bundled, global, and unknown metadata."""

    bundled = load_bundled_model_metadata()
    normalized_ids = sorted({normalize_upstream_model_id(model_id) for model_id in upstream_ids})
    return tuple(_resolve_model(model_id, config, bundled) for model_id in normalized_ids)


def resolve_model_metadata(upstream_id: str, config: AppConfig) -> ModelInfo:
    """Resolve capabilities for one raw NVIDIA model ID."""

    bundled = load_bundled_model_metadata()
    return _resolve_model(normalize_upstream_model_id(upstream_id), config, bundled)


def default_model_cache_path() -> Path:
    """Return the per-user model cache path."""

    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "claude-nim" / "models.json"


def load_model_cache(path: Path) -> ModelCache | None:
    """Load a model cache, returning ``None`` when the cache does not exist."""

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ModelCacheError(f"cannot read model cache {path}: {exc.strerror}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCacheError(f"invalid JSON in model cache {path}") from exc
    return _parse_model_cache(payload, path)


def save_model_cache(cache: ModelCache, path: Path) -> None:
    """Atomically write a credential-free model cache."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(cache.to_dict(), indent=2, ensure_ascii=False) + "\n"
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
    except OSError as exc:
        raise ModelCacheError(f"cannot write model cache {path}: {exc.strerror}") from exc
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            os.chmod(temporary_path, 0o600)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise ModelCacheError(f"cannot write model cache {path}: {exc.strerror}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def compare_catalog_ids(
    previous_ids: Iterable[str],
    current_ids: Iterable[str],
) -> CatalogDiff:
    """Compare two model ID lists for update reporting."""

    previous = set(previous_ids)
    current = set(current_ids)
    return CatalogDiff(
        added=tuple(sorted(current - previous)),
        removed=tuple(sorted(previous - current)),
        unchanged=tuple(sorted(current & previous)),
    )


def _resolve_model(
    upstream_id: str,
    config: AppConfig,
    bundled: Mapping[str, ModelMetadata],
) -> ModelInfo:
    override = config.model_overrides.get(upstream_id)
    curated = bundled.get(upstream_id)

    display_name = _first_string(
        override.display_name if override is not None else None,
        curated.display_name if curated is not None else None,
        f"NVIDIA {upstream_id} (NIM)",
    )
    context_window = _first_integer(
        override.context_window if override is not None else None,
        curated.context_window if curated is not None else None,
        config.context_window,
    )
    max_output_tokens = _first_integer(
        override.max_output_tokens if override is not None else None,
        curated.max_output_tokens if curated is not None else None,
        config.max_output_tokens,
    )
    reasoning = _first_optional(
        override.reasoning if override is not None else None,
        curated.reasoning if curated is not None else None,
        None,
    )
    modalities = _first_modalities(
        override.input if override is not None else None,
        curated.input if curated is not None else None,
        ("text",),
    )
    tool_calling = _first_optional(
        override.tool_calling if override is not None else None,
        curated.tool_calling if curated is not None else None,
        None,
    )

    if _has_override_values(override):
        metadata_source = "user"
    elif curated is not None:
        metadata_source = "bundled"
    else:
        metadata_source = "unknown"

    return ModelInfo(
        upstream_id=upstream_id,
        display_name=display_name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        input=modalities,
        tool_calling=tool_calling,
        source=("nvidia",),
        metadata_source=metadata_source,
    )


def _first_string(*values: str | None) -> str:
    for value in values:
        if value is not None:
            return value
    raise RuntimeError("all string fallback values were None")


def _first_integer(*values: int | None) -> int:
    for value in values:
        if value is not None:
            return value
    raise RuntimeError("all integer fallback values were None")


def _first_modalities(*values: tuple[str, ...] | None) -> tuple[str, ...]:
    for value in values:
        if value is not None:
            return value
    raise RuntimeError("all modality fallback values were None")


def _first_optional(*values: bool | None) -> bool | None:
    for value in values:
        if value is not None:
            return value
    return None


def _has_override_values(override: ModelOverride | None) -> bool:
    if override is None:
        return False
    return any(
        value is not None
        for value in (
            override.display_name,
            override.context_window,
            override.max_output_tokens,
            override.reasoning,
            override.input,
            override.tool_calling,
        )
    )


def _parse_optional_metadata(upstream_id: str, raw: Mapping[str, Any]) -> ModelMetadata:
    raw_source = raw.get("source")
    if not isinstance(raw_source, list) or not all(isinstance(item, str) for item in raw_source):
        raise ModelCatalogError(f"model {upstream_id} source must be an array of strings")
    if raw.get("metadata_source") != "bundled":
        raise ModelCatalogError(f"model {upstream_id} metadata_source must be bundled")

    display_name = raw.get("display_name")
    if display_name is not None and (not isinstance(display_name, str) or not display_name):
        raise ModelCatalogError(f"model {upstream_id} display_name must be a non-empty string")

    context_window = _optional_positive_int(
        raw.get("context_window"), f"model {upstream_id} context_window"
    )
    max_output_tokens = _optional_positive_int(
        raw.get("max_output_tokens"), f"model {upstream_id} max_output_tokens"
    )
    reasoning = _optional_boolean(raw.get("reasoning"), f"model {upstream_id} reasoning")
    tool_calling = _optional_boolean(raw.get("tool_calling"), f"model {upstream_id} tool_calling")

    raw_input = raw.get("input")
    modalities: tuple[str, ...] | None = None
    if raw_input is not None:
        if not isinstance(raw_input, list) or not all(isinstance(item, str) for item in raw_input):
            raise ModelCatalogError(f"model {upstream_id} input must be an array of strings")
        invalid = sorted(set(raw_input) - _VALID_INPUT_MODALITIES)
        if invalid:
            raise ModelCatalogError(f"model {upstream_id} has invalid input modality {invalid[0]}")
        modalities = tuple(raw_input)

    return ModelMetadata(
        upstream_id=upstream_id,
        display_name=display_name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        input=modalities,
        tool_calling=tool_calling,
    )


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelCatalogError(f"{name} must be a positive integer")
    return value


def _optional_boolean(value: Any, name: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ModelCatalogError(f"{name} must be a boolean or null")


def _parse_model_cache(payload: Any, path: Path) -> ModelCache:
    if not isinstance(payload, dict):
        raise ModelCacheError(f"model cache {path} must contain a JSON object")
    version = payload.get("version")
    if version != MODEL_CACHE_VERSION:
        raise ModelCacheError(f"model cache {path} has unsupported version {version!r}")
    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        raise ModelCacheError(f"model cache {path} requires updated_at")
    try:
        datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ModelCacheError(f"model cache {path} has an invalid updated_at") from exc
    upstream_base_url = payload.get("upstream_base_url")
    if not isinstance(upstream_base_url, str) or not upstream_base_url:
        raise ModelCacheError(f"model cache {path} requires upstream_base_url")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ModelCacheError(f"model cache {path} requires a models array")

    models: list[ModelInfo] = []
    seen_ids: set[str] = set()
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ModelCacheError(f"model cache {path} model {index} must be an object")
        unknown = sorted(set(raw_model) - _CACHE_MODEL_KEYS)
        if unknown:
            raise ModelCacheError(f"model cache {path} model {index} has unknown key {unknown[0]}")
        models.append(_parse_cached_model(raw_model, index, path))
        upstream_id = models[-1].upstream_id
        if upstream_id in seen_ids:
            raise ModelCacheError(f"model cache {path} contains duplicate model {upstream_id}")
        seen_ids.add(upstream_id)

    return ModelCache(
        updated_at=updated_at,
        upstream_base_url=upstream_base_url,
        models=tuple(models),
    )


def _parse_cached_model(raw: Mapping[str, Any], index: int, path: Path) -> ModelInfo:
    upstream_id = raw.get("upstream_id")
    if not isinstance(upstream_id, str) or not upstream_id:
        raise ModelCacheError(f"model cache {path} model {index} requires upstream_id")
    try:
        normalized_id = normalize_upstream_model_id(upstream_id)
    except ModelCatalogError as exc:
        raise ModelCacheError(f"model cache {path} model {index}: {exc}") from exc
    gateway_id = raw.get("gateway_id")
    if gateway_id != f"{GATEWAY_MODEL_PREFIX}{normalized_id}":
        raise ModelCacheError(f"model cache {path} model {index} has an invalid gateway_id")

    display_name = raw.get("display_name")
    if not isinstance(display_name, str) or not display_name:
        raise ModelCacheError(f"model cache {path} model {index} requires display_name")
    context_window = _cached_positive_int(raw.get("context_window"), index, "context_window", path)
    max_output_tokens = _cached_positive_int(
        raw.get("max_output_tokens"), index, "max_output_tokens", path
    )
    reasoning = _optional_boolean(raw.get("reasoning"), f"cached model {index} reasoning")
    tool_calling = _optional_boolean(raw.get("tool_calling"), f"cached model {index} tool_calling")
    raw_input = raw.get("input")
    if not isinstance(raw_input, list) or not all(isinstance(item, str) for item in raw_input):
        raise ModelCacheError(f"model cache {path} model {index} input must be an array of strings")
    invalid = sorted(set(raw_input) - _VALID_INPUT_MODALITIES)
    if invalid:
        raise ModelCacheError(
            f"model cache {path} model {index} has invalid input modality {invalid[0]}"
        )
    raw_source = raw.get("source")
    if not isinstance(raw_source, list) or not all(isinstance(item, str) for item in raw_source):
        raise ModelCacheError(
            f"model cache {path} model {index} source must be an array of strings"
        )
    metadata_source = raw.get("metadata_source")
    if metadata_source not in _VALID_METADATA_SOURCES:
        raise ModelCacheError(f"model cache {path} model {index} has invalid metadata_source")

    return ModelInfo(
        upstream_id=normalized_id,
        display_name=display_name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        input=tuple(raw_input),
        tool_calling=tool_calling,
        source=tuple(raw_source),
        metadata_source=str(metadata_source),
    )


def _cached_positive_int(value: Any, index: int, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelCacheError(f"model cache {path} model {index} {name} must be a positive integer")
    return value
