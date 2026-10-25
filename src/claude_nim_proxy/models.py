"""Curated model metadata, active-model catalog storage, and capability resolution."""

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

from .config import AppConfig
from .translate.anthropic_to_openai import GATEWAY_MODEL_PREFIX

MODEL_CATALOG_VERSION = 2
UNKNOWN_CONTEXT_WINDOW = 32_768
UNKNOWN_MAX_OUTPUT_TOKENS = 4_096
_VALID_INPUT_MODALITIES = {"text", "image"}
_VALID_CATALOG_SOURCES = {"bundled", "pi", "manual"}
_VALID_METADATA_SOURCES = {"bundled", "pi", "manual", "unknown"}
_CATALOG_MODEL_KEYS = {
    "upstream_id",
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
    """Raised when bundled or active model catalog data is invalid."""


class ModelCacheError(ModelCatalogError):
    """Backward-compatible name for active model catalog errors."""


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
    source: tuple[str, ...] = ("nvidia",)
    metadata_source: str = "bundled"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible catalog representation."""

        return {
            "upstream_id": self.upstream_id,
            "display_name": self.display_name,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": self.reasoning,
            "input": list(self.input) if self.input is not None else None,
            "tool_calling": self.tool_calling,
            "source": list(self.source),
            "metadata_source": self.metadata_source,
        }


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Fully resolved model metadata used by the gateway and runner."""

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
        """Return the JSON-compatible resolved-model representation."""

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
class ModelCatalog:
    """Versioned active catalog of curated NVIDIA model IDs and capabilities."""

    source: str
    updated_at: str | None
    models: tuple[ModelMetadata, ...]
    version: int = MODEL_CATALOG_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible active-catalog representation."""

        return {
            "version": self.version,
            "source": self.source,
            "updated_at": self.updated_at,
            "models": [model.to_dict() for model in self.models],
        }


@dataclass(frozen=True, slots=True)
class CatalogDiff:
    """Changes between two active model ID lists."""

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


def load_bundled_model_metadata() -> dict[str, ModelMetadata]:
    """Load and validate the bundled initialization/reset model catalog."""

    resource = resources.files("claude_nim_proxy").joinpath("data/nvidia_models.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCatalogError("bundled NVIDIA model metadata is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("version") != MODEL_CATALOG_VERSION:
        raise ModelCatalogError("bundled NVIDIA model metadata has an unsupported version")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ModelCatalogError("bundled NVIDIA model metadata must contain a models array")

    result: dict[str, ModelMetadata] = {}
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ModelCatalogError(f"bundled model entry {index} must be an object")
        metadata = _parse_model_metadata(raw_model, index, default_source="bundled")
        if metadata.upstream_id in result:
            raise ModelCatalogError(
                f"bundled model metadata contains duplicate ID {metadata.upstream_id}"
            )
        result[metadata.upstream_id] = metadata
    return result


def load_bundled_model_catalog() -> ModelCatalog:
    """Return the bundled catalog used only for initialization and reset."""

    metadata = load_bundled_model_metadata()
    return ModelCatalog(
        source="bundled",
        updated_at=None,
        models=tuple(metadata[model_id] for model_id in sorted(metadata)),
    )


def default_model_catalog_path() -> Path:
    """Return the per-user active model catalog path."""

    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "claude-nim" / "models.json"


def default_model_cache_path() -> Path:
    """Backward-compatible alias for the active model catalog path."""

    return default_model_catalog_path()


def load_model_catalog(path: Path) -> ModelCatalog | None:
    """Load an active model catalog, returning ``None`` when it does not exist."""

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ModelCacheError(f"cannot read model catalog {path}: {exc.strerror}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCacheError(f"invalid JSON in model catalog {path}") from exc
    return _parse_model_catalog(payload, path)


def load_model_cache(path: Path) -> ModelCatalog | None:
    """Backward-compatible alias for :func:`load_model_catalog`."""

    return load_model_catalog(path)


def save_model_catalog(catalog: ModelCatalog, path: Path) -> None:
    """Atomically write a credential-free active model catalog."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(catalog.to_dict(), indent=2, ensure_ascii=False) + "\n"
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
    except OSError as exc:
        raise ModelCacheError(f"cannot write model catalog {path}: {exc.strerror}") from exc
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            os.chmod(temporary_path, 0o600)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise ModelCacheError(f"cannot write model catalog {path}: {exc.strerror}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def save_model_cache(catalog: ModelCatalog, path: Path) -> None:
    """Backward-compatible alias for :func:`save_model_catalog`."""

    save_model_catalog(catalog, path)


def ensure_model_catalog(path: Path) -> ModelCatalog:
    """Load the active catalog, initializing or archiving known legacy data as needed."""

    if not path.exists():
        catalog = load_bundled_model_catalog()
        save_model_catalog(catalog, path)
        return catalog
    try:
        loaded = load_model_catalog(path)
    except ModelCacheError:
        if not _is_legacy_nvidia_catalog(path):
            raise
        _archive_legacy_nvidia_catalog(path)
        catalog = load_bundled_model_catalog()
        save_model_catalog(catalog, path)
        return catalog
    if loaded is None:
        catalog = load_bundled_model_catalog()
        save_model_catalog(catalog, path)
        return catalog
    return loaded


def reset_model_catalog(path: Path) -> ModelCatalog:
    """Replace the active catalog with the bundled initialization/reset template."""

    catalog = load_bundled_model_catalog()
    save_model_catalog(catalog, path)
    return catalog


def resolve_model_catalog(catalog: ModelCatalog, config: AppConfig) -> tuple[ModelInfo, ...]:
    """Resolve every model in one active catalog against global fallbacks."""

    return tuple(_resolve_model(model, config, catalog.source) for model in catalog.models)


def resolve_model_metadata(
    upstream_id: str,
    config: AppConfig,
    catalog: ModelCatalog | None = None,
) -> ModelInfo:
    """Resolve capabilities for one raw NVIDIA model ID from the active catalog."""

    active = catalog if catalog is not None else ensure_model_catalog(default_model_catalog_path())
    normalized_id = normalize_upstream_model_id(upstream_id)
    metadata = next(
        (model for model in active.models if model.upstream_id == normalized_id),
        None,
    )
    if metadata is None:
        metadata = ModelMetadata(
            upstream_id=normalized_id,
            metadata_source="unknown",
        )
    return _resolve_model(metadata, config, active.source)


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
    metadata: ModelMetadata,
    config: AppConfig,
    catalog_source: str,
) -> ModelInfo:
    display_name = metadata.display_name or f"NVIDIA {metadata.upstream_id} (NIM)"
    context_window = metadata.context_window or config.context_window
    max_output_tokens = metadata.max_output_tokens or config.max_output_tokens
    modalities = metadata.input or ("text",)
    metadata_source = metadata.metadata_source
    if catalog_source == "manual" and metadata_source != "unknown":
        metadata_source = "manual"

    return ModelInfo(
        upstream_id=metadata.upstream_id,
        display_name=display_name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        reasoning=metadata.reasoning,
        input=modalities,
        tool_calling=metadata.tool_calling,
        source=metadata.source,
        metadata_source=metadata_source,
    )


def _parse_model_catalog(payload: Any, path: Path) -> ModelCatalog:
    if not isinstance(payload, dict):
        raise ModelCacheError(f"model catalog {path} must contain a JSON object")
    version = payload.get("version")
    if version != MODEL_CATALOG_VERSION:
        raise ModelCacheError(f"model catalog {path} has unsupported version {version!r}")
    source = payload.get("source")
    if source not in _VALID_CATALOG_SOURCES:
        raise ModelCacheError(f"model catalog {path} has invalid source {source!r}")
    updated_at = payload.get("updated_at")
    if updated_at is not None:
        if not isinstance(updated_at, str) or not updated_at:
            raise ModelCacheError(
                f"model catalog {path} requires updated_at to be a string or null"
            )
        try:
            datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ModelCacheError(f"model catalog {path} has an invalid updated_at") from exc
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ModelCacheError(f"model catalog {path} requires a models array")

    models: list[ModelMetadata] = []
    seen_ids: set[str] = set()
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ModelCacheError(f"model catalog {path} model {index} must be an object")
        unknown = sorted(set(raw_model) - _CATALOG_MODEL_KEYS)
        if unknown:
            raise ModelCacheError(
                f"model catalog {path} model {index} has unknown key {unknown[0]}"
            )
        metadata = _parse_model_metadata(
            raw_model,
            index,
            default_source=str(source),
            path=path,
        )
        if metadata.upstream_id in seen_ids:
            raise ModelCacheError(
                f"model catalog {path} contains duplicate model {metadata.upstream_id}"
            )
        seen_ids.add(metadata.upstream_id)
        models.append(metadata)

    return ModelCatalog(
        source=str(source),
        updated_at=updated_at,
        models=tuple(models),
    )


def _parse_model_metadata(
    raw_model: Mapping[str, Any],
    index: int,
    *,
    default_source: str,
    path: Path | None = None,
) -> ModelMetadata:
    location = f"model {index}"
    if path is not None:
        location = f"model catalog {path} model {index}"
    upstream_id = raw_model.get("upstream_id")
    if not isinstance(upstream_id, str) or not upstream_id.strip():
        raise ModelCatalogError(f"{location} requires an upstream_id")
    try:
        normalized_id = normalize_upstream_model_id(upstream_id)
    except ModelCatalogError as exc:
        raise ModelCatalogError(f"{location}: {exc}") from exc

    display_name = raw_model.get("display_name")
    if display_name is not None and (not isinstance(display_name, str) or not display_name):
        raise ModelCatalogError(f"{location} display_name must be a non-empty string")

    context_window = _optional_positive_int(
        raw_model.get("context_window"), f"{location} context_window"
    )
    max_output_tokens = _optional_positive_int(
        raw_model.get("max_output_tokens"), f"{location} max_output_tokens"
    )
    reasoning = _optional_boolean(raw_model.get("reasoning"), f"{location} reasoning")
    tool_calling = _optional_boolean(raw_model.get("tool_calling"), f"{location} tool_calling")

    raw_input = raw_model.get("input")
    modalities: tuple[str, ...] | None = None
    if raw_input is not None:
        if not isinstance(raw_input, list) or not all(isinstance(item, str) for item in raw_input):
            raise ModelCatalogError(f"{location} input must be an array of strings")
        invalid = sorted(set(raw_input) - _VALID_INPUT_MODALITIES)
        if invalid:
            raise ModelCatalogError(f"{location} has invalid input modality {invalid[0]}")
        modalities = tuple(raw_input)

    raw_source = raw_model.get("source", ["nvidia"])
    if not isinstance(raw_source, list) or not all(isinstance(item, str) for item in raw_source):
        raise ModelCatalogError(f"{location} source must be an array of strings")
    metadata_source = raw_model.get("metadata_source", default_source)
    if metadata_source not in _VALID_METADATA_SOURCES:
        raise ModelCatalogError(f"{location} has invalid metadata_source {metadata_source!r}")

    return ModelMetadata(
        upstream_id=normalized_id,
        display_name=display_name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        input=modalities,
        tool_calling=tool_calling,
        source=tuple(raw_source),
        metadata_source=str(metadata_source),
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


def _is_legacy_nvidia_catalog(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("version") == 1


def _archive_legacy_nvidia_catalog(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.nvidia-v1.bak")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.nvidia-v1.{counter}.bak")
        counter += 1
    try:
        os.replace(path, backup)
    except OSError as exc:
        raise ModelCacheError(
            f"cannot archive legacy NVIDIA model catalog {path}: {exc.strerror}"
        ) from exc
    return backup
