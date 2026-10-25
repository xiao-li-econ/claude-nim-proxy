from __future__ import annotations

import json
import stat
from pathlib import Path

import aiohttp
import pytest

from claude_nim_proxy.cli import _gateway_settings, main
from claude_nim_proxy.config import load_config
from claude_nim_proxy.models import (
    ModelCacheError,
    ModelCatalog,
    ModelMetadata,
    compare_catalog_ids,
    default_model_cache_path,
    ensure_model_catalog,
    load_bundled_model_metadata,
    load_model_catalog,
    normalize_upstream_model_id,
    reset_model_catalog,
    resolve_model_catalog,
    resolve_model_metadata,
    save_model_catalog,
)
from claude_nim_proxy.pi_models import PiModel, PiModelListError
from claude_nim_proxy.server import GatewayServer

PI_VISIBLE_NVIDIA_MODEL_IDS = (
    "deepseek-ai/deepseek-v4.1-flash",
    "google/diffusiongemma-26b-a4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "meta/muse-glimmer-30b",
    "minimaxai/minimax-m3.1-flash",
    "mistralai/mistral-7b-instruct-v0.3",
    "moonshotai/kimi-k2.6",
    "nvidia/cosmos-reason2-8b",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "openai/gpt-oss-20b",
    "poolside/laguna-xs-2.1",
    "stepfun-ai/step-5-preview",
    "z-ai/glm-5.3",
    "z-ai/glm-5.3-flash",
)


def test_gateway_alias_is_reversible() -> None:
    alias = "anthropic-nim/nvidia/nemotron-3.5-lightning-30b-a3b"

    assert alias == "anthropic-nim/nvidia/nemotron-3.5-lightning-30b-a3b"
    assert normalize_upstream_model_id(alias) == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_bundled_metadata_exactly_matches_current_pi_nvidia_list() -> None:
    bundled = load_bundled_model_metadata()

    assert set(bundled) == set(PI_VISIBLE_NVIDIA_MODEL_IDS)
    assert bundled["nvidia/nemotron-3.5-lightning-30b-a3b"].tool_calling is None
    assert bundled["google/diffusiongemma-26b-a4b-it"].context_window == 250_000
    assert bundled["google/diffusiongemma-26b-a4b-it"].input == ("text", "image")
    assert "nvidia/nemotron-3-super-120b-a12b" not in bundled
    assert "nvidia/nemotron-nano-3-30b-a3b" not in bundled


def test_catalog_diff_reports_additions_and_removals() -> None:
    difference = compare_catalog_ids(
        ["nvidia/old", "nvidia/same"],
        ["nvidia/new", "nvidia/same"],
    )

    assert difference.added == ("nvidia/new",)
    assert difference.removed == ("nvidia/old",)
    assert difference.unchanged == ("nvidia/same",)


def test_default_catalog_path_uses_xdg_config_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert default_model_cache_path() == tmp_path / "claude-nim" / "models.json"


def test_active_catalog_is_initialized_from_bundled_template(tmp_path: Path) -> None:
    path = tmp_path / "models.json"

    catalog = ensure_model_catalog(path)

    assert catalog.source == "bundled"
    assert catalog.updated_at is None
    assert set(model.upstream_id for model in catalog.models) == set(
        PI_VISIBLE_NVIDIA_MODEL_IDS
    )
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_legacy_nvidia_cache_is_archived_before_bundled_initialization(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text('{"version": 1, "models": []}\n', encoding="utf-8")

    catalog = ensure_model_catalog(path)

    assert catalog.source == "bundled"
    assert path.exists()
    assert (tmp_path / "models.json.nvidia-v1.bak").exists()
    assert json.loads((tmp_path / "models.json.nvidia-v1.bak").read_text())["version"] == 1


def test_invalid_active_catalog_is_not_silently_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(ModelCacheError, match="invalid JSON"):
        ensure_model_catalog(path)

    assert path.read_text(encoding="utf-8") == "not json\n"


def test_active_catalog_round_trip_is_atomic_and_credential_free(tmp_path: Path) -> None:
    path = tmp_path / "catalog" / "models.json"
    catalog = ModelCatalog(
        source="manual",
        updated_at="2026-10-25T12:00:00Z",
        models=(
            ModelMetadata(
                upstream_id="nvidia/model",
                display_name="Manual model",
                context_window=32_768,
                max_output_tokens=4_096,
                reasoning=None,
                input=("text",),
                tool_calling=None,
                metadata_source="manual",
            ),
        ),
    )

    save_model_catalog(catalog, path)

    loaded = load_model_catalog(path)
    assert loaded == catalog
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    raw = path.read_text(encoding="utf-8")
    assert "NVIDIA_API_KEY" not in raw
    assert "authorization" not in raw.lower()
    assert not list(path.parent.glob("*.tmp"))


def test_model_capabilities_resolve_from_active_catalog_with_global_fallback(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
context_window = 12345
max_output_tokens = 6789
""",
        encoding="utf-8",
    )
    config = load_config(config_path, environ={})
    catalog = ModelCatalog(
        source="manual",
        updated_at=None,
        models=(
            ModelMetadata(
                upstream_id="nvidia/model",
                display_name=None,
                context_window=None,
                max_output_tokens=None,
                reasoning=None,
                input=None,
                tool_calling=None,
                metadata_source="manual",
            ),
        ),
    )

    model = resolve_model_metadata("nvidia/model", config, catalog)

    assert model.display_name == "NVIDIA nvidia/model (NIM)"
    assert model.context_window == 12_345
    assert model.max_output_tokens == 6_789
    assert model.reasoning is None
    assert model.tool_calling is None
    assert model.input == ("text",)
    assert model.metadata_source == "manual"


def test_selected_model_outside_active_catalog_uses_conservative_fallback(
    tmp_path: Path,
) -> None:
    config = load_config(tmp_path / "missing.toml", environ={})
    catalog = ModelCatalog(source="manual", updated_at=None, models=())

    model = resolve_model_metadata("vendor/unknown", config, catalog)

    assert model.context_window == 32_768
    assert model.max_output_tokens == 4_096
    assert model.metadata_source == "unknown"


@pytest.mark.asyncio
async def test_gateway_discovery_uses_active_catalog_plus_selected_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'default_model = "vendor/selected-uncurated"\n',
        encoding="utf-8",
    )
    config = load_config(config_path, environ={})
    catalog = ModelCatalog(
        source="manual",
        updated_at=None,
        models=(
            ModelMetadata(
                upstream_id="nvidia/active-model",
                display_name="Active model",
                context_window=32_768,
                max_output_tokens=4_096,
                reasoning=False,
                input=("text",),
                tool_calling=True,
                metadata_source="manual",
            ),
        ),
    )
    save_model_catalog(catalog, default_model_cache_path())

    settings = _gateway_settings(
        config,
        api_key=None,
        allow_non_loopback=False,
        local_token="local-token",
    )
    server = GatewayServer(settings)
    await server.start()
    try:
        async with aiohttp.ClientSession() as session, session.get(
            f"{server.base_url}/v1/models",
            headers={"Authorization": "Bearer local-token"},
        ) as response:
            assert response.status == 200
            payload = await response.json()
    finally:
        await server.stop()

    assert [entry["id"] for entry in payload["data"]] == [
        "anthropic-nim/vendor/selected-uncurated",
        "anthropic-nim/nvidia/active-model",
    ]


def test_models_list_uses_bundled_template_without_prior_cache(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    status = main(["models", "list", "--json"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "bundled"
    assert payload["updated_at"] is None
    assert len(payload["models"]) == len(PI_VISIBLE_NVIDIA_MODEL_IDS)


def test_models_update_replaces_active_catalog_with_pi_list(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = default_model_cache_path()
    save_model_catalog(
        ModelCatalog(
            source="manual",
            updated_at=None,
            models=(
                ModelMetadata(
                    upstream_id="vendor/old-model",
                    display_name="Old model",
                    context_window=32_768,
                    max_output_tokens=4_096,
                    reasoning=False,
                    input=("text",),
                    tool_calling=True,
                    metadata_source="manual",
                ),
            ),
        ),
        path,
    )
    pi_models = (
        PiModel(
            upstream_id="nvidia/new-model",
            display_name="new-model (NIM)",
            context_window=131_072,
            max_output_tokens=16_384,
            reasoning=True,
            input=("text", "image"),
        ),
    )
    monkeypatch.setattr("claude_nim_proxy.cli.run_pi_model_list", lambda: pi_models)

    status = main(["models", "update", "--json"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "pi"
    assert payload["added"] == ["nvidia/new-model"]
    assert payload["removed"] == ["vendor/old-model"]
    active = load_model_catalog(path)
    assert active is not None
    assert [model.upstream_id for model in active.models] == ["nvidia/new-model"]
    assert active.models[0].tool_calling is None
    assert active.models[0].metadata_source == "pi"


def test_models_update_preserves_active_catalog_when_pi_is_unavailable(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = default_model_cache_path()
    ensure_model_catalog(path)
    before = path.read_bytes()

    def fail() -> tuple[PiModel, ...]:
        raise PiModelListError("Pi command not found")

    monkeypatch.setattr("claude_nim_proxy.cli.run_pi_model_list", fail)

    status = main(["models", "update"])

    assert status == 2
    assert path.read_bytes() == before
    error = capsys.readouterr().err
    assert "Pi model refresh failed" in error
    assert "~/.config/claude-nim/models.json" in error


def test_models_reset_restores_bundled_template(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = default_model_cache_path()
    save_model_catalog(
        ModelCatalog(source="pi", updated_at=None, models=()),
        path,
    )

    status = main(["models", "reset", "--json"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "bundled"
    active = load_model_catalog(path)
    assert active is not None
    assert len(active.models) == len(PI_VISIBLE_NVIDIA_MODEL_IDS)


def test_reset_model_catalog_restores_bundled_models(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    save_model_catalog(ModelCatalog(source="pi", updated_at=None, models=()), path)

    catalog = reset_model_catalog(path)

    assert catalog.source == "bundled"
    assert set(model.upstream_id for model in catalog.models) == set(
        PI_VISIBLE_NVIDIA_MODEL_IDS
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["models", "--help"],
        ["models", "list", "--help"],
        ["models", "update", "--help"],
        ["models", "reset", "--help"],
    ],
)
def test_model_help_uses_portable_paths(
    arguments: list[str],
    capsys,
) -> None:
    with pytest.raises(SystemExit) as captured:
        main(arguments)

    assert captured.value.code == 0
    output = capsys.readouterr().out
    assert "~/.config/claude-nim/models.json" in output
    assert "src/claude_nim_proxy/data/nvidia_models.json" in output
    assert "manually edit" in output
    assert "/home/" not in output


def test_resolve_model_catalog_uses_catalog_entries(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml", environ={})
    catalog = ModelCatalog(
        source="manual",
        updated_at=None,
        models=(
            ModelMetadata(
                upstream_id="nvidia/model",
                display_name="Model",
                context_window=64_000,
                max_output_tokens=8_000,
                reasoning=True,
                input=("text",),
                tool_calling=True,
                metadata_source="manual",
            ),
        ),
    )

    models = resolve_model_catalog(catalog, config)

    assert models[0].gateway_id == "anthropic-nim/nvidia/model"
    assert models[0].context_window == 64_000
    assert models[0].metadata_source == "manual"
