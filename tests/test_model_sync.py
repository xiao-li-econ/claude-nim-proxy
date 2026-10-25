from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from claude_nim_proxy.cli import _gateway_settings, main
from claude_nim_proxy.config import load_config
from claude_nim_proxy.models import (
    ModelCache,
    ModelCacheError,
    compare_catalog_ids,
    default_model_cache_path,
    gateway_alias,
    load_bundled_model_metadata,
    load_model_cache,
    merge_model_metadata,
    normalize_upstream_model_id,
    parse_nvidia_model_catalog,
    resolve_model_metadata,
    save_model_cache,
)
from claude_nim_proxy.server import GatewayServer

CURATED_AGENTIC_MODEL_IDS = (
    "deepseek-ai/deepseek-v4.1-flash",
    "minimaxai/minimax-m3.1-flash",
    "moonshotai/kimi-k2.6",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "openai/gpt-oss-20b",
    "stepfun-ai/step-5-preview",
    "z-ai/glm-5.3",
)


def test_gateway_alias_is_reversible() -> None:
    alias = gateway_alias("nvidia/nemotron-3-super-120b-a12b")

    assert alias == "anthropic-nim/nvidia/nemotron-3-super-120b-a12b"
    assert normalize_upstream_model_id(alias) == "nvidia/nemotron-3-super-120b-a12b"


def test_catalog_parser_keeps_authoritative_ids_and_removes_duplicates() -> None:
    payload = {
        "object": "list",
        "data": [
            {"id": "nvidia/model-b"},
            {"id": "nvidia/model-a"},
            {"id": "nvidia/model-b"},
        ],
    }

    assert parse_nvidia_model_catalog(payload) == ("nvidia/model-a", "nvidia/model-b")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": {}},
        {"data": [{"object": "model"}]},
        {"data": [{"id": ""}]},
    ],
)
def test_malformed_catalog_is_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_nvidia_model_catalog(payload)


def test_bundled_metadata_provides_curated_capabilities(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml", environ={})

    model = resolve_model_metadata("nvidia/nemotron-3-super-120b-a12b", config)

    assert model.display_name == "NVIDIA Nemotron 3 Super 120B A12B (NIM)"
    assert model.context_window == 1_048_576
    assert model.max_output_tokens == 32_768
    assert model.reasoning is True
    assert model.tool_calling is True
    assert model.input == ("text",)
    assert model.metadata_source == "bundled"


def test_unknown_model_uses_global_and_conservative_fallbacks(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
context_window = 12345
max_output_tokens = 6789
""",
        encoding="utf-8",
    )
    config = load_config(config_path, environ={})

    model = resolve_model_metadata("vendor/unknown", config)

    assert model.context_window == 12345
    assert model.max_output_tokens == 6789
    assert model.reasoning is None
    assert model.tool_calling is None
    assert model.input == ("text",)
    assert model.metadata_source == "unknown"


def test_user_override_has_highest_capability_precedence(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
context_window = 12345
max_output_tokens = 6789

[model_overrides."nvidia/nemotron-3-super-120b-a12b"]
display_name = "My NVIDIA model"
context_window = 200000
reasoning = false
input = ["text", "image"]
tool_calling = false
""",
        encoding="utf-8",
    )
    config = load_config(config_path, environ={})

    model = resolve_model_metadata("nvidia/nemotron-3-super-120b-a12b", config)

    assert model.display_name == "My NVIDIA model"
    assert model.context_window == 200_000
    assert model.max_output_tokens == 32_768
    assert model.reasoning is False
    assert model.tool_calling is False
    assert model.input == ("text", "image")
    assert model.metadata_source == "user"


def test_cache_round_trip_is_atomic_and_credential_free(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml", environ={})
    models = merge_model_metadata(["nvidia/model", "vendor/unknown"], config)
    cache = ModelCache(
        updated_at="2026-10-24T16:18:00Z",
        upstream_base_url="https://integrate.api.nvidia.com/v1",
        models=models,
    )
    path = tmp_path / "cache" / "models.json"

    save_model_cache(cache, path)

    loaded = load_model_cache(path)
    assert loaded == cache
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    raw = path.read_text(encoding="utf-8")
    assert "NVIDIA_API_KEY" not in raw
    assert "authorization" not in raw.lower()
    assert not list(path.parent.glob("*.tmp"))


def test_invalid_cache_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text('{"version": 99, "models": []}', encoding="utf-8")

    with pytest.raises(ModelCacheError, match="unsupported version"):
        load_model_cache(path)


def test_catalog_diff_reports_additions_and_removals() -> None:
    difference = compare_catalog_ids(
        ["nvidia/old", "nvidia/same"],
        ["nvidia/new", "nvidia/same"],
    )

    assert difference.added == ("nvidia/new",)
    assert difference.removed == ("nvidia/old",)
    assert difference.unchanged == ("nvidia/same",)


def test_default_cache_path_uses_xdg_config_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert default_model_cache_path() == tmp_path / "claude-nim" / "models.json"


def test_bundled_metadata_file_is_valid() -> None:
    bundled = load_bundled_model_metadata()

    assert "nvidia/nemotron-3-super-120b-a12b" in bundled
    assert bundled["nvidia/nemotron-3-super-120b-a12b"].tool_calling is True
    assert "nvidia/nemotron-nano-3-30b-a3b" in bundled
    assert "nvidia/nemotron-3-nano-30b-a3b" not in bundled


def test_bundled_metadata_covers_curated_agentic_models() -> None:
    bundled = load_bundled_model_metadata()

    assert set(CURATED_AGENTIC_MODEL_IDS) <= set(bundled)
    for model_id in CURATED_AGENTIC_MODEL_IDS:
        metadata = bundled[model_id]
        assert metadata.tool_calling is True
        assert metadata.reasoning is True
        assert metadata.context_window is not None
        assert metadata.context_window >= 128_000


def test_bundled_metadata_marks_image_capable_models() -> None:
    bundled = load_bundled_model_metadata()

    assert bundled["nvidia/nemotron-3.5-super-vl-120b-a12b"].input == ("text", "image")
    assert bundled["moonshotai/kimi-k2.6"].input == ("text", "image")
    assert bundled["nvidia/nemotron-3-super-120b-a12b"].input == ("text",)


def test_unresolved_bundled_output_limit_falls_back_to_global_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("max_output_tokens = 6789\n", encoding="utf-8")
    config = load_config(config_path, environ={})

    model = resolve_model_metadata("z-ai/glm-5.3", config)

    assert model.max_output_tokens == 6789
    assert model.context_window == 1_048_576
    assert model.metadata_source == "bundled"


@pytest_asyncio.fixture
async def catalog_server() -> tuple[str, list[dict[str, Any]], list[str]]:
    app = web.Application()
    requests: list[dict[str, Any]] = []
    model_ids = ["nvidia/nemotron-3-super-120b-a12b", "vendor/unknown"]

    async def handle_models(request: web.Request) -> web.StreamResponse:
        requests.append({"headers": dict(request.headers), "path": request.path})
        return web.json_response(
            {"object": "list", "data": [{"id": model_id} for model_id in model_ids]}
        )

    app.router.add_get("/v1/models", handle_models)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    addresses = runner.addresses
    assert addresses
    port = int(addresses[0][1])
    try:
        yield f"http://127.0.0.1:{port}/v1", requests, model_ids
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gateway_discovery_is_limited_to_curated_models_plus_selected_model(
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
    cache_models = merge_model_metadata(
        (
            "nvidia/nemotron-3-super-120b-a12b",
            "vendor/selected-uncurated",
            "vendor/unknown",
        ),
        config,
    )
    save_model_cache(
        ModelCache(
            updated_at="2026-10-24T16:18:00Z",
            upstream_base_url=config.upstream_base_url,
            models=cache_models,
        ),
        default_model_cache_path(),
    )

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
        "anthropic-nim/nvidia/nemotron-3-super-120b-a12b",
    ]
    assert "anthropic-nim/vendor/unknown" not in {
        entry["id"] for entry in payload["data"]
    }


@pytest.mark.asyncio
async def test_models_update_is_public_atomic_and_reports_changes(
    tmp_path: Path,
    monkeypatch,
    capsys,
    catalog_server: tuple[str, list[dict[str, Any]], list[str]],
) -> None:
    base_url, requests, catalog_ids = catalog_server
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-must-not-be-persisted")
    monkeypatch.setenv("NVIDIA_BASE_URL", "http://127.0.0.1:1/v1")

    status = await asyncio.to_thread(
        main,
        [
            "models",
            "update",
            "--upstream-base-url",
            base_url,
            "--json",
        ],
    )
    output = capsys.readouterr().out

    assert status == 0
    report = json.loads(output)
    assert report["model_count"] == 2
    assert report["added"] == [
        "nvidia/nemotron-3-super-120b-a12b",
        "vendor/unknown",
    ]
    assert report["missing_metadata"] == ["vendor/unknown"]
    assert requests[0]["path"] == "/v1/models"
    assert "Authorization" not in requests[0]["headers"]

    cache_path = tmp_path / "claude-nim" / "models.json"
    cache_text = cache_path.read_text(encoding="utf-8")
    assert "nvapi-must-not-be-persisted" not in cache_text
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600
    cache = load_model_cache(cache_path)
    assert cache is not None
    assert [model.gateway_id for model in cache.models] == [
        "anthropic-nim/nvidia/nemotron-3-super-120b-a12b",
        "anthropic-nim/vendor/unknown",
    ]

    catalog_ids[:] = ["vendor/unknown", "vendor/new"]
    second_status = await asyncio.to_thread(
        main,
        [
            "models",
            "update",
            "--upstream-base-url",
            base_url,
            "--json",
        ],
    )
    second_output = capsys.readouterr().out
    second_report = json.loads(second_output)

    assert second_status == 0
    assert second_report["added"] == ["vendor/new"]
    assert second_report["removed"] == ["nvidia/nemotron-3-super-120b-a12b"]
    assert second_report["unchanged"] == ["vendor/unknown"]


@pytest.mark.asyncio
async def test_models_list_shows_gateway_aliases(
    tmp_path: Path,
    monkeypatch,
    capsys,
    catalog_server: tuple[str, list[dict[str, Any]], list[str]],
) -> None:
    base_url, _, _ = catalog_server
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    update_status = await asyncio.to_thread(
        main,
        ["models", "update", "--upstream-base-url", base_url],
    )
    capsys.readouterr()

    list_status = await asyncio.to_thread(main, ["models", "list", "--json"])
    output = capsys.readouterr().out

    assert update_status == 0
    assert list_status == 0
    payload = json.loads(output)
    assert payload["models"][0]["gateway_id"] == ("anthropic-nim/nvidia/nemotron-3-super-120b-a12b")
    assert payload["models"][0]["metadata_source"] == "bundled"
    assert payload["models"][1]["metadata_source"] == "unknown"


def test_models_list_without_cache_has_actionable_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    status = main(["models", "list"])

    assert status == 2
    assert "models update" in capsys.readouterr().err


def test_environment_base_url_is_used_by_models_update(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("NVIDIA_BASE_URL", "http://127.0.0.1:1/v1")

    status = main(["models", "update"])

    assert status == 2
    assert "Model catalog refresh failed" in capsys.readouterr().err


def test_cache_permissions_are_not_world_readable(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml", environ={})
    cache = ModelCache(
        updated_at="2026-10-24T16:18:00Z",
        upstream_base_url="https://integrate.api.nvidia.com/v1",
        models=merge_model_metadata(["vendor/model"], config),
    )
    path = tmp_path / "models.json"

    save_model_cache(cache, path)

    assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
    assert os.path.isfile(path)
