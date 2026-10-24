from __future__ import annotations

import aiohttp
import pytest

from claude_nim_proxy.config import UpstreamAuth
from claude_nim_proxy.translate.errors import AnthropicError
from claude_nim_proxy.upstream import NvidiaCatalogClient, NvidiaUpstreamClient


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://integrate.api.nvidia.com/v1",
            "https://integrate.api.nvidia.com/v1/chat/completions",
        ),
        (
            "https://integrate.api.nvidia.com/v1/",
            "https://integrate.api.nvidia.com/v1/chat/completions",
        ),
        ("https://example.test", "https://example.test/v1/chat/completions"),
        (
            "https://example.test/v1/chat/completions",
            "https://example.test/v1/chat/completions",
        ),
    ],
)
async def test_chat_completions_endpoint_is_normalized(base_url: str, expected: str) -> None:
    async with aiohttp.ClientSession() as session:
        client = NvidiaUpstreamClient(
            session,
            base_url=base_url,
            api_key=None,
            auth_mode=UpstreamAuth.NONE,
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
        )
        assert client.chat_completions_url == expected


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://integrate.api.nvidia.com/v1",
            "https://integrate.api.nvidia.com/v1/models",
        ),
        (
            "https://example.test",
            "https://example.test/v1/models",
        ),
        (
            "https://example.test/v1/models",
            "https://example.test/v1/models",
        ),
    ],
)
async def test_models_endpoint_is_normalized(base_url: str, expected: str) -> None:
    async with aiohttp.ClientSession() as session:
        client = NvidiaCatalogClient(
            session,
            base_url=base_url,
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
        )
        assert client.models_url == expected


async def test_missing_required_credential_fails_before_network_io() -> None:
    async with aiohttp.ClientSession() as session:
        client = NvidiaUpstreamClient(
            session,
            base_url="http://127.0.0.1:1/v1",
            api_key=None,
            auth_mode=UpstreamAuth.REQUIRED,
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
        )

        with pytest.raises(AnthropicError) as captured:
            await client.complete({"model": "nvidia/model", "messages": []})

    assert captured.value.status == 500
    assert captured.value.error_type == "api_error"
