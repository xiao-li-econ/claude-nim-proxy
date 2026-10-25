from __future__ import annotations

import json

import aiohttp
import pytest

from claude_nim_proxy.config import UpstreamAuth
from claude_nim_proxy.diagnostics import (
    configure_debug,
    debug_logger,
    safe_headers,
    set_debug_secrets,
)
from claude_nim_proxy.server import GatewayServer, GatewaySettings


@pytest.mark.asyncio
async def test_debug_logs_are_redacted_and_exclude_request_and_response_bodies(
    fake_upstream, capsys
) -> None:
    local_token = "local-debug-token"
    upstream_key = "upstream-debug-secret"
    prompt_secret = "prompt-that-must-not-be-logged"
    response_secret = "Hello from fake"
    configure_debug(True)
    set_debug_secrets((local_token, upstream_key))
    settings = GatewaySettings(
        local_token=local_token,
        default_model="nvidia/model",
        upstream_base_url=fake_upstream.base_url,
        api_key=upstream_key,
        upstream_auth=UpstreamAuth.REQUIRED,
    )
    server = GatewayServer(settings)
    await server.start()
    try:
        async with aiohttp.ClientSession() as session, session.post(
            f"{server.base_url}/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": prompt_secret}],
                "tools": [
                    {
                        "name": "Read",
                        "description": prompt_secret,
                        "input_schema": {"type": "object"},
                    }
                ],
            },
            headers={"Authorization": f"Bearer {local_token}"},
        ) as response:
            assert response.status == 200
            body = await response.text()
    finally:
        await server.stop()
        configure_debug(False)
        set_debug_secrets(())

    output = capsys.readouterr().err
    assert "gateway request" in output
    assert "model=claude-sonnet-4-5" in output
    assert "upstream response status=200" in output
    assert local_token not in output
    assert upstream_key not in output
    assert prompt_secret not in output
    assert response_secret not in output
    assert "Bearer" not in output
    assert json.loads(body)["content"][0]["text"] == response_secret


def test_header_redaction_covers_credential_headers() -> None:
    redacted = safe_headers(
        {
            "Authorization": "Bearer secret",
            "x-api-key": "secret",
            "retry-after": "3",
        }
    )

    assert redacted == {
        "Authorization": "[REDACTED]",
        "x-api-key": "[REDACTED]",
        "retry-after": "3",
    }


def test_debug_filter_redacts_registered_secrets(capsys) -> None:
    configure_debug(True)
    set_debug_secrets(("exact-secret-value",))
    try:
        debug_logger().debug("credential=exact-secret-value")
    finally:
        configure_debug(False)
        set_debug_secrets(())

    output = capsys.readouterr().err
    assert "exact-secret-value" not in output
    assert "[REDACTED]" in output
