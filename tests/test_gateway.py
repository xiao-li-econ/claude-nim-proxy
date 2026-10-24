from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from claude_nim_proxy.config import UpstreamAuth
from claude_nim_proxy.server import GatewayServer, GatewaySettings

LOCAL_TOKEN = "local-test-token"
UPSTREAM_KEY = "upstream-test-secret"


@asynccontextmanager
async def running_gateway(
    upstream_base_url: str,
    **overrides: Any,
) -> AsyncIterator[GatewayServer]:
    settings_values: dict[str, Any] = {
        "local_token": LOCAL_TOKEN,
        "default_model": "nvidia/model",
        "upstream_base_url": upstream_base_url,
        "upstream_auth": UpstreamAuth.NONE,
    }
    settings_values.update(overrides)
    server = GatewayServer(GatewaySettings(**settings_values))
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


def anthropic_request(*, stream: bool = False) -> dict[str, Any]:
    return {
        "model": "anthropic-nim/nvidia/model",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
    }


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {LOCAL_TOKEN}"}


def parse_sse_events(payload: bytes) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in payload.decode().split("\n\n"):
        if not block:
            continue
        event_name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        events.append((event_name, json.loads(data)))
    return events


@pytest.mark.asyncio
async def test_non_streaming_request_is_translated_and_authenticated(fake_upstream) -> None:
    async with (
        running_gateway(
            fake_upstream.base_url,
            api_key=UPSTREAM_KEY,
            upstream_auth=UpstreamAuth.REQUIRED,
        ) as server,
        aiohttp.ClientSession() as session,
        session.post(
            f"{server.base_url}/v1/messages?beta=true",
            json=anthropic_request(),
            headers=auth_headers(),
        ) as response,
    ):
        assert response.status == 200
        body = await response.json()

    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "nvidia/model"
    assert body["content"] == [{"type": "text", "text": "Hello from fake"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 3}
    assert re.fullmatch(r"msg_[0-9a-f]{32}", body["id"])

    assert len(fake_upstream.requests) == 1
    upstream_request = fake_upstream.requests[0]
    assert upstream_request["path"] == "/v1/chat/completions"
    assert upstream_request["headers"]["Authorization"] == f"Bearer {UPSTREAM_KEY}"
    assert upstream_request["body"]["model"] == "nvidia/model"
    assert upstream_request["body"]["messages"] == [{"role": "user", "content": "Hello"}]
    assert LOCAL_TOKEN not in json.dumps(body)
    assert UPSTREAM_KEY not in json.dumps(body)


@pytest.mark.asyncio
async def test_upstream_auth_none_does_not_send_authorization(fake_upstream) -> None:
    async with running_gateway(fake_upstream.base_url) as server:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server.base_url}/v1/messages",
                json=anthropic_request(),
                headers=auth_headers(),
            ) as response:
                assert response.status == 200

    assert "Authorization" not in fake_upstream.requests[0]["headers"]


@pytest.mark.asyncio
async def test_local_authentication_is_required_for_messages_and_models(fake_upstream) -> None:
    async with (
        running_gateway(fake_upstream.base_url) as server,
        aiohttp.ClientSession() as session,
    ):
        async with session.post(
            f"{server.base_url}/v1/messages",
            json=anthropic_request(),
        ) as response:
            assert response.status == 401
            body = await response.json()
            assert body["type"] == "error"
            assert body["error"]["type"] == "authentication_error"
            assert re.fullmatch(r"req_[0-9a-f]{32}", body["request_id"])

        async with session.get(f"{server.base_url}/v1/models") as response:
            assert response.status == 401
            body = await response.json()
            assert body["error"]["type"] == "authentication_error"

        async with session.get(f"{server.base_url}/v1/models?token={LOCAL_TOKEN}") as response:
            assert response.status == 401


@pytest.mark.asyncio
async def test_health_and_hello_probes_are_unauthenticated(fake_upstream) -> None:
    async with (
        running_gateway(fake_upstream.base_url) as server,
        aiohttp.ClientSession() as session,
    ):
        async with session.get(f"{server.base_url}/healthz") as response:
            assert response.status == 200
            assert await response.json() == {"status": "ok"}

        async with session.head(f"{server.base_url}/api/hello") as response:
            assert response.status == 204


@pytest.mark.asyncio
async def test_models_endpoint_returns_reversible_gateway_alias(fake_upstream) -> None:
    async with (
        running_gateway(
            fake_upstream.base_url,
            model_ids=("nvidia/second-model",),
        ) as server,
        aiohttp.ClientSession() as session,
        session.get(
            f"{server.base_url}/v1/models?limit=10",
            headers=auth_headers(),
        ) as response,
    ):
        assert response.status == 200
        body = await response.json()

    assert body == {
        "data": [
            {
                "id": "anthropic-nim/nvidia/model",
                "display_name": "NVIDIA nvidia/model (NIM)",
                "description": "NVIDIA NIM model",
            },
            {
                "id": "anthropic-nim/nvidia/second-model",
                "display_name": "NVIDIA nvidia/second-model (NIM)",
                "description": "NVIDIA NIM model",
            },
        ]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream_status", "local_status", "error_type"),
    [
        (401, 401, "authentication_error"),
        (403, 403, "permission_error"),
        (400, 400, "invalid_request_error"),
        (422, 400, "invalid_request_error"),
        (404, 404, "not_found_error"),
        (429, 429, "rate_limit_error"),
        (408, 504, "timeout_error"),
        (504, 504, "timeout_error"),
        (500, 500, "api_error"),
        (502, 500, "api_error"),
        (503, 529, "overloaded_error"),
    ],
)
async def test_upstream_errors_use_anthropic_status_and_type_mapping(
    fake_upstream,
    upstream_status: int,
    local_status: int,
    error_type: str,
) -> None:
    async def error_handler(request: web.Request) -> web.StreamResponse:
        del request
        return web.json_response(
            {"error": {"message": f"upstream detail {UPSTREAM_KEY}"}},
            status=upstream_status,
            headers={"x-request-id": "upstream-request-123"},
        )

    fake_upstream.handler = error_handler
    async with (
        running_gateway(
            fake_upstream.base_url,
            api_key=UPSTREAM_KEY,
            upstream_auth=UpstreamAuth.REQUIRED,
        ) as server,
        aiohttp.ClientSession() as session,
        session.post(
            f"{server.base_url}/v1/messages",
            json=anthropic_request(),
            headers=auth_headers(),
        ) as response,
    ):
        assert response.status == local_status
        body = await response.json()

    assert body["type"] == "error"
    assert body["error"]["type"] == error_type
    assert body["request_id"] == "upstream-request-123"
    assert UPSTREAM_KEY not in json.dumps(body)


@pytest.mark.asyncio
async def test_retry_headers_are_preserved(fake_upstream) -> None:
    async def rate_limit_handler(request: web.Request) -> web.StreamResponse:
        del request
        return web.json_response(
            {"error": {"message": "slow down"}},
            status=429,
            headers={
                "retry-after": "3",
                "x-should-retry": "true",
                "anthropic-ratelimit-requests-limit": "100",
            },
        )

    fake_upstream.handler = rate_limit_handler
    async with (
        running_gateway(fake_upstream.base_url) as server,
        aiohttp.ClientSession() as session,
    ):
        async with session.post(
            f"{server.base_url}/v1/messages",
            json=anthropic_request(),
            headers=auth_headers(),
        ) as response:
            assert response.status == 429
            assert response.headers["retry-after"] == "3"
            assert response.headers["x-should-retry"] == "true"
            assert response.headers["anthropic-ratelimit-requests-limit"] == "100"


@pytest.mark.asyncio
async def test_malformed_local_json_is_an_invalid_request(fake_upstream) -> None:
    async with (
        running_gateway(fake_upstream.base_url) as server,
        aiohttp.ClientSession() as session,
    ):
        async with session.post(
            f"{server.base_url}/v1/messages",
            data=b"not-json",
            headers={**auth_headers(), "Content-Type": "application/json"},
        ) as response:
            assert response.status == 400
            body = await response.json()
            assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_malformed_upstream_json_is_an_api_error(fake_upstream) -> None:
    async def malformed_handler(request: web.Request) -> web.StreamResponse:
        del request
        return web.Response(text="not-json", content_type="application/json")

    fake_upstream.handler = malformed_handler
    async with (
        running_gateway(fake_upstream.base_url) as server,
        aiohttp.ClientSession() as session,
    ):
        async with session.post(
            f"{server.base_url}/v1/messages",
            json=anthropic_request(),
            headers=auth_headers(),
        ) as response:
            assert response.status == 502
            body = await response.json()
            assert body["error"]["type"] == "api_error"


@pytest.mark.asyncio
async def test_malformed_upstream_message_is_an_api_error(fake_upstream) -> None:
    async def malformed_message_handler(request: web.Request) -> web.StreamResponse:
        del request
        return web.json_response({"choices": []})

    fake_upstream.handler = malformed_message_handler
    async with running_gateway(fake_upstream.base_url) as server:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server.base_url}/v1/messages",
                json=anthropic_request(),
                headers=auth_headers(),
            ) as response:
                assert response.status == 502
                body = await response.json()
                assert body["error"]["type"] == "api_error"


@pytest.mark.asyncio
async def test_upstream_connection_failure_is_an_api_error(fake_upstream) -> None:
    unused_base_url = fake_upstream.base_url
    await fake_upstream.stop()
    async with (
        running_gateway(unused_base_url) as server,
        aiohttp.ClientSession() as session,
        session.post(
            f"{server.base_url}/v1/messages",
            json=anthropic_request(),
            headers=auth_headers(),
        ) as response,
    ):
        assert response.status == 502
        body = await response.json()
        assert body["error"]["type"] == "api_error"


@pytest.mark.asyncio
async def test_upstream_timeout_maps_to_gateway_timeout(fake_upstream) -> None:
    async def slow_handler(request: web.Request) -> web.StreamResponse:
        del request
        await asyncio.sleep(0.3)
        return web.json_response({"choices": []})

    fake_upstream.handler = slow_handler
    async with (
        running_gateway(
            fake_upstream.base_url,
            upstream_connect_timeout_seconds=0.05,
            upstream_read_timeout_seconds=0.05,
        ) as server,
        aiohttp.ClientSession() as session,
        session.post(
            f"{server.base_url}/v1/messages",
            json=anthropic_request(),
            headers=auth_headers(),
        ) as response,
    ):
        assert response.status == 504
        body = await response.json()
        assert body["error"]["type"] == "timeout_error"


@pytest.mark.asyncio
async def test_streaming_response_is_incremental_and_has_anthropic_order(
    fake_upstream,
) -> None:
    release_second = asyncio.Event()

    async def stream_handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        )
        await release_second.wait()
        await response.write(
            b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    fake_upstream.handler = stream_handler
    async with (
        running_gateway(fake_upstream.base_url) as server,
        aiohttp.ClientSession() as session,
    ):
        async with session.post(
            f"{server.base_url}/v1/messages",
            json=anthropic_request(stream=True),
            headers=auth_headers(),
        ) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/event-stream")
            initial_chunks = [
                await asyncio.wait_for(response.content.readuntil(b"\n\n"), timeout=1)
                for _ in range(3)
            ]
            assert not release_second.is_set()
            release_second.set()
            payload = b"".join(initial_chunks) + await response.read()

    events = parse_sse_events(payload)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2][1]["delta"]["text"] == "Hello"
    assert events[3][1]["delta"]["text"] == " world"
    assert fake_upstream.requests[0]["body"]["stream"] is True
    assert fake_upstream.requests[0]["body"]["stream_options"] == {"include_usage": True}
    assert fake_upstream.requests[0]["headers"]["Accept"] == "text/event-stream"


@pytest.mark.asyncio
async def test_fatal_stream_error_closes_block_and_emits_terminal_error(
    fake_upstream,
) -> None:
    async def broken_stream_handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
        await response.write(b"data: not-json\n\n")
        await response.write_eof()
        return response

    fake_upstream.handler = broken_stream_handler
    async with (
        running_gateway(fake_upstream.base_url) as server,
        aiohttp.ClientSession() as session,
    ):
        async with session.post(
            f"{server.base_url}/v1/messages",
            json=anthropic_request(stream=True),
            headers=auth_headers(),
        ) as response:
            assert response.status == 200
            payload = await response.read()

    events = parse_sse_events(payload)
    assert [name for name, _ in events][-2:] == ["content_block_stop", "error"]
    assert events[-1][1]["error"]["type"] == "api_error"
    assert "message_stop" not in [name for name, _ in events]


@pytest.mark.asyncio
async def test_client_disconnect_cancels_the_upstream_request(fake_upstream) -> None:
    upstream_cancelled = asyncio.Event()

    async def hanging_stream_handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        await response.write(b": waiting\n\n")
        while request.transport is not None and not request.transport.is_closing():
            await asyncio.sleep(0.01)
        upstream_cancelled.set()
        return response

    fake_upstream.handler = hanging_stream_handler
    async with (
        running_gateway(fake_upstream.base_url) as server,
        aiohttp.ClientSession() as session,
    ):
        response = await session.post(
            f"{server.base_url}/v1/messages",
            json=anthropic_request(stream=True),
            headers=auth_headers(),
        )
        assert response.status == 200
        await response.content.readuntil(b"\n\n")
        response.close()
        await session.close()
        await asyncio.wait_for(upstream_cancelled.wait(), timeout=2)


def test_non_loopback_binding_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="non-loopback"):
        GatewaySettings(
            local_token=LOCAL_TOKEN,
            default_model="nvidia/model",
            upstream_base_url="http://127.0.0.1:1/v1",
            listen_host="0.0.0.0",
        )
