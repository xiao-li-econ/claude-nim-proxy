from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest_asyncio
from aiohttp import web


class FakeNvidiaUpstream:
    """Controllable fake NVIDIA OpenAI-compatible upstream."""

    def __init__(self) -> None:
        self.app = web.Application()
        self.runner: web.AppRunner | None = None
        self.port: int | None = None
        self.requests: list[dict[str, Any]] = []
        self.catalog_requests: list[dict[str, Any]] = []
        self.handler: Callable[[web.Request], Awaitable[web.StreamResponse]] = self._default_handler
        self.app.router.add_post("/v1/chat/completions", self._handle)
        self.app.router.add_get("/v1/models", self._handle_catalog)

    async def start(self) -> None:
        self.runner = web.AppRunner(self.app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, host="127.0.0.1", port=0)
        await site.start()
        addresses = self.runner.addresses
        assert addresses
        self.port = int(addresses[0][1])

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None

    @property
    def base_url(self) -> str:
        assert self.port is not None
        return f"http://127.0.0.1:{self.port}/v1"

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(
            {
                "method": request.method,
                "path": request.path,
                "query": dict(request.query),
                "headers": dict(request.headers),
                "body": body,
            }
        )
        return await self.handler(request)

    async def _handle_catalog(self, request: web.Request) -> web.StreamResponse:
        self.catalog_requests.append({"method": request.method, "path": request.path})
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {"id": "nvidia/model", "object": "model"},
                ],
            }
        )

    async def _default_handler(self, request: web.Request) -> web.StreamResponse:
        del request
        return web.json_response(
            {
                "id": "chatcmpl-fake",
                "model": "nvidia/model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from fake"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
        )


@pytest_asyncio.fixture
async def fake_upstream() -> FakeNvidiaUpstream:
    upstream = FakeNvidiaUpstream()
    await upstream.start()
    try:
        yield upstream
    finally:
        await upstream.stop()
