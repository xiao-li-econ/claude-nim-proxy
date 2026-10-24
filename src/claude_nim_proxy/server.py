"""Loopback-only Anthropic Messages gateway backed by NVIDIA NIM."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import secrets
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import aiohttp
from aiohttp import web

from .config import UpstreamAuth
from .translate.anthropic_to_openai import (
    GATEWAY_MODEL_PREFIX,
    RequestTranslationError,
    RequestTranslationOptions,
    UnsupportedImageError,
    resolve_upstream_model,
    translate_request,
)
from .translate.errors import AnthropicError, safe_upstream_headers
from .translate.openai_to_anthropic import ResponseTranslationError, translate_response
from .translate.stream import translate_stream
from .upstream import NvidiaUpstreamClient

_DISCONNECT_POLL_SECONDS = 0.05


class ClientDisconnected(Exception):
    """Raised internally when the Claude-facing client disconnects."""


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    """Configuration required by the local gateway."""

    local_token: str
    default_model: str
    upstream_base_url: str
    api_key: str | None = None
    upstream_auth: UpstreamAuth = UpstreamAuth.AUTO
    listen_host: str = "127.0.0.1"
    listen_port: int = 0
    upstream_connect_timeout_seconds: float = 10.0
    upstream_read_timeout_seconds: float = 600.0
    stream_ping_interval_seconds: float = 1.0
    supports_images: bool = True
    model_ids: tuple[str, ...] = ()
    model_display_names: Mapping[str, str] = field(default_factory=dict)
    model_image_support: Mapping[str, bool] = field(default_factory=dict)
    known_model_ids: tuple[str, ...] = ()
    context_window: int = 32_768
    max_output_tokens: int = 4_096
    allow_non_loopback: bool = False
    resolved_default_model: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.local_token:
            raise ValueError("local_token must not be empty")
        if not self.default_model:
            raise ValueError("default_model must not be empty")
        object.__setattr__(
            self,
            "resolved_default_model",
            resolve_upstream_model(self.default_model),
        )
        if not _is_loopback_address(self.listen_host) and not self.allow_non_loopback:
            raise ValueError("non-loopback binding requires allow_non_loopback=True")

    @property
    def binds_non_loopback(self) -> bool:
        """Return whether this gateway listens beyond the loopback interface."""

        return not _is_loopback_address(self.listen_host)

    @property
    def raw_model_ids(self) -> tuple[str, ...]:
        """Return raw upstream IDs advertised by the model endpoint."""

        models = [self.resolved_default_model]
        for model_id in self.model_ids:
            raw_id = resolve_upstream_model(model_id)
            if raw_id not in models:
                models.append(raw_id)
        return tuple(models)

    def supports_images_for(self, upstream_model_id: str) -> bool:
        """Return image support for one resolved upstream model ID."""

        if upstream_model_id in self.model_image_support:
            return self.model_image_support[upstream_model_id]
        if self.known_model_ids and upstream_model_id not in self.known_model_ids:
            return False
        return self.supports_images


class Gateway:
    """Implement the Anthropic-facing HTTP routes."""

    def __init__(
        self,
        settings: GatewaySettings,
        upstream_client: NvidiaUpstreamClient,
    ) -> None:
        self.settings = settings
        self.upstream_client = upstream_client

    async def handle_messages(self, request: web.Request) -> web.StreamResponse:
        """Handle POST /v1/messages with streaming or non-streaming translation."""

        if not self._authenticated(request):
            return self._error_response(
                AnthropicError(
                    401,
                    "authentication_error",
                    "Invalid local bearer token.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            )

        try:
            request_body = await _read_json_object(request)
        except AnthropicError as exc:
            return self._error_response(exc)

        supports_images = self.settings.supports_images
        requested_model = request_body.get("model")
        if isinstance(requested_model, str):
            try:
                resolved_model = resolve_upstream_model(
                    requested_model,
                    self.settings.resolved_default_model,
                )
            except RequestTranslationError:
                return self._error_response(
                    AnthropicError(400, "invalid_request_error", "The request was invalid.")
                )
            supports_images = self.settings.supports_images_for(resolved_model)

        try:
            translated = translate_request(
                request_body,
                RequestTranslationOptions(
                    default_model=self.settings.resolved_default_model,
                    supports_images=supports_images,
                ),
            )
        except UnsupportedImageError:
            return self._error_response(
                AnthropicError(
                    400,
                    "invalid_request_error",
                    "The selected model does not support image input.",
                )
            )
        except RequestTranslationError:
            return self._error_response(
                AnthropicError(400, "invalid_request_error", "The request was invalid.")
            )

        if translated.get("stream") is True:
            return await self._handle_streaming(request, translated)
        return await self._handle_non_streaming(translated)

    async def handle_models(self, request: web.Request) -> web.StreamResponse:
        """Handle authenticated Claude Code model discovery."""

        if not self._authenticated(request):
            return self._error_response(
                AnthropicError(
                    401,
                    "authentication_error",
                    "Invalid local bearer token.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            )
        data = [
            {
                "id": f"{GATEWAY_MODEL_PREFIX}{model_id}",
                "display_name": self.settings.model_display_names.get(
                    model_id,
                    f"NVIDIA {model_id} (NIM)",
                ),
                "description": "NVIDIA NIM model",
            }
            for model_id in self.settings.raw_model_ids
        ]
        return web.json_response({"data": data})

    async def handle_health(self, request: web.Request) -> web.StreamResponse:
        """Handle the unauthenticated local health probe."""

        del request
        return web.json_response({"status": "ok"})

    async def handle_hello(self, request: web.Request) -> web.StreamResponse:
        """Handle Claude Code's best-effort startup probe."""

        del request
        return web.Response(status=204)

    def _authenticated(self, request: web.Request) -> bool:
        provided = request.headers.get("Authorization", "")
        expected = f"Bearer {self.settings.local_token}"
        return secrets.compare_digest(provided, expected)

    async def _handle_non_streaming(self, translated: dict[str, Any]) -> web.StreamResponse:
        try:
            upstream_response = await self.upstream_client.complete(translated)
        except AnthropicError as exc:
            return self._error_response(exc)

        fallback_model = str(translated["model"])
        try:
            response_body = translate_response(upstream_response.data, fallback_model)
        except ResponseTranslationError:
            return self._error_response(
                AnthropicError(
                    502,
                    "api_error",
                    "NVIDIA NIM returned a malformed response.",
                )
            )
        headers = safe_upstream_headers(upstream_response.headers)
        return web.json_response(response_body, headers=headers)

    async def _handle_streaming(
        self, request: web.Request, translated: dict[str, Any]
    ) -> web.StreamResponse:
        try:
            upstream_stream = await self.upstream_client.open_stream(translated)
        except AnthropicError as exc:
            return self._error_response(exc)

        response_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        response_headers.update(safe_upstream_headers(upstream_stream.headers))
        response = web.StreamResponse(status=200, headers=response_headers)
        try:
            await response.prepare(request)
            chunks = _DisconnectAwareChunks(upstream_stream.chunks(), request)
            fallback_model = str(translated["model"])
            async for message in translate_stream(
                chunks,
                fallback_model,
                ping_interval_seconds=self.settings.stream_ping_interval_seconds,
            ):
                await response.write(message.encode())
            await response.write_eof()
        except (ClientDisconnected, ConnectionResetError):
            return response
        except asyncio.CancelledError:
            raise
        finally:
            await upstream_stream.close()
        return response

    def _error_response(self, error: AnthropicError) -> web.StreamResponse:
        headers = dict(error.headers)
        if error.request_id:
            headers["request-id"] = error.request_id
        return web.json_response(error.payload(), status=error.status, headers=headers)


class GatewayServer:
    """Run the local gateway on a loopback listener."""

    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings
        self._runner: web.AppRunner | None = None
        self._session: aiohttp.ClientSession | None = None
        self._base_url: str | None = None

    @property
    def base_url(self) -> str:
        """Return the selected local base URL."""

        if self._base_url is None:
            raise RuntimeError("gateway server has not started")
        return self._base_url

    async def start(self) -> str:
        """Start listening and return the local base URL."""

        if self._runner is not None:
            return self.base_url
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        upstream_client = NvidiaUpstreamClient(
            self._session,
            base_url=self.settings.upstream_base_url,
            api_key=self.settings.api_key,
            auth_mode=self.settings.upstream_auth,
            connect_timeout_seconds=self.settings.upstream_connect_timeout_seconds,
            read_timeout_seconds=self.settings.upstream_read_timeout_seconds,
        )
        gateway = Gateway(self.settings, upstream_client)
        app = web.Application(client_max_size=16 * 1024 * 1024)
        app.router.add_post("/v1/messages", gateway.handle_messages)
        app.router.add_get("/v1/models", gateway.handle_models)
        app.router.add_get("/healthz", gateway.handle_health)
        app.router.add_route("HEAD", "/api/hello", gateway.handle_hello)

        self._runner = web.AppRunner(app, access_log=None)
        try:
            await self._runner.setup()
            site = web.TCPSite(
                self._runner,
                host=self.settings.listen_host,
                port=self.settings.listen_port,
            )
            await site.start()
        except BaseException:
            await self.stop()
            raise
        addresses = self._runner.addresses
        if not addresses:
            await self.stop()
            raise RuntimeError("gateway server did not bind a listener")
        address = cast(tuple[str, int], addresses[0])
        self._base_url = _base_url(self.settings.listen_host, address[1])
        return self._base_url

    async def stop(self) -> None:
        """Stop the listener and close upstream connections."""

        runner = self._runner
        self._runner = None
        self._base_url = None
        try:
            if runner is not None:
                await runner.cleanup()
        finally:
            session = self._session
            self._session = None
            if session is not None:
                await session.close()

    async def __aenter__(self) -> GatewayServer:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()


class _DisconnectAwareChunks:
    """Cancel an in-progress upstream read when the Claude client disconnects."""

    def __init__(
        self,
        chunks: AsyncIterator[bytes],
        request: web.Request,
    ) -> None:
        self._chunks = chunks
        self._request = request

    def __aiter__(self) -> _DisconnectAwareChunks:
        return self

    async def __anext__(self) -> bytes:
        if self._disconnected():
            raise ClientDisconnected
        pending: asyncio.Task[bytes] = asyncio.create_task(_next_chunk(self._chunks))
        try:
            while True:
                done, _ = await asyncio.wait(
                    (pending,),
                    timeout=_DISCONNECT_POLL_SECONDS,
                )
                if done:
                    return pending.result()
                if self._disconnected():
                    raise ClientDisconnected
        finally:
            if not pending.done():
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                    await pending

    def _disconnected(self) -> bool:
        transport = self._request.transport
        return transport is None or transport.is_closing()


async def _next_chunk(chunks: AsyncIterator[bytes]) -> bytes:
    return await chunks.__anext__()


async def _read_json_object(request: web.Request) -> dict[str, Any]:
    try:
        raw = await request.text()
        decoded = json.loads(raw)
    except web.HTTPRequestEntityTooLarge as exc:
        raise AnthropicError(
            400,
            "invalid_request_error",
            "The request body is too large.",
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnthropicError(
            400,
            "invalid_request_error",
            "The request body must contain valid JSON.",
        ) from exc
    if not isinstance(decoded, dict):
        raise AnthropicError(
            400,
            "invalid_request_error",
            "The request body must be a JSON object.",
        )
    return decoded


def _is_loopback_address(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _base_url(host: str, port: int) -> str:
    if host in {"0.0.0.0", ""}:
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"
