"""Asynchronous client for NVIDIA's OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp

from .config import UpstreamAuth
from .translate.errors import (
    AnthropicError,
    default_error_message,
    error_type_for_status,
    local_status_for_upstream,
    safe_upstream_headers,
    sanitize_upstream_message,
    upstream_request_id,
)

_MAX_ERROR_BODY_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class UpstreamJSONResponse:
    """A successful non-streaming upstream response and its safe metadata."""

    data: dict[str, Any]
    headers: Mapping[str, str]


class UpstreamStream:
    """An open streaming upstream response."""

    def __init__(self, response: aiohttp.ClientResponse) -> None:
        self._response = response

    @property
    def headers(self) -> Mapping[str, str]:
        """Return the upstream response headers."""

        return self._response.headers

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield upstream response bytes as they arrive."""

        async for chunk in self._response.content.iter_any():
            if chunk:
                yield chunk

    async def close(self) -> None:
        """Close the upstream response and cancel any remaining body read."""

        self._response.close()


class NvidiaCatalogClient:
    """Fetch NVIDIA's public model catalog without sending a credential."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
    ) -> None:
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=connect_timeout_seconds,
            sock_read=read_timeout_seconds,
        )

    @property
    def models_url(self) -> str:
        """Return the normalized public model catalog endpoint."""

        return _join_endpoint(self._base_url, "models")

    async def fetch_catalog(self) -> dict[str, Any]:
        """Fetch and decode the OpenAI-style NVIDIA model catalog."""

        try:
            response = await self._session.get(
                self.models_url,
                headers={"Accept": "application/json"},
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            raise AnthropicError(
                504,
                "timeout_error",
                "NVIDIA NIM model catalog request timed out.",
            ) from exc
        except aiohttp.ClientError as exc:
            raise AnthropicError(
                502,
                "api_error",
                "Could not connect to the NVIDIA NIM model catalog.",
            ) from exc

        try:
            if 200 <= response.status < 300:
                body = await response.read()
            else:
                body = await _read_limited(response)
        except (TimeoutError, aiohttp.ClientError) as exc:
            raise _connection_or_timeout_error(exc) from exc
        finally:
            response.release()

        if not 200 <= response.status < 300:
            raise _http_error(response.status, response.headers, body, None)

        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AnthropicError(
                502,
                "api_error",
                "NVIDIA NIM returned a malformed model catalog.",
            ) from exc
        if not isinstance(decoded, dict):
            raise AnthropicError(
                502,
                "api_error",
                "NVIDIA NIM returned a malformed model catalog.",
            )
        return decoded


class NvidiaUpstreamClient:
    """Send translated Chat Completions requests to NVIDIA NIM."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        api_key: str | None,
        auth_mode: UpstreamAuth,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
    ) -> None:
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._auth_mode = auth_mode
        self._timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=connect_timeout_seconds,
            sock_read=read_timeout_seconds,
        )

    @property
    def chat_completions_url(self) -> str:
        """Return the normalized Chat Completions endpoint."""

        return _join_endpoint(self._base_url, "chat/completions")

    async def complete(self, payload: dict[str, Any]) -> UpstreamJSONResponse:
        """Send a non-streaming request and decode its JSON response."""

        response = await self._post(self.chat_completions_url, payload)
        try:
            if 200 <= response.status < 300:
                body = await response.read()
            else:
                body = await _read_limited(response)
        except (TimeoutError, aiohttp.ClientError) as exc:
            raise _connection_or_timeout_error(exc) from exc
        finally:
            response.release()

        if not 200 <= response.status < 300:
            raise _http_error(response.status, response.headers, body, self._api_key)

        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AnthropicError(
                502,
                "api_error",
                "NVIDIA NIM returned a malformed JSON response.",
            ) from exc
        if not isinstance(decoded, dict):
            raise AnthropicError(
                502,
                "api_error",
                "NVIDIA NIM returned a malformed JSON response.",
            )
        return UpstreamJSONResponse(data=decoded, headers=response.headers)

    async def open_stream(self, payload: dict[str, Any]) -> UpstreamStream:
        """Open a streaming request and return its live response body."""

        response = await self._post(self.chat_completions_url, payload)
        if not 200 <= response.status < 300:
            try:
                body = await _read_limited(response)
            finally:
                response.release()
            raise _http_error(response.status, response.headers, body, self._api_key)

        content_type = response.headers.get("Content-Type", "")
        if not content_type.lower().startswith("text/event-stream"):
            try:
                await _read_limited(response)
            finally:
                response.release()
            raise AnthropicError(
                502,
                "api_error",
                "NVIDIA NIM returned a non-streaming response for a streaming request.",
            )
        return UpstreamStream(response)

    async def _post(self, url: str, payload: dict[str, Any]) -> aiohttp.ClientResponse:
        accept = "text/event-stream" if payload.get("stream") is True else "application/json"
        headers = {"Accept": accept}
        if self._auth_mode is not UpstreamAuth.NONE:
            if not self._api_key:
                raise AnthropicError(
                    500,
                    "api_error",
                    "An NVIDIA API credential is required but was not configured.",
                )
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            return await self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            raise AnthropicError(
                504,
                "timeout_error",
                "NVIDIA NIM request timed out.",
            ) from exc
        except aiohttp.ClientError as exc:
            raise AnthropicError(
                502,
                "api_error",
                "Could not connect to NVIDIA NIM.",
            ) from exc


def _join_endpoint(base_url: str, endpoint: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith(f"/{endpoint}"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/{endpoint}"
    return f"{normalized}/v1/{endpoint}"


async def _read_limited(response: aiohttp.ClientResponse) -> bytes:
    try:
        return await response.content.read(_MAX_ERROR_BODY_BYTES + 1)
    except (TimeoutError, aiohttp.ClientError) as exc:
        raise _connection_or_timeout_error(exc) from exc


def _http_error(
    status: int,
    headers: Mapping[str, str],
    body: bytes,
    api_key: str | None,
) -> AnthropicError:
    local_status = local_status_for_upstream(status)
    error_type = error_type_for_status(status)
    message = _extract_error_message(body)
    sanitized = sanitize_upstream_message(message, secrets=(api_key or "",))
    if sanitized is None:
        sanitized = default_error_message(local_status)
    return AnthropicError(
        local_status,
        error_type,
        sanitized,
        headers=safe_upstream_headers(headers),
        request_id=upstream_request_id(headers),
    )


def _extract_error_message(body: bytes) -> str | None:
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    error = decoded.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    message = decoded.get("message")
    return message if isinstance(message, str) else None


def _connection_or_timeout_error(exc: BaseException) -> AnthropicError:
    if isinstance(exc, TimeoutError):
        return AnthropicError(504, "timeout_error", "NVIDIA NIM request timed out.")
    return AnthropicError(502, "api_error", "Could not connect to NVIDIA NIM.")
