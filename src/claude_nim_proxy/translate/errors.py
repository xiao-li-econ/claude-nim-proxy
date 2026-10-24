"""Anthropic-style error envelopes and safe upstream error conversion."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

_ERROR_TYPES_BY_STATUS: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    504: "timeout_error",
    529: "overloaded_error",
}
_DEFAULT_MESSAGES: dict[int, str] = {
    400: "The request was invalid.",
    401: "Authentication with NVIDIA NIM failed.",
    403: "NVIDIA NIM denied access to the requested resource.",
    404: "The requested NVIDIA NIM resource was not found.",
    429: "NVIDIA NIM rate limit exceeded.",
    502: "NVIDIA NIM upstream error.",
    504: "NVIDIA NIM request timed out.",
    529: "NVIDIA NIM is temporarily overloaded.",
}
_SAFE_HEADER_NAMES = {
    "retry-after",
    "x-should-retry",
}
_REDACTIONS = (
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:api[_ -]?key|token)\s*[:=]\s*)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\bnvapi-[a-z0-9_-]{8,}\b"), "[REDACTED]"),
)


@dataclass(slots=True)
class AnthropicError(Exception):
    """An error that can be rendered using Anthropic's HTTP error envelope."""

    status: int
    error_type: str
    message: str
    headers: Mapping[str, str] = field(default_factory=dict)
    request_id: str | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def payload(self) -> dict[str, Any]:
        """Return the exact Anthropic error envelope for this error."""

        return {
            "type": "error",
            "error": {"type": self.error_type, "message": self.message},
            "request_id": self.request_id or make_request_id(),
        }


def make_request_id() -> str:
    """Generate a non-secret request identifier."""

    return f"req_{uuid.uuid4().hex}"


def error_type_for_status(status: int) -> str:
    """Map an upstream HTTP status to an Anthropic error type and local status."""

    if status in {400, 422}:
        return "invalid_request_error"
    if status in {500, 502}:
        return "api_error"
    if status in {408, 504}:
        return "timeout_error"
    if status == 503:
        return "overloaded_error"
    return _ERROR_TYPES_BY_STATUS.get(status, "api_error")


def local_status_for_upstream(status: int) -> int:
    """Map an upstream HTTP status to the status returned to Claude Code."""

    if status in {400, 422}:
        return 400
    if status == 503:
        return 529
    if status == 408:
        return 504
    if status in {401, 403, 404, 429, 500, 504}:
        return status
    if status == 502:
        return 500
    return 502


def default_error_message(status: int) -> str:
    """Return a safe generic message for an upstream status."""

    return _DEFAULT_MESSAGES.get(status, "NVIDIA NIM upstream error.")


def sanitize_upstream_message(
    raw_message: object,
    *,
    secrets: Iterable[str] = (),
    max_length: int = 500,
) -> str | None:
    """Sanitize and truncate a useful upstream error message."""

    if not isinstance(raw_message, str):
        return None
    message = raw_message.strip()
    if not message:
        return None
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    for pattern, replacement in _REDACTIONS:
        message = pattern.sub(replacement, message)
    if len(message) > max_length:
        message = message[: max_length - 1].rstrip() + "…"
    return message


def safe_upstream_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Copy retry and rate-limit headers that are safe to expose downstream."""

    safe: dict[str, str] = {}
    for name, value in headers.items():
        normalized = name.lower()
        if normalized in _SAFE_HEADER_NAMES or normalized.startswith(
            ("anthropic-ratelimit-", "x-ratelimit-")
        ):
            safe[name] = value
    return safe


def upstream_request_id(headers: Mapping[str, str]) -> str | None:
    """Return a safe upstream request identifier when one is available."""

    for name in ("request-id", "x-request-id", "x-requestid"):
        value = headers.get(name)
        if value:
            sanitized = "".join(
                character for character in value if character.isalnum() or character in "-_"
            )
            if sanitized:
                return sanitized[:128]
    return None
