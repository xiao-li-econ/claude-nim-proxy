"""Redacted diagnostic logging for the proxy.

Debug logs intentionally exclude prompts, message bodies, tool schemas, and response
content. A logging filter also removes configured secrets and common credential headers
as a defense in depth.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable, Mapping

LOGGER_NAME = "claude_nim_proxy"
_REDACTIONS = (
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:api[_ -]?key|token)\s*[:=]\s*)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\bnvapi-[a-z0-9_-]{8,}\b"), "[REDACTED]"),
)


class _RedactionFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._secrets: tuple[str, ...] = ()

    def set_secrets(self, secrets: Iterable[str]) -> None:
        normalized = tuple(secret for secret in secrets if secret)
        self._secrets = normalized

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in self._secrets:
            message = message.replace(secret, "[REDACTED]")
        for pattern, replacement in _REDACTIONS:
            message = pattern.sub(replacement, message)
        record.msg = message
        record.args = ()
        return True


class _DynamicStderrHandler(logging.Handler):
    """Write to the current standard error stream at emit time."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            print(message, file=sys.stderr)
        except Exception:
            self.handleError(record)


_logger = logging.getLogger(LOGGER_NAME)
_logger.propagate = False
_filter = _RedactionFilter()
_logger.addFilter(_filter)
_handler: logging.Handler | None = None


def configure_debug(enabled: bool) -> None:
    """Enable or disable the proxy's redacted debug log stream."""

    global _handler
    if enabled:
        if _handler is None:
            _handler = _DynamicStderrHandler()
            _handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            _handler.addFilter(_filter)
            _logger.addHandler(_handler)
        _logger.setLevel(logging.DEBUG)
        return
    _logger.setLevel(logging.WARNING)


def set_debug_secrets(secrets: Iterable[str]) -> None:
    """Register values that must be redacted from all debug output."""

    _filter.set_secrets(secrets)


def debug_logger() -> logging.Logger:
    """Return the dedicated proxy debug logger."""

    return _logger


def safe_model(model: object) -> str:
    """Return a bounded, single-line model identifier safe for logs."""

    text = str(model)
    text = "".join(character if character.isprintable() else "?" for character in text)
    return text[:200]


def safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return header values with credential-bearing headers redacted."""

    redacted: dict[str, str] = {}
    for name, value in headers.items():
        normalized = name.lower()
        if normalized in {"authorization", "x-api-key", "cookie", "set-cookie"}:
            redacted[name] = "[REDACTED]"
        else:
            redacted[name] = value
    return redacted
