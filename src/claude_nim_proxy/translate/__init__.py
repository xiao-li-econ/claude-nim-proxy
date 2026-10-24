"""Pure protocol translation helpers."""

from .anthropic_to_openai import (
    GATEWAY_MODEL_PREFIX,
    RequestTranslationError,
    RequestTranslationOptions,
    resolve_upstream_model,
    translate_request,
)
from .openai_to_anthropic import ResponseTranslationError, translate_response

__all__ = [
    "GATEWAY_MODEL_PREFIX",
    "RequestTranslationError",
    "RequestTranslationOptions",
    "ResponseTranslationError",
    "resolve_upstream_model",
    "translate_request",
    "translate_response",
]
