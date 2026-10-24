"""Translate OpenAI Chat Completions responses into Anthropic Messages responses."""

from __future__ import annotations

import json
import uuid
from typing import Any


class ResponseTranslationError(ValueError):
    """Raised when an upstream response cannot be represented as an Anthropic response."""


def translate_response(response: dict[str, Any], fallback_model: str) -> dict[str, Any]:
    """Translate a non-streaming OpenAI response to the Anthropic Messages envelope."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ResponseTranslationError("upstream response contains no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ResponseTranslationError("upstream response choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ResponseTranslationError("upstream response contains no message")

    content: list[dict[str, Any]] = []
    text = _extract_text(message.get("content"))
    if text:
        content.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for raw_tool_call in tool_calls:
            if not isinstance(raw_tool_call, dict):
                raise ResponseTranslationError("upstream tool call must be an object")
            content.append(_translate_tool_call(raw_tool_call))

    legacy_function_call = message.get("function_call")
    if isinstance(legacy_function_call, dict):
        content.append(_translate_legacy_function_call(legacy_function_call))

    finish_reason = first_choice.get("finish_reason")
    stop_reason = map_stop_reason(finish_reason)
    usage = _translate_usage(response.get("usage"))
    upstream_model = response.get("model")
    model = upstream_model if isinstance(upstream_model, str) and upstream_model else fallback_model

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def map_stop_reason(finish_reason: Any) -> str:
    """Map an OpenAI finish reason to an Anthropic stop reason."""

    if finish_reason == "length":
        return "max_tokens"
    if finish_reason in {"tool_calls", "function_call"}:
        return "tool_use"
    return "end_turn"


def _extract_text(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for part in raw_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def _translate_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ResponseTranslationError("upstream tool call contains no function")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ResponseTranslationError("upstream tool call contains no function name")
    call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"toolu_{uuid.uuid4().hex}"
    arguments = function.get("arguments", "{}")
    tool_input = _parse_tool_input(arguments)
    return {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}


def _translate_legacy_function_call(function_call: dict[str, Any]) -> dict[str, Any]:
    name = function_call.get("name")
    if not isinstance(name, str) or not name:
        raise ResponseTranslationError("legacy function call contains no name")
    return {
        "type": "tool_use",
        "id": f"toolu_{uuid.uuid4().hex}",
        "name": name,
        "input": _parse_tool_input(function_call.get("arguments", "{}")),
    }


def _parse_tool_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if arguments is None or arguments == "":
        return {}
    if not isinstance(arguments, str):
        raise ResponseTranslationError("tool call arguments must be a JSON object string")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ResponseTranslationError("tool call arguments are not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ResponseTranslationError("tool call arguments must decode to an object")
    return parsed


def _translate_usage(raw_usage: Any) -> dict[str, int]:
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    if not isinstance(raw_usage, dict):
        return usage
    prompt_tokens = raw_usage.get("prompt_tokens")
    completion_tokens = raw_usage.get("completion_tokens")
    if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
        usage["input_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
        usage["output_tokens"] = completion_tokens
    details = raw_usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached_tokens = details.get("cached_tokens")
        if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool):
            usage["cache_read_input_tokens"] = cached_tokens
    return usage
