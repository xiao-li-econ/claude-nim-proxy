"""Translate Anthropic Messages requests into OpenAI Chat Completions requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

GATEWAY_MODEL_PREFIX = "anthropic-nim/"
_CLAUDE_MODEL_NAMES = {"opus", "sonnet", "haiku", "fable"}
_CLAUDE_MODEL_PREFIXES = ("claude-", "claude/", "claude_")


class RequestTranslationError(ValueError):
    """Raised when an Anthropic request cannot be represented safely."""


class UnsupportedImageError(RequestTranslationError):
    """Raised when a request contains an image for a text-only model."""


@dataclass(frozen=True, slots=True)
class RequestTranslationOptions:
    """Options that affect request translation."""

    default_model: str | None = None
    supports_images: bool = True
    include_stream_usage: bool = True


def resolve_upstream_model(requested: str | None, default_model: str | None = None) -> str:
    """Resolve a Claude-facing model name to a raw NVIDIA model ID.

    Reversible ``anthropic-nim/`` aliases and raw NVIDIA IDs are accepted. Built-in
    Claude names are defensive fallbacks to the explicitly selected default model.
    """

    if requested and requested.startswith(GATEWAY_MODEL_PREFIX):
        resolved = requested[len(GATEWAY_MODEL_PREFIX) :]
        if not resolved:
            raise RequestTranslationError("gateway model alias has no upstream model ID")
        return resolved

    if requested and _is_claude_model(requested):
        if not default_model:
            raise RequestTranslationError(
                f"Claude model {requested!r} requires an explicitly selected NVIDIA model"
            )
        return resolve_upstream_model(default_model)

    if requested:
        return requested
    if default_model:
        return resolve_upstream_model(default_model)
    raise RequestTranslationError("no upstream model was selected")


def translate_request(
    request: dict[str, Any], options: RequestTranslationOptions | None = None
) -> dict[str, Any]:
    """Translate an Anthropic Messages request to an OpenAI Chat Completions request."""

    opts = options or RequestTranslationOptions()
    requested_model = request.get("model")
    if requested_model is not None and not isinstance(requested_model, str):
        raise RequestTranslationError("model must be a string")
    model = resolve_upstream_model(requested_model, opts.default_model)

    messages: list[dict[str, Any]] = []
    system_text = _translate_system(request.get("system"))
    if system_text is not None:
        messages.append({"role": "system", "content": system_text})

    raw_messages = request.get("messages")
    if not isinstance(raw_messages, list):
        raise RequestTranslationError("messages must be an array")
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            raise RequestTranslationError("each message must be an object")
        messages.extend(_translate_message(raw_message, opts.supports_images))

    translated: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    max_tokens = request.get("max_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
        translated["max_tokens"] = max_tokens

    stop_sequences = request.get("stop_sequences")
    if isinstance(stop_sequences, list) and all(isinstance(item, str) for item in stop_sequences):
        translated["stop"] = list(stop_sequences)

    for key in ("temperature", "top_p"):
        value = request.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            translated[key] = value

    stream = request.get("stream")
    if isinstance(stream, bool):
        translated["stream"] = stream
        if stream and opts.include_stream_usage:
            translated["stream_options"] = {"include_usage": True}

    tools = _translate_tools(request.get("tools"))
    if tools:
        translated["tools"] = tools

    tool_choice = _translate_tool_choice(request.get("tool_choice"))
    if tool_choice is not None:
        translated["tool_choice"] = tool_choice

    disable_parallel = request.get("disable_parallel_tool_use")
    if not isinstance(disable_parallel, bool):
        choice = request.get("tool_choice")
        if isinstance(choice, dict):
            disable_parallel = choice.get("disable_parallel_tool_use")
    if disable_parallel is True:
        translated["parallel_tool_calls"] = False

    reasoning_effort = _translate_reasoning_effort(request)
    if reasoning_effort is not None:
        translated["reasoning_effort"] = reasoning_effort

    return translated


def _is_claude_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized in _CLAUDE_MODEL_NAMES or normalized.startswith(_CLAUDE_MODEL_PREFIXES)


def _translate_system(raw_system: Any) -> str | None:
    if raw_system is None:
        return None
    if isinstance(raw_system, str):
        return raw_system
    if not isinstance(raw_system, list):
        raise RequestTranslationError("system must be a string or an array of content blocks")

    parts: list[str] = []
    for block in raw_system:
        if not isinstance(block, dict):
            raise RequestTranslationError("system content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise RequestTranslationError("system text block requires string text")
            parts.append(text)
        elif block_type not in {"redacted_thinking", "thinking"}:
            raise RequestTranslationError(f"unsupported system content block: {block_type!r}")
    return "\n".join(parts)


def _translate_message(message: dict[str, Any], supports_images: bool) -> list[dict[str, Any]]:
    role = message.get("role")
    if role not in {"user", "assistant", "system"}:
        raise RequestTranslationError(f"unsupported message role: {role!r}")
    content = message.get("content")
    if isinstance(content, str):
        if role == "system":
            return [{"role": "user", "content": f"[System reminder] {content}"}]
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        raise RequestTranslationError("message content must be a string or an array of blocks")

    if role == "system":
        text = _text_from_blocks(content)
        return [{"role": "user", "content": f"[System reminder] {text}"}]

    if role == "assistant":
        return _translate_assistant_blocks(content, supports_images)
    return _translate_user_blocks(content, supports_images)


def _translate_assistant_blocks(blocks: list[Any], supports_images: bool) -> list[dict[str, Any]]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise RequestTranslationError("message content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise RequestTranslationError("text block requires string text")
            text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(_translate_tool_use(block))
        elif block_type in {"thinking", "redacted_thinking"}:
            continue
        elif block_type == "image":
            raise RequestTranslationError("assistant messages cannot contain image blocks")
        else:
            raise RequestTranslationError(f"unsupported assistant content block: {block_type!r}")

    if not text_parts and not tool_calls:
        return []
    translated: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
    if tool_calls:
        translated["tool_calls"] = tool_calls
    return [translated]


def _translate_user_blocks(blocks: list[Any], supports_images: bool) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    pending_content: list[dict[str, Any]] = []

    def flush_user_content() -> None:
        if pending_content:
            translated.append({"role": "user", "content": _content_value(pending_content)})
            pending_content.clear()

    for block in blocks:
        if not isinstance(block, dict):
            raise RequestTranslationError("message content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise RequestTranslationError("text block requires string text")
            pending_content.append({"type": "text", "text": text})
        elif block_type == "image":
            pending_content.append(_translate_image(block, supports_images))
        elif block_type == "tool_result":
            flush_user_content()
            translated.append(_translate_tool_result(block, supports_images))
        elif block_type in {"thinking", "redacted_thinking"}:
            continue
        else:
            raise RequestTranslationError(f"unsupported user content block: {block_type!r}")

    flush_user_content()
    return translated


def _translate_tool_use(block: dict[str, Any]) -> dict[str, Any]:
    tool_id = block.get("id")
    name = block.get("name")
    if not isinstance(tool_id, str) or not tool_id:
        raise RequestTranslationError("tool_use block requires a non-empty id")
    if not isinstance(name, str) or not name:
        raise RequestTranslationError("tool_use block requires a non-empty name")
    tool_input = block.get("input", {})
    if not isinstance(tool_input, dict):
        raise RequestTranslationError("tool_use input must be an object")
    try:
        arguments = json.dumps(tool_input, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise RequestTranslationError("tool_use input must be JSON serializable") from exc
    return {
        "id": tool_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _translate_tool_result(block: dict[str, Any], supports_images: bool) -> dict[str, Any]:
    tool_id = block.get("tool_use_id")
    if not isinstance(tool_id, str) or not tool_id:
        raise RequestTranslationError("tool_result block requires a non-empty tool_use_id")
    raw_content = block.get("content", "")
    content = _translate_tool_result_content(raw_content, supports_images)
    return {"role": "tool", "tool_call_id": tool_id, "content": content}


def _translate_tool_result_content(raw_content: Any, supports_images: bool) -> Any:
    if isinstance(raw_content, str):
        return raw_content
    if not isinstance(raw_content, list):
        return str(raw_content)

    parts: list[dict[str, Any]] = []
    texts: list[str] = []
    for block in raw_content:
        if not isinstance(block, dict):
            texts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise RequestTranslationError("tool_result text block requires string text")
            texts.append(text)
        elif block_type == "image":
            parts.append(_translate_image(block, supports_images))
        else:
            raise RequestTranslationError(f"unsupported tool_result content block: {block_type!r}")
    if parts:
        if texts:
            parts.insert(0, {"type": "text", "text": "\n".join(texts)})
        return parts
    return "\n".join(texts)


def _translate_image(block: dict[str, Any], supports_images: bool) -> dict[str, Any]:
    if not supports_images:
        raise UnsupportedImageError("the selected model does not support image input")
    source = block.get("source")
    if not isinstance(source, dict):
        raise RequestTranslationError("image block requires a source object")
    source_type = source.get("type")
    url: str
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not isinstance(data, str):
            raise RequestTranslationError("base64 image source requires media_type and data")
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url":
        source_url = source.get("url")
        if not isinstance(source_url, str) or not source_url:
            raise RequestTranslationError("URL image source requires a non-empty url")
        url = source_url
    else:
        raise RequestTranslationError(f"unsupported image source type: {source_type!r}")
    return {"type": "image_url", "image_url": {"url": url}}


def _content_value(parts: list[dict[str, Any]]) -> Any:
    if len(parts) == 1 and parts[0].get("type") == "text":
        text = parts[0].get("text")
        return text if isinstance(text, str) else ""
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(str(part.get("text", "")) for part in parts)
    return list(parts)


def _text_from_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise RequestTranslationError("message content blocks must be objects")
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            raise RequestTranslationError("text block requires string text")
        parts.append(text)
    return "\n".join(parts)


def _translate_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        raise RequestTranslationError("tools must be an array")
    tools: list[dict[str, Any]] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            raise RequestTranslationError("tools must contain objects")
        name = raw_tool.get("name")
        schema = raw_tool.get("input_schema")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(schema, dict):
            # Anthropic server-side tools have no input_schema and are outside the MVP.
            continue
        function: dict[str, Any] = {"name": name, "parameters": schema}
        description = raw_tool.get("description")
        if isinstance(description, str):
            function["description"] = description
        tools.append({"type": "function", "function": function})
    return tools


def _translate_tool_choice(raw_choice: Any) -> Any:
    if raw_choice is None:
        return None
    if isinstance(raw_choice, str):
        if raw_choice in {"auto", "none", "required"}:
            return raw_choice
        raise RequestTranslationError(f"unsupported tool_choice: {raw_choice!r}")
    if not isinstance(raw_choice, dict):
        raise RequestTranslationError("tool_choice must be an object")
    choice_type = raw_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool":
        name = raw_choice.get("name")
        if not isinstance(name, str) or not name:
            raise RequestTranslationError("named tool_choice requires a name")
        return {"type": "function", "function": {"name": name}}
    raise RequestTranslationError(f"unsupported tool_choice type: {choice_type!r}")


def _translate_reasoning_effort(request: dict[str, Any]) -> str | None:
    output_config = request.get("output_config")
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if isinstance(effort, str) and effort in {"low", "medium", "high"}:
            return effort
    thinking = request.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        return "medium"
    return None
