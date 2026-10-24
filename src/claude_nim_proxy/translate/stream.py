"""Incremental OpenAI SSE to Anthropic SSE translation state machine."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any

from .openai_to_anthropic import map_stop_reason


@dataclass(frozen=True, slots=True)
class SSEFrame:
    """A parsed upstream Server-Sent Events frame."""

    event: str | None
    data: str


@dataclass(frozen=True, slots=True)
class SSEMessage:
    """A client-facing Anthropic SSE event."""

    event: str
    data: dict[str, Any]

    def encode(self) -> bytes:
        """Encode this event using Anthropic's SSE wire format."""

        payload = json.dumps(self.data, separators=(",", ":"), ensure_ascii=False)
        return f"event: {self.event}\ndata: {payload}\n\n".encode()


class SSEParser:
    """Incrementally parse SSE frames, including fragmented and multi-line data."""

    def __init__(self) -> None:
        self._buffer = ""
        self._event: str | None = None
        self._data: list[str] = []

    def feed(self, chunk: bytes | str) -> list[SSEFrame]:
        """Consume a byte or text chunk and return all newly completed frames."""

        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        self._buffer += text

        frames: list[SSEFrame] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            frame = self._consume_line(line.rstrip("\r"))
            if frame is not None:
                frames.append(frame)
        return frames

    def flush(self) -> list[SSEFrame]:
        """Finish parsing and dispatch a final frame that had no blank-line terminator."""

        frames: list[SSEFrame] = []
        if self._buffer:
            line = self._buffer
            self._buffer = ""
            frame = self._consume_line(line.rstrip("\r"))
            if frame is not None:
                frames.append(frame)
        frame = self._dispatch()
        if frame is not None:
            frames.append(frame)
        return frames

    def _consume_line(self, line: str) -> SSEFrame | None:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if not separator:
            field, value = line, ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        return None

    def _dispatch(self) -> SSEFrame | None:
        if not self._data:
            self._event = None
            return None
        frame = SSEFrame(event=self._event, data="\n".join(self._data))
        self._event = None
        self._data = []
        return frame


@dataclass(slots=True)
class _ToolState:
    upstream_index: int
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""
    arguments_emitted: int = 0
    anthropic_index: int | None = None


class OpenAIStreamTranslator:
    """Stateful translator for one upstream OpenAI-compatible stream."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.started = False
        self.finished = False
        self.text_index: int | None = None
        self.next_block_index = 0
        self.stop_reason = "end_turn"
        self.output_tokens = 0
        self._tools: dict[int, _ToolState] = {}

    def start(self) -> list[SSEMessage]:
        """Start the Anthropic message and return the message_start event."""

        if self.started:
            return []
        self.started = True
        return [
            SSEMessage(
                event="message_start",
                data={
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{uuid.uuid4().hex}",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        ]

    def handle_frame(self, frame: SSEFrame) -> list[SSEMessage]:
        """Translate one upstream SSE frame into zero or more Anthropic events."""

        if self.finished:
            return []
        if frame.data.strip() == "[DONE]":
            return self.finalize()
        if frame.event not in {
            None,
            "",
            "message",
            "completion",
            "chat.completion.chunk",
            "error",
        }:
            return []
        if frame.event == "error":
            return self.fail()

        try:
            payload = json.loads(frame.data)
        except json.JSONDecodeError:
            return self.fail()
        if not isinstance(payload, dict):
            return self.fail()
        if "error" in payload:
            return self.fail()

        messages = self.start()
        usage = payload.get("usage")
        if isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens")
            if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
                self.output_tokens = completion_tokens

        choices = payload.get("choices")
        if not isinstance(choices, list):
            return messages
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                self.stop_reason = map_stop_reason(finish_reason)
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = choice.get("message")
            if not isinstance(delta, dict):
                continue
            messages.extend(self._translate_delta(delta))
        return messages

    def finalize(self) -> list[SSEMessage]:
        """Finalize a successful stream, even if the upstream omitted a finish chunk."""

        if self.finished:
            return []
        if not self.started:
            self.start()
        self.finished = True

        messages: list[SSEMessage] = []
        if self.text_index is not None:
            messages.append(
                SSEMessage(
                    event="content_block_stop",
                    data={"type": "content_block_stop", "index": self.text_index},
                )
            )
            self.text_index = None

        opened_tools: list[_ToolState] = []
        for tool in sorted(self._tools.values(), key=lambda item: item.upstream_index):
            if tool.anthropic_index is None:
                if not tool.name:
                    continue
                messages.extend(self._open_tool(tool))
            if len(tool.arguments) > tool.arguments_emitted:
                messages.append(self._tool_argument_delta(tool))
            opened_tools.append(tool)

        for tool in opened_tools:
            if tool.anthropic_index is None:
                continue
            messages.append(
                SSEMessage(
                    event="content_block_stop",
                    data={"type": "content_block_stop", "index": tool.anthropic_index},
                )
            )

        if any(tool.name for tool in self._tools.values()):
            self.stop_reason = "tool_use"
        messages.append(
            SSEMessage(
                event="message_delta",
                data={
                    "type": "message_delta",
                    "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": self.output_tokens},
                },
            )
        )
        messages.append(SSEMessage(event="message_stop", data={"type": "message_stop"}))
        return messages

    def fail(self) -> list[SSEMessage]:
        """Close open blocks and emit a terminal Anthropic error event."""

        if self.finished:
            return []
        if not self.started:
            self.start()
        self.finished = True
        messages: list[SSEMessage] = []
        if self.text_index is not None:
            messages.append(
                SSEMessage(
                    event="content_block_stop",
                    data={"type": "content_block_stop", "index": self.text_index},
                )
            )
            self.text_index = None
        for tool in sorted(self._tools.values(), key=lambda item: item.upstream_index):
            if tool.anthropic_index is not None:
                messages.append(
                    SSEMessage(
                        event="content_block_stop",
                        data={"type": "content_block_stop", "index": tool.anthropic_index},
                    )
                )
        messages.append(
            SSEMessage(
                event="error",
                data={
                    "type": "error",
                    "error": {"type": "api_error", "message": "NVIDIA NIM upstream error"},
                },
            )
        )
        return messages

    def _translate_delta(self, delta: dict[str, Any]) -> list[SSEMessage]:
        messages: list[SSEMessage] = []
        content = delta.get("content")
        if isinstance(content, str) and content:
            if self.text_index is None:
                index = self.next_block_index
                self.next_block_index += 1
                self.text_index = index
                messages.append(
                    SSEMessage(
                        event="content_block_start",
                        data={
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                )
            messages.append(
                SSEMessage(
                    event="content_block_delta",
                    data={
                        "type": "content_block_delta",
                        "index": self.text_index,
                        "delta": {"type": "text_delta", "text": content},
                    },
                )
            )

        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            return messages
        for raw_tool_call in tool_calls:
            if not isinstance(raw_tool_call, dict):
                continue
            upstream_index = raw_tool_call.get("index", 0)
            if not isinstance(upstream_index, int) or isinstance(upstream_index, bool):
                upstream_index = 0
            tool = self._tools.setdefault(upstream_index, _ToolState(upstream_index=upstream_index))

            call_id = raw_tool_call.get("id")
            if isinstance(call_id, str) and call_id:
                tool.call_id = call_id
            function = raw_tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                tool.name = name
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                tool.arguments += arguments

            if tool.anthropic_index is None and tool.call_id and tool.name:
                messages.extend(self._open_tool(tool))
                if len(tool.arguments) > tool.arguments_emitted:
                    messages.append(self._tool_argument_delta(tool))
            elif tool.anthropic_index is not None and len(tool.arguments) > tool.arguments_emitted:
                messages.append(self._tool_argument_delta(tool))
        return messages

    def _open_tool(self, tool: _ToolState) -> list[SSEMessage]:
        if tool.anthropic_index is not None:
            return []
        if not tool.name:
            return []
        index = self.next_block_index
        self.next_block_index += 1
        tool.anthropic_index = index
        call_id = tool.call_id or f"toolu_{uuid.uuid4().hex}"
        return [
            SSEMessage(
                event="content_block_start",
                data={
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call_id,
                        "name": tool.name,
                        "input": {},
                    },
                },
            )
        ]

    def _tool_argument_delta(self, tool: _ToolState) -> SSEMessage:
        if tool.anthropic_index is None:
            raise RuntimeError("tool block is not open")
        arguments = tool.arguments[tool.arguments_emitted :]
        tool.arguments_emitted = len(tool.arguments)
        return SSEMessage(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": tool.anthropic_index,
                "delta": {"type": "input_json_delta", "partial_json": arguments},
            },
        )


async def _next_chunk(iterator: AsyncIterator[bytes]) -> bytes:
    return await iterator.__anext__()


async def translate_stream(
    chunks: AsyncIterable[bytes],
    model: str,
    *,
    ping_interval_seconds: float = 10.0,
) -> AsyncIterator[SSEMessage]:
    """Translate an OpenAI SSE byte stream into Anthropic SSE messages.

    The upstream iterator is consumed incrementally. While waiting for upstream bytes,
    Anthropic ping events are emitted so Claude Code does not mistake a quiet generation
    for a disconnected stream.
    """

    translator = OpenAIStreamTranslator(model)
    parser = SSEParser()
    for message in translator.start():
        yield message

    iterator = chunks.__aiter__()
    pending: asyncio.Task[bytes] | None = None
    try:
        while not translator.finished:
            if pending is None:
                pending = asyncio.create_task(_next_chunk(iterator))
            if ping_interval_seconds > 0:
                done, _ = await asyncio.wait((pending,), timeout=ping_interval_seconds)
                if not done:
                    yield SSEMessage(event="ping", data={"type": "ping"})
                    continue
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                break
            finally:
                pending = None

            for frame in parser.feed(chunk):
                for message in translator.handle_frame(frame):
                    yield message
                if translator.finished:
                    break
            if translator.finished:
                break

        if not translator.finished:
            for frame in parser.flush():
                for message in translator.handle_frame(frame):
                    yield message
                if translator.finished:
                    break
            if not translator.finished:
                for message in translator.finalize():
                    yield message
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
