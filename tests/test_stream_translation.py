from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from claude_nim_proxy.translate.stream import SSEParser, translate_stream


async def _chunks(*values: bytes) -> AsyncIterator[bytes]:
    for value in values:
        yield value


async def _collect(*values: bytes) -> list[object]:
    return [message async for message in translate_stream(_chunks(*values), "nvidia/model")]


def _event(data: dict[str, object]) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


async def test_text_stream_has_exact_anthropic_event_order() -> None:
    messages = await _collect(
        _event({"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]}),
        _event({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    )

    assert [message.event for message in messages] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert messages[0].data["type"] == "message_start"
    assert messages[0].data["message"]["model"] == "nvidia/model"
    assert messages[1].data["index"] == 0
    assert messages[2].data["delta"] == {"type": "text_delta", "text": "Hel"}
    assert messages[3].data["delta"] == {"type": "text_delta", "text": "lo"}
    assert messages[5].data["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}
    assert messages[5].data["usage"] == {"output_tokens": 0}
    assert messages[6].data == {"type": "message_stop"}


async def test_stream_state_machine_handles_fragmented_frames_and_comments() -> None:
    messages = await _collect(
        b": keepalive comment\n",
        b'data: {"choices":[{"delta":{"content":"Hel',
        b'lo"},"finish_reason":null}]}\n',
        b"\n",
        b"data: [DONE]\n\n",
    )

    assert [message.event for message in messages] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert messages[2].data["delta"] == {"type": "text_delta", "text": "Hello"}


async def test_multi_line_data_is_joined_before_json_parsing() -> None:
    parser = SSEParser()
    frames = parser.feed(b"event: message\ndata: {\ndata:   \"choices\": []\ndata: }\n\n")
    assert len(frames) == 1
    assert frames[0].event == "message"
    assert json.loads(frames[0].data) == {"choices": []}


async def test_tool_stream_opens_blocks_and_emits_argument_fragments() -> None:
    messages = await _collect(
        _event(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {"name": "Read", "arguments": '{"file_'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _event(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'path":"/tmp/x"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        b"data: [DONE]\n\n",
    )

    assert [message.event for message in messages] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert messages[1].data["content_block"] == {
        "type": "tool_use",
        "id": "call_abc123",
        "name": "Read",
        "input": {},
    }
    assert messages[2].data["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"file_',
    }
    assert messages[3].data["delta"] == {
        "type": "input_json_delta",
        "partial_json": 'path":"/tmp/x"}',
    }
    assert messages[4].data["index"] == 0
    assert messages[5].data["delta"]["stop_reason"] == "tool_use"


async def test_text_and_parallel_tool_blocks_have_contiguous_indices() -> None:
    messages = await _collect(
        _event({"choices": [{"delta": {"content": "Working"}}]}),
        _event(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {"name": "Read", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        _event(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "function": {"name": "Write", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        b"data: [DONE]\n\n",
    )

    starts = [message for message in messages if message.event == "content_block_start"]
    stops = [message for message in messages if message.event == "content_block_stop"]
    assert [message.data["index"] for message in starts] == [0, 1, 2]
    assert [message.data["index"] for message in stops] == [0, 1, 2]


async def test_fatal_error_closes_open_block_and_is_terminal() -> None:
    messages = await _collect(
        _event({"choices": [{"delta": {"content": "partial"}}]}),
        b"event: error\ndata: {\"error\":{\"message\":\"secret upstream detail\"}}\n\n",
    )

    assert [message.event for message in messages] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "error",
    ]
    assert messages[-1].data == {
        "type": "error",
        "error": {"type": "api_error", "message": "NVIDIA NIM upstream error"},
    }


async def test_missing_finish_chunk_still_finishes_successfully() -> None:
    messages = await _collect(_event({"choices": [{"delta": {"content": "Hello"}}]}))

    assert [message.event for message in messages][-2:] == ["message_delta", "message_stop"]
    assert messages[-2].data["delta"]["stop_reason"] == "end_turn"


async def test_usage_only_chunk_updates_final_cumulative_output_tokens() -> None:
    messages = await _collect(
        _event({"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}]}),
        _event({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 7}}),
        b"data: [DONE]\n\n",
    )

    message_delta = next(message for message in messages if message.event == "message_delta")
    assert message_delta.data["usage"] == {"output_tokens": 7}


async def test_unknown_upstream_event_is_ignored() -> None:
    messages = await _collect(
        b"event: unknown.event\ndata: {\"choices\":[{\"delta\":{\"content\":\"hidden\"}}]}\n\n",
        _event({"choices": [{"delta": {"content": "visible"}}]}),
        b"data: [DONE]\n\n",
    )

    text = [
        message.data["delta"]["text"]
        for message in messages
        if message.event == "content_block_delta"
    ]
    assert text == ["visible"]


async def test_ping_is_emitted_while_upstream_is_quiet() -> None:
    async def slow_chunks() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.04)
        yield _event({"choices": [{"delta": {"content": "done"}}]})

    messages = [
        message
        async for message in translate_stream(
            slow_chunks(), "nvidia/model", ping_interval_seconds=0.005
        )
    ]

    assert any(message.event == "ping" for message in messages)
    assert messages[-1].event == "message_stop"


async def test_sse_message_encoding_has_matching_event_and_type() -> None:
    messages = await _collect(_event({"choices": [{"delta": {"content": "Hello"}}]}))
    encoded = messages[0].encode().decode()
    assert encoded.startswith("event: message_start\ndata: {")
    assert encoded.endswith("\n\n")
    payload = json.loads(encoded.split("data: ", 1)[1])
    assert payload["type"] == "message_start"
