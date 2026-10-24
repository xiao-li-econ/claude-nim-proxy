from __future__ import annotations

import json
import re

import pytest

from claude_nim_proxy.translate.openai_to_anthropic import (
    ResponseTranslationError,
    translate_response,
)


def test_text_response_has_exact_anthropic_envelope() -> None:
    response = translate_response(
        {
            "id": "chatcmpl-upstream-id",
            "model": "nvidia/upstream-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
        "nvidia/fallback-model",
    )

    assert re.fullmatch(r"msg_[0-9a-f]{32}", response["id"])
    assert response == {
        "id": response["id"],
        "type": "message",
        "role": "assistant",
        "model": "nvidia/upstream-model",
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


def test_tool_call_id_is_preserved_and_arguments_become_an_object() -> None:
    response = translate_response(
        {
            "model": "nvidia/model",
            "choices": [
                {
                    "message": {
                        "content": "I will inspect the file.",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"file_path":"/tmp/example"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        "nvidia/model",
    )

    assert response["content"] == [
        {"type": "text", "text": "I will inspect the file."},
        {
            "type": "tool_use",
            "id": "call_abc123",
            "name": "Read",
            "input": {"file_path": "/tmp/example"},
        },
    ]
    assert response["stop_reason"] == "tool_use"


def test_legacy_function_call_is_translated() -> None:
    response = translate_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "function_call": {"name": "Read", "arguments": '{"path":"x"}'},
                    },
                    "finish_reason": "function_call",
                }
            ]
        },
        "nvidia/model",
    )

    content = response["content"]
    assert len(content) == 1
    assert content[0]["type"] == "tool_use"
    assert content[0]["name"] == "Read"
    assert content[0]["input"] == {"path": "x"}
    assert re.fullmatch(r"toolu_[0-9a-f]{32}", content[0]["id"])


def test_missing_usage_values_are_zero_and_fallback_model_is_used() -> None:
    response = translate_response(
        {"choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]},
        "nvidia/fallback",
    )

    assert response["model"] == "nvidia/fallback"
    assert response["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_cached_prompt_tokens_are_reported() -> None:
    response = translate_response(
        {
            "choices": [{"message": {"content": "Hi"}}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        },
        "nvidia/model",
    )

    assert response["usage"]["cache_read_input_tokens"] == 8


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("function_call", "tool_use"),
        (None, "end_turn"),
    ],
)
def test_stop_reason_mapping(finish_reason: str | None, expected: str) -> None:
    response = translate_response(
        {
            "choices": [
                {"message": {"content": "Hi"}, "finish_reason": finish_reason},
            ]
        },
        "nvidia/model",
    )
    assert response["stop_reason"] == expected


def test_invalid_tool_arguments_are_rejected() -> None:
    with pytest.raises(ResponseTranslationError, match="valid JSON"):
        translate_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"name": "Read", "arguments": "not-json"},
                                }
                            ]
                        }
                    }
                ]
            },
            "nvidia/model",
        )


def test_tool_arguments_may_be_an_object() -> None:
    response = translate_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "Read", "arguments": {"path": "x"}},
                            }
                        ]
                    }
                }
            ]
        },
        "nvidia/model",
    )

    assert response["content"][0]["input"] == {"path": "x"}
    assert json.dumps(response["content"][0]["input"]) == '{"path": "x"}'
