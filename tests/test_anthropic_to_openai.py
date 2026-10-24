from __future__ import annotations

import json

import pytest

from claude_nim_proxy.translate.anthropic_to_openai import (
    RequestTranslationError,
    RequestTranslationOptions,
    UnsupportedImageError,
    resolve_upstream_model,
    translate_request,
)


def test_text_request_translation_preserves_model_system_and_sampling() -> None:
    translated = translate_request(
        {
            "model": "anthropic-nim/nvidia/nemotron-3-super-120b-a12b",
            "system": [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            "max_tokens": 123,
            "temperature": 0.2,
            "top_p": 0.8,
            "stop_sequences": ["END", "STOP"],
            "metadata": {"user_id": "not-forwarded"},
            "service_tier": "not-forwarded",
        }
    )

    assert translated == {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {"role": "system", "content": "First\nSecond"},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 123,
        "temperature": 0.2,
        "top_p": 0.8,
        "stop": ["END", "STOP"],
    }


def test_mid_conversation_system_message_becomes_system_reminder() -> None:
    translated = translate_request(
        {
            "model": "nvidia/model",
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "system", "content": "Important constraint"},
                {"role": "assistant", "content": "Acknowledged"},
            ],
        }
    )

    assert translated["messages"] == [
        {"role": "user", "content": "First"},
        {"role": "user", "content": "[System reminder] Important constraint"},
        {"role": "assistant", "content": "Acknowledged"},
    ]


def test_images_are_converted_to_openai_image_url_parts() -> None:
    translated = translate_request(
        {
            "model": "nvidia/vision-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
                        },
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.test/image.jpg"},
                        },
                    ],
                }
            ],
        }
    )

    content = translated["messages"][0]["content"]
    assert content == [
        {"type": "text", "text": "Compare"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "image_url", "image_url": {"url": "https://example.test/image.jpg"}},
    ]


def test_image_is_rejected_for_text_only_model() -> None:
    with pytest.raises(UnsupportedImageError, match="does not support image"):
        translate_request(
            {
                "model": "nvidia/text-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://example.test/image.png"},
                            }
                        ],
                    }
                ],
            },
            RequestTranslationOptions(supports_images=False),
        )


def test_tools_and_all_tool_choice_modes_are_translated() -> None:
    tools = [
        {
            "name": "Read",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        }
    ]

    auto = translate_request(
        {"model": "nvidia/model", "messages": [], "tools": tools, "tool_choice": {"type": "auto"}}
    )
    required = translate_request(
        {"model": "nvidia/model", "messages": [], "tools": tools, "tool_choice": {"type": "any"}}
    )
    named = translate_request(
        {
            "model": "nvidia/model",
            "messages": [],
            "tools": tools,
            "tool_choice": {"type": "tool", "name": "Read"},
        }
    )
    none = translate_request(
        {
            "model": "nvidia/model",
            "messages": [],
            "tools": tools,
            "tool_choice": {"type": "none"},
            "disable_parallel_tool_use": True,
        }
    )

    assert auto["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a file",
                "parameters": tools[0]["input_schema"],
            },
        }
    ]
    assert auto["tool_choice"] == "auto"
    assert required["tool_choice"] == "required"
    assert named["tool_choice"] == {"type": "function", "function": {"name": "Read"}}
    assert none["tool_choice"] == "none"
    assert none["parallel_tool_calls"] is False


def test_assistant_tool_use_and_user_tool_results_round_trip_ids() -> None:
    translated = translate_request(
        {
            "model": "nvidia/model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I will read it."},
                        {
                            "type": "tool_use",
                            "id": "call_abc123",
                            "name": "Read",
                            "input": {"file_path": "/tmp/example"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "The result is:"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_abc123",
                            "content": [{"type": "text", "text": "file contents"}],
                        },
                        {"type": "text", "text": "Continue."},
                    ],
                },
            ],
        }
    )

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": "I will read it.",
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
        {"role": "user", "content": "The result is:"},
        {"role": "tool", "tool_call_id": "call_abc123", "content": "file contents"},
        {"role": "user", "content": "Continue."},
    ]


def test_builtin_server_tools_are_omitted() -> None:
    translated = translate_request(
        {
            "model": "nvidia/model",
            "messages": [],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ],
        }
    )

    assert [tool["function"]["name"] for tool in translated["tools"]] == ["Read"]


def test_streaming_adds_usage_option_and_beta_only_objects_are_dropped() -> None:
    translated = translate_request(
        {
            "model": "nvidia/model",
            "messages": [],
            "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "output_config": {"effort": "high"},
        }
    )

    assert translated["stream"] is True
    assert translated["stream_options"] == {"include_usage": True}
    assert translated["reasoning_effort"] == "high"
    assert "thinking" not in translated
    assert "output_config" not in translated


@pytest.mark.parametrize(
    ("requested", "default", "expected"),
    [
        ("anthropic-nim/nvidia/model", None, "nvidia/model"),
        ("nvidia/model", None, "nvidia/model"),
        ("claude-sonnet-4-5", "nvidia/model", "nvidia/model"),
        ("sonnet", "anthropic-nim/nvidia/other", "nvidia/other"),
    ],
)
def test_model_resolution(requested: str, default: str | None, expected: str) -> None:
    assert resolve_upstream_model(requested, default) == expected


def test_claude_fallback_requires_explicit_default_model() -> None:
    with pytest.raises(RequestTranslationError, match="explicitly selected"):
        resolve_upstream_model("claude-sonnet-4-5")


def test_tool_input_is_compact_json() -> None:
    translated = translate_request(
        {
            "model": "nvidia/model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Read",
                            "input": {"path": "/tmp/a"},
                        }
                    ],
                }
            ],
        }
    )
    arguments = translated["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"path": "/tmp/a"}
    assert arguments == '{"path":"/tmp/a"}'
