"""Opt-in compatibility smoke tests using the installed Claude Code CLI.

These tests use a local fake NVIDIA upstream and never make a billable external request.
Enable them explicitly with ``CLAUDE_NIM_RUN_CLAUDE_COMPAT=1``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

pytestmark = pytest.mark.skipif(
    os.environ.get("CLAUDE_NIM_RUN_CLAUDE_COMPAT") != "1" or shutil.which("claude") is None,
    reason="set CLAUDE_NIM_RUN_CLAUDE_COMPAT=1 with Claude Code installed",
)

_IMAGE_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeNvidia:
    def __init__(self, scenario: str, fixture_path: Path | None = None) -> None:
        self.scenario = scenario
        self.fixture_path = fixture_path
        self.requests: list[dict[str, Any]] = []
        self.request_count = 0
        self.app = web.Application()
        self.app.router.add_get("/v1/models", self._models)
        self.app.router.add_post("/v1/chat/completions", self._chat)
        self.runner: web.AppRunner | None = None
        self.port: int | None = None

    async def start(self) -> None:
        self.runner = web.AppRunner(self.app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, host="127.0.0.1", port=0)
        await site.start()
        addresses = self.runner.addresses
        assert addresses
        self.port = int(addresses[0][1])

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None

    @property
    def base_url(self) -> str:
        assert self.port is not None
        return f"http://127.0.0.1:{self.port}/v1"

    async def _models(self, request: web.Request) -> web.StreamResponse:
        del request
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {"id": "nvidia/model", "object": "model"},
                    {
                        "id": "nvidia/nemotron-3.5-super-vl-120b-a12b",
                        "object": "model",
                    },
                ],
            }
        )

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.request_count += 1
        self.requests.append(body)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in self._chunks(body):
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    def _chunks(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        model = str(body.get("model"))
        has_tool_result = any(
            message.get("role") == "tool" for message in body.get("messages", [])
        )
        if self.scenario == "text" or has_tool_result:
            text = self._final_text()
            return self._text_chunks(model, text)
        if self.scenario in {"tool", "image"}:
            return self._tool_chunks(model)
        if self.scenario in {"subagent", "background"}:
            if self.request_count == 1:
                return self._agent_chunks(model, background=self.scenario == "background")
            if not has_tool_result:
                return self._text_chunks(model, "SUBAGENT_OK")
            return self._text_chunks(model, self._final_text())
        raise AssertionError(f"unknown scenario {self.scenario}")

    def _final_text(self) -> str:
        if self.scenario in {"subagent", "background"}:
            return "Parent received subagent completion: SUBAGENT_OK"
        return f"{self.scenario.upper()}_COMPATIBILITY_OK"

    def _tool_chunks(self, model: str) -> list[dict[str, Any]]:
        assert self.fixture_path is not None
        arguments = json.dumps({"file_path": str(self.fixture_path)})
        name = "Read"
        return [
            {
                "id": "chatcmpl-tool",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_compat_123",
                                    "type": "function",
                                    "function": {"name": name, "arguments": arguments},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-tool",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
            {
                "id": "chatcmpl-tool",
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        ]

    def _agent_chunks(self, model: str, *, background: bool) -> list[dict[str, Any]]:
        arguments = json.dumps(
            {
                "description": "Compatibility subagent check",
                "prompt": "Return exactly SUBAGENT_OK and do not use tools.",
                "model": "haiku",
                "run_in_background": background,
            }
        )
        return [
            {
                "id": "chatcmpl-agent",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_agent_123",
                                    "type": "function",
                                    "function": {"name": "Agent", "arguments": arguments},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-agent",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
            {
                "id": "chatcmpl-agent",
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        ]

    def _text_chunks(self, model: str, text: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "chatcmpl-text",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-text",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8},
            },
            {
                "id": "chatcmpl-text",
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8},
            },
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "model", "expected"),
    [
        ("text", "nvidia/model", "TEXT_COMPATIBILITY_OK"),
        ("tool", "nvidia/model", "TOOL_COMPATIBILITY_OK"),
        ("image", "nvidia/nemotron-3.5-super-vl-120b-a12b", "IMAGE_COMPATIBILITY_OK"),
        ("subagent", "nvidia/model", "Parent received subagent completion: SUBAGENT_OK"),
        ("background", "nvidia/model", "SUBAGENT_OK"),
    ],
)
async def test_installed_claude_code_compatibility(
    tmp_path: Path,
    scenario: str,
    model: str,
    expected: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    fixture_path: Path | None = None
    if scenario == "tool":
        fixture_path = tmp_path / "tool.txt"
        fixture_path.write_text("TOOL_COMPATIBILITY_OK\n", encoding="utf-8")
    elif scenario == "image":
        fixture_path = tmp_path / "image.png"
        fixture_path.write_bytes(_IMAGE_BYTES)

    upstream = FakeNvidia(scenario, fixture_path)
    await upstream.start()
    try:
        environment = os.environ.copy()
        environment["HOME"] = str(tmp_path / "home")
        environment["XDG_CONFIG_HOME"] = str(tmp_path / "config")
        environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        environment["DISABLE_AUTOUPDATER"] = "1"
        (tmp_path / "home").mkdir()
        prompt = {
            "text": "Reply with the requested greeting.",
            "tool": "Use Read to read the fixture, then report its exact marker.",
            "image": "Use Read to inspect the image, then report completion.",
            "subagent": "Delegate this check to a subagent with Agent, then report its result.",
            "background": (
                "Delegate this check to a background subagent with Agent, then report completion."
            ),
        }[scenario]
        completed = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "claude_nim_proxy.cli",
                "run",
                str(project),
                "--model",
                model,
                "--upstream-base-url",
                upstream.base_url,
                "--upstream-auth",
                "none",
                "--",
                "-p",
                prompt,
                "--output-format",
                "json",
                "--dangerously-skip-permissions",
            ],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    finally:
        await upstream.stop()

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["type"] == "result"
    assert result["is_error"] is False
    assert expected in result["result"]
    assert upstream.requests
    assert all(request["model"] == model for request in upstream.requests)
