from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from claude_nim_proxy.config import AppConfig
from claude_nim_proxy.models import ModelInfo
from claude_nim_proxy.runner import (
    create_temporary_settings,
    gateway_environment,
    launch_claude,
    remove_temporary_settings,
    sanitized_child_environment,
)


def _model() -> ModelInfo:
    return ModelInfo(
        upstream_id="nvidia/test-model",
        display_name="NVIDIA Test Model",
        context_window=131072,
        max_output_tokens=32768,
        reasoning=True,
        input=("text", "image"),
        tool_calling=True,
        source=("nvidia",),
        metadata_source="bundled",
    )


def test_gateway_environment_uses_reversible_alias_and_known_capabilities() -> None:
    values = gateway_environment(
        AppConfig(),
        base_url="http://127.0.0.1:54321",
        local_token="local-token",
        model=_model(),
    )

    assert values == {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:54321",
        "ANTHROPIC_AUTH_TOKEN": "local-token",
        "ANTHROPIC_MODEL": "anthropic-nim/nvidia/test-model",
        "ANTHROPIC_CUSTOM_MODEL_OPTION": "anthropic-nim/nvidia/test-model",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "anthropic-nim/nvidia/test-model",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "anthropic-nim/nvidia/test-model",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "anthropic-nim/nvidia/test-model",
        "ANTHROPIC_DEFAULT_FABLE_MODEL": "anthropic-nim/nvidia/test-model",
        "CLAUDE_CODE_SUBAGENT_MODEL": "anthropic-nim/nvidia/test-model",
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "131072",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    }


def test_child_environment_removes_inherited_provider_credentials() -> None:
    parent = {
        "PATH": "/usr/bin",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "ANTHROPIC_AUTH_TOKEN": "old-token",
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-secret",
        "NVIDIA_API_KEY": "nvidia-secret",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
    }
    values = gateway_environment(
        AppConfig(),
        base_url="http://127.0.0.1:54321",
        local_token="local-token",
        model=_model(),
    )

    child = sanitized_child_environment(parent, values)

    assert child["ANTHROPIC_AUTH_TOKEN"] == "local-token"
    assert child["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:54321"
    assert "ANTHROPIC_API_KEY" not in child
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in child
    assert "NVIDIA_API_KEY" not in child
    assert "nvidia-secret" not in child.values()


def test_temporary_settings_has_exact_shape_and_mode_0600() -> None:
    values = gateway_environment(
        AppConfig(),
        base_url="http://127.0.0.1:54321",
        local_token="local-token",
        model=_model(),
    )

    path = create_temporary_settings(values)
    try:
        assert path.is_absolute()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert json.loads(path.read_text(encoding="utf-8")) == {"env": values}
    finally:
        remove_temporary_settings(path)
    assert not path.exists()


@pytest.mark.asyncio
async def test_launch_claude_uses_settings_contract_and_propagates_exit_code(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture.json"
    executable = tmp_path / "fake-claude"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
with open(os.environ['CAPTURE_PATH'], 'w', encoding='utf-8') as output:
    json.dump({
        'argv': sys.argv[1:],
        'cwd': os.getcwd(),
        'env': dict(os.environ),
        'settings': json.load(open(sys.argv[2], encoding='utf-8')),
    }, output)
raise SystemExit(int(os.environ.get('FAKE_EXIT_CODE', '0')))
""",
        encoding="utf-8",
    )
    os.chmod(executable, 0o700)
    settings = create_temporary_settings({"ANTHROPIC_BASE_URL": "http://127.0.0.1:1"})
    environment = sanitized_child_environment(
        {
            **os.environ,
            "CAPTURE_PATH": str(capture),
            "FAKE_EXIT_CODE": "7",
            "NVIDIA_API_KEY": "must-not-be-inherited",
        },
        {},
    )
    try:
        status = await launch_claude(
            project_dir=tmp_path,
            claude_arguments=["--resume", "value with spaces"],
            environment=environment,
            settings_path=settings,
            executable=str(executable),
        )
    finally:
        remove_temporary_settings(settings)

    assert status == 7
    payload = json.loads(capture.read_text(encoding="utf-8"))
    assert payload["argv"] == [
        "--settings",
        str(settings),
        "--resume",
        "value with spaces",
    ]
    assert payload["cwd"] == str(tmp_path)
    assert payload["settings"] == {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}}
    assert "NVIDIA_API_KEY" not in payload["env"]
