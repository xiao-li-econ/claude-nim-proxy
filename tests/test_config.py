from __future__ import annotations

from pathlib import Path

import pytest

from claude_nim_proxy.config import ConfigError, UpstreamAuth, load_config


def test_missing_config_file_uses_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml", environ={})

    assert config.default_model is None
    assert config.upstream_base_url == "https://integrate.api.nvidia.com/v1"
    assert config.upstream_auth is UpstreamAuth.AUTO
    assert config.disable_experimental_betas is True


def test_config_file_is_parsed_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
default_model = "nvidia/model"
upstream_base_url = "https://example.test/v1"
upstream_auth = "none"
listen_port = 4321
context_window = 131072
disable_experimental_betas = false

[model_overrides."nvidia/model"]
display_name = "Custom model"
context_window = 200000
reasoning = true
input = ["text", "image"]
""",
        encoding="utf-8",
    )

    config = load_config(path, environ={})

    assert config.default_model == "nvidia/model"
    assert config.verify_model == "nvidia/model"
    assert config.upstream_auth is UpstreamAuth.NONE
    assert config.listen_port == 4321
    assert config.context_window == 131072
    assert config.disable_experimental_betas is False
    override = config.model_overrides["nvidia/model"]
    assert override.display_name == "Custom model"
    assert override.context_window == 200000
    assert override.reasoning is True
    assert override.input == ("text", "image")
    assert override.tool_calling is None


def test_environment_overrides_config(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
default_model = "nvidia/config-model"
upstream_base_url = "https://config.test/v1"
upstream_auth = "required"
listen_port = 1000
debug = false
""",
        encoding="utf-8",
    )

    config = load_config(
        path,
        environ={
            "NVIDIA_MODEL": "nvidia/environment-model",
            "NVIDIA_BASE_URL": "https://environment.test/v1",
            "CLAUDE_NIM_UPSTREAM_AUTH": "none",
            "CLAUDE_NIM_LISTEN_PORT": "2000",
            "CLAUDE_NIM_DEBUG": "yes",
        },
    )

    assert config.default_model == "nvidia/environment-model"
    assert config.upstream_base_url == "https://environment.test/v1"
    assert config.upstream_auth is UpstreamAuth.NONE
    assert config.listen_port == 2000
    assert config.debug is True


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("unknown_key = true\n", "unknown configuration key: unknown_key"),
        ("upstream_auth = 'sometimes'\n", "upstream_auth must be one of"),
        ("listen_port = '8080'\n", "listen_port must be an integer"),
        ("context_window = 0\n", "context_window must be positive"),
        ("debug = 'perhaps'\n", "debug must be a boolean"),
        (
            "upstream_base_url = 'https://user:secret@example.test/v1'\n",
            "must not contain credentials",
        ),
        ("upstream_base_url = 'not-a-url'\n", "must be a valid HTTP"),
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path, environ={})


def test_unknown_model_override_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[model_overrides."nvidia/model"]
unknown = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="model_overrides.nvidia/model.unknown"):
        load_config(path, environ={})
