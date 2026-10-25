from __future__ import annotations

import stat
from pathlib import Path

import pytest

from claude_nim_proxy.config import (
    DEFAULT_MODEL,
    ConfigError,
    UpstreamAuth,
    config_sources,
    load_config,
    set_config_value,
)


def test_missing_config_file_uses_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml", environ={})

    assert config.default_model == DEFAULT_MODEL
    assert config.verify_model == DEFAULT_MODEL
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


def test_model_overrides_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[model_overrides."nvidia/model"]
context_window = 200000
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown configuration key: model_overrides"):
        load_config(path, environ={})


def test_set_config_value_creates_a_private_config_file(tmp_path: Path) -> None:
    path = tmp_path / "claude-nim" / "config.toml"

    config = set_config_value(path, "default_model", "nvidia/my-model")

    assert config.default_model == "nvidia/my-model"
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text(encoding="utf-8") == 'default_model = "nvidia/my-model"\n'
    assert load_config(path, environ={}).default_model == "nvidia/my-model"


def test_set_config_value_preserves_existing_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('default_model = "nvidia/old"\ndebug = false\n', encoding="utf-8")

    set_config_value(path, "default_model", "nvidia/new")

    content = path.read_text(encoding="utf-8")
    assert 'default_model = "nvidia/new"' in content
    assert "debug = false" in content
    config = load_config(path, environ={})
    assert config.default_model == "nvidia/new"
    assert config.debug is False


def test_set_config_value_escapes_strings(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"

    set_config_value(path, "default_model", 'nvidia/weird "quoted" \\ model')

    assert load_config(path, environ={}).default_model == 'nvidia/weird "quoted" \\ model'


def test_set_config_value_rejects_an_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"

    with pytest.raises(ConfigError, match="unknown configuration key: unknown_key"):
        set_config_value(path, "unknown_key", "value")

    assert not path.exists()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("listen_port", "not-a-number", "listen_port must be an integer"),
        ("debug", "perhaps", "debug must be a boolean"),
        ("upstream_auth", "sometimes", "upstream_auth must be one of"),
        ("default_model", "", "default_model must be a non-empty string"),
    ],
)
def test_set_config_value_rejects_invalid_values(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    path = tmp_path / "config.toml"

    with pytest.raises(ConfigError, match=message):
        set_config_value(path, key, value)

    assert not path.exists()


def test_set_config_value_does_not_overwrite_an_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("listen_port = 'not-a-number'\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="listen_port must be an integer"):
        set_config_value(path, "default_model", "nvidia/my-model")

    assert path.read_text(encoding="utf-8") == "listen_port = 'not-a-number'\n"


def test_config_sources_report_the_effective_origin(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('default_model = "nvidia/file-model"\n', encoding="utf-8")

    sources = config_sources(path, environ={"NVIDIA_BASE_URL": "https://env.test/v1"})

    assert sources["default_model"] == "config.toml"
    assert sources["upstream_base_url"] == "NVIDIA_BASE_URL"
    assert sources["listen_port"] == "default"
