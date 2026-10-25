from __future__ import annotations

import asyncio
import io
import json
import os
import re
import signal
import stat
import sys
from pathlib import Path

import pytest

from claude_nim_proxy.cli import _resolve_local_token, build_parser, main
from claude_nim_proxy.config import DEFAULT_MODEL
from claude_nim_proxy.models import load_model_catalog
from claude_nim_proxy.pi_models import PiModel, PiModelListError


def test_cli_exposes_the_planned_command_tree() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    for command in ("run", "serve", "auth", "models", "config", "doctor"):
        assert command in help_text


def test_run_passes_claude_arguments_verbatim_and_cleans_up(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    capture = tmp_path / "capture.json"
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    executable = executable_dir / "claude"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
with open(os.environ['CAPTURE_PATH'], 'w', encoding='utf-8') as output:
    json.dump({'argv': sys.argv[1:], 'env': dict(os.environ)}, output)
""",
        encoding="utf-8",
    )
    os.chmod(executable, 0o700)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("PATH", f"{executable_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CAPTURE_PATH", str(capture))
    monkeypatch.setenv("NVIDIA_API_KEY", "must-not-reach-claude")

    status = main(
        [
            "run",
            str(project),
            "--model",
            "nvidia/model",
            "--upstream-auth",
            "none",
            "--",
            "--resume",
            "--permission-mode",
            "acceptEdits",
        ]
    )

    assert status == 0
    assert capsys.readouterr().err == ""
    payload = json.loads(capture.read_text(encoding="utf-8"))
    assert payload["argv"][0:2] == ["--settings", payload["argv"][1]]
    assert payload["argv"][2:] == ["--resume", "--permission-mode", "acceptEdits"]
    settings_path = Path(payload["argv"][1])
    assert not settings_path.exists()
    assert "NVIDIA_API_KEY" not in payload["env"]
    assert payload["env"]["ANTHROPIC_MODEL"] == "anthropic-nim/nvidia/model"


def _capture_run_model(monkeypatch) -> list[str]:
    captured: list[str] = []

    async def fake_gateway(config, selected_model_id, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(selected_model_id)
        return 0

    monkeypatch.setattr("claude_nim_proxy.cli._run_gateway_and_claude", fake_gateway)
    monkeypatch.setattr(
        "claude_nim_proxy.cli.resolve_claude_executable",
        lambda: "/usr/local/bin/claude",
    )
    return captured


def test_run_uses_the_shipped_default_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    captured = _capture_run_model(monkeypatch)

    status = main(["run", str(tmp_path), "--upstream-auth", "none"])

    assert status == 0
    assert captured == [DEFAULT_MODEL]


def test_run_uses_the_configured_default_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    config_path = tmp_path / "claude-nim" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('default_model = "nvidia/configured-model"\n', encoding="utf-8")
    captured = _capture_run_model(monkeypatch)

    status = main(["run", str(tmp_path), "--upstream-auth", "none"])

    assert status == 0
    assert captured == ["nvidia/configured-model"]


def test_run_model_flag_wins_over_the_configured_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    config_path = tmp_path / "claude-nim" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('default_model = "nvidia/configured-model"\n', encoding="utf-8")
    captured = _capture_run_model(monkeypatch)

    status = main(
        [
            "run",
            str(tmp_path),
            "--model",
            "nvidia/flag-model",
            "--upstream-auth",
            "none",
        ]
    )

    assert status == 0
    assert captured == ["nvidia/flag-model"]


def test_run_refresh_models_warns_and_continues_when_pi_is_unavailable(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    captured = _capture_run_model(monkeypatch)

    def fail() -> tuple[PiModel, ...]:
        raise PiModelListError("Pi command not found")

    monkeypatch.setattr("claude_nim_proxy.cli.run_pi_model_list", fail)

    status = main(["run", str(tmp_path), "--refresh-models", "--upstream-auth", "none"])

    assert status == 0
    error = capsys.readouterr().err
    assert "WARNING: Pi model refresh failed" in error
    assert "using the existing active model JSON" in error
    assert captured == [DEFAULT_MODEL]


def test_run_refresh_models_uses_pi_catalog_before_starting(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    pi_models = (
        PiModel(
            upstream_id="nvidia/refreshed-model",
            display_name="refreshed-model (NIM)",
            context_window=131_072,
            max_output_tokens=16_384,
            reasoning=True,
            input=("text",),
        ),
    )
    monkeypatch.setattr("claude_nim_proxy.cli.run_pi_model_list", lambda: pi_models)
    captured_models = _capture_run_model(monkeypatch)

    status = main(["run", str(tmp_path), "--refresh-models", "--upstream-auth", "none"])

    assert status == 0
    captured = capsys.readouterr()
    assert "Refreshed active model JSON from Pi: 1 models" in captured.out
    catalog = load_model_catalog(tmp_path / "claude-nim" / "models.json")
    assert catalog is not None
    assert catalog.source == "pi"
    assert [model.upstream_id for model in catalog.models] == ["nvidia/refreshed-model"]
    assert captured_models == [DEFAULT_MODEL]


def test_auth_status_does_not_reveal_a_secret(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-value")

    status = main(["auth", "status"])

    assert status == 0
    output = capsys.readouterr().err
    assert "configured (environment)" in output
    assert "secret-value" not in output


def test_auth_login_uses_a_private_fallback_when_no_os_keyring_exists(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("getpass.getpass", lambda prompt: "nvapi-private-value")

    status = main(["auth", "login"])

    assert status == 0
    output = capsys.readouterr().err
    assert "file fallback" in output
    assert "nvapi-private-value" not in output
    credential_path = tmp_path / "claude-nim" / "credentials.json"
    assert stat.S_IMODE(credential_path.stat().st_mode) == 0o600
    assert "nvapi-private-value" in credential_path.read_text(encoding="utf-8")

    assert main(["auth", "login"]) == 2
    assert main(["auth", "login", "--replace"]) == 0


def test_auth_logout_does_not_remove_environment_credential(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("NVIDIA_API_KEY", "environment-value")

    status = main(["auth", "logout"])

    assert status == 0
    output = capsys.readouterr().err
    assert "No stored NVIDIA credential was found" in output
    assert "environment variable was not changed" in output
    assert "environment-value" not in output


@pytest.mark.asyncio
async def test_doctor_checks_claude_credential_and_entitlement(
    fake_upstream, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "doctor-secret")
    monkeypatch.setattr(
        "claude_nim_proxy.cli.resolve_claude_executable",
        lambda: "/usr/local/bin/claude",
    )

    status = await asyncio.to_thread(
        main,
        [
            "doctor",
            "--model",
            "nvidia/model",
            "--upstream-base-url",
            fake_upstream.base_url,
        ],
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "PASS: Claude Code" in output
    assert "PASS: Credential" in output
    assert "PASS: Listener" in output
    assert "PASS: NVIDIA upstream" in output
    assert "doctor-secret" not in output
    assert len(fake_upstream.requests) == 1


@pytest.mark.asyncio
async def test_auth_verify_sends_one_authenticated_minimal_request(
    fake_upstream, monkeypatch, capsys, tmp_path: Path
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "verify-secret")
    config_path = tmp_path / "config.toml"
    config_path.write_text("upstream_auth = 'none'\n", encoding="utf-8")

    status = await asyncio.to_thread(
        main,
        [
            "auth",
            "verify",
            "--config",
            str(config_path),
            "--model",
            "nvidia/model",
            "--upstream-base-url",
            fake_upstream.base_url,
        ],
    )

    assert status == 0
    output = capsys.readouterr().err
    assert "may consume a small amount of API credit" in output
    assert "verified successfully" in output
    assert "verify-secret" not in output
    assert len(fake_upstream.requests) == 1
    request = fake_upstream.requests[0]
    assert request["headers"]["Authorization"] == "Bearer verify-secret"
    assert request["body"]["max_tokens"] == 1
    assert request["body"]["stream"] is False


@pytest.mark.asyncio
async def test_run_uses_the_same_local_token_for_gateway_and_claude(
    fake_upstream, tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    capture = tmp_path / "response.json"
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    executable = executable_dir / "claude"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
import urllib.request
settings = json.load(open(sys.argv[2], encoding='utf-8'))['env']
payload = json.dumps({
    'model': 'claude-sonnet-4-5',
    'max_tokens': 32,
    'messages': [{'role': 'user', 'content': 'Hello'}],
}).encode()
request = urllib.request.Request(
    settings['ANTHROPIC_BASE_URL'] + '/v1/messages',
    data=payload,
    headers={
        'Authorization': 'Bearer ' + settings['ANTHROPIC_AUTH_TOKEN'],
        'Content-Type': 'application/json',
    },
)
with urllib.request.urlopen(request, timeout=5) as response:
    body = response.read().decode()
with open(os.environ['CAPTURE_PATH'], 'w', encoding='utf-8') as output:
    json.dump({'status': response.status, 'body': body}, output)
""",
        encoding="utf-8",
    )
    os.chmod(executable, 0o700)
    monkeypatch.setenv("PATH", f"{executable_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CAPTURE_PATH", str(capture))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    status = await asyncio.to_thread(
        main,
        [
            "run",
            str(project),
            "--model",
            "nvidia/model",
            "--upstream-base-url",
            fake_upstream.base_url,
            "--upstream-auth",
            "none",
        ],
    )

    assert status == 0
    payload = json.loads(capture.read_text(encoding="utf-8"))
    assert payload["status"] == 200
    assert json.loads(payload["body"])["content"][0]["text"] == "Hello from fake"


@pytest.mark.asyncio
async def test_run_forwards_termination_and_leaves_no_orphan(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    pid_file = tmp_path / "claude.pid"
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    executable = executable_dir / "claude"
    executable.write_text(
        """#!/bin/sh
printf '%s' "$$" > "$CLAUDE_PID_FILE"
sleep 30
""",
        encoding="utf-8",
    )
    os.chmod(executable, 0o700)
    environment = os.environ.copy()
    environment["PATH"] = f"{executable_dir}{os.pathsep}{environment['PATH']}"
    environment["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    environment["CLAUDE_PID_FILE"] = str(pid_file)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "claude_nim_proxy.cli",
        "run",
        str(project),
        "--model",
        "nvidia/model",
        "--upstream-auth",
        "none",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
    )
    try:
        for _ in range(100):
            if pid_file.exists() and pid_file.read_text(encoding="utf-8"):
                break
            await asyncio.sleep(0.02)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        process.send_signal(signal.SIGTERM)
        await asyncio.wait_for(process.wait(), timeout=5)
        assert process.returncode is not None
        for _ in range(50):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("Claude Code child was left running")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def test_config_set_stores_the_default_model(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    status = main(["config", "set", "default_model", "nvidia/my-model"])

    assert status == 0
    path = tmp_path / "claude-nim" / "config.toml"
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert 'default_model = "nvidia/my-model"' in path.read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "default_model" in output
    assert "nvidia/my-model" in output


def test_config_set_rejects_an_unknown_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    with pytest.raises(SystemExit) as captured:
        main(["config", "set", "unknown_key", "value"])

    assert captured.value.code == 2
    assert not (tmp_path / "claude-nim" / "config.toml").exists()


def test_config_set_rejects_an_invalid_value(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    status = main(["config", "set", "listen_port", "not-a-number"])

    assert status == 2
    assert "Configuration error" in capsys.readouterr().err
    assert not (tmp_path / "claude-nim" / "config.toml").exists()


def test_config_show_reports_the_effective_default_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)

    status = main(["config", "show"])

    assert status == 0
    output = capsys.readouterr().out
    assert DEFAULT_MODEL in output
    assert "built-in default" in output


def test_config_show_json_reports_value_sources(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    assert main(["config", "set", "default_model", "nvidia/my-model"]) == 0
    capsys.readouterr()

    status = main(["config", "show", "--json"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["values"]["default_model"] == "nvidia/my-model"
    assert payload["sources"]["default_model"] == "config.toml"
    assert payload["exists"] is True


def test_config_show_reports_the_environment_as_a_source(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/env-model")

    status = main(["config", "show", "--json"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["values"]["default_model"] == "nvidia/env-model"
    assert payload["sources"]["default_model"] == "NVIDIA_MODEL"


def test_generated_local_token_is_random_urlsafe_text() -> None:
    token, generated = _resolve_local_token(
        token_stdin=False,
        token_file=None,
        allow_insecure_token_file=False,
    )

    assert generated is True
    assert len(token) >= 32
    assert " " not in token


def test_local_token_can_be_read_from_stdin(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("token-from-stdin\n"))

    token, generated = _resolve_local_token(
        token_stdin=True,
        token_file=None,
        allow_insecure_token_file=False,
    )

    assert token == "token-from-stdin"
    assert generated is False


def test_insecure_token_file_requires_explicit_acceptance(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("file-token\n", encoding="utf-8")
    os.chmod(token_file, 0o644)

    with pytest.raises(ValueError, match="group/world accessible"):
        _resolve_local_token(
            token_stdin=False,
            token_file=token_file,
            allow_insecure_token_file=False,
        )

    token, generated = _resolve_local_token(
        token_stdin=False,
        token_file=token_file,
        allow_insecure_token_file=True,
    )
    assert token == "file-token"
    assert generated is False


def test_serve_uses_the_shipped_default_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    captured: list[str] = []

    async def fake_serve(settings, token, generated):  # type: ignore[no-untyped-def]
        captured.append(settings.default_model)
        return 0

    monkeypatch.setattr("claude_nim_proxy.cli._serve_until_stopped", fake_serve)

    status = main(["serve", "--upstream-auth", "none"])

    assert status == 0
    assert captured == [DEFAULT_MODEL]


@pytest.mark.asyncio
async def test_serve_starts_and_stops_cleanly_on_sigterm(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["XDG_CONFIG_HOME"] = str(tmp_path)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "claude_nim_proxy.cli",
        "serve",
        "--model",
        "nvidia/model",
        "--upstream-auth",
        "none",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
    )
    try:
        base_url_line = await asyncio.wait_for(process.stderr.readline(), timeout=2)
        token_line = await asyncio.wait_for(process.stderr.readline(), timeout=2)
        assert re.fullmatch(
            r"export ANTHROPIC_BASE_URL='http://127\.0\.0\.1:\d+'\n",
            base_url_line.decode(),
        )
        assert re.fullmatch(
            r"export ANTHROPIC_AUTH_TOKEN='[A-Za-z0-9_-]+'\n",
            token_line.decode(),
        )

        process.send_signal(signal.SIGTERM)
        exit_code = await asyncio.wait_for(process.wait(), timeout=2)
        assert exit_code == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def test_serve_token_can_come_from_stdin_or_file_but_not_command_line() -> None:
    parser = build_parser()
    serve = parser._subparsers._group_actions[0].choices["serve"]  # type: ignore[attr-defined]
    option_strings = {option for action in serve._actions for option in action.option_strings}

    assert "--token-stdin" in option_strings
    assert "--token-file" in option_strings
    assert "--token" not in option_strings
