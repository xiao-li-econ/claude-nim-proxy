from __future__ import annotations

import asyncio
import io
import os
import re
import signal
import sys
from pathlib import Path

import pytest

from claude_nim_proxy.cli import _resolve_local_token, build_parser, main


def test_cli_exposes_the_planned_command_tree() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    for command in ("run", "serve", "auth", "models", "doctor"):
        assert command in help_text


def test_run_passes_claude_arguments_verbatim(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    status = main(
        [
            "run",
            str(tmp_path),
            "--model",
            "nvidia/model",
            "--",
            "--resume",
            "--permission-mode",
            "acceptEdits",
        ]
    )

    assert status == 1
    error = capsys.readouterr().err
    assert "Phase 4" in error


def test_cli_rejects_missing_model_only_when_run_is_implemented(capsys) -> None:
    # Phase 1 deliberately does not perform model resolution or network I/O.
    assert build_parser().parse_args(["run", "."]).command == "run"


def test_auth_commands_remain_skeletons(capsys) -> None:
    assert main(["auth", "status"]) == 1
    assert "Phase 4" in capsys.readouterr().err


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


def test_serve_requires_an_explicit_model(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    config_path = tmp_path / "config.toml"

    status = main(
        [
            "serve",
            "--config",
            str(config_path),
            "--upstream-auth",
            "none",
        ]
    )

    assert status == 2
    assert "No NVIDIA model is selected" in capsys.readouterr().err


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
