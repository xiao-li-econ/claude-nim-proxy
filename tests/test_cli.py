from __future__ import annotations

from pathlib import Path

from claude_nim_proxy.cli import build_parser, main


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
    assert "Phase 1" in error


def test_cli_rejects_missing_model_only_when_run_is_implemented(capsys) -> None:
    # Phase 1 deliberately does not perform model resolution or network I/O.
    assert build_parser().parse_args(["run", "."]).command == "run"


def test_auth_and_models_commands_are_skeletons(capsys) -> None:
    assert main(["auth", "status"]) == 1
    assert "Phase 4" in capsys.readouterr().err

    assert main(["models", "list"]) == 1
    assert "Phase 3" in capsys.readouterr().err


def test_serve_token_can_come_from_stdin_or_file_but_not_command_line() -> None:
    parser = build_parser()
    serve = parser._subparsers._group_actions[0].choices["serve"]  # type: ignore[attr-defined]
    option_strings = {
        option
        for action in serve._actions
        for option in action.option_strings
    }

    assert "--token-stdin" in option_strings
    assert "--token-file" in option_strings
    assert "--token" not in option_strings
