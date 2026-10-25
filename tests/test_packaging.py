from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import claude_nim_proxy
from claude_nim_proxy.cli import main

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _project_version() -> str:
    payload = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def test_cli_reports_the_installed_package_version(capsys) -> None:
    with pytest.raises(SystemExit) as captured:
        main(["--version"])

    assert captured.value.code == 0
    assert capsys.readouterr().out.strip() == f"claude-nim {claude_nim_proxy.__version__}"


def test_pyproject_is_the_single_source_of_the_version() -> None:
    assert _project_version() == claude_nim_proxy.__version__
