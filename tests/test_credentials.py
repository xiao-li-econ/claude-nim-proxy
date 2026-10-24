from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from keyring.errors import NoKeyringError

from claude_nim_proxy.credentials import (
    CredentialManager,
    CredentialSource,
)


class MemoryKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, password: str) -> None:
        self.values[(service, account)] = password

    def delete_password(self, service: str, account: str) -> None:
        self.values.pop((service, account), None)


class FailingKeyring:
    def get_password(self, service: str, account: str) -> str | None:
        del service, account
        raise NoKeyringError("no backend")

    def set_password(self, service: str, account: str, password: str) -> None:
        del service, account, password
        raise NoKeyringError("no backend")

    def delete_password(self, service: str, account: str) -> None:
        del service, account
        raise NoKeyringError("no backend")


def test_environment_credential_takes_precedence(tmp_path: Path) -> None:
    backend = MemoryKeyring()
    backend.set_password("claude-nim", "nvidia", "stored-key")
    manager = CredentialManager(
        fallback_path=tmp_path / "credentials.json",
        environ={"NVIDIA_API_KEY": "environment-key"},
        keyring_module=backend,
        interactive=False,
    )

    credential = manager.resolve(allow_prompt=False)

    assert credential is not None
    assert credential.api_key == "environment-key"
    assert credential.source is CredentialSource.ENVIRONMENT


def test_keyring_credential_is_resolved_without_revealing_it_in_source(tmp_path: Path) -> None:
    backend = MemoryKeyring()
    backend.set_password("claude-nim", "nvidia", "stored-key")
    manager = CredentialManager(
        fallback_path=tmp_path / "credentials.json",
        environ={},
        keyring_module=backend,
        interactive=False,
    )

    credential = manager.resolve(allow_prompt=False)

    assert credential is not None
    assert credential.api_key == "stored-key"
    assert credential.source is CredentialSource.KEYRING


def test_interactive_prompt_stores_credential_in_keyring(tmp_path: Path) -> None:
    backend = MemoryKeyring()
    manager = CredentialManager(
        fallback_path=tmp_path / "credentials.json",
        environ={},
        keyring_module=backend,
        prompt=lambda label: "prompted-key",
        interactive=True,
    )

    credential = manager.resolve(allow_prompt=True)

    assert credential is not None
    assert credential.api_key == "prompted-key"
    assert credential.source is CredentialSource.KEYRING
    assert backend.values[("claude-nim", "nvidia")] == "prompted-key"


def test_file_fallback_is_private_and_used_when_keyring_is_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    manager = CredentialManager(
        fallback_path=path,
        environ={},
        keyring_module=FailingKeyring(),
        interactive=False,
    )

    source = manager.store("fallback-key")
    credential = manager.resolve(allow_prompt=False)

    assert source is CredentialSource.FILE
    assert credential is not None
    assert credential.api_key == "fallback-key"
    assert credential.source is CredentialSource.FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert payload["api_key"] == "fallback-key"
    assert payload["service"] == "claude-nim"
    assert payload["account"] == "nvidia"


def test_delete_stored_removes_fallback_but_not_environment(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    manager = CredentialManager(
        fallback_path=path,
        environ={"NVIDIA_API_KEY": "environment-key"},
        keyring_module=FailingKeyring(),
        interactive=False,
    )
    manager.store("fallback-key")

    assert manager.delete_stored() is True
    assert not path.exists()
    credential = manager.resolve(allow_prompt=False)
    assert credential is not None
    assert credential.api_key == "environment-key"
    assert credential.source is CredentialSource.ENVIRONMENT


def test_insecure_fallback_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text('{"version": 1, "api_key": "secret"}\n', encoding="utf-8")
    os.chmod(path, 0o644)
    manager = CredentialManager(
        fallback_path=path,
        environ={},
        keyring_module=FailingKeyring(),
        interactive=False,
    )

    try:
        manager.resolve(allow_prompt=False)
    except Exception as exc:
        assert "group/world accessible" in str(exc)
    else:
        raise AssertionError("expected an insecure fallback file to be rejected")
