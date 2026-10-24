"""Secure NVIDIA credential resolution and persistence.

The environment always takes precedence over persistent storage. Persistent credentials use
an OS keyring when one is available and otherwise fall back to a mode-0600 JSON file with
an explicit warning.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import keyring
from keyring.errors import KeyringError, NoKeyringError, PasswordDeleteError

SERVICE_NAME = "claude-nim"
ACCOUNT_NAME = "nvidia"
_CREDENTIALS_VERSION = 1


class CredentialSource(StrEnum):
    """Describe where a resolved NVIDIA credential came from."""

    ENVIRONMENT = "environment"
    KEYRING = "credential store"
    FILE = "file fallback"
    NONE = "not configured"


class CredentialError(RuntimeError):
    """Raised when a credential cannot be safely read, stored, or removed."""


@dataclass(frozen=True, slots=True)
class ResolvedCredential:
    """A credential and its non-secret source description."""

    api_key: str
    source: CredentialSource


class CredentialManager:
    """Resolve and manage the NVIDIA credential owned by this CLI."""

    def __init__(
        self,
        *,
        fallback_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
        keyring_module: Any = keyring,
        prompt: Callable[[str], str] | None = None,
        interactive: bool | None = None,
    ) -> None:
        self._fallback_path = fallback_path or default_credentials_path()
        self._environ = os.environ if environ is None else environ
        self._keyring = keyring_module
        self._prompt = prompt
        self._interactive = interactive

    @property
    def fallback_path(self) -> Path:
        """Return the path used by the unencrypted file fallback."""

        return self._fallback_path

    def resolve(self, *, allow_prompt: bool = True) -> ResolvedCredential | None:
        """Resolve a credential without exposing its value.

        Environment credentials win, followed by the OS keyring and the file fallback. If
        none exists and an interactive terminal is available, securely prompt and store a
        new credential.
        """

        environment_key = self._environ.get("NVIDIA_API_KEY", "").strip()
        if environment_key:
            return ResolvedCredential(environment_key, CredentialSource.ENVIRONMENT)

        stored = self._read_keyring()
        if stored:
            return ResolvedCredential(stored, CredentialSource.KEYRING)

        stored = self._read_fallback()
        if stored:
            return ResolvedCredential(stored, CredentialSource.FILE)

        if allow_prompt and self._is_interactive():
            api_key = self._secure_prompt("NVIDIA API key: ").strip()
            if not api_key:
                raise CredentialError("No NVIDIA API key was entered.")
            source = self.store(api_key)
            return ResolvedCredential(api_key, source)
        return None

    def prompt_and_store(self) -> CredentialSource:
        """Securely prompt for a credential and persist it."""

        api_key = self._secure_prompt("NVIDIA API key: ").strip()
        if not api_key:
            raise CredentialError("No NVIDIA API key was entered.")
        return self.store(api_key)

    def has_stored_credential(self) -> bool:
        """Return whether a persistent CLI-owned credential exists."""

        return self._read_keyring() is not None or self._fallback_path.exists()

    def store(self, api_key: str) -> CredentialSource:
        """Store a credential in the OS keyring, or in the warned file fallback."""

        normalized = api_key.strip()
        if not normalized:
            raise CredentialError("The NVIDIA API key must not be empty.")
        try:
            self._keyring.set_password(SERVICE_NAME, ACCOUNT_NAME, normalized)
        except KeyringError:
            self._write_fallback(normalized)
            return CredentialSource.FILE
        except Exception:
            # Third-party backends are allowed to raise plain exceptions. Falling back
            # keeps credential acquisition usable while retaining the explicit warning.
            self._write_fallback(normalized)
            return CredentialSource.FILE
        if self._fallback_path.exists():
            try:
                self._fallback_path.unlink()
            except OSError as exc:
                raise CredentialError(
                    f"Stored the OS credential, but cannot remove the stale fallback file "
                    f"{self._fallback_path}: {exc.strerror}"
                ) from exc
        return CredentialSource.KEYRING

    def delete_stored(self) -> bool:
        """Remove persistent CLI-owned credentials without changing the environment."""

        removed = False
        try:
            stored = self._keyring.get_password(SERVICE_NAME, ACCOUNT_NAME)
        except NoKeyringError:
            stored = None
        except KeyringError as exc:
            raise CredentialError(f"Could not access the OS credential store: {exc}") from exc
        except Exception as exc:
            raise CredentialError(f"Could not access the OS credential store: {exc}") from exc
        if stored is not None:
            try:
                self._keyring.delete_password(SERVICE_NAME, ACCOUNT_NAME)
                removed = True
            except PasswordDeleteError:
                pass
            except KeyringError as exc:
                raise CredentialError(
                    f"Could not remove the OS credential-store entry: {exc}"
                ) from exc
            except Exception as exc:
                raise CredentialError(
                    f"Could not remove the OS credential-store entry: {exc}"
                ) from exc
        if self._fallback_path.exists():
            try:
                self._fallback_path.unlink()
                removed = True
            except OSError as exc:
                raise CredentialError(
                    f"Cannot remove fallback credential file {self._fallback_path}: {exc.strerror}"
                ) from exc
        return removed

    def fallback_exists(self) -> bool:
        """Return whether an unencrypted fallback credential file exists."""

        return self._fallback_path.exists()

    def _read_keyring(self) -> str | None:
        try:
            value = self._keyring.get_password(SERVICE_NAME, ACCOUNT_NAME)
        except NoKeyringError:
            return None
        except KeyringError as exc:
            raise CredentialError(f"Could not access the OS credential store: {exc}") from exc
        except Exception as exc:
            raise CredentialError(f"Could not access the OS credential store: {exc}") from exc
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _read_fallback(self) -> str | None:
        path = self._fallback_path
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CredentialError(
                f"Cannot read fallback credential file {path}: {exc.strerror}"
            ) from exc
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise CredentialError(
                f"Fallback credential file {path} is group/world accessible; use chmod 600"
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CredentialError(f"Invalid JSON in fallback credential file {path}") from exc
        if not isinstance(payload, dict) or payload.get("version") != _CREDENTIALS_VERSION:
            raise CredentialError(f"Unsupported fallback credential file format: {path}")
        api_key = payload.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            raise CredentialError(f"Fallback credential file {path} does not contain an API key")
        return api_key.strip()

    def _write_fallback(self, api_key: str) -> None:
        path = self._fallback_path
        payload = {
            "version": _CREDENTIALS_VERSION,
            "service": SERVICE_NAME,
            "account": ACCOUNT_NAME,
            "api_key": api_key,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                text=True,
            )
        except OSError as exc:
            raise CredentialError(
                f"Cannot write fallback credential file {path}: {exc.strerror}"
            ) from exc
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
                os.chmod(temporary_path, 0o600)
                json.dump(payload, temporary_file, indent=2)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
            os.chmod(path, 0o600)
        except OSError as exc:
            raise CredentialError(
                f"Cannot write fallback credential file {path}: {exc.strerror}"
            ) from exc
        finally:
            temporary_path.unlink(missing_ok=True)

    def _is_interactive(self) -> bool:
        if self._interactive is not None:
            return self._interactive
        try:
            return os.isatty(0)
        except OSError:
            return False

    def _secure_prompt(self, label: str) -> str:
        if self._prompt is not None:
            return self._prompt(label)
        import getpass

        return getpass.getpass(label)


def default_credentials_path() -> Path:
    """Return the per-user fallback credential path."""

    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "claude-nim" / "credentials.json"
