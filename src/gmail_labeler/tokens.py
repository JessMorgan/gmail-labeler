"""OAuth token storage: OS keyring first, 0600 file fallback (ADR-0004).

The refresh token grants `gmail.modify` over the user's whole mailbox — it is
handled as secret material end to end: values NEVER appear in logs, error
messages, or chained tracebacks (all keyring/OS exceptions are re-raised
`from None` with only the exception TYPE name; tests assert even a backend
that echoes the value cannot leak it), files are 0600 at WRITE time (fchmod
closes the pre-existing-file hole), written atomically (temp+fsync+rename),
and read defensively (truncated/corrupt/non-UTF-8 files — reachable via a
crash mid-save — surface as TokenStoreError, never raw tracebacks).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Protocol

SERVICE_NAME = "gmail-labeler"
ACCOUNT = "gmail-oauth-refresh"
GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"

_MALFORMED = "stored token JSON is malformed; re-run `gmail-labeler auth login`"


class TokenStoreError(Exception):
    """Token store unavailable/misconfigured (message never carries values)."""


class CorruptTokenError(TokenStoreError):
    """Stored bytes are undecodable (crash-mid-save is reachable). The store
    itself WORKS — login may overwrite; only readers must not dead-end."""


class TokenStore(Protocol):
    def load(self) -> str | None:
        """The stored token JSON, or None if signed out."""

    def save(self, token_json: str) -> None: ...

    def probe(self) -> None:
        """Availability check for login's pre-consent gate: reads must not
        dead-end on corrupt content, and WRITABILITY is exercised (a read-
        only store discovered after consent loses the user's token)."""


class KeyringStore:
    """OS keyring (Secret Service / Keychain / Credential Manager).

    Import is lazy (headless boxes have the package but no backend), and
    every backend call is wrapped `from None` — third-party backend
    exception text is untrusted: it can contain the value it failed on.
    """

    def load(self) -> str | None:
        try:
            import keyring

            return keyring.get_password(SERVICE_NAME, ACCOUNT)
        except Exception as e:
            raise TokenStoreError(
                f"OS keyring read failed ({type(e).__name__}); "
                'set gmail.token_backend = "file" in config.toml if headless'
            ) from None

    def save(self, token_json: str) -> None:
        try:
            import keyring

            keyring.set_password(SERVICE_NAME, ACCOUNT, token_json)
        except Exception as e:
            raise TokenStoreError(
                f"OS keyring write failed ({type(e).__name__}); "
                'set gmail.token_backend = "file" in config.toml if headless'
            ) from None

    def probe(self) -> None:
        self.load()  # any backend failure raises TokenStoreError


class FileTokenStore:
    """0600 file fallback for headless boxes — atomic and enforced."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._warned_loose = False

    def load(self) -> str | None:
        try:
            if not self.path.is_file():
                return None
            mode = self.path.stat().st_mode
            if mode & 0o077 and not self._warned_loose:
                self._warned_loose = True
                print(
                    f"WARNING: {self.path} is group/world-readable (mode "
                    f"{mode & 0o777:o}); chmod 600 it — it holds a gmail.modify token",
                    file=sys.stderr,
                )
            raw = self.path.read_bytes()
        except OSError as e:
            raise TokenStoreError(
                f"token file unreadable ({type(e).__name__})"
            ) from None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise CorruptTokenError(_MALFORMED) from None

    def _ensure_parent(self) -> Path:
        parent = self.path.parent
        if not parent.exists():
            # chmod ONLY dirs we create — a user-chosen parent (e.g. a
            # shared dir, or /tmp) is not ours to re-permission (review
            # MAJOR: blanket chmod could take /tmp to 0700 system-wide).
            # Confidentiality rides on the 0600 temp + atomic replace.
            # Intermediate ancestors get default modes (mkdir -p semantics);
            # only the leaf token dir is clamped to 0700.
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(parent, 0o700)
        return parent

    def probe(self) -> None:
        # content is save()'s problem, not availability's
        with contextlib.suppress(CorruptTokenError):
            self.load()
        try:
            parent = self._ensure_parent()
            fd, tmp = tempfile.mkstemp(dir=parent, prefix=".probe-", suffix=".tmp")
            os.close(fd)
            os.unlink(tmp)
        except OSError as e:
            raise TokenStoreError(
                f"token store unwritable ({type(e).__name__})"
            ) from None

    def save(self, token_json: str) -> None:
        parent = self._ensure_parent()
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=".tok-", suffix=".tmp")
        replaced = False
        try:
            try:
                os.fchmod(fd, 0o600)  # atomic replace keeps the temp's mode
            except OSError:
                os.close(fd)  # qwen NIT: never leak our own fd even here
                raise
            try:
                fh = os.fdopen(fd, "w", encoding="utf-8")
            except BaseException:
                os.close(fd)
                raise
            with fh:
                fh.write(token_json)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)  # replaces any symlink at path too
            replaced = True
        except OSError as e:
            raise TokenStoreError(
                f"token file unwritable ({type(e).__name__})"
            ) from None
        except BaseException:
            raise
        finally:
            # a token-bearing temp must never survive ANY failure (ENOSPC
            # mid-write is the realistic one)
            if not replaced:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
        # rename durability: best-effort fsync of the directory entry
        with contextlib.suppress(OSError):
            dfd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)


def store_for_backend(backend: str, token_file: Path) -> TokenStore:
    if backend == "keyring":
        return KeyringStore()
    if backend == "file":
        return FileTokenStore(token_file)
    raise TokenStoreError(f"unknown token_backend {backend!r}")  # config guards this


def make_token_json(refresh_token: str, client_id: str, client_secret: str) -> str:
    return json.dumps(
        {
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "token_uri": "https://oauth2.googleapis.com/token",
            "scopes": [GMAIL_MODIFY_SCOPE],
        },
        sort_keys=True,
    )


def parse_token_json(raw: str) -> dict[str, object]:
    """Single parser for both status() and credentials() (no divergent copies)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise TokenStoreError(_MALFORMED) from None
    if not isinstance(data, dict):
        raise TokenStoreError(_MALFORMED)
    return data


def is_signed_in_data(data: dict[str, object]) -> bool:
    """The fields credentials() will actually require — status must not
    report SIGNED IN on a token that cannot be used."""
    return all(data.get(k) for k in ("refresh_token", "client_id", "client_secret"))
