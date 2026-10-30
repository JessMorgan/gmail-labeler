"""`auth` commands: one-time desktop OAuth flow + credential status (ADR-0004).

The flow is the installed-app (loopback) pattern against the USER's own GCP
OAuth client (desktop-type ONLY: web clients cannot complete loopback
redirects and would strand the user post-consent). Nothing here is logged or
echoed with token values; third-party exception text is replaced by type
names (`from None`) because backend errors can embed the value they touched.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .tokens import (
    GMAIL_MODIFY_SCOPE,
    TokenStore,
    TokenStoreError,
    is_signed_in_data,
    make_token_json,
    parse_token_json,
    store_for_backend,
)

_UNVERIFIED_APP_NOTICE = (
    "Google will show a 'Google hasn't verified this app' warning — expected "
    "for a personal OAuth client (ADR-0004). Click Advanced → Go to project → "
    "Allow. Make sure the OAuth app is published to 'In production' (not "
    "'Testing'), or the refresh token expires every 7 days."
)


class AuthError(Exception):
    """Auth unusable (no client secret, cancelled flow, revoked token)."""


def _load_client_secret(path: object) -> tuple[str, str]:
    p = Path(str(path))
    if not p.is_file():
        raise AuthError(
            f"OAuth client file not found: {p}\n"
            "Create a Desktop-app OAuth client in YOUR GCP project (see the "
            "README setup section) and set gmail.credentials_path."
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        # str(e) is position-only for JSONDecodeError; e.doc holds file
        # CONTENTS (incl. client_secret) — never echo beyond the type name.
        raise AuthError(
            f"OAuth client file unreadable/malformed: {type(e).__name__}"
        ) from None
    installed = data.get("installed")
    if not isinstance(installed, dict):
        raise AuthError(
            "OAuth client file is not a DESKTOP-app client (ADR-0004 requires "
            "type Desktop; web clients cannot complete the loopback flow)"
        )
    client_id = installed.get("client_id")
    client_secret = installed.get("client_secret")
    if not client_id or not client_secret:
        raise AuthError("desktop client missing client_id/client_secret")
    return str(client_id), str(client_secret)


def login(cfg: Config, store: TokenStore | None = None) -> None:
    """Run the interactive consent flow and persist the refresh token."""
    if cfg.gmail.credentials_path is None:
        raise AuthError("gmail.credentials_path is not set (see example.toml)")
    store = store or store_for_backend(cfg.gmail.token_backend, cfg.gmail.token_file)
    # Probe the store BEFORE asking the user to consent — a keyring-less or
    # write-blocked box must not lose a just-obtained refresh token on save.
    # Probe = availability (read AND writability), not content: a corrupt
    # stored file must not dead-end the very command that recovers from it.
    store.probe()
    client_id, client_secret = _load_client_secret(cfg.gmail.credentials_path)
    print(_UNVERIFIED_APP_NOTICE, file=sys.stderr)
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:  # pragma: no cover - dependency present by default
        raise AuthError(
            "OAuth libraries unavailable (google-auth-oauthlib missing?)"
        ) from None
    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        },
        scopes=[GMAIL_MODIFY_SCOPE],
    )
    try:
        creds = flow.run_local_server(port=0, open_browser=True)
    except Exception as e:
        # google_auth_oauthlib raises AttributeError-derived WSGITimeoutError
        # and oauthlib OAuth2Error types — catch-all with type name only:
        # exception text around token exchange can carry fragments.
        raise AuthError(f"consent flow did not complete ({type(e).__name__})") from None
    refresh = getattr(creds, "refresh_token", None)
    if not refresh:
        raise AuthError(
            "flow completed without a refresh token (account not re-authorized?)"
        )
    store.save(make_token_json(refresh, client_id, client_secret))


@dataclass(frozen=True, slots=True)
class AuthStatus:
    signed_in: bool
    backend: str
    detail: str  # presence/absence + cause — never a token value


def status(cfg: Config, store: TokenStore | None = None) -> AuthStatus:
    backend = cfg.gmail.token_backend
    try:
        store = store or store_for_backend(backend, cfg.gmail.token_file)
        raw = store.load()
    except TokenStoreError as e:
        return AuthStatus(False, backend, str(e))
    if raw is None:
        return AuthStatus(
            False, backend, "no stored token — run `gmail-labeler auth login`"
        )
    try:
        ok = is_signed_in_data(parse_token_json(raw))
    except TokenStoreError:
        ok = False
    return AuthStatus(
        ok,
        backend,
        "usable refresh token present (gmail.modify)"
        if ok
        else "stored token incomplete/malformed — re-run `gmail-labeler auth login`",
    )


def _scopes(data: dict[str, object]) -> list[str]:
    raw = data.get("scopes")
    if isinstance(raw, list) and all(isinstance(s, str) for s in raw):
        return list(raw)
    return [GMAIL_MODIFY_SCOPE]


def resolve_store(cfg: Config) -> TokenStore:
    return store_for_backend(cfg.gmail.token_backend, cfg.gmail.token_file)


def credentials(cfg: Config) -> object:
    """Build google Credentials from the stored refresh token (for the client)."""
    from google.oauth2.credentials import Credentials

    store = resolve_store(cfg)
    try:
        raw = store.load()
    except TokenStoreError as e:
        raise AuthError(str(e)) from None  # documented contract
    if raw is None:
        raise AuthError("not signed in — run `gmail-labeler auth login`")
    try:
        data = parse_token_json(raw)
    except TokenStoreError as e:
        raise AuthError(str(e)) from None
    if not is_signed_in_data(data):
        raise AuthError("stored token incomplete — re-run `gmail-labeler auth login`")
    return Credentials(
        token=None,
        refresh_token=str(data["refresh_token"]),
        client_id=str(data["client_id"]),
        client_secret=str(data["client_secret"]),
        token_uri=str(data.get("token_uri", "https://oauth2.googleapis.com/token")),
        scopes=_scopes(data),
    )
