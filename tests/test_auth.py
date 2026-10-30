from __future__ import annotations

import json
import traceback
from pathlib import Path

import pytest

from gmail_labeler.auth import AuthError, login, status
from gmail_labeler.config import Config, GmailConfig
from gmail_labeler.tokens import FileTokenStore, TokenStoreError, make_token_json

REFRESH = "1//0REALREFRESHneverlogged"


class MemoryStore:
    def __init__(self) -> None:
        self.value: str | None = None

    def load(self) -> str | None:
        return self.value

    def save(self, token_json: str) -> None:
        self.value = token_json

    def probe(self) -> None:
        self.load()


def cfg_with_client(tmp_path: Path) -> Config:
    secret_file = tmp_path / "client.json"
    secret_file.write_text(
        json.dumps({"installed": {"client_id": "abc.apps", "client_secret": "csec"}}),
        encoding="utf-8",
    )
    return Config(
        gmail=GmailConfig(credentials_path=secret_file, token_backend="file"),
    )


class FakeCreds:
    refresh_token = REFRESH


class FakeFlow:
    @classmethod
    def from_client_config(cls, config: dict, scopes: list[str]) -> FakeFlow:
        assert scopes == ["https://www.googleapis.com/auth/gmail.modify"]
        assert "installed" in config
        return cls()

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def run_local_server(self, **kw: object) -> FakeCreds:
        assert kw.get("port") == 0  # loopback ephemeral, per ADR-0004
        return FakeCreds()


@pytest.fixture(autouse=True)
def _stub_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    import google_auth_oauthlib.flow as flow_mod

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", FakeFlow)


def test_login_stores_token_and_shows_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = cfg_with_client(tmp_path)
    store = MemoryStore()
    login(cfg, store)  # type: ignore[arg-type]
    assert store.value is not None
    assert REFRESH in store.value  # the STORE holds it...
    err = capsys.readouterr().err
    assert REFRESH not in err  # ...stdout/stderr never do
    assert "hasn't verified" in err  # unverified-app guidance printed first


def test_login_requires_credentials_path() -> None:
    with pytest.raises(AuthError, match="credentials_path is not set"):
        login(Config(), MemoryStore())  # type: ignore[arg-type]


def test_login_missing_client_file(tmp_path: Path) -> None:
    cfg = Config(
        gmail=GmailConfig(credentials_path=tmp_path / "gone.json"),
    )
    with pytest.raises(AuthError, match="not found"):
        login(cfg, MemoryStore())  # type: ignore[arg-type]


def test_login_rejects_web_app_json(tmp_path: Path) -> None:
    p = tmp_path / "client.json"
    p.write_text("{}", encoding="utf-8")
    cfg = Config(gmail=GmailConfig(credentials_path=p))
    with pytest.raises(AuthError, match=r"(?i)desktop-app"):
        login(cfg, MemoryStore())  # type: ignore[arg-type]


def test_status_absent(tmp_path: Path) -> None:
    st = status(Config(), MemoryStore())  # type: ignore[arg-type]
    assert not st.signed_in
    assert "auth login" in st.detail


def test_status_present_without_leaking(tmp_path: Path) -> None:
    store = MemoryStore()
    store.save(make_token_json(REFRESH, "cid", "csec"))
    st = status(Config(), store)  # type: ignore[arg-type]
    assert st.signed_in
    assert REFRESH not in st.detail
    assert "gmail.modify" in st.detail


def test_status_malformed_token(tmp_path: Path) -> None:
    store = MemoryStore()
    store.save("garbage")
    st = status(Config(), store)  # type: ignore[arg-type]
    assert not st.signed_in
    assert "malformed" in st.detail


def test_login_rejects_web_client(tmp_path: Path) -> None:
    p = tmp_path / "client.json"
    p.write_text(
        json.dumps({"web": {"client_id": "a", "client_secret": "b"}}), encoding="utf-8"
    )
    cfg = Config(gmail=GmailConfig(credentials_path=p, token_backend="file"))
    with pytest.raises(AuthError, match="DESKTOP"):
        login(cfg, MemoryStore())  # type: ignore[arg-type]


def test_login_flow_exception_never_leaks_and_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import google_auth_oauthlib.flow as flow_mod

    class BoomFlow(FakeFlow):
        def run_local_server(self, **kw: object) -> FakeCreds:
            raise AttributeError(f"timeout while handling {REFRESH}")

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", BoomFlow)
    cfg = cfg_with_client(tmp_path)
    with pytest.raises(AuthError) as ei:
        login(cfg, MemoryStore())  # type: ignore[arg-type]
    assert "AttributeError" in str(ei.value)  # type name kept for diagnosis
    assert REFRESH not in str(ei.value)
    assert REFRESH not in traceback.format_exc()  # from None severs the cause


def test_login_probes_store_before_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Headless keyring must fail BEFORE the browser flow (qwen MAJOR:
    losing a just-obtained refresh token on save)."""
    import google_auth_oauthlib.flow as flow_mod

    started: list[bool] = []

    class SpyFlow(FakeFlow):
        def run_local_server(self, **kw: object) -> FakeCreds:
            started.append(True)
            return super().run_local_server(**kw)

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", SpyFlow)

    class DeadStore(MemoryStore):
        def load(self) -> str | None:
            from gmail_labeler.tokens import TokenStoreError

            raise TokenStoreError("OS keyring read failed (RuntimeError)")

        def probe(self) -> None:
            self.load()

    cfg = cfg_with_client(tmp_path)
    with pytest.raises(Exception) as ei:  # TokenStoreError surfaces to CLI handler
        login(cfg, DeadStore())  # type: ignore[arg-type]
    assert "keyring" in str(ei.value)
    assert started == []  # flow never started


def test_status_requires_usable_token() -> None:
    store = MemoryStore()
    store.save(json.dumps({"refresh_token": REFRESH}))  # no client creds
    st = status(Config(), store)  # type: ignore[arg-type]
    assert not st.signed_in  # would-be credentials() failure must not read SIGNED IN
    assert REFRESH not in st.detail


def test_credentials_builds_and_detects_corruption(tmp_path: Path) -> None:
    from gmail_labeler.auth import credentials as build_creds
    from gmail_labeler.config import GmailConfig

    cfg = Config(
        gmail=GmailConfig(token_backend="file", token_file=tmp_path / "t.json")
    )
    store = FileTokenStore(cfg.gmail.token_file)
    store.save(make_token_json(REFRESH, "cid", "csec"))
    creds = build_creds(cfg)
    assert getattr(creds, "refresh_token", None) == REFRESH

    store.save("{truncated")
    with pytest.raises(AuthError, match=r"malformed|incomplete"):
        build_creds(cfg)


def test_login_recovers_from_corrupt_token_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # deepseek MAJOR: a crash-mid-save corrupt file must not dead-end the
    # command that recovers from it.
    cfg = cfg_with_client(tmp_path)
    token = tmp_path / "tok.json"
    cfg = Config(
        gmail=GmailConfig(
            credentials_path=cfg.gmail.credentials_path,
            token_backend="file",
            token_file=token,
        )
    )
    token.write_bytes(b"\xff\xfe broken")
    store = FileTokenStore(token)
    login(cfg, store)
    stored = store.load()
    assert stored is not None and REFRESH in stored


def test_login_aborts_before_consent_on_unwritable_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # deepseek MINOR: read-only probe missed WRITE-blocked stores; the token
    # loss class the probe exists to prevent must fail before the browser.
    import google_auth_oauthlib.flow as flow_mod

    started: list[bool] = []

    class SpyFlow(FakeFlow):
        def run_local_server(self, **kw: object) -> FakeCreds:
            started.append(True)
            return super().run_local_server(**kw)

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", SpyFlow)
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir(mode=0o500)
    cfg = Config(
        gmail=GmailConfig(
            credentials_path=cfg_with_client(tmp_path).gmail.credentials_path,
            token_backend="file",
            token_file=ro_dir / "tok.json",
        )
    )
    with pytest.raises(TokenStoreError, match=r"unwritable|unreadable"):
        login(cfg)
    assert started == []
    ro_dir.chmod(0o700)  # let tmp_path cleanup work
