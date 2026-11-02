from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pytest

# Captured at import time — before the module-wide `_stub_flow` autouse
# fixture swaps in FakeFlow — so the #53 lib-contract test below can run
# the REAL InstalledAppFlow while every other test in this file stays on
# the fake.
from google_auth_oauthlib.flow import InstalledAppFlow as REAL_INSTALLED_APP_FLOW

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

    calls: list[dict[str, object]] = []

    class BoomFlow(FakeFlow):
        def run_local_server(self, **kw: object) -> FakeCreds:
            calls.append(kw)
            raise AttributeError(f"timeout while handling {REFRESH}")

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", BoomFlow)
    cfg = cfg_with_client(tmp_path)
    with pytest.raises(AuthError) as ei:
        login(cfg, MemoryStore())  # type: ignore[arg-type]
    assert "AttributeError" in str(ei.value)  # type name kept for diagnosis
    assert REFRESH not in str(ei.value)
    # from None severs the cause: a chained traceback would replay the
    # raise-site frames, whose exception text carries the token fragment.
    assert ei.value.__cause__ is None and ei.value.__suppress_context__
    # content-level too (deepseek-endorsed defense-in-depth): the rendered
    # chain must not hold the fragment even if a future edit re-chains.
    assert REFRESH not in "".join(traceback.format_exception(ei.value))
    # #53: only webbrowser.Error retries — a general flow failure must not
    # relaunch a second browser-less consent server behind the user.
    assert len(calls) == 1


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


# --- #53: headless boxes — webbrowser.Error must not kill login() ---------
# In google-auth-oauthlib 1.5.0, run_local_server calls
# webbrowser.get(browser).open() (flow.py:458-460) BEFORE printing the
# prompt with the URL (flow.py:463-464); with zero runnable-browser
# candidates webbrowser.get() RAISES webbrowser.Error, so the desktop
# attempt dies without ever showing the URL. login() must retry once
# browser-less. (A merely-dead launcher whose open() only returns False
# never raises — the lib prints its default prompt; that path is the
# library's, not ours, and stays as-is.)


def test_login_headless_retry_renders_prompt_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import webbrowser

    import google_auth_oauthlib.flow as flow_mod

    calls: list[dict[str, object]] = []

    class HeadlessFlow(FakeFlow):
        def run_local_server(self, **kw: object) -> FakeCreds:
            calls.append(kw)
            if len(calls) == 1:
                raise webbrowser.Error("can't open display (set BROWSER?)")
            return super().run_local_server(**kw)

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", HeadlessFlow)

    class CountingStore(MemoryStore):
        def __init__(self) -> None:
            self.probes = 0
            super().__init__()

        def probe(self) -> None:
            self.probes += 1
            super().probe()

    cfg = cfg_with_client(tmp_path)
    store = CountingStore()
    login(cfg, store)  # type: ignore[arg-type]
    assert store.value is not None and REFRESH in store.value  # consent done
    assert store.probes == 1  # the retry must not re-probe the token store

    assert len(calls) == 2  # exactly one browser-less retry
    assert calls[0].get("open_browser") is True  # desktop attempt first
    assert calls[1].get("open_browser") is False  # the #53 retry
    prompt = calls[1].get("authorization_prompt_message")
    assert isinstance(prompt, str) and prompt  # custom message, not the default

    # The lib renders the prompt via message.format(url=...) (flow.py:464);
    # a stray literal brace in our text would raise there instead of
    # printing — pin the render against a URL carrying the redirect_uri port.
    sample_url = (
        "https://accounts.google.com/o/oauth2/auth?client_id=abc.apps"
        "&redirect_uri=http%3A%2F%2Flocalhost%3A43217%2F&response_type=code"
    )
    rendered = prompt.format(url=sample_url)
    assert sample_url in rendered  # the URL itself is shown
    assert "ssh -L" in rendered  # remote-consent port-forward recipe
    assert "token_backend" in rendered  # headless storage pointer
    err = capsys.readouterr().err
    assert err.count("hasn't verified") == 1  # notice not re-printed for retry


def test_login_desktop_path_single_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import google_auth_oauthlib.flow as flow_mod

    calls: list[dict[str, object]] = []

    class SpyFlow(FakeFlow):
        def run_local_server(self, **kw: object) -> FakeCreds:
            calls.append(kw)
            return super().run_local_server(**kw)

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", SpyFlow)
    login(cfg_with_client(tmp_path), MemoryStore())  # type: ignore[arg-type]
    assert len(calls) == 1  # browser opened fine → no retry, no custom prompt
    assert calls[0].get("open_browser") is True
    # pin whatever the code passes on the desktop path: nothing at all, so
    # the lib default prompt ("Please visit this URL...") stands.
    assert "authorization_prompt_message" not in calls[0]


def test_login_headless_retry_failure_keeps_type_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import webbrowser

    import google_auth_oauthlib.flow as flow_mod

    class RetryBoomFlow(FakeFlow):
        def run_local_server(self, **kw: object) -> FakeCreds:
            if kw.get("open_browser") is False:
                raise TimeoutError(f"no redirect arrived for {REFRESH}")
            raise webbrowser.Error("can't open display (set BROWSER?)")

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", RetryBoomFlow)
    cfg = cfg_with_client(tmp_path)
    with pytest.raises(AuthError) as ei:
        login(cfg, MemoryStore())  # type: ignore[arg-type]
    # Retry failures land in the existing catch-all: type name only, exit 4.
    assert "TimeoutError" in str(ei.value)
    assert REFRESH not in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__suppress_context__
    err = capsys.readouterr().err
    assert err.count("hasn't verified") == 1  # notice not printed twice


# --- #53 follow-up: stdout buffering ---------------------------------------
# The retry's prompt rides the lib's print() to stdout, which Python
# BLOCK-buffers when stdout is not a TTY: over `ssh box 'gmail-labeler auth
# login'` or `| tee` the URL would stay invisible while the loopback server
# waits — the ticket's symptom class one layer down. login() must force
# line-buffering before the flow attempts.


def test_login_forces_line_buffered_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingStdout:
        def __init__(self) -> None:
            self.reconfigure_calls: list[dict[str, object]] = []
            self.applied = False

        def write(self, s: str) -> int:
            return len(s)

        def flush(self) -> None:
            pass

        def reconfigure(self, **kw: object) -> None:
            self.reconfigure_calls.append(kw)
            self.applied = kw.get("line_buffering") is True

    fake = RecordingStdout()
    monkeypatch.setattr(sys, "stdout", fake)

    import google_auth_oauthlib.flow as flow_mod

    class OrderingSpyFlow(FakeFlow):
        def run_local_server(self, **kw: object) -> FakeCreds:
            # ORDERING contract: buffering must be in effect BEFORE the
            # flow can print anything — the lib's prompt rides print(), so
            # a reconfigure after the attempts is the bug this pins out.
            assert fake.applied, "flow attempted before line-buffering applied"
            return super().run_local_server(**kw)

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", OrderingSpyFlow)
    login(cfg_with_client(tmp_path), MemoryStore())  # type: ignore[arg-type]
    assert fake.reconfigure_calls == [{"line_buffering": True}]


def test_login_survives_stdout_without_reconfigure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Objects without reconfigure (io.StringIO, injected custom targets —
    # NOT pytest's CaptureIO, which is a TextIOWrapper and has it) must hit
    # the attribute guard: skip, not abort, the flow.
    class PlainStdout:
        def write(self, s: str) -> int:
            return len(s)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stdout", PlainStdout())
    store = MemoryStore()
    login(cfg_with_client(tmp_path), store)  # type: ignore[arg-type]
    assert store.value is not None and REFRESH in store.value


def test_login_survives_reconfigure_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A reconfigure that EXISTS but raises (TextIOWrapper over a closed
    # buffer -> ValueError) must not abort login either — the flow proceeds
    # on whatever buffering is in effect.
    class ExplodingStdout:
        def write(self, s: str) -> int:
            return len(s)

        def flush(self) -> None:
            pass

        def reconfigure(self, **kw: object) -> None:
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys, "stdout", ExplodingStdout())
    store = MemoryStore()
    login(cfg_with_client(tmp_path), store)  # type: ignore[arg-type]
    assert store.value is not None and REFRESH in store.value


def test_login_line_buffering_applied_on_real_text_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The guard must not silently skip the REAL path: on a plain
    # TextIOWrapper the flag has to end up set.
    import io

    wrapper = io.TextIOWrapper(io.BytesIO())
    monkeypatch.setattr(sys, "stdout", wrapper)
    login(cfg_with_client(tmp_path), MemoryStore())  # type: ignore[arg-type]
    assert wrapper.line_buffering is True


# --- #53's ordering contract, pinned against the INSTALLED lib -------------
# The whole retry-safety design rests on lib-internal ORDERING asserted
# elsewhere only in comments: bind → browser attempt (raises, nothing
# printed yet) → finally server_close (port released) → browser-less retry:
# bind → print → wait, webbrowser never touched. This test runs the REAL
# google_auth_oauthlib.flow — a lib bump that reorders these steps fails it
# LOUDLY, because login()'s retry argument dies with it. It pins existing
# lib behavior (green since the first run), not new code. Network-free: the
# auth URL is built locally and timeout_seconds cuts the wait short before
# any token exchange.


def test_lib_ordering_contract_headless_raise_then_browserless_print(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import errno
    import socket
    import webbrowser
    import wsgiref.simple_server
    from typing import Any
    from urllib.parse import parse_qs, urlparse

    import google_auth_oauthlib.flow as flow_mod

    bound: list[int] = []
    real_make_server = wsgiref.simple_server.make_server

    def spy_make_server(*args: Any, **kwargs: Any) -> Any:
        server = real_make_server(*args, **kwargs)
        bound.append(server.server_port)  # the port ACTUALLY bound (port=0)
        return server

    monkeypatch.setattr(wsgiref.simple_server, "make_server", spy_make_server)

    flow = REAL_INSTALLED_APP_FLOW.from_client_config(
        {
            "installed": {
                "client_id": "abc.apps",
                "client_secret": "csec",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        },
        scopes=["https://www.googleapis.com/auth/gmail.modify"],
    )

    def assert_port_free(addr: str, port: int, family: int) -> None:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.bind((addr, port))
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                pytest.fail(
                    f"{addr}:{port} still has the first attempt's listener — "
                    "the lib's finally-block no longer closes the server "
                    "before the raise propagates, so login()'s browser-less "
                    "retry would serve the consent flow alongside a leaked "
                    "listener (fd + stray accept path) on the abandoned "
                    "port (#53)"
                )
            # family unavailable here (IPv6 disabled): the lib skips that
            # reservation the same way — nothing to pin.

    # (a) headless box: webbrowser.get raises AFTER the bind, BEFORE the
    # prompt print — the #53 symptom, at the source.
    def raise_get(*args: Any, **kwargs: Any) -> Any:
        raise webbrowser.Error("could not locate runnable browser")

    monkeypatch.setattr(webbrowser, "get", raise_get)
    # timeout_seconds=1 is defensive: a lib regression that skipped the
    # browser attempt would otherwise hang the suite inside handle_request
    # (no pytest-timeout here); the real lib raises long before any wait.
    # `as ei` is load-bearing: the exception's traceback pins the
    # run_local_server frame, whose locals hold the server — destroying the
    # exception (bare pytest.raises) would GC-close the sockets and make the
    # re-bind probe below pass VACUOUSLY even with the finally removed.
    with pytest.raises(webbrowser.Error) as ei:
        flow.run_local_server(port=0, open_browser=True, timeout_seconds=1)
    assert len(bound) == 1  # server was bound before the raise (flow.py:441)
    assert capsys.readouterr().out == ""  # ...and nothing printed yet
    # the finally-block (flow.py:483-484) closed BOTH loopback sockets the
    # lib holds for "localhost" (the served one + the ::1 reservation) —
    # probed while ei keeps the server object (hence any leaked socket)
    # alive, so a missing finally cannot hide behind refcounting:
    assert_port_free("127.0.0.1", bound[0], socket.AF_INET)
    if hasattr(socket, "AF_INET6"):
        assert_port_free("::1", bound[0], socket.AF_INET6)
    del ei  # release the traceback (and the server it pins) before part (b)

    # (b) browser-less retry on the SAME flow object: open_browser=False
    # never touches webbrowser, and the custom prompt prints with the
    # URL's redirect_uri port == the port actually listening.
    touched: list[str] = []

    def raise_stub(name: str) -> Any:
        def stub(*args: Any, **kwargs: Any) -> Any:
            touched.append(name)
            raise AssertionError(f"webbrowser.{name} touched")

        return stub

    monkeypatch.setattr(webbrowser, "get", raise_stub("get"))
    monkeypatch.setattr(webbrowser, "open", raise_stub("open"))
    with pytest.raises(flow_mod.WSGITimeoutError):
        flow.run_local_server(
            port=0,
            open_browser=False,
            authorization_prompt_message="SENTINEL {url}",
            timeout_seconds=1,  # abort the wait before any token exchange
        )
    assert touched == []  # the open_browser=False branch skipped webbrowser
    assert len(bound) == 2  # the retry rebound (fresh ephemeral port)
    out = capsys.readouterr().out
    assert out.startswith("SENTINEL ")  # custom template rendered via print
    redirect_uri = parse_qs(urlparse(out.strip()).query)["redirect_uri"][0]
    parsed = urlparse(redirect_uri)
    assert parsed.hostname in ("localhost", "127.0.0.1")
    assert parsed.port == bound[1]  # prompt URL points at the bound port
