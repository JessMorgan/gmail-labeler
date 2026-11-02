"""Ticket #28 C2 — the CLI wiring of the auth wizard.

The pure machine (test_wizard.py) is driven here through `typer.testing` with
the REAL probe + the OAuth exchange behind the SAME injectable seam `auth login`
uses today (monkeypatched ``InstalledAppFlow``) — no network, no real keyring.
The single test ``test_ac1_fresh_profile_setup_reaches_green_status`` IS
acceptance criterion #1 (a fresh profile, no config, reaches a working
`auth status` without reading source) — it is named loudly per the design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import gmail_labeler.cli as cli
from gmail_labeler.cli import app

runner = CliRunner()

# secret-discipline sentinels for the client JSON + the refresh token
CLIENT_ID = "SENTINEL-client-id.apps"
CLIENT_SECRET = "SENTINEL-client-secret-never-echoed"
REFRESH = "1//0SENTINELrefreshneverlogged"
PROJECT_ID = "my-project-42"


class FakeCreds:
    refresh_token = REFRESH


class FakeFlow:
    """Stands in for google_auth_oauthlib.flow.InstalledAppFlow — the SAME
    seam test_auth.py fakes for `auth login`, so the wizard (which COMPOSES
    login) is faked by the identical mechanism."""

    def __init__(self, config: dict, scopes: list[str]) -> None:
        assert scopes == ["https://www.googleapis.com/auth/gmail.modify"]
        assert "installed" in config

    @classmethod
    def from_client_config(cls, config: dict, scopes: list[str], **kw: object):  # type: ignore[no-untyped-def]
        return cls(config, scopes)

    def run_local_server(self, **kw: object) -> FakeCreds:
        assert kw.get("port") == 0
        return FakeCreds()


@pytest.fixture
def fake_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    import google_auth_oauthlib.flow as flow_mod

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", FakeFlow)


@pytest.fixture
def headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the file token backend (a headless box) so the token lands in a
    test-readable file — deterministic regardless of the CI keyring posture."""
    monkeypatch.setattr(
        cli._RealEnvironmentProbe, "keyring_available", lambda self: False
    )


def _client_json(tmp_path: Path) -> Path:
    p = tmp_path / "desktop-client.json"
    p.write_text(
        json.dumps(
            {"installed": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}}
        ),
        encoding="utf-8",
    )
    return p


def _setup_input(client_path: Path, project_id: str = PROJECT_ID) -> str:
    # prompt order: project-id, then client-json
    return f"{project_id}\n{client_path}\n"


# ===========================================================================
# AC1 — THE acceptance-criterion-#1 end-to-end. Fresh profile, NO config dir,
# scripted wizard completion, faked token seam → a green `auth status`.
# ===========================================================================
def test_ac1_fresh_profile_setup_reaches_green_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_flow: None,
    headless: None,
) -> None:
    # AC1: "End-to-end wizard reaches a working auth status from a fresh
    # machine profile without reading source." This one test IS that AC.
    monkeypatch.setenv("HOME", str(tmp_path))  # token default lands under tmp
    client = _client_json(tmp_path)
    config = tmp_path / "config.toml"
    assert not config.exists()  # fresh profile: no config yet

    result = runner.invoke(
        app, ["auth", "setup", "--config", str(config)], input=_setup_input(client)
    )
    assert result.exit_code == 0, result.output
    assert config.is_file()  # the wizard wired the config

    # the wizard's own terminal status step reported green...
    assert "signed in" in result.output.lower()

    # ...AND a SEPARATE follow-up command (the next thing a user runs) reads
    # the persisted config and agrees — the profile now works without source.
    followup = runner.invoke(app, ["auth", "--config", str(config)])
    assert followup.exit_code == 0, followup.output
    assert "SIGNED IN" in followup.output

    # secret discipline (invariant #8) across the WHOLE run: the client JSON's
    # id/secret and the refresh token never appear in any output.
    out = result.output + followup.output
    for sentinel in (CLIENT_ID, CLIENT_SECRET, REFRESH):
        assert sentinel not in out, f"{sentinel} leaked"


def test_ac1_written_config_is_loadable_and_value_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_flow: None,
    headless: None,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    client = _client_json(tmp_path)
    config = tmp_path / "config.toml"
    runner.invoke(
        app, ["auth", "setup", "--config", str(config)], input=_setup_input(client)
    )
    text = config.read_text(encoding="utf-8")
    # the config holds PATHS only (invariant #8) — never the client secret
    assert CLIENT_SECRET not in text
    assert CLIENT_ID not in text
    assert str(client) in text  # the path IS recorded (a path, not a secret)
    assert 'token_backend = "file"' in text


# --- bare `auth` entry decision (the TTY rule) ------------------------------
def _unsigned_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[gmail]\ntoken_backend = "file"\ntoken_file = "{tmp_path / "tok.json"}"\n',
        encoding="utf-8",
    )
    return cfg


def test_bare_auth_no_creds_non_tty_exits_5_naming_setup(tmp_path: Path) -> None:
    # a script must never half-prompt: exit 5, names the interactive command
    cfg = _unsigned_cfg(tmp_path)
    result = runner.invoke(app, ["auth", "--config", str(cfg)])
    assert result.exit_code == 5
    assert "NOT SIGNED IN" in result.output
    assert "auth setup" in (result.output + result.stderr)


def test_bare_auth_no_creds_tty_launches_the_wizard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # on a real terminal the same bare `auth` enters the wizard (prompts)
    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: True)
    cfg = _unsigned_cfg(tmp_path)
    client = _client_json(tmp_path)
    # feed the wizard's prompts (project-id, client-json); keyring forced off
    monkeypatch.setattr(
        cli._RealEnvironmentProbe, "keyring_available", lambda self: False
    )
    import google_auth_oauthlib.flow as flow_mod

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", FakeFlow)
    monkeypatch.setenv("HOME", str(tmp_path))
    # the existing (unsigned) config is the write target → the wizard asks to
    # overwrite it; answer the prompts in order: project-id, client-json, y
    result = runner.invoke(
        app, ["auth", "--config", str(cfg)], input=f"{PROJECT_ID}\n{client}\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert "setup" in result.output.lower() or "console" in result.output.lower()


def test_bare_auth_signed_in_still_exits_0(tmp_path: Path) -> None:
    # signed-in bare `auth` keeps its status meaning (exit 0), no wizard
    from gmail_labeler.tokens import FileTokenStore, make_token_json

    cfg = _unsigned_cfg(tmp_path)
    FileTokenStore(tmp_path / "tok.json").save(
        make_token_json(REFRESH, CLIENT_ID, CLIENT_SECRET)
    )
    result = runner.invoke(app, ["auth", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "SIGNED IN" in result.output


# --- one name: `auth wizard` is rejected at exit 2 --------------------------
def test_auth_wizard_alias_is_rejected_exit_2() -> None:
    result = runner.invoke(app, ["auth", "wizard"])
    assert result.exit_code == 2
    assert (
        "no such command" in result.output.lower() or "wizard" in result.output.lower()
    )


# --- exit-contract mapping ---------------------------------------------------
def test_setup_cancel_exits_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gmail_labeler.wizard import WizardCancelled

    def _boom(prompt: object) -> str:
        raise WizardCancelled()

    monkeypatch.setattr(cli, "_typer_ask", _boom)
    config = tmp_path / "config.toml"
    result = runner.invoke(app, ["auth", "setup", "--config", str(config)], input="")
    assert result.exit_code == 130
    assert not config.exists()  # cancel wrote nothing


def test_setup_login_failure_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BoomFlow(FakeFlow):
        def run_local_server(self, **kw: object) -> FakeCreds:
            raise AttributeError(f"timeout handling {REFRESH}")

    import google_auth_oauthlib.flow as flow_mod

    monkeypatch.setattr(flow_mod, "InstalledAppFlow", BoomFlow)
    monkeypatch.setattr(
        cli._RealEnvironmentProbe, "keyring_available", lambda self: False
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    client = _client_json(tmp_path)
    config = tmp_path / "config.toml"
    result = runner.invoke(
        app, ["auth", "setup", "--config", str(config)], input=_setup_input(client)
    )
    assert result.exit_code == 4  # auth/config unusable
    # the failure names the console steps and never echoes the secret token
    assert "enable" in result.output.lower() and "production" in result.output.lower()
    assert REFRESH not in result.output


def test_setup_declined_overwrite_exits_5_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    original = "# user's hand-written config — must survive\n"
    config.write_text(original, encoding="utf-8")
    monkeypatch.setattr(cli, "_typer_confirm", lambda text: False)  # decline overwrite
    client = _client_json(tmp_path)
    result = runner.invoke(
        app, ["auth", "setup", "--config", str(config)], input=_setup_input(client)
    )
    assert result.exit_code == 5  # incomplete (declined)
    assert config.read_text(encoding="utf-8") == original  # NOT overwritten


# --- the real probe's local checks (adapter side) ---------------------------
def test_real_probe_desktop_ok(tmp_path: Path) -> None:
    client = _client_json(tmp_path)
    assert (
        cli._RealEnvironmentProbe().client_file_state(str(client)).value == "desktop-ok"
    )


def test_real_probe_missing(tmp_path: Path) -> None:
    assert (
        cli._RealEnvironmentProbe().client_file_state(str(tmp_path / "nope.json")).value
        == "missing"
    )


def test_real_probe_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert cli._RealEnvironmentProbe().client_file_state(str(bad)).value == "malformed"


def test_real_probe_web_client_is_not_desktop(tmp_path: Path) -> None:
    web = tmp_path / "web.json"
    web.write_text(
        json.dumps({"web": {"client_id": "a", "client_secret": "b"}}), encoding="utf-8"
    )
    assert (
        cli._RealEnvironmentProbe().client_file_state(str(web)).value == "not-desktop"
    )


def test_project_id_shape_guard() -> None:
    assert cli._project_id_shape_ok("my-project-42")
    assert not cli._project_id_shape_ok("My-Project")  # uppercase
    assert not cli._project_id_shape_ok("1abc")  # too short / starts wrong
    assert not cli._project_id_shape_ok("has spaces here")


# --- the raw `auth login` path is untouched (composition, not fork) ---------
def test_auth_login_command_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_flow: None
) -> None:
    # the wizard COMPOSES login; login itself must still stand alone
    monkeypatch.setenv("HOME", str(tmp_path))
    client = _client_json(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[gmail]\ncredentials_path = "{client}"\ntoken_backend = "file"\n'
        f'token_file = "{tmp_path / "tok.json"}"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["auth", "login", "--config", str(cfg)])
    assert result.exit_code == 0
    assert REFRESH not in result.output
