"""Ticket #28 C2 — the CLI wiring of the auth wizard.

The pure machine (test_wizard.py) is driven here through `typer.testing` with
the REAL probe + the OAuth exchange behind the SAME injectable seam `auth login`
uses today (monkeypatched ``InstalledAppFlow``) — no network, no real keyring.
The single test ``test_ac1_fresh_profile_setup_reaches_green_status`` IS
acceptance criterion #1 (a fresh profile, no config, reaches a working
`auth status` without reading source) — it is named loudly per the design.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
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


# NOTE (F1): the non-TTY-unsigned bare-auth contract (exit 4 + names `auth
# setup`) is pinned once, in tests/test_cli.py::
# test_auth_status_reports_and_exits_4_when_unsigned — this suite keeps only
# the TTY-launch and signed-in pins to avoid duplicating that assertion.
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
def test_setup_cancel_exits_130(tmp_path: Path) -> None:
    # REAL prompt path (no monkeypatch of _typer_ask): an empty stdin makes
    # typer.prompt hit EOF -> Abort -> _typer_ask maps it to WizardCancelled ->
    # _run_setup_wizard exits 130. This is the code the monkeypatched variant
    # below skips (review MINOR 4: the Abort/EOF->130 mapping was unpinned).
    config = tmp_path / "config.toml"
    result = runner.invoke(app, ["auth", "setup", "--config", str(config)], input="")
    assert result.exit_code == 130
    assert "setup cancelled" in (result.output + result.stderr)
    assert not config.exists()  # cancel wrote nothing


def test_setup_confirm_eof_exits_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # reach the overwrite CONFIRM (config exists), then EOF there -> the
    # _typer_confirm Abort/EOF->130 mapping (its own arm, not _typer_ask's).
    monkeypatch.setattr(
        cli._RealEnvironmentProbe, "keyring_available", lambda self: False
    )
    config = tmp_path / "config.toml"
    config.write_text("# existing\n", encoding="utf-8")
    client = _client_json(tmp_path)
    # answer project-id + client-json, then NOTHING (EOF at the confirm)
    result = runner.invoke(
        app,
        ["auth", "setup", "--config", str(config)],
        input=f"{PROJECT_ID}\n{client}\n",
    )
    assert result.exit_code == 130
    assert (
        config.read_text(encoding="utf-8") == "# existing\n"
    )  # confirm cancelled -> no write


def test_setup_cancelled_marker_exits_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gmail_labeler.wizard import WizardCancelled

    def _boom(prompt: object) -> str:
        raise WizardCancelled()

    # pins the _run_setup_wizard WizardCancelled -> 130 mapping directly
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


def test_setup_tilde_client_path_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_flow: None,
    headless: None,
) -> None:
    # review MAJOR 1: the prompt invites `~`. The value the probe verified must
    # be the value login receives — before the expanduser fix a `~/…` path
    # passed the local check then failed inside login with a misleading
    # console-side diagnosis (exit 4). It must now sign in (exit 0).
    monkeypatch.setenv("HOME", str(tmp_path))
    client = tmp_path / "desktop-client.json"
    client.write_text(
        json.dumps(
            {"installed": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}}
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.toml"
    result = runner.invoke(
        app,
        ["auth", "setup", "--config", str(config)],
        input=f"{PROJECT_ID}\n~/desktop-client.json\n",
    )
    assert result.exit_code == 0, result.output
    assert "signed in" in result.output.lower()


def test_setup_config_write_failure_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # review MINOR 6: a config-write OSError exits 5 (the contract), never a
    # raw traceback (which would exit 1). Parent path is a FILE -> mkdir fails.
    monkeypatch.setattr(
        cli._RealEnvironmentProbe, "keyring_available", lambda self: False
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    client = _client_json(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    config = blocker / "config.toml"
    result = runner.invoke(
        app, ["auth", "setup", "--config", str(config)], input=_setup_input(client)
    )
    assert result.exit_code == 5
    assert "Traceback" not in result.output
    # F3-3 one-surface: the diagnosis appears EXACTLY ONCE and ONLY on stderr —
    # a machine-side io.out echo (the old double-print) would land on stdout too.
    assert "could not write" in result.stderr.lower()
    assert result.stderr.lower().count("could not write") == 1
    assert "could not write" not in result.stdout.lower()  # NOT on stdout


def test_bare_auth_invalid_config_no_flag_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # review MAJOR 3: a config that EXISTS but is INVALID is a config error
    # (exit 4, message preserved) — NOT "no config found" (which would falsely
    # offer the wizard). Discovered via the no-flag search path.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        '[gmail]\nbogus_key = "x"\n', encoding="utf-8"
    )
    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: False)
    result = runner.invoke(app, ["auth"])
    assert result.exit_code == 4
    out = result.output + result.stderr
    assert (
        "config error" in out and "invalid config" in out
    )  # identical to the --config arm
    assert "no config.toml found" not in out  # the false claim is gone


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


def test_completed_but_not_signed_in_exits_5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_flow: None,
    headless: None,
) -> None:
    # A4: the "green status this ticket sells" surface. login succeeds but the
    # status check reports NOT signed in -> the wizard must NOT claim success:
    # exit 5 + the honest "not signed in" line (mutant: hardcode signed_in=True
    # in the result -> this reddens to exit 0).
    import gmail_labeler.auth as auth_mod
    from gmail_labeler.auth import AuthStatus

    monkeypatch.setattr(
        auth_mod,
        "status",
        lambda cfg: AuthStatus(False, "file", "stored token incomplete"),
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    client = _client_json(tmp_path)
    config = tmp_path / "config.toml"
    result = runner.invoke(
        app, ["auth", "setup", "--config", str(config)], input=_setup_input(client)
    )
    assert result.exit_code == 5
    assert "not signed in" in result.output.lower()


def test_status_check_crash_cli_exits_5_no_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_flow: None,
    headless: None,
) -> None:
    # A3 at the CLI layer: an injected status() that RAISES must map to exit 5
    # (the status-error arm), never a raw traceback (exit 1, off-contract). The
    # machine-layer test alone left the cli.py status-error->5 arm uncovered.
    import gmail_labeler.auth as auth_mod
    from gmail_labeler.auth import AuthStatus

    def _boom(cfg: object) -> AuthStatus:
        raise RuntimeError(f"status exploded {REFRESH}")

    monkeypatch.setattr(auth_mod, "status", _boom)
    monkeypatch.setenv("HOME", str(tmp_path))
    client = _client_json(tmp_path)
    config = tmp_path / "config.toml"
    result = runner.invoke(
        app, ["auth", "setup", "--config", str(config)], input=_setup_input(client)
    )
    assert result.exit_code == 5
    assert "Traceback" not in result.output
    # F3-3 one-surface: exactly once, stderr only (no machine-side stdout echo)
    assert "status check failed" in result.stderr.lower()
    assert result.stderr.lower().count("status check failed") == 1
    assert "status check failed" not in result.stdout.lower()
    assert REFRESH not in result.output  # the crash text is never echoed


def test_status_broken_pipe_is_cancel_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_flow: None,
    headless: None,
) -> None:
    # A5 symmetric case (review MINOR): a BrokenPipeError escaping the status
    # check must reach the CLI's 130 handler, not be caught as a status-error (5).
    import gmail_labeler.auth as auth_mod
    from gmail_labeler.auth import AuthStatus

    def _bp(cfg: object) -> AuthStatus:
        raise BrokenPipeError()

    monkeypatch.setattr(auth_mod, "status", _bp)
    monkeypatch.setenv("HOME", str(tmp_path))
    client = _client_json(tmp_path)
    config = tmp_path / "config.toml"
    result = runner.invoke(
        app, ["auth", "setup", "--config", str(config)], input=_setup_input(client)
    )
    assert result.exit_code == 130


def test_broken_pipe_stdout_closed_is_cancel_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A5: `auth setup | head` closes stdout mid-run -> BrokenPipeError from the
    # output sink. Cancel-class (130), NOT mislabeled "setup failed writing
    # config.toml" (the old broad OSError catch's lie — nothing was being written).
    def _broken(text: str) -> None:
        raise BrokenPipeError()

    monkeypatch.setattr(cli, "_echo_out", _broken)
    config = tmp_path / "config.toml"
    result = runner.invoke(app, ["auth", "setup", "--config", str(config)], input="")
    assert result.exit_code == 130
    assert "writing" not in (result.output + result.stderr).lower()


def test_broken_pipe_in_login_is_cancel_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, headless: None
) -> None:
    # A5 (review MINOR): a BrokenPipeError escaping the composed LOGIN (auth.login
    # prints the auth URL to stdout; a closed pipe there) must reach the CLI's 130
    # handler, NOT be swallowed by the login arm's `except Exception` as a
    # console-side sign-in failure (exit 4).
    import gmail_labeler.auth as auth_mod

    monkeypatch.setenv("HOME", str(tmp_path))

    def _broken_login(cfg: object) -> None:
        raise BrokenPipeError()

    monkeypatch.setattr(auth_mod, "login", _broken_login)
    client = _client_json(tmp_path)
    config = tmp_path / "config.toml"
    result = runner.invoke(
        app, ["auth", "setup", "--config", str(config)], input=_setup_input(client)
    )
    assert result.exit_code == 130


def test_auth_explicit_invalid_config_exits_4(tmp_path: Path) -> None:
    # A6: an EXPLICIT --config that exists but is INVALID refuses identically to
    # the discovered-path arm — exit 4 + "config error"/"invalid config", never
    # the "no config.toml found" false claim (m3 survivor now pinned).
    cfg = tmp_path / "config.toml"
    cfg.write_text('[gmail]\nbogus_key = "x"\n', encoding="utf-8")
    result = runner.invoke(app, ["auth", "--config", str(cfg)])
    assert result.exit_code == 4
    out = result.output + result.stderr
    assert "config error" in out and "invalid config" in out
    assert "no config.toml found" not in out


def test_setup_decline_via_real_n_input_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B10: the decline path exercised through the REAL typer.confirm (an 'n' on
    # stdin), not a monkeypatched _typer_confirm — and it must change NOTHING.
    monkeypatch.setattr(
        cli._RealEnvironmentProbe, "keyring_available", lambda self: False
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / "config.toml"
    original = "# user's hand-written config — must survive\n"
    config.write_text(original, encoding="utf-8")
    client = _client_json(tmp_path)
    # project-id, client-json, then 'n' at the overwrite confirm
    result = runner.invoke(
        app,
        ["auth", "setup", "--config", str(config)],
        input=f"{PROJECT_ID}\n{client}\nn\n",
    )
    assert result.exit_code == 5
    assert config.read_text(encoding="utf-8") == original  # byte-identical


def test_bare_auth_no_config_hint_names_paths_and_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B7: the fresh-profile (no config anywhere) bare `auth` keeps exit 4 AND the
    # informative hint that the C2 rewrite dropped — looked-in paths + copy
    # example.toml + the `auth setup` route.
    empty_home = tmp_path / "home"
    empty_home.mkdir()
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.chdir(tmp_path)  # no ./config.toml here either
    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: False)
    result = runner.invoke(app, ["auth"])
    assert result.exit_code == 4
    out = result.output + result.stderr
    assert "~/.config/gmail-labeler/" in out  # names where it looked
    assert "example.toml" in out  # the copy hint
    assert "auth setup" in out  # the wizard route


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


# --- the PROCESS-boundary tests (F3 MAJOR) ----------------------------------
# CliRunner runs the CLI IN-PROCESS, so it always saw the intended exit code.
# These spawn the real interpreter and assert the code is true at the BOUNDARY —
# where a buffered stdout that only flushes at shutdown used to turn 130 into 120.
def _boundary_env(tmp_path: Path) -> dict[str, str]:
    # run THIS src tree (not a stale install) — the daemon subprocess precedent
    src = str(Path(cli.__file__).parent.parent)
    return {
        **os.environ,
        "PYTHONPATH": src,
        "HOME": str(tmp_path),
        "PYTHONTZPATH": "/nonexistent/tzdb",
    }


def test_boundary_closed_stdout_exits_130(tmp_path: Path) -> None:
    # `auth setup | head`: the reader closes stdout's read end before the child
    # flushes. Must exit 130 with NO interpreter-shutdown flush error. Pre-fix
    # this returned 120 + "Exception ignored on flushing sys.stdout".
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gmail_labeler.cli",
            "auth",
            "setup",
            "--config",
            str(tmp_path / "config.toml"),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_boundary_env(tmp_path),
        cwd=str(tmp_path),
    )
    assert p.stdout is not None and p.stderr is not None  # PIPE was requested
    p.stdout.close()  # the reader (`head`) is gone
    err = p.stderr.read().decode()
    p.stderr.close()
    rc = p.wait()
    assert rc == 130, f"expected 130 at the boundary, got {rc}: {err!r}"
    assert "Exception ignored" not in err
    assert "BrokenPipeError" not in err
    assert "Traceback" not in err
    # Self-validating: 130 is ALSO what the EOF-cancel path returns (stdin is
    # DEVNULL), so a bare rc check would pass even if the pipe never broke. The
    # closed-stdout handler prints NO "setup cancelled." line (unlike the cancel
    # path), so its absence proves the child actually hit the BrokenPipeError
    # branch — a future speed-up that let the prompt EOF first would fail here.
    assert "setup cancelled" not in err


def test_boundary_lost_stdout_fd_closed_exits_130(tmp_path: Path) -> None:
    # `auth setup >&-`: fd 1 is CLOSED (sys.stdout is None), not a pipe. Must
    # exit 130 with a one-line stderr diagnosis — pre-fix it crashed with a
    # traceback and exited 1 (off the documented 0/2/3/4/5/130 contract).
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gmail_labeler.cli",
            "auth",
            "setup",
            "--config",
            str(tmp_path / "config.toml"),
        ],
        stdin=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=_boundary_env(tmp_path),
        cwd=str(tmp_path),
        preexec_fn=lambda: os.close(1),  # close fd 1 in the child, like >&-
    )
    assert p.stderr is not None  # PIPE was requested
    err = p.stderr.read().decode()
    p.stderr.close()
    rc = p.wait()
    assert rc == 130, f"expected 130 for a closed stdout fd, got {rc}: {err!r}"
    assert "Traceback" not in err
    assert "stdout is unavailable" in err


def test_boundary_merged_streams_exits_130(tmp_path: Path) -> None:
    # `auth setup 2>&1 | head`: stdout AND stderr share ONE pipe. When the reader
    # exits, BOTH fds are dead. The cancel diagnosis is written to STDERR — if that
    # write EPIPEs out of the handler (or the shutdown flush of stderr fails), the
    # process exits 1/120, off the closed set (F3 round-2 MAJOR: the earlier fix
    # silenced fd 1 only). The handlers must retarget BOTH fds so 130 holds even
    # when the two streams share a pipe. Deterministic in outcome: whichever of the
    # intro-write (fd 1) or the cancel-echo (fd 2) trips first, both roads exit 130.
    read_fd, write_fd = os.pipe()
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gmail_labeler.cli",
            "auth",
            "setup",
            "--config",
            str(tmp_path / "config.toml"),
        ],
        stdin=subprocess.DEVNULL,
        stdout=write_fd,
        stderr=write_fd,  # the SAME pipe as stdout, like `2>&1`
        env=_boundary_env(tmp_path),
        cwd=str(tmp_path),
    )
    os.close(write_fd)  # parent drops its write-end dup
    os.close(read_fd)  # the reader (`head`) is gone -> both child fds EPIPE on write
    rc = p.wait()
    assert rc == 130, f"expected 130 with merged streams, got {rc}"


def test_boundary_dead_stderr_cancel_diagnosis_exits_130(tmp_path: Path) -> None:
    # The load-bearing pin for the F3 round-2 MAJOR. deepseek's strace showed the
    # real failure was `write(2, "setup cancelled.") = -1 EPIPE` — the CANCEL
    # diagnosis going to a dead stderr. To reach that path deterministically the
    # two streams must be decoupled: stdout stays a live pipe (so the intro write
    # SUCCEEDS and no BrokenPipeError fires there), while stderr's reader is closed
    # so the first prompt's EOF -> cancel -> diagnosis echo hits a dead fd 2. The
    # handler must swallow that EPIPE and retarget both fds, or the process exits 1.
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gmail_labeler.cli",
            "auth",
            "setup",
            "--config",
            str(tmp_path / "config.toml"),
        ],
        stdin=subprocess.DEVNULL,  # the prompt EOFs -> WizardCancelled
        stdout=out_w,
        stderr=err_w,
        env=_boundary_env(tmp_path),
        cwd=str(tmp_path),
    )
    os.close(out_w)
    os.close(err_w)  # parent drops its write-end dups
    os.close(err_r)  # stderr reader gone -> the cancel echo to fd 2 EPIPEs
    with contextlib.suppress(OSError):
        os.read(out_r, 65536)  # drain the intro so the stdout write never blocks
    os.close(out_r)
    rc = p.wait()
    assert rc == 130, (
        f"expected 130 when the cancel diagnosis hits a dead stderr, got {rc}"
    )
