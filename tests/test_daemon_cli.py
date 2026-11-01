"""daemon CLI (ticket #25, C3): the `daemon` command — identity gates
first (the shadow-run shape), startup tier resolution ONCE via
ensure_label (the collision ValueError → exit 4 NAMING the [labels]
key — the bare-ValueError arm per the C0 Protocol contract), the
interruptible poll sleep, SIGTERM/SIGINT graceful shutdown, and the
exit-code contract 0/2/3/4/5 (project contract: 4 = auth/config
unusable — the daemon's collision/config/auth refusals all land there;
3 stays the reserved-not-implemented arm, unreachable).
"""

from __future__ import annotations

import json
import signal
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from gmail_labeler.appdb import AppDB
from gmail_labeler.cli import app
from gmail_labeler.fake_gmail import FakeGmailClient
from gmail_labeler.gmail_client import HistoryPage, LabelInfo
from gmail_labeler.gmail_errors import GmailAuthError
from test_shadow_cli import (
    _backlog_messages,
    _backlog_scores,
    _prepare,
    _shadow_cfg,
    _shadow_factory,
    _shadow_labels,
)

runner = CliRunner()


def _daemon_cfg(tmp_path: Path, *, poll: int | None = None) -> Path:
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    if poll is not None:
        cfg.write_text(
            f"[gmail]\nhistory_poll_seconds = {poll}\n"
            + cfg.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return cfg


def _preset_cursor(tmp_path: Path, cursor: int) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    db.set_history_cursor(cursor)
    db.close()


def _run_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake: FakeGmailClient | None = None,
    cfg: Path | None = None,
    extra: tuple[str, ...] = (),
    preset_cursor: int | None = 900,
    install_capture: list[tuple[int, object]] | None = None,
    sleep_record: list[float] | None = None,
    fire_handler_in_sleep: bool = False,
) -> tuple[Result, FakeGmailClient]:
    import gmail_labeler.cli as cli

    fake = fake or FakeGmailClient(
        messages=_backlog_messages(),
        labels=_shadow_labels(),
        current_history_id=900,
    )
    monkeypatch.setattr(cli, "_daemon_client", lambda cfg_: fake)
    monkeypatch.setattr(
        cli, "_daemon_model_factory", _shadow_factory(_backlog_scores())
    )
    from datetime import UTC, datetime

    monkeypatch.setattr(
        cli, "_now", lambda: datetime(2026, 11, 1, 12, 30, 0, tzinfo=UTC)
    )

    # the signal-install seam: capture instead of touching the test
    # process's dispositions; the handler is REAL (a real threading.Event)
    captured: dict[str, object] = {}
    real_signal = cli.signal.signal

    def fake_signal(sig: int, handler: object) -> object:
        if install_capture is not None:
            install_capture.append((sig, handler))
        captured["last_handler"] = handler
        return real_signal  # report a "previous" handler, install nothing

    monkeypatch.setattr(cli.signal, "signal", fake_signal)

    def fake_sleep(flag: object, seconds: float) -> None:
        if sleep_record is not None:
            sleep_record.append(seconds)
        if fire_handler_in_sleep:
            # the mid-sleep SIGTERM: the REAL handler fires (it is the
            # command's own closure over the real Event)
            handler = captured.get("last_handler")
            assert callable(handler)
            handler(signal.SIGTERM, None)  # pyright: ignore[reportCallIssue]
        else:
            # default: stop after this sleep (the graceful-exit demo)
            flag.set()  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr(cli, "_sleep_until_stopped", fake_sleep)
    if preset_cursor is not None:
        _preset_cursor(tmp_path, preset_cursor)
    cfg = cfg or _daemon_cfg(tmp_path)
    result = runner.invoke(
        app,
        [
            "daemon",
            "--config",
            str(cfg),
            "--bands",
            str(tmp_path / "bands.json"),
            "--graduation",
            str(tmp_path / "graduation.json"),
            *extra,
        ],
    )
    return result, fake


def test_daemon_wired_idle_tick_exits_0_and_resolves_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the golden idle path: gates pass, the three tier names resolve via
    # ensure_label EXACTLY ONCE each (startup, not per-tick), the idle
    # delta commits its cursor advance, the sleep lands, the flag exits
    # clean 0
    _prepare(tmp_path, monkeypatch)
    result, fake = _run_daemon(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert "daemon: serving" in result.stderr
    assert "daemon: stopped" in result.stderr
    ensures = [c for c in fake.calls if c.startswith("ensure_label:")]
    assert ensures == [
        "ensure_label:ai-applied",
        "ensure_label:ai-high",
        "ensure_label:ai-med",
    ]
    assert fake.calls.count("history:900:") == 1
    db = AppDB(tmp_path / "state.db")
    db.open()
    assert db.get_history_cursor() == 900
    db.close()


def test_daemon_startup_resolution_precedes_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # exact-call pin: bridge list → three ensure_label round-trips →
    # THEN the poll (identity gates first, then serve)
    _prepare(tmp_path, monkeypatch)
    result, fake = _run_daemon(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.stderr
    first_history = fake.calls.index("history:900:")
    assert fake.calls[:5] == [
        "list_labels",  # the bridge
        "ensure_label:ai-applied",
        "list_labels",  # the ensure round-trips (Q1 ruling: signature
        # kept, one round-trip per name)
        "ensure_label:ai-high",
        "list_labels",
    ]
    assert fake.calls[5] == "ensure_label:ai-med"
    assert first_history == 7  # bridge + 3x (ensure + its list round-trip)


def test_daemon_serves_a_write_through_the_full_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CLI→loop→store integration: a delta with one would-apply message
    # writes (managed + umbrella + tier ids, sorted) and commits the
    # row + cursor in one transaction
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(
        messages=_backlog_messages(),
        labels=_shadow_labels(),
        history_pages=[
            HistoryPage(
                history_id=None,
                next_page_token=None,
                records=({"id": 5, "messages": [{"id": "aa22", "threadId": "t2"}]},),
            )
        ],
        current_history_id=900,
    )
    result, _ = _run_daemon(tmp_path, monkeypatch, fake=fake)
    assert result.exit_code == 0, result.stderr
    label_writes = [w for w in fake.writes if w["method"] == "apply_labels"]
    assert len(label_writes) == 1
    assert label_writes[0]["message_id"] == "aa22"
    assert label_writes[0]["body"] == {
        "addLabelIds": ["L2", "Label_fake_1", "Label_fake_2"]
    }
    db = AppDB(tmp_path / "state.db")
    db.open()
    rows = db.applications_for_message("aa22")
    assert [(r.label, r.source) for r in rows] == [("Money/Bills", "daemon")]
    assert db.get_history_cursor() == 900
    db.close()


def test_daemon_poll_cadence_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # history_poll_seconds reaches the sleep EXACTLY (config.py: ge=5
    # default 30 — 7 is a legal override)
    _prepare(tmp_path, monkeypatch)
    sleeps: list[float] = []
    result, _ = _run_daemon(
        tmp_path,
        monkeypatch,
        cfg=_daemon_cfg(tmp_path, poll=7),
        sleep_record=sleeps,
    )
    assert result.exit_code == 0, result.stderr
    assert sleeps == [7]


def test_daemon_sigterm_mid_sleep_exits_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the AC pin at the CLI layer: the REAL handler (captured from the
    # real signal.install seam) fires mid-sleep → the loop exits 0
    # WITHOUT a second poll, cursor untouched
    _prepare(tmp_path, monkeypatch)
    install: list[tuple[int, object]] = []
    result, fake = _run_daemon(
        tmp_path,
        monkeypatch,
        install_capture=install,
        fire_handler_in_sleep=True,
    )
    assert result.exit_code == 0, result.stderr
    assert fake.calls.count("history:900:") == 1  # no second tick
    assert "daemon: stopped" in result.stderr
    # both dispositions were registered (and restored in finally)
    assert [sig for sig, _ in install] == [signal.SIGTERM, signal.SIGINT] * 2


def test_daemon_usage_refusal_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # empty --bands is a usage refusal BEFORE any work (the shadow precedent)
    _prepare(tmp_path, monkeypatch)
    import gmail_labeler.cli as cli

    result = runner.invoke(
        cli.app,
        [
            "daemon",
            "--config",
            str(_daemon_cfg(tmp_path)),
            "--bands",
            " ",
            "--graduation",
            str(tmp_path / "graduation.json"),
        ],
    )
    assert result.exit_code == 2
    assert "PATH must be non-empty" in result.stderr


def test_daemon_config_error_exit_4(tmp_path: Path) -> None:
    # the PROJECT contract (cli.py epilog + every command suite):
    # config load failure = 4 (auth/config unusable), NOT 2 — the
    # design brief's "config → 2 / auth → 3" line contradicts the
    # contract and the brief's OWN collision→4 mapping; implemented per
    # the contract, flagged as an Open Question in the ticket report
    import gmail_labeler.cli as cli

    bad = tmp_path / "bad.toml"
    bad.write_text('[labels]\napplied = ""\n', encoding="utf-8")
    result = runner.invoke(
        cli.app, ["daemon", "--config", str(bad), "--bands", "b", "--graduation", "g"]
    )
    assert result.exit_code == 4
    assert "config error" in result.stderr


def test_daemon_auth_gate_exit_4_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path, monkeypatch)
    import gmail_labeler.cli as cli
    from gmail_labeler.auth import AuthError

    def _boom(cfg_: object) -> None:
        raise AuthError("not signed in")

    monkeypatch.setattr(cli, "_daemon_client", _boom)
    result = runner.invoke(
        cli.app,
        [
            "daemon",
            "--config",
            str(_daemon_cfg(tmp_path)),
            "--bands",
            str(tmp_path / "bands.json"),
            "--graduation",
            str(tmp_path / "graduation.json"),
        ],
    )
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_daemon_midrun_auth_error_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the loop's propagating arm lands on the contract: GmailAuthError
    # → 4 (human-gated), never a traceback
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(
        messages=_backlog_messages(),
        labels=_shadow_labels(),
        current_history_id=900,
    )
    fake.fail("history", GmailAuthError("token revoked"))
    install: list[tuple[int, object]] = []
    result, _ = _run_daemon(tmp_path, monkeypatch, fake=fake, install_capture=install)
    assert result.exit_code == 4
    assert "auth:" in result.stderr
    # handlers restored on the exit-4 path too (the finally covers it)
    assert len(install) == 4  # 2 installs + 2 restores


def test_daemon_collision_refusal_exit_4_names_the_labels_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the carryover's headline: ensure_label's BARE ValueError (the C0
    # Protocol contract: not the gmail_errors taxonomy) maps to exit 4
    # (config error) and the message NAMES the offending [labels] key
    # — never the exit-1 "a bug" landing a bare ValueError would take
    _prepare(tmp_path, monkeypatch)
    labels = [*_shadow_labels(), LabelInfo("IMPORTANT", "ai-high", "system")]
    fake = FakeGmailClient(
        messages=_backlog_messages(), labels=labels, current_history_id=900
    )
    result, _ = _run_daemon(tmp_path, monkeypatch, fake=fake)
    assert result.exit_code == 4
    assert "[labels] high" in result.stderr
    assert "collides with a System Label" in result.stderr


def test_daemon_ensure_label_auth_error_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the review round's BLOCKER (both families, executed probes):
    # ensure_label's own list/create round-trips raise the gmail_errors
    # family; a token revoked between the auth gate and startup
    # resolution must land on 4 (human-gated), never exit-1 traceback
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(
        messages=_backlog_messages(),
        labels=_shadow_labels(),
        current_history_id=900,
    )
    fake.fail("ensure_label", GmailAuthError("token revoked"))
    result, _ = _run_daemon(tmp_path, monkeypatch, fake=fake)
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_daemon_ensure_label_transient_exit_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a rate-limit blip during the 3-6 startup wire calls of a
    # long-running service: operation failure (5), named, no traceback
    _prepare(tmp_path, monkeypatch)
    from gmail_labeler.gmail_errors import GmailTransientError

    fake = FakeGmailClient(
        messages=_backlog_messages(),
        labels=_shadow_labels(),
        current_history_id=900,
    )
    fake.fail("ensure_label", GmailTransientError("blip"))
    result, _ = _run_daemon(tmp_path, monkeypatch, fake=fake)
    assert result.exit_code == 5
    assert "tier label resolution failed" in result.stderr


def test_daemon_apply_auth_error_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F1 MAJOR arm (c) at the CLI seam: auth on the WRITE re-raises out
    # of the loop (never held-forever) and lands on the contract's 4
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(
        messages=_backlog_messages(),
        labels=_shadow_labels(),
        history_pages=[
            HistoryPage(
                history_id=None,
                next_page_token=None,
                records=({"id": 5, "messages": [{"id": "aa22", "threadId": "t2"}]},),
            )
        ],
        current_history_id=900,
    )
    fake.fail("apply_labels", GmailAuthError("token revoked mid-delta"))
    result, _ = _run_daemon(tmp_path, monkeypatch, fake=fake)
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_daemon_state_db_fail_fast_exit_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S5 posture rides the daemon: a corrupt state DB is refused BEFORE
    # any network work
    _prepare(tmp_path, monkeypatch)
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a sqlite file")
    result, fake = _run_daemon(
        tmp_path, monkeypatch, preset_cursor=None, extra=("--db", str(corrupt))
    )
    assert result.exit_code == 5
    assert "state DB unusable" in result.stderr
    assert fake.calls == []


def test_daemon_tampered_cursor_exit_5_not_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the corrupt-row doctrine at the CLI edge: a hand-edited cursor
    # refuses with exit 5 (operation failure), never an exit-1 traceback
    _prepare(tmp_path, monkeypatch)
    _preset_cursor(tmp_path, 900)
    conn = sqlite3.connect(str(tmp_path / "state.db"))
    conn.execute("UPDATE daemon_state SET history_cursor = 0")
    conn.commit()
    conn.close()
    install: list[tuple[int, object]] = []
    result, _ = _run_daemon(
        tmp_path, monkeypatch, preset_cursor=None, install_capture=install
    )
    assert result.exit_code == 5
    assert "state DB" in result.stderr
    assert len(install) == 4  # restored on the exit-5 path as well


def test_daemon_tampered_bands_artifact_exit_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the artifact refusals land on 5 (the shadow doctrine): a bands
    # artifact whose identity fields were hand-edited dies on its own
    # CONTENT HASH (from_json refuses) — the hash binding makes the
    # drift-detection parse-level, and the daemon refuses either way
    _prepare(tmp_path, monkeypatch)
    bands = tmp_path / "bands.json"
    payload = json.loads(bands.read_text(encoding="utf-8"))
    payload["encoder_id"] = "other-encoder"
    bands.write_text(json.dumps(payload), encoding="utf-8")
    result, _ = _run_daemon(tmp_path, monkeypatch)
    assert result.exit_code == 5
    assert "bands artifact" in result.stderr


def test_daemon_exit_3_stays_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the contract's 3-arm is reserved for WIRED-BUT-STUBBED commands;
    # daemon is wired (listed in --help) — scripts can branch on
    # 0/2/4/5 for it today, exactly as the epilog states
    import gmail_labeler.cli as cli

    help_result = runner.invoke(cli.app, ["--help"])
    assert help_result.exit_code == 0
    assert "daemon" in help_result.output


def test_daemon_bodies_absent_cli_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # invariant #5 end-to-end through the CLI: the marker rides a real
    # scored message; stdout/stderr/caplog stay clean
    import logging

    caplog.set_level(logging.DEBUG)
    marker = "SECRET-BODY-MARKER"
    from test_shadow_cli import _eml

    messages = _backlog_messages()
    for m in messages:
        if m.id == "aa22":
            m.eml = _eml("backlog two", f"{marker} body two")
    scores = dict(_backlog_scores())
    scores.pop("backlog two\nbody two")
    scores[f"backlog two\n{marker} body two"] = (0.85, 0.3)
    import gmail_labeler.cli as cli

    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(
        messages=messages,
        labels=_shadow_labels(),
        history_pages=[
            HistoryPage(
                history_id=None,
                next_page_token=None,
                records=({"id": 5, "messages": [{"id": "aa22", "threadId": "t2"}]},),
            )
        ],
        current_history_id=900,
    )
    monkeypatch.setattr(cli, "_daemon_model_factory", _shadow_factory(scores))
    result, _ = _run_daemon(tmp_path, monkeypatch, fake=fake)
    assert result.exit_code == 0, result.stderr
    # non-vacuity: the leak path (the predict-failure warning quoting
    # the unprogrammed text) MUST have executed this run
    assert "prediction failed" in caplog.text
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert marker not in caplog.text
