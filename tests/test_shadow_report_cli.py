"""shadow report CLI (ticket #23, C3): the Phase 1 exit review — diff a
Shadow Pass run's would-apply verdicts against the mailbox's ACTUAL
labels. Deterministic per-message diff (ids + labels + scores + tiers
ONLY — subjects OUT, bodies out by construction, invariant #5);
ADDED / ALREADY-PRESENT / NOT-APPLIED outcomes; per-label summary with
the would-be tier mix; --json single-object contract; exit contract
0/2/3/4/5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from gmail_labeler.cli import app
from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
from gmail_labeler.gmail_errors import GmailTransientError
from test_shadow_cli import (
    READ_SURFACE,
    _backlog_messages,
    _prepare,
    _run_shadow,
    _shadow_cfg,
    _shadow_labels,
)

runner = CliRunner()


def _run_report(
    tmp_path: Path,
    *extra: str,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeGmailClient | None = None,
    cfg: Path | None = None,
) -> tuple[Result, FakeGmailClient]:
    import gmail_labeler.cli as cli

    fake = (
        fake
        if fake is not None
        else FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
    )
    monkeypatch.setattr(cli, "_shadow_client", lambda cfg_: fake)
    args = [
        "shadow",
        "report",
        "--config",
        str(cfg if cfg is not None else _shadow_cfg(tmp_path)),
        *extra,
    ]
    return runner.invoke(app, args), fake


def test_shadow_report_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC2: the report diffs the would-apply verdicts against the
    # actual labels — ADDED / ALREADY-PRESENT / NOT-APPLIED outcomes +
    # per-label summary with the would-be tier mix
    _prepare(tmp_path, monkeypatch)
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    report, _fake2 = _run_report(tmp_path, monkeypatch=monkeypatch)
    assert report.exit_code == 0, report.stderr
    # m1 (aa11): Shipping present → ALREADY-PRESENT; Money/Bills none
    # → NOT-APPLIED
    assert "aa11:" in report.stdout
    assert "ALREADY-PRESENT" in report.stdout
    assert "NOT-APPLIED" in report.stdout
    # m2 (aa22): Money/Bills absent → ADDED (high)
    assert "ADDED" in report.stdout
    # the per-label summary with the would-be tier mix
    assert "per-label:" in report.stdout
    assert "Money/Bills: added=1" in report.stdout
    assert "Shipping: added=1" in report.stdout
    # the header
    assert "shadow report: run=" in report.stdout
    assert "messages=4" in report.stdout


def test_shadow_report_refetches_actual_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the report re-fetches the ACTUAL labels (the labels may have
    # changed since the run): a label the user applied AFTER the run
    # flips ADDED → ALREADY-PRESENT
    _prepare(tmp_path, monkeypatch)
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    # m2 (aa22) had NO labels at run time → Money/Bills was ADDED. The
    # user then applies Money/Bills (L2) to m2:
    messages = _backlog_messages()
    messages[1] = FakeMessage(
        id="aa22",
        eml=messages[1].eml,
        label_ids=("INBOX", "L2"),  # Money/Bills now present
        thread_id="t2",
    )
    fake = FakeGmailClient(messages=messages, labels=_shadow_labels())
    report, _fake2 = _run_report(tmp_path, monkeypatch=monkeypatch, fake=fake)
    assert report.exit_code == 0, report.stderr
    # m2's Money/Bills is now ALREADY-PRESENT (re-fetched), not ADDED
    aa22_block = report.stdout.split("aa22:")[1].split("aa33:")[0]
    assert "ALREADY-PRESENT" in aa22_block
    assert "ADDED" not in aa22_block


def test_shadow_report_latest_run_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the default is the LATEST run (by run_ts): two runs → the report
    # is the second one
    _prepare(tmp_path, monkeypatch)
    r1, _f1 = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert r1.exit_code == 0, r1.stderr
    r2, _f2 = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert r2.exit_code == 0, r2.stderr
    report, _fake = _run_report(tmp_path, monkeypatch=monkeypatch)
    assert report.exit_code == 0, report.stderr
    # the reported run id is the second run's (from r2's summary)
    run2_id = r2.stderr.split("run=")[1].split()[0]
    assert f"run={run2_id}" in report.stdout


def test_shadow_report_run_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --run picks a specific run (not the latest)
    _prepare(tmp_path, monkeypatch)
    r1, _f1 = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert r1.exit_code == 0, r1.stderr
    r2, _f2 = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert r2.exit_code == 0, r2.stderr
    run1_id = r1.stderr.split("run=")[1].split()[0]
    report, _fake = _run_report(tmp_path, "--run", run1_id, monkeypatch=monkeypatch)
    assert report.exit_code == 0, report.stderr
    assert f"run={run1_id}" in report.stdout


def test_shadow_report_no_runs_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an empty DB (no runs) → exit 5, named (run `shadow run` first)
    _prepare(tmp_path, monkeypatch)
    report, _fake = _run_report(tmp_path, monkeypatch=monkeypatch)
    assert report.exit_code == 5
    assert "no shadow runs recorded" in report.stderr


def test_shadow_report_unknown_run_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a --run naming a non-existent run → exit 5, named
    _prepare(tmp_path, monkeypatch)
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    report, _fake2 = _run_report(
        tmp_path, "--run", "nonexistent", monkeypatch=monkeypatch
    )
    assert report.exit_code == 5
    assert "no shadow run" in report.stderr


def test_shadow_report_degraded_fetch_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a message whose labels can't be re-fetched is a per-message SKIP
    # (warning + count, the report continues) — the same posture as
    # shadow run
    _prepare(tmp_path, monkeypatch)
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    fake = FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
    fake.fail("get_message", GmailTransientError("network blip"))
    report, _fake2 = _run_report(tmp_path, monkeypatch=monkeypatch, fake=fake)
    assert report.exit_code == 0, report.stderr
    assert "WARNING: skipped message aa11" in report.stderr
    assert "skipped=1" in report.stdout
    assert "messages=3" in report.stdout


def test_shadow_report_bodies_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # invariant #5: bodies (and subjects) never enter the report —
    # ids + labels + scores + tiers only
    import logging

    marker = "SECRET-BODY-MARKER"
    _prepare(tmp_path, monkeypatch)
    messages = [
        FakeMessage(
            id="aa11",
            eml=(
                f"Subject: {marker} subj\r\nFrom: a@x.com\r\nTo: me@x.com\r\n"
                "Content-Type: text/plain\r\n\r\n" + f"{marker} body"
            ).encode(),
            label_ids=("INBOX", "L1"),
            thread_id="t1",
        ),
        FakeMessage(
            id="aa22",
            eml=(
                f"Subject: {marker} subj2\r\nFrom: a@x.com\r\nTo: me@x.com\r\n"
                "Content-Type: text/plain\r\n\r\n" + f"{marker} body2"
            ).encode(),
            label_ids=("INBOX",),
            thread_id="t2",
        ),
    ]
    scores = {
        f"{marker} subj\n{marker} body": (0.5, 0.95),
        f"{marker} subj2\n{marker} body2": (0.85, 0.3),
    }
    caplog.set_level(logging.DEBUG)
    fake = FakeGmailClient(messages=messages, labels=_shadow_labels())
    result, _f1 = _run_shadow(
        tmp_path, monkeypatch=monkeypatch, fake=fake, scores=scores
    )
    assert result.exit_code == 0, result.stderr
    report, _f2 = _run_report(tmp_path, monkeypatch=monkeypatch, fake=fake)
    assert report.exit_code == 0, report.stderr
    assert marker not in report.stdout
    assert marker not in report.stderr
    assert marker not in caplog.text


def test_shadow_report_json_single_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json contract: stdout is EXACTLY one parseable JSON object
    _prepare(tmp_path, monkeypatch)
    result, _f1 = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    report, _f2 = _run_report(tmp_path, "--json", monkeypatch=monkeypatch)
    assert report.exit_code == 0, report.stderr
    payload = json.loads(report.stdout)  # the whole stdout is one object
    assert payload["message_count"] == 4
    assert payload["skipped"] == 0
    assert payload["outcomes"] == {
        "added": 2,
        "already_present": 2,
        "not_applied": 4,
    }
    assert len(payload["messages"]) == 4
    first = payload["messages"][0]
    assert set(first) == {"message_id", "rows"}
    assert set(first["rows"][0]) == {
        "label",
        "probability",
        "band",
        "outcome",
        "tier",
    }
    assert len(payload["per_label"]) == 2
    # the per-label would-be tier mix reflects ONLY labels that would
    # ACTUALLY be added (skip-present labels are not applied, so not
    # counted): Money/Bills added=m2 (high); its m3 is already-present
    # (med band, NOT counted). Shipping added=m4 (med); its m1 is
    # already-present (high band, NOT counted).
    mb = next(p for p in payload["per_label"] if p["label"] == "Money/Bills")
    assert mb["would_be_high"] == 1  # m2
    assert mb["would_be_med"] == 0  # m3 is already-present (not added)
    sh = next(p for p in payload["per_label"] if p["label"] == "Shipping")
    assert sh["would_be_high"] == 0  # m1 is already-present (not added)
    assert sh["would_be_med"] == 1  # m4


def test_shadow_report_usage_empty_run_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an empty --run is a usage error (exit 2) — the latest-run default
    # is the ABSENT case, not an empty-string run id
    _prepare(tmp_path, monkeypatch)
    report, _fake = _run_report(tmp_path, "--run", "  ", monkeypatch=monkeypatch)
    assert report.exit_code == 2
    assert "non-empty" in report.stderr


def test_shadow_report_auth_gate_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the auth gate (exit 4 precedent): the report re-fetches the
    # actual labels via the read path
    _prepare(tmp_path, monkeypatch)
    import gmail_labeler.cli as cli
    from gmail_labeler.auth import AuthError

    def _boom(cfg):
        raise AuthError("no credentials_path set")

    monkeypatch.setattr(cli, "_shadow_client", _boom)
    result = runner.invoke(
        app,
        [
            "shadow",
            "report",
            "--config",
            str(_shadow_cfg(tmp_path)),
        ],
    )
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_shadow_report_all_skipped_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # T2: if EVERY message is skipped (labels can't be re-fetched), the
    # report has no evidence — refuse with exit 5 (mirroring shadow
    # run's zero-verdict posture). Partial-skip still exits 0.
    _prepare(tmp_path, monkeypatch)
    result, _f1 = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    # a client whose get_message ALWAYS fails → every message skipped
    fake = FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
    for _ in _backlog_messages():
        fake.fail("get_message", GmailTransientError("network blip"))
    report, _f2 = _run_report(tmp_path, monkeypatch=monkeypatch, fake=fake)
    assert report.exit_code == 5
    assert "no evidence to report" in report.stderr
    assert "all 4 message(s) were skipped" in report.stderr


def test_shadow_report_determinism_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the report is deterministic: rendering the same run twice yields
    # byte-identical stdout (text) and payload (--json)
    _prepare(tmp_path, monkeypatch)
    result, _f1 = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    r1, _a = _run_report(tmp_path, monkeypatch=monkeypatch)
    r2, _b = _run_report(tmp_path, monkeypatch=monkeypatch)
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert r1.stdout == r2.stdout  # byte-identical text
    r3, _c = _run_report(tmp_path, "--json", monkeypatch=monkeypatch)
    r4, _d = _run_report(tmp_path, "--json", monkeypatch=monkeypatch)
    assert r3.stdout == r4.stdout  # byte-identical json


def test_shadow_report_zero_gmail_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the report is read-only: the recorded calls are a subset of the
    # READ surface (no write method of any shape)
    _prepare(tmp_path, monkeypatch)
    result, _f1 = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    report, fake = _run_report(tmp_path, monkeypatch=monkeypatch)
    assert report.exit_code == 0, report.stderr
    methods = {call.split(":", 1)[0] for call in fake.calls}
    assert methods <= READ_SURFACE, f"write call recorded: {methods - READ_SURFACE}"
    assert "get_message" in methods
    assert "list_labels" in methods


def test_shadow_report_json_refusal_combos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json path shares the same refusal exits as text mode:
    # empty store (no runs) and unknown --run both exit 5 with a named
    # cause (stdout stays empty — no partial payload)
    _prepare(tmp_path, monkeypatch)
    # empty store: a fresh config dir with no runs
    empty_cfg = _shadow_cfg(tmp_path, store='db_path = "empty.db"\n')
    report, _f = _run_report(tmp_path, "--json", monkeypatch=monkeypatch, cfg=empty_cfg)
    assert report.exit_code == 5
    assert "no shadow runs recorded" in report.stderr
    assert report.stdout == ""  # no partial payload
    # unknown --run
    result, _f1 = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    report2, _f2 = _run_report(
        tmp_path, "--json", "--run", "nonexistent", monkeypatch=monkeypatch
    )
    assert report2.exit_code == 5
    assert "no shadow run" in report2.stderr
    assert report2.stdout == ""
