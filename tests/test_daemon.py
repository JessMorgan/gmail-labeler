"""daemon.py (ticket #25): the poll loop — Durable Handling commit
point (write FIRST, then ONE transaction of rows + cursor: COMMIT =
handled, invariant #3), skip-present replay idempotency (the crash
matrix's convergence proof), the 404 resync arm, graceful shutdown,
and the ids-only logging surface (invariant #5, caplog pins).

The loop is FULLY injected (client/db/model/clock/sleep/should_stop) —
tests never real-sleep. The crash matrix: injected-raise between every
persistence step (mid-page after-write, mid-transaction, after-commit —
the _CrashOn precedent, proxy-wrapped) PLUS a REAL os._exit subprocess
kill for the after-write row.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sqlite3
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gmail_labeler.appdb import AppDB, AppDBError
from gmail_labeler.daemon import ServingContext, bootstrap_cursor, daemon_loop
from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
from gmail_labeler.gmail_client import HistoryPage, LabelInfo
from gmail_labeler.gmail_errors import (
    GmailAuthError,
    GmailPermanentError,
    GmailResyncError,
    GmailTransientError,
)
from gmail_labeler.model import FakeModel

TS = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)
VOCAB = ("Money/Bills", "Shipping")  # canonical sorted Label Paths
SRC = str(Path(__file__).resolve().parents[1] / "src")


def _eml(subject: str, body: str) -> bytes:
    return (
        f"Subject: {subject}\r\nFrom: a@x.com\r\nTo: me@x.com\r\n"
        "Content-Type: text/plain\r\n\r\n" + body
    ).encode()


def _messages() -> list[FakeMessage]:
    """m1 (aa11): Shipping present (skip-present candidate); m2 (aa22):
    nothing present. history_id programmed for the bootstrap pins."""
    return [
        FakeMessage(
            id="aa11",
            eml=_eml("one", "body one"),
            label_ids=("INBOX", "L1"),
            thread_id="t1",
            history_id=777,
        ),
        FakeMessage(
            id="aa22",
            eml=_eml("two", "body two"),
            label_ids=("INBOX",),
            thread_id="t2",
            history_id=777,
        ),
    ]


def _labels() -> list[LabelInfo]:
    return [
        LabelInfo("INBOX", "INBOX", "system"),
        LabelInfo("L1", "Shipping", "user"),
        LabelInfo("L2", "Money/Bills", "user"),
        LabelInfo("LA", "ai-applied", "user"),
        LabelInfo("LH", "ai-high", "user"),
        LabelInfo("LM", "ai-med", "user"),
    ]


def _scores() -> dict[str, tuple[float, ...]]:
    """m1: Shipping high (present → skip-present); m2: Money/Bills high
    (absent → the write)."""
    return {
        "one\nbody one": (0.2, 0.95),
        "two\nbody two": (0.9, 0.1),
    }


def _ctx(**over: Any) -> ServingContext:
    base: dict[str, Any] = {
        "bands": {"Money/Bills": (0.8, 0.6), "Shipping": (0.8, 0.6)},
        "graduated": frozenset({"Money/Bills", "Shipping"}),
        "system_labels": frozenset({"INBOX"}),
        "id_to_name": {"L1": "Shipping", "L2": "Money/Bills"},
        "managed_label_ids": {"Money/Bills": "L2", "Shipping": "L1"},
        "tier_label_ids": {"ai-applied": "LA", "ai-high": "LH", "ai-med": "LM"},
    }
    base.update(over)
    return ServingContext(**base)  # pyright: ignore[reportArgumentType]


def _record(*message_ids: str, hid: int = 1) -> dict[str, object]:
    return {
        "id": hid,
        "messages": [{"id": m, "threadId": f"t{m}"} for m in message_ids],
    }


def _page(*records: dict[str, object], token: str | None = None) -> HistoryPage:
    return HistoryPage(history_id=None, next_page_token=token, records=records)


class _LoopRun:
    """Driver: the fake sleep counts ticks; the flag flips after
    ``stop_after_ticks`` sleeps (the loop checks before each tick)."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self.stopped = False
        self.stop_after = 1

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if len(self.sleeps) >= self.stop_after:
            self.stopped = True

    def should_stop(self) -> bool:
        return self.stopped


def _db(tmp_path: Path) -> AppDB:
    db = AppDB(tmp_path / "state.db")
    db.open()
    return db


def _run(
    tmp_path: Path,
    client: Any,
    *,
    db: Any | None = None,
    model: Any | None = None,
    ctx: ServingContext | None = None,
    clock: Any = None,
    poll_seconds: int = 30,
    stop_after_ticks: int = 1,
) -> int:
    driver = _LoopRun()
    driver.stop_after = stop_after_ticks
    own_db = db is None
    run_db: Any = db if db is not None else _db(tmp_path)
    try:
        return daemon_loop(
            client=client,
            db=run_db,
            model=model or FakeModel(VOCAB, scores=_scores()),
            ctx=ctx or _ctx(),
            poll_seconds=poll_seconds,
            clock=clock or (lambda: TS),
            sleep=driver.sleep,
            should_stop=driver.should_stop,
        )
    finally:
        if own_db:
            run_db.close()


# -- bootstrap (the export prior-art path: list_messages + get_message,
#    the sync guide's full-sync cursor source) --------------------------------


def test_bootstrap_establishes_cursor_from_most_recent_message(
    tmp_path: Path,
) -> None:
    # fresh daemon (no cursor): the guide's full-sync bootstrap —
    # list_messages, get_message(the FIRST stub = the most recent
    # message), store its per-message historyId. Exact-call pin: the
    # bootstrap pair, then the poll from the new cursor.
    fake = FakeGmailClient(
        messages=_messages(), labels=_labels(), current_history_id=777
    )
    db = _db(tmp_path)
    # tick 1 establishes the cursor (bootstrap pair), tick 2 polls from
    # it — the bootstrap tick does NOT also poll (one wake per tick)
    assert _run(tmp_path, fake, db=db, stop_after_ticks=2) == 0
    assert db.get_history_cursor() == 777
    assert fake.calls[:3] == ["list_messages::", "get_message:aa11", "history:777:"]
    db.close()


def test_bootstrap_empty_mailbox_holds_no_cursor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # no messages → no per-message historyId → the cursor stays
    # un-established (named warning, retry next tick) — never a guess
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(messages=[], labels=_labels(), current_history_id=5)
    db = _db(tmp_path)
    assert _run(tmp_path, fake, db=db, stop_after_ticks=2) == 0
    assert db.get_history_cursor() is None
    assert "cannot establish history cursor" in caplog.text
    # every tick retried the bootstrap — no history call was ever paid
    # for without a cursor
    assert all(not c.startswith("history:") for c in fake.calls)
    db.close()


def test_bootstrap_history_id_absent_holds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # the tolerance branch (the internalDate doctrine): the wire omitted
    # historyId on the newest message → hold, never invent
    caplog.set_level(logging.WARNING)
    msgs = _messages()
    for m in msgs:
        m.history_id = None
    fake = FakeGmailClient(messages=msgs, labels=_labels(), current_history_id=777)
    db = _db(tmp_path)
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() is None
    assert "cannot establish history cursor" in caplog.text
    db.close()


# -- the poll: decide, write FIRST, commit rows+cursor once ------------------


def test_page_writes_first_then_commits_rows_and_cursor(tmp_path: Path) -> None:
    # the Durable Handling commit point: apply_labels BEFORE the ONE
    # commit_page(rows + cursor) — order pinned on the call log, the
    # write carries the FULL composed set (managed + umbrella + band
    # tier), and the cursor takes the PAGE's id (the mailbox CURRENT),
    # not the request's
    order: list[str] = []

    class _OrderClient(FakeGmailClient):
        def apply_labels(self, message_id: str, add_label_ids: Any) -> None:
            order.append(f"apply:{message_id}")
            super().apply_labels(message_id, add_label_ids)

    fake = _OrderClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    orig_commit = db.commit_page

    def commit_then(rows: Any, cursor: int) -> int:
        order.append(f"commit:{cursor}")
        return orig_commit(rows, cursor)

    db.commit_page = commit_then  # type: ignore[method-assign]
    assert _run(tmp_path, fake, db=db) == 0
    assert order == ["apply:aa22", "commit:900"]
    assert fake.writes == [
        {
            "method": "apply_labels",
            "message_id": "aa22",
            "body": {"addLabelIds": ["L2", "LA", "LH"]},
        }
    ]
    assert db.get_history_cursor() == 900
    rows = db.applications_for_message("aa22")
    assert [(r.label, r.band, r.tier, r.source) for r in rows] == [
        ("Money/Bills", "high", "ai-high", "daemon")
    ]
    assert rows[0].applied_at == TS
    db.close()


def test_skip_present_message_no_write_no_rows_cursor_advances(
    tmp_path: Path,
) -> None:
    # the replay's engine (invariant #2): the label is present → empty
    # plan → NO API call at all, no row — and the cursor still advances
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa11"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert fake.writes == []
    assert db.applications_for_message("aa11") == ()
    assert db.get_history_cursor() == 900
    db.close()


def test_idle_delta_advances_cursor_with_empty_commit(tmp_path: Path) -> None:
    # the fake's exhausted-sequence shape IS the real idle delta: empty
    # records + the mailbox's CURRENT id → commit_page([], id) — the
    # cursor moves, nothing else happens
    fake = FakeGmailClient(
        messages=_messages(), labels=_labels(), current_history_id=900
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() == 900
    assert fake.writes == []
    db.close()


def test_multi_message_page_one_predict_one_commit_batch_homogeneity(
    tmp_path: Path,
) -> None:
    # the #24 batch-homogeneity contract AT THE DAEMON CALL SITE (a
    # contract, not an accident): the page's rows share the ONE clock
    # read taken at page-handling time; ONE predict batch over the
    # page's texts; ONE commit for the delta
    import gmail_labeler.daemon as daemon_module

    class _CountingModel(FakeModel):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            self.batch_count = 0

        def predict(self, texts: Any) -> Any:
            self.batch_count += 1
            return super().predict(texts)

    model = _CountingModel(VOCAB, scores=_scores())
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa11"), _record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)

    class _CountDb:
        def __init__(self, inner: AppDB) -> None:
            self._inner = inner
            self.commits = 0

        def commit_page(self, rows: Any, cursor: int) -> int:
            self.commits += 1
            return self._inner.commit_page(rows, cursor)

        def get_history_cursor(self) -> int | None:
            return self._inner.get_history_cursor()

        def set_history_cursor(self, cursor: int) -> None:
            self._inner.set_history_cursor(cursor)

        def close(self) -> None:
            pass  # the driver owns the real db

    counting = _CountDb(db)
    applied_ats: list[datetime] = []
    real_plan = daemon_module.plan_message_write

    def spy_plan(message_id: str, **kw: Any) -> Any:
        applied_ats.append(kw["applied_at"])
        return real_plan(message_id, **kw)

    # a TICKING clock: every read returns a different microsecond — the
    # homogeneity pin is only real if a second read per page CHANGES
    # the rows (a constant clock would bless the double-read mutant)
    clock_calls = 0

    def ticking_clock() -> datetime:
        nonlocal clock_calls
        clock_calls += 1
        return TS.replace(microsecond=clock_calls)

    daemon_module.plan_message_write = spy_plan  # type: ignore[assignment]
    try:
        assert _run(tmp_path, fake, db=counting, model=model, clock=ticking_clock) == 0
    finally:
        daemon_module.plan_message_write = real_plan  # type: ignore[assignment]
    # the counting pin (fold-5 A4 doctrine): the clock is read EXACTLY
    # once per handled delta and every row carries THAT one stamp
    assert clock_calls == 1
    assert len(applied_ats) == 2 and len(set(applied_ats)) == 1
    assert applied_ats[0] == TS.replace(microsecond=1)
    assert model.batch_count == 1 and len(model.calls) == 1
    assert counting.commits == 1
    assert db.get_history_cursor() == 900
    db.close()


# -- the error arms -------------------------------------------------------------


def test_transient_history_failure_holds_cursor_counted_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # per-page GmailError: cursor does NOT advance, counted id-only
    # warning, next tick retries (fail-once consumed → tick 2 commits)
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    fake.fail("history", GmailTransientError("blip"))
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db, stop_after_ticks=2) == 0
    assert "cursor 777" in caplog.text  # the held cursor is NAMED (id-only)
    assert db.get_history_cursor() == 900  # tick 2 retried and committed
    assert len(fake.writes) == 1
    db.close()


def test_expired_cursor_resyncs_via_bootstrap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # the 404 arm PER THE TAXONOMY (checked, not invented): the fake's
    # out-of-date-window GmailPermanentError(status=404) routes to the
    # re-bootstrap — via the export prior-art path, named warning,
    # cursor re-established, gap mail NOT processed. F1 moved this onto
    # the REAL expiry direction: the daemon down past retention leaves
    # the cursor BEHIND the floor (the ahead-of-mailbox arm stays
    # fake-level pinned; the injected-GmailResyncError adapter test
    # below keeps its own route)
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        current_history_id=900,
        retention_floor=700,
    )
    db = _db(tmp_path)
    db.set_history_cursor(400)  # BEHIND the retained window → out-of-date 404
    assert _run(tmp_path, fake, db=db) == 0
    assert "expired" in caplog.text
    assert "NOT processed" in caplog.text  # the gap is named, honestly
    assert db.get_history_cursor() == 777  # re-bootstrapped (aa11's id)
    assert fake.writes == []
    db.close()


def test_adapter_resync_error_routes_to_same_arm(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # GmailResyncError (the adapter's reading of the 404) is caught by
    # the same arm — it IS a GmailPermanentError with status=404 (the
    # taxonomy's documented subclassing)
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    fake.fail("history", GmailResyncError("startHistoryId not retained", status=404))
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert "expired" in caplog.text
    # the resync re-established the cursor from the newest message
    # (777 — the same value: aa11.history_id) and the SAME tick does
    # not also poll: the scripted page was NOT consumed
    assert db.get_history_cursor() == 777
    assert fake.calls.count("history:777:") == 1
    assert fake.writes == []
    db.close()


def test_resync_bootstrap_failure_keeps_old_cursor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 404 + an empty bootstrap mailbox: the DEAD cursor stays (the
    # retry's problem, named) — never a fake cursor written over the
    # durable record
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(messages=[], labels=_labels(), current_history_id=100)
    db = _db(tmp_path)
    db.set_history_cursor(500)
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() == 500
    assert "cannot establish history cursor" in caplog.text
    db.close()


def test_auth_error_propagates_for_exit_4(tmp_path: Path) -> None:
    # the human-gated arm: GmailAuthError is NOT a page-hold — it
    # propagates (the CLI maps it to the exit-code contract)
    fake = FakeGmailClient(
        messages=_messages(), labels=_labels(), current_history_id=900
    )
    fake.fail("history", GmailAuthError("token revoked"))
    db = _db(tmp_path)
    db.set_history_cursor(777)
    with pytest.raises(GmailAuthError):
        _run(tmp_path, fake, db=db)
    assert db.get_history_cursor() == 777  # cursor held, nothing lost
    db.close()


def test_get_message_permanent_404_skips_message_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # a message the mailbox no longer has (deleted mid-window): the
    # per-message fail-safe — skip + counted id-only warning, the page
    # still commits (the cursor must not wedge on a gone message)
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("dead00"), _record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert "skipped message dead00" in caplog.text
    assert db.get_history_cursor() == 900  # NOT held — the page was handled
    assert len(fake.writes) == 1  # aa22 still applied
    db.close()


def test_get_message_transient_holds_page_and_rewalks(
    tmp_path: Path,
) -> None:
    # a transient fetch failure: the page is NOT committed (cursor
    # held); the next tick RE-WALKS from the held cursor (the cursor IS
    # the resume point — ADR-0003's "the next poll replays the gap";
    # the fake re-serves its scripted window for the same start id)
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    fake.fail("get_message", GmailTransientError("blip"))
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db, stop_after_ticks=2) == 0
    # exact-call pin: page fetched, held, RE-WALKED (not skipped, not
    # double-advanced), committed on tick 2
    assert fake.calls == [
        "history:777:",
        "get_message:aa22",
        "history:777:",
        "get_message:aa22",
        "apply_labels:aa22",
    ]
    assert db.get_history_cursor() == 900
    assert len(fake.writes) == 1
    db.close()


def test_label_only_records_pass_through_and_delta_commits(tmp_path: Path) -> None:
    # F1 round 2: a history record carrying ONLY label changes (no
    # messages key) is the COMMON real wire shape — it passes through
    # the event extraction and the delta commits normally
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[
            _page({"id": 7, "labels": {"labels": [{"id": "L1", "added": ["INBOX"]}]}})
        ],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() == 900  # handled: nothing to fetch, cursor moves
    assert fake.writes == []
    db.close()


def test_malformed_messages_not_a_list_holds_cursor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # F1 round 2: the sibling malformed shape — messages present but
    # not a list of objects — is the same lying-wire hold arm
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page({"id": 1, "messages": "not-a-list"})],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() == 777
    assert "delta failed" in caplog.text
    db.close()


def test_bootstrap_auth_error_propagates(tmp_path: Path) -> None:
    # F1 round 2 (the same dead-line class): auth during bootstrap must
    # reach the CLI's exit 4, never hold-retry forever on revoked creds
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    fake.fail("list_messages", GmailAuthError("token revoked"))
    db = _db(tmp_path)  # no cursor: bootstrap tick
    with pytest.raises(GmailAuthError):
        _run(tmp_path, fake, db=db)
    assert db.get_history_cursor() is None
    db.close()


def test_bootstrap_transient_holds_and_retries_next_tick(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # F1 round 2: a transient bootstrap read is the NAMED hold (retry
    # next tick), not a crash and not a fake cursor
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    fake.fail("list_messages", GmailTransientError("503 blip"))
    db = _db(tmp_path)
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() is None  # never a fake value
    assert "cannot establish history cursor" in caplog.text
    # the name's promise, kept (review round-2 NIT): the NEXT tick's
    # retried bootstrap succeeds (fail-once queue) and lands the cursor
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() == 777  # aa11's id (the bootstrap pin)
    db.close()


def test_shutdown_mid_delta_still_drains_to_commit_point(tmp_path: Path) -> None:
    # F1 round 2: the BETWEEN-PAGES stop check — consulted after a
    # page with a continuation token — records the stop and the delta
    # still drains to its commit point (no half-committed state)
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[
            _page(_record("aa11"), token="p1"),
            _page(_record("aa22"), token=None),
        ],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    seen = {"n": 0}

    def should_stop() -> bool:
        # 1st call: loop-top (False, tick runs); 2nd: after page 1
        # (True — recorded, drain continues); later: stay True
        seen["n"] += 1
        return seen["n"] > 2

    driver = _LoopRun()
    rc = daemon_loop(
        client=fake,
        db=db,
        model=FakeModel(VOCAB, scores=_scores()),
        ctx=_ctx(),
        poll_seconds=30,
        clock=lambda: TS,
        sleep=driver.sleep,
        should_stop=should_stop,
    )
    assert rc == 0
    assert driver.sleeps == []  # stopped before any sleep: straight out
    assert db.get_history_cursor() == 900  # drained to the commit point
    assert [w["message_id"] for w in fake.writes] == ["aa22"]
    db.close()


def test_apply_transient_holds_cursor_and_retries_next_tick(tmp_path: Path) -> None:
    # F1 MAJOR (the manager's mutant T2): a transient failure on the
    # WRITE must not let the cursor advance past the failed message.
    # Unpinned, the FULL suite stayed green while the daemon starved the
    # mailbox: cursor 900, message never labeled, never retried,
    # silently. Phase 1: the delta HOLDS (cursor stays, no writes, no
    # rows). Phase 2: the next tick re-walks from the held cursor, the
    # write lands, the rows commit.
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    fake.fail("apply_labels", GmailTransientError("503 blip"))
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() == 777  # THE T2 PIN: no advance past a failed write
    assert fake.writes == []
    assert db.applications_for_message("aa22") == ()
    assert _run(tmp_path, fake, db=db) == 0  # next tick: re-walk, retry
    assert db.get_history_cursor() == 900
    assert [w["message_id"] for w in fake.writes] == ["aa22"]
    # one Application row (the umbrella/tier labels are Gmail-side state,
    # not rows — the CLI full-stack pin shows the row shape)
    assert [r.label for r in db.applications_for_message("aa22")] == ["Money/Bills"]
    db.close()


def test_apply_404_skips_message_delta_still_commits(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # F1 MAJOR arm (b): the message vanished between fetch and write
    # (per-message 404 ON THE WRITE). The fail-safe skips it COUNTED,
    # the other message's write is unaffected, the delta still commits.
    # Counter pin (F1 nit): the ghost counts SKIPPED only — the
    # end-of-delta line reads written=1 skipped=1 warnings=0.
    caplog.set_level(logging.INFO)

    def _both_write() -> dict[str, tuple[float, ...]]:
        # aa11: Money/Bills high (absent — write); aa22: Shipping high
        # (absent — write). The delta order makes aa11 the FIRST apply.
        return {"one\nbody one": (0.9, 0.1), "two\nbody two": (0.1, 0.95)}

    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa11"), _record("aa22"))],
        current_history_id=900,
    )
    fake.fail("apply_labels", GmailPermanentError("message aa11 not found", status=404))
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert (
        _run(tmp_path, fake, db=db, model=FakeModel(VOCAB, scores=_both_write())) == 0
    )
    assert [w["message_id"] for w in fake.writes] == ["aa22"]  # aa11 skipped
    assert db.get_history_cursor() == 900  # the delta COMMITS (a ghost must not wedge)
    assert (
        db.applications_for_message("aa11") == ()
    )  # no row for a write that never landed
    assert [r.label for r in db.applications_for_message("aa22")] == ["Shipping"]
    assert "skipped message aa11: apply refused" in caplog.text
    assert "written=1 skipped=1 warnings=0" in caplog.text
    db.close()


def test_apply_auth_error_propagates_for_exit_4(tmp_path: Path) -> None:
    # F1 MAJOR arm (c): auth on the WRITE is human-gated — it re-raises
    # out of the loop (the CLI maps 4; pinned at the CLI seam in
    # test_daemon_cli), NEVER held-forever like a transient.
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    fake.fail("apply_labels", GmailAuthError("token revoked"))
    db = _db(tmp_path)
    db.set_history_cursor(777)
    with pytest.raises(GmailAuthError):
        _run(tmp_path, fake, db=db)
    assert db.get_history_cursor() == 777  # nothing committed over an auth death
    assert fake.writes == []
    db.close()


def test_get_message_auth_error_propagates(tmp_path: Path) -> None:
    # F1 MINOR 1 (the dead line): the fetch arm's `except GmailAuthError:
    # raise` had NO node — a refactor swallowing auth there would hang
    # the daemon in hold-retry forever instead of the exit-4 landing
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    fake.fail("get_message", GmailAuthError("token revoked"))
    db = _db(tmp_path)
    db.set_history_cursor(777)
    with pytest.raises(GmailAuthError):
        _run(tmp_path, fake, db=db)
    assert db.get_history_cursor() == 777
    db.close()


def test_malformed_messages_list_holds_cursor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # F1 MINOR 1: a history record whose messages event carries no id is
    # a lying wire — the typed permanent from _message_ids routes to
    # the HOLD arm (never a skip: the whole delta is untrustworthy)
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page({"id": 1, "messages": [{"threadId": "tx"}]})],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() == 777
    assert "delta failed" in caplog.text
    db.close()


def test_undecodable_raw_skips_message_and_delta_commits(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # F1 MINOR 1: corrupt raw is the per-message fail-safe (skipped,
    # counted, id-only warning) — the delta still commits (one ghost
    # body must not wedge the cursor)
    caplog.set_level(logging.WARNING)

    # the decode surface is BASE64URL (decode_gmail_raw does not utf-8
    # decode — parse_mime never raises on bytes), so the corrupt field
    # is an unpaddable base64url string, injected at the wire shape
    class _BadRawClient(FakeGmailClient):
        def get_message(self, message_id: str):
            raw = super().get_message(message_id)
            if message_id == "aa22":
                return dataclasses.replace(raw, raw_b64url="A")  # ValueError
            return raw

    fake = _BadRawClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa11"), _record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() == 900
    assert fake.writes == []  # aa11 skip-present, aa22 skipped: nothing to write
    assert "skipped message aa22: raw undecodable" in caplog.text
    db.close()


def test_bootstrap_anchors_on_the_documented_newest_first_order() -> None:
    # F1 MINOR 6 (resolved by PRIMARY source, not by guessing): the List
    # Gmail messages guide states the response is "listed in reverse
    # chronological order (newest first)" and the sync guide names the
    # first entry the full-sync seed source. The fake MODELS that order
    # (documented_order=True sorts by internal_date_ms desc); the
    # insertion order here is deliberately OLDEST-first, so a bootstrap
    # taking the wrong stub changes the seed value (the wrong-stub
    # scenario is pinned by construction, not assumed away)
    msgs = [
        FakeMessage(
            id="old", eml=_eml("o", "ob"), internal_date_ms=1_000, history_id=500
        ),
        FakeMessage(
            id="new", eml=_eml("n", "nb"), internal_date_ms=2_000, history_id=900
        ),
    ]
    fake = FakeGmailClient(messages=msgs, labels=_labels(), documented_order=True)
    assert bootstrap_cursor(fake) == 900
    assert fake.calls == ["list_messages::", "get_message:new"]
    # the modeling is honest: a message without the recency field is
    # refused, never silently sorted to the top
    with pytest.raises(ValueError, match="internal_date_ms"):
        FakeGmailClient(messages=[FakeMessage(id="x", eml=b"x")], documented_order=True)


def test_predict_failure_holds_cursor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # a model defect (unprogrammed text) is systematic, not per-message:
    # the page is held (cursor does NOT advance), the warning is counted
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db, model=FakeModel(VOCAB, scores={})) == 0
    assert db.get_history_cursor() == 777
    assert fake.writes == []
    assert "delta failed" in caplog.text
    # the review round's MAJOR pin: the leak fix itself. FakeModel's
    # unprogrammed-text refusal QUOTES THE SCORED TEXT (which carries
    # the body) — the warning must name the exception CLASS only,
    # never the message. (Mutant: reverting to log(..., e) reddens the
    # body-absence assertion below.)
    assert "prediction failed (ValueError)" in caplog.text
    assert "body two" not in caplog.text
    db.close()


def test_prediction_shape_defect_holds_and_names(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # the self-authored shape defects (row count / row width) are
    # body-free by construction and log their MESSAGE (operator-
    # diagnosable), while still holding the cursor
    caplog.set_level(logging.WARNING)

    class _ShortModel(FakeModel):
        def predict(self, texts: Any) -> Any:
            self.calls.append(tuple(texts))
            return ((0.9, 0.1),)  # ONE row for a two-message delta

    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa11"), _record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db, model=_ShortModel(VOCAB, scores={})) == 0
    assert db.get_history_cursor() == 777  # held
    assert "score rows for 2 texts" in caplog.text
    db.close()


def test_prediction_width_defect_holds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # a lying checkpoint with the right row COUNT but a misaligned row
    # WIDTH must not escape the guarded arm and kill the loop (the
    # daemon is long-lived: hold and warn, unlike shadow's one-shot
    # exit-1 posture for width defects)
    caplog.set_level(logging.WARNING)

    class _WideModel(FakeModel):
        def predict(self, texts: Any) -> Any:
            self.calls.append(tuple(texts))
            return tuple((0.9, 0.1, 0.5) for _ in tuple(texts))

    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db, model=_WideModel(VOCAB, scores={})) == 0
    assert db.get_history_cursor() == 777  # held, loop alive
    assert "does not align" in caplog.text
    db.close()


def test_history_response_without_history_id_holds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # the wire owes the mailbox's CURRENT id on every response; a
    # response without it cannot establish the advance — hold, never
    # invent a cursor (the arm had no node before the review round)
    caplog.set_level(logging.WARNING)

    class _NoIdClient(FakeGmailClient):
        def history(
            self, start_history_id: int, page_token: str | None = None
        ) -> HistoryPage:
            return HistoryPage(history_id=None, next_page_token=None, records=())

    fake = _NoIdClient(messages=_messages(), labels=_labels(), current_history_id=900)
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert db.get_history_cursor() == 777  # held
    assert "no historyId" in caplog.text
    db.close()


def test_fail_safe_warning_shape_and_end_of_page_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # the #24 warning shape surfaces EXACT (plan.warning verbatim) and
    # the end-of-page report line carries the count (bodies never). The
    # lying row bypasses FakeModel's programming check (a real lying
    # model row reaches the plan from a defective checkpoint — the
    # test_applications precedent)
    caplog.set_level(logging.INFO)

    class _LyingModel(FakeModel):
        def predict(self, texts: Any) -> Any:
            self.calls.append(tuple(texts))
            return tuple(
                (1.5, 0.1) if "two" in t else (0.2, 0.95) for t in tuple(texts)
            )

    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db, model=_LyingModel(VOCAB, scores={})) == 0
    assert (
        "aa22: ValueError: tiers for 'Money/Bills': score 1.5 is out of [0, 1] "
        "— a lying model row" in caplog.text
    )
    assert "warnings=1" in caplog.text  # the end-of-page report count
    assert fake.writes == []  # no write for the failed message
    assert db.get_history_cursor() == 900  # the page WAS handled (no write)
    db.close()


def test_poisoned_tier_id_never_reaches_the_wire(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # the plan's System-Label id screen at the daemon layer: a tier
    # resolved to INBOX composes NO write and lands as a counted warning
    caplog.set_level(logging.INFO)
    ctx = _ctx(tier_label_ids={"ai-applied": "LA", "ai-high": "INBOX", "ai-med": "LM"})
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db, ctx=ctx) == 0
    assert fake.writes == []
    assert "warnings=1" in caplog.text
    db.close()


# -- paging ---------------------------------------------------------------------


def test_multi_page_delta_commits_once_with_final_cursor(tmp_path: Path) -> None:
    # every page's historyId is the mailbox's CURRENT id (the documented
    # trap): the cursor advances ONCE, at the delta's end, from the
    # FINAL response — rows from BOTH pages ride the one commit
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[
            _page(_record("aa11"), token="p1"),
            _page(_record("aa22")),
        ],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db) == 0
    assert fake.calls == [
        "history:777:",
        "get_message:aa11",
        "history:777:p1",
        "get_message:aa22",
        "apply_labels:aa22",
    ]
    assert db.get_history_cursor() == 900
    assert len(db.applications_for_message("aa22")) == 1
    db.close()


def test_mid_delta_failure_rewalks_from_the_held_cursor(tmp_path: Path) -> None:
    # failure while handling page 1 of a multi-page delta: the cursor
    # holds AND the next tick RE-WALKS from the cursor (the consumed
    # pages' events are re-served by the retained window — the fake
    # models this by resetting its script for the same start id); page
    # 2's token chain then continues normally to the final page
    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[
            _page(_record("aa11"), token="p1"),
            _page(_record("aa22"), token="p2"),
            _page(),
        ],
        current_history_id=900,
    )
    fake.fail("get_message", GmailTransientError("blip"))  # page 1's fetch fails
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db, stop_after_ticks=2) == 0
    assert fake.calls == [
        "history:777:",
        "get_message:aa11",  # transient failure → hold
        "history:777:",  # re-walk: page 1 re-served
        "get_message:aa11",
        "history:777:p1",
        "get_message:aa22",
        "history:777:p2",
        "apply_labels:aa22",
    ]
    assert db.get_history_cursor() == 900
    db.close()


# -- graceful shutdown ------------------------------------------------------------


def test_shutdown_mid_page_completes_to_commit_point(tmp_path: Path) -> None:
    # AC pin: SIGTERM between messages → the current page RUNS TO ITS
    # COMMIT POINT (cursor written), clean exit 0, no further tick
    calls = 0

    def flag_after_two() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2  # the flag lands mid-page

    fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[_page(_record("aa11"), _record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    driver = _LoopRun()
    assert (
        daemon_loop(
            client=fake,
            db=db,
            model=FakeModel(VOCAB, scores=_scores()),
            ctx=_ctx(),
            poll_seconds=30,
            clock=lambda: TS,
            sleep=driver.sleep,
            should_stop=flag_after_two,
        )
        == 0
    )
    # the page completed: aa22 written, rows + cursor committed
    assert [w["message_id"] for w in fake.writes] == ["aa22"]
    assert db.get_history_cursor() == 900
    assert len(db.applications_for_message("aa22")) == 1
    # and the loop exited WITHOUT a second tick or sleep
    assert driver.sleeps == []
    assert fake.calls.count("history:777:") == 1
    db.close()


def test_shutdown_mid_sleep_exits_immediately_cursor_untouched(
    tmp_path: Path,
) -> None:
    # AC pin: SIGTERM mid-sleep → immediate exit 0, cursor untouched.
    # The injected sleep stands in for the interruptible real one: it
    # raises the flag as the signal handler would, then returns.
    fake = FakeGmailClient(
        messages=_messages(), labels=_labels(), current_history_id=900
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)

    class _SleepThatGetsSignaled:
        def __init__(self) -> None:
            self.stopped = False
            self.slept: list[float] = []

        def sleep(self, seconds: float) -> None:
            self.slept.append(seconds)
            self.stopped = True  # the signal handler fires mid-sleep

        def should_stop(self) -> bool:
            return self.stopped

    sig = _SleepThatGetsSignaled()
    assert (
        daemon_loop(
            client=fake,
            db=db,
            model=FakeModel(VOCAB, scores=_scores()),
            ctx=_ctx(),
            poll_seconds=42,
            clock=lambda: TS,
            sleep=sig.sleep,
            should_stop=sig.should_stop,
        )
        == 0
    )
    assert sig.slept == [42]  # config cadence honored EXACTLY
    assert db.get_history_cursor() == 900  # tick 1's commit stands
    assert fake.calls.count("history:900:") == 0  # NO second tick
    db.close()


def test_poll_seconds_passed_to_sleep(tmp_path: Path) -> None:
    fake = FakeGmailClient(
        messages=_messages(), labels=_labels(), current_history_id=900
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    driver = _LoopRun()
    driver.stop_after = 2
    assert (
        daemon_loop(
            client=fake,
            db=db,
            model=FakeModel(VOCAB, scores=_scores()),
            ctx=_ctx(),
            poll_seconds=15,
            clock=lambda: TS,
            sleep=driver.sleep,
            should_stop=driver.should_stop,
        )
        == 0
    )
    assert driver.sleeps == [15, 15]
    db.close()


# -- bodies never (invariant #5, daemon layer) ------------------------------------


def test_bodies_absent_from_logs_and_db(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    marker = "SECRET-BODY-MARKER"
    msgs = [
        FakeMessage(
            id="aa11",
            eml=_eml("one", f"{marker} body one"),
            label_ids=("INBOX", "L1"),
            history_id=777,
        ),
        FakeMessage(
            id="aa22",
            eml=_eml("two", f"{marker} body two"),
            label_ids=("INBOX",),
            history_id=777,
        ),
    ]
    scores = {
        f"one\n{marker} body one": (0.2, 0.95),
        f"two\n{marker} body two": (0.9, 0.1),
    }
    fake = FakeGmailClient(
        messages=msgs,
        labels=_labels(),
        history_pages=[_page(_record("aa11"), _record("aa22"))],
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    assert _run(tmp_path, fake, db=db, model=FakeModel(VOCAB, scores=scores)) == 0
    assert marker not in caplog.text
    conn = sqlite3.connect(str(db.path))
    try:
        for table in ("applications", "daemon_state"):
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            assert marker not in str(rows)
    finally:
        conn.close()
    db.close()


# -- the crash matrix (the AC's kill -9 list) --------------------------------------


class _Boom(Exception):
    pass


def _delta_pages() -> list[HistoryPage]:
    return [_page(_record("aa11"), _record("aa22"))]


def test_crash_mid_page_after_write_replay_converges(tmp_path: Path) -> None:
    # crash row (a) mid-page AFTER-WRITE: die after message 2's Gmail
    # write, before the DB commit. Replay (fresh loop, mailbox state
    # kept — the fake's mutation IS the server state): m2's label is
    # present → skip-present → EMPTY plan → no second write, no row;
    # zero double-writes, cursor advanced exactly once.
    msgs = _messages()
    pages = _delta_pages()

    class _KillAfterFirstWrite:
        def __init__(self, inner: FakeGmailClient) -> None:
            self._inner = inner

        def apply_labels(self, message_id: str, add_label_ids: Any) -> None:
            self._inner.apply_labels(message_id, add_label_ids)
            if len(self._inner.writes) == 1:
                raise _Boom("kill -9 between the Gmail write and the commit")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    fake1 = FakeGmailClient(
        messages=msgs,
        labels=_labels(),
        history_pages=list(pages),
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)
    with pytest.raises(_Boom):
        _run(tmp_path, _KillAfterFirstWrite(fake1), db=db)  # type: ignore[arg-type]
    assert len(fake1.writes) == 1  # aa22 written, nothing committed
    assert db.get_history_cursor() == 777  # cursor HELD (invariant #3)
    assert db.applications_for_message("aa22") == ()  # no rows survived

    # replay: fresh loop, SAME mailbox objects (Gmail's durable truth),
    # the retained window re-serving the same events
    fake2 = FakeGmailClient(
        messages=msgs,
        labels=_labels(),
        history_pages=list(pages),
        current_history_id=900,
    )
    assert _run(tmp_path, fake2, db=db) == 0
    # aa22's labels are PRESENT now (the pre-crash write) → skip-present
    # → NO second write; aa11 stays skip-present
    assert fake2.writes == []
    assert db.get_history_cursor() == 900
    # the audit gap is the design's documented face of the convergence:
    # the pre-crash write's row was never committed (write-first +
    # commit-at-delta-end); Gmail truth + zero double-writes is the
    # safety claim, and it holds
    assert len(db.applications_for_message("aa22")) == 0
    db.close()


def test_crash_mid_transaction_holds_and_converges(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # crash row (b) mid-transaction: the commit dies. The store-side
    # all-or-nothing (rows AND cursor rolling back together) is pinned
    # at the store layer in test_appdb; at the DAEMON layer the contract
    # is the HOLD + convergence: the daemon catches the commit failure
    # (fail-safe arm — a crash has the same durable outcome, rows
    # (a)/(c)), the cursor stays, and the replay recomposes — the
    # already-written message skip-presents (no second write, no row)
    caplog.set_level(logging.WARNING)
    msgs = _messages()
    pages = _delta_pages()
    fake1 = FakeGmailClient(
        messages=msgs,
        labels=_labels(),
        history_pages=list(pages),
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)

    class _CommitDies:
        def __init__(self, inner: AppDB) -> None:
            self._inner = inner

        def commit_page(self, rows: Any, cursor: int) -> int:
            raise AppDBError("simulated commit failure (mid-transaction death)")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    assert _run(tmp_path, fake1, db=_CommitDies(db)) == 0  # type: ignore[arg-type]
    assert "state DB" in caplog.text  # the failure is surfaced, named
    assert db.get_history_cursor() == 777  # held
    assert db.applications_for_message("aa22") == ()  # nothing committed

    fake2 = FakeGmailClient(
        messages=msgs,
        labels=_labels(),
        history_pages=list(pages),
        current_history_id=900,
    )
    assert _run(tmp_path, fake2, db=db) == 0
    assert fake2.writes == []  # skip-present did its job
    assert db.get_history_cursor() == 900
    db.close()


def test_crash_after_commit_replay_converges(tmp_path: Path) -> None:
    # crash row (c) after-commit: die one heartbeat AFTER the durable
    # commit — the restart polls from the NEW cursor: zero second
    # writes, zero duplicate rows, one commit
    msgs = _messages()
    pages = _delta_pages()
    fake1 = FakeGmailClient(
        messages=msgs,
        labels=_labels(),
        history_pages=list(pages),
        current_history_id=900,
    )
    db = _db(tmp_path)
    db.set_history_cursor(777)

    class _CommitThenDie:
        def __init__(self, inner: AppDB) -> None:
            self._inner = inner

        def commit_page(self, rows: Any, cursor: int) -> int:
            self._inner.commit_page(rows, cursor)
            raise _Boom("kill -9 one heartbeat after COMMIT")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    with pytest.raises(_Boom):
        _run(tmp_path, fake1, db=_CommitThenDie(db))  # type: ignore[arg-type]
    assert db.get_history_cursor() == 900  # the commit STANDS
    assert len(db.applications_for_message("aa22")) == 1

    fake2 = FakeGmailClient(
        messages=msgs,
        labels=_labels(),
        history_pages=list(pages),
        current_history_id=900,
    )
    assert _run(tmp_path, fake2, db=db) == 0
    # the events are behind the cursor AND skip-present covers the
    # overlap — no second write either way
    assert fake2.writes == []
    assert len(db.applications_for_message("aa22")) == 1  # no duplicate
    assert db.get_history_cursor() == 900
    db.close()


_CHILD_SCRIPT = textwrap.dedent(
    """
    import json, os, sys
    from datetime import UTC, datetime

    # the import-path assertion (the editable-.pth false-green trap is
    # subprocess-capable too): the child must run THIS src tree
    import gmail_labeler
    assert gmail_labeler.__file__ == os.path.join(sys.argv[2], "gmail_labeler",
                                                  "__init__.py"), gmail_labeler.__file__

    from gmail_labeler.appdb import AppDB
    from gmail_labeler.daemon import ServingContext, daemon_loop
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
    from gmail_labeler.gmail_client import HistoryPage, LabelInfo
    from gmail_labeler.model import FakeModel
    from pathlib import Path

    cfg = json.loads(sys.argv[1])
    msgs = []
    for m in cfg["messages"]:
        eml = ("Subject: %s\\r\\nFrom: a@x.com\\r\\nTo: me@x.com\\r\\n"
               "Content-Type: text/plain\\r\\n\\r\\n%s"
               % (m["subject"], m["body"])).encode()
        msgs.append(FakeMessage(id=m["id"], eml=eml, label_ids=tuple(m["label_ids"]),
                                history_id=777))
    scores = {"one\\nbody one": (0.2, 0.95), "two\\nbody two": (0.9, 0.1)}
    records = ({"id": 1, "messages": [
        {"id": i, "threadId": "t" + i} for i in ("aa11", "aa22")]},)
    pages = [HistoryPage(history_id=None, next_page_token=None,
                         records=records)]
    fake = FakeGmailClient(messages=msgs, labels=[
        LabelInfo("INBOX", "INBOX", "system"),
        LabelInfo("L1", "Shipping", "user"),
        LabelInfo("L2", "Money/Bills", "user"),
        LabelInfo("LA", "ai-applied", "user"),
        LabelInfo("LH", "ai-high", "user"),
        LabelInfo("LM", "ai-med", "user"),
    ], history_pages=pages, current_history_id=900)

    class _RealKill:
        def __init__(self, inner):
            self._inner = inner

        def apply_labels(self, mid, adds):
            self._inner.apply_labels(mid, adds)
            if len(self._inner.writes) == 1:
                # the mailbox truth at death, dumped BEFORE the kill
                # (the fake is in-memory by nature; a real Gmail's
                # server state survives the kill — the dump stands in)
                with open(cfg["dump"], "w") as fh:
                    json.dump({m.id: list(m.label_ids)
                               for m in self._inner.messages.values()}, fh)
                os._exit(137)  # the kill -9: no finally, no cleanup, no rollback

        def __getattr__(self, name):
            return getattr(self._inner, name)

    db = AppDB(Path(cfg["db"]))
    db.open()
    ctx = ServingContext(
        bands={"Money/Bills": (0.8, 0.6), "Shipping": (0.8, 0.6)},
        graduated=frozenset({"Money/Bills", "Shipping"}),
        system_labels=frozenset({"INBOX"}),
        id_to_name={"L1": "Shipping", "L2": "Money/Bills"},
        managed_label_ids={"Money/Bills": "L2", "Shipping": "L1"},
        tier_label_ids={"ai-applied": "LA", "ai-high": "LH", "ai-med": "LM"},
    )
    slept = []
    ticks = {"n": 0}
    real_history = fake.history

    def counting_history(start, token=None):
        ticks["n"] += 1
        return real_history(start, token)

    fake.history = counting_history
    client = _RealKill(fake) if cfg["die"] else fake

    def should_stop():
        return ticks["n"] >= cfg["ticks"] and len(slept) >= 1

    def sleep(s):
        slept.append(s)

    try:
        daemon_loop(client=client, db=db,
                    model=FakeModel(("Money/Bills", "Shipping"), scores=scores),
                    ctx=ctx, poll_seconds=30,
                    clock=lambda: datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC),
                    sleep=sleep, should_stop=should_stop)
    finally:
        conn = db._conn
        rows = [list(r) for r in conn.execute(
            "SELECT message_id, label FROM applications ORDER BY message_id, label")]
        print("RESULT " + json.dumps({
            "cursor": db.get_history_cursor(),
            "rows": rows,
            "writes": [w["message_id"] for w in fake.writes],
            "mailbox": {m.id: list(m.label_ids) for m in fake.messages.values()},
        }))
    """
)


def test_real_os_exit_kill_after_write_converges(tmp_path: Path) -> None:
    # THE after-write row with a REAL kill: a child process runs the
    # loop and os._exit()s (no finally, no cleanup, no rollback — the
    # process is simply gone) right after the first Gmail write landed.
    # The parent restarts a SECOND process on the same DB file:
    # skip-present converges — zero second writes, zero duplicate rows,
    # cursor exactly once.
    db_path = tmp_path / "state.db"
    mbox_path = tmp_path / "mailbox.json"
    seed_messages = [
        {
            "id": "aa11",
            "subject": "one",
            "body": "body one",
            "label_ids": ["INBOX", "L1"],
        },
        {"id": "aa22", "subject": "two", "body": "body two", "label_ids": ["INBOX"]},
    ]
    env = dict(os.environ, PYTHONPATH=SRC)

    def child(payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT, json.dumps(payload), SRC],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )

    # the DB file (fresh v3 schema) with the cursor at 777 — the state
    # the pre-crash daemon had durably committed BEFORE this delta
    setup = _db(tmp_path)
    setup.set_history_cursor(777)
    setup.close()

    # child 1: DIES right after the first Gmail write (before any commit)
    r1 = child(
        {
            "messages": seed_messages,
            "dump": str(mbox_path),
            "die": True,
            "db": str(db_path),
            "ticks": 1,
        }
    )
    # the REAL kill: os._exit(137) is an exit STATUS chosen to resemble
    # SIGKILL (no signal is delivered) — what matters is that the
    # process died with no finally, no cleanup, and no rollback
    assert r1.returncode == 137, r1.stderr
    mailbox_at_death = json.loads(mbox_path.read_text(encoding="utf-8"))
    assert "L2" in mailbox_at_death["aa22"]  # the write LANDED server-side
    conn = sqlite3.connect(str(db_path))
    try:
        # the DB is EXACTLY the pre-delta state: the kill wrote nothing
        assert conn.execute("SELECT history_cursor FROM daemon_state").fetchone() == (
            777,
        )
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone() == (0,)
    finally:
        conn.close()

    # child 2 (the restart): fresh process, same DB file, the mailbox
    # AS CHILD 1 LEFT IT, the retained window re-serving the same
    # events — converge
    restarted = [{**m, "label_ids": mailbox_at_death[m["id"]]} for m in seed_messages]
    r2 = child(
        {
            "messages": restarted,
            "dump": str(mbox_path),
            "die": False,
            "db": str(db_path),
            "ticks": 1,
        }
    )
    assert r2.returncode == 0, r2.stderr
    line = [ln for ln in r2.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    result = json.loads(line[len("RESULT ") :])
    assert result["cursor"] == 900  # advanced exactly once, past the delta
    assert result["writes"] == []  # ZERO second writes (skip-present)
    assert result["rows"] == []  # ZERO application rows (the audit gap
    # of the write-first design — documented, never a double-write)
    assert result["mailbox"] == mailbox_at_death  # replay changed NOTHING


# -- import-direction pins ---------------------------------------------------------


def test_daemon_serves_through_the_injected_surface() -> None:
    # the loop talks to the injected seams only — no transport, no real
    # sleep, no direct clock read (the Model/GmailClient Protocols are
    # structural, pyright's job; this pins the import direction)
    import gmail_labeler.daemon as daemon_module

    source = Path(daemon_module.__file__).read_text(encoding="utf-8")
    assert "googleapiclient" not in source
    assert "time.sleep" not in source  # the sleep is INJECTED, never real
    assert "datetime.now" not in source  # the clock is INJECTED, never read
