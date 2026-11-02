"""apply core (ticket #26, C2): the batch loop tested DIRECTLY against
the fake (the test_daemon.py pattern — invariant #9), NOT through the
CLI (the CLI wiring + exit contract + provenance warning are C3). The
headline pins:

- AC1 preview: run_apply(dry_run=True) performs ZERO wire writes (no
  apply_labels, no writes recorded) and reports the would-apply lines;
- AC2 one-core parity: the structural identity (apply's composition entry
  IS daemon's `compose_plan`, whose globals reach the one
  `plan_message_write` core) + the semantic parity (one daemon tick and
  one apply batch over identical mail land IDENTICAL plans through the
  SAME core) — mutation-pinned (give apply a local copy of the
  composition and the identity check reddens);
- AC3: backfilled rows are recorded identically to daemon rows (source
  differs, the literal that entered the write path);
- AC4: mid-page SIGTERM completes the batch to its commit point, and a
  re-run is idempotent (skip-present, the fake really mutates label_ids —
  the #25 non-vacuity standard);
- AC6: a lying resolution map pointing a would-apply label at a System
  Label id composes NO write on THIS path (named-mutation pin);
- the landmines: 404-vanished mail is a counted skip (never fatal); a
  transient write failure HOLDS nothing (no cursor) and a re-run picks it
  up; >1-page paging fidelity; the empty plan is no write AND no row;
  daemon_state is byte-for-byte untouched across a whole backfill.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gmail_labeler.appdb import AppDB
from gmail_labeler.applications import plan_message_write as applications_plan
from gmail_labeler.apply import run_apply
from gmail_labeler.daemon import ServingContext, daemon_loop
from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
from gmail_labeler.gmail_client import HistoryPage, LabelInfo
from gmail_labeler.gmail_errors import (
    GmailPermanentError,
    GmailRateLimitedError,
    GmailTransientError,
)
from gmail_labeler.model import FakeModel

# the read surface the preview zero-write pin allows (the shadow idiom,
# minus history — apply never polls history; the core makes no label
# calls at all, the startup does)
READ_SURFACE = frozenset(
    {"list_messages", "get_message", "list_labels", "list_filters"}
)

TS = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)
VOCAB = ("Money/Bills", "Shipping")


def _eml(subject: str, body: str) -> bytes:
    return (
        f"Subject: {subject}\r\nFrom: a@x.com\r\nTo: me@x.com\r\n"
        "Content-Type: text/plain\r\n\r\n" + body
    ).encode()


def _messages() -> list[FakeMessage]:
    """m1 (aa11): Shipping present (skip-present candidate); m2 (aa22):
    nothing present (the write candidate)."""
    return [
        FakeMessage(
            id="aa11",
            eml=_eml("one", "body one"),
            label_ids=("INBOX", "L1"),
            thread_id="t1",
        ),
        FakeMessage(
            id="aa22",
            eml=_eml("two", "body two"),
            label_ids=("INBOX",),
            thread_id="t2",
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
    return {
        "one\nbody one": (0.2, 0.95),  # Shipping high, present → skip
        "two\nbody two": (0.9, 0.1),  # Money/Bills high, absent → write
    }


def _ctx(**over: Any) -> ServingContext:
    # typed locals (no `# pyright: ignore`): the **over unpack into a
    # dict[str, Any] then ServingContext(**base) forced an argument-type
    # suppression; naming each field's type lets pyright check the call
    bands: Mapping[str, tuple[float | None, float | None]] = over.get(
        "bands", {"Money/Bills": (0.8, 0.6), "Shipping": (0.8, 0.6)}
    )
    graduated: frozenset[str] = over.get(
        "graduated", frozenset({"Money/Bills", "Shipping"})
    )
    system_labels: frozenset[str] = over.get("system_labels", frozenset({"INBOX"}))
    id_to_name: Mapping[str, str] = over.get(
        "id_to_name", {"L1": "Shipping", "L2": "Money/Bills"}
    )
    managed_label_ids: Mapping[str, str] = over.get(
        "managed_label_ids", {"Money/Bills": "L2", "Shipping": "L1"}
    )
    tier_label_ids: Mapping[str, str] = over.get(
        "tier_label_ids", {"ai-applied": "LA", "ai-high": "LH", "ai-med": "LM"}
    )
    return ServingContext(
        bands=bands,
        graduated=graduated,
        system_labels=system_labels,
        id_to_name=id_to_name,
        managed_label_ids=managed_label_ids,
        tier_label_ids=tier_label_ids,
    )


def _db(tmp_path: Path) -> AppDB:
    db = AppDB(tmp_path / "state.db")
    db.open()
    return db


def _never_stop() -> bool:
    return False


def _run_apply(
    tmp_path: Path,
    client: Any,
    *,
    db: Any | None = None,
    model: Any | None = None,
    ctx: ServingContext | None = None,
    dry_run: bool = True,
    should_stop: Any = _never_stop,
    query: str = "after:0",
) -> Any:
    own_db = db is None
    run_db: Any = db if db is not None else _db(tmp_path)
    try:
        return run_apply(
            client=client,
            db=run_db,
            model=model or FakeModel(VOCAB, scores=_scores()),
            ctx=ctx or _ctx(),
            query=query,
            clock=lambda: TS,
            should_stop=should_stop,
            dry_run=dry_run,
        )
    finally:
        if own_db:
            run_db.close()


# -- AC1: preview = zero wire writes ------------------------------------------


def test_preview_writes_nothing_and_reports(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # AC1: dry preview, ZERO Gmail writes — the recorded calls are a
    # subset of the READ surface, no apply_labels, and the fake's write
    # ledger is EMPTY; the would-apply line is on the log (the
    # shadow-report vocabulary: id + label NAMES + band/tier + scores)
    caplog.set_level(logging.INFO)
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    outcome = _run_apply(tmp_path, fake, dry_run=True)
    assert outcome.written == 0 and outcome.rows == 0
    assert fake.writes == []  # the AC1 fact, not a display convention
    methods = {call.split(":", 1)[0] for call in fake.calls}
    assert methods <= READ_SURFACE, f"write call recorded: {methods - READ_SURFACE}"
    assert "list_messages" in methods and "get_message" in methods
    # the write candidate (aa22) is REPORTED would-apply; the skip-present
    # one (aa11) earns no would-apply line (empty plan)
    assert "aa22 would-apply Money/Bills" in caplog.text
    assert "aa11 would-apply" not in caplog.text
    # the label_ids on the fake are UNTOUCHED (a preview mutates nothing)
    assert fake.messages["aa22"].label_ids == ("INBOX",)


def test_preview_bodies_never_in_output(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # invariant #5 at the apply surface: the marker rides a real scored
    # message; the preview line carries ids/labels/scores only
    caplog.set_level(logging.INFO)
    marker = "SECRET-BODY-MARKER"
    msgs = _messages()
    for m in msgs:
        if m.id == "aa22":
            m.eml = _eml("two", f"{marker} body two")
    scores = dict(_scores())
    del scores["two\nbody two"]
    scores[f"two\n{marker} body two"] = (0.9, 0.1)
    fake = FakeGmailClient(messages=msgs, labels=_labels())
    _run_apply(tmp_path, fake, dry_run=True, model=FakeModel(VOCAB, scores=scores))
    assert "aa22 would-apply" in caplog.text  # non-vacuity: the line landed
    assert marker not in caplog.text


# -- AC2: one-core parity (structural + semantic) -----------------------------


def test_apply_imports_the_daemon_decision_core_not_a_copy() -> None:
    # the STRUCTURAL one-core claim (AC2, mutation-pinned): apply's
    # composition entry IS the daemon's — the SAME compose_plan object
    # (which threads the ctx maps to the one plan_message_write core).
    # Mutant: give apply a local re-implementation of the composition (or
    # of the core) and this identity check reddens — the review's
    # "never mirrors" is now true of the PLUMBING (compose_plan), not just
    # the decision (plan_message_write)
    import gmail_labeler.apply as apply_module
    import gmail_labeler.daemon as daemon_module

    assert apply_module.compose_plan is daemon_module.compose_plan
    # and the shared composition reaches the ONE core, not a copy
    assert (
        daemon_module.compose_plan.__globals__["plan_message_write"]
        is applications_plan
    )


def test_one_daemon_tick_and_one_apply_batch_land_identical_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC2 semantic parity: instrument the ONE core (plan_message_write,
    # which BOTH paths reach through the shared compose_plan), run ONE
    # daemon tick AND ONE apply batch over IDENTICAL fixture mail (same
    # scores, same present sets, same clock) → both land with IDENTICAL
    # plans (same core, same inputs, same outputs; only the source literal
    # differs — the literal that entered the path). The spy is on the
    # daemon module's plan_message_write because compose_plan (daemon.py)
    # calls THAT module-global — so a single patch intercepts both the
    # daemon's and apply's compositions, which IS the one-core proof.
    import gmail_labeler.daemon as daemon_module

    recorded: list[Any] = []
    real = applications_plan

    def spy(message_id: str, **kw: Any) -> Any:
        plan = real(message_id, **kw)
        recorded.append((message_id, kw["source"], plan))
        return plan

    monkeypatch.setattr(daemon_module, "plan_message_write", spy)

    # the daemon tick: a delta naming aa11 + aa22, cursor preset
    daemon_fake = FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        history_pages=[
            HistoryPage(
                history_id=None,
                next_page_token=None,
                records=(
                    {
                        "id": 5,
                        "messages": [
                            {"id": "aa11", "threadId": "t1"},
                            {"id": "aa22", "threadId": "t2"},
                        ],
                    },
                ),
            )
        ],
        current_history_id=900,
    )
    ddb = _db(tmp_path)
    ddb.set_history_cursor(900)
    driver_sleeps: list[float] = []

    def stop_after_one_sleep(seconds: float) -> None:
        driver_sleeps.append(seconds)

    daemon_loop(
        client=daemon_fake,
        db=ddb,
        model=FakeModel(VOCAB, scores=_scores()),
        ctx=_ctx(),
        poll_seconds=30,
        clock=lambda: TS,
        sleep=stop_after_one_sleep,
        should_stop=lambda: len(driver_sleeps) >= 1,
    )
    ddb.close()
    daemon_calls = [(m, p) for m, s, p in recorded if s == "daemon"]
    assert len(daemon_calls) == 2  # both messages planned through the core
    recorded.clear()

    # the apply batch: list_messages serves the SAME two messages
    apply_fake = FakeGmailClient(messages=_messages(), labels=_labels())
    adb = AppDB(tmp_path / "apply-state.db")  # a separate DB for the apply run
    adb.open()
    _run_apply(tmp_path, apply_fake, db=adb, dry_run=False)
    adb.close()
    apply_calls = [(m, p) for m, s, p in recorded if s == "apply"]
    assert len(apply_calls) == 2

    # IDENTICAL plans (same message order, same adds, same rows modulo
    # the source literal — the one core, same inputs, same outputs)
    assert [m for m, _ in daemon_calls] == [m for m, _ in apply_calls]
    for (dm, dplan), (am, aplan) in zip(daemon_calls, apply_calls, strict=True):
        assert dm == am
        assert dplan.adds == aplan.adds
        assert dplan.warning == aplan.warning is None
        # rows identical except the source literal
        assert [(r.label, r.label_id, r.score, r.band, r.tier) for r in dplan.rows] == [
            (r.label, r.label_id, r.score, r.band, r.tier) for r in aplan.rows
        ]
        # the source literal is the ONLY difference (empty plans — aa11
        # skip-present — carry no rows to source; the write candidate
        # aa22 proves each path stamps its own literal)
        assert {r.source for r in dplan.rows} <= {"daemon"}
        assert {r.source for r in aplan.rows} <= {"apply"}
    # non-vacuity: the write candidate really produced rows on both paths
    aa22_d = next(p for m, p in daemon_calls if m == "aa22")
    aa22_a = next(p for m, p in apply_calls if m == "aa22")
    assert {r.source for r in aa22_d.rows} == {"daemon"}
    assert {r.source for r in aa22_a.rows} == {"apply"}


# -- AC3: backfilled rows recorded identically to daemon rows -----------------


def test_backfill_records_apply_sourced_rows(tmp_path: Path) -> None:
    # AC3: with --apply the backfill records Applications IDENTICALLY to
    # daemon-applied ones (same table, same row shape; the source literal
    # is "apply" — the literal that entered the write path)
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    db = _db(tmp_path)
    outcome = _run_apply(tmp_path, fake, db=db, dry_run=False)
    assert outcome.written == 1 and outcome.rows == 1
    rows = db.applications_for_message("aa22")
    assert [(r.label, r.label_id, r.score, r.band, r.tier, r.source) for r in rows] == [
        ("Money/Bills", "L2", 0.9, "high", "ai-high", "apply")
    ]
    db.close()


def test_backfill_writes_first_then_commits(tmp_path: Path) -> None:
    # the write-first ordering IDENTICAL to the daemon: apply_labels lands
    # BEFORE the DB transaction (the crash-replay safety proof, inherited)
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    db = _db(tmp_path)
    order: list[str] = []
    real_commit = db.commit_applications

    def commit_then(rows: Any) -> int:
        order.append("commit")
        return real_commit(rows)

    db.commit_applications = commit_then  # type: ignore[method-assign]
    real_apply = fake.apply_labels

    def apply_then(message_id: str, adds: Any) -> None:
        order.append("gmail")
        return real_apply(message_id, adds)

    fake.apply_labels = apply_then  # type: ignore[method-assign]
    _run_apply(tmp_path, fake, db=db, dry_run=False)
    assert order == ["gmail", "commit"]  # Gmail write first, DB commit last
    db.close()


# -- AC4: interrupt + re-run idempotent (the fake really mutates) -------------


def test_midpage_interrupt_completes_batch_then_rerun_is_idempotent(
    tmp_path: Path,
) -> None:
    # AC4: SIGTERM mid-page → the batch STILL runs to its commit point
    # (every message in the page is written + committed), exit; a RE-RUN
    # over the same window is idempotent — skip-present (invariant #2)
    # makes the already-landed label compose an EMPTY plan: zero
    # double-writes, zero duplicate rows. Non-vacuity: the fake really
    # mutates label_ids (the #25 standard)
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    db = _db(tmp_path)
    # the stop flag fires after the FIRST message is fetched (mid-page)
    seen = {"n": 0}

    def stop_after_one() -> bool:
        return seen["n"] >= 1

    real_get = fake.get_message

    def counting_get(message_id: str) -> Any:
        result = real_get(message_id)
        seen["n"] += 1
        return result

    fake.get_message = counting_get  # type: ignore[method-assign]
    outcome = _run_apply(
        tmp_path, fake, db=db, dry_run=False, should_stop=stop_after_one
    )
    # the batch completed to its commit point: aa22's write landed + row
    assert outcome.written == 1 and outcome.rows == 1
    assert fake.messages["aa22"].label_ids == (
        "INBOX",
        "L2",
        "LA",
        "LH",
    )  # really mutated
    writes_after_first = len([w for w in fake.writes if w["method"] == "apply_labels"])
    rows_after_first = db.applications_for_message("aa22")
    assert len(rows_after_first) == 1
    db.close()

    # RE-RUN over the same window: aa22's labels are now PRESENT →
    # skip-present → EMPTY plan → no second write, no duplicate row
    db2 = _db(tmp_path)
    outcome2 = _run_apply(tmp_path, fake, db=db2, dry_run=False)
    writes_after_second = len([w for w in fake.writes if w["method"] == "apply_labels"])
    assert writes_after_second == writes_after_first  # zero double-writes
    assert outcome2.written == 0 and outcome2.rows == 0
    assert len(db2.applications_for_message("aa22")) == 1  # zero duplicate rows
    db2.close()


# -- AC6: System Labels never targeted on the apply path ----------------------


def test_apply_never_targets_a_system_label_even_via_a_lying_map(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # AC6 (this path's named pin): a lying resolution map that points a
    # would-apply label at a System Label id composes NO write — the
    # composition-level _resolve_id guard refuses it into the fail-safe
    # net (a counted warning, no wire call). Mutant: drop the guard and
    # the write lands on INBOX → fake.writes non-empty → red
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    ctx = _ctx(managed_label_ids={"Money/Bills": "INBOX", "Shipping": "L1"})
    outcome = _run_apply(tmp_path, fake, ctx=ctx, dry_run=False)
    assert outcome.written == 0
    assert fake.writes == []
    assert "fail-safe" in caplog.text and "INBOX" in caplog.text


# -- landmine: 404-vanished mail is a counted skip, never fatal ---------------


def test_vanished_message_is_counted_skip_not_fatal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # a backfill over years WILL hit deleted mail: a typed 404 on the read
    # is a counted SKIP (the run continues, the other message still
    # applies), never fatal (there is no cursor to wedge)
    caplog.set_level(logging.WARNING)
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    fake.fail("get_message", GmailPermanentError("message gone", status=404))
    db = _db(tmp_path)
    outcome = _run_apply(tmp_path, fake, db=db, dry_run=False)
    assert outcome.skipped == 1  # the vanished message counted
    assert outcome.written == 1  # the surviving message still applied
    assert "skipped message" in caplog.text
    db.close()


def test_message_vanished_between_fetch_and_write_is_counted_skip(
    tmp_path: Path,
) -> None:
    # the WRITE-path 404 arm (a message deleted between get_message and
    # apply_labels): counted SKIPPED only, no row (rows record
    # Applications that DID land) — the delta-free claim the read-path
    # test does not cover
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    fake.fail("apply_labels", GmailPermanentError("message gone", status=404))
    db = _db(tmp_path)
    outcome = _run_apply(tmp_path, fake, db=db, dry_run=False)
    assert outcome.skipped == 1 and outcome.written == 0 and outcome.rows == 0
    assert (
        db.applications_for_message("aa22") == ()
    )  # no row for a write that did not land
    assert fake.messages["aa22"].label_ids == ("INBOX",)  # the mail is untouched
    db.close()


# -- AC6: the mailbox System Label NAME set is forwarded to the core ----------


def test_apply_forwards_system_label_names_to_the_core(tmp_path: Path) -> None:
    # AC6 on THIS path, the NAME guard (not just the id guard): a lying
    # Graduation Table that graduated a System Label NAME, resolved to a
    # real USER id (so the composition's _resolve_id id-screen does NOT
    # fire) — only tiers' graduated-System-Label NAME refusal, fed by the
    # forwarded ctx.system_labels, keeps it off the wire. Mutant: drop
    # system_labels from the plan call → the name sails through and WRITES
    ctx = _ctx(
        graduated=frozenset({"INBOX"}),
        system_labels=frozenset({"INBOX"}),
        managed_label_ids={"INBOX": "L1"},  # a USER id — the id guard is blind here
        bands={"INBOX": (0.8, 0.6)},
    )
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    db = _db(tmp_path)
    outcome = _run_apply(
        tmp_path,
        fake,
        ctx=ctx,
        db=db,
        dry_run=False,
        model=FakeModel(
            ("INBOX",),
            scores={"one\nbody one": (0.95,), "two\nbody two": (0.95,)},
        ),
    )
    assert outcome.written == 0 and outcome.rows == 0
    assert fake.writes == []  # the graduated System Label NAME never reached the wire
    db.close()


# -- landmine: transient write failure holds nothing, re-run picks it up ------


def test_transient_write_failure_holds_nothing_rerun_picks_it_up(
    tmp_path: Path,
) -> None:
    # the distinguishing shape (no cursor to hold): a transient write
    # error leaves the message UN-labeled and UN-rowed (a counted warning);
    # a RE-RUN picks it up (not in the DB, not on the mail). Contrast the
    # daemon, which HOLDS the cursor and retries the whole delta
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    fake.fail("apply_labels", GmailTransientError("blip"))
    db = _db(tmp_path)
    outcome = _run_apply(tmp_path, fake, db=db, dry_run=False)
    assert outcome.warnings == 1
    assert outcome.written == 0
    assert fake.messages["aa22"].label_ids == ("INBOX",)  # NOT labeled
    assert db.applications_for_message("aa22") == ()  # NOT rowed
    db.close()

    # re-run (the blip cleared): the message is picked up — label lands,
    # row commits
    db2 = _db(tmp_path)
    outcome2 = _run_apply(tmp_path, fake, db=db2, dry_run=False)
    assert outcome2.written == 1 and outcome2.rows == 1
    assert "L2" in fake.messages["aa22"].label_ids
    assert len(db2.applications_for_message("aa22")) == 1
    db2.close()


def test_transient_READ_failure_holds_nothing_continues(tmp_path: Path) -> None:
    # item C (the read/write asymmetry the review named): a NON-404 per-
    # message READ error takes the SAME safe shape as a write blip — a
    # counted warning, the run CONTINUES (the surviving message still
    # applies), and the failed message stays un-labeled for a re-run. The
    # daemon's hold-the-cursor rationale does NOT transfer to a cursorless
    # path, so a transient read blip must not abort a whole backfill.
    # Mutant: revert the read arm to `raise` → run_apply raises
    # GmailTransientError → this node errors (no outcome).
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    fake.fail("get_message", GmailTransientError("read blip"))  # aa11 (first) blips
    db = _db(tmp_path)
    outcome = _run_apply(tmp_path, fake, db=db, dry_run=False)
    assert outcome.warnings == 1  # the blipped read counted as a WARNING
    assert outcome.skipped == 0  # NOT a skip (it is not a 404 vanish)
    assert outcome.written == 1  # the run CONTINUED: aa22 still applied
    assert fake.messages["aa22"].label_ids != ("INBOX",)
    # the un-labeled AND un-rowed half of the shape: the blipped message
    # left no audit row (a re-run is what picks it up), and the one row
    # that exists is aa22's
    assert outcome.rows == 1
    assert db.applications_for_message("aa11") == ()
    db.close()


def test_duplicate_message_id_in_one_page_processed_once(tmp_path: Path) -> None:
    # item D (the daemon's _message_ids dedupe, now on the apply path): a
    # message served TWICE in one page is processed ONCE. Without the
    # dedupe the second pass re-composes an IDENTICAL row and
    # commit_applications refuses the within-batch (message_id, label)
    # DUPLICATE with exit 5 — AFTER the first pass's Gmail write LANDED
    # (labels on the mail, no audit row: the review's executed probe).
    # Mutant: drop the `seen` dedupe → AppDBError (duplicate row) → red.
    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    fake.order = ["aa22", "aa22"]  # the same id served twice in one page
    db = _db(tmp_path)
    outcome = _run_apply(tmp_path, fake, db=db, dry_run=False)
    assert outcome.messages == 1  # deduped: one message, not two
    assert outcome.written == 1 and outcome.rows == 1  # single write, single row
    assert len([w for w in fake.writes if w["method"] == "apply_labels"]) == 1
    assert len(db.applications_for_message("aa22")) == 1
    db.close()


def _hammer_mailbox() -> list[FakeMessage]:
    # three clean WRITE candidates: Money/Bills absent (INBOX only), so a
    # successful read composes exactly ONE write — warnings then count one
    # per message, which is what the posture pin asserts on.
    return [
        FakeMessage(
            id=f"hh{i}",
            eml=_eml(f"s{i}", f"b{i}"),
            label_ids=("INBOX",),
            thread_id=f"t{i}",
        )
        for i in (1, 2, 3)
    ]


def _hammer_model() -> FakeModel:
    # Money/Bills high (0.9 >= the 0.8 band), Shipping low -> one write per
    # message. Used by the write arm; the read arm never reaches predict.
    return FakeModel(VOCAB, scores={f"s{i}\nb{i}": (0.9, 0.1) for i in (1, 2, 3)})


def test_rate_limited_mailbox_walks_to_end_counted_v1_posture(
    tmp_path: Path,
) -> None:
    # F2 (manager-ruled v1 posture): a quota-limited mailbox
    # (GmailRateLimitedError, kind=BACKOFF) is walked to the END with
    # counted warnings — NO abort, NO backoff, exit 0. This is the shape the
    # cursorless apply path trades for its read/write symmetry (item C): it
    # hammers the window one message at a time and keeps going. Pinning the
    # OBSERVABLE (N per-message attempts + warnings=N + written=0 + exit 0)
    # makes any future posture change (back off, or abort on a BACKOFF-kind
    # error) a deliberate RED, not a silent drift. Both arms share the one
    # posture node — the manager's "both arms, one node" structure call.
    #
    # READ arm: every get_message 429s -> fetched is EMPTY -> no predict,
    # no write, no row; three counted warnings; three get_message attempts
    # (the hammer). Mutant: escalate the read arm on kind=BACKOFF (raise)
    # -> run_apply propagates -> this node errors (no outcome).
    fake = FakeGmailClient(messages=_hammer_mailbox(), labels=_labels())
    for _ in range(3):
        fake.fail("get_message", GmailRateLimitedError("429", status=429))
    db = _db(tmp_path)
    outcome = _run_apply(tmp_path, fake, db=db, dry_run=False, model=_hammer_model())
    assert outcome.warnings == 3  # counted per message, no abort
    assert outcome.written == 0 and outcome.rows == 0  # nothing landed
    assert len([c for c in fake.calls if c.startswith("get_message:")]) == 3
    assert not [c for c in fake.calls if c.startswith("apply_labels:")]
    db.close()

    # WRITE arm: reads succeed, every apply_labels 429s -> three composed
    # writes each HELD as a counted warning (no row — rows record writes
    # that LANDED), three apply_labels attempts. Mutant: escalate the write
    # arm on kind=BACKOFF -> propagates -> red.
    fake2 = FakeGmailClient(messages=_hammer_mailbox(), labels=_labels())
    for _ in range(3):
        fake2.fail("apply_labels", GmailRateLimitedError("429", status=429))
    db2 = _db(tmp_path)
    outcome2 = _run_apply(tmp_path, fake2, db=db2, dry_run=False, model=_hammer_model())
    assert outcome2.warnings == 3
    assert outcome2.written == 0 and outcome2.rows == 0
    assert len([c for c in fake2.calls if c.startswith("apply_labels:")]) == 3
    assert len([c for c in fake2.calls if c.startswith("get_message:")]) == 3
    db2.close()
    # the re-run-picks-up-the-missed-messages half of this posture is
    # ALREADY pinned (test_transient_write_failure_holds_nothing_rerun_picks_it_up
    # + the AC4 re-run idempotency node) — cited here, not re-pinned.


# -- landmine: >1-page paging fidelity ----------------------------------------


def test_stop_between_pages_does_not_pay_for_another_page(tmp_path: Path) -> None:
    # the between-PAGES should_stop sample (the daemon's shape): a flag
    # that flips AFTER a batch's message loop but before the next page
    # must stop the loop — the current batch reached its commit point, so
    # no further Gmail writes. Mutant: drop the between-pages check and
    # page 2 is fetched, scored, WRITTEN and committed (batches=2)
    msgs = [
        FakeMessage(
            id="aa22", eml=_eml("two", "body two"), label_ids=("INBOX",), thread_id="t2"
        ),
        FakeMessage(
            id="aa44",
            eml=_eml("four", "body four"),
            label_ids=("INBOX",),
            thread_id="t4",
        ),
    ]
    scores = {
        "two\nbody two": (0.9, 0.1),  # Money/Bills high → write
        "four\nbody four": (0.2, 0.9),  # Shipping high → write
    }
    fake = FakeGmailClient(messages=msgs, labels=_labels(), page_size=1)
    db = _db(tmp_path)
    calls: list[int] = []

    def stop_after_first_page() -> bool:
        # the batch-1 message-loop sample returns False (the batch
        # completes); the between-pages sample (the 2nd call) returns True
        calls.append(1)
        return len(calls) >= 2

    outcome = run_apply(
        client=fake,
        db=db,
        model=FakeModel(VOCAB, scores=scores),
        ctx=_ctx(),
        query="after:0",
        clock=lambda: TS,
        should_stop=stop_after_first_page,
        dry_run=False,
    )
    assert outcome.batches == 1  # page 2 never processed
    assert outcome.written == 1  # only aa22 landed; aa44's page was not paid for
    assert "get_message:aa44" not in fake.calls
    db.close()


def test_preview_pages_through_multiple_pages(tmp_path: Path) -> None:
    # the wire pageToken chain: a preview over >1 page must page correctly
    # (idle/last-page behavior mirrors history paging fidelity)
    msgs = _messages()
    msgs.append(
        FakeMessage(
            id="aa33",
            eml=_eml("three", "body three"),
            label_ids=("INBOX",),
            thread_id="t3",
        )
    )
    scores = dict(_scores())
    scores["three\nbody three"] = (0.9, 0.1)  # Money/Bills high, absent
    fake = FakeGmailClient(messages=msgs, labels=_labels(), page_size=1)
    outcome = _run_apply(
        tmp_path, fake, dry_run=True, model=FakeModel(VOCAB, scores=scores)
    )
    # 3 messages over 3 pages of size 1
    assert outcome.batches == 3
    assert outcome.messages == 3
    tokens = [c for c in fake.calls if c.startswith("list_messages:")]
    assert tokens == [
        "list_messages:after:0:",
        "list_messages:after:0:1",
        "list_messages:after:0:2",
    ]


# -- landmine: empty plan = no write AND no row ------------------------------


def test_empty_plan_no_write_no_row(tmp_path: Path) -> None:
    # the #24 consequence holds: skip-present + empty add set → no API
    # call AND no row (aa11's Shipping is present → empty plan)
    fake = FakeGmailClient(messages=[_messages()[0]], labels=_labels())
    db = _db(tmp_path)
    outcome = _run_apply(tmp_path, fake, db=db, dry_run=False)
    assert outcome.written == 0 and outcome.rows == 0
    assert fake.writes == []
    assert db.applications_for_message("aa11") == ()
    db.close()


# -- landmine: daemon_state byte-for-byte untouched across a backfill ---------


def test_backfill_leaves_daemon_state_untouched(tmp_path: Path) -> None:
    # apply NEVER reads or writes daemon_state (invariant #3 is the
    # daemon's alone): a live daemon's cursor stands where it was across
    # a whole backfill
    import sqlite3

    def _state_rows() -> list[tuple[int, int]]:
        conn = sqlite3.connect(str(db.path))
        try:
            return conn.execute(
                "SELECT rowid, history_cursor FROM daemon_state ORDER BY rowid"
            ).fetchall()
        finally:
            conn.close()

    fake = FakeGmailClient(messages=_messages(), labels=_labels())
    db = _db(tmp_path)
    db.set_history_cursor(12345)
    before = _state_rows()
    _run_apply(tmp_path, fake, db=db, dry_run=False)
    after = _state_rows()
    assert before == after == [(1, 12345)]
    assert db.get_history_cursor() == 12345
    db.close()


# -- the batch homogeneity contract at the apply call site --------------------


def test_one_clock_read_per_batch(tmp_path: Path) -> None:
    # the #24 homogeneity contract REUSED at the apply call site: ONE
    # clock() read per batch stamps every row of that batch. NON-VACUOUS:
    # TWO messages both produce rows, so a per-message clock read would
    # stamp them :00 and :01 → commit_applications' homogeneity guard
    # raises → the run dies. A single per-batch read keeps them equal
    msgs = [
        FakeMessage(
            id="aa22",
            eml=_eml("two", "body two"),
            label_ids=("INBOX",),
            thread_id="t2",
        ),
        FakeMessage(
            id="aa44",
            eml=_eml("four", "body four"),
            label_ids=("INBOX",),
            thread_id="t4",
        ),
    ]
    scores = {
        "two\nbody two": (0.9, 0.1),  # Money/Bills high, absent → row
        "four\nbody four": (0.2, 0.9),  # Shipping high, absent → row
    }
    ticks = iter(
        [
            datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC),
            datetime(2026, 11, 1, 12, 0, 1, tzinfo=UTC),
        ]
    )
    fake = FakeGmailClient(messages=msgs, labels=_labels())
    db = _db(tmp_path)
    outcome = run_apply(
        client=fake,
        db=db,
        model=FakeModel(VOCAB, scores=scores),
        ctx=_ctx(),
        query="after:0",
        clock=lambda: next(ticks),
        should_stop=_never_stop,
        dry_run=False,
    )
    # both rows landed with the SAME stamp (one clock read for the batch)
    assert outcome.rows == 2
    stamps = {r.applied_at for r in db.applications_for_message("aa22")} | {
        r.applied_at for r in db.applications_for_message("aa44")
    }
    assert stamps == {TS}  # a per-message read would carry :01 too
    db.close()
