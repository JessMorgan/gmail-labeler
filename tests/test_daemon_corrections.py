"""Correction DETECTION in the daemon tick (ticket #27).

A SEPARATE suite from the sealed #25 daemon tests (which stay verbatim —
this file drives the SAME tick with history records that carry
labelsAdded/labelsRemoved events, which the #25 fixtures never produce).
The heart: detection is read + local-DB only, never writes back to
Gmail (AC2), the ignore-boundary records nothing, a corrected message is
never re-written, and the correction rides the delta's one commit point.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gmail_labeler.appdb import AppDB
from gmail_labeler.applications import ApplicationRow
from gmail_labeler.cli import _ServeStartup
from gmail_labeler.daemon import ServingContext, daemon_loop
from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
from gmail_labeler.gmail_client import HistoryPage, LabelInfo
from gmail_labeler.model import FakeModel

# the read surface the correction path may touch (the shadow/apply
# idiom, EXTENDED to the daemon's correction arm — AC2: a Correction is
# detected by READING history + the local join, never by a Gmail write)
READ_SURFACE = frozenset(
    {"history", "get_message", "list_messages", "list_labels", "list_filters"}
)

TS = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)
VOCAB = ("Money/Bills", "Shipping")


def _eml(subject: str) -> bytes:
    return (
        f"Subject: {subject}\r\nFrom: a@x.com\r\nTo: me@x.com\r\n"
        "Content-Type: text/plain\r\n\r\nbody"
    ).encode()


def _messages() -> list[FakeMessage]:
    # aa11: Shipping applied (a prior Application); aa22: Money/Bills
    # applied. Both INBOX-only now (the human's removal is the delta).
    return [
        FakeMessage(
            id="aa11",
            eml=_eml("one"),
            label_ids=("INBOX", "L2"),
            thread_id="t1",
            history_id=901,
        ),
        FakeMessage(
            id="aa22",
            eml=_eml("two"),
            label_ids=("INBOX",),
            thread_id="t2",
            history_id=901,
        ),
    ]


def _labels() -> list[LabelInfo]:
    return [
        LabelInfo("INBOX", "INBOX", "system"),
        LabelInfo("L1", "Shipping", "user"),
        LabelInfo("L2", "Money/Bills", "user"),
        LabelInfo("LX", "NotGraduated", "user"),
        LabelInfo("LA", "ai-applied", "user"),
        LabelInfo("LH", "ai-high", "user"),
        LabelInfo("LM", "ai-med", "user"),
    ]


def _ctx(**over: Any) -> ServingContext:
    base: dict[str, Any] = {
        "bands": {"Money/Bills": (0.8, 0.6), "Shipping": (0.8, 0.6)},
        "graduated": frozenset({"Money/Bills", "Shipping"}),
        "system_labels": frozenset({"INBOX"}),
        "id_to_name": {"L1": "Shipping", "L2": "Money/Bills", "LX": "NotGraduated"},
        "managed_label_ids": {"Money/Bills": "L2", "Shipping": "L1"},
        "tier_label_ids": {"ai-applied": "LA", "ai-high": "LH", "ai-med": "LM"},
        "filters": (),
    }
    base.update(over)
    return ServingContext(**base)  # pyright: ignore[reportArgumentType]


def _record(
    *message_ids: str,
    hid: int = 1,
    added: list[tuple[str, list[str]]] | None = None,
    removed: list[tuple[str, list[str]]] | None = None,
) -> dict[str, object]:
    # the REAL Gmail wire shape (discovery rev 20261026): PLURAL
    # labelsAdded/labelsRemoved, each element nesting the Message under
    # 'message' (id at message.id). This is what the adapter delivers raw.
    rec: dict[str, object] = {
        "id": hid,
        "messages": [{"id": m, "threadId": f"t{m}"} for m in message_ids],
    }
    if added:
        rec["labelsAdded"] = [
            {"message": {"id": m}, "labelIds": lids} for m, lids in added
        ]
    if removed:
        rec["labelsRemoved"] = [
            {"message": {"id": m}, "labelIds": lids} for m, lids in removed
        ]
    return rec


def _page(*records: dict[str, object], token: str | None = None) -> HistoryPage:
    return HistoryPage(history_id=None, next_page_token=token, records=records)


def _fake(*pages: HistoryPage) -> FakeGmailClient:
    return FakeGmailClient(
        messages=_messages(),
        labels=_labels(),
        current_history_id=901,
        history_pages=list(pages),
    )


def _db(tmp_path: Path) -> AppDB:
    db = AppDB(tmp_path / "state.db")
    db.open()
    db.set_history_cursor(900)  # the daemon polls from an established cursor
    return db


def _applied(db: AppDB, message_id: str, label: str, label_id: str) -> None:
    db.commit_applications(
        [
            ApplicationRow(
                message_id=message_id,
                label=label,
                label_id=label_id,
                score=0.9,
                band="high",
                tier="ai-high",
                applied_at=TS,
                source="daemon",
            )
        ]
    )


class _LoopRun:
    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self.stopped = False

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.stopped = True  # stop after the first tick's sleep

    def should_stop(self) -> bool:
        return self.stopped


def _tick(
    tmp_path: Path,
    client: Any,
    db: Any,
    *,
    model: Any | None = None,
    ctx: ServingContext | None = None,
) -> int:
    driver = _LoopRun()
    return daemon_loop(
        client=client,
        db=db,
        model=model or FakeModel(VOCAB, scores={}, default=0.1),
        ctx=ctx or _ctx(),
        poll_seconds=30,
        clock=lambda: TS,
        sleep=driver.sleep,
        should_stop=driver.should_stop,
    )


# -- 'removed': a human undid our applied label -------------------------------


def test_removed_applied_label_records_correction(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    fake = _fake(_page(_record("aa22", removed=[("aa22", ["L2"])])))
    assert _tick(tmp_path, fake, db) == 0
    corr = db.corrections_for_message("aa22")
    assert [(c.label, c.kind) for c in corr] == [("Money/Bills", "removed")]
    assert db.get_history_cursor() == 901  # the delta committed atomically
    db.close()


def test_corrected_message_is_never_written(tmp_path: Path) -> None:
    # the LANDMINE: the human removed Money/Bills from aa22; aa22 still
    # scores high for it, so the plan WOULD re-add it — but a corrected
    # message is NEVER written this delta. Pin: no apply_labels for aa22,
    # no new Application row, yet the correction IS recorded.
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    fake = _fake(_page(_record("aa22", removed=[("aa22", ["L2"])])))
    # aa22 scores high for Money/Bills (would re-add if not suppressed)
    model = FakeModel(VOCAB, scores={"two\nbody": (0.95, 0.1)})
    assert _tick(tmp_path, fake, db, model=model) == 0
    assert "apply_labels:aa22" not in fake.calls  # never written
    assert fake.writes == []  # the AC2 fact: zero Gmail writes recorded
    assert len(db.applications_for_message("aa22")) == 1  # only the PRIOR row
    assert len(db.corrections_for_message("aa22")) == 1  # the correction landed
    db.close()


# -- 'added': a human did our job (positive signal) ---------------------------


def test_added_managed_label_records_correction(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")  # aa11 applied for Shipping
    # the human ADDED Money/Bills (L2) to aa11 — a graduated label we did
    # not write for it (aa11's Application is Shipping, not Money/Bills)
    fake = _fake(_page(_record("aa11", added=[("aa11", ["L2"])])))
    assert _tick(tmp_path, fake, db) == 0
    corr = db.corrections_for_message("aa11")
    assert [(c.label, c.kind) for c in corr] == [("Money/Bills", "added")]
    # _ctx()'s default filters=() is an ATTESTED-NONE snapshot (the operator
    # ran the export and has no filters) -> no relevant filter claims the
    # label -> the human added it (positive training signal). A MISSING
    # snapshot is the SEPARATE None case, pinned below as 'unverifiable'.
    assert corr[0].attribution == "human"
    db.close()


def test_added_correction_unknown_snapshot_is_unverifiable(tmp_path: Path) -> None:
    # F1 MAJOR (invariant #6, one level up): a MISSING filters snapshot
    # (ctx.filters is None) is authorship-UNKNOWN, NOT "no filters" -- every
    # 'added' is withheld as 'unverifiable', never credited to the human.
    # This is the same fail-open F1 removed (the bool defaulted an
    # unverifiable add to human), now at the snapshot level; analyze.py's
    # filters=None posture is the reused #22 precedent.
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _fake(_page(_record("aa11", added=[("aa11", ["L2"])])))
    ctx = _ctx(filters=None)  # NO usable snapshot
    assert _tick(tmp_path, fake, db, ctx=ctx) == 0
    corr = db.corrections_for_message("aa11")
    assert [(c.label, c.kind, c.attribution) for c in corr] == [
        ("Money/Bills", "added", "unverifiable")
    ]
    db.close()


def test_removed_correction_still_human_under_unknown_snapshot(
    tmp_path: Path,
) -> None:
    # the unknown-snapshot withhold applies ONLY to 'added'; a 'removed' is
    # ALWAYS 'human' (a human removing is human signal regardless of whether
    # we can see the filters -- invariant #6), even with ctx.filters None.
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    fake = _fake(_page(_record("aa22", removed=[("aa22", ["L2"])])))
    ctx = _ctx(filters=None)
    assert _tick(tmp_path, fake, db, ctx=ctx) == 0
    corr = db.corrections_for_message("aa22")
    assert corr[0].kind == "removed" and corr[0].attribution == "human"
    db.close()


# -- the ignore-boundary, BOTH directions -------------------------------------


def test_removed_nonapplied_label_records_nothing(tmp_path: Path) -> None:
    # a labelsRemoved event for a label that is NOT this message's Application
    # noise (the human removed something we never applied) -> NOTHING
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    fake = _fake(_page(_record("aa22", removed=[("aa22", ["L1"])])))  # L1 not applied
    assert _tick(tmp_path, fake, db) == 0
    assert db.corrections_for_message("aa22") == ()
    assert db.get_history_cursor() == 901  # the delta still handled
    db.close()


def test_added_nonvocabulary_label_records_nothing(tmp_path: Path) -> None:
    # a labelsAdded event for a NON-graduated label (LX=NotGraduated) or a
    # System Label (INBOX) is not signal -> NOTHING recorded
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _fake(_page(_record("aa11", added=[("aa11", ["LX"]), ("aa11", ["INBOX"])])))
    assert _tick(tmp_path, fake, db) == 0
    assert db.corrections_for_message("aa11") == ()
    db.close()


def test_added_label_the_daemon_already_wrote_records_nothing(tmp_path: Path) -> None:
    # a labelsAdded event for a label that IS this message's Application (the
    # daemon wrote it) is NOT a human 'added' -> nothing
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _fake(_page(_record("aa11", added=[("aa11", ["L1"])])))
    assert _tick(tmp_path, fake, db) == 0
    assert db.corrections_for_message("aa11") == ()
    db.close()


def test_correction_on_unapplied_message_records_nothing(tmp_path: Path) -> None:
    # the JOIN AC: a label event on a message NOT in the applications
    # table is out of scope (we only track corrections on APPLIED mail).
    # BOTH kinds are exercised: a 'removed' non-applied label is already
    # caught by the per-kind check, but an 'added' graduated label on an
    # UNAPPLIED message is caught ONLY by the join gate -- so this is the
    # case that makes `if not applied: continue` load-bearing (mutant M4:
    # drop the gate -> the added row records -> red).
    db = _db(tmp_path)  # NO _applied -> nothing in the join
    fake = _fake(
        _page(
            _record(removed=[("aa22", ["L2"])]),
            _record(added=[("aa11", ["L2"])]),  # graduated label, unapplied msg
        )
    )
    assert _tick(tmp_path, fake, db) == 0
    assert db.corrections_for_message("aa22") == ()
    assert db.corrections_for_message("aa11") == ()
    db.close()


# -- AC2: the correction path NEVER writes back to Gmail ----------------------


def test_correction_detection_makes_no_write_calls(tmp_path: Path) -> None:
    # AC2 mutation-pin: classification of label events is READ + local
    # DB only — the fake's call log is a subset of the READ surface and
    # its write ledger is EMPTY (no apply_labels, no ensure_label)
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    fake = _fake(_page(_record("aa22", removed=[("aa22", ["L2"])])))
    assert _tick(tmp_path, fake, db) == 0
    methods = {call.split(":", 1)[0] for call in fake.calls}
    assert methods <= READ_SURFACE, f"write call recorded: {methods - READ_SURFACE}"
    assert fake.writes == []
    assert len(db.corrections_for_message("aa22")) == 1  # the correction landed
    db.close()


# -- dedupe, non-vacuous by the #25 bar ---------------------------------------


def test_duplicate_label_event_in_one_delta_is_one_correction(tmp_path: Path) -> None:
    # the same hand-fix reported by TWO history records of one delta is
    # ONE Correction (detect_corrections dedupes the batch so the store's
    # within-batch refusal is never tripped). STATE assertion: one row.
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    fake = _fake(
        _page(
            _record("aa22", removed=[("aa22", ["L2"])]),
            _record("aa22", removed=[("aa22", ["L2"])]),
        )
    )
    assert _tick(tmp_path, fake, db) == 0
    assert len(db.corrections_for_message("aa22")) == 1  # STATE: one, not two
    db.close()


# -- filter attribution (invariant #6) ----------------------------------------


def test_added_correction_attributed_filter_covered(tmp_path: Path) -> None:
    # an 'added' Correction whose label a SIMULATABLE Gmail filter verifiably
    # would have applied (from: a@x.com adds L2, and the message IS from
    # a@x.com) is attributed 'filter_covered' -- withheld from human credit
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _fake(_page(_record("aa11", added=[("aa11", ["L2"])])))
    ctx = _ctx(
        filters=[{"criteria": {"from": "a@x.com"}, "action": {"addLabelIds": ["L2"]}}]
    )
    assert _tick(tmp_path, fake, db, ctx=ctx) == 0
    corr = db.corrections_for_message("aa11")
    assert [(c.label, c.kind, c.attribution) for c in corr] == [
        ("Money/Bills", "added", "filter_covered")
    ]
    db.close()


def test_added_correction_human_when_no_filter_adds_the_label(tmp_path: Path) -> None:
    # the SAME added label, but the snapshot's filter does NOT add it (it
    # adds L1) -> no RELEVANT filter claims Money/Bills -> 'human' (a
    # filter that verifiably does not touch the label is not a reason to
    # withhold the add; only a filter that ADDS the label is relevant)
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _fake(_page(_record("aa11", added=[("aa11", ["L2"])])))
    ctx = _ctx(
        filters=[{"criteria": {"from": "a@x.com"}, "action": {"addLabelIds": ["L1"]}}]
    )
    assert _tick(tmp_path, fake, db, ctx=ctx) == 0
    corr = db.corrections_for_message("aa11")
    assert corr[0].attribution == "human"
    db.close()


def test_removed_correction_always_attributed_human(tmp_path: Path) -> None:
    # invariant #6: a 'removed' Correction is human signal regardless of
    # filters -> it is ALWAYS 'human', even with a matching snapshot that
    # would have made an ADD filter_covered (a human removing is not the
    # filter's doing)
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    fake = _fake(_page(_record("aa22", removed=[("aa22", ["L2"])])))
    ctx = _ctx(
        filters=[{"criteria": {"from": "a@x.com"}, "action": {"addLabelIds": ["L2"]}}]
    )
    assert _tick(tmp_path, fake, db, ctx=ctx) == 0
    corr = db.corrections_for_message("aa22")
    assert corr[0].kind == "removed" and corr[0].attribution == "human"
    db.close()


def test_added_correction_unverifiable_when_filter_criteria_outside(
    tmp_path: Path,
) -> None:
    # F1 (invariant #6 restored): a RELEVANT filter (it adds the label)
    # whose criteria are OUTSIDE the simulatable set {from,to,subject}
    # (here: a 'query' criterion) makes authorship UNDECIDABLE -> the add
    # is attributed 'unverifiable' -- WITHHELD, not credited to the human.
    # The old bool INVERTED this (an unverifiable filter -> False -> human);
    # #22's exclusion principle says guessing either way is forbidden.
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _fake(_page(_record("aa11", added=[("aa11", ["L2"])])))
    ctx = _ctx(
        filters=[
            {"criteria": {"query": "has:attachment"}, "action": {"addLabelIds": ["L2"]}}
        ]
    )
    assert _tick(tmp_path, fake, db, ctx=ctx) == 0
    corr = db.corrections_for_message("aa11")
    assert [(c.label, c.kind, c.attribution) for c in corr] == [
        ("Money/Bills", "added", "unverifiable")
    ]
    db.close()


def test_added_correction_unreadable_addlabelids_is_unverifiable(
    tmp_path: Path,
) -> None:
    # F2 item 3 (entry-level #6 inversion): the reviewer's exact probe -- a
    # relevant filter whose addLabelIds is a bare STRING 'L2' (wrong shape).
    # The LENIENT reader saw zero ids -> the filter looked irrelevant -> the
    # add was credited HUMAN. The STRICT reader (analyze's, shared with the
    # builder + CLI bridge) refuses it, so authorship is undecidable ->
    # 'unverifiable' (withheld, not guessed -- #22).
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _fake(_page(_record("aa11", added=[("aa11", ["L2"])])))
    ctx = _ctx(
        filters=[{"criteria": {"from": "a@x.com"}, "action": {"addLabelIds": "L2"}}]
    )
    assert _tick(tmp_path, fake, db, ctx=ctx) == 0
    corr = db.corrections_for_message("aa11")
    assert [(c.label, c.kind, c.attribution) for c in corr] == [
        ("Money/Bills", "added", "unverifiable")
    ]
    db.close()


def test_label_events_plural_takes_precedence_over_singular() -> None:
    # F2 item 2 (dual-shape double-mint): a record carrying BOTH the plural
    # and the singular key with DIFFERENT labelIds mints events from the
    # PLURAL only -- the singular is a fallback read ONLY when the plural is
    # absent, never additive (reading both would double-mint one hand-fix).
    from gmail_labeler.daemon import _label_events

    rec = {
        "id": 1,
        "messages": [{"id": "aa11", "threadId": "t1"}],
        "labelsAdded": [{"message": {"id": "aa11"}, "labelIds": ["L2"]}],
        "labelAdded": [{"message": {"id": "aa11"}, "labelIds": ["L1"]}],
    }
    assert _label_events((rec,)) == [("added", "aa11", "L2")]


def test_label_events_singular_read_only_when_plural_absent() -> None:
    # the fallback still works: with NO plural key the singular IS read (the
    # defensive tolerance for the design brief's named shape)
    from gmail_labeler.daemon import _label_events

    rec = {
        "id": 1,
        "messages": [{"id": "aa11", "threadId": "t1"}],
        "labelRemoved": [{"message": {"id": "aa11"}, "labelIds": ["L1"]}],
    }
    assert _label_events((rec,)) == [("removed", "aa11", "L1")]


def test_added_correction_unverifiable_when_fields_unavailable(
    tmp_path: Path,
) -> None:
    # F1: a RELEVANT filter exists but the message's fields were NOT
    # fetched this delta (parsed_by_id lacks it) -> authorship cannot be
    # checked -> 'unverifiable' (withheld), NOT 'human'. The old bool
    # recorded False (human) here -- the same inversion, other branch.
    # (aa11 is the applied message; the added label L2 rides a DIFFERENT
    # message id 'ff00' that is NOT in _messages(), so it is unfetched --
    # but it must still be in the Applications join to be a correction.)
    db = _db(tmp_path)
    _applied(db, "ff00", "Shipping", "L1")  # ff00 applied (join row exists)
    fake = _fake(_page(_record("ff00", added=[("ff00", ["L2"])])))
    ctx = _ctx(
        filters=[{"criteria": {"from": "a@x.com"}, "action": {"addLabelIds": ["L2"]}}]
    )
    assert _tick(tmp_path, fake, db, ctx=ctx) == 0
    corr = db.corrections_for_message("ff00")
    assert [(c.label, c.kind, c.attribution) for c in corr] == [
        ("Money/Bills", "added", "unverifiable")
    ]
    db.close()


def test_added_correction_human_when_relevant_filter_verifiably_absent(
    tmp_path: Path,
) -> None:
    # F1 the OTHER direction (the exclusion is not a default-to-withhold):
    # a relevant SIMULATABLE filter that verifiably does NOT match this
    # message (from: nobody@x.com, message is from a@x.com) leaves the add
    # 'human' -- we PROVE the filter did not author it, so the human did.
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _fake(_page(_record("aa11", added=[("aa11", ["L2"])])))
    ctx = _ctx(
        filters=[
            {"criteria": {"from": "nobody@x.com"}, "action": {"addLabelIds": ["L2"]}}
        ]
    )
    assert _tick(tmp_path, fake, db, ctx=ctx) == 0
    corr = db.corrections_for_message("aa11")
    assert corr[0].attribution == "human"
    db.close()


def test_added_correction_filter_covered_dominates_unverifiable(
    tmp_path: Path,
) -> None:
    # mixed case, order-INDEPENDENT: the UNVERIFIABLE filter is listed
    # FIRST, the verifiably-matching one SECOND. A proven match must win
    # (filter_covered) regardless of iteration order -- an unverifiable
    # filter must not mask a filter we can PROVE authored the add.
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _fake(_page(_record("aa11", added=[("aa11", ["L2"])])))
    ctx = _ctx(
        filters=[
            {
                "criteria": {"query": "has:attachment"},
                "action": {"addLabelIds": ["L2"]},
            },
            {"criteria": {"from": "a@x.com"}, "action": {"addLabelIds": ["L2"]}},
        ]
    )
    assert _tick(tmp_path, fake, db, ctx=ctx) == 0
    corr = db.corrections_for_message("aa11")
    assert corr[0].attribution == "filter_covered"
    db.close()


def test_added_correction_unverifiable_when_relevant_filter_lacks_criteria(
    tmp_path: Path,
) -> None:
    # a relevant filter (it adds L2) whose 'criteria' key is ABSENT (a
    # malformed / non-mapping snapshot entry) -> authorship undecidable ->
    # 'unverifiable' (withheld), the same posture as outside-{from,to,subject}
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _page(_record("aa11", added=[("aa11", ["L2"])]))
    ctx = _ctx(filters=[{"action": {"addLabelIds": ["L2"]}}])
    assert _tick(tmp_path, _fake(fake), db, ctx=ctx) == 0
    corr = db.corrections_for_message("aa11")
    assert corr[0].attribution == "unverifiable"
    db.close()


# -- malformed label-event wire is a lying delta (hold, not crash) ------------


def test_malformed_label_event_holds_cursor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # a labelsAdded value that is not a list of objects is a lying wire -> the
    # delta HOLDS the cursor (the _message_ids shape doctrine extended to
    # label events), never a raw traceback
    caplog.set_level(logging.WARNING)
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    fake = _fake(_page({"id": 1, "labelsAdded": "not-a-list"}))
    assert _tick(tmp_path, fake, db) == 0
    assert db.get_history_cursor() == 900  # HELD, not advanced
    assert "cursor held" in caplog.text
    db.close()


def test_real_gmail_wire_shape_is_detected(tmp_path: Path) -> None:
    # ADAPTER-CONFORMANCE pin: the EXACT documented Gmail shape (discovery
    # rev 20261026) — PLURAL labelsRemoved, message id NESTED under
    # 'message' — is what the adapter delivers raw; detection must see it.
    # (The design brief's singular labelAdded/flat-id shape never appears
    # on the wire; this node is the guard against regressing to it.)
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    real_record = {
        "id": 1,
        "messages": [{"id": "aa22", "threadId": "t2"}],
        "labelsRemoved": [{"message": {"id": "aa22"}, "labelIds": ["L2"]}],
    }
    fake = _fake(_page(real_record))
    assert _tick(tmp_path, fake, db) == 0
    corr = db.corrections_for_message("aa22")
    assert [(c.label, c.kind) for c in corr] == [("Money/Bills", "removed")]
    db.close()


def test_flat_id_label_event_is_tolerated(tmp_path: Path) -> None:
    # defensive: a flat event['id'] (the generic messages[] form) is also
    # accepted, so the daemon never wedges on an id-nesting surprise
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    flat_record = {
        "id": 1,
        "messages": [{"id": "aa22", "threadId": "t2"}],
        "labelsRemoved": [{"id": "aa22", "labelIds": ["L2"]}],
    }
    fake = _fake(_page(flat_record))
    assert _tick(tmp_path, fake, db) == 0
    assert len(db.corrections_for_message("aa22")) == 1
    db.close()


def test_serving_context_filters_default_is_none_fail_closed() -> None:
    # F2 item 1: the DEFAULT is None (fail-closed) -- a ServingContext
    # construction that does not KNOW the filters withholds every 'added'
    # rather than crediting it to the human. The old ()-default was only a
    # suite-riding convenience (dead now the wiring is pinned); a revert to
    # () would make an omitted-filters construction fail OPEN.
    import dataclasses

    fld = {f.name: f for f in dataclasses.fields(ServingContext)}["filters"]
    assert fld.default is None


# -- the startup snapshot reader (the None-vs-attested-none wiring) -----------


def test_load_filters_snapshot_absent_is_unknown(tmp_path: Path) -> None:
    # a MISSING sidecar is UNKNOWN (None), NOT attested-none ([]) -- the
    # fail-open F1 removes at the snapshot level
    from gmail_labeler.cli import _load_filters_snapshot

    assert _load_filters_snapshot(tmp_path / "filters.json") is None


def test_load_filters_snapshot_malformed_is_unknown(tmp_path: Path) -> None:
    # a CORRUPT sidecar is UNKNOWN (None), never an attested-empty [] (a
    # broken file must not masquerade as "the operator has no filters")
    from gmail_labeler.cli import _load_filters_snapshot

    bad = tmp_path / "filters.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _load_filters_snapshot(bad) is None
    # a JSON list containing a non-dict element is also malformed -> None
    bad.write_text('["not-a-filter"]', encoding="utf-8")
    assert _load_filters_snapshot(bad) is None


def test_load_filters_snapshot_present_empty_is_attested_none(tmp_path: Path) -> None:
    # a present sidecar listing NO filters is the operator's ATTESTATION
    # ([]), distinct from the unknown (None) case above
    from gmail_labeler.cli import _load_filters_snapshot

    empty = tmp_path / "filters.json"
    empty.write_text("[]", encoding="utf-8")
    assert _load_filters_snapshot(empty) == []


def test_load_filters_snapshot_present_returns_the_list(tmp_path: Path) -> None:
    from gmail_labeler.cli import _load_filters_snapshot

    snap = tmp_path / "filters.json"
    snap.write_text('[{"criteria": {"from": "a@x.com"}}]', encoding="utf-8")
    assert _load_filters_snapshot(snap) == [{"criteria": {"from": "a@x.com"}}]


# -- the END-TO-END wiring: the loader's output reaches ServingContext.filters
# (the unit tests above prove the loader and _attribution each behave; only
# this proves _serve_startup CONNECTS them — a `... or []` typo at the single
# production wiring site would otherwise re-open the MAJOR with a green suite)


def _startup_with_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    command: str = "daemon",
    sidecar_text: str | None = None,
) -> _ServeStartup:
    """Run the real _serve_startup gates against the shared _prepare corpus
    (client/model seams monkeypatched, no Gmail), optionally planting a
    filters.json sidecar in the corpus root; return the startup bundle."""
    import gmail_labeler.cli as cli
    from gmail_labeler.cli import _serve_startup
    from test_shadow_cli import (
        _backlog_messages,
        _backlog_scores,
        _prepare,
        _shadow_cfg,
        _shadow_factory,
        _shadow_labels,
    )

    _prepare(tmp_path, monkeypatch)
    db = AppDB(tmp_path / "state.db")
    db.open()
    db.close()  # a real v4 state DB for the fail-fast gate
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    fake = FakeGmailClient(
        messages=_backlog_messages(),
        labels=_shadow_labels(),
        current_history_id=900,
    )
    monkeypatch.setattr(cli, "_daemon_client", lambda cfg_: fake)
    monkeypatch.setattr(
        cli, "_daemon_model_factory", _shadow_factory(_backlog_scores())
    )
    # _prepare/_seed_corpus plants a [] sidecar; the ABSENT case must remove
    # it (the present cases overwrite it) so the three states are genuinely
    # distinct at the wiring site
    sidecar = tmp_path / "corpus" / "filters.json"
    if sidecar_text is None:
        sidecar.unlink(missing_ok=True)
    else:
        sidecar.write_text(sidecar_text, encoding="utf-8")
    return _serve_startup(
        command=command,
        config=str(cfg),
        bands=str(tmp_path / "bands.json"),
        graduation=str(tmp_path / "graduation.json"),
        db=str(tmp_path / "state.db"),
    )


def test_serve_startup_no_sidecar_yields_unknown_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # the load-bearing wiring: NO sidecar -> ctx.filters is None (UNKNOWN),
    # NOT [] (which would fail every 'added' open to 'human'); and the daemon
    # warns at startup that attribution is UNKNOWN
    startup = _startup_with_sidecar(tmp_path, monkeypatch)  # no sidecar planted
    assert startup.ctx.filters is None
    err = capsys.readouterr().err
    assert "UNKNOWN" in err and "unverifiable" in err


def test_serve_startup_empty_sidecar_yields_attested_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # a present [] sidecar -> ctx.filters == [] (the operator's ATTESTATION
    # of no filters), distinct from the None/unknown case; no warning
    startup = _startup_with_sidecar(tmp_path, monkeypatch, sidecar_text="[]")
    assert startup.ctx.filters == []
    assert "UNKNOWN" not in capsys.readouterr().err


def test_serve_startup_malformed_sidecar_yields_unknown_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # a corrupt sidecar is UNKNOWN (None), never a masquerading [] — and the
    # daemon warns
    startup = _startup_with_sidecar(tmp_path, monkeypatch, sidecar_text="{broken")
    assert startup.ctx.filters is None
    assert "UNKNOWN" in capsys.readouterr().err


def test_serve_startup_apply_path_does_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # the UNKNOWN warning is daemon-scoped: apply shares _serve_startup but
    # never reads the correction arm, so it must NOT print the correction
    # attribution warning
    startup = _startup_with_sidecar(tmp_path, monkeypatch, command="apply")
    assert startup.ctx.filters is None  # the wiring is the same...
    assert "UNKNOWN" not in capsys.readouterr().err  # ...but apply stays quiet


def test_daemon_command_passes_command_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F2 item 13: the startup UNKNOWN warning is gated on command=="daemon".
    # The F1 wiring pins call _serve_startup DIRECTLY with command="daemon",
    # so mutating the daemon COMMAND's own literal to "serve" rode the whole
    # suite green (the reviewer's mut4). Pin the PRODUCTION call site: capture
    # the command kwarg the daemon command hands _serve_startup.
    import gmail_labeler.cli as cli
    from test_daemon_cli import _run_daemon
    from test_shadow_cli import _prepare

    _prepare(tmp_path, monkeypatch)
    captured: dict[str, str] = {}
    real = cli._serve_startup

    def spy(*, command: str, **kw: Any) -> Any:
        captured["command"] = command
        return real(command=command, **kw)

    monkeypatch.setattr(cli, "_serve_startup", spy)
    result, _fake = _run_daemon(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert captured.get("command") == "daemon"
