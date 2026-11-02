"""Correction DETECTION in the daemon tick (ticket #27).

A SEPARATE suite from the sealed #25 daemon tests (which stay verbatim —
this file drives the SAME tick with history records that carry
labelAdded/labelRemoved events, which the #25 fixtures never produce).
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
    # no snapshot -> no relevant filter claims the label -> the human added
    # it (positive training signal; F1 replaced the old bool's False here)
    assert corr[0].attribution == "human"
    db.close()


# -- the ignore-boundary, BOTH directions -------------------------------------


def test_removed_nonapplied_label_records_nothing(tmp_path: Path) -> None:
    # a labelRemoved of a label that is NOT this message's Application is
    # noise (the human removed something we never applied) -> NOTHING
    db = _db(tmp_path)
    _applied(db, "aa22", "Money/Bills", "L2")
    fake = _fake(_page(_record("aa22", removed=[("aa22", ["L1"])])))  # L1 not applied
    assert _tick(tmp_path, fake, db) == 0
    assert db.corrections_for_message("aa22") == ()
    assert db.get_history_cursor() == 901  # the delta still handled
    db.close()


def test_added_nonvocabulary_label_records_nothing(tmp_path: Path) -> None:
    # a labelAdded of a NON-graduated label (LX=NotGraduated) or a System
    # Label (INBOX) is not signal -> NOTHING recorded
    db = _db(tmp_path)
    _applied(db, "aa11", "Shipping", "L1")
    fake = _fake(_page(_record("aa11", added=[("aa11", ["LX"]), ("aa11", ["INBOX"])])))
    assert _tick(tmp_path, fake, db) == 0
    assert db.corrections_for_message("aa11") == ()
    db.close()


def test_added_label_the_daemon_already_wrote_records_nothing(tmp_path: Path) -> None:
    # a labelAdded for a label that IS this message's Application (the
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


# -- malformed label-event wire is a lying delta (hold, not crash) ------------


def test_malformed_label_event_holds_cursor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # a labelAdded that is not a list of objects is a lying wire -> the
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
