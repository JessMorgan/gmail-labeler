"""Label Sync event walk (ticket #73): seam = sync_labels(client, tmp store,
injected sleeper/clock). FakeGmailClient's deterministic history_records is
the event source; the resolved view is the read interface. Never below the
seam."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gmail_labeler.corpus_record import CorpusRecord, body_hash
from gmail_labeler.corpus_store import CorpusStore
from gmail_labeler.fake_gmail import (
    FakeGmailClient,
    FakeMessage,
    HistoryEvent,
    LabelsAdded,
    LabelsRemoved,
    MessageDeleted,
    history_records,
)
from gmail_labeler.gmail_client import HistoryPage
from gmail_labeler.label_sync import sync_labels
from gmail_labeler.sync_cursor import SyncCursor

NAMES = {"L1": "Shipping", "L2": "Money/Bills"}
NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def store_at(tmp_path: Path) -> CorpusStore:
    return CorpusStore(tmp_path / "corpus")


def _record(mid: str, labels: tuple[str, ...]) -> CorpusRecord:
    return CorpusRecord(
        id=mid,
        date=datetime(2026, 10, 30, 10, 0, tzinfo=UTC),
        subject=f"hi {mid}",
        sender="a@x.com",
        recipients=("me@x.com",),
        labels=labels,
        body_hash=body_hash(f"body {mid}"),
        source="api",
        thread_id=f"t{mid}",
    )


def _seed(store: CorpusStore, *records: CorpusRecord) -> None:
    for rec in records:
        store.put(rec, f"body {rec.id}")


def _clock() -> datetime:
    return NOW


def test_event_walk_refreshes_resident_ids_advances_cursor_once(
    tmp_path: Path,
) -> None:
    """Fake-driven end-to-end (AC #1): seed corpus + synthetic label-change
    events -> correct revisions appended, cursor advanced exactly once,
    counts right. messageRemoved-touched resident id: record kept, counted.
    Non-resident event id: never written, counted as new-match only."""
    store = store_at(tmp_path)
    _seed(
        store,
        _record("a1", ("Shipping",)),
        _record("b2", ("Money/Bills",)),
        _record("c3", ("Shipping",)),
    )
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        # mailbox truth AFTER the events (the fake serves records, never
        # applies them): a1 gained an unknown label id, b2 lost L2, c3 is
        # gone from the mailbox entirely
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9")),
            FakeMessage(id="b2", eml=b"x", label_ids=()),
        ],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records(
                    [
                        LabelsAdded("a1", ("L9",)),
                        LabelsRemoved("b2", ("L2",)),
                        LabelsAdded("d4", ("L1",)),
                        MessageDeleted("c3"),
                    ]
                ),
            )
        ],
    )
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.events == 4
    assert stats.stale == 3  # a1, b2, c3 are corpus-resident touches
    assert stats.revised == 2  # a1, b2; c3 stays (gone upstream)
    assert stats.deleted_kept == 1
    assert stats.new_match == 1
    assert stats.gains == {"id:L9": 1}  # unresolvable id keeps the id: convention
    assert stats.losses == {"Money/Bills": 1}
    assert not stats.failed

    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "id:L9")
    assert resolved["b2"].labels == ()
    assert resolved["c3"].labels == ("Shipping",)  # kept, not tombstoned
    assert "d4" not in resolved  # ingest stays the acquisition paths' alone
    # field parity (ADR-0007): only labels differ on a Revision
    assert resolved["a1"].body_hash == body_hash("body a1")
    assert resolved["a1"].date == datetime(2026, 10, 30, 10, 0, tzinfo=UTC)
    assert resolved["a1"].source == "api"
    assert resolved["a1"].thread_id == "ta1"

    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # advanced once
    assert cursor.last_sync == NOW

    # exact call sequence at the fake's call-log seam
    assert fake.calls == [
        "history:100:",
        "get_message_metadata:a1",
        "get_message_metadata:b2",
        "get_message_metadata:c3",
    ]


def _page(*events: HistoryEvent, history_id: int = 500) -> HistoryPage:
    return HistoryPage(
        history_id=history_id, next_page_token=None, records=history_records(events)
    )


def test_sync_path_is_structurally_body_free(tmp_path: Path) -> None:
    """Invariant #5 as an executable check (AC #2): the sync call log
    contains no raw get and no message-list call — only history, metadata
    gets, and (for a birth) profile."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1",))],
        history_pages=[_page(LabelsAdded("a1", ("L1",)))],
    )
    sync_labels(fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock)
    assert not any(c.startswith("get_message:") for c in fake.calls)
    assert not any(c.startswith("list_messages") for c in fake.calls)


def test_expired_cursor_raises_resync_not_swallowed(tmp_path: Path) -> None:
    """The _retrying swallow fix (manager note 3): a GmailResyncError (kind
    STOP) must surface typed — never a logged warning + silent success."""
    from gmail_labeler.gmail_errors import GmailResyncError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=900, last_sync=NOW))
    fake = FakeGmailClient(current_history_id=500)  # cursor ahead = outside window
    with pytest.raises(GmailResyncError):
        sync_labels(
            fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
        )
    assert store.read_cursor() is not None
    assert store.read_cursor() == SyncCursor(history_id=900, last_sync=NOW)


def test_bare_permanent_404_on_history_converts_to_resync(tmp_path: Path) -> None:
    """Both-spine taxonomy (manager note 2): a spine that raises the bare
    GmailPermanentError(status=404) — not the adapter's GmailResyncError —
    still reads as resync."""
    from gmail_labeler.gmail_errors import GmailPermanentError, GmailResyncError

    class BareFourOhFour(FakeGmailClient):
        def history(self, start_history_id: int, page_token: str | None = None):  # type: ignore[no-untyped-def]
            self.calls.append(f"history:{start_history_id}:{page_token or ''}")
            raise GmailPermanentError("history 404 (unconverted spine)", status=404)

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    with pytest.raises(GmailResyncError):
        sync_labels(
            BareFourOhFour(),
            store,
            id_to_name=NAMES,
            sleeper=lambda _s: None,
            clock=_clock,
        )


def test_auth_error_propagates_typed(tmp_path: Path) -> None:
    from gmail_labeler.gmail_errors import GmailAuthError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(history_pages=[_page(LabelsAdded("a1", ("L1",)))])
    fake.fail("history", GmailAuthError("revoked"))
    with pytest.raises(GmailAuthError):
        sync_labels(
            fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
        )


def test_transient_history_retries_with_pacing(tmp_path: Path) -> None:
    from gmail_labeler.gmail_errors import GmailTransientError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        history_pages=[_page(LabelsAdded("a1", ("L9",)))],
    )
    fake.fail("history", GmailTransientError("blip"))
    slept: list[float] = []
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=slept.append, clock=_clock
    )
    assert stats.retried == 1 and slept == [1.0]  # _BASE_BACKOFF_S * 2**0
    assert stats.revised == 1 and not stats.failed


def test_page_level_permanent_stops_run_without_advancing_cursor(
    tmp_path: Path,
) -> None:
    from gmail_labeler.gmail_errors import GmailPermanentError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(history_pages=[_page(LabelsAdded("a1", ("L1",)))])
    fake.fail("history", GmailPermanentError("bad token", status=400))
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.failed == [("<page>", "history stopped (permanent or exhausted)")]
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)


def test_multipage_walk_exhausts_and_lands_on_final_id(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    _seed(store, _record("a1", ()), _record("b2", ()))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1",)),
            FakeMessage(id="b2", eml=b"x", label_ids=("L2",)),
        ],
        history_pages=[
            _page(LabelsAdded("a1", ("L1",)), history_id=500),
            _page(LabelsAdded("b2", ("L2",)), history_id=600),
        ],
    )
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.events == 2 and stats.revised == 2
    assert fake.calls[:2] == ["history:100:", "history:100:1"]  # token chain
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 600  # the FINAL page's id


def test_event_ids_are_case_insensitive_against_the_corpus(
    tmp_path: Path,
) -> None:
    """Uppercase twins defeat the fold — ids are lowercased at the gate
    (caller-side duty, note 9a)."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=())],
        history_pages=[_page(LabelsRemoved("A1", ("L1",)))],  # uppercase on the wire
    )
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.revised == 1 and stats.new_match == 0
    assert fake.calls == ["history:100:", "get_message_metadata:a1"]
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ()
    assert stats.losses == {"Shipping": 1}


def test_failed_refresh_keeps_cursor_behind_the_delta(tmp_path: Path) -> None:
    """Invariant #3: the cursor never runs ahead of durability — a failed
    member of the delta keeps the whole window replayable."""
    from gmail_labeler.gmail_errors import GmailPermanentError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)), _record("b2", ()))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9")),
            FakeMessage(id="b2", eml=b"x", label_ids=("L2",)),
        ],
        history_pages=[_page(LabelsAdded("a1", ("L9",)), LabelsAdded("b2", ("L2",)))],
    )
    fake.fail("get_message_metadata", GmailPermanentError("boom", status=400))
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.failed == [("a1", "metadata fetch failed (permanent/retries)")]
    assert stats.revised == 1  # b2's revision is durable, a1's is not
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)


def test_crash_retry_absorbs_duplicate_revisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #3: a sync killed after the revisions are durable but before the
    cursor write is safe to re-run — duplicate revisions fold, the cursor
    advances exactly once, the resolved view is right."""
    from gmail_labeler.corpus_store import CorpusError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ()), _record("b2", ()))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))

    def fake_mailbox() -> FakeGmailClient:
        return FakeGmailClient(
            messages=[
                FakeMessage(id="a1", eml=b"x", label_ids=("L1",)),
                FakeMessage(id="b2", eml=b"x", label_ids=("L2",)),
            ],
            history_pages=[
                _page(LabelsAdded("a1", ("L1",)), LabelsAdded("b2", ("L2",)))
            ],
        )

    def crash(*_a: object) -> None:
        raise CorpusError("simulated kill between durable revisions and cursor write")

    monkeypatch.setattr(store, "write_cursor", crash)
    with pytest.raises(CorpusError):
        sync_labels(
            fake_mailbox(),
            store,
            id_to_name=NAMES,
            sleeper=lambda _s: None,
            clock=_clock,
        )
    # revisions durable, cursor NOT advanced (durability ordering proof)
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)
    raw_lines = (store.root / "records.jsonl").read_text().count("\n")
    assert raw_lines == 4  # 2 seeds + 2 durable revisions

    # re-run with a FRESH store instance (new process) and fresh fake: the
    # window replays, the last-wins fold absorbs the duplicates
    store2 = store_at(tmp_path)
    stats = sync_labels(
        fake_mailbox(), store2, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.revised == 2 and not stats.failed
    resolved = {r.id: r for r in store2.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping",)
    assert resolved["b2"].labels == ("Money/Bills",)
    cursor = store2.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # advanced exactly once


def test_dry_run_plans_without_touching_anything(
    tmp_path: Path,
) -> None:
    """AC #4: identical report shape (same stats), zero record writes,
    cursor byte-identical."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)), _record("b2", ("Money/Bills",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    records_bytes = (store.root / "records.jsonl").read_bytes()
    cursor_bytes = (store.root / "sync-cursor.json").read_bytes()
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9")),
            FakeMessage(id="b2", eml=b"x", label_ids=()),
        ],
        history_pages=[_page(LabelsAdded("a1", ("L9",)), LabelsRemoved("b2", ("L2",)))],
    )
    stats = sync_labels(
        fake,
        store,
        id_to_name=NAMES,
        sleeper=lambda _s: None,
        clock=_clock,
        dry_run=True,
    )
    assert stats.revised == 2
    assert stats.gains == {"id:L9": 1} and stats.losses == {"Money/Bills": 1}
    assert not stats.failed
    assert (store.root / "records.jsonl").read_bytes() == records_bytes
    assert (store.root / "sync-cursor.json").read_bytes() == cursor_bytes


def test_cursorless_corpus_births_and_applies_nothing(
    tmp_path: Path,
) -> None:
    """ADR-0008 consequence 1: first sync on a cursorless corpus births
    from one profile call, reports the provenance, applies nothing — no
    event walk."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    fake = FakeGmailClient(current_history_id=777)
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.birthed and stats.revised == 0 and not stats.failed
    assert fake.calls == ["get_profile"]  # no history walk, no metadata gets
    assert store.read_cursor() == SyncCursor(history_id=777, last_sync=NOW)
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping",)  # untouched


def test_dry_run_on_cursorless_corpus_does_not_birth(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    fake = FakeGmailClient(current_history_id=777)
    stats = sync_labels(
        fake,
        store,
        id_to_name=NAMES,
        sleeper=lambda _s: None,
        clock=_clock,
        dry_run=True,
    )
    assert stats.birthed  # the plan says it would birth
    assert not (store.root / "sync-cursor.json").exists()


def test_unusable_seed_on_cursorless_corpus_fails_loudly(tmp_path: Path) -> None:
    from gmail_labeler.gmail_errors import GmailPermanentError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    fake = FakeGmailClient(current_history_id=777)
    fake.fail("get_profile", GmailPermanentError("profile down", status=400))
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.failed == [("<cursor>", "no usable historyId seed (profile call)")]
    assert not (store.root / "sync-cursor.json").exists()


def test_max_attempts_below_one_is_rejected(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    with pytest.raises(ValueError, match="max_attempts"):
        sync_labels(
            FakeGmailClient(),
            store,
            id_to_name=NAMES,
            max_attempts=0,
            sleeper=lambda _s: None,
            clock=_clock,
        )


def test_gone_confirmed_without_delete_event_counts_as_deleted_kept(
    tmp_path: Path,
) -> None:
    """The other half of the deleted_kept union: an id a label event names
    whose metadata get 404s (vanished upstream with no messagesDeleted in
    the window) — record kept, counted, NOT a failure."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[],  # a1 is gone from the mailbox: the metadata get 404s
        history_pages=[_page(LabelsAdded("a1", ("L1",)))],
    )
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.deleted_kept == 1 and stats.revised == 0 and not stats.failed
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping",)  # kept, not tombstoned
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # delta fully handled


def test_metadata_transient_retries_then_revises(tmp_path: Path) -> None:
    """The metadata spine rides the same taxonomy: one transient blip is a
    retry, not a failed id."""
    from gmail_labeler.gmail_errors import GmailTransientError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        history_pages=[_page(LabelsAdded("a1", ("L9",)))],
    )
    fake.fail("get_message_metadata", GmailTransientError("blip"))
    slept: list[float] = []
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=slept.append, clock=_clock
    )
    assert stats.retried == 1 and slept == [1.0]
    assert stats.revised == 1 and not stats.failed


def test_metadata_retry_exhaustion_fails_and_holds_cursor(tmp_path: Path) -> None:
    """Exhausted metadata retries land in failed (never silently skipped)
    and keep the cursor behind the delta."""
    from gmail_labeler.gmail_errors import GmailTransientError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        history_pages=[_page(LabelsAdded("a1", ("L9",)))],
    )
    fake.fail("get_message_metadata", GmailTransientError("down"))
    fake.fail("get_message_metadata", GmailTransientError("still down"))
    stats = sync_labels(
        fake,
        store,
        id_to_name=NAMES,
        max_attempts=2,
        sleeper=lambda _s: None,
        clock=_clock,
    )
    assert stats.retried == 2
    assert stats.failed == [("a1", "metadata fetch failed (permanent/retries)")]
    assert stats.revised == 0
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)
