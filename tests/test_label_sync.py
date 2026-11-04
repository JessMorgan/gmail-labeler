"""Label Sync event walk (ticket #73): seam = sync_labels(client, tmp store,
injected sleeper/clock). FakeGmailClient's deterministic history_records is
the event source; the resolved view is the read interface. Never below the
seam.

--full-resync (ticket #74) joins the same module and the same test seam:
full_resync(client, tmp store, injected sleeper/clock/progress) — the
estimate, both strategies, and the seed-before-scan cursor discipline are
all pinned at that boundary plus the fake's call log."""

from __future__ import annotations

import dataclasses
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
from gmail_labeler.gmail_client import HistoryPage, LabelInfo
from gmail_labeler.label_sync import full_resync, sync_labels
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


def _assert_no_raw_fetches(fake: FakeGmailClient) -> None:
    """Invariant #5 call-log postcondition, asserted by EVERY seam test in
    this file (not just the named structural test): the log holds no raw
    get and no message-list call — a future raw fetch fails the suite in
    ANY sync scenario, whichever test catches it first."""
    offenders = [
        c
        for c in fake.calls
        if c.startswith("get_message:") or c.startswith("list_messages")
    ]
    assert not offenders, offenders


def test_event_walk_refreshes_resident_ids_advances_cursor_once(
    tmp_path: Path,
) -> None:
    """Fake-driven end-to-end (AC #1): seed corpus + synthetic label-change
    events -> correct revisions appended, cursor advanced exactly once,
    counts right. messageRemoved-touched resident id: record kept, counted.
    Non-resident event id: never written, counted as an Unseen Id only."""
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
    _assert_no_raw_fetches(fake)
    assert stats.events == 4
    assert stats.stale == 3  # a1, b2, c3 are corpus-resident touches
    assert stats.revised == 2  # a1, b2; c3 stays (gone upstream)
    assert stats.deleted_kept == 1
    assert stats.unseen == 1
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
    gets, and (for a birth) profile. The named taxonomy proof; the shared
    _assert_no_raw_fetches helper now runs in every seam test too."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1",))],
        history_pages=[_page(LabelsAdded("a1", ("L1",)))],
    )
    sync_labels(fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock)
    _assert_no_raw_fetches(fake)


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
    _assert_no_raw_fetches(fake)
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
    fake = BareFourOhFour()
    with pytest.raises(GmailResyncError):
        sync_labels(
            fake,
            store,
            id_to_name=NAMES,
            sleeper=lambda _s: None,
            clock=_clock,
        )
    _assert_no_raw_fetches(fake)


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
    _assert_no_raw_fetches(fake)


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
    _assert_no_raw_fetches(fake)
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
    _assert_no_raw_fetches(fake)
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
    _assert_no_raw_fetches(fake)
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
    _assert_no_raw_fetches(fake)
    assert stats.revised == 1 and stats.unseen == 0
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
    _assert_no_raw_fetches(fake)
    assert stats.failed == [("a1", "metadata fetch failed (permanent/retries)")]
    assert stats.failed_ids == ["a1"]  # sentinel-safe accessor (gmail_import parity)
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
    fake1 = fake_mailbox()
    with pytest.raises(CorpusError):
        sync_labels(
            fake1,
            store,
            id_to_name=NAMES,
            sleeper=lambda _s: None,
            clock=_clock,
        )
    _assert_no_raw_fetches(fake1)  # the crashed run's call log
    # revisions durable, cursor NOT advanced (durability ordering proof)
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)
    raw_lines = (store.root / "records.jsonl").read_text().count("\n")
    assert raw_lines == 4  # 2 seeds + 2 durable revisions

    # re-run with a FRESH store instance (new process) and fresh fake: the
    # window replays, the last-wins fold absorbs the duplicates
    store2 = store_at(tmp_path)
    fake2 = fake_mailbox()
    stats = sync_labels(
        fake2, store2, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    _assert_no_raw_fetches(fake2)  # the re-run's call log
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
    _assert_no_raw_fetches(fake)
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
    _assert_no_raw_fetches(fake)
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
    _assert_no_raw_fetches(fake)
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
    _assert_no_raw_fetches(fake)
    assert stats.failed == [("<cursor>", "no usable historyId seed (profile call)")]
    assert not (store.root / "sync-cursor.json").exists()


def test_max_attempts_below_one_is_rejected(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    fake = FakeGmailClient()
    with pytest.raises(ValueError, match="max_attempts"):
        sync_labels(
            fake,
            store,
            id_to_name=NAMES,
            max_attempts=0,
            sleeper=lambda _s: None,
            clock=_clock,
        )
    _assert_no_raw_fetches(fake)  # rejected before any call at all


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
    _assert_no_raw_fetches(fake)
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
    _assert_no_raw_fetches(fake)
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
    _assert_no_raw_fetches(fake)
    assert stats.retried == 2
    assert stats.failed == [("a1", "metadata fetch failed (permanent/retries)")]
    assert stats.revised == 0
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)


def test_final_page_without_history_id_holds_the_cursor(tmp_path: Path) -> None:
    """A delta whose final page carries no usable historyId cannot honestly
    claim the window is consumed: the old cursor stays, the gap is a loud
    failure (a replay is harmless; a guess is the silent-gap class)."""

    class NoEcho(FakeGmailClient):
        def history(self, start_history_id: int, page_token: str | None = None):  # type: ignore[no-untyped-def]
            page = super().history(start_history_id, page_token)
            # the wire-faithful field is present; this spine drops it
            return HistoryPage(
                history_id=None,
                next_page_token=page.next_page_token,
                records=page.records,
            )

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = NoEcho(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        history_pages=[_page(LabelsAdded("a1", ("L9",)))],
    )
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    _assert_no_raw_fetches(fake)
    assert stats.revised == 1  # the delta IS durable…
    assert stats.failed == [
        ("<cursor>", "no usable historyId on the final history page")
    ]
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)  # …unmarked


def test_malformed_history_entries_name_no_messages(tmp_path: Path) -> None:
    """The tolerant parser's defensive arms: garbage entries in typed fields
    contribute no touches (and no crash) — the walk never guesses at
    malformed wire shapes."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=(
                    {
                        "id": "1",
                        "labelsAdded": [
                            "not-an-object",  # entry garbage
                            {"message": "also-not-an-object"},  # nested garbage
                            {"message": {"noId": True}},  # no id
                        ],
                    },
                ),
            )
        ],
    )
    stats = sync_labels(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    _assert_no_raw_fetches(fake)
    assert stats.events == 1 and stats.stale == 0 and stats.unseen == 0
    assert fake.calls == ["history:100:"]  # nothing was refreshed
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # empty delta consumed


# ---- --full-resync (ticket #74): pure core ---------------------------------


def _big_mailbox_labels() -> list[LabelInfo]:
    """Two user labels with big totals: the scan's enumeration floor dwarfs
    a walk over any small corpus — the estimator's small-corpus case."""
    return [
        LabelInfo("L1", "Shipping", "user", messages_total=50_000),
        LabelInfo("L2", "Money/Bills", "user", messages_total=30_000),
    ]


def test_estimator_picks_corpus_walk_for_small_corpus_big_mailbox(
    tmp_path: Path,
) -> None:
    """AC #2 (one direction): 2 corpus messages (walk = 2x20 = 40 units) vs
    an 800-unit scan enumeration floor -> the walk runs, and its call log
    shows metadata gets for EVERY corpus id and no inventory list calls."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)), _record("b2", ()))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1",)),
            FakeMessage(id="b2", eml=b"x", label_ids=()),
        ],
        labels=_big_mailbox_labels(),
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.strategy == "walk"
    assert stats.corpus_messages == 2
    assert stats.walk_units == 40 and stats.scan_list_units == 800
    assert stats.flagged == 2  # the walk flags every corpus id
    assert stats.revised == 0 and stats.unchanged == 2  # nothing actually moved
    assert not stats.failed
    assert fake.calls == [
        "get_profile",  # seed FIRST (note 9): before any enumeration spend
        "list_labels",  # ONE labels.list for the estimate
        "get_message_metadata:a1",
        "get_message_metadata:b2",
    ]
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # the up-front seed
    assert cursor.last_sync == NOW


def test_walk_revises_only_drifted_ids_and_keeps_field_parity(
    tmp_path: Path,
) -> None:
    """The walk diffs before appending: a drifted id gets an ADR-0007
    Revision (field parity intact), an identical one gets NOTHING."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)), _record("b2", ("Money/Bills",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9")),  # gained L9
            FakeMessage(id="b2", eml=b"x", label_ids=("L2",)),  # unchanged
        ],
        labels=_big_mailbox_labels(),
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.strategy == "walk"
    assert stats.revised == 1 and stats.unchanged == 1
    assert stats.gains == {"id:L9": 1} and stats.losses == {}
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "id:L9")
    raw_lines = (store.root / "records.jsonl").read_text().count("\n")
    assert raw_lines == 3  # 2 seeds + exactly one revision line
    assert resolved["a1"].body_hash  # field parity: replace()-built revision


def test_resync_path_is_structurally_body_free_both_strategies(
    tmp_path: Path,
) -> None:
    """Invariant #5 re-pinned for #74 (AC #4): no raw get and — for the walk
    — no list call at all; for the scan, only labelIds inventory calls."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        labels=_big_mailbox_labels(),
        current_history_id=500,
    )
    walk_stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert walk_stats.strategy == "walk"
    assert not any(c.startswith("get_message:") for c in fake.calls)

    store2 = store_at(tmp_path / "second")
    _seed(store2, _record("a1", ("Shipping",)))
    store2.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake2 = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        labels=[LabelInfo("L1", "Shipping", "user", messages_total=3)],
        current_history_id=500,
    )
    scan_stats = full_resync(
        fake2, store2, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert scan_stats.strategy == "scan"
    assert not any(c.startswith("get_message:") for c in fake2.calls)
    # every list call is a labelIds inventory call, never a query sweep
    assert all(":" + "L1" in c for c in fake2.calls if c.startswith("list_messages"))


def test_seed_persisted_only_after_revisions_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Note 9 / invariant #3: the seed rides the SAME after-durability write
    as the event walk's advance — crash between durable revisions and the
    cursor write, and the ledger stays behind the delta."""
    from gmail_labeler.corpus_store import CorpusError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ()))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1",))],
        labels=_big_mailbox_labels(),
        current_history_id=500,
    )

    def crash(*_a: object) -> None:
        raise CorpusError("simulated kill between durable revisions and cursor")

    monkeypatch.setattr(store, "write_cursor", crash)
    with pytest.raises(CorpusError):
        full_resync(
            fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
        )
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping",)  # the revision IS durable


class _AdvancingFake(FakeGmailClient):
    """Every metadata get stands in for one mailbox event landing mid-scan:
    the current historyId climbs, so a last-page-wins-style end-of-run read
    (the bug #73's handoff warns the resync must not inherit) would print a
    ledger value strictly ahead of the up-front seed."""

    def get_message_metadata(self, message_id: str):  # type: ignore[no-untyped-def]
        meta = super().get_message_metadata(message_id)
        self.current_history_id += 1
        return meta


def test_seed_captured_up_front_is_the_ledger_value_mid_run_change_survives(
    tmp_path: Path,
) -> None:
    """Note 9 (binding): seed = ONE profile call BEFORE the scan; the value
    persisted is that seed, never anything read later — so a change made
    between seed capture and scan end (id > seed) lands in the NEXT event
    sync's window instead of being skipped."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = _AdvancingFake(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        labels=_big_mailbox_labels(),
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.revised == 1
    assert fake.calls.count("get_profile") == 1  # never re-read for the ledger
    assert fake.current_history_id == 501  # the mailbox moved mid-scan
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # the SEED, not 501

    # the change made after seed capture (event 501) is the next sync's job:
    fake2 = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9", "L2"))],
        labels=_big_mailbox_labels(),
        history_pages=[_page(LabelsAdded("a1", ("L2",)), history_id=501)],
        current_history_id=501,
    )
    stats2 = sync_labels(
        fake2, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats2.revised == 1 and not stats2.failed
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "id:L9", "Money/Bills")


def test_full_resync_unusable_seed_and_bad_attempts_fail_loudly(
    tmp_path: Path,
) -> None:
    from gmail_labeler.gmail_errors import GmailPermanentError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    fake = FakeGmailClient(
        labels=_big_mailbox_labels(),
        current_history_id=777,
    )
    fake.fail("get_profile", GmailPermanentError("profile down", status=400))
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.failed == [("<cursor>", "no usable historyId seed (profile call)")]
    assert not (store.root / "sync-cursor.json").exists()
    assert fake.calls == ["get_profile"]  # nothing else was spent

    with pytest.raises(ValueError, match="max_attempts"):
        full_resync(
            FakeGmailClient(),
            store,
            id_to_name=NAMES,
            max_attempts=0,
            sleeper=lambda _s: None,
            clock=_clock,
        )


def _small_mailbox_labels() -> list[LabelInfo]:
    """Tiny totals: the scan's enumeration floor (2 x 5 units) is far below
    any walk — the estimator's other direction (AC #2)."""
    return [
        LabelInfo("L1", "Shipping", "user", messages_total=4),
        LabelInfo("L2", "Money/Bills", "user", messages_total=2),
    ]


def test_estimator_picks_scan_and_fetches_metadata_only_for_flagged_ids(
    tmp_path: Path,
) -> None:
    """AC #1/#2/#4 (scan direction): 4 corpus messages vs a 10-unit scan
    floor -> scan runs; membership unchanged -> ZERO metadata gets, yet
    the ledger still advances (the reconciliation completed, nothing moved)."""
    store = store_at(tmp_path)
    _seed(
        store,
        _record("a1", ("Shipping",)),
        _record("b2", ("Shipping",)),
        _record("c3", ()),
        _record("d4", ()),
    )
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1",)),
            FakeMessage(id="b2", eml=b"x", label_ids=("L1",)),
        ],
        labels=_small_mailbox_labels(),
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.strategy == "scan"
    assert stats.scan_list_units == 10 and stats.walk_units == 80
    assert stats.flagged == 0 and stats.revised == 0
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500
    assert fake.calls == [
        "get_profile",
        "list_labels",
        "list_messages:::L1",
        "list_messages:::L2",
    ]  # not one metadata get over a quiet mailbox


def test_scan_flags_gained_and_lost_membership_including_unlabeled_mail(
    tmp_path: Path,
) -> None:
    """AC #3 (completeness, gained half): a1 gains L2, previously-unlabeled
    c3 gains L2 — both flagged; only flagged ids receive metadata gets; the
    revisions carry the gained names."""
    store = store_at(tmp_path)
    _seed(
        store,
        _record("a1", ("Shipping",)),
        _record("b2", ("Shipping",)),
        _record("c3", ()),  # previously unlabeled mail
    )
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L2")),
            FakeMessage(id="b2", eml=b"x", label_ids=("L1",)),
            FakeMessage(id="c3", eml=b"x", label_ids=("L2",)),
        ],
        labels=_small_mailbox_labels(),
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.strategy == "scan"
    assert stats.flagged == 2 and stats.revised == 2
    assert stats.gains == {"Money/Bills": 2} and stats.losses == {}
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "Money/Bills")
    assert resolved["c3"].labels == ("Money/Bills",)
    assert fake.calls == [
        "get_profile",
        "list_labels",
        "list_messages:::L1",
        "list_messages:::L2",
        "get_message_metadata:a1",  # ONLY the flagged two — b2 untouched
        "get_message_metadata:c3",
    ]


def test_scan_flags_lost_membership_and_reports_unseen_without_writing(
    tmp_path: Path,
) -> None:
    """AC #3 (completeness, lost half) + ingest discipline: b2's L2 is gone
    upstream -> flagged via the corpus-side set difference (the mailbox
    inventory never names it); e5 is a gained member NOT in the corpus ->
    counted as Unseen, never written."""
    store = store_at(tmp_path)
    _seed(store, _record("b2", ("Money/Bills",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="b2", eml=b"x", label_ids=()),  # L2 removed
            FakeMessage(id="e5", eml=b"x", label_ids=("L2",)),  # non-resident
        ],
        labels=_small_mailbox_labels(),
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.flagged == 1 and stats.revised == 1
    assert stats.losses == {"Money/Bills": 1} and stats.gains == {}
    assert stats.unseen == 1  # e5: report only
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["b2"].labels == ()
    assert "e5" not in resolved
    assert fake.calls[-1] == "get_message_metadata:b2"


def test_scan_reconciles_vanished_labels_and_keeps_id_convention(
    tmp_path: Path,
) -> None:
    """AC #3 (vanished labels): the corpus knows "Old/News" and the raw id
    L9 under the id: convention; neither label survives in the mailbox, so
    no inventory can name them — their corpus members are metadata-fetched
    directly. The get proves L9 still rides a1 while Old/News is gone: the
    id: convention is KEPT where unresolvable, the vanished name drops."""
    store = store_at(tmp_path)
    _seed(
        store,
        _record("a1", ("Shipping", "Old/News")),
        _record("b2", ("id:L9", "Shipping")),
    )
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9")),
            FakeMessage(id="b2", eml=b"x", label_ids=("L1", "L9")),
        ],
        labels=_small_mailbox_labels(),  # no Old/News, no L9 in the map
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.strategy == "scan"
    assert stats.flagged == 2 and stats.revised == 1  # b2's set is unchanged
    assert stats.unchanged == 1
    assert stats.losses == {"Old/News": 1} and stats.gains == {"id:L9": 1}
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "id:L9")  # name -> id: form
    assert resolved["b2"].labels == ("id:L9", "Shipping")  # untouched: kept
    assert fake.calls == [
        "get_profile",
        "list_labels",
        "list_messages:::L1",
        "list_messages:::L2",
        "get_message_metadata:a1",
        "get_message_metadata:b2",
    ]


def test_scan_ignores_system_label_only_churn(tmp_path: Path) -> None:
    """AC #3 (negative half): INBOX/UNREAD movement touches no enumerated
    user-label set — not one metadata get, not one revision, ledger still
    advances."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("INBOX", "UNREAD", "L1"))],
        labels=_small_mailbox_labels(),
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.flagged == 0 and stats.revised == 0
    assert fake.calls == [
        "get_profile",
        "list_labels",
        "list_messages:::L1",
        "list_messages:::L2",
    ]
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500


def test_scan_pages_multi_page_labels(tmp_path: Path) -> None:
    """Enumeration pages to exhaustion: the fake's small page_size proves
    the second page's members reach the diff."""
    store = store_at(tmp_path)
    _seed(store, _record("a4", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id=f"a{i}", eml=b"x", label_ids=("L1",)) for i in range(5)
        ],
        labels=_small_mailbox_labels(),
        page_size=2,
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    # a0..a3 gained L1 upstream but are NOT in the corpus; a4 IS in the
    # corpus with the right label. The third page's member (a4) is only
    # seen if paging is honored — nothing is flagged, and the non-resident
    # four are Unseen, report-only.
    assert stats.flagged == 0 and stats.unseen == 4
    inv_calls = [c for c in fake.calls if c.startswith("list_messages")]
    assert inv_calls == [
        "list_messages:::L1",
        "list_messages::2:L1",
        "list_messages::4:L1",  # the second page's members reach the diff
        "list_messages:::L2",
    ]


def test_resync_gone_upstream_message_keeps_record_and_advances(
    tmp_path: Path,
) -> None:
    """A flagged id whose metadata get 404s (deleted upstream): record
    kept, counted deleted_kept, NOT a failure — the ledger advances."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[],  # a1 gone: the get 404s
        labels=_small_mailbox_labels(),
        current_history_id=500,
    )
    # a1's Shipping membership: L1 inventory is empty -> a1 flagged (lost)
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.deleted_kept == 1 and stats.revised == 0 and not stats.failed
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping",)  # kept, not tombstoned
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500


def test_resync_failed_fetch_holds_ledger_and_replay_absorbs(
    tmp_path: Path,
) -> None:
    """Invariant #3 + mid-run failure replay (AC #5): a permanent metadata
    failure keeps the ledger behind; the durable revision stays, the re-run
    redoes the reconciliation, the fold absorbs the replay, and only then
    does the ledger advance."""
    from gmail_labeler.gmail_errors import GmailPermanentError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ()), _record("b2", ()))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))

    def mailbox() -> FakeGmailClient:
        return FakeGmailClient(
            messages=[
                FakeMessage(id="a1", eml=b"x", label_ids=("L1",)),
                FakeMessage(id="b2", eml=b"x", label_ids=("L2",)),
            ],
            labels=_small_mailbox_labels(),
            current_history_id=500,
        )

    fake = mailbox()
    fake.fail("get_message_metadata", GmailPermanentError("boom", status=400))
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.failed == [("a1", "metadata fetch failed (permanent/retries)")]
    assert stats.revised == 1  # b2's revision IS durable
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)

    # replay with a fresh mailbox: a1 reconciles at last; b2's durable
    # revision from run 1 means its membership now agrees — the fold
    # absorbed the replay, the ledger advances only over a clean delta
    stats2 = full_resync(
        mailbox(), store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats2.revised == 1 and not stats2.failed  # a1; b2 already agrees
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping",)
    assert resolved["b2"].labels == ("Money/Bills",)


def test_resync_dry_run_writes_nothing_cursor_byte_identical(
    tmp_path: Path,
) -> None:
    store = store_at(tmp_path)
    _seed(store, _record("a1", ()))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    records_bytes = (store.root / "records.jsonl").read_bytes()
    cursor_bytes = (store.root / "sync-cursor.json").read_bytes()
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1",))],
        labels=_small_mailbox_labels(),
        current_history_id=500,
    )
    stats = full_resync(
        fake,
        store,
        id_to_name=NAMES,
        sleeper=lambda _s: None,
        clock=_clock,
        dry_run=True,
    )
    assert stats.revised == 1  # the plan says what WOULD be written
    assert (store.root / "records.jsonl").read_bytes() == records_bytes
    assert (store.root / "sync-cursor.json").read_bytes() == cursor_bytes


def test_resync_gyb_corpus_births_ledger_when_absent_end_to_end(
    tmp_path: Path,
) -> None:
    """AC #7: a GYB-sourced corpus (no cursor ever existed) reconciles
    end-to-end against the fake and births the ledger with the up-front
    seed — the same before/after ordering as an advance (note 9)."""
    store = store_at(tmp_path)
    rec = _record("a1", ("Shipping",))
    store.put(dataclasses.replace(rec, source="gyb"), f"body {rec.id}")
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=())],
        labels=_small_mailbox_labels(),
        current_history_id=777,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.birthed and not stats.failed
    assert stats.revised == 1 and stats.losses == {"Shipping": 1}
    assert store.read_cursor() == SyncCursor(history_id=777, last_sync=NOW)
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ()
    assert resolved["a1"].source == "gyb"  # field parity across the revision


def test_resync_dry_run_on_cursorless_corpus_plans_birth_without_writing(
    tmp_path: Path,
) -> None:
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=())],
        labels=_small_mailbox_labels(),
        current_history_id=777,
    )
    stats = full_resync(
        fake,
        store,
        id_to_name=NAMES,
        sleeper=lambda _s: None,
        clock=_clock,
        dry_run=True,
    )
    assert stats.birthed  # the plan says it would birth
    assert not (store.root / "sync-cursor.json").exists()


def test_resync_progress_prints_estimate_before_spend_and_flood_story(
    tmp_path: Path,
) -> None:
    """AC #1 (printed before enumeration spend) + AC #6 (flag-rate + the
    honest worst-case story): the first progress line lands while the call
    log holds ONLY the seed + estimate reads; a corpus-wide flag triggers
    the reload-comparison NOTE."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)), _record("b2", ("Money/Bills",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=()),  # drift everywhere
            FakeMessage(id="b2", eml=b"x", label_ids=("L1",)),
        ],
        labels=_small_mailbox_labels(),
        current_history_id=500,
    )
    seen: list[tuple[str, list[str]]] = []
    stats = full_resync(
        fake,
        store,
        id_to_name=NAMES,
        sleeper=lambda _s: None,
        clock=_clock,
        progress=lambda line: seen.append((line, list(fake.calls))),
    )
    assert stats.flagged == 2 and stats.flag_rate == 1.0
    estimate_line, spend_at_estimate = seen[0]
    assert "estimate" in estimate_line and "strategy" in seen[1][0]
    assert spend_at_estimate == ["get_profile", "list_labels"]  # BEFORE scan
    joined = "\n".join(line for line, _ in seen)
    assert "flag-rate 100%" in joined
    assert "from-scratch reload" in joined and "60 units/msg" in joined
    # the flag-rate line itself precedes the metadata spend:
    flag_line = next((calls for line, calls in seen if "flag-rate" in line))
    assert flag_line == [
        "get_profile",
        "list_labels",
        "list_messages:::L1",
        "list_messages:::L2",
    ]


def test_resync_enumeration_failure_stops_before_any_spend(
    tmp_path: Path,
) -> None:
    """A permanently-failing inventory page leaves the flagged set
    incomplete: the run stops with the sentinel failure, ZERO metadata
    gets, ledger behind (replay redoes the scan harmlessly)."""
    from gmail_labeler.gmail_errors import GmailPermanentError

    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1",))],
        labels=_small_mailbox_labels(),
        current_history_id=500,
    )
    fake.fail("list_messages", GmailPermanentError("inventory down", status=400))
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.failed == [
        (
            "<label:L1>",
            "inventory enumeration stopped (permanent or exhausted)",
        )
    ]
    assert not any(c.startswith("get_message_metadata") for c in fake.calls)
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)


def test_resync_refuses_id_less_user_label_from_the_wire(tmp_path: Path) -> None:
    """A blank label id on labels.list is corrupt wire: enumerating it would
    emit a blank labelIds= filter (the empty-LIST guard cannot see [""]).
    Refused loud, ledger behind, zero enumeration spend (review carry-over)."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1",))],
        labels=[
            LabelInfo("L1", "Shipping", "user", messages_total=4),
            LabelInfo("", "Corrupt", "user", messages_total=1),
        ],
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.failed == [("<labels>", "labels.list carried an id-less user label")]
    assert not any(c.startswith("list_messages") for c in fake.calls)
    assert not any(c.startswith("get_message_metadata") for c in fake.calls)
    assert store.read_cursor() == SyncCursor(history_id=100, last_sync=NOW)


def test_estimator_picks_walk_when_the_scan_is_unpriceable(tmp_path: Path) -> None:
    """messagesTotal absent on the wire leaves the scan unpriceable — the
    walk (cost scales with the KNOWN corpus size) is the only honest
    choice, with its own rationale line. Discriminating: dropping the
    unpriceable arm leaves scan at 5 units vs walk 20 and silently picks
    the scan."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1",))],
        labels=[
            LabelInfo("L1", "Shipping", "user", messages_total=4),
            LabelInfo("L2", "Money/Bills", "user"),  # total absent: unpriceable
        ],
        current_history_id=500,
    )
    lines: list[str] = []
    stats = full_resync(
        fake,
        store,
        id_to_name=NAMES,
        sleeper=lambda _s: None,
        clock=_clock,
        progress=lines.append,
    )
    assert stats.strategy == "walk"
    assert any("unpriceable" in line for line in lines)
    assert not any(c.startswith("list_messages") for c in fake.calls)


def test_estimator_ties_go_to_the_exact_cost_walk(tmp_path: Path) -> None:
    """At parity the WALK wins: its number is exact, the scan's is only an
    enumeration floor that flagged metadata can exceed (deepseek nit)."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)), _record("b2", ("Shipping",)))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1",)),
            FakeMessage(id="b2", eml=b"x", label_ids=("L1",)),
        ],
        labels=[LabelInfo("L1", "Shipping", "user", messages_total=4000)],
        current_history_id=500,
    )
    stats = full_resync(
        fake, store, id_to_name=NAMES, sleeper=lambda _s: None, clock=_clock
    )
    assert stats.walk_units == 40 == stats.scan_list_units
    assert stats.strategy == "walk"


def test_walk_reports_its_plan_not_a_false_drift_alarm(tmp_path: Path) -> None:
    """The flood NOTE is the SCAN's drift signal; on the walk every id is
    fetched by construction, so "corpus-wide drift" would fire on a quiet
    mailbox — the walk prints its priced plan line instead (deepseek
    review, major)."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("Shipping",)), _record("b2", ()))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1",)),
            FakeMessage(id="b2", eml=b"x", label_ids=()),
        ],
        labels=_big_mailbox_labels(),
        current_history_id=500,
    )
    lines: list[str] = []
    stats = full_resync(
        fake,
        store,
        id_to_name=NAMES,
        sleeper=lambda _s: None,
        clock=_clock,
        progress=lines.append,
    )
    assert stats.strategy == "walk"
    assert stats.revised == 0 and stats.unchanged == 2  # a quiet mailbox
    joined = "\n".join(lines)
    assert "corpus-wide drift" not in joined
    assert "plan: metadata-get all 2" in joined


def test_scan_converges_id_names_that_resolved_since_import(tmp_path: Path) -> None:
    """A record holding "id:L9" whose id the mailbox NOW lists (named
    "News") membership-matches via the id: set — but the NAME is stale.
    The scan must fetch and normalize it (one self-limiting extra get:
    the revision drops the id: entry), or the two strategies would
    disagree on ground truth forever (deepseek review, minor)."""
    store = store_at(tmp_path)
    _seed(store, _record("a1", ("id:L9", "Shipping")))
    store.write_cursor(SyncCursor(history_id=100, last_sync=NOW))
    fake = FakeGmailClient(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        labels=[
            LabelInfo("L1", "Shipping", "user", messages_total=4),
            LabelInfo("L9", "News", "user", messages_total=4),
        ],
        current_history_id=500,
    )
    stats = full_resync(
        fake,
        store,
        id_to_name={"L1": "Shipping", "L9": "News"},
        sleeper=lambda _s: None,
        clock=_clock,
    )
    assert stats.strategy == "scan"
    assert stats.revised == 1
    assert stats.gains == {"News": 1} and stats.losses == {"id:L9": 1}
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "News")
    assert "get_message_metadata:a1" in fake.calls
