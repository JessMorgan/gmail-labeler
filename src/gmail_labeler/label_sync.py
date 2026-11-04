"""Label Sync event walk (GLOSSARY: Label Sync; ADR-0007/ADR-0008).

The headline freshness loop: after a Backfill, bring the Corpus's label
state up to the mailbox's — no re-assembly. Read the Sync Cursor, page
``history`` to exhaustion on the shared retry/pacing taxonomy, mark every
event-named corpus-resident id stale (refresh-on-any-touch: identical
completeness to event-type parsing, insures against unfaithful diffs),
then metadata-fetch each stale id and append a Revision (ADR-0007: a full
revised line, last-wins fold; only ``labels`` differ — built with
``dataclasses.replace()`` on the resolved record, the caller-side field-
parity duty). The Cursor advances only after the delta's revision lines
are fsynced (invariant #3), and only when the delta had no failures: a
replayed window is harmless (the fold absorbs it), a skipped one is the
silent-gap class the ledger exists to prevent.

Structurally body-free (invariant #5): the only message read is
``get_message_metadata`` — no raw payload path is even imported here.
Ids in events that are not resident in the Corpus are never written
(ingest stays the acquisition paths' alone) and counted as new-match for
the report. ``messagesDeleted``-touched resident ids: record kept (v1: no
tombstones), counted. Now-unresolvable label ids keep the ``id:``
deleted-label convention via the shared normalizer.

Pure core (CODING-STANDARDS, like gmail_import): client + store +
injected sleeper/clock, no typer/config imports — the ADR-0003 daemon
reuses this function verbatim. ``dry_run`` computes the identical plan
and writes nothing: no revisions, no cursor, no birth.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from .corpus_record import CorpusRecord
from .corpus_store import CorpusStore
from .gmail_client import GmailClient, HistoryPage, MessageMetadata
from .gmail_errors import GmailPermanentError, GmailResyncError
from .labels import normalize_label_names
from .retry import MAX_ATTEMPTS, retrying
from .sync_cursor import SyncCursor

log = logging.getLogger("gmail_labeler.label_sync")

#: History records are opaque wire dicts; these are the typed entry fields
#: that name a message (the flat ``messages[]`` duplicates them — Google
#: recommends the typed fields, and the fake emits only those). Unknown
#: fields are ignored: events with NO entry for a changed message are
#: --full-resync's domain, not a reason to fail the walk.
_EVENT_MESSAGE_FIELDS = (
    "messagesAdded",
    "labelsAdded",
    "labelsRemoved",
    "messagesDeleted",
)


#: Sentinel for "the metadata get 404'd" — the message is unretrievable
#: upstream (deleted). A distinct type, not None: None already means
#: "fetch failed, record it" through the retry helper.
class _Gone:
    __slots__ = ()


_GONE: Final = _Gone()


@dataclass
class SyncStats:
    events: int = 0  # history records walked
    stale: int = 0  # corpus-resident ids the walk marked for refresh
    revised: int = 0  # revisions appended (planned, under dry_run)
    deleted_kept: int = 0  # resident ids confirmed gone; records kept
    new_match: int = 0  # event ids not resident in the corpus (report only)
    retried: int = 0  # shared RetryStats protocol counter
    birthed: bool = False  # ADR-0008 consequence 1: cursorless corpus
    gains: dict[str, int] = field(default_factory=dict)  # label -> count
    losses: dict[str, int] = field(default_factory=dict)
    # (id, reason); "<page>"/"<cursor>" are SENTINELS, not message ids —
    # report renderers must not treat them as ids (gmail_import parity)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def failed_ids(self) -> list[str]:
        return [i for i, _ in self.failed]


def sync_labels(
    client: GmailClient,
    store: CorpusStore,
    *,
    id_to_name: Mapping[str, str],
    max_attempts: int = MAX_ATTEMPTS,
    sleeper: Callable[[float], None] | None = None,
    clock: Callable[[], datetime] | None = None,
    dry_run: bool = False,
) -> SyncStats:
    """Event-walk Label Sync. Returns counts + per-label gains/losses.

    Raises (typed, never swallowed to a silent success): GmailResyncError
    on an expired cursor (history 404 — the CLI prints recovery guidance),
    GmailAuthError (exit-4 territory), CorpusError on store faults. A
    permanently-failing metadata fetch is recorded in ``failed`` and keeps
    the cursor behind the delta (invariant #3).
    """
    sleeper = sleeper or time.sleep
    clock = clock or (lambda: datetime.now(UTC))
    if max_attempts < 1:
        # a zero-attempt loop would be a silent no-op sync (gmail_import precedent)
        raise ValueError("max_attempts must be >= 1")
    stats = SyncStats()
    cursor = store.read_cursor()
    if cursor is None:
        return _birth_without_applying(
            client,
            store,
            stats,
            max_attempts=max_attempts,
            sleeper=sleeper,
            clock=clock,
            dry_run=dry_run,
        )

    stale: dict[str, None] = {}  # insertion-ordered set: walk order is the report order
    deleted_touched: set[str] = set()
    new_match: set[str] = set()
    page_token: str | None = None
    final_history_id: int | None = None
    while True:
        page = retrying(
            lambda tok=page_token: _history(client, cursor.history_id, tok),
            max_attempts=max_attempts,
            stats=stats,
            sleeper=sleeper,
            what="history",
        )
        if page is None:
            # '<page>' is a SENTINEL, not a message id (gmail_import parity)
            stats.failed.append(("<page>", "history stopped (permanent or exhausted)"))
            return stats
        stats.events += len(page.records)
        for record in page.records:
            for message_id, is_delete in _event_message_ids(record):
                if store.has(message_id):
                    if is_delete:
                        deleted_touched.add(message_id)
                    # refresh-on-any-touch — a delete touch marks stale too:
                    # if the message came back, the metadata get says so
                    stale.setdefault(message_id, None)
                else:
                    new_match.add(message_id)
        final_history_id = page.history_id
        page_token = page.next_page_token
        if page_token is None:
            break
    stats.stale = len(stale)
    stats.new_match = len(new_match)

    gone_confirmed: set[str] = set()
    if stale:
        # The resolved view is the ONLY read interface (ADR-0007): the
        # replace() base must be the newest line, and gains/losses compare
        # against it. Folded only when the delta actually touches the
        # corpus — the daemon's idle 30 s poll never pays the whole-log
        # fold (ADR-0007 accepts the fold cost; it need not be paid empty).
        resolved = {rec.id: rec for rec in store.iter_resolved()}
        for message_id in stale:
            meta = retrying(
                lambda mid=message_id: _metadata_or_gone(client, mid),
                max_attempts=max_attempts,
                stats=stats,
                sleeper=sleeper,
                what=f"get_message_metadata:{message_id}",
            )
            if isinstance(meta, _Gone):  # narrows the union for the get below
                gone_confirmed.add(message_id)
                continue
            if meta is None:
                stats.failed.append(
                    (message_id, "metadata fetch failed (permanent/retries)")
                )
                continue
            resident = resolved.get(message_id)
            if resident is None or not store.has(message_id):
                # the has() gate (caller-side duty): a Revision supersedes
                # a resident record, it never creates one. Unreachable for
                # a consistent log (has() and iter_resolved() share one
                # reader) — if it ever fires that IS a gap, so record it,
                # never drop it silently (invariant #3's posture)
                stats.failed.append((message_id, "resolved record vanished mid-run"))
                continue
            _revise(
                resident,
                meta,
                id_to_name,
                stats,
                store=store if not dry_run else None,
            )
    stats.deleted_kept = len(deleted_touched | gone_confirmed)

    if stats.failed:
        # never ahead of durability — including the delta's own failed
        # members: the next run replays the window and the fold absorbs
        log.warning(
            "sync-labels: %d failure(s) — Sync Cursor NOT advanced; "
            "re-running resumes the same window",
            len(stats.failed),
        )
        return stats
    if final_history_id is None or final_history_id < 1:
        # the mailbox current id is the new ledger value; without it the
        # walk cannot honestly claim the window is consumed — keep the old
        # cursor (a replay is harmless) and report the gap loudly
        stats.failed.append(
            ("<cursor>", "no usable historyId on the final history page")
        )
        return stats
    if not dry_run:
        # durability ordering (invariant #3): every revision line above was
        # appended+fsynced through append_revision BEFORE this write
        store.write_cursor(SyncCursor(history_id=final_history_id, last_sync=clock()))
    return stats


def _birth_without_applying(
    client: GmailClient,
    store: CorpusStore,
    stats: SyncStats,
    *,
    max_attempts: int,
    sleeper: Callable[[float], None],
    clock: Callable[[], datetime],
    dry_run: bool,
) -> SyncStats:
    """ADR-0008 consequence 1: the first Label Sync on a cursorless corpus
    births the ledger and reports — it applies nothing, because there is no
    consumed baseline to diff against yet (a GYB-sourced corpus always
    lands here). The seed is one profile call (sync has no list page); the
    report discloses the provenance: the cursor is fresh while the corpus's
    label state may be months behind. A seed failure is a recorded failure
    (exit non-zero) — unlike Gmail Import, where birth is a mere side
    effect of the real job."""
    profile = retrying(
        client.get_profile,
        max_attempts=max_attempts,
        stats=stats,
        sleeper=sleeper,
        what="get_profile",
    )
    seed = profile.history_id if profile is not None else None
    if seed is None or seed <= 0:
        stats.failed.append(("<cursor>", "no usable historyId seed (profile call)"))
        return stats
    stats.birthed = True
    if not dry_run:
        store.write_cursor(SyncCursor(history_id=seed, last_sync=clock()))
    return stats


def _history(
    client: GmailClient, start_history_id: int, page_token: str | None
) -> HistoryPage:
    """One history page, with the 404 taxonomy reading normalized to ONE
    type: resync detection reads status == 404 off GmailPermanentError so
    every spine works — the fake and the real adapter raise
    GmailResyncError (a subclass), a bare permanent-404 still converts."""
    try:
        return client.history(start_history_id, page_token)
    except GmailResyncError:
        raise
    except GmailPermanentError as e:
        if e.status == 404:
            raise GmailResyncError(
                f"history.list 404: startHistoryId or pageToken no longer "
                f"retained — full re-sync required: {e}",
                status=404,
            ) from e
        raise


def _metadata_or_gone(client: GmailClient, message_id: str) -> MessageMetadata | _Gone:
    """Metadata-only get (invariant #5: the raw path is never reached from
    here). A 404 is the wire's 'message unretrievable upstream' — deleted
    mail whose corpus record stays kept-and-reported, not a failure; any
    other GmailError rides the shared taxonomy untouched."""
    try:
        return client.get_message_metadata(message_id)
    except GmailResyncError:
        raise  # resync is never a per-message fault
    except GmailPermanentError as e:
        if e.status == 404:
            return _GONE
        raise


def _revise(
    resident: CorpusRecord,
    meta: MessageMetadata,
    id_to_name: Mapping[str, str],
    stats: SyncStats,
    *,
    store: CorpusStore | None,
) -> None:
    """Append the Revision (or plan it, dry_run): field parity is
    CALLER-owned (ADR-0007 / note 9a) — dataclasses.replace() on the
    resolved record keeps body_hash/date/source/envelope byte-equal; only
    the normalized label set changes. Label normalization is the same map
    the import uses: System Labels filtered, now-unresolvable ids under
    the id: convention. Revisions append even when labels look unchanged
    (self-healing: the walk said something moved; the fold absorbs)."""
    revised = dataclasses.replace(
        resident,
        labels=tuple(normalize_label_names(list(meta.label_ids), id_to_name)),
    )
    old = set(resident.labels)
    new = set(revised.labels)
    for name in new - old:
        stats.gains[name] = stats.gains.get(name, 0) + 1
    for name in old - new:
        stats.losses[name] = stats.losses.get(name, 0) + 1
    if store is not None:
        store.append_revision(revised)  # CorpusError propagates: exit-5 surface
    stats.revised += 1


def _event_message_ids(record: Mapping[str, object]) -> Iterator[tuple[str, bool]]:
    """(lowercased message id, is messagesDeleted) for every entry naming a
    message in one wire History record. Absent-tolerant by shape: a field
    that is not the documented list-of-objects contributes nothing — the
    walk never guesses at malformed entries, and events that name no
    message cannot touch anything."""
    for field_name in _EVENT_MESSAGE_FIELDS:
        entries = record.get(field_name)
        if not isinstance(entries, list):
            continue
        is_delete = field_name == "messagesDeleted"
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            mid = message.get("id")
            if isinstance(mid, str) and mid:
                yield (mid.lower(), is_delete)
