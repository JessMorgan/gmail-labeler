"""Label Sync — event walk + full reconciliation (GLOSSARY: Label Sync;
ADR-0007/ADR-0008).

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

``full_resync`` (ticket #74) is the recovery path that needs no history:
the remedy when the cursor aged past Gmail's retention and the first
reconciliation for GYB-sourced Corpora. ONE ``labels.list`` (per-label
``messagesTotal``) plus local resolved-view counts yield a printed cost
estimate BEFORE any enumeration spend, and the cheaper of two strategies
runs: (a) the label-inventory scan — enumerate current members of every
mailbox user label via the list ``labelIds`` parameter (ids, not search
names), diff against the Corpus's sets, metadata-fetch only flagged ids;
vanished labels' corpus members are fetched directly; or (b) the corpus-
metadata walk — fetch every corpus id's current labels, diff, revise.
Seed discipline (ADR-0008 / manager note 9): the profile historyId is
captured ONCE UP FRONT, before the scan, and persisted only after the
reconciliation's revisions are durable — an ADVANCE of the ledger (a
birth on a cursorless corpus, same ordering), never a re-birth and never
a last-page-wins value the event walk reads. A change made between seed
capture and scan end carries a higher id and lands in the next event
sync's window.

Structurally body-free (invariant #5): the only message read is
``get_message_metadata`` — no raw payload path is even imported here.
Ids in events or inventories that are not resident in the Corpus are
never written (ingest stays the acquisition paths' alone) and counted as
new-match for the report. ``messagesDeleted``-touched resident ids and
metadata-confirmed-gone ids: record kept (v1: no tombstones), counted.
Now-unresolvable label ids keep the ``id:`` deleted-label convention via
the shared normalizer.

Pure core (CODING-STANDARDS, like gmail_import): client + store +
injected sleeper/clock/progress, no typer/config imports — the ADR-0003
daemon reuses these functions verbatim. ``dry_run`` computes the
identical plan and writes nothing: no revisions, no cursor, no birth.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from .corpus_record import CorpusRecord
from .corpus_store import CorpusStore
from .gmail_client import GmailClient, HistoryPage, LabelInfo, MessageMetadata
from .gmail_errors import GmailPermanentError, GmailResyncError
from .labels import is_system_label_id, normalize_label_names
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
    stats: SyncStats | FullResyncStats,
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


# -- --full-resync: estimate-picked ground-truth reconciliation (#74) --------

#: Cost constants for the up-front estimate (spec #64 "Further Notes",
#: primary-sourced 2026-11-03): messages.list is 5 units/page (500 max),
#: messages.get is 20 units at format=metadata on the NEW quota regime —
#: pre-May-2026 GCP projects keep the flat-5 regime, so these estimates are
#: ceiling-conservative for them. The from-scratch-reload comparison (60
#: units/msg raw + full body bandwidth) is the honest worst-case foil the
#: flood NOTE quotes.
_LIST_PAGE_SIZE: Final = 500
_LIST_PAGE_UNITS: Final = 5
_METADATA_GET_UNITS: Final = 20
_RELOAD_UNITS_PER_MSG: Final = 60


@dataclass
class FullResyncStats:
    """Report counts for one reconciliation. ``strategy`` names the choice
    the estimator printed; the unit fields are the numbers behind it."""

    strategy: str = ""  # "scan" | "walk" ("" until the estimate lands)
    corpus_messages: int = 0
    enumerated_labels: int = 0  # mailbox user labels the scan would page
    scan_list_units: int = 0  # enumeration floor; flagged metadata is on top
    walk_units: int = 0
    flagged: int = 0
    revised: int = 0  # revisions appended (planned, under dry_run)
    unchanged: int = 0  # flagged/fetched but the diff came back empty
    deleted_kept: int = 0  # metadata get 404'd; record kept
    new_match: int = 0  # inventory ids not resident (report only)
    retried: int = 0  # shared RetryStats protocol counter
    birthed: bool = False  # cursorless corpus: this run births (ADR-0008)
    seed_history_id: int | None = None  # the up-front profile value written
    gains: dict[str, int] = field(default_factory=dict)  # label -> count
    losses: dict[str, int] = field(default_factory=dict)
    # (id, reason); "<cursor>"/"<labels>"/"<label:ID>" are SENTINELS, not
    # message ids — report renderers must not treat them as ids
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def failed_ids(self) -> list[str]:
        return [i for i, _ in self.failed]

    @property
    def flag_rate(self) -> float:
        """flagged / corpus size — the flood guard's number (0.0 on an
        empty corpus)."""
        if not self.corpus_messages:
            return 0.0
        return self.flagged / self.corpus_messages


def full_resync(
    client: GmailClient,
    store: CorpusStore,
    *,
    id_to_name: Mapping[str, str],
    max_attempts: int = MAX_ATTEMPTS,
    sleeper: Callable[[float], None] | None = None,
    clock: Callable[[], datetime] | None = None,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> FullResyncStats:
    """Ground-truth label reconciliation needing no history (ticket #74).

    Raises (typed, never swallowed to a silent success): GmailAuthError
    (exit-4 territory), CorpusError on store faults. A permanently-failing
    read or fetch is recorded in ``failed`` and keeps the ledger behind the
    reconciliation (invariant #3) — the next run redoes the whole thing,
    and the fold absorbs the replays.

    ``progress`` receives the estimate + strategy lines BEFORE any
    enumeration spend, and — on the SCAN — the flag-rate line BEFORE the
    metadata pass (the walk prints its priced plan line there instead: a
    100% flag-rate on the walk is the strategy, not drift). The CLI
    echoes it; the daemon may ignore it. ``dry_run`` computes the
    identical plan (reads included: the plan IS the diff) and writes
    nothing.
    """
    sleeper = sleeper or time.sleep
    clock = clock or (lambda: datetime.now(UTC))
    say = progress or (lambda _line: None)
    if max_attempts < 1:
        # a zero-attempt loop would be a silent no-op resync (import parity)
        raise ValueError("max_attempts must be >= 1")
    stats = FullResyncStats()
    # the birth flag reads the ledger locally — no quota call, and an
    # unparseable cursor raises CorpusError here (never a silent re-birth)
    had_cursor = store.read_cursor() is not None

    # 1. SEED UP FRONT (note 9, binding): ONE profile call before any
    #    enumeration. Events after capture carry higher ids, so persisting
    #    THIS value keeps mid-scan changes inside the next sync's window.
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
    stats.seed_history_id = seed

    # 2. ONE labels.list — the estimate's only mailbox read besides the seed
    labels = retrying(
        client.list_labels,
        max_attempts=max_attempts,
        stats=stats,
        sleeper=sleeper,
        what="list_labels",
    )
    if labels is None:
        stats.failed.append(("<labels>", "labels.list failed (permanent/retries)"))
        return stats

    # 3. local resolved-view counts (ADR-0007: the ONLY read interface)
    resolved = {rec.id: rec for rec in store.iter_resolved()}
    user_labels = [lbl for lbl in labels if not is_system_label_id(lbl.id)]
    if any(not lbl.id for lbl in user_labels):
        # a blank label id on labels.list is corrupt wire: enumerating it
        # would emit a blank labelIds= filter (the empty-LIST guard in
        # list_messages cannot see [""]). Refuse loud, ledger behind —
        # guessing here is the silent-gap class (review carry-over, #74)
        stats.failed.append(("<labels>", "labels.list carried an id-less user label"))
        return stats
    stats.corpus_messages = len(resolved)
    stats.enumerated_labels = len(user_labels)
    # a label whose messagesTotal the wire omitted leaves the scan
    # unpriceable — the walk (cost scales with the KNOWN corpus size) is
    # then the only honest choice, and the printed rationale says so
    unpriceable = any(lbl.messages_total is None for lbl in user_labels)
    stats.scan_list_units = sum(
        _LIST_PAGE_UNITS * -((lbl.messages_total or 0) // -_LIST_PAGE_SIZE)
        for lbl in user_labels
    )
    stats.walk_units = _METADATA_GET_UNITS * len(resolved)
    # tie goes to the WALK (deepseek review nit): the scan number is only
    # an enumeration FLOOR that flagged metadata can exceed, while the
    # walk number is exact — at parity the exact cost wins
    strategy = (
        "walk" if (unpriceable or stats.walk_units <= stats.scan_list_units) else "scan"
    )
    stats.strategy = strategy
    say(
        f"full-resync estimate: corpus={stats.corpus_messages} messages; "
        f"mailbox user labels={stats.enumerated_labels}; "
        f"scan enumeration floor ≈ {stats.scan_list_units} units "
        f"(+{_METADATA_GET_UNITS}/flagged metadata), "
        f"walk ≈ {stats.walk_units} units "
        f"[{_METADATA_GET_UNITS} units/msg]"
    )
    if unpriceable:
        say(
            f"full-resync strategy: walk — labels.list omitted messagesTotal "
            f"for some labels, the scan is unpriceable; the walk costs a known "
            f"{stats.walk_units} units for {stats.corpus_messages} corpus messages"
        )
    else:
        say(
            f"full-resync strategy: {strategy} — walk {stats.walk_units} units "
            f"vs scan {stats.scan_list_units} units enumeration floor "
            f"(flagged metadata adds up to "
            f"{_METADATA_GET_UNITS * stats.corpus_messages} more)"
        )

    # 4. flag the ids whose label state may have drifted
    if strategy == "walk":
        flagged: list[str] = sorted(resolved)
    else:
        flagged_or_none = _inventory_flagged(
            client, resolved, user_labels, stats, max_attempts, sleeper
        )
        if flagged_or_none is None:
            # enumeration stopped (failure recorded): the flagged set is
            # INCOMPLETE — nothing has been fetched or written, bail with
            # the ledger behind (a re-run redoes the whole scan harmlessly)
            return stats
        flagged = flagged_or_none
    stats.flagged = len(flagged)

    # Flood guard BEFORE the metadata spend — strategy-shaped (deepseek
    # review): on the SCAN, flagged IS the drift signal, so a corpus-wide
    # flag-rate must be visible while the run can still be stopped, with
    # the honest worst-case cost story (spec #64's numbers). On the WALK,
    # every corpus id is fetched by construction — the flag count is the
    # plan, not evidence of drift (drift is measured by the diff), and the
    # estimate line already priced the walk's exact cost. A drift-blind
    # alarm on the walk would lie on its common path.
    if strategy == "scan":
        say(
            f"full-resync flagged={stats.flagged} of {stats.corpus_messages} "
            f"corpus messages (flag-rate {stats.flag_rate:.0%})"
        )
        if resolved and stats.flagged * 2 >= stats.corpus_messages:
            say(
                f"NOTE: corpus-wide drift — the metadata pass costs up to "
                f"{_METADATA_GET_UNITS * stats.flagged} units worst-case "
                f"({_METADATA_GET_UNITS}/msg new-regime), still below a "
                f"from-scratch reload (≈ {_RELOAD_UNITS_PER_MSG} units/msg + "
                f"full body bandwidth)"
            )
    else:
        say(
            f"full-resync plan: metadata-get all {stats.flagged} corpus ids "
            f"(≈ {stats.walk_units} units priced up front); the walk fetches "
            f"everything by construction — drift is measured by the diff"
        )

    # 5. metadata-fetch ONLY flagged ids; diff; revise drifted ones
    gone_confirmed: set[str] = set()
    for message_id in flagged:
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
            # the has() gate (caller-side duty, note 9a): a Revision
            # supersedes a resident record, it never creates one
            stats.failed.append((message_id, "resolved record vanished mid-run"))
            continue
        new_labels = tuple(normalize_label_names(list(meta.label_ids), id_to_name))
        if set(new_labels) == set(resident.labels):
            # reconciliation diffs before appending (unlike the event
            # walk's self-heal): a full-corridor re-append on every
            # resync would double the revision log for no signal
            stats.unchanged += 1
            continue
        _revise(
            resident,
            meta,
            id_to_name,
            stats,
            store=store if not dry_run else None,
        )
    stats.deleted_kept = len(gone_confirmed)

    # 6. the ledger moves only over durable revisions (invariant #3) —
    #    and to the UP-FRONT seed, never an end-of-run read (note 9)
    if stats.failed:
        log.warning(
            "full-resync: %d failure(s) — Sync Cursor NOT written; "
            "re-running redoes the reconciliation",
            len(stats.failed),
        )
        return stats
    stats.birthed = not had_cursor
    if not dry_run:
        # every revision line above was appended+fsynced through
        # append_revision BEFORE this write
        store.write_cursor(SyncCursor(history_id=seed, last_sync=clock()))
    return stats


def _inventory_flagged(
    client: GmailClient,
    resolved: Mapping[str, CorpusRecord],
    user_labels: Sequence[LabelInfo],
    stats: FullResyncStats,
    max_attempts: int,
    sleeper: Callable[[float], None],
) -> list[str] | None:
    """Strategy (a): enumerate every mailbox user label's current members by
    label id and symmetric-diff against the Corpus's member sets; a corpus
    id whose membership moved either way is flagged. None means an
    enumeration stopped permanently — the set is incomplete, the caller
    must not proceed.

    Completeness argument (pinned by tests, spec #64): every user-label
    gain or loss changes membership in an enumerated set — the enumerated
    family is ALL mailbox user labels, which is exactly (corpus-known union
    mailbox-listed) minus the vanished ones; a corpus label no mailbox
    label maps to is vanished and cannot be listed, so its corpus members
    are flagged for direct metadata reads instead. System-label-only
    churn touches no enumerated set and cannot drift a corpus record (the
    Corpus stores user-label names only — labels.py filters System Labels
    at write time). Non-resident inventory ids are never written: ingest
    stays the acquisition paths' alone.
    """
    # corpus membership index: label name -> corpus ids holding it (built
    # once — per-label full scans would be O(labels x corpus))
    corpus_by_name: dict[str, set[str]] = {}
    for record_id, rec in resolved.items():
        for name in rec.labels:
            corpus_by_name.setdefault(name, set()).add(record_id)

    flagged: dict[str, None] = {}  # insertion-ordered set
    non_resident: set[str] = set()
    for lbl in user_labels:
        members = _enumerate_members(client, lbl.id, stats, max_attempts, sleeper)
        if members is None:
            return None
        # the id: convention names the same label for records imported
        # while its mapping was missing — they belong to this set too
        corpus_members = corpus_by_name.get(lbl.name, set()) | corpus_by_name.get(
            f"id:{lbl.id}", set()
        )
        for mid in members:
            if mid in resolved:
                if mid not in corpus_members:  # gained membership, resident
                    flagged[mid] = None
            else:
                non_resident.add(mid)
        for mid in corpus_members:
            if mid not in members:  # lost membership
                flagged[mid] = None
    # id: entries and vanished names: no inventory can speak for them, so
    # their corpus members are metadata-fetched directly (spec #64
    # deleted-label subcase). This covers the RESOLVED-SINCE case too
    # (deepseek review): a record holding "id:L9" whose id the mailbox now
    # lists would membership-match via the id: set above and never be
    # fetched, keeping the stale id: form forever under the scan while the
    # walk would normalize it — the flag count is one extra get per such
    # record, self-limiting for the converging case (the revision drops
    # the id: entry). A record whose metadata STILL normalizes back to the
    # same id: form (truly vanished id, or a blank-name map edge the CLI's
    # own labels.list map makes unreachable) is re-fetched every run with
    # NO revision churn — the diff gate counts it unchanged — one bounded
    # get, which is the spec's direct-fetch mandate for vanished labels
    # anyway.
    mailbox_names = {lbl.name for lbl in user_labels}
    for record_id, rec in resolved.items():
        for name in rec.labels:
            if name.startswith("id:") or name not in mailbox_names:
                flagged[record_id] = None
    stats.new_match = len(non_resident)
    return sorted(flagged)  # deterministic pass order (stable call logs)


def _enumerate_members(
    client: GmailClient,
    label_id: str,
    stats: FullResyncStats,
    max_attempts: int,
    sleeper: Callable[[float], None],
) -> set[str] | None:
    """Current member ids of one label via the list labelIds parameter
    (ids, not search names — dodges slash-path quoting), paged to
    exhaustion. Lowercased at the gate like every other id crossing into
    corpus comparisons (note 9a)."""
    members: set[str] = set()
    page_token: str | None = None
    while True:
        page = retrying(
            lambda tok=page_token: client.list_messages("", tok, label_ids=[label_id]),
            max_attempts=max_attempts,
            stats=stats,
            sleeper=sleeper,
            what=f"list_messages labelIds={label_id}",
        )
        if page is None:
            # '<label:ID>' is a SENTINEL, not a message id (import parity)
            stats.failed.append(
                (
                    f"<label:{label_id}>",
                    "inventory enumeration stopped (permanent or exhausted)",
                )
            )
            return None
        members.update(stub.id.lower() for stub in page.stubs)
        page_token = page.next_page_token
        if page_token is None:
            return members


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
