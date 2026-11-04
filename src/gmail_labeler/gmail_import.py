"""Resumable Gmail Import (GLOSSARY: Corpus, Gmail Import; source mode ``api``).

Pulls messages through the GmailClient Protocol, parses via the single shared
MIME path (invariant #4), and writes CorpusRecords through CorpusStore.
Resume is *structural*: the store's id set is the resume point (skip-present is the
same property that makes crash-retry harmless — invariant #2's corpus twin),
so the import can die at any moment and simply be re-run. Retry policy is driven
by the typed taxonomy only (transient→backoff-retry, rate_limited→honored
backoff, permanent→record-and-continue: one unreadable message must never
abort a 394k-message import). Messages whose timestamp cannot be
established from either source (API internalDate or Date header) are FAILED,
not fabricated — the temporal split (invariant #7) starts honest.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from .attribution import SIDECAR_FILTERS, SIDECAR_LABELS
from .corpus_record import CorpusRecord, body_hash, utc_datetime
from .corpus_store import CorpusStore
from .gmail_client import FilterInfo, GmailClient, MessagePage
from .gmail_errors import GmailError
from .labels import normalize_label_names
from .mime_extract import decode_gmail_raw, parse_mime
from .retry import MAX_ATTEMPTS
from .retry import retrying as _retrying
from .sync_cursor import SyncCursor

log = logging.getLogger("gmail_labeler.gmail_import")

DEFAULT_QUERY = "in:anywhere -in:spam -in:trash"
# MAX_ATTEMPTS and the pacing loop are shared with Label Sync via retry.py
# (ticket #73 generalization); the constant is imported here for the
# default arguments below — the loop itself no longer lives in this module.


@dataclass
class GmailImportStats:
    scanned: int = 0
    written: int = 0
    skipped_present: int = 0
    retried: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (id, reason)

    @property
    def failed_ids(self) -> list[str]:
        return [i for i, _ in self.failed]


def _ms_to_utc(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=UTC)
    except (ValueError, OSError, OverflowError):
        return None


def import_gmail(
    client: GmailClient,
    store: CorpusStore,
    *,
    id_to_name: Mapping[str, str],
    query: str = DEFAULT_QUERY,
    max_messages: int | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    sleeper: Callable[[float], None] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> GmailImportStats:
    """Import the query's messages into the store. Returns counts + failures.

    Side effect beyond the record log (ADR-0008, additive): births the Sync
    Cursor when — and only when — the corpus has none. Advancement is Label
    Sync's exclusive act; see _birth_cursor."""
    sleeper = sleeper or time.sleep
    clock = clock or (lambda: datetime.now(UTC))
    if max_attempts < 1:
        # a zero-attempt loop would be a silent no-op import (deepseek:
        # the 'exhausted' return below was its only executed line)
        raise ValueError("max_attempts must be >= 1")
    stats = GmailImportStats()
    if max_messages is not None and max_messages <= 0:
        return stats  # explicit no-op, never "process one anyway"
    page_token: str | None = None
    seeding_done = False
    while True:
        page = _retrying(
            lambda tok=page_token: client.list_messages(query, tok),
            max_attempts=max_attempts,
            stats=stats,
            sleeper=sleeper,
            what="list_messages",
        )
        if page is None:
            # '<page>' is a SENTINEL, not a message id (failed_ids callers)
            stats.failed.append(
                ("<page>", "list_messages stopped (permanent or exhausted)")
            )
            return stats
        if not seeding_done:
            # ADR-0008: EVERY seed — page or profile — is captured BEFORE any
            # message is ingested, on the first page only. Events after
            # capture carry higher ids, so the seed is a lower bound and
            # relabels made during the import land in the next sync's window
            # instead of being silently skipped.
            seeding_done = True
            _birth_cursor(
                client,
                store,
                page,
                max_attempts=max_attempts,
                stats=stats,
                sleeper=sleeper,
                clock=clock,
            )
        page_token = page.next_page_token
        for stub in page.stubs:
            stats.scanned += 1
            if store.has(stub.id.lower()):  # ids are stored lowercased
                stats.skipped_present += 1
            else:
                outcome = _import_one(
                    client,
                    store,
                    stub.id,
                    id_to_name,
                    max_attempts=max_attempts,
                    stats=stats,
                    sleeper=sleeper,
                )
                if outcome == "written":
                    stats.written += 1
                elif outcome == "present":
                    stats.skipped_present += 1
            # bound checked on EVERY path (deepseek engine pass: continue
            # used to bypass it, scanning past max_messages on resume runs)
            if max_messages is not None and stats.scanned >= max_messages:
                return stats
        if page_token is None:
            return stats


def _birth_cursor(
    client: GmailClient,
    store: CorpusStore,
    page: MessagePage,
    *,
    max_attempts: int,
    stats: GmailImportStats,
    sleeper: Callable[[float], None],
    clock: Callable[[], datetime],
) -> None:
    """Birth the Sync Cursor when — and only when — the corpus has none
    (ADR-0008: birth-when-absent; advancement is Label Sync's exclusive act,
    so an existing ledger is left byte-identical without a quota call).

    Seed order (the sync guide): the FIRST (newest) stub's historyId where
    the wire carries it — Gmail's documented stub promises only id/threadId,
    so absence is schema-legal, never an error — falling back to ONE
    get_profile call (1 quota unit, the reliable path). Both are captured
    before any message is ingested, keeping the seed a lower bound. A
    negative seed is a wire lie the ledger refuses. When no usable seed
    exists the import degrades LOUDLY but not fatally: the cursor is
    additive, the corpus stays birth-eligible, and failing a healthy
    394k-message import over one bootstrap read would regress the
    acquisition contract. A write failure (CorpusError) propagates like any
    structural store fault — typed exit 5."""
    if store.read_cursor() is not None:
        return  # birth-when-absent: an existing ledger is sync's to advance
    seed = page.stubs[0].history_id if page.stubs else None
    if seed is not None and seed <= 0:
        # a non-positive page seed is a wire lie (history ids count from 1):
        # treat it as absent so the guide-ordered profile fallback gets its
        # turn, instead of birthing a ledger the first sync must 400 over
        seed = None
    if seed is None:
        profile = _retrying(
            client.get_profile,
            max_attempts=max_attempts,
            stats=stats,
            sleeper=sleeper,
            what="get_profile",
        )
        if profile is not None:
            seed = profile.history_id
    if seed is None or seed <= 0:
        log.warning(
            "import: no usable historyId seed (first page and profile) — "
            "sync cursor NOT birthed; the first Label Sync will birth it"
        )
        return
    store.write_cursor(SyncCursor(history_id=seed, last_sync=clock()))


def _import_one(
    client: GmailClient,
    store: CorpusStore,
    message_id: str,
    id_to_name: Mapping[str, str],
    *,
    max_attempts: int,
    stats: GmailImportStats,
    sleeper: Callable[[float], None],
) -> Literal["written", "present", "failed"]:
    msg = _retrying(
        lambda: client.get_message(message_id),
        max_attempts=max_attempts,
        stats=stats,
        sleeper=sleeper,
        what=f"get_message:{message_id}",
    )
    if msg is None:
        stats.failed.append((message_id, "fetch failed (permanent/retries)"))
        return "failed"
    try:
        parsed = parse_mime(decode_gmail_raw(msg.raw_b64url))
    except ValueError as e:
        # decode_gmail_raw is intentionally loud on corrupt raw (documented;
        # binascii.Error and UnicodeEncodeError are ValueError subclasses)
        stats.failed.append((message_id, f"raw decode: {type(e).__name__}"))
        return "failed"
    # Date provenance: server internalDate when present; the MIME Date
    # header is SENDER-CLAIMED (a forged/future header can move a message
    # across the temporal split — a per-record provenance flag is
    # daemon/eval-phase work, see follow-up). No header => no guess.
    date = _ms_to_utc(msg.internal_date_ms) or (
        utc_datetime(parsed.date) if parsed.date is not None else None
    )
    if date is None:
        stats.failed.append((message_id, "no authoritative timestamp"))
        return "failed"
    try:
        record = CorpusRecord(
            id=message_id.lower(),
            date=date,
            subject=parsed.subject,
            sender=parsed.sender,
            recipients=parsed.recipients,
            labels=tuple(normalize_label_names(list(msg.label_ids), id_to_name)),
            body_hash=body_hash(parsed.body_text),
            source="api",
            thread_id=msg.thread_id,
        )
    except ValueError as e:
        # a wire value violating the record contract fails THIS message
        # only — one impossible id must not abort a 394k import
        stats.failed.append((message_id, f"record shape: {e}"))
        return "failed"
    # Structural store failures (CorpusError) PROPAGATE and stop the run
    # loudly — continuing against a corrupt index would compound damage.
    added = store.put(record, parsed.body_text)
    return "written" if added else "present"


def fetch_settings(
    client: GmailClient,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    sleeper: Callable[[float], None] | None = None,
) -> tuple[dict[str, str], list[FilterInfo], GmailImportStats]:
    """Bootstrap reads (label map + filter snapshot) share the message
    loop's retry taxonomy: one transient blip on settings must not kill a
    healthy import. Exhaustion RAISES — attribution (invariant #6) is never
    silently skipped; the caller exits non-zero."""
    sleeper = sleeper or time.sleep
    stats = GmailImportStats()  # ride along on the return: retried= stays honest
    labels = _retrying(
        client.list_labels,
        max_attempts=max_attempts,
        stats=stats,
        sleeper=sleeper,
        what="list_labels",
    )
    if labels is None:
        raise GmailError("label map unavailable after retries")
    filters = _retrying(
        client.list_filters,
        max_attempts=max_attempts,
        stats=stats,
        sleeper=sleeper,
        what="list_filters",
    )
    if filters is None:
        raise GmailError("filter snapshot unavailable after retries")
    return {label.id: label.name for label in labels}, list(filters), stats


def write_filters_snapshot(store: CorpusStore, filters: list[FilterInfo]) -> None:
    """filters.json beside the index: the OFFLINE attribution snapshot
    dataset analyze needs (invariant #6) — metadata only, never message
    content (invariant #5)."""
    payload = json.dumps(
        [dataclasses.asdict(f) for f in filters], indent=0, sort_keys=True
    )
    store.write_snapshot(SIDECAR_FILTERS, payload)


def write_labels_snapshot(store: CorpusStore, id_to_name: Mapping[str, str]) -> None:
    """labels.json: the id->name bridge (same offline-attribution purpose)."""
    payload = json.dumps(dict(sorted(id_to_name.items())), indent=0, sort_keys=True)
    store.write_snapshot(SIDECAR_LABELS, payload)
