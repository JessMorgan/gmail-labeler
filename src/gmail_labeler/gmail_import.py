"""Resumable Gmail Import (GLOSSARY: Corpus, Gmail Import; source mode ``api``).

Fetches messages through the GmailClient Protocol, parses via the single shared
MIME path (invariant #4), and writes CorpusRecords through CorpusStore.
Resume is *structural*: the store's id set is the resume point (skip-present is
the same property that makes crash-retry harmless — invariant #2's corpus
twin), so the import can die at any moment and simply be re-run. Retry policy
is driven by the typed taxonomy only (transient→backoff-retry,
rate_limited→honored backoff, permanent→record-and-continue: one unreadable
message must never abort the whole import). Messages whose timestamp
cannot be established from either source (API internalDate or Date header) are
FAILED, not fabricated — the temporal split (invariant #7) starts honest.
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
from .corpus_record import CorpusRecord, DateProvenance, body_hash, utc_datetime
from .corpus_store import CorpusStore
from .gmail_client import FilterInfo, GmailClient, MessagePage
from .gmail_errors import GmailError
from .labels import EXCLUDED_CORPUS_LABEL_IDS, normalize_label_names
from .mime_extract import decode_gmail_raw, parse_mime
from .retry import MAX_ATTEMPTS
from .retry import retrying as _retrying
from .sync_cursor import SyncCursor

log = logging.getLogger("gmail_labeler.gmail_import")

# Export scope decision (maintainer, 2026-11-01; ticket #17 / issue #10
# item 4): DRAFT and CHAT are EXCLUDED, not just spam/trash. With
# Additive-only + skip-present (invariants #1/#2) and the append-only
# corpus, the old implicit inclusion (in:anywhere) would become permanent
# and unrecoverable — excluding them now is the safe direction under the
# one-way door.
#
# Documented tokens only (provenance standard, labels.py precedent):
#   -in:draft — Google-documented: "-in:draft means exclude drafts from
#     the search results" (developers.google.com/workspace/gmail/api/
#     reference/mcp/tools_list/search_threads, page 2026-07-21). High
#     confidence as a token; medium for exact REST q composition — the
#     ingest rejection below covers the gap.
#   CHAT — NO documented exclusion token: is:chat documented 2014→2021,
#     gone by 2023-01 (Wayback 7190 lineage); -in:chat is undocumented
#     and carries a silent word-degradation hazard (an unknown in: value
#     may be word-degraded and drop legit mail containing the word
#     "chat"). Chat exclusion therefore lives at INGEST:
#     labels.EXCLUDED_CORPUS_LABEL_IDS (the shared home in labels.py)
#     hard-skips raw-wire DRAFT/CHAT labelIds — the load-bearing layer,
#     which an explicit `--query` override cannot bypass. The GYB importer
#     imports the SAME constant (ticket #32: two ingest layers, one
#     policy, no drift).
# Pinned in test_import_gmail_scope_pins_both_layers.
DEFAULT_QUERY = "in:anywhere -in:spam -in:trash -in:draft"

# MAX_ATTEMPTS and the pacing loop are shared with Label Sync via retry.py
# (ticket #73 generalization); the constant is imported here for the
# default arguments below — the loop itself no longer lives in this module.

# Per-stub outcome vocabulary (#61): the shared name for the loop's branches,
# used by _import_one, the on_message callback, and the CLI renderer —
# documentation + completion today (these modules are pyright `basic`, where
# a typo'd comparison is NOT gated until the strict migration); a type error
# then (#62's bar is the declared next consumer).
ImportOutcome = Literal["written", "present", "failed", "policy_skip"]


@dataclass
class GmailImportStats:
    scanned: int = 0
    written: int = 0
    skipped_present: int = 0
    skipped_draft_chat: int = 0  # ingest-layer policy skips (see above)
    retried: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (id, reason)

    @property
    def failed_ids(self) -> list[str]:
        return [i for i, _ in self.failed]


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """Immutable count view handed to the #61 progress callbacks.

    Frozen, slots, ints-only: constructed fresh at every fire, so a callback
    can neither mutate the run's accounting nor hold a live reference into it
    (the snapshot is a value, not a window). Scope is the EXPORT LOOP only:
    fetch_settings bootstrap retries live in the caller's own GmailImportStats
    and are the adapter's to merge (deepseek r2: #62 must not read retried
    as a run total). Counts ONLY — message ids ride on_message, never here
    — and both are metadata, never content (invariant #5), so the engine
    stays I/O-free (CODING-STANDARDS: pure core, bounded adapters print)."""

    scanned: int
    written: int
    skipped_present: int
    skipped_draft_chat: int
    retried: int
    failed_count: int


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
    on_page: Callable[[ProgressSnapshot], None] | None = None,
    on_message: Callable[[str, ImportOutcome], None] | None = None,
) -> GmailImportStats:
    """Import the query's messages into the store. Returns counts + failures.

    Side effect beyond the record log (ADR-0008, additive): births the Sync
    Cursor when — and only when — the corpus has none. Advancement is Label
    Sync's exclusive act; see _birth_cursor.

    Progress callbacks (#61, opt-in): ``on_page`` receives a fresh
    ProgressSnapshot at each completed page and on every early-return path
    (max_messages bound hit mid-page; the ``<page>`` sentinel fetch-failure
    return) so operators see final counts before the summary — EXCEPT the
    ``max_messages <= 0`` no-op, which fires nothing. ``on_message`` receives
    (message id, outcome) per stub once the outcome is decided, from the
    ImportOutcome vocabulary: 'written', 'present' (skip-present, either the
    has()-pre-check or a concurrent put), 'policy_skip' (DRAFT/CHAT ingest
    layer), 'failed'.

    Callbacks MUST NOT RAISE — the engine deliberately does NOT catch their
    exceptions (fail-loud posture, pinned by
    test_raising_progress_callback_propagates): a broken progress renderer
    is a defect to surface, not noise to swallow, and resume is structural
    (the id-set cursor) so a stopped run is simply re-run. Satisfying that
    contract is the CALLER's duty: the CLI adapter closes its stderr channel
    permanently on the first BrokenPipeError so its renderers never raise
    (#61 fix round, deepseek-measured EPIPE abort)."""
    sleeper = sleeper or time.sleep
    clock = clock or (lambda: datetime.now(UTC))
    if max_attempts < 1:
        # a zero-attempt loop would be a silent no-op import (deepseek:
        # the 'exhausted' return below was its only executed line)
        raise ValueError("max_attempts must be >= 1")
    stats = GmailImportStats()
    if max_messages is not None and max_messages <= 0:
        return stats  # explicit no-op, never "process one anyway"

    def _fire_page() -> None:
        # fresh value object per fire — no mutable-state leak to callbacks
        if on_page is not None:
            on_page(
                ProgressSnapshot(
                    scanned=stats.scanned,
                    written=stats.written,
                    skipped_present=stats.skipped_present,
                    skipped_draft_chat=stats.skipped_draft_chat,
                    retried=stats.retried,
                    failed_count=len(stats.failed),
                )
            )

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
            _fire_page()  # early return still reports final counts (#61)
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
                outcome = "present"  # same class as a concurrent put=False
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
                elif outcome == "policy_skip":
                    stats.skipped_draft_chat += 1
            # per-stub event AFTER the outcome is decided (#61); the id is
            # normalized like the store key so a -vv line and the corpus
            # index always agree (uppercase wire stubs would otherwise
            # render two spellings of one message); metadata, never content
            # (invariant #5)
            if on_message is not None:
                on_message(stub.id.lower(), outcome)
            # bound checked on EVERY path (deepseek engine pass: continue
            # used to bypass it, scanning past max_messages on resume runs)
            if max_messages is not None and stats.scanned >= max_messages:
                _fire_page()  # bound return still reports final counts (#61)
                return stats
        _fire_page()  # completed page: one snapshot per page (#61)
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
    non-positive seed is a wire lie the ledger refuses. When no usable seed
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
) -> ImportOutcome:
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
    if EXCLUDED_CORPUS_LABEL_IDS.intersection(msg.label_ids):
        # policy skip on the RAW wire labelIds (before parse/normalize):
        # counted by the caller, never stored, never a failure — the
        # exclusion is a decision, not a defect
        return "policy_skip"
    try:
        parsed = parse_mime(decode_gmail_raw(msg.raw_b64url))
    except ValueError as e:
        # decode_gmail_raw is intentionally loud on corrupt raw (documented;
        # binascii.Error and UnicodeEncodeError are ValueError subclasses)
        stats.failed.append((message_id, f"raw decode: {type(e).__name__}"))
        return "failed"
    # Date Provenance (GLOSSARY): the stamp must say where the date ACTUALLY
    # came from. Server internalDate is authoritative; the MIME Date header
    # is SENDER-CLAIMED — a forged/future header can move a message across
    # the temporal split (invariant #7), so the fallback is stamped
    # "mime_date", never blended. (A present-but-unconvertible internalDate
    # falls through to the header and is stamped accordingly.) No header =>
    # no guess.
    server_date = _ms_to_utc(msg.internal_date_ms)
    provenance: DateProvenance
    if server_date is not None:
        date = server_date
        provenance = "internal_date"
    elif parsed.date is not None:
        date = utc_datetime(parsed.date)
        provenance = "mime_date"
    else:
        # no authoritative timestamp: fail here, never stored — a record's
        # provenance is always one of the two real sources
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
            date_provenance=provenance,
            thread_id=msg.thread_id,
        )
    except ValueError as e:
        # a wire value violating the record contract fails THIS message
        # only — one impossible id must not abort the whole import
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
