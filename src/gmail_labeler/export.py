"""Resumable corpus export (GLOSSARY: Corpus; source mode ``api``).

Pulls messages through the GmailClient Protocol, parses via the single shared
MIME path (invariant #4), and writes CorpusRecords through CorpusStore.
Resume is *structural*: the store's id set is the cursor (skip-present is the
same property that makes crash-retry harmless — invariant #2's corpus twin),
so export can die at any moment and simply be re-run. Retry policy is driven
by the typed taxonomy only (transient→backoff-retry, rate_limited→honored
backoff, permanent→record-and-continue: one unreadable message must never
abort a 394k-message import). Messages whose timestamp cannot be
established from either source (API internalDate or Date header) are FAILED,
not fabricated — the temporal split (invariant #7) starts honest.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from .corpus_record import CorpusRecord, body_hash, utc_datetime
from .corpus_store import CorpusStore
from .gmail_client import GmailClient
from .gmail_errors import BACKOFF, RETRY, GmailError
from .labels import normalize_label_names
from .mime_extract import decode_gmail_raw, parse_mime

log = logging.getLogger("gmail_labeler.export")

DEFAULT_QUERY = "in:anywhere -in:spam -in:trash"
MAX_ATTEMPTS = 5
_BASE_BACKOFF_S = 1.0


@dataclass
class ExportStats:
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


def export_corpus(
    client: GmailClient,
    store: CorpusStore,
    *,
    id_to_name: Mapping[str, str],
    query: str = DEFAULT_QUERY,
    max_messages: int | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    sleeper: Callable[[float], None] | None = None,
) -> ExportStats:
    """Export the query's messages into the store. Returns counts + failures."""
    sleeper = sleeper or time.sleep
    if max_attempts < 1:
        # a zero-attempt loop would be a silent no-op export (deepseek:
        # the 'exhausted' return below was its only executed line)
        raise ValueError("max_attempts must be >= 1")
    stats = ExportStats()
    if max_messages is not None and max_messages <= 0:
        return stats  # explicit no-op, never "process one anyway"
    page_token: str | None = None
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
        page_token = page.next_page_token
        for stub in page.stubs:
            stats.scanned += 1
            if store.has(stub.id.lower()):  # ids are stored lowercased
                stats.skipped_present += 1
            else:
                outcome = _export_one(
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


def _retrying[T](
    fn: Callable[[], T],
    *,
    max_attempts: int,
    stats: ExportStats,
    sleeper: Callable[[float], None],
    what: str,
) -> T | None:
    for attempt in range(max_attempts):
        try:
            return fn()
        except GmailError as e:
            if e.kind in (RETRY, BACKOFF):
                stats.retried += 1
                if attempt == max_attempts - 1:
                    log.warning("export: %s retries exhausted (%s)", what, e.kind)
                    return None
                delay = getattr(e, "retry_after_s", None)
                if delay is None:  # an explicit server 0.0 stays 0.0
                    delay = _BASE_BACKOFF_S * (2**attempt)
                # 60 s CEILING on server hints: a dailyLimitExceeded-scale
                # hint is honest-but-bounded here per message; a global
                # stop-vs-backoff policy is a daemon-phase decision (the
                # ceiling keeps a poisoned hint from sleeping an hour)
                sleeper(min(delay, 60.0))
            else:
                log.warning("export: %s permanent error: %s", what, e)
                return None
    return None


def _export_one(
    client: GmailClient,
    store: CorpusStore,
    message_id: str,
    id_to_name: Mapping[str, str],
    *,
    max_attempts: int,
    stats: ExportStats,
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
