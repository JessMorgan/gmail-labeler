"""Resumable corpus export (GLOSSARY: Corpus; source mode ``api``).

Pulls messages through the GmailClient Protocol, parses via the single shared
MIME path (invariant #4), and writes CorpusRecords through CorpusStore.
Resume is *structural*: the store's id set is the cursor (skip-present is the
same property that makes crash-retry harmless — invariant #2's corpus twin),
so export can die at any moment and simply be re-run. Retry policy is driven
by the typed taxonomy only (transient→backoff-retry, rate_limited→honored
backoff, permanent→record-and-continue: one unreadable message must never
abort the whole import). Messages whose timestamp cannot be
established from either source (API internalDate or Date header) are FAILED,
not fabricated — the temporal split (invariant #7) starts honest.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from .corpus_record import (
    CorpusRecord,
    DateProvenance,
    body_hash,
    utc_datetime,
)
from .corpus_store import CorpusError, CorpusStore
from .corpus_store import _fsync_dir as _fsync_dir
from .gmail_client import FilterInfo, GmailClient
from .gmail_errors import BACKOFF, RETRY, GmailAuthError, GmailError
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
                if isinstance(e, GmailAuthError):
                    # every later call would fail identically — the fix is
                    # human, so surface it typed (CLI exit 4), never swallow
                    raise
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


SIDECAR_FILTERS = "filters.json"
SIDECAR_LABELS = "labels.json"  # id->name snapshot: filters speak ids, the
# corpus speaks names — attribution (invariant #6) needs the bridge offline


def fetch_settings(
    client: GmailClient,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    sleeper: Callable[[float], None] | None = None,
) -> tuple[dict[str, str], list[FilterInfo], ExportStats]:
    """Bootstrap reads (label map + filter snapshot) share the message
    loop's retry taxonomy: one transient blip on settings must not kill a
    healthy export. Exhaustion RAISES — attribution (invariant #6) is never
    silently skipped; the caller exits non-zero."""
    sleeper = sleeper or time.sleep
    stats = ExportStats()  # ride along on the return: retried= stays honest
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
    _write_snapshot_atomic(store, SIDECAR_FILTERS, payload)


def write_labels_snapshot(store: CorpusStore, id_to_name: Mapping[str, str]) -> None:
    """labels.json: the id->name bridge (same offline-attribution purpose)."""
    payload = json.dumps(dict(sorted(id_to_name.items())), indent=0, sort_keys=True)
    _write_snapshot_atomic(store, SIDECAR_LABELS, payload)


def _write_snapshot_atomic(store: CorpusStore, name: str, payload: str) -> None:
    """Created 0600 at the fd (never 0644-then-chmod), fsynced, atomically
    replaced; any OS failure is a CorpusError (CLI's typed exit-5 surface)."""
    tmp = store.root / (name + ".tmp")
    try:
        if not store.root.is_dir():
            # leaf only: intermediate ancestors follow umask, same
            # documented semantics as the token store
            store.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(store.root, 0o700)  # secret material from the FIRST byte
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)  # tighten a pre-existing tmp (O_CREAT trap)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fd = -1  # owned by the file object now
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            with contextlib.suppress(OSError):  # no partial-tmp litter
                tmp.unlink(missing_ok=True)
            raise
        finally:
            if fd >= 0:  # fchmod/write failure would otherwise leak the fd
                os.close(fd)
        os.replace(tmp, store.root / name)
        # a FAILED os.replace deliberately leaves the complete, 0600 tmp:
        # the next run re-writes it in place (O_TRUNC) and renames it away
        # — the same never-read-a-partial-state guarantee as the corpus
        # index, by rewrite-and-rename rather than tail repair
    except OSError as e:
        raise CorpusError(
            f"corpus snapshot {name!r} unwritable ({type(e).__name__})"
        ) from None
    with contextlib.suppress(OSError):  # rename durability, best-effort
        _fsync_dir(store.root)
