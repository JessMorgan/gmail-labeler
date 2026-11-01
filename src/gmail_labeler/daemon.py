"""The daemon poll loop (ticket #25; GLOSSARY: Durable Handling,
History Cursor).

THE COMMIT POINT (invariant #3 — the whole ticket's spine). Per poll:
read the persisted cursor, drain the delta (the token chain), fetch and
score each message, compose the write through ``applications.
plan_message_write`` (the plan is called, never re-derived), and per
message with a non-empty plan: ``apply_labels`` FIRST (the Gmail
write), then — after the whole delta — ONE DB transaction carrying the
page's Application rows AND the cursor advance
(``appdb.commit_page``). COMMIT = handled.

THE SAFETY PROOF IS THE SKIP-PRESENT PIN, NOT THE TRANSACTION. The
Gmail write cannot ride the DB transaction — there is no two-phase
commit with Google — so the ordering is deliberate: write first, commit
last. A crash between the two replays the page from the held cursor;
the replay re-composes each message's plan against the mailbox's
PRESENT labels, and every already-written label now reads present →
skip-present (invariant #2) → an EMPTY plan → NO second write and no
row. Zero double-writes, zero duplicate rows — that pin IS the safety
proof (pinned by the crash matrix in tests/test_daemon.py, including a
real ``os._exit`` kill). The documented face of the same design: a
crash after a write but before the commit leaves that write's audit
ROW uncommitted (the Gmail truth stands; the row is not re-added). The
audit truth is the Applications table plus the mailbox, not a
guaranteed 1:1.

The cursor takes the id from the delta's FINAL response, never the
request's — every page's historyId is the mailbox's CURRENT id (the
documented wire trap, gmail_client.HistoryPage).

THE ARMS (the gmail_errors taxonomy, checked not string-matched):
GmailAuthError propagates (human-gated — the CLI's exit-4 mapping);
a 404 on the history call (GmailResyncError or the fake's typed
GmailPermanentError(status=404) — the taxonomy's documented subclass
relationship) means the cursor fell outside the retained window:
re-bootstrap via the export prior-art path (below), named warning, the
gap mail NOT processed (the operator reviews it with shadow/apply);
any other per-page GmailError HOLDS the cursor (counted id-only
warning, next tick retries); per-message 404s (a deleted message) skip
that message counted — the cursor must not wedge on a ghost; any
application-path error means NO write for that message (the plan's
fail-safe net, invariant #1).

THE CURSOR BOOTSTRAP (the export prior-art mechanism, REUSED): the
Gmail sync guide's full-sync step — ``list_messages`` (unfiltered: the
most recent message first), ``get_message`` on that first stub, and
the response's per-message ``historyId`` becomes the starting cursor.
Export's read surface is exactly this pair; the adapter surfaces the
Message.historyId field (RawMessage.history_id) for the purpose. An
empty mailbox or an absent field holds the bootstrap (named warning,
retry next tick) — never a guessed cursor.

GRACEFUL SHUTDOWN: ``should_stop`` is consulted between messages and
between pages; once seen, the current delta still runs to its commit
point (AC: mid-page SIGTERM → page completes, cursor written, exit 0);
mid-sleep SIGTERM exits immediately with the cursor untouched (the
CLI's sleep is the interruptible one — this module never sleeps by
itself).

BODIES NEVER (invariant #5): the loop's inputs are ids/labels/scores;
the fetched body feeds ONLY the in-memory preprocess_text call — every
log line carries ids, counts, and cursor values. NOTHING here reads
the clock or sleeps by itself: ``clock``/``sleep`` are injected (the
cli._now seam doctrine, #23/#24).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from .appdb import AppDB, AppDBError
from .applications import ApplicationRow, plan_message_write
from .corpus_record import preprocess_text
from .gmail_client import GmailClient
from .gmail_errors import GmailAuthError, GmailError, GmailPermanentError
from .labels import normalize_label_names
from .mime_extract import decode_gmail_raw, parse_mime
from .model import Model

__all__ = ["BOOTSTRAP_QUERY", "ServingContext", "bootstrap_cursor", "daemon_loop"]

log = logging.getLogger("gmail_labeler.daemon")

#: The bootstrap list query: UNFILTERED — the sync guide's full-sync
#: step 1 lists the mailbox and the FIRST stub is the most recent
#: message (the guide's own claim); its historyId is the cursor seed.
BOOTSTRAP_QUERY = ""


@dataclass(frozen=True, slots=True)
class ServingContext:
    """Everything the loop serves with, resolved ONCE at startup by the
    CLI (identity gates first, then serve — the shadow-run shape): the
    calibrated bands map, the graduated label set, the System Label
    names, the id→name bridge (the present-set reader), and the two
    name→id resolution maps the plan consumes (the tiers bridge
    contract — Managed Label names and the DEFAULT tier constants →
    Gmail label ids)."""

    bands: Mapping[str, tuple[float | None, float | None]]
    graduated: frozenset[str]
    system_labels: frozenset[str]
    id_to_name: Mapping[str, str]
    managed_label_ids: Mapping[str, str]
    tier_label_ids: Mapping[str, str]


def bootstrap_cursor(client: GmailClient) -> int | None:
    """The export prior-art cursor seed (module docstring). GmailError
    propagates (the caller maps the arms); ``None`` = cannot establish
    (empty mailbox, or the newest message's historyId absent — the
    tolerance branch)."""
    page = client.list_messages(BOOTSTRAP_QUERY)
    if not page.stubs:
        return None
    return client.get_message(page.stubs[0].id).history_id


def _message_ids(records: tuple[dict[str, object], ...]) -> list[str]:
    """The message-event ids of one page's history records, in wire
    order, deduplicated (a message touched twice in one delta is
    processed once — the present-set read makes the later event's state
    authoritative either way). Records carrying only labels/threadThreads
    have no message events and pass through; a MALFORMED messages list
    is a lying wire — loud permanent (the adapter's shape doctrine)."""
    ids: list[str] = []
    for record in records:
        events = record.get("messages")
        if events is None:
            continue
        if not isinstance(events, list) or not all(
            isinstance(event, dict) for event in events
        ):
            raise GmailPermanentError("history record 'messages' not a list of objects")
        for event in events:
            message_id = event.get("id")
            if not isinstance(message_id, str) or not message_id:
                raise GmailPermanentError("history message event without a message id")
            if message_id not in ids:
                ids.append(message_id)
    return ids


def _bootstrap_tick(client: GmailClient, db: AppDB) -> tuple[str, bool]:
    """Establish the cursor via the prior-art path (no planning, no
    writes — the next tick polls). Returns (status, stop_requested)."""
    try:
        cursor = bootstrap_cursor(client)
    except GmailAuthError:
        raise
    except GmailError as e:
        log.warning(
            "daemon: cannot establish history cursor (bootstrap read failed: "
            "%s) — retry next tick",
            e,
        )
        return "bootstrap-held", False
    if cursor is None:
        log.warning(
            "daemon: cannot establish history cursor (no message to read a "
            "historyId from — empty mailbox or the field absent) — retry "
            "next tick"
        )
        return "bootstrap-held", False
    db.set_history_cursor(cursor)
    log.info("daemon: history cursor established at %d", cursor)
    return "bootstrapped", False


def _resync_tick(client: GmailClient, db: AppDB, dead_cursor: int) -> tuple[str, bool]:
    """The 404 arm: the cursor fell outside the retained window (the
    sync guide's documented expiry). Re-bootstrap via the prior-art
    path; the DEAD cursor stays durable until a fresh one is ESTABLISHED
    (never a fake value written over it), and the gap is named — the
    mail in it was NOT processed."""
    log.warning(
        "daemon: history cursor %d expired (history 404) — re-bootstrapping "
        "via the full-sync prior art; mail that arrived since the last "
        "handled cursor was NOT processed (review it with shadow/apply)",
        dead_cursor,
    )
    return _bootstrap_tick(client, db)


def _run_tick(
    client: GmailClient,
    db: AppDB,
    model: Model,
    ctx: ServingContext,
    clock: Callable[[], datetime],
    should_stop: Callable[[], bool],
) -> tuple[str, bool]:
    """One poll: drain the delta, write, commit. Returns
    (status, stop_requested); GmailAuthError propagates (exit-4 arm)."""
    cursor = db.get_history_cursor()
    if cursor is None:
        return _bootstrap_tick(client, db)
    stop = False
    # ONE clock read per handled delta (the #24 batch-homogeneity
    # contract: every row this delta commits carries THIS stamp — the
    # daemon call site, pinned by a ticking clock)
    applied_at = clock()
    fetched: list[tuple[str, frozenset[str], str]] = []  # (id, present, text)
    skipped = 0
    token: str | None = None
    new_cursor: int | None = None
    while True:
        try:
            page = client.history(cursor, token)
        except GmailError as e:
            if isinstance(e, GmailAuthError):
                raise  # human-gated, never a page-hold
            if isinstance(e, GmailPermanentError) and e.status == 404:
                # the expiry arm PER THE TAXONOMY: GmailResyncError IS
                # a GmailPermanentError with status=404 (documented
                # subclass) — the fake's raw 404 and the adapter's
                # reading route to the SAME handling
                return _resync_tick(client, db, cursor)
            log.warning(
                "daemon: delta failed at cursor %d — cursor held, retry next tick: %s",
                cursor,
                e,
            )
            return "held", stop
        if page.history_id is None:
            # the wire owes the mailbox's CURRENT id on every response;
            # without it the advance cannot be established — hold
            log.warning(
                "daemon: delta failed at cursor %d — cursor held, retry "
                "next tick: history response carried no historyId",
                cursor,
            )
            return "held", stop
        new_cursor = page.history_id  # the PAGE's id, never the request's
        try:
            message_ids = _message_ids(page.records)
        except GmailPermanentError as e:
            log.warning(
                "daemon: delta failed at cursor %d — cursor held, retry next tick: %s",
                cursor,
                e,
            )
            return "held", stop
        for message_id in message_ids:
            try:
                raw = client.get_message(message_id)
            except GmailAuthError:
                raise
            except GmailError as e:
                if isinstance(e, GmailPermanentError) and e.status == 404:
                    # the message is gone (deleted mid-window): the
                    # per-message fail-safe — skip + counted id-only
                    # warning. The cursor must NOT wedge on a ghost.
                    skipped += 1
                    log.warning("daemon: skipped message %s: %s", message_id, e)
                    continue
                log.warning(
                    "daemon: delta failed at cursor %d — cursor held, retry "
                    "next tick: %s",
                    cursor,
                    e,
                )
                return "held", stop
            try:
                data = decode_gmail_raw(raw.raw_b64url)
            except (ValueError, UnicodeEncodeError) as e:
                # decode_gmail_raw is intentionally loud on corrupt raw
                # (the export doctrine): per-message fail-safe
                skipped += 1
                log.warning(
                    "daemon: skipped message %s: raw undecodable (%s)",
                    message_id,
                    type(e).__name__,
                )
                continue
            parsed = parse_mime(data)  # never raises (the mime contract)
            present = frozenset(normalize_label_names(raw.label_ids, ctx.id_to_name))
            # the body feeds ONLY this in-memory preprocess (invariant
            # #5 — it never reaches a log line, the DB, or the plan)
            fetched.append(
                (message_id, present, preprocess_text(parsed.subject, parsed.body_text))
            )
            if should_stop():
                stop = True  # the delta still runs to its commit point
        token = page.next_page_token
        if token is None:
            break
        if should_stop():
            stop = True  # the delta still drains to its commit point

    # the batched predict (the shadow seam: ONE call over the delta's
    # texts, vocab zip like shadow). A model defect is SYSTEMATIC —
    # hold the cursor (every retry fails identically and loudly; a
    # silent advance would label-starve the whole mailbox on a bad
    # checkpoint)
    try:
        score_rows = model.predict([text for _, _, text in fetched])
        if len(score_rows) != len(fetched):
            raise ValueError(
                f"the model returned {len(score_rows)} score rows for "
                f"{len(fetched)} texts — checkpoint/recipe defect"
            )
    except (ValueError, OSError, RuntimeError) as e:
        log.warning(
            "daemon: delta failed at cursor %d — cursor held, retry next "
            "tick: prediction failed: %s",
            cursor,
            e,
        )
        return "held", stop

    # per message: compose (the plan's fail-safe is the net), then the
    # Gmail write FIRST — see the module docstring's safety proof
    page_rows: list[ApplicationRow] = []
    written = 0
    warnings = 0
    for (message_id, present, _text), score_row in zip(
        fetched, score_rows, strict=True
    ):
        scores = dict(zip(model.vocabulary, score_row, strict=True))
        plan = plan_message_write(
            message_id,
            scores=scores,
            bands=ctx.bands,
            graduated=ctx.graduated,
            present=present,
            system_labels=ctx.system_labels,
            managed_label_ids=ctx.managed_label_ids,
            tier_label_ids=ctx.tier_label_ids,
            applied_at=applied_at,
            source="daemon",
        )
        if plan.warning is not None:
            # the fail-safe net (invariant #1): NO write for this
            # message; the warning is the #24-pinned shape verbatim
            # (ids/labels/scores only — structurally body-free)
            warnings += 1
            log.warning("daemon: fail-safe %s", plan.warning)
            continue
        if not plan.adds:
            # the legitimate empty (skip-present / no band): no API
            # call at all, no row, no warning (the two empty states are
            # distinguishable — the plan's contract)
            continue
        try:
            client.apply_labels(message_id, plan.adds)
        except GmailAuthError:
            raise
        except GmailError as e:
            if isinstance(e, GmailPermanentError) and e.status == 404:
                # the message vanished between fetch and write: no
                # write happened → no rows (rows record Applications
                # that DID land); skip counted
                warnings += 1
                skipped += 1
                log.warning(
                    "daemon: skipped message %s: apply refused (%s)",
                    message_id,
                    e,
                )
                continue
            # transient/rate-limited: the page retries — the messages
            # already written this delta skip-present on the replay
            # (the safety proof again), and their rows are NOT
            # committed (this transaction is the only writer)
            log.warning(
                "daemon: delta failed at cursor %d — cursor held, retry next tick: %s",
                cursor,
                e,
            )
            return "held", stop
        written += 1
        page_rows.extend(plan.rows)

    # THE durable-handling commit point: rows AND cursor in ONE
    # transaction — COMMIT = handled (invariant #3; the store-side
    # all-or-nothing is pinned in test_appdb)
    assert new_cursor is not None  # every served page set it (loop invariant)
    try:
        db.commit_page(page_rows, new_cursor)
    except AppDBError as e:
        log.warning(
            "daemon: delta failed at cursor %d — cursor held, retry next "
            "tick: state DB: %s",
            cursor,
            e,
        )
        return "held", stop
    log.info(
        "daemon: delta handled cursor=%d messages=%d written=%d skipped=%d warnings=%d",
        new_cursor,
        len(fetched),
        written,
        skipped,
        warnings,
    )
    return "committed", stop


def daemon_loop(
    *,
    client: GmailClient,
    db: AppDB,
    model: Model,
    ctx: ServingContext,
    poll_seconds: int,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
    should_stop: Callable[[], bool],
) -> int:
    """The poll loop (module docstring for the full contract). Fully
    injected — tests never real-sleep, never real-poll. Returns 0 on
    graceful stop; GmailAuthError propagates for the CLI's exit-4
    mapping."""
    while True:
        if should_stop():
            return 0
        _status, stop_requested = _run_tick(client, db, model, ctx, clock, should_stop)
        if stop_requested or should_stop():
            return 0
        sleep(poll_seconds)
