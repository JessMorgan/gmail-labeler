"""The daemon poll loop (ticket #25; GLOSSARY: Durable Handling,
History Cursor).

THE COMMIT POINT (invariant #3 — the whole ticket's spine). Per poll:
read the persisted cursor, drain the delta (the token chain), fetch and
score each message, compose the write through ``applications.
plan_message_write`` (the plan is called, never re-derived), and per
message with a non-empty plan: ``apply_labels`` FIRST (the Gmail
write), then — after the whole delta — ONE DB transaction carrying the
delta's Application rows AND the cursor advance
(``appdb.commit_page``). COMMIT = handled.

THE SAFETY PROOF IS THE SKIP-PRESENT PIN, NOT THE TRANSACTION. The
Gmail write cannot ride the DB transaction — there is no two-phase
commit with Google — so the ordering is deliberate: write first, commit
last. A crash between the two replays the delta from the held cursor;
the replay re-composes each message's plan against the mailbox's
PRESENT labels, and every already-written label now reads present →
skip-present (invariant #2) → an EMPTY plan → NO second write and no
row. Zero double-writes, zero duplicate rows — that pin IS the safety
proof (pinned by the crash matrix in tests/test_daemon.py, including a
real ``os._exit`` kill). The documented face of the same design: ANY
delta that fails to commit — a crash, or a mid-delta write error that
holds the cursor (the F1 widening: that loss is ROUTINE, not only
crash-scoped) — leaves its landed writes' audit ROWS uncommitted (the
Gmail truth stands; the rows are not re-added). The audit truth is the
Applications table plus the mailbox, not a guaranteed 1:1.

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
any other per-page GmailError HOLDS the cursor (a warning naming the
cursor and the typed error — page-level holds increment NO counter: the
counters cover per-message events of a committed delta; next tick
retries, and a PERMANENT non-404 therefore retries forever — the
intentional v1 posture: loud hold beats silent advance, escalation to a
stop path is out of #25's scope); per-message 404s (a deleted message)
skip that message counted — the cursor must not wedge on a ghost; any
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
point (AC: mid-delta SIGTERM → the delta completes, cursor written, exit 0);
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
from collections.abc import Callable, Mapping, Sized
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


class _PredictionShapeError(ValueError):
    # SELF-AUTHORED prediction-shape defects (row count / row width):
    # the message is built from COUNTS only — structurally body-free —
    # so it is safe to log verbatim. Model-raised errors stay class-only
    # in the log (a model error may quote the scored text, which carries
    # the body — invariant #5).
    pass


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
    tolerance branch).

    The FIRST stub is the newest message BY DOCUMENTED CONTRACT: the
    List Gmail messages guide states "The returned ``messages`` objects
    are listed in reverse chronological order (newest first)"
    (https://developers.google.com/workspace/gmail/api/guides/list-messages)
    and the sync guide's full-sync step stores the historyId of "the most
    recent message (the first message in the ``list`` response)"
    (https://developers.google.com/workspace/gmail/api/guides/sync). The
    fake models that order (``documented_order=True``) and the bootstrap
    pins exercise a reversed insertion order so a first-stub regression
    changes the seed value."""
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


def _bootstrap_tick(client: GmailClient, db: AppDB) -> bool:
    """Establish the cursor via the prior-art path (no planning, no
    writes — the next tick polls). Returns stop_requested (always
    False: a bootstrap never drains a delta, so never observes a stop)."""
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
        return False
    if cursor is None:
        log.warning(
            "daemon: cannot establish history cursor (no message to read a "
            "historyId from — empty mailbox or the field absent) — retry "
            "next tick"
        )
        return False
    db.set_history_cursor(cursor)
    log.info("daemon: history cursor established at %d", cursor)
    return False


def _resync_tick(client: GmailClient, db: AppDB, dead_cursor: int) -> bool:
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
) -> bool:
    """One poll: drain the delta, write, commit. Returns
    stop_requested; GmailAuthError propagates (exit-4 arm). Every
    outcome is already operator-visible on the log (a warning at each
    hold, an info line at the commit) — the F1 round deleted the
    parallel status string that no caller consumed."""
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
            return stop
        if page.history_id is None:
            # the wire owes the mailbox's CURRENT id on every response;
            # without it the advance cannot be established — hold
            log.warning(
                "daemon: delta failed at cursor %d — cursor held, retry "
                "next tick: history response carried no historyId",
                cursor,
            )
            return stop
        new_cursor = page.history_id  # the PAGE's id, never the request's
        try:
            message_ids = _message_ids(page.records)
        except GmailPermanentError as e:
            log.warning(
                "daemon: delta failed at cursor %d — cursor held, retry next tick: %s",
                cursor,
                e,
            )
            return stop
        for message_id in message_ids:
            try:
                raw = client.get_message(message_id)
            except GmailAuthError:
                raise
            except GmailError as e:
                if isinstance(e, GmailPermanentError) and e.status == 404:
                    # the message is gone (deleted mid-window): the
                    # per-message fail-safe — skip + a counted warning
                    # naming the id and the typed error (body-free by the
                    # taxonomy, invariant #5). The cursor must NOT wedge
                    # on a ghost.
                    skipped += 1
                    log.warning("daemon: skipped message %s: %s", message_id, e)
                    continue
                log.warning(
                    "daemon: delta failed at cursor %d — cursor held, retry "
                    "next tick: %s",
                    cursor,
                    e,
                )
                return stop
            try:
                data = decode_gmail_raw(raw.raw_b64url)
            except ValueError as e:  # UnicodeEncodeError IS a ValueError
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
            raise _PredictionShapeError(
                f"the model returned {len(score_rows)} score rows for "
                f"{len(fetched)} texts — checkpoint/recipe defect"
            )
        # the row WIDTH check lives HERE (inside the guarded arm): a
        # lying checkpoint with the right count but a misaligned row
        # would otherwise raise from the compose zip OUTSIDE the net and
        # kill the loop. Divergence from shadow's one-shot posture (cli.py
        # treats a width defect as exit-1 'a bug'): the daemon is a
        # long-lived service — hold, warn, keep the cursor.
        # Sized checks FIRST (F2 MINOR 1): a flat vector served as the
        # whole batch (len == texts but rows are floats) would raise
        # TypeError from len(row) — outside the caught tuple, exit 1.
        # Every shape defect must land in _PredictionShapeError territory.
        vocab_len = len(model.vocabulary)
        if any(
            not isinstance(row, Sized) or len(row) != vocab_len for row in score_rows
        ):
            raise _PredictionShapeError(
                f"a score row does not align to the {vocab_len}-label "
                f"vocabulary (an unsized row counts as misaligned) — "
                f"checkpoint/recipe defect"
            )
    except _PredictionShapeError as e:
        # self-authored, counts-only message: safe to log verbatim
        log.warning(
            "daemon: delta failed at cursor %d — cursor held, retry next tick: %s",
            cursor,
            e,
        )
        return stop
    except (ValueError, OSError, RuntimeError) as e:
        # invariant #5 at THIS surface: a model error may quote the
        # scored TEXT (FakeModel's unprogrammed-text refusal does — and
        # the text carries the body) — the log names the exception
        # CLASS only, never the message. The held cursor is the
        # operator-visible fact.
        log.warning(
            "daemon: delta failed at cursor %d — cursor held, retry next "
            "tick: prediction failed (%s)",
            cursor,
            type(e).__name__,
        )
        return stop

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
                # that DID land); counted SKIPPED only — the F1 round's
                # nit: a ghost message is the skip arm, not a warning
                # (double-counting it inflated both counters)
                skipped += 1
                log.warning(
                    "daemon: skipped message %s: apply refused (%s)",
                    message_id,
                    e,
                )
                continue
            # ANY non-404 GmailError reaches HERE (transient, rate-
            # limited, OR permanent — the F2 pin: the label below used
            # to say 'transient/rate-limited' while a 400 'invalid label
            # id' fell through just the same). The DELTA retries: the
            # messages already written skip-present on the replay (the
            # safety proof again), and their rows are NOT committed
            # (this transaction is the only writer). A permanent error
            # therefore retries forever — intentional posture: loud
            # hold beats the silent advance that starves the mailbox;
            # escalation to a stop path is out of #25's scope
            log.warning(
                "daemon: delta failed at cursor %d — cursor held, retry next tick: %s",
                cursor,
                e,
            )
            return stop
        written += 1
        page_rows.extend(plan.rows)

    # THE durable-handling commit point: rows AND cursor in ONE
    # transaction — COMMIT = handled (invariant #3; the store-side
    # all-or-nothing is pinned in test_appdb)
    if new_cursor is None:  # unreachable: every served page set it
        # (the repo rule: a raise, not an assert — python -O strips
        # asserts; the applications.py internal-contradiction precedent)
        raise AssertionError("delta drained without a cursor value")
    try:
        db.commit_page(page_rows, new_cursor)
    except AppDBError as e:
        log.warning(
            "daemon: delta failed at cursor %d — cursor held, retry next "
            "tick: state DB: %s",
            cursor,
            e,
        )
        return stop
    log.info(
        "daemon: delta handled cursor=%d messages=%d written=%d skipped=%d warnings=%d",
        new_cursor,
        len(fetched),
        written,
        skipped,
        warnings,
    )
    return stop


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
        stop_requested = _run_tick(client, db, model, ctx, clock, should_stop)
        if stop_requested or should_stop():
            return 0
        sleep(poll_seconds)
