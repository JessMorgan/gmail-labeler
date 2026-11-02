"""The ``apply`` bulk path (ticket #26; GLOSSARY: Application, Backfill).

``apply --since <when> [--apply]`` is the ONLY bulk path: without
``--apply`` a dry preview scores the historical window and writes
NOTHING; with ``--apply`` it backfills graduated labels over that mail —
always an explicit human decision, never automatic (the daemon never
invokes anything here: the AC's "never automatic" is a grep pin on
daemon.py, which carries no reference to this module).

ONE CORE, NO SECOND IMPLEMENTATION (AC2 — the whole ticket's spine):
every band/tier/skip-present decision is the daemon's, imported from
``applications.plan_message_write`` (composition over ``tiers.would_apply``)
— apply calls it, never mirrors it. The batched predict is the daemon's
positive-conformance seam (``daemon.predict_conformant``), imported
whole. The startup is ``cli._serve_startup`` (the gates-first chain the
daemon and apply share). A re-derivation of ANY of these is the exact
bug class this layout exists to prevent; the parity is mutation-pinned
in tests/test_apply.py (one daemon tick + one apply batch over identical
mail land IDENTICAL plans through the SAME function object).

THE BATCH UNIT is one ``list_messages`` page (the wire's paging spine,
not the history delta — apply has no cursor): per batch ONE ``clock()``
read stamps every row (the #24 homogeneity contract, reused at the call
site), and the write-first ordering is IDENTICAL to the daemon — the
Gmail ``apply_labels`` lands FIRST, then the batch's ONE DB transaction
(``commit_applications``) carries its rows. COMMIT = the batch is done.

NO CURSOR, AND THAT IS THE POINT (invariant #3 belongs to the daemon
alone): apply never reads or writes ``daemon_state`` (the store pin: the
state-table snapshot is byte-identical across a whole backfill). The
safety consequence is the re-run shape, not a held cursor:
skip-present (invariant #2) makes a re-run over the same window
idempotent — a label that already landed reads present, composes an
EMPTY plan, and earns no second write and no duplicate row. A crash
between a Gmail write and the commit leaves the write standing with its
audit row UNCOMMITTED, and the re-run does NOT re-add it: the label is
now present, so skip-present composes an empty plan and no row is
written (the documented face of the same design the daemon carries —
"the rows are not re-added"). The audit truth is the Applications table
PLUS the mailbox, never a guaranteed 1:1 (the daemon's posture,
inherited).

THE ARMS (the gmail_errors taxonomy, checked not string-matched):
GmailAuthError propagates (human-gated — the CLI's exit-4 mapping); a
404 on a message read or a write is a VANISHED message (a backfill over
years WILL hit deleted mail) — counted SKIPPED, never fatal, delta-free
(there is no cursor to hold); any OTHER per-message write error HOLDS
nothing (there is no cursor) — the message stays un-labeled and un-
rowed, a counted warning, and a RE-RUN picks it up (not in the DB, not
on the mail — the distinguishing shape, pinned); a model shape defect
(``predict_conformant`` raising) or a page-read failure ABORTS the run
(the CLI's exit-5 mapping — the shadow prediction-failure precedent: a
systematic defect is a bug, not a per-message skip).

GRACEFUL SHUTDOWN: ``should_stop`` (the daemon's shared handler helper
— no third copy) is consulted between messages and between pages; once
seen, the current batch still runs to its commit point, then the loop
returns (exit 0). Mid-page SIGTERM therefore completes the page it was
on — the AC4 interrupt + re-run idempotency matrix lives in
tests/test_apply.py.

BODIES NEVER (invariant #5): the loop's inputs are ids/labels/scores;
the fetched body feeds ONLY the in-memory preprocess_text call; every
preview line and log line carries ids, label NAMES, bands, tiers, and
scores — never a subject, snippet, or body (the shadow-report vocabulary
discipline; caplog-pinned).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from .appdb import AppDB
from .applications import ApplicationRow
from .corpus_record import preprocess_text
from .daemon import ServingContext, compose_plan, predict_conformant, present_names
from .gmail_client import GmailClient, MessageStub
from .gmail_errors import GmailAuthError, GmailError, is_404
from .mime_extract import decode_gmail_raw, parse_mime
from .model import Model

__all__ = ["ApplyOutcome", "run_apply"]

log = logging.getLogger("gmail_labeler.apply")


@dataclass(frozen=True, slots=True)
class ApplyOutcome:
    """The aggregate counters of one apply run (the operator-visible
    vocabulary — the SAME count discipline as the daemon's per-delta
    line, summed over batches). ``messages`` is the count of messages
    FETCHED and scored (after the 404 skips and the per-page dedupe — the
    denominator the other counters run against); ``written`` is the Gmail
    writes that landed (ZERO in a preview — the AC1 fact, not a display
    convention); ``skipped`` counts vanished (404) / undecodable messages;
    ``warnings`` counts fail-safe composes and per-message READ-or-WRITE
    errors held off the wire with no cursor to retry them (the re-run
    picks each up); ``rows`` is the Application rows committed (ZERO in a
    preview); ``batches`` is the pages processed."""

    messages: int
    written: int
    skipped: int
    warnings: int
    rows: int
    batches: int


def _log_preview(message_id: str, rows: Sequence[ApplicationRow]) -> None:
    """One preview line: id + would-be label NAMES + band/tier + scores
    (the shadow-report vocabulary — no subjects, no snippets, bodies
    never; invariant #5). The rows are the composed Application rows, so
    the preview shows EXACTLY what the write path would record (one
    core, one output)."""
    parts = ", ".join(
        f"{r.label}={r.score:.3f}({r.band}/{r.tier})"
        for r in sorted(rows, key=lambda r: r.label)
    )
    log.info("apply: %s would-apply %s", message_id, parts)


def _run_batch(
    client: GmailClient,
    model: Model,
    ctx: ServingContext,
    stubs: Sequence[MessageStub],
    *,
    applied_at: datetime,
    should_stop: Callable[[], bool],
    dry_run: bool,
) -> tuple[list[ApplicationRow], int, int, int, int, bool]:
    """One page: fetch, predict, compose, write (unless dry). Returns
    ``(rows, messages, written, skipped, warnings, stop)`` — ``rows`` are
    the composed Application rows the caller commits (EMPTY in a preview,
    which never records); the commit is the CALLER's (one transaction per
    batch). ``applied_at`` is the caller's ONE clock read for this batch
    (the homogeneity contract, stamped at the call site). GmailAuthError
    and a page-level ``list_messages`` failure propagate (the CLI's arms);
    a 404 message (read OR write) is a counted skip; any OTHER per-message
    read or write error is a counted WARNING that holds nothing and
    CONTINUES — the message stays un-labeled and un-rowed and a RE-RUN
    picks it up (the cursorless path's safe shape: the daemon's
    hold-the-cursor retry has no analog here, so a blip must not abort a
    whole backfill)."""
    fetched: list[tuple[str, frozenset[str], str]] = []  # (id, present, text)
    seen: set[str] = set()  # per-page dedupe (the daemon's _message_ids shape)
    skipped = 0
    warnings = 0
    stop = False
    for stub in stubs:
        if stub.id in seen:
            # a message served twice in one page (a wire/replay overlap):
            # process it ONCE. Without this the second pass re-fetches,
            # re-composes an IDENTICAL row, and commit_applications refuses
            # the within-batch (message_id, label) DUPLICATE with exit 5 —
            # AFTER the first pass's Gmail write already LANDED (the
            # review's executed probe: labels on the mail, no audit row).
            continue
        seen.add(stub.id)
        try:
            raw = client.get_message(stub.id)
        except GmailAuthError:
            raise
        except GmailError as e:
            if is_404(e):
                # the message is gone (deleted since the export — a
                # backfill over years WILL hit this): counted skip, the
                # run continues (there is no cursor to wedge)
                skipped += 1
                log.warning("apply: skipped message %s: %s", stub.id, e)
                continue
            # a non-404 read error is the SAME safe shape as a write blip:
            # counted warning, continue — the message stays un-labeled and
            # un-rowed and a re-run picks it up (there is no cursor to
            # hold, so aborting a whole multi-year backfill on one blip is
            # the wrong posture the daemon's cursor-hold rationale would
            # have imported)
            warnings += 1
            log.warning(
                "apply: read failed for %s (no cursor held — a re-run picks it up): %s",
                stub.id,
                e,
            )
            continue
        try:
            data = decode_gmail_raw(raw.raw_b64url)
        except ValueError as e:  # UnicodeEncodeError IS a ValueError
            skipped += 1
            log.warning(
                "apply: skipped message %s: raw undecodable (%s)",
                stub.id,
                type(e).__name__,
            )
            continue
        parsed = parse_mime(data)  # never raises (the mime contract)
        present = present_names(ctx, raw.label_ids)  # the SHARED reader
        # the body feeds ONLY this in-memory preprocess (invariant #5)
        fetched.append(
            (stub.id, present, preprocess_text(parsed.subject, parsed.body_text))
        )
        if should_stop():
            stop = True  # the batch still runs to its commit point

    # the batched predict — the daemon's SHARED positive-conformance
    # seam (imported, never re-implemented); a shape defect raises and
    # aborts the run (a systematic bug, not a per-message skip)
    vocab, score_rows = predict_conformant(model, [text for _, _, text in fetched])

    batch_rows: list[ApplicationRow] = []
    written = 0
    for (message_id, present, _text), score_row in zip(
        fetched, score_rows, strict=True
    ):
        scores = dict(zip(vocab, score_row, strict=True))
        # the SHARED composition call-shape (compose_plan → the one
        # plan_message_write core the daemon also rides — AC2, one core)
        plan = compose_plan(
            ctx,
            message_id,
            scores=scores,
            present=present,
            applied_at=applied_at,
            source="apply",
        )
        if plan.warning is not None:
            # the fail-safe net (invariant #1): NO write for this message
            warnings += 1
            log.warning("apply: fail-safe %s", plan.warning)
            continue
        if not plan.adds:
            # the legitimate empty (skip-present / no band): no API call,
            # no row, no warning (the two empty states stay distinguishable)
            continue
        if dry_run:
            # preview: the composed plan is REPORTED, never written (AC1)
            _log_preview(message_id, plan.rows)
            continue
        try:
            client.apply_labels(message_id, plan.adds)
        except GmailAuthError:
            raise
        except GmailError as e:
            if is_404(e):
                # vanished between fetch and write: no write happened →
                # no rows (rows record Applications that DID land);
                # counted SKIPPED only (the daemon's F1 nit, inherited)
                skipped += 1
                log.warning(
                    "apply: skipped message %s: apply refused (%s)", message_id, e
                )
                continue
            # a non-404 write error HOLDS NOTHING (there is no cursor):
            # the message stays un-labeled and un-rowed, a counted
            # warning, and a RE-RUN picks it up — the distinguishing
            # shape vs the daemon's held-cursor retry
            warnings += 1
            log.warning(
                "apply: write failed for %s (no cursor held — a re-run "
                "picks it up): %s",
                message_id,
                e,
            )
            continue
        written += 1
        batch_rows.extend(plan.rows)

    return batch_rows, len(fetched), written, skipped, warnings, stop


def run_apply(
    *,
    client: GmailClient,
    db: AppDB,
    model: Model,
    ctx: ServingContext,
    query: str,
    clock: Callable[[], datetime],
    should_stop: Callable[[], bool],
    dry_run: bool,
) -> ApplyOutcome:
    """The apply batch loop (module docstring for the full contract).
    Pages ``list_messages`` (the wire pageToken chain) and processes one
    page per batch: fetch → predict → compose → (write → commit). Fully
    injected — tests never real-poll, never real-write. Only
    ``db.commit_applications`` is called, and never in a preview.
    Returns the aggregate ``ApplyOutcome``; GmailAuthError propagates
    (exit 4), a model/page defect propagates (exit 5)."""
    total_messages = total_written = total_skipped = total_warnings = total_rows = 0
    batches = 0
    token: str | None = None
    while True:
        # a page-read failure propagates (GmailAuthError → exit 4, any
        # other GmailError → exit 5): apply has no cursor to resume
        # from, so the honest posture is a loud stop — the re-run
        # re-pages from the top and skip-present makes it idempotent
        page = client.list_messages(query, token)
        # ONE clock read per batch (the #24 homogeneity contract at the
        # call site — every row this batch commits carries THIS stamp)
        applied_at = clock()
        (
            batch_rows,
            messages,
            written,
            skipped,
            warnings,
            stop,
        ) = _run_batch(
            client,
            model,
            ctx,
            page.stubs,
            applied_at=applied_at,
            should_stop=should_stop,
            dry_run=dry_run,
        )
        batches += 1
        total_messages += messages
        total_written += written
        total_skipped += skipped
        total_warnings += warnings
        if batch_rows:
            # the batch's commit point (backfill only — a preview never
            # reaches here with rows: it records nothing). The empty
            # batch is NOT committed (commit_applications refuses it;
            # an empty plan is no Application)
            total_rows += db.commit_applications(batch_rows)
        log.info(
            "apply: batch handled page=%d messages=%d written=%d skipped=%d "
            "warnings=%d dry_run=%s",
            batches,
            messages,
            written,
            skipped,
            warnings,
            dry_run,
        )
        # the between-PAGES sample (the daemon's shape): a flag that
        # flips after this batch's message loop but before the next page
        # must not pay for another page of Gmail writes — the current
        # batch already reached its commit point. `stop` (the loop's own
        # sample) is subsumed here: the flag is a sticky Event, so a stop
        # seen mid-batch means should_stop() is True now too
        if stop or should_stop():
            break
        token = page.next_page_token
        if token is None:
            break
    return ApplyOutcome(
        messages=total_messages,
        written=total_written,
        skipped=total_skipped,
        warnings=total_warnings,
        rows=total_rows,
        batches=batches,
    )
