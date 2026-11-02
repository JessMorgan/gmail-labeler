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

CORRECTION DETECTION (ticket #27; GLOSSARY: Correction): each delta's
history records also carry ``labelsAdded``/``labelsRemoved`` events (the
documented Gmail response keys — the singular ``labelAdded``/``labelRemoved``
are only ``historyTypes[]`` request-enum values, never response keys; see
``_label_events``). They
are classified — read + local-DB only, NEVER a Gmail write — against the
applications join: only messages ALREADY present in the Applications
table are candidates (the join AC). A ``labelsRemoved`` of a label that
IS that message's Application label is a 'removed' Correction (a human
undid our work); a ``labelsAdded`` of a MANAGED-VOCABULARY (graduated,
non-System) label the daemon did NOT write (no Application row for it)
is an 'added' Correction CANDIDATE — whether it is positive training signal
depends on its attribution (only ``"human"`` is; ``"filter_covered"`` and
``"unverifiable"`` — including the crash-replay withhold below — are not).
ANYTHING else on an applied message — System Labels, non-vocabulary
labels, a removed NON-applied label, INBOX churn — records NOTHING:
noise is not correction. An 'added' Correction carries an
``attribution`` (the #22 ``simulates_filter`` mechanism over the offline
filters.json snapshot): ``"human"`` (an attested snapshot with no relevant
filter, or a verifiable one that does not match — training signal),
``"filter_covered"`` (a simulatable filter verifiably would have added it —
withheld), or ``"unverifiable"`` (authorship cannot be determined —
WITHHELD, not guessed, #22's exclusion principle). A MISSING or malformed
snapshot is UNKNOWN attribution, not "no filters": every 'added' is then
``"unverifiable"`` (withheld), and the daemon warns at startup. Only
``"human"`` adds are training signal. A 'removed' Correction is always
attributed ``"human"`` (a human removing is human signal regardless of
filters — invariant #6). A fourth UNVERIFIABLE trigger (S1 C2): when the
delta carries a ``labelsAdded`` for a label this tick's plan SKIPPED as
already-present yet no Application row exists for it, the HUMAN credit is the
crash-replay ambiguity — our own landed write whose row was lost before the
commit, indistinguishable from a human who added exactly what we would have —
so a would-be ``"human"`` add is routed to ``"unverifiable"`` (withheld), never
credited (which would poison human_added and the training fold). It overrides
ONLY the human credit: a PROVEN ``"filter_covered"`` (a filter verifiably
authored it) or an already-``"unverifiable"`` add is not ambiguous and stands.
A corrected message is NEVER written this delta (the daemon does not
fight the human's hand-fix — the landmine: detection never produces a
write for the corrected message). The correction rows ride the SAME
delta transaction as the Application rows and the cursor (one commit
point, unchanged); the store's UNIQUE (message_id, label, kind) + INSERT
OR IGNORE makes a replayed delta's re-detected correction idempotent.

BODIES NEVER (invariant #5): the loop's inputs are ids/labels/scores;
the fetched body feeds ONLY the in-memory preprocess_text call — every
log line carries ids, counts, and cursor values. NOTHING here reads
the clock or sleeps by itself: ``clock``/``sleep`` are injected (the
cli._now seam doctrine, #23/#24).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence, Sized
from dataclasses import dataclass
from datetime import datetime

from .analyze import filter_label_ids_strict, simulates_filter
from .appdb import AppDB, AppDBError, CorrectionRow
from .applications import ApplicationRow, WritePlan, plan_message_write
from .corpus_record import preprocess_text
from .gmail_client import GmailClient
from .gmail_errors import GmailAuthError, GmailError, GmailPermanentError, is_404
from .labels import normalize_label_names
from .mime_extract import decode_gmail_raw, parse_mime
from .model import Model
from .tiers import would_apply

__all__ = [
    "BOOTSTRAP_QUERY",
    "ServingContext",
    "bootstrap_cursor",
    "compose_plan",
    "daemon_loop",
    "predict_conformant",
    "present_names",
]

log = logging.getLogger("gmail_labeler.daemon")


class _PredictionShapeError(ValueError):
    # SELF-AUTHORED prediction-shape defects (vocabulary, container,
    # row type, count, width, element type): the message is built from
    # COUNTS and TYPE NAMES
    # only — never an element VALUE (a lying score element is an
    # arbitrary string and must not reach a log line) — structurally
    # body-free, safe to log verbatim. Model-raised errors stay
    # class-only in the log (a model error may quote the scored text,
    # which carries the body — invariant #5).
    pass


def predict_conformant(
    model: Model, texts: Sequence[str]
) -> tuple[tuple[str, ...], list[tuple[float, ...]]]:
    """The ONE positive-conformance predict seam (ticket #25's F4,
    SHARED with apply #26 — the design's 'positive-conformance seam —
    REUSE, do not re-implement'): ONE ``model.predict`` call over the
    batch's texts, validated against the Model Protocol's DECLARED
    shape. Returns ``(vocab, rows)`` — the vocabulary SNAPSHOT (the
    single truth for the width check AND the caller's compose zip) and
    the materialized score rows.

    Raises ``_PredictionShapeError`` (a ValueError subclass) on a
    self-authored shape defect — the message is counts + type names
    only, never an element VALUE (invariant #5); the model's OWN
    ``ValueError``/``OSError``/``RuntimeError`` propagate untouched.
    THE CALLER OWNS THE POSTURE: the daemon HOLDS the cursor on either
    family (a systematic defect retries identically and loudly); apply
    has no cursor — the same defect is a genuine bug and aborts the
    run (the CLI maps it to exit 5, the shadow prediction-failure
    precedent).

    F4 POSITIVE PROTOCOL CONFORMANCE (the cycle-ender): every earlier
    round ADDED A TYPE NAME to a row denylist (float → Sized/Iterable →
    bytes/bytearray) and a NEW int-materializing container escaped each
    time — memoryview, array('B'), int-keyed dict and range ALL
    materialize to (0, 1) and WROTE a garbage label at ef208b7
    (executed). The vocabulary denylist likewise let an INT vocabulary
    (label ids where Label Paths belong — the mis-wired-checkpoint case
    verify_sidecar cannot catch: it hashes the sidecar against the
    EXPECTED vocabulary, model.py) sail into the per-message fail-safe:
    cursor advancing forever, zero writes. The positive form closes the
    family: model.predict's DECLARED return type IS the contract (Model
    Protocol: vocabulary: tuple[str, ...]; predict -> tuple[tuple[float,
    ...], ...]) — a row is accepted ONLY as a tuple, a vocabulary ONLY
    as tuple[str, ...]. Off-conformance is a WIRING bug: it belongs in
    the daemon's hold arm, never in the per-message fail-safe. The
    checks are now TOTAL for off-Protocol containers and rows and for
    lying lengths (positive conformance leaves no type-name to miss);
    the residuals are EXACTLY an __iter__ (container, row, or
    vocabulary) raising OUTSIDE the caught set — TypeError, ValueError,
    OSError and RuntimeError are ALL netted (F4 round-2 restored the
    loop and snapshot arms); an unbounded-yield container drained before
    the count fires; numpy-style scalar rows (numbers.Real when the
    real encoder lands)."""
    score_rows = model.predict(texts)
    # the container must be SIZED — a Model-Protocol conformance
    # refusal (the Protocol declares tuple[tuple[float, ...], ...]):
    # materialization would VERIFY an unsized container, but a
    # generator batch is already off-contract (F3 round-2: this gate's
    # original len()-protection rationale retired when the count moved
    # to the materialized rows)
    if not isinstance(score_rows, Sized):
        raise _PredictionShapeError(
            "the model returned an unsized score-row container "
            "(not a sequence) — checkpoint/recipe defect"
        )
    try:
        vocab_src = model.vocabulary  # read ONCE (F3 round-2 NIT)
    except AttributeError as e:
        raise _PredictionShapeError(
            f"the model exposes no vocabulary attribute "
            f"({type(e).__name__}) — checkpoint/recipe defect"
        ) from e
    if not isinstance(vocab_src, tuple):
        # POSITIVE guard (replaces the F3 str/bytes denylist): str,
        # bytes, memoryview, int vocabularies, lists, sets, dicts — ALL
        # refused by class name
        raise _PredictionShapeError(
            f"the model vocabulary is not a tuple[str, ...] as the Model "
            f"Protocol declares (got {type(vocab_src).__name__}; str "
            f"elements required) — checkpoint/recipe defect"
        )
    try:
        vocab = tuple(vocab_src)  # honest SNAPSHOT (F4 round-2,
        # reviewer MAJOR): the live re-read was the vocabulary twin of
        # 098be2a — a tuple SUBCLASS with a lying __len__ escaped the
        # width check at compose (exit 1), and a STATEFUL __iter__
        # passed the str-check pass, reversed at the compose pass, and
        # WROTE THE WRONG LABEL (executed probe). The snapshot is the
        # single truth for len, the element check, and the compose zip
    except TypeError as e:
        raise _PredictionShapeError(
            f"the model vocabulary is not iterable ({type(e).__name__}) "
            f"— checkpoint/recipe defect"
        ) from e
    if not all(isinstance(v, str) for v in vocab):
        raise _PredictionShapeError(
            "the model vocabulary is not a tuple[str, ...] as the "
            "Model Protocol declares (got tuple with non-str "
            "elements; str elements required) — checkpoint/recipe defect"
        )
    vocab_len = len(vocab)
    rows: list[tuple[float, ...]] = []
    try:
        for row in score_rows:  # the container iterated EXACTLY ONCE
            # (F3 round-2 double-iteration MINOR: the denylist pass
            # touched it twice — a one-shot Sized container could lie
            # about its contents between the passes)
            if not isinstance(row, tuple):
                # memoryview / array / dict / range / frozenset / set /
                # list / float — refused by class name. frozenset dies
                # here for the RIGHT reason (F3 P5): its iteration order
                # is arbitrary, so materialize-and-check could MIS-ALIGN
                # scores to labels, not just miscount them
                raise _PredictionShapeError(
                    f"a score row is not a tuple as the Model Protocol "
                    f"declares (got {type(row).__name__}) — "
                    f"checkpoint/recipe defect"
                )
            item = tuple(row)  # honest materialization: a tuple
            # SUBCLASS with a lying __len__ cannot survive the width
            # check (iteration is the authority); a raising __iter__ is
            # netted by the TypeError arm below (F4 round-2)
            if len(item) != vocab_len:
                raise _PredictionShapeError(
                    f"a score row does not align to the {vocab_len}-label "
                    f"vocabulary — checkpoint/recipe defect"
                )
            for element in item:
                # TYPE NAMES only in every message (class comment): an
                # element VALUE is an arbitrary object and never
                # interpolates
                if isinstance(element, bool) or not isinstance(element, (int, float)):
                    raise _PredictionShapeError(
                        f"a score element is not numeric "
                        f"({type(element).__name__}) — checkpoint/recipe defect"
                    )
            rows.append(item)
    except TypeError as e:
        # F4 round-2 (reviewer MAJOR): the single-pass loop and
        # tuple(row) can THEMSELVES raise TypeError (Sized-not-iterable
        # container; tuple-subclass row whose __iter__ raises) — the net
        # the F3 materialization carried is restored, so the residual
        # stays exactly what the comment says: raises OUTSIDE
        # (TypeError/ValueError/OSError/RuntimeError)
        raise _PredictionShapeError(
            f"a score row is not a tuple as the Model Protocol "
            f"declares (unmaterializable container or row: "
            f"{type(e).__name__}) — checkpoint/recipe defect"
        ) from e
    if len(rows) != len(texts):
        # the count check on the materialized rows (honest length)
        raise _PredictionShapeError(
            f"the model returned {len(rows)} score rows for "
            f"{len(texts)} texts — checkpoint/recipe defect"
        )
    return vocab, rows


#: The bootstrap list query: UNFILTERED — the sync guide's full-sync
#: step 1 lists the mailbox and the FIRST stub is the most recent
#: message (the guide's own claim); its historyId is the cursor seed.
BOOTSTRAP_QUERY = ""

#: The Gmail system labels marking mail that is NOT incoming mail the daemon
#: should classify: SENT (the operator's own outgoing), DRAFT (pre-send), and
#: CHAT (not email). `dataset export` already EXCLUDES DRAFT+CHAT from the
#: corpus (export.py _EXCLUDED_SYSTEM_LABEL_IDS); labelling them on the live
#: path would apply ai-applied/tier labels to a draft before the operator even
#: sends it — an inference target deliberately outside the training
#: distribution. A sent message arrives on the history stream as a messageAdded
#: too — MEDIUM confidence: a single practitioner report, but the product need
#: to skip our own outgoing / non-incoming mail holds regardless of which field
#: Gmail signals it with. "SELF" is a search-query term, not a label id, so the
#: system label ids are the machine-checkable signal here (see the guard).
_OUTGOING_LABEL_IDS = frozenset({"SENT", "DRAFT", "CHAT"})


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
    # the offline filter-attribution snapshot (ticket #27, invariant #6):
    # the filters.json the export path wrote, read at startup (a LOCAL
    # file read — never a Gmail call, so the daemon's exact startup-call
    # pins stand). The correction arm uses it to attribute an 'added'
    # Correction (human / filter_covered / unverifiable — see
    # _attribution). THREE states, mirroring analyze.py's filters=None
    # posture (the #22 mechanism REUSED, not forked):
    #   None  — NO usable snapshot (absent or malformed): attribution is
    #           UNKNOWN, so an 'added' is 'unverifiable' (WITHHELD, not
    #           credited to the human — #22's exclusion principle; treating
    #           an absent snapshot as "no filters" is the fail-open F1
    #           removes, one level up). The CLI warns at startup.
    #   ()    — a snapshot was read and lists NO filters: the operator's
    #           ATTESTATION that none exist, so no relevant filter claims a
    #           label and an 'added' is 'human' (positive signal). The CLI
    #           passes this EXPLICITLY (an empty parsed list) when it reads a
    #           real, empty filters.json; the daemon's correction arm is the
    #           ONLY reader.
    #   [...] — the filters; an 'added' whose relevant filter is simulatable
    #           and matches is 'filter_covered'; one whose criteria are
    #           outside {from,to,subject} is 'unverifiable' (withheld).
    # The DEFAULT is None (fail-closed): a construction that does not KNOW
    # the filters withholds every 'added' rather than crediting it to the
    # human. An attested-none snapshot is passed EXPLICITLY as () by the
    # CLI (it read a real, empty filters.json); the old ()-default was only
    # ever a suite-riding convenience and is dead now the wiring is pinned.
    filters: Sequence[Mapping[str, object]] | None = None


def present_names(ctx: ServingContext, raw_label_ids: Sequence[str]) -> frozenset[str]:
    """The ONE present-set reader (ticket #26 F1): a message's Gmail
    label ids → the normalized NAMES the plan tests skip-present against,
    System Labels dropped. The daemon's live path and the apply bulk path
    ride THIS reader — the id→name bridge is a ctx field, so a change to
    how presence is read reaches BOTH paths together (the decision core
    was always shared; this closes the plumbing mirror the review named).
    Lives in daemon.py (where ServingContext is defined) so the apply
    module imports it the same way it imports predict_conformant — never
    a re-derivation."""
    return frozenset(normalize_label_names(raw_label_ids, ctx.id_to_name))


def compose_plan(
    ctx: ServingContext,
    message_id: str,
    *,
    scores: Mapping[str, float],
    present: frozenset[str],
    applied_at: datetime,
    source: str,
) -> WritePlan:
    """The ONE composition call-shape both the daemon (#25) and the apply
    bulk path (#26) ride (ticket #26 F1): it threads the six
    ``ServingContext`` maps to ``plan_message_write`` in a SINGLE place,
    so a new ctx field reaches BOTH paths together and neither caller can
    drift a kwarg. ``plan_message_write`` is THE decision core (composition
    over ``tiers.would_apply``) — this wrapper adds NO decision, only the
    ctx→kwarg wiring; the daemon call site and the apply call site both
    reach the core through HERE. It calls the module-global
    ``plan_message_write`` (not a re-import) so the daemon suite's spy on
    ``daemon.plan_message_write`` still intercepts every plan this helper
    composes — the extraction is behavior-preserving at the seam the tests
    watch."""
    return plan_message_write(
        message_id,
        scores=scores,
        bands=ctx.bands,
        graduated=ctx.graduated,
        present=present,
        system_labels=ctx.system_labels,
        managed_label_ids=ctx.managed_label_ids,
        tier_label_ids=ctx.tier_label_ids,
        applied_at=applied_at,
        source=source,
    )


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
    """The ARRIVAL message-event ids of one page's history records, in wire
    order, deduplicated (a message touched twice in one delta is processed
    once — the present-set read makes the later event's state authoritative
    either way).

    ARRIVALS ONLY — the scoring set is ``messagesAdded[]``, NOT the generic
    ``messages[]`` field. The Gmail users.history reference describes the
    generic field as "List of messages changed in this history record" and
    explicitly recommends the specific change-type fields instead; ``
    messagesAdded[]`` is "Messages added to the mailbox" (arrivals). Reading
    the generic field re-enrolled OLD mail that merely got a label change —
    a user archive/relabel, or the daemon's OWN apply_labels echo (the push
    guide's notification-loop warning) — for scoring, which is RETROACTIVE
    LABELING (README 'scores arriving mail'; ROADMAP Never: no retroactive
    labeling except via the deliberate `apply` command). Records carrying
    only labels/threadThreads/messages[]-without-messagesAdded have no
    arrival event and pass through; a MALFORMED messagesAdded list is a
    lying wire — loud permanent (the adapter's shape doctrine). The nested
    Message id is read; a flat id is tolerated defensively (the
    _label_event_message_id precedent — a wrong shape reads as FEWER
    arrivals, never a bad write, ADR-0005 fail-safe).

    HONEST SCOPE of "arrivals": messagesAdded[] is literally "added to the
    MAILBOX", which is not identical to "newly received" — a Trash/Spam
    RESTORE is plausibly reported as an add, so a restored OLD message could
    still be scored and receive a graduated label it lacks (a narrow
    retroactivity the field cannot distinguish; the apply command remains the
    deliberate bulk path). Conversely, if a future wire ever omitted
    messagesAdded for a true arrival the failure is SILENT (no scoring,
    visible only as messages=0 in the counter line). Both edges are wire
    facts to re-confirm at the live opt-in seam, not CI-provable here.

    Why NOT the ``historyTypes[]`` SERVER filter: one might ask history.list
    to return only messageAdded records. REJECTED — the correction arm needs
    the SAME delta's labelsAdded/labelsRemoved events, and a type-suppressed
    list's cursor-advance semantics are undocumented (does the cursor still
    cover the suppressed records?). We read the full delta and select arrivals
    client-side, so the correction join and the cursor stay on one honest
    stream."""
    ids: list[str] = []
    for record in records:
        events = record.get("messagesAdded")
        if events is None:
            continue
        if not isinstance(events, list) or not all(
            isinstance(event, dict) for event in events
        ):
            raise GmailPermanentError(
                "history record 'messagesAdded' not a list of objects"
            )
        for event in events:
            message_id = _label_event_message_id(event, "messagesAdded")
            if message_id not in ids:
                ids.append(message_id)
    return ids


def _label_event_message_id(event: dict[str, object], field: str) -> str:
    # the documented HistoryLabelAdded/Removed nests the Message under
    # 'message' (id at event['message']['id']); the generic messages[]
    # form is a flat Message. Tolerate BOTH id nestings — the discovery
    # doc is the authority but no raw HTTP body was capturable, so the
    # daemon must not wedge if the wire differs (a wrong shape reads as
    # 'no events', never a bad write — ADR-0005 fail-safe)
    msg = event.get("message")
    if isinstance(msg, dict):
        mid = msg.get("id")
        if isinstance(mid, str) and mid:
            return mid
    mid = event.get("id")
    if isinstance(mid, str) and mid:
        return mid
    raise GmailPermanentError(f"history {field} event without a message id")


def _label_events(
    records: tuple[dict[str, object], ...],
) -> list[tuple[str, str, str]]:
    """The (kind, message_id, label_id) label-change events of one page's
    history records, in wire order (ticket #27; GLOSSARY: Correction).

    THE WIRE SHAPE (verified against the Gmail discovery document,
    revision 20261026): the response keys are the PLURAL ``labelsAdded``
    / ``labelsRemoved`` (the singular ``labelAdded``/``labelRemoved`` are
    only ``historyTypes[]`` REQUEST-enum values, never response keys),
    each a list of ``{"message": {"id": ...}, "labelIds": [...]}``. The
    adapter passes record internals RAW (gmail_client.history), so this
    is the ONLY place the shape is interpreted. We read the plural keys
    (correct) and fall back to the singular ones ONLY when the plural is
    ABSENT (defensive — the locked design named them; harmless if absent),
    and tolerate both ``message.id`` and a flat ``id`` (see
    _label_event_message_id). The singular is NOT additive: a record
    carrying BOTH a plural and a singular key (a lying / hand-made wire)
    mints events from the PLURAL only — reading both would double-mint,
    contradicting the fail-safe (a wrong shape must read as FEWER events,
    never a bad write). kind is 'added' for a labelsAdded event, 'removed'
    for a labelsRemoved one. A malformed label-event list is a lying wire —
    loud permanent (the _message_ids shape doctrine). Records with no label
    events pass through."""
    events: list[tuple[str, str, str]] = []
    for kind, plural, singular in (
        ("added", "labelsAdded", "labelAdded"),
        ("removed", "labelsRemoved", "labelRemoved"),
    ):
        for record in records:
            # plural present (even an empty list) wins outright; the
            # singular is a fallback read ONLY when the plural is absent
            field = plural if record.get(plural) is not None else singular
            raw = record.get(field)
            if raw is None:
                continue
            if not isinstance(raw, list) or not all(isinstance(e, dict) for e in raw):
                raise GmailPermanentError(
                    f"history record {field!r} not a list of objects"
                )
            for event in raw:
                message_id = _label_event_message_id(event, field)
                label_ids = event.get("labelIds")
                if not isinstance(label_ids, list) or not all(
                    isinstance(x, str) for x in label_ids
                ):
                    raise GmailPermanentError(
                        f"history {field} event 'labelIds' not a list of strings"
                    )
                for label_id in label_ids:
                    events.append((kind, message_id, label_id))
    return events


def _attribution(
    ctx: ServingContext,
    label_id: str,
    fields: tuple[str, tuple[str, ...], str] | None,
) -> str:
    """Invariant #6 attribution for an 'added' Correction (the #22
    mechanism, REUSED not forked) — one of CORRECTION_ATTRIBUTIONS:
    - ``"human"`` — a relevant filter verifiably does NOT explain the add
      (no readable filter adds this label, or every relevant filter is
      simulatable and does not match this message): the human added it
      (positive training signal);
    - ``"filter_covered"`` — a relevant filter verifiably WOULD have added
      it (``simulates_filter`` True): withheld from the human credit and
      the corpus fold (the export dump still shows it, attributed);
    - ``"unverifiable"`` — withheld, not guessed, when authorship cannot be
      determined: no usable filters snapshot (absent or malformed); a
      relevant filter whose criteria cannot be simulated from the kept
      fields (outside ``from``/``to``/``subject``, or malformed); or an add
      whose authorship is otherwise undecidable (its message fields were
      not fetched, or a filter's ``addLabelIds`` is unreadable). An
      unreadable ``addLabelIds`` is GLOBAL, not per-filter: because we
      cannot read which labels that entry adds, we cannot rule it out as
      any add's author, so every 'added' Correction, for every label, is
      withheld until the snapshot is clean.
    The credit arm reads each filter's added labels through the STRICT
    ``filter_label_ids_strict`` (analyze's, the same one the builder and
    the CLI bridge check use): an unreadable ``addLabelIds`` must NOT
    contribute zero ids and let the filter look irrelevant (that lenient
    read was an entry-level inversion of #6 — it credited the human for an
    add a malformed filter may well have authored); an unreadable filter
    therefore withholds the add as 'unverifiable', the #22 posture. An
    attested-empty snapshot (``()``) is NOT the no-snapshot case — it means
    "no filters exist", so an 'added' is 'human'."""
    from types import SimpleNamespace

    if ctx.filters is None:
        return "unverifiable"  # no snapshot -> authorship unknown -> withhold
    relevant: list[Mapping[str, object]] = []
    saw_unreadable = False
    for filter_ in ctx.filters:
        try:
            ids = filter_label_ids_strict(filter_)
        except ValueError:
            # unreadable addLabelIds: we cannot tell which labels this
            # filter adds, so we cannot rule it out as the author -> the
            # add is withheld, NOT credited to the human (#22)
            saw_unreadable = True
            continue
        if label_id in ids:
            relevant.append(filter_)
    if not relevant:
        # no READABLE filter claims the label, but an unreadable one might
        # have -> undecidable, not 'human'
        return "unverifiable" if saw_unreadable else "human"
    if fields is None:
        return "unverifiable"  # a relevant filter exists but nothing to check
    sender, recipients, subject = fields
    rec = SimpleNamespace(sender=sender, recipients=recipients, subject=subject)
    # an unreadable filter keeps authorship open even if a readable one
    # verifiably misses; a PROVEN match (filter_covered) still beats unknown
    saw_unverifiable = saw_unreadable
    for filter_ in relevant:
        criteria = filter_.get("criteria")
        if not isinstance(criteria, dict):
            saw_unverifiable = True
            continue
        verdict = simulates_filter(criteria, rec)  # type: ignore[arg-type]
        if verdict is True:
            return "filter_covered"  # proven filter authorship
        if verdict is None:
            saw_unverifiable = True
    # no filter PROVED coverage; unverifiable only when one was inconclusive
    return "unverifiable" if saw_unverifiable else "human"


def _skip_present_pairs(
    ctx: ServingContext,
    fetched: list[tuple[str, frozenset[str], str]],
    vocab: tuple[str, ...],
    rows: list[tuple[float, ...]],
) -> frozenset[tuple[str, str]]:
    """The (message_id, label_name) pairs this tick would have APPLIED but
    found already PRESENT — a band verdict AND present (skip-present,
    invariant #2). S1 C2 feeds these to detect_corrections: a labelsAdded
    event for such a pair with no Application row is the crash-replay
    ambiguity (our own lost write vs a human who added the same label).

    Reuses ``tiers.would_apply`` (the SAME decision core the plan composes
    through — never re-derived). A message whose would_apply raises is a
    lying input the plan loop already routes to a per-message warning; here
    it contributes NO pair (we cannot claim a skip-present we could not
    compute), so attribution falls through to the ordinary path."""
    pairs: set[tuple[str, str]] = set()
    # the zips are strict-safe: predict_conformant already pinned the row COUNT
    # (== len(fetched)) and each row's WIDTH (== len(vocab)) before we get here,
    # so strict=True can never fire — noted so it does not silently become
    # load-bearing if predict_conformant's checks ever move.
    for (message_id, present, _text), score_row in zip(fetched, rows, strict=True):
        scores = dict(zip(vocab, score_row, strict=True))
        try:
            decisions = would_apply(
                scores=scores,
                bands=ctx.bands,
                graduated=ctx.graduated,
                present=present,
                system_labels=ctx.system_labels,
            )
        except Exception:
            # a lying input contributes NO pair here (we cannot claim a
            # skip-present we could not compute) AND must not escape _run_tick to
            # kill daemon_loop — the plan loop turns the same input into a
            # per-message warning. Netting Exception (not just ValueError) is the
            # defensive choice; unlike plan_message_write we do NOT re-raise
            # AssertionError, but that divergence is unreachable: would_apply's
            # refusals are all ValueError (a lying score/band/vocab), never the
            # internal-contradiction AssertionError the plan guards against.
            continue
        for d in decisions:
            if d.band is not None and d.present:
                pairs.add((message_id, d.label))
    return frozenset(pairs)


def detect_corrections(
    records: tuple[dict[str, object], ...],
    *,
    ctx: ServingContext,
    db: AppDB,
    parsed_by_id: Mapping[str, tuple[str, tuple[str, ...], str]],
    detected_at: datetime,
    skip_present: frozenset[tuple[str, str]] = frozenset(),
) -> list[CorrectionRow]:
    """Classify one page's label-change events into Corrections
    (ticket #27; GLOSSARY: Correction; the module docstring's CORRECTION
    DETECTION section for the full contract). Only messages PRESENT in
    the applications join are candidates (the join AC); the ignore
    boundary (system labels, non-vocabulary labels, a removed
    non-applied label) records NOTHING. The batch is deduplicated on
    (message_id, label, kind) so a hand-fix reported by two events of
    one delta is ONE Correction (the store also refuses a within-batch
    duplicate — this keeps the batch honest). ``parsed_by_id`` carries
    (sender, recipients, subject) for the messages fetched this delta
    (the attribution fields); an 'added' on a message NOT fetched cannot
    be checked against a relevant filter and is attributed 'unverifiable'
    (withheld, not guessed) — see _attribution."""
    # the join: which labels are THIS message's Applications (a name set)
    applied_cache: dict[str, frozenset[str]] = {}

    def applied_labels(message_id: str) -> frozenset[str]:
        if message_id not in applied_cache:
            applied_cache[message_id] = frozenset(
                row.label for row in db.applications_for_message(message_id)
            )
        return applied_cache[message_id]

    rows: list[CorrectionRow] = []
    seen: set[tuple[str, str, str]] = set()
    for kind, message_id, label_id in _label_events(records):
        name = ctx.id_to_name.get(label_id)
        if name is None:
            continue  # an unbridged id (a System Label id, an unknown): noise
        applied = applied_labels(message_id)
        if not applied:
            continue  # not in the join — corrections are only for applied mail
        if kind == "removed":
            if name not in applied:
                continue  # a removed NON-applied label records nothing
            key = (message_id, name, "removed")
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                CorrectionRow(
                    message_id=message_id,
                    label=name,
                    kind="removed",
                    detected_at=detected_at,
                    attribution="human",
                )
            )
        else:  # 'added'
            if name in applied:
                continue  # the daemon wrote it (a row exists) — not a human add
            if name not in ctx.graduated or name in ctx.system_labels:
                continue  # non-vocabulary / System Label churn: noise, not signal
            key = (message_id, name, "added")
            if key in seen:
                continue
            seen.add(key)
            fields = parsed_by_id.get(message_id)
            attribution = _attribution(ctx, label_id, fields)
            if attribution == "human" and (message_id, name) in skip_present:
                # S1 C2 (crash-replay window): _attribution found no filter
                # explaining the add, so it would credit the HUMAN — but this
                # tick's plan SKIPPED this add as already-present and there is
                # NO Application row for it. That makes the human credit
                # AMBIGUOUS: equally it is the daemon's own landed write whose
                # row was lost before the commit (a replay). We withhold only
                # the human credit (invariant #6, withheld-not-guessed); a
                # PROVEN filter_covered (a filter verifiably authored it) or an
                # already-unverifiable add is NOT ambiguous here and stands —
                # downgrading a proof to "unverifiable" would claim an
                # ignorance we do not have.
                attribution = "unverifiable"
            rows.append(
                CorrectionRow(
                    message_id=message_id,
                    label=name,
                    kind="added",
                    detected_at=detected_at,
                    attribution=attribution,
                )
            )
    return rows


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
    # the #27 correction-detection inputs, accumulated across the delta's
    # pages: the raw history records (label events) and the fetched
    # messages' filter-simulation fields (sender, recipients, subject)
    all_records: list[dict[str, object]] = []
    parsed_by_id: dict[str, tuple[str, tuple[str, ...], str]] = {}
    skipped = 0
    token: str | None = None
    new_cursor: int | None = None
    while True:
        try:
            page = client.history(cursor, token)
        except GmailError as e:
            if isinstance(e, GmailAuthError):
                raise  # human-gated, never a page-hold
            if is_404(e):
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
        all_records.extend(page.records)  # #27: label events ride the same drain
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
                if is_404(e):
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
            if outgoing := sorted(_OUTGOING_LABEL_IDS & set(raw.label_ids)):
                # S1 sent/non-incoming guard: the operator's own SENT/DRAFT
                # mail and CHAT arrive as a messageAdded too (see
                # _OUTGOING_LABEL_IDS for the rationale + confidence note). A
                # counted SKIP — named by the triggering label id (body-free),
                # never scored, never written — the posture of a ghost skip.
                skipped += 1
                log.warning(
                    "daemon: skipped message %s: non-incoming (%s)",
                    message_id,
                    ",".join(outgoing),
                )
                continue
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
            present = present_names(ctx, raw.label_ids)  # the SHARED reader
            # the #27 filter-coverage fields (sender/recipients/subject) for
            # a correction on THIS message; the body never enters this tuple
            parsed_by_id[message_id] = (
                parsed.sender,
                parsed.recipients,
                parsed.subject,
            )
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
    # checkpoint). The positive-conformance validation is the SHARED
    # seam (``predict_conformant`` — apply #26 imports it, never
    # re-implements it); the arms below are the DAEMON's posture: the
    # shape defect and the model's own failure both HOLD the cursor.
    #
    # S1 C2: predict now runs BEFORE correction detection — the ambiguous-
    # add check needs this tick's band verdicts (which labels were skipped
    # as already-present), and those come from the scores. Both predict and
    # detection are read-only (model + local DB, no Gmail write), and the
    # commit point is unchanged, so the reorder is behavior-preserving: on a
    # model hold nothing is committed either way.
    try:
        vocab, rows = predict_conformant(model, [text for _, _, text in fetched])
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

    skip_present = _skip_present_pairs(ctx, fetched, vocab, rows)

    # #27 CORRECTION DETECTION (read + local-DB only — never a Gmail
    # write): classify this delta's label-change events against the
    # applications join. A corrected message is EXCLUDED from the write
    # below (the daemon does not fight the human's hand-fix — the landmine
    # "detection never produces a write for the corrected message, ever").
    # The correction rows ride the SAME delta transaction as the rows and
    # the cursor (one commit point, unchanged). skip_present (computed above
    # from this tick's verdicts) lets an added-correction that is really the
    # daemon's own lost write be WITHHELD, not credited (S1 C2).
    try:
        corrections = detect_corrections(
            tuple(all_records),
            ctx=ctx,
            db=db,
            parsed_by_id=parsed_by_id,
            detected_at=applied_at,
            skip_present=skip_present,
        )
    except GmailPermanentError as e:
        # a malformed label-event wire is a lying delta — hold the cursor
        # (the same posture as a malformed messages list above)
        log.warning(
            "daemon: delta failed at cursor %d — cursor held, retry next tick: %s",
            cursor,
            e,
        )
        return stop
    corrected_ids = frozenset(row.message_id for row in corrections)

    # per message: compose (the plan's fail-safe is the net), then the
    # Gmail write FIRST — see the module docstring's safety proof
    page_rows: list[ApplicationRow] = []
    written = 0
    warnings = 0
    for (message_id, present, _text), score_row in zip(fetched, rows, strict=True):
        if message_id in corrected_ids:
            # #27 landmine: a message carrying a correction event this
            # delta is NEVER written (the daemon does not re-add a label
            # the human just removed, nor fight a label they just added).
            # It was fetched + scored (it flowed through the plan pass),
            # but no write, no Application row — the correction RECORD is
            # the only thing this delta books for it.
            continue
        scores = dict(zip(vocab, score_row, strict=True))
        # the SHARED composition call-shape (compose_plan → the one
        # plan_message_write core); the daemon suite's spy on
        # daemon.plan_message_write still intercepts every plan here
        plan = compose_plan(
            ctx,
            message_id,
            scores=scores,
            present=present,
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
            if is_404(e):
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

    # THE durable-handling commit point: rows, Corrections AND cursor in
    # ONE transaction — COMMIT = handled (invariant #3; the store-side
    # all-or-nothing is pinned in test_appdb). The corrections are the
    # delta's human hand-fixes detected above (empty when none).
    if new_cursor is None:  # unreachable: every served page set it
        # (the repo rule: a raise, not an assert — python -O strips
        # asserts; the applications.py internal-contradiction precedent)
        raise AssertionError("delta drained without a cursor value")
    try:
        # The corrections argument is passed ONLY when the delta detected
        # some. The sealed #25 crash-matrix fakes override commit_page
        # with the 2-arg (rows, cursor) signature and never carry label
        # events, so an unconditional 3-arg call would break them verbatim
        # — passing the third argument only when non-empty keeps those
        # fakes green while the real AppDB (corrections defaulting to ())
        # still commits rows + corrections + cursor in ONE transaction.
        if corrections:
            db.commit_page(page_rows, new_cursor, corrections)
        else:
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
        "daemon: delta handled cursor=%d messages=%d written=%d skipped=%d "
        "warnings=%d corrections=%d",
        new_cursor,
        len(fetched),
        written,
        skipped,
        warnings,
        len(corrections),
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
