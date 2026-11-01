"""The Application write plan (ticket #24; GLOSSARY: Application).

Pure composition (CODING-STANDARDS): deterministic on any input — no
clock, no I/O, no Gmail call. ``plan_message_write`` composes ONE
message's would-apply verdicts into the write the daemon (#25) and
``apply`` (#26) perform — and it is COMPOSITION ONLY: every band/tier/
skip-present decision is IMPORTED from ``tiers`` (the single decision
core; #26's AC is that apply reuses the same function, and a
re-derivation anywhere is the bug class this layout exists to prevent).

The composition (per message):

- the verdicts come from ``tiers.would_apply`` (graduated labels only,
  band edges, skip-present, the System Label name guard);
- the ADD set is the would-apply Managed Labels' Gmail ids PLUS the
  umbrella Tier Label id PLUS the best band Tier Label id (the GLOSSARY
  Application contract: umbrella + exactly one band tier — high beats
  med). Name→id resolution stays at the adapter edge: the maps are
  passed IN (resolved once per startup by the daemon, pure thereafter —
  the tiers docstring's bridge contract);
- skip-present consequence (invariant #2): an already-present label
  earns no row and no Tier Labels — nothing new added means an EMPTY
  plan, and the caller's rule is no API call at all (the crash-retry
  idempotency: the retry sees the label present and composes empty);
- fail-safe (invariant #1): ANY exception inside the per-message
  compose — a tiers refusal, a missing id in the resolution maps, a
  lying input surface — turns into NO WRITE for that message (empty
  adds) plus a counted warning naming the message id. The caller logs
  the warning (stderr, counted) and moves on; a legitimate empty plan
  carries NO warning, so the two empty states are distinguishable. The
  compose inputs are ids/labels/scores ONLY — bodies never enter the
  plan, so the warning text structurally cannot carry one (invariant
  #5);
- System Labels are never targeted, at TWO levels: tiers refuses a
  graduated System Label NAME, and this module screens every resolved
  id through ``labels.is_system_label_id`` — a lying resolution map
  that points a label (or a tier) at a System Label id composes no
  write;
- deterministic total order: ``adds`` is sorted (a set union of ids —
  the input mapping order never leaks), rows are sorted by
  (message_id, label).

The caller bug vs data error line: a bad ``source`` literal or a naive
``applied_at`` is a CALLER bug (the daemon's clock seam is ``cli._now``
— aware by construction) and RAISES — silently warning every message
would turn a programming defect into silent label starvation.
Everything the message's own data says (scores, bands, ids) goes
through the fail-safe net.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from .labels import is_system_label_id
from .tiers import TIER_APPLIED, best_tier, would_apply

__all__ = ["SOURCES", "ApplicationRow", "WritePlan", "plan_message_write"]


@dataclass(frozen=True, slots=True)
class ApplicationRow:
    """One applied-label row (NO bodies — invariant #5): the message
    id, the Managed Label NAME, its Gmail label id, the sigmoid score,
    the band verdict (``"high"`` / ``"med"`` — an applied row always has
    a band), the Tier Label name the Application carried (the tiers-core
    spelling: the configured ``[labels]`` wire spelling is config-
    derivable), the UTC applied-at stamp, and the source (``"daemon"``
    or ``"apply"`` — the literal that entered the write path). One row
    per applied label; the Application's tier is the per-message best
    band tier, repeated on the message's rows (the audit join reads
    either way). Lives in the PURE module (the corpus_record/corpus_store
    direction: the adapter imports the row shape from the core, never
    the reverse); ``appdb`` validates and stores it.

    Band-tier ACCUMULATION semantics (pinned here, decided by #25): the
    GLOSSARY's "exactly one band Tier Label" is per-Application — a
    later Application on a message that already carries an earlier band
    tier adds ITS OWN best band tier (additive-only forbids correcting
    the old one). Whether the daemon counts Tier Labels in the
    message's ``present`` set (making a carried tier skip-present) is
    #25's policy line; the composition is identical either way."""

    message_id: str
    label: str
    label_id: str
    score: float
    band: str
    tier: str
    applied_at: datetime
    source: str


#: The Application source literals — mirrored import-light from the
#: store's CHECK vocabulary (appdb._SOURCES); the equality is pinned by
#: test_applications, not by an import (the _BANDS doctrine).
SOURCES = ("daemon", "apply")


@dataclass(frozen=True, slots=True)
class WritePlan:
    """One message's composed write. ``adds`` is the FULL sorted Gmail
    label-id set the single modify call would carry (managed ids + the
    umbrella + the best band tier — deduplicated: a managed label whose
    id equals a tier id is ONE add); empty means NO API call at all.
    ``rows`` are the Application rows (one per applied label, sorted).
    ``warning`` is non-None exactly when the compose failed (fail-safe:
    adds/rows are then empty AND the caller must count + log it) — the
    legitimate-empty state carries no warning."""

    message_id: str
    adds: tuple[str, ...]
    rows: tuple[ApplicationRow, ...]
    warning: str | None


def plan_message_write(
    message_id: str,
    *,
    scores: Mapping[str, float],
    bands: Mapping[str, tuple[float | None, float | None]],
    graduated: frozenset[str],
    present: frozenset[str],
    system_labels: frozenset[str],
    managed_label_ids: Mapping[str, str],
    tier_label_ids: Mapping[str, str],
    applied_at: datetime,
    source: str,
) -> WritePlan:
    """Compose ONE message's Application write (see the module docstring
    for the full contract).

    ``managed_label_ids`` maps Managed Label NAME → Gmail label id;
    ``tier_label_ids`` maps the tiers' DEFAULT tier constants
    (``TIER_APPLIED`` / ``TIER_HIGH`` / ``TIER_MED``) → Gmail label ids
    (the daemon has already resolved the operator's configured
    ``[labels] applied/high/med`` names to ids and passes them under
    these keys — the bridge contract the tiers docstring records).
    ``applied_at`` is the caller's clock read (``cli._now`` — the ONE
    seam; the pure core never reads the clock).

    Raises ValueError for CALLER bugs (a source outside ``SOURCES``, a
    naive ``applied_at``). Per-message DATA errors never raise: they
    compose an empty plan with a warning (invariant #1).
    """
    # the pre-compose caller guards (raise — a programming defect must
    # crash loudly, never warn every message into silence)
    if source not in SOURCES:
        raise ValueError(
            f"plan_message_write: source {source!r} must be one of {SOURCES}"
        )
    if applied_at.tzinfo is None:
        raise ValueError(
            "plan_message_write: applied_at must be timezone-aware (the "
            "clock seam is cli._now — a naive timestamp is a caller bug)"
        )
    try:
        decisions = would_apply(
            scores=scores,
            bands=bands,
            graduated=graduated,
            present=present,
            system_labels=system_labels,
        )
        applied = [d for d in decisions if d.applied]
        if not applied:
            # nothing new (no band, or every banded label is present):
            # NO Application row, NO Tier Labels, no API call (the
            # skip-present consequence, invariant #2)
            return WritePlan(message_id=message_id, adds=(), rows=(), warning=None)
        best = best_tier(decisions)
        if best is None:  # unreachable while `applied` is non-empty — defended
            raise AssertionError("applied decisions without a best tier")
        managed_ids = [_resolve_id(managed_label_ids, d.label) for d in applied]
        tier_ids = [
            _resolve_id(tier_label_ids, TIER_APPLIED),
            _resolve_id(tier_label_ids, best),
        ]
        row_items: list[ApplicationRow] = []
        for d, label_id in zip(applied, managed_ids, strict=True):
            if d.band is None:  # unreachable: applied implies a band verdict
                raise AssertionError(f"applied decision {d.label!r} without a band")
            row_items.append(
                ApplicationRow(
                    message_id=message_id,
                    label=d.label,
                    label_id=label_id,
                    score=d.score,
                    band=d.band,
                    tier=best,  # the per-message best band tier (one per Application)
                    applied_at=applied_at,
                    source=source,
                )
            )
        rows = tuple(sorted(row_items, key=lambda r: (r.message_id, r.label)))
        adds = tuple(sorted(set(managed_ids) | set(tier_ids)))
        return WritePlan(message_id=message_id, adds=adds, rows=rows, warning=None)
    except Exception as e:  # the fail-safe net IS the point (invariant #1)
        # invariant #1: ANY application-path error means NO write for
        # that message. The warning names the id (counted, logged by
        # the caller); the compose inputs are ids/labels/scores only,
        # so the error text cannot carry a body (invariant #5).
        return WritePlan(
            message_id=message_id,
            adds=(),
            rows=(),
            warning=f"{message_id}: {type(e).__name__}: {e}",
        )


def _resolve_id(id_map: Mapping[str, str], name: str) -> str:
    # the adapter-edge lookup: a missing entry is a lying edge (the
    # startup resolution covers the vocabulary), refused into the net;
    # a System Label ID is refused HERE — the composition-level guard
    # on top of tiers' name guard (a tier or managed name resolved at
    # INBOX never reaches the wire)
    try:
        label_id = id_map[name]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"no Gmail label id resolved for {name!r} — the startup "
            "resolution must cover every would-apply name"
        ) from e
    if is_system_label_id(label_id):
        raise ValueError(
            f"{name!r} resolved to System Label id {label_id!r} — System "
            "Labels are never write targets"
        )
    return label_id
