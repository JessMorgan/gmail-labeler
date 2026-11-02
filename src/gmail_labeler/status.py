"""Realized precision from the local Corrections + Applications store
(ticket #27; GLOSSARY: Correction, Status).

Pure core (CODING-STANDARDS): deterministic on any input — no clock, no
I/O, no Gmail call. ``compute_precision`` turns the store's aggregate
counts into per-graduated-label realized precision — the answer to "is
it working?" WITHOUT asking Gmail anything (the AC: status is DB-only).

The math (design decision 4, v1 honest-and-simple):

- ``written``  = the Application ROWS for the label (we applied it); a
  re-earned label is a NEW row (the applications table has no UNIQUE
  constraint), so ``written`` can exceed the number of distinct messages
  that carry the label;
- ``removed``  = the matching 'removed' Corrections (a human undid it),
  deduped per (message, label) by the store's UNIQUE index — so the
  denominator (rows) and the numerator's correction count are on
  DIFFERENT units when a label has been re-earned; the CLI's basis line
  states this (v1, per the manager's "all rows" ruling);
- ``precision`` = (written - removed) / written, or ``None`` ('no data')
  when the label has no Applications yet. The join gate makes
  ``removed <= written`` (a 'removed' Correction exists only for an
  applied (message, label), deduped), so precision never goes negative on
  an honest store; a tampered DB where ``removed > written`` is refused
  NAMED (``PrecisionInvariantError`` → CLI exit 5), never clamped to 0.0;
- ``added`` / ``added_filter_covered`` / ``added_unverifiable`` = the
  positive-signal Corrections and the two invariant-#6 WITHHELD slices: a
  Gmail filter verifiably applied it (``filter_covered``), or authorship
  cannot be determined (``unverifiable``). BOTH slices are EXCLUDED from
  ``human_added`` (the net human signal) and DISCLOSED separately — status
  shows the total and each withheld slice. An unverifiable add is WITHHELD,
  not credited to the human (#22's exclusion principle: guessing either way
  is what #22 forbids).

Temporal honesty (design landmine): precision counts removals over ALL
rows (v1); the CLI's basis line states the 'assumes unseen-but-removed
mail is rare' caveat and the archive-equals-dismiss assumption, and
notes the rows can span artifact versions (the per-run binding stays
deferred to post-merge calibration)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["LabelPrecision", "PrecisionInvariantError", "compute_precision"]


class PrecisionInvariantError(Exception):
    """compute_precision found ``removed > written`` for a label — a state
    the join gate makes impossible on an honest store, so this names a
    hand-edited/corrupt DB. The CLI maps it to exit 5 (refused, named)."""


@dataclass(frozen=True, slots=True)
class LabelPrecision:
    """One graduated label's realized-precision line (NO bodies —
    invariant #5: ids/labels/counts/rates only)."""

    label: str
    written: int
    removed: int
    precision: float | None  # None == 'no data' (written == 0)
    added: int
    added_filter_covered: int
    added_unverifiable: int

    @property
    def human_added(self) -> int:
        """The positive-signal Corrections credited to the human — the
        'added' total minus BOTH withheld slices (invariant #6: a
        filter-covered add is not human signal, and an unverifiable one is
        WITHHELD, not guessed either way — #22's exclusion principle)."""
        return self.added - self.added_filter_covered - self.added_unverifiable


def compute_precision(
    graduated: frozenset[str],
    written_by_label: Mapping[str, int],
    correction_stats: Mapping[str, tuple[int, int, int, int]],
    *,
    db_path: str | None = None,
) -> list[LabelPrecision]:
    """Per GRADUATED label (sorted — a total order for the report), the
    realized-precision line. ``correction_stats`` maps label ->
    (removed, added, added_filter_covered, added_unverifiable) (the store's
    ``correction_label_stats``). A graduated label with no Applications
    reports 'no data' (precision None); one with no Corrections reports
    precision 1.0 (everything applied is still standing). ``db_path`` (when
    given) names the state DB in the refusal — with several --db files in
    play the operator must know WHICH store is corrupt.

    Raises ``PrecisionInvariantError`` when ``removed > written`` for a
    label: the join gate keeps ``removed <= written`` on the honest path
    (a 'removed' Correction exists only for an applied (message, label),
    deduped), so a violation is a hand-edited/corrupt store — refused
    NAMED with a recovery clause (the appdb tamper-family tone), never
    clamped to 0.0 (a clamp would hide the corruption behind a
    plausible-looking precision)."""
    out: list[LabelPrecision] = []
    for label in sorted(graduated):
        written = written_by_label.get(label, 0)
        removed, added, added_fc, added_uv = correction_stats.get(label, (0, 0, 0, 0))
        if removed > written:
            where = f": {db_path}" if db_path else ""
            recovery = (
                " — recovery: a 'removed' Correction has no matching "
                "Application row; hand-fix the corrections rows, or delete "
                "the state DB (ALL local history — applications, shadow, "
                "corrections — is lost and CANNOT be rebuilt: the daemon "
                "re-bootstraps the cursor without replaying history, so past "
                "Applications are never restored, the 'written' count "
                "restarts at zero and grows only from mail applied AFTER the "
                "reset, and mail before the new bootstrap cursor is not "
                "processed)"
                if db_path
                else ""
            )
            raise PrecisionInvariantError(
                f"label {label!r}: {removed} 'removed' Corrections exceed "
                f"{written} Application rows — the join gate makes this "
                f"impossible on an honest store; the state DB is corrupt "
                f"(refusing, not clamping){where}{recovery}"
            )
        precision = (written - removed) / written if written > 0 else None
        out.append(
            LabelPrecision(
                label=label,
                written=written,
                removed=removed,
                precision=precision,
                added=added,
                added_filter_covered=added_fc,
                added_unverifiable=added_uv,
            )
        )
    return out
