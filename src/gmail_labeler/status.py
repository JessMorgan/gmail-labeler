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
  applied (message, label), deduped), so precision never goes negative;
- ``added`` / ``added_filter_covered`` = the positive-signal Corrections
  and the invariant-#6 slice a Gmail filter would have applied anyway.
  The filter-covered slice is EXCLUDED from ``human_added`` (the net
  human signal) and DISCLOSED separately — status shows BOTH totals.

Temporal honesty (design landmine): precision counts removals over ALL
rows (v1); the CLI's basis line states the 'assumes unseen-but-removed
mail is rare' caveat and the archive-equals-dismiss assumption, and
notes the rows can span artifact versions (the per-run binding stays
deferred to #23)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["LabelPrecision", "compute_precision"]


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

    @property
    def human_added(self) -> int:
        """The positive-signal Corrections NOT explained by a Gmail
        filter (invariant #6: the filter-covered slice is excluded)."""
        return self.added - self.added_filter_covered


def compute_precision(
    graduated: frozenset[str],
    written_by_label: Mapping[str, int],
    correction_stats: Mapping[str, tuple[int, int, int]],
) -> list[LabelPrecision]:
    """Per GRADUATED label (sorted — a total order for the report), the
    realized-precision line. ``correction_stats`` maps label ->
    (removed, added, added_filter_covered) (the store's
    ``correction_label_stats``). A graduated label with no Applications
    reports 'no data' (precision None); one with no Corrections reports
    precision 1.0 (everything applied is still standing)."""
    out: list[LabelPrecision] = []
    for label in sorted(graduated):
        written = written_by_label.get(label, 0)
        removed, added, added_fc = correction_stats.get(label, (0, 0, 0))
        precision = (written - removed) / written if written > 0 else None
        out.append(
            LabelPrecision(
                label=label,
                written=written,
                removed=removed,
                precision=precision,
                added=added,
                added_filter_covered=added_fc,
            )
        )
    return out
