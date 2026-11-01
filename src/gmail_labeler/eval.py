"""Per-label precision/recall at candidate thresholds on the temporal
test split (ticket #20, GLOSSARY: Eval Report).

Pure core (CODING-STANDARDS): deterministic on any input — no clock,
no I/O; sorting is total and stable (the analyze discipline).

This module does NOT split: the caller passes the TEST-side examples
rebuilt through the Split Manifest (invariant #7 — the builder is the
only split implementation; the manifest is authoritative). The
split-boundary trust filter (``server_authoritative_only``) is a
split-identity decision owned by the manifest: eval inherits it, never
re-applies it.

Candidate thresholds: a FIXED grid of tenths, named in exactly one
place (``DEFAULT_THRESHOLDS``) — never scattered. #21's calibration
replaces the grid (Threshold Bands) without changing this interface.
A non-tenth grid is REFUSED: the text report labels its columns to
one decimal (``0.11`` renders as ``0.1``), so a non-tenth grid would
render mislabeled columns — fail-loud beats a lying table.

Alignment is by EXAMPLE ID, not position: ``scores`` maps example id →
score row (column j ↔ ``vocabulary[j]``). A positional read would
silently misalign whenever the input order is not the canonical one —
the refusal below makes a lying or incomplete score vector impossible.

Conventions (pinned by golden tests):

- a score EXACTLY at the threshold is a positive (``score >= t``);
- precision = TP/(TP+FP), recall = TP/(TP+FN); a zero denominator is
  0.0 — never a fabricated 1.0 and never undefined;
- rows are in vocabulary order (the vocabulary is canonically sorted —
  a total order, input-order invariant);
- nothing body-shaped enters this module: inputs are TrainingExamples
  (metadata) + score rows; the report carries labels, counts, and
  probabilities only (invariant #5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

from .dataset_build import TrainingExample, is_tenth, sorted_unique

__all__ = [
    "DEFAULT_THRESHOLDS",
    "EvalTable",
    "LabelEvalRow",
    "ThresholdCell",
    "eval_metrics",
    "render_report",
]

#: The candidate-threshold grid (GLOSSARY: Eval Report): tenths, 0.1..0.9.
#: Named in exactly one place — the report, the JSON payload, and any
#: future consumer read THIS, never a re-typed constant. #21's
#: calibration (Threshold Bands) supersedes it per label.
DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


@dataclass(frozen=True, slots=True)
class ThresholdCell:
    """One (label, threshold) verdict: the confusion counts and the two
    ratios derived from them (0.0 on a zero denominator — the pinned
    convention)."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float


@dataclass(frozen=True, slots=True)
class LabelEvalRow:
    """One label's row across the threshold grid. ``support`` is the
    number of ACTUAL positives on the test side (TP+FN) — the volume a
    Graduation gate reads."""

    label: str
    support: int
    cells: tuple[ThresholdCell, ...]  # aligned to the table's thresholds


@dataclass(frozen=True, slots=True)
class EvalTable:
    """The measured table: rows = label classes (vocabulary order),
    columns = candidate thresholds. Deterministic — same inputs give
    the same table (and a byte-identical ``render_report``)."""

    vocabulary: tuple[str, ...]
    thresholds: tuple[float, ...]
    test_examples: int
    rows: tuple[LabelEvalRow, ...]


def eval_metrics(
    examples: Sequence[TrainingExample],
    scores: Mapping[str, tuple[float, ...]],
    *,
    vocabulary: tuple[str, ...],
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> EvalTable:
    """Score the test side at every candidate threshold.

    ``examples`` are the TEST-side Training Examples (manifest mode —
    the caller rebuilt them through the Split Manifest); ``scores``
    maps example id → score row (column j ↔ ``vocabulary[j]``, the
    Model Protocol's alignment). Refusals (fail-loud doctrine): an
    empty/non-canonical vocabulary, a threshold grid that is empty,
    out of (0, 1), not strictly ascending, or not a grid of tenths
    (the text report labels columns to one decimal — a non-tenth
    would render mislabeled), duplicate example ids, an example label
    outside the vocabulary (it would have no score column — silently
    unmeasured), a score row that does not align to the vocabulary or
    leaves the sigmoid range, a missing score (undercounting), and a
    score for an id that is not an example (a lying vector) all raise
    ValueError — the table is a measured claim, never a guess.
    """
    if not vocabulary:
        raise ValueError("vocabulary must not be empty")
    if not sorted_unique(vocabulary):
        raise ValueError("vocabulary must be sorted and unique (canonical form)")
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    for t in thresholds:
        if not 0.0 < t < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {t}")
        if not is_tenth(t):
            raise ValueError(
                f"threshold {t} is not a tenth — the report labels its "
                "columns to one decimal, so a non-tenth grid would render "
                "mislabeled columns"
            )
    for a, b in pairwise(thresholds):
        if a >= b:
            raise ValueError(f"thresholds must be strictly ascending, got {a} then {b}")

    seen: set[str] = set()
    vocab_set = set(vocabulary)
    for example in examples:
        if example.id in seen:
            raise ValueError(f"duplicate example id: {example.id}")
        seen.add(example.id)
        outside = [name for name in example.labels if name not in vocab_set]
        if outside:
            raise ValueError(
                f"example {example.id} carries label(s) {outside} outside "
                "the vocabulary — refusing to score silently (the label "
                "would have no column)"
            )

    missing = seen - set(scores)
    if missing:
        raise ValueError(
            f"no score for example(s) {sorted(missing)} — the table would "
            "silently undercount; score every example or exclude it"
        )
    unknown = set(scores) - seen
    if unknown:
        raise ValueError(
            f"score for {sorted(unknown)} is not an example — a lying "
            "score vector is refused"
        )

    width = len(vocabulary)
    for example in examples:
        row = scores[example.id]
        if len(row) != width:
            raise ValueError(
                f"score row for {example.id!r} does not align to the "
                f"vocabulary: {len(row)} scores for {width} labels"
            )
        if any(not 0.0 <= v <= 1.0 for v in row):
            raise ValueError(
                f"score row for {example.id!r} out of sigmoid range [0, 1]"
            )

    # the score rows, aligned to `examples` (index i ↔ examples[i]) —
    # the dict lookup happens once per example, not per (label,
    # threshold) cell
    score_rows = [scores[example.id] for example in examples]

    rows: list[LabelEvalRow] = []
    for j, label in enumerate(vocabulary):
        cells: list[ThresholdCell] = []
        for t in thresholds:
            tp = fp = fn = 0
            for example, row in zip(examples, score_rows, strict=True):
                if label in example.labels:
                    if row[j] >= t:
                        tp += 1
                    else:
                        fn += 1
                elif row[j] >= t:
                    fp += 1
            cells.append(
                ThresholdCell(
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    precision=tp / (tp + fp) if tp + fp else 0.0,
                    recall=tp / (tp + fn) if tp + fn else 0.0,
                )
            )
        rows.append(
            LabelEvalRow(
                label=label,
                support=sum(1 for example in examples if label in example.labels),
                cells=tuple(cells),
            )
        )
    return EvalTable(
        vocabulary=vocabulary,
        thresholds=thresholds,
        test_examples=len(examples),
        rows=tuple(rows),
    )


def render_report(
    table: EvalTable, *, split_date: datetime, split_manifest_hash: str
) -> str:
    """The deterministic, diff-friendly text report: a split-
    provenance header (split date + the manifest's content hash, short
    form — a saved .txt is never indistinguishable from a stale one),
    per-label rows x candidate-threshold columns, each cell "P/R"
    (two decimals, precision first), the support count per label, and
    a convention footer naming the zero-denominator rendering. Same
    inputs → byte-identical output. Nothing body-shaped is in the
    table, so nothing body-shaped can be rendered (invariant #5)."""
    if not table.thresholds:
        raise ValueError("table.thresholds must not be empty")
    # no cap, no truncation: the width grows with the longest label so
    # every row lines up with the header, however long the label is
    label_w = max((len(row.label) for row in table.rows), default=5) + 1
    lines = [
        f"eval: split {split_date.isoformat()} manifest={split_manifest_hash[:16]}",
        f"eval: test examples={table.test_examples} labels={len(table.rows)} "
        f"thresholds={table.thresholds[0]:.1f}..{table.thresholds[-1]:.1f}",
        f"{'label':<{label_w}} {'support':>7}"
        + "".join(f"{t:.1f}".rjust(10) for t in table.thresholds),
    ]
    for row in table.rows:
        lines.append(
            f"{row.label:<{label_w}} {row.support:>7}"
            + "".join(
                f"{cell.precision:.2f}/{cell.recall:.2f}".rjust(10)
                for cell in row.cells
            )
        )
    lines.append(
        "precision/recall render 0.00 where the denominator is zero; "
        "see tp/fp/fn in --json"
    )
    return "\n".join(lines) + "\n"
