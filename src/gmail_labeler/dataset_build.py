"""Corpus → training examples (Phase 1 core; GLOSSARY: Training Example).

Pure core (CODING-STANDARDS): deterministic on any input — ``now`` is a
required argument, never read from the clock; sorting is total and stable;
no I/O — consumers pass already-loaded structures (the analyze discipline).

The vocabulary is Label Paths (GLOSSARY: Label Path): classes are the full
label names as seen in the corpus — ``Money/Bills`` is a distinct class
from ``Money`` — and membership is path-aware through the labels.py
primitives (``label_path`` / ``path_is_at_or_under``), never re-parsed ad
hoc: exact membership never crosses paths, while a parent class may scope
its subtree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from .corpus_record import DateProvenance
from .labels import label_path, path_is_at_or_under

if TYPE_CHECKING:
    from .corpus_record import CorpusRecord


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One corpus record turned for training (GLOSSARY: Training
    Example). ``labels`` are the record's label names; ``weight`` is the
    per-example Recency Weight (see ``example_weight``);
    ``date_provenance`` is stamped so temporal-split consumers can filter
    fallback-sourced records at the split boundary (invariant #7)."""

    id: str
    date: datetime
    labels: tuple[str, ...]
    weight: float
    date_provenance: DateProvenance


@dataclass(frozen=True, slots=True)
class BuildResult:
    """The build's output: examples sorted by record id — a total order,
    input-order invariant."""

    examples: tuple[TrainingExample, ...]
    vocabulary: tuple[str, ...]  # sorted full label names (Label Paths)

    def members_of(self, class_name: str) -> tuple[str, ...]:
        """Record ids carrying ``class_name`` exactly. Its label path is
        one class: ``Money/Bills`` is NOT a member of ``Money``."""
        return tuple(
            sorted(
                example.id for example in self.examples if class_name in example.labels
            )
        )

    def subtree_members(self, class_name: str) -> tuple[str, ...]:
        """Path-aware scoping (GLOSSARY: Label Path): ids carrying
        ``class_name`` or any descendant of its label path — how a parent
        Managed Label's gate scopes its subtree. Comparison is per-segment
        (``Money2`` is not under ``Money``)."""
        ancestor = label_path(class_name)
        if not ancestor:
            return ()
        return tuple(
            sorted(
                example.id
                for example in self.examples
                if any(
                    path_is_at_or_under(label_path(name), ancestor)
                    for name in example.labels
                )
            )
        )


def build_dataset(records: Sequence[CorpusRecord], *, now: datetime) -> BuildResult:
    """Turn CorpusRecords into training examples + the Label Path
    vocabulary. Deterministic: examples sorted by id, vocabulary sorted —
    identical inputs give identical results, input order irrelevant. The
    store guarantees unique ids; a duplicate would make the stable sort
    order-dependent, so build_dataset rejects duplicates (defensive).
    ``now`` is required (the analyze discipline): the core never reads
    the clock."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise ValueError(f"duplicate record id: {record.id}")
        seen.add(record.id)
    examples = tuple(
        TrainingExample(
            id=record.id,
            date=record.date,
            labels=tuple(sorted(record.labels)),
            weight=1.0,
            date_provenance=record.date_provenance,
        )
        for record in sorted(records, key=lambda record: record.id)
    )
    vocabulary = tuple(
        sorted({name for example in examples for name in example.labels})
    )
    return BuildResult(examples=examples, vocabulary=vocabulary)
