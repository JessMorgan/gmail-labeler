"""dataset build: path-aware vocabulary, training examples, determinism."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gmail_labeler.corpus_record import CorpusRecord, DateProvenance, body_hash
from gmail_labeler.dataset_build import build_dataset

NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)


def _rec(
    i: int,
    labels: tuple[str, ...],
    *,
    days_ago: int = 1,
    provenance: DateProvenance = "internal_date",
) -> CorpusRecord:
    return CorpusRecord(
        id=f"{i:016x}",
        date=NOW - timedelta(days=days_ago),
        subject="hello",
        sender="x@y.z",
        recipients=("me@x.y",),
        labels=labels,
        body_hash=body_hash(f"unique body {i}"),
        source="api",
        date_provenance=provenance,
    )


# ---- vocabulary: Label Paths ----


def test_vocabulary_is_label_paths() -> None:
    # vocabulary = the label paths as seen (full names), incl. a parent
    # used as a label alongside its children; GLOSSARY "Label Path" is
    # authoritative — Money/Bills is a distinct path from Money
    recs = [
        _rec(1, ("Money",)),
        _rec(2, ("Money/Bills",)),
        _rec(3, ("Money", "Money/Bills")),
    ]
    result = build_dataset(recs, now=NOW)
    assert result.vocabulary == ("Money", "Money/Bills")


def test_vocabulary_sorted_order_pinned() -> None:
    # ≥6 labels where sorted order != insertion order: a set() (unordered)
    # or a mis-sorted vocabulary fails this — the 2-name fixture was
    # seed-luck (deepseek). Reversed input must give the identical result.
    labels = ["Zeta", "Alpha", "Mike", "Bob", "Yan", "Ann"]
    recs = [_rec(i, (lab,)) for i, lab in enumerate(labels, start=1)]
    result = build_dataset(recs, now=NOW)
    assert result.vocabulary == tuple(sorted(labels))
    assert result.vocabulary != tuple(labels)  # sorted order != insertion order
    assert (
        build_dataset(list(reversed(recs)), now=NOW).vocabulary == result.vocabulary
    )  # input order irrelevant


def test_members_of_is_exact_path() -> None:
    recs = [_rec(1, ("Money",)), _rec(2, ("Money/Bills",))]
    result = build_dataset(recs, now=NOW)
    assert result.members_of("Money") == ("0000000000000001",)
    assert result.members_of("Money/Bills") == ("0000000000000002",)
    assert result.members_of("Bills") == ()  # no class named "Bills"


def test_subtree_members_scopes_descendants() -> None:
    recs = [
        _rec(1, ("Money",)),
        _rec(2, ("Money/Bills",)),
        _rec(3, ("Money/Bills/2025",)),
        _rec(4, ("Money2",)),
    ]
    result = build_dataset(recs, now=NOW)
    # a parent Managed Label's gate may scope its subtree
    assert result.subtree_members("Money") == (
        "0000000000000001",
        "0000000000000002",
        "0000000000000003",
    )
    assert result.subtree_members("Money2") == ("0000000000000004",)
    # per-segment: ("Money","Bills") is NOT under ("Bills",)
    assert result.subtree_members("Bills") == ()


# ---- examples: shape + determinism ----


def test_examples_sorted_by_id_input_order_invariant() -> None:
    recs = [_rec(3, ("A",)), _rec(1, ("B",)), _rec(2, ("A",))]
    r1 = build_dataset(recs, now=NOW)
    r2 = build_dataset(list(reversed(recs)), now=NOW)
    assert r1 == r2  # input order irrelevant
    assert [e.id for e in r1.examples] == [
        "0000000000000001",
        "0000000000000002",
        "0000000000000003",
    ]


def test_example_carries_labels_and_provenance_stamp() -> None:
    (example,) = build_dataset(
        [_rec(1, ("A", "B"), provenance="mime_date")], now=NOW
    ).examples
    assert example.labels == ("A", "B")
    assert example.date_provenance == "mime_date"
    assert example.weight == 1.0  # flat until recency weighting lands


def test_example_labels_are_sorted() -> None:
    # neither Gmail labelIds nor the GYB label map guarantees stable
    # order — the example's label tuple is canonical (sorted)
    (example,) = build_dataset([_rec(1, ("B", "A"))], now=NOW).examples
    assert example.labels == ("A", "B")


def test_subtree_members_empty_class_name() -> None:
    result = build_dataset([_rec(1, ("Money",))], now=NOW)
    assert result.subtree_members("") == ()


def test_duplicate_record_id_rejected() -> None:
    # the store guarantees unique ids; a duplicate would make the stable
    # sort order-dependent, so the build rejects it (build path, not
    # manifest mode)
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        build_dataset([_rec(1, ("A",)), _rec(1, ("B",))], now=NOW)


def test_empty_corpus() -> None:
    result = build_dataset([], now=NOW)
    assert result.examples == ()
    assert result.vocabulary == ()


def test_naive_now_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="timezone-aware"):
        build_dataset([_rec(1, ("A",))], now=datetime(2026, 1, 1))
