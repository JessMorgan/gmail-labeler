"""eval metric table: golden determinism, alignment by example id,
refusals (the test_analyze pattern — in-memory examples + hand-authored
score vectors, no I/O, no clock).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gmail_labeler.dataset_build import TrainingExample
from gmail_labeler.eval import (
    DEFAULT_THRESHOLDS,
    eval_metrics,
    render_report,
)

VOCAB = ("Bills", "Shipping")
NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
SPLIT_DATE = datetime(2026, 1, 1, tzinfo=UTC)
MANIFEST_HASH = "0" * 64  # rendered short: manifest=0000000000000000


def _ex(i: int, labels: tuple[str, ...]) -> TrainingExample:
    return TrainingExample(
        id=f"{i:016x}",
        date=NOW,
        labels=labels,
        weight=1.0,
        date_provenance="internal_date",
    )


def _golden_examples() -> list[TrainingExample]:
    # ex0: Bills, scores (0.9, 0.1)
    # ex1: Bills, scores (0.4, 0.6)
    # ex2: Shipping, scores (0.2, 0.8)
    # ex3: no labels, scores (0.5, 0.5)
    return [_ex(0, ("Bills",)), _ex(1, ("Bills",)), _ex(2, ("Shipping",)), _ex(3, ())]


def _golden_scores() -> dict[str, tuple[float, ...]]:
    return {
        "0000000000000000": (0.9, 0.1),
        "0000000000000001": (0.4, 0.6),
        "0000000000000002": (0.2, 0.8),
        "0000000000000003": (0.5, 0.5),
    }


# ---- the golden table (hand-computed, independent of the code) ----


def test_golden_metric_table() -> None:
    table = eval_metrics(_golden_examples(), _golden_scores(), vocabulary=VOCAB)
    assert table.test_examples == 4
    assert table.vocabulary == VOCAB
    assert table.thresholds == DEFAULT_THRESHOLDS
    assert [row.label for row in table.rows] == list(VOCAB)  # total order

    bills = table.rows[0]
    shipping = table.rows[1]
    assert bills.support == 2
    assert shipping.support == 1

    # Bills (column 0): actual positives ex0, ex1
    assert [c.tp for c in bills.cells] == [2, 2, 2, 2, 1, 1, 1, 1, 1]
    assert [c.fp for c in bills.cells] == [2, 2, 1, 1, 1, 0, 0, 0, 0]
    assert [c.fn for c in bills.cells] == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert [c.precision for c in bills.cells] == [
        0.5,
        0.5,
        2 / 3,
        2 / 3,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert [c.recall for c in bills.cells] == [
        1.0,
        1.0,
        1.0,
        1.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    ]

    # Shipping (column 1): actual positive ex2
    assert [c.tp for c in shipping.cells] == [1, 1, 1, 1, 1, 1, 1, 1, 0]
    assert [c.fp for c in shipping.cells] == [3, 2, 2, 2, 2, 1, 0, 0, 0]
    assert [c.fn for c in shipping.cells] == [0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert [c.precision for c in shipping.cells] == [
        0.25,
        1 / 3,
        1 / 3,
        1 / 3,
        1 / 3,
        0.5,
        1.0,
        1.0,
        0.0,  # no predictions at 0.9: precision is 0.0, not a guess
    ]
    assert [c.recall for c in shipping.cells] == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
    ]


def test_score_equal_threshold_counts_positive() -> None:
    # the >= boundary: a score EXACTLY at the threshold is a positive
    examples = [_ex(0, ("Bills",))]
    scores = {"0000000000000000": (0.5, 0.0)}
    table = eval_metrics(examples, scores, vocabulary=VOCAB)
    cell = table.rows[0].cells[4]  # t=0.5
    assert cell.tp == 1 and cell.fp == 0 and cell.precision == 1.0
    # Shipping: no score at 0.5 → no prediction
    assert table.rows[1].cells[4].tp == 0 and table.rows[1].cells[4].fp == 0


def test_zero_support_label_is_zero_not_undefined() -> None:
    # a label with no actual positives on the test side: support 0,
    # recall 0.0; precision 0.0 when it has false positives (never a
    # fabricated 1.0)
    examples = [_ex(0, ())]
    scores = {"0000000000000000": (0.9, 0.9)}
    table = eval_metrics(examples, scores, vocabulary=VOCAB)
    for row in table.rows:
        assert row.support == 0
        assert all(c.recall == 0.0 for c in row.cells)
        assert all(c.precision == 0.0 for c in row.cells)  # FP > 0 everywhere


def test_no_predictions_no_positives_is_zero() -> None:
    # both denominators zero: P and R are 0.0 (the pinned convention)
    examples = [_ex(0, ())]
    scores = {"0000000000000000": (0.0, 0.0)}
    table = eval_metrics(examples, scores, vocabulary=VOCAB)
    for row in table.rows:
        assert all(c.precision == 0.0 and c.recall == 0.0 for c in row.cells)


def test_default_thresholds_are_the_named_tenths_grid() -> None:
    assert DEFAULT_THRESHOLDS == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    table = eval_metrics(_golden_examples(), _golden_scores(), vocabulary=VOCAB)
    assert table.thresholds == DEFAULT_THRESHOLDS


def test_custom_thresholds_are_respected() -> None:
    examples = _golden_examples()
    scores = _golden_scores()
    table = eval_metrics(examples, scores, vocabulary=VOCAB, thresholds=(0.5, 0.9))
    assert table.thresholds == (0.5, 0.9)
    assert all(len(row.cells) == 2 for row in table.rows)
    # t=0.5 Bills: TP=1 FP=1 FN=1 (the golden's 0.5 column)
    assert table.rows[0].cells[0].precision == 0.5
    assert table.rows[0].cells[0].recall == 0.5


# ---- determinism / alignment ----


def test_input_order_invariant() -> None:
    examples = _golden_examples()
    scores = _golden_scores()
    forward = eval_metrics(examples, scores, vocabulary=VOCAB)
    backward = eval_metrics(list(reversed(examples)), scores, vocabulary=VOCAB)
    assert forward == backward  # byte-identical tables


def test_alignment_by_id_not_position() -> None:
    # the scores mapping is keyed by example id: dict key order must not
    # matter (a positional implementation would read rows in insertion
    # order and misalign)
    examples = _golden_examples()
    scores = _golden_scores()
    reordered = dict(reversed(list(scores.items())))
    assert eval_metrics(examples, scores, vocabulary=VOCAB) == eval_metrics(
        examples, reordered, vocabulary=VOCAB
    )


def test_score_swap_between_ids_changes_the_table() -> None:
    # a determinism-flavored check (NOT the positional pin — that is
    # test_alignment_by_id_not_position): swapping two ids' score rows
    # (across DIFFERENT labels) MUST move the metrics. ex0 carries
    # Bills, ex2 carries Shipping.
    examples = _golden_examples()
    scores = _golden_scores()
    swapped = dict(scores)
    swapped["0000000000000000"] = scores["0000000000000002"]
    swapped["0000000000000002"] = scores["0000000000000000"]
    baseline = eval_metrics(examples, scores, vocabulary=VOCAB)
    mutated = eval_metrics(examples, swapped, vocabulary=VOCAB)
    assert baseline != mutated
    # Bills t=0.5: baseline ex0=0.9 TP → P=0.5 R=0.5; mutated ex0=0.2
    # (FN) → no TP at all → P=0.0 R=0.0
    assert baseline.rows[0].cells[4].precision == 0.5
    assert baseline.rows[0].cells[4].recall == 0.5
    assert mutated.rows[0].cells[4].precision == 0.0
    assert mutated.rows[0].cells[4].recall == 0.0
    # Shipping t=0.5: baseline ex2=0.8 → TP=1 FP=2 → P=1/3 R=1.0;
    # mutated ex2=0.1 (the swap moved the WHOLE row) → no TP → P=0.0
    # R=0.0
    assert baseline.rows[1].cells[4].precision == 1 / 3
    assert baseline.rows[1].cells[4].recall == 1.0
    assert mutated.rows[1].cells[4].precision == 0.0
    assert mutated.rows[1].cells[4].recall == 0.0


# ---- refusals (fail-loud doctrine) ----


def test_unsorted_vocabulary_refused() -> None:
    with pytest.raises(ValueError, match="sorted"):
        eval_metrics(
            _golden_examples(), _golden_scores(), vocabulary=("Shipping", "Bills")
        )


def test_duplicate_example_id_refused() -> None:
    examples = _golden_examples()
    examples.append(examples[0])
    with pytest.raises(ValueError, match="duplicate"):
        eval_metrics(examples, _golden_scores(), vocabulary=VOCAB)


def test_missing_score_refused() -> None:
    # an example with no score row: the table would silently undercount —
    # refuse, never guess
    scores = dict(_golden_scores())
    del scores["0000000000000002"]
    with pytest.raises(ValueError, match="no score"):
        eval_metrics(_golden_examples(), scores, vocabulary=VOCAB)


def test_unknown_score_id_refused() -> None:
    # a score row for an id that is not an example: a lying vector
    scores = dict(_golden_scores())
    scores["ffffffffffffffff"] = (0.5, 0.5)
    with pytest.raises(ValueError, match="not an example"):
        eval_metrics(_golden_examples(), scores, vocabulary=VOCAB)


def test_misaligned_score_row_refused() -> None:
    scores = dict(_golden_scores())
    scores["0000000000000000"] = (0.9,)  # one score for two labels
    with pytest.raises(ValueError, match="align"):
        eval_metrics(_golden_examples(), scores, vocabulary=VOCAB)


def test_out_of_range_score_refused() -> None:
    scores = dict(_golden_scores())
    scores["0000000000000000"] = (1.5, 0.1)  # not a sigmoid output
    with pytest.raises(ValueError, match="sigmoid"):
        eval_metrics(_golden_examples(), scores, vocabulary=VOCAB)


def test_threshold_out_of_range_refused() -> None:
    for bad in ((0.0,), (1.0,)):
        with pytest.raises(ValueError, match="threshold"):
            eval_metrics(
                _golden_examples(), _golden_scores(), vocabulary=VOCAB, thresholds=bad
            )


def test_non_ascending_thresholds_refused() -> None:
    with pytest.raises(ValueError, match="ascending"):
        eval_metrics(
            _golden_examples(),
            _golden_scores(),
            vocabulary=VOCAB,
            thresholds=(0.5, 0.5),
        )
    with pytest.raises(ValueError, match="ascending"):
        eval_metrics(
            _golden_examples(),
            _golden_scores(),
            vocabulary=VOCAB,
            thresholds=(0.9, 0.1),
        )


def test_empty_thresholds_refused() -> None:
    with pytest.raises(ValueError, match="threshold"):
        eval_metrics(
            _golden_examples(), _golden_scores(), vocabulary=VOCAB, thresholds=()
        )


def test_empty_vocabulary_refused() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        eval_metrics(_golden_examples(), {}, vocabulary=())


def test_non_tenth_thresholds_refused() -> None:
    # the text report labels columns to one decimal: 0.11 and 0.12
    # would BOTH render "0.1" — mislabeled columns are refused, never
    # rendered (probed: the grid is a grid of tenths)
    for bad in ((0.11, 0.12), (0.15, 0.5), (0.33,)):
        with pytest.raises(ValueError, match="tenth"):
            eval_metrics(
                _golden_examples(), _golden_scores(), vocabulary=VOCAB, thresholds=bad
            )


def test_label_outside_vocabulary_refused() -> None:
    # defense-in-depth beyond the builder's guarantee: a label with no
    # score column would be silently unmeasured — refuse
    examples = [_ex(0, ("Bills", "Ghost"))]
    scores = {"0000000000000000": (0.9, 0.1)}
    with pytest.raises(ValueError, match="outside the vocabulary"):
        eval_metrics(examples, scores, vocabulary=VOCAB)


# ---- report rendering (deterministic, diff-friendly) ----


def _render(table) -> str:
    return render_report(
        table, split_date=SPLIT_DATE, split_manifest_hash=MANIFEST_HASH
    )


def test_render_report_golden() -> None:
    # format spec: split-provenance header (split date + manifest hash
    # short form), then the summary line; label column = max label
    # length + 1 (no truncation); support right-aligned width 7; one
    # 10-char column per threshold — header right-aligned "0.1", cell
    # "P/R" as "0.50/1.00" (two decimals each, P first); a convention
    # footer naming the zero-denominator rendering
    table = eval_metrics(_golden_examples(), _golden_scores(), vocabulary=VOCAB)
    assert _render(table) == (
        "eval: split 2026-01-01T00:00:00+00:00 manifest=0000000000000000\n"
        "eval: test examples=4 labels=2 thresholds=0.1..0.9\n"
        "label     support       0.1       0.2       0.3       0.4       "
        "0.5       0.6       0.7       0.8       0.9\n"
        "Bills           2 0.50/1.00 0.50/1.00 0.67/1.00 0.67/1.00 "
        "0.50/0.50 1.00/0.50 1.00/0.50 1.00/0.50 1.00/0.50\n"
        "Shipping        1 0.25/1.00 0.33/1.00 0.33/1.00 0.33/1.00 "
        "0.33/1.00 0.50/1.00 1.00/1.00 1.00/1.00 0.00/0.00\n"
        "precision/recall render 0.00 where the denominator is zero; "
        "see tp/fp/fn in --json\n"
    )


def test_render_report_deterministic_across_reorder() -> None:
    # byte-stability across input order: two tables built from reordered
    # inputs render byte-identically
    examples = _golden_examples()
    scores = _golden_scores()
    forward = eval_metrics(examples, scores, vocabulary=VOCAB)
    backward = eval_metrics(list(reversed(examples)), scores, vocabulary=VOCAB)
    assert _render(forward) == _render(backward)


def test_render_report_long_label_no_truncation() -> None:
    # no cap, no truncation: a label longer than the old 64-char bound
    # must still line up — the row's support column stays under the
    # header's (a width cap would shift the row right of the header)
    long_label = "A" * 70
    examples = [_ex(0, (long_label,))]
    scores = {"0000000000000000": (0.9,)}
    table = eval_metrics(examples, scores, vocabulary=(long_label,))
    report = _render(table)
    lines = report.splitlines()
    assert long_label in lines[3]
    # precise pin: the row is the label (width = max+1, one trailing
    # space) + the same column tail the header carries, so the total
    # line width matches the header's — the columns stay aligned
    header = lines[2]
    row = lines[3]
    assert row.startswith(long_label + " ")
    assert len(row) == len(header)


def test_render_report_empty_thresholds_refused() -> None:
    # argument validation, not an IndexError: an empty grid refuses
    from gmail_labeler.eval import EvalTable

    table = EvalTable(vocabulary=VOCAB, thresholds=(), test_examples=0, rows=())
    with pytest.raises(ValueError, match="thresholds"):
        _render(table)


def test_vacuous_table_is_all_zero() -> None:
    # no test-side examples: the table is vacuous (all-zero cells,
    # test_examples=0), not a refusal — the CLI guards the empty case,
    # the core pins its behavior for future callers (#21)
    table = eval_metrics([], {}, vocabulary=VOCAB)
    assert table.test_examples == 0
    assert [row.label for row in table.rows] == list(VOCAB)
    for row in table.rows:
        assert row.support == 0
        assert all(
            c.tp == 0
            and c.fp == 0
            and c.fn == 0
            and c.precision == 0.0
            and c.recall == 0.0
            for c in row.cells
        )
