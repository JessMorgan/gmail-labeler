"""Per-label Graduation gates (ticket #22): golden determinism on
hand-built metric tables (the test_analyze / test_eval / test_thresholds
pattern — in-memory inputs, no I/O, no clock), gate-edge strictness
(analyze's strict-gt precedent: a value EXACTLY at a gate passes),
named reasons, refusals, and the Graduation Table (versioned JSON +
content hash + identity-binding refusals, the SplitManifest discipline).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, date, datetime
from typing import cast

import pytest

from gmail_labeler.dataset_build import TrainingExample
from gmail_labeler.eval import (
    DEFAULT_THRESHOLDS,
    EvalTable,
    eval_metrics,
)
from gmail_labeler.graduation import (
    DEFAULT_MIN_EXAMPLES,
    DEFAULT_MIN_PRECISION,
    GRADUATION_FORMAT_VERSION,
    GateTargets,
    Graduation,
    GraduationRefusedError,
    GraduationTable,
    build_table,
    evaluate_graduation,
    render_graduation_diff,
    verify_graduation,
)
from gmail_labeler.model import vocabulary_hash
from gmail_labeler.thresholds import BandsArtifact, build_artifact, calibrate_bands

VOCAB = ("Bills", "Shipping")
NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
SPLIT_DATE = datetime(2026, 1, 1, tzinfo=UTC)
MANIFEST_HASH = "0" * 64
CORPUS_FP = "f" * 64
ENCODER = "tiny-encoder"


def _ex(i: int, labels: tuple[str, ...]) -> TrainingExample:
    return TrainingExample(
        id=f"{i:016x}",
        date=NOW,
        labels=labels,
        weight=1.0,
        date_provenance="internal_date",
    )


def _golden_examples() -> list[TrainingExample]:
    # Bills positives: nine at 0.95, one at 0.5 (support = 10 — EXACTLY
    # the default min_examples gate)
    # x0: unlabeled, Bills score 0.99 — the one FP that keeps P at the
    # high band EXACTLY at the default min_precision gate (9/10 = 0.9)
    # Shipping positives: five at 0.4 (support = 5 — below the gate)
    # y0/y1: unlabeled, Shipping score 0.45 — Shipping P = 5/7 at t<=0.4
    # (0.714 < 0.9: no high band), 0.0 above 0.4
    return [
        *[_ex(i, ("Bills",)) for i in range(9)],
        _ex(9, ("Bills",)),
        _ex(10, ()),
        *[_ex(i, ("Shipping",)) for i in range(11, 16)],
        _ex(16, ()),
        _ex(17, ()),
    ]


def _golden_scores() -> dict[str, tuple[float, ...]]:
    scores = {f"{i:016x}": (0.95, 0.0) for i in range(9)}
    scores["0000000000000009"] = (0.5, 0.0)
    scores["000000000000000a"] = (0.99, 0.0)
    for i in range(11, 16):
        scores[f"{i:016x}"] = (0.0, 0.4)
    scores["0000000000000010"] = (0.0, 0.45)
    scores["0000000000000011"] = (0.0, 0.45)
    return scores


def _golden_table() -> EvalTable:
    return eval_metrics(_golden_examples(), _golden_scores(), vocabulary=VOCAB)


def _golden_bands() -> BandsArtifact:
    # defaults (high 0.9 / med 0.7):
    #   Bills: P(t) = 10/11 (t<=0.5), 9/10 = 0.9 (0.6..0.9) → high = 0.9
    #   (P EXACTLY 0.9), med = 0.8
    #   Shipping: P(t) = 5/7 (t<=0.4), 0.0 (0.5..0.8), UNDEFINED (0.9)
    #   → high = none, med = 0.4
    bands = calibrate_bands(_golden_table())
    return build_artifact(
        bands,
        vocabulary=VOCAB,
        corpus_fingerprint=CORPUS_FP,
        split_manifest_hash=MANIFEST_HASH,
        encoder_id=ENCODER,
        split_date=SPLIT_DATE,
    )


def _table(graduation: Graduation, **overrides: object) -> GraduationTable:
    # the eval identity (test_examples) and the bands binding are
    # DERIVED from the graduation's provenance — build_table no longer
    # takes caller-told values
    kwargs: dict[str, object] = {
        "vocabulary": VOCAB,
        "corpus_fingerprint": CORPUS_FP,
        "split_manifest_hash": MANIFEST_HASH,
        "encoder_id": ENCODER,
        "split_date": SPLIT_DATE,
    }
    kwargs.update(overrides)
    return build_table(graduation, **kwargs)  # type: ignore[arg-type]


# ---- the golden graduation (hand-computed, independent of the code) ----
#
# Bills: support 10 (EXACTLY the default min_examples gate), high = 0.9
#   with measured precision EXACTLY 0.9 (the default min_precision gate)
#   → graduated (both gates pass AT the boundary — analyze's strict-gt
#   precedent: a value exactly at a gate passes, not per intuition)
# Shipping: support 5 (< 10), no high band (med 0.4 alone never
#   graduates) → not graduated, two named reasons


def test_golden_graduation() -> None:
    grad = evaluate_graduation(
        _golden_table(), _golden_bands(), managed_labels=list(VOCAB)
    )
    assert grad.vocabulary == VOCAB
    assert grad.candidate_thresholds == DEFAULT_THRESHOLDS
    assert grad.test_examples == 18
    assert grad.bands_content_hash == _golden_bands().content_hash
    assert [entry.label for entry in grad.labels] == list(VOCAB)  # total order

    bills = grad.labels[0]
    assert bills.examples == 10
    assert bills.high == 0.9
    assert bills.med == 0.8
    assert bills.precision_at_high == pytest.approx(0.9)
    assert bills.min_examples == 10
    assert bills.min_precision == 0.9
    assert bills.graduated is True
    assert bills.reasons == ()

    shipping = grad.labels[1]
    assert shipping.examples == 5
    assert shipping.high is None
    assert shipping.med == 0.4  # a med band — and it graduates nothing
    assert shipping.precision_at_high is None
    assert shipping.graduated is False
    assert shipping.reasons == ("below-min-examples", "no-high-band")

    # provenance: the per-label EFFECTIVE targets ride along (defaults
    # here — no overrides)
    assert grad.targets == (
        ("Bills", GateTargets(10, 0.9)),
        ("Shipping", GateTargets(10, 0.9)),
    )
    assert grad.default_targets == GateTargets(10, 0.9)


def test_defaults_are_named_constants() -> None:
    assert DEFAULT_MIN_EXAMPLES == 10
    assert DEFAULT_MIN_PRECISION == 0.9


def test_gate_edge_strictness_examples_at_gate_graduates() -> None:
    # analyze's operator precedent (analyze.py: `total < min_total`):
    # a value EXACTLY at the gate passes — support == min_examples
    # graduates (a `<=` implementation is red here)
    grad = evaluate_graduation(
        _golden_table(), _golden_bands(), managed_labels=["Bills"]
    )
    assert grad.labels[0].examples == grad.labels[0].min_examples == 10
    assert grad.labels[0].graduated is True


def test_gate_edge_strictness_precision_at_gate_graduates() -> None:
    # analyze's operator precedent (analyze.py: `dup_rate > max_dup_rate`):
    # a value EXACTLY at the gate passes — precision == min_precision
    # graduates (a `>` implementation is red here)
    grad = evaluate_graduation(
        _golden_table(), _golden_bands(), managed_labels=["Bills"]
    )
    assert grad.labels[0].precision_at_high == pytest.approx(0.9)
    assert grad.labels[0].min_precision == 0.9
    assert grad.labels[0].graduated is True


def test_examples_one_below_gate_fails() -> None:
    # the boundary from the other side: min_examples = 11 puts the
    # golden Bills (support 10) below the gate
    grad = evaluate_graduation(
        _golden_table(),
        _golden_bands(),
        min_examples=11,
        managed_labels=["Bills"],
    )
    entry = grad.labels[0]
    assert entry.graduated is False
    assert entry.reasons == ("below-min-examples",)


def test_precision_just_below_gate_fails() -> None:
    # the boundary from the other side: min_precision = 0.91 puts the
    # golden Bills (P at high = 0.9 exactly) below the gate
    grad = evaluate_graduation(
        _golden_table(),
        _golden_bands(),
        min_precision=0.91,  # NOT a tenth — gate targets are not grid-bound
        managed_labels=["Bills"],
    )
    entry = grad.labels[0]
    assert entry.graduated is False
    assert entry.reasons == ("precision-below-gate",)


def test_med_band_never_graduates_by_itself() -> None:
    # Shipping carries a med band (0.4) and no high band: the med band
    # is not a graduation path — the reason is no-high-band, full stop
    grad = evaluate_graduation(
        _golden_table(), _golden_bands(), managed_labels=["Shipping"]
    )
    entry = grad.labels[0]
    assert entry.med == 0.4
    assert entry.high is None
    assert entry.graduated is False
    assert "no-high-band" in entry.reasons


def test_managed_labels_only_are_in_the_table() -> None:
    # the table covers the Managed Labels, not the whole vocabulary
    grad = evaluate_graduation(
        _golden_table(), _golden_bands(), managed_labels=["Shipping"]
    )
    assert [entry.label for entry in grad.labels] == ["Shipping"]


def test_deterministic_total_order_label_sorted() -> None:
    # the entries are in label-sorted order (the vocabulary is
    # canonically sorted — a total order, input-order invariant)
    grad = evaluate_graduation(
        _golden_table(), _golden_bands(), managed_labels=["Shipping", "Bills"]
    )
    assert [entry.label for entry in grad.labels] == ["Bills", "Shipping"]


def test_byte_identical_table_across_runs() -> None:
    # AC5: same eval → byte-identical table (deterministic total order —
    # no clock, no I/O nondeterminism in the core)
    a = _table(
        evaluate_graduation(
            _golden_table(), _golden_bands(), managed_labels=list(VOCAB)
        )
    )
    b = _table(
        evaluate_graduation(
            _golden_table(), _golden_bands(), managed_labels=list(VOCAB)
        )
    )
    assert a.to_json() == b.to_json()
    assert a.content_hash == b.content_hash


def test_per_label_override_applied() -> None:
    # a per-label gate override changes only that label's gates (and the
    # effective targets are visible per label — provenance)
    grad = evaluate_graduation(
        _golden_table(),
        _golden_bands(),
        overrides={"Bills": GateTargets(min_examples=5, min_precision=0.95)},
        managed_labels=list(VOCAB),
    )
    bills, shipping = grad.labels
    assert bills.min_examples == 5
    assert bills.min_precision == 0.95
    # P at high is 0.9 < 0.95 → the override flips the golden graduate
    assert bills.graduated is False
    assert bills.reasons == ("precision-below-gate",)
    # Shipping: defaults unchanged
    assert shipping.min_examples == 10
    assert shipping.min_precision == 0.9


# ---- refusals (fail-loud doctrine) ----


def test_empty_managed_labels_refused() -> None:
    with pytest.raises(ValueError, match="managed_labels must not be empty"):
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=[])


def test_duplicate_managed_labels_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_graduation(
            _golden_table(), _golden_bands(), managed_labels=["Bills", "Bills"]
        )


def test_managed_label_outside_vocabulary_refused() -> None:
    with pytest.raises(ValueError, match="Ghost"):
        evaluate_graduation(
            _golden_table(), _golden_bands(), managed_labels=["Bills", "Ghost"]
        )


def test_override_for_label_outside_vocabulary_refused() -> None:
    with pytest.raises(ValueError, match="Ghost"):
        evaluate_graduation(
            _golden_table(),
            _golden_bands(),
            overrides={"Ghost": GateTargets(min_examples=1, min_precision=0.5)},
            managed_labels=list(VOCAB),
        )


def test_bands_vocabulary_mismatch_refused() -> None:
    # the bands were calibrated on a DIFFERENT vocabulary than the table
    # being measured — a lying identity (the artifact carries the
    # vocabulary as its label entries)
    other_table = eval_metrics(
        [_ex(0, ("Bills",))],
        {"0000000000000000": (0.9,)},
        vocabulary=("Bills",),
    )
    lying = build_artifact(
        calibrate_bands(other_table),
        vocabulary=("Bills",),
        corpus_fingerprint=CORPUS_FP,
        split_manifest_hash=MANIFEST_HASH,
        encoder_id=ENCODER,
        split_date=SPLIT_DATE,
    )
    with pytest.raises(ValueError, match="vocabulary"):
        evaluate_graduation(_golden_table(), lying, managed_labels=list(VOCAB))


def test_bands_grid_mismatch_refused() -> None:
    # the bands' candidate grid does not match the table's thresholds —
    # the high/med thresholds would not be grid points of the table
    lying = dataclasses.replace(_golden_bands(), candidate_thresholds=(0.1, 0.2, 0.3))
    with pytest.raises(ValueError, match="candidate"):
        evaluate_graduation(_golden_table(), lying, managed_labels=list(VOCAB))


def test_negative_min_examples_refused() -> None:
    with pytest.raises(ValueError, match="min_examples"):
        evaluate_graduation(
            _golden_table(), _golden_bands(), min_examples=-1, managed_labels=VOCAB
        )


def test_bool_min_examples_refused() -> None:
    with pytest.raises(ValueError, match="min_examples"):
        evaluate_graduation(
            _golden_table(),
            _golden_bands(),
            min_examples=True,  # type: ignore[arg-type]
            managed_labels=VOCAB,
        )


def test_out_of_range_min_precision_refused() -> None:
    for value in (-0.1, 1.5):
        with pytest.raises(ValueError, match="min_precision"):
            evaluate_graduation(
                _golden_table(),
                _golden_bands(),
                min_precision=value,
                managed_labels=VOCAB,
            )


def test_bool_min_precision_refused() -> None:
    with pytest.raises(ValueError, match="min_precision"):
        evaluate_graduation(
            _golden_table(),
            _golden_bands(),
            min_precision=True,  # type: ignore[arg-type]
            managed_labels=VOCAB,
        )


def test_bands_high_band_without_predictions_on_table_refused() -> None:
    # the bands artifact is a SEPARATE input: if it was calibrated on a
    # DIFFERENT eval table (same vocabulary + grid), the cell at the
    # high band may have NO predictions on this table (tp=fp=0,
    # undefined precision) — refused with ValueError (the refusal
    # family), never a ZeroDivisionError traceback
    table_a = eval_metrics(
        [_ex(0, ("Bills",))],
        {"0000000000000000": (0.95,)},
        vocabulary=("Bills",),
    )
    bands = build_artifact(
        calibrate_bands(table_a),
        vocabulary=("Bills",),
        corpus_fingerprint=CORPUS_FP,
        split_manifest_hash=MANIFEST_HASH,
        encoder_id=ENCODER,
        split_date=SPLIT_DATE,
    )
    assert bands.labels[0].high == 0.9
    # table B: the positive scores below the high band, no predictions
    # at 0.9 (tp=fp=0)
    table_b = eval_metrics(
        [_ex(0, ("Bills",)), _ex(1, ())],
        {
            "0000000000000000": (0.5,),
            "0000000000000001": (0.4,),
        },
        vocabulary=("Bills",),
    )
    with pytest.raises(ValueError, match="no measured predictions"):
        evaluate_graduation(table_b, bands, managed_labels=["Bills"])


def test_float_min_examples_refused() -> None:
    # min_examples is a COUNT: a float (even an integral one) is not an
    # integer — the two-entry-path symmetry (validate vs from_json)
    # requires the mint to refuse what the parse refuses
    for value in (10.5, 10.0):
        with pytest.raises(ValueError, match="min_examples"):
            evaluate_graduation(
                _golden_table(),
                _golden_bands(),
                min_examples=value,  # type: ignore[arg-type]
                managed_labels=VOCAB,
            )


def test_override_target_refused() -> None:
    # the override targets get the SAME validation as the run's — an
    # out-of-range per-label target would miscalibrate silently
    for override in (
        GateTargets(min_examples=-1, min_precision=0.9),
        GateTargets(min_examples=10, min_precision=1.5),
    ):
        with pytest.raises(ValueError):
            evaluate_graduation(
                _golden_table(),
                _golden_bands(),
                overrides={"Bills": override},
                managed_labels=list(VOCAB),
            )


# ---- the Graduation Table artifact (the SplitManifest discipline) ----


def test_table_identity_fields() -> None:
    # AC3: the table carries the six eval identity fields + the Bands
    # Artifact's content hash it was computed from
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    assert table.format_version == GRADUATION_FORMAT_VERSION == 1
    assert table.vocabulary_hash == vocabulary_hash(VOCAB)
    assert table.corpus_fingerprint == CORPUS_FP
    assert table.split_manifest_hash == MANIFEST_HASH
    assert table.encoder_id == ENCODER
    assert table.split_date == SPLIT_DATE
    assert table.test_examples == 18
    assert table.bands_content_hash == _golden_bands().content_hash
    assert table.candidate_thresholds == DEFAULT_THRESHOLDS
    assert table.default_targets == GateTargets(10, 0.9)


def test_table_roundtrip_from_json() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    parsed = GraduationTable.from_json(table.to_json())
    assert parsed == table
    assert parsed.content_hash == table.content_hash


def test_table_hash_mismatch_refused() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    lying = dataclasses.replace(table, content_hash="1" * 64)
    with pytest.raises(ValueError, match="content_hash"):
        lying.validate()
    with pytest.raises(ValueError, match="content_hash"):
        GraduationTable.from_json(lying.to_json())


def test_table_unknown_format_version_refused() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    lying = dataclasses.replace(table, format_version=99)
    with pytest.raises(ValueError, match="format_version"):
        lying.validate()


def test_table_naive_split_date_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _table(
            evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB),
            split_date=datetime(2026, 1, 1),
        )
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    naive = dataclasses.replace(table, split_date=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="UTC"):
        naive.validate()


def test_table_vocabulary_mismatch_refused() -> None:
    # a caller-told vocabulary that does not match the graduation's is a
    # lying identity — refused at mint
    grad = evaluate_graduation(
        _golden_table(), _golden_bands(), managed_labels=list(VOCAB)
    )
    with pytest.raises(ValueError, match="vocabulary"):
        build_table(
            grad,
            vocabulary=("Bills",),
            corpus_fingerprint=CORPUS_FP,
            split_manifest_hash=MANIFEST_HASH,
            encoder_id=ENCODER,
            split_date=SPLIT_DATE,
        )


def test_table_lies_about_graduated_refused() -> None:
    # a machine-checked artifact may not contradict itself: an entry
    # marked graduated while carrying a reason (or vice versa) is
    # corrupt, not a valid variant
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    lying = dataclasses.replace(
        table,
        labels=tuple(
            dataclasses.replace(e, graduated=not e.graduated) for e in table.labels
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        lying.validate()


def test_table_lies_about_reasons_refused() -> None:
    # a duplicated/missing reason would pass a set equality — the
    # machine-checked boundary must not accept a lying count
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    shipping = table.labels[1]
    lying_entry = dataclasses.replace(
        shipping, reasons=("below-min-examples", "below-min-examples")
    )
    lying = dataclasses.replace(table, labels=(table.labels[0], lying_entry))
    with pytest.raises(ValueError, match="do not match"):
        lying.validate()


def test_table_precision_at_high_must_track_high() -> None:
    # precision_at_high is None iff the high band is None — a measured
    # precision with no high band (or a high band with no measurement)
    # is self-contradictory
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    bills = table.labels[0]
    lying = dataclasses.replace(
        table,
        labels=(dataclasses.replace(bills, precision_at_high=None), table.labels[1]),
    )
    with pytest.raises(ValueError, match="precision_at_high"):
        lying.validate()
    shipping = table.labels[1]
    lying2 = dataclasses.replace(
        table,
        labels=(table.labels[0], dataclasses.replace(shipping, precision_at_high=0.5)),
    )
    with pytest.raises(ValueError, match="precision_at_high"):
        lying2.validate()


def test_table_band_thresholds_must_be_grid_points() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    bills = table.labels[0]
    lying = dataclasses.replace(
        table,
        labels=(dataclasses.replace(bills, high=0.95), table.labels[1]),
    )
    with pytest.raises(ValueError, match="not a candidate threshold"):
        lying.validate()


def test_table_bad_hash_field_refused() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    for field in ("vocabulary_hash", "corpus_fingerprint", "split_manifest_hash"):
        lying = dataclasses.replace(table, **{field: "x" * 64})
        with pytest.raises(ValueError, match="64-char hex"):
            lying.validate()
    lying = dataclasses.replace(table, bands_content_hash="")
    with pytest.raises(ValueError, match="64-char hex"):
        lying.validate()


def test_table_empty_encoder_id_refused() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    lying = dataclasses.replace(table, encoder_id="   ")
    with pytest.raises(ValueError, match="encoder_id"):
        lying.validate()


def test_table_negative_test_examples_refused() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    lying = dataclasses.replace(table, test_examples=-1)
    with pytest.raises(ValueError, match="test_examples"):
        lying.validate()


def test_table_unsorted_labels_refused() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    lying = dataclasses.replace(table, labels=(table.labels[1], table.labels[0]))
    with pytest.raises(ValueError, match="sorted and unique"):
        lying.validate()


def test_table_malformed_json_refused() -> None:
    for payload in (
        "[1, 2, 3]",  # not an object
        json.dumps({"format_version": 1}),  # missing fields
        json.dumps(
            {
                "format_version": 1,
                "vocabulary_hash": "0" * 64,
                "corpus_fingerprint": "f" * 64,
                "split_manifest_hash": "0" * 64,
                "encoder_id": "tiny-encoder",
                "split_date": "2026-01-01T00:00:00+00:00",
                "test_examples": 18,
                "bands_content_hash": "0" * 64,
                "candidate_thresholds": [0.1, 0.2],
                "default_targets": {"min_examples": 10, "min_precision": 0.9},
                "labels": [],  # empty labels
                "content_hash": "0" * 64,
            }
        ),
    ):
        with pytest.raises(ValueError):
            GraduationTable.from_json(payload)


# ---- verify_graduation (the consumer check, the sidecar doctrine) ----


def _verify_with(table: GraduationTable, overrides: dict[str, object]) -> None:
    """verify_graduation with the matching identity + the given
    overrides (the test_thresholds pattern — explicit casts at the
    boundary)."""
    kwargs: dict[str, object] = {
        "vocabulary": VOCAB,
        "corpus_fingerprint": CORPUS_FP,
        "split_manifest_hash": MANIFEST_HASH,
        "encoder_id": ENCODER,
        "split_date": SPLIT_DATE,
        "test_examples": 18,
        "bands_content_hash": _golden_bands().content_hash,
    }
    kwargs.update(overrides)
    verify_graduation(
        table,
        vocabulary=cast("tuple[str, ...]", kwargs["vocabulary"]),
        corpus_fingerprint=cast("str", kwargs["corpus_fingerprint"]),
        split_manifest_hash=cast("str", kwargs["split_manifest_hash"]),
        encoder_id=cast("str", kwargs["encoder_id"]),
        split_date=cast("datetime | date", kwargs["split_date"]),
        test_examples=cast("int", kwargs["test_examples"]),
        bands_content_hash=cast("str", kwargs["bands_content_hash"]),
    )


def test_verify_graduation_passes_on_matching_identity() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    _verify_with(table, {})  # no refusal


def test_verify_graduation_names_every_disagreeing_field() -> None:
    # every field checked INDEPENDENTLY; ALL disagreements collected in
    # one GraduationRefusedError (a single-field drift is refused on its
    # own — no cross-field coupling)
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    with pytest.raises(GraduationRefusedError) as ei:
        _verify_with(
            table,
            {
                "vocabulary": ("Bills",),
                "corpus_fingerprint": "e" * 64,
                "split_manifest_hash": "1" * 64,
                "encoder_id": "other-encoder",
                "split_date": datetime(2026, 2, 1, tzinfo=UTC),
                "test_examples": 17,
                "bands_content_hash": "2" * 64,
            },
        )
    assert ei.value.fields == (
        "vocabulary",
        "corpus_fingerprint",
        "split_manifest_hash",
        "encoder_id",
        "split_date",
        "test_examples",
        "bands_content_hash",
    )
    # a single-field drift is refused on its own
    with pytest.raises(GraduationRefusedError) as ei2:
        _verify_with(table, {"test_examples": 17})
    assert ei2.value.fields == ("test_examples",)
    assert isinstance(ei2.value, ValueError)  # the CLI catches the family


def test_verify_graduation_naive_split_date_is_a_caller_bug() -> None:
    # a naive-dated split_date is a CALLER bug, not identity drift:
    # refused with a plain ValueError naming it (the verify_bands rule)
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    with pytest.raises(ValueError, match="caller bug"):
        _verify_with(table, {"split_date": datetime(2026, 1, 1)})


# ---- render_graduation_diff (the human-confirm surface) ----


def test_diff_baseline_empty_when_no_existing_table() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    diff = render_graduation_diff(None, table, path="grad.json")
    assert "no existing table at grad.json" in diff
    # the everything-vs-nothing baseline: every label is (absent) on the
    # old side
    assert "Bills: (absent) -> graduated" in diff
    assert (
        "Shipping: (absent) -> not graduated (below-min-examples, no-high-band)" in diff
    )


def test_diff_shows_old_to_new_per_label() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    # the old table: Bills NOT graduated (a stricter gate), Shipping
    # graduated — the diff must show both directions
    old = dataclasses.replace(
        table,
        labels=tuple(
            dataclasses.replace(
                e,
                graduated=not e.graduated,
                reasons=("below-min-examples",) if e.graduated else (),
            )
            for e in table.labels
        ),
    )
    # fix the hash so the old table is self-consistent
    old = dataclasses.replace(old, content_hash=old.compute_hash())
    diff = render_graduation_diff(old, table, path="grad.json")
    assert "Bills: not graduated (below-min-examples) -> graduated" in diff
    assert (
        "Shipping: graduated -> not graduated (below-min-examples, no-high-band)"
        in diff
    )


def test_diff_names_absent_labels() -> None:
    # a label in the new table that the old one does not carry is named
    # as absent, not silently dropped
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    old = dataclasses.replace(
        table,
        labels=(table.labels[0],),
        content_hash="",
    )
    old = dataclasses.replace(old, content_hash=old.compute_hash())
    diff = render_graduation_diff(old, table, path="grad.json")
    assert "(absent) -> not graduated (below-min-examples, no-high-band)" in diff


def test_diff_names_identity_disagreement() -> None:
    # a foreign-bound table (identity drift) is named in the diff — the
    # human sees they are comparing a table bound to a different
    # checkpoint/eval
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    diff = render_graduation_diff(
        table, table, path="grad.json", identity_disagreement=("split_date",)
    )
    assert (
        "existing table at grad.json is bound to a different identity "
        "(split_date)" in diff
    )
    assert "foreign-bound" in diff


def test_diff_is_deterministic() -> None:
    table = _table(
        evaluate_graduation(_golden_table(), _golden_bands(), managed_labels=VOCAB)
    )
    a = render_graduation_diff(None, table, path="grad.json")
    b = render_graduation_diff(None, table, path="grad.json")
    assert a == b
