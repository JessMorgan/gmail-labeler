"""Per-label threshold calibration (ticket #21): golden determinism on
hand-built metric tables (the test_analyze / test_eval pattern — in-memory
inputs, no I/O, no clock), band-edge strictness, zero-support cells,
refusals, and the Bands Artifact (versioned JSON + content hash +
identity-binding refusals, the SplitManifest discipline).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from typing import cast

import pytest

from gmail_labeler.dataset_build import TrainingExample
from gmail_labeler.eval import (
    DEFAULT_THRESHOLDS,
    EvalTable,
    LabelEvalRow,
    ThresholdCell,
    eval_metrics,
)
from gmail_labeler.model import vocabulary_hash
from gmail_labeler.thresholds import (
    BANDS_FORMAT_VERSION,
    DEFAULT_MIN_HIGH_PRECISION,
    DEFAULT_MIN_MED_PRECISION,
    Bands,
    BandsArtifact,
    BandsLabelEntry,
    BandsRefusedError,
    BandTargets,
    build_artifact,
    calibrate_bands,
    verify_bands,
)

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


def _golden_table() -> EvalTable:
    return eval_metrics(_golden_examples(), _golden_scores(), vocabulary=VOCAB)


def _artifact(bands: Bands, **overrides: object) -> BandsArtifact:
    # the eval identity (test_examples) and the default targets are
    # DERIVED from the bands' provenance (calibrate_bands' inputs) —
    # build_artifact no longer takes caller-told values
    kwargs: dict[str, object] = {
        "vocabulary": VOCAB,
        "corpus_fingerprint": CORPUS_FP,
        "split_manifest_hash": MANIFEST_HASH,
        "encoder_id": ENCODER,
        "split_date": SPLIT_DATE,
    }
    kwargs.update(overrides)
    return build_artifact(bands, **kwargs)  # type: ignore[arg-type]


# ---- the golden bands (hand-computed, independent of the code) ----
#
# Bills precision by threshold (tp/fp from the golden table):
#   0.1: 2/4 = 0.5   0.2: 2/4 = 0.5   0.3: 2/3   0.4: 2/3
#   0.5: 1/2 = 0.5   0.6: 1/1 = 1.0   0.7: 1.0   0.8: 1.0   0.9: 1.0
# Shipping precision:
#   0.1: 1/4 = 0.25  0.2..0.5: 1/3   0.6: 1/2 = 0.5   0.7: 1.0   0.8: 1.0
#   0.9: tp=0 fp=0 — UNDEFINED (the zero-denominator cell)


def test_golden_bands() -> None:
    bands = calibrate_bands(_golden_table())
    assert bands.vocabulary == VOCAB
    assert bands.candidate_thresholds == DEFAULT_THRESHOLDS
    assert [lb.label for lb in bands.labels] == list(VOCAB)  # total order

    bills = bands.labels[0]
    shipping = bands.labels[1]
    # Bills: high = highest t with P >= 0.9 → 0.9; med = highest t < 0.9
    # with P >= 0.7 → 0.8
    assert bills.high == 0.9
    assert bills.med == 0.8
    assert bills.reasons == ()
    # Shipping: high = highest t with P >= 0.9 → 0.8 (0.9 is undefined);
    # med = highest t < 0.8 with P >= 0.7 → 0.7 (P=1.0)
    assert shipping.high == 0.8
    assert shipping.med == 0.7
    assert shipping.reasons == ()

    # provenance: the per-label EFFECTIVE targets ride along (defaults
    # here — no overrides)
    assert bands.targets == (
        ("Bills", BandTargets(0.9, 0.7)),
        ("Shipping", BandTargets(0.9, 0.7)),
    )


def test_defaults_are_named_constants() -> None:
    assert DEFAULT_MIN_HIGH_PRECISION == 0.9
    assert DEFAULT_MIN_MED_PRECISION == 0.7


def test_band_edge_strictness_ge_not_gt() -> None:
    # a cell whose precision is EXACTLY the target earns the band (>= is
    # the pinned boundary; a > implementation is red here)
    # Bills: t=0.5 → tp=1 fp=1 → P = 0.5 exactly
    table = _golden_table()
    bands = calibrate_bands(table, min_high_precision=0.5)
    bills = bands.labels[0]
    # highest t with P >= 0.5: 0.9 (P=1.0) — the edge cell (0.5) is what
    # a > implementation would lose at the target itself; pin it directly
    # with a table where the edge cell is the highest qualifying one
    assert bills.high == 0.9
    # direct edge pin: target 1.0 → only the P=1.0 cells qualify
    bands2 = calibrate_bands(table, min_high_precision=1.0)
    assert bands2.labels[0].high == 0.9  # P(0.9)=1.0 exactly meets 1.0
    assert bands2.labels[1].high == 0.8  # Shipping P(0.8)=1.0; 0.9 undefined
    # med target exactly met: med 0.5 → Bills t=0.5 (P=0.5) qualifies
    bands3 = calibrate_bands(table, min_med_precision=0.5)
    assert bands3.labels[0].med == 0.8  # highest below high (0.9) with P>=0.5


def test_highest_qualifying_threshold_wins_not_first() -> None:
    # precision is NOT monotone in the threshold: a lower threshold can
    # qualify while a higher one also does — the HIGHEST wins (the band
    # is the strictest threshold the measured precision still earns)
    table = _golden_table()
    # Bills P >= 0.5 at 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 —
    # the first qualifier is 0.1, the highest is 0.9
    bands = calibrate_bands(table, min_high_precision=0.5)
    assert bands.labels[0].high == 0.9


def test_med_strictly_below_high() -> None:
    # the ordering constraint: med must be STRICTLY below high — a med
    # band equal to high would be empty (meaningless), and med > high
    # would invert the tiers. The highest med-qualifier at the same
    # threshold as high does NOT earn med; the next lower one does.
    # Bills: P >= 0.5 at 0.5 AND at 0.9; high=0.9; med must be < 0.9
    table = _golden_table()
    bands = calibrate_bands(table, min_high_precision=0.5, min_med_precision=0.5)
    bills = bands.labels[0]
    assert bills.high == 0.9
    assert bills.med == 0.8  # 0.8 < 0.9, P=1.0 >= 0.5
    # no threshold below high qualifies → med is an explicit none
    # (Bills: P=0.5 at every t < 0.9 — the unlabelled example scores
    # 0.85, an FP below 0.9 — and P=1.0 only at 0.9)
    examples = [_ex(0, ("Bills",)), _ex(1, ())]
    scores = {
        "0000000000000000": (0.9, 0.0),
        "0000000000000001": (0.85, 0.0),
    }
    table2 = eval_metrics(examples, scores, vocabulary=VOCAB)
    bands2 = calibrate_bands(table2, min_high_precision=1.0, min_med_precision=0.9)
    bills2 = bands2.labels[0]
    assert bills2.high == 0.9
    assert bills2.med is None
    # the reason must not LIE: t=0.9 meets the med target, it is simply
    # not below high — its own name
    assert bills2.reasons == ("no-candidate-below-high-meets-med-target",)


def test_med_unconstrained_when_high_not_earned() -> None:
    # high not earned → med is the highest qualifier on the whole grid
    # (no ordering constraint to violate)
    # Bills: tp=1, fp=1 at every threshold → P = 0.5 everywhere
    examples = [_ex(0, ("Bills",)), _ex(1, ())]
    scores = {
        "0000000000000000": (0.9, 0.0),
        "0000000000000001": (0.9, 0.0),
    }
    table = eval_metrics(examples, scores, vocabulary=VOCAB)
    bands = calibrate_bands(table, min_high_precision=0.7, min_med_precision=0.5)
    bills = bands.labels[0]
    assert bills.high is None
    assert bills.reasons == ("no-candidate-meets-high-target",)
    assert bills.med == 0.9  # highest qualifier, unconstrained


def test_zero_support_earns_no_band() -> None:
    # a label with no actual positives on the test side: every cell has
    # tp=0; where fp=0 the precision is UNDEFINED (not 0.0) and where
    # fp>0 it is 0.0 — neither meets a positive target. The no-band
    # state is EXPLICIT (named reasons), never a silent zero
    examples = [_ex(0, ())]
    scores = {"0000000000000000": (0.9, 0.0)}
    table = eval_metrics(examples, scores, vocabulary=VOCAB)
    bands = calibrate_bands(table)
    for lb in bands.labels:
        assert lb.high is None
        assert lb.med is None
        assert lb.reasons == (
            "no-candidate-meets-high-target",
            "no-candidate-meets-med-target",
        )


def test_undefined_precision_is_not_zero() -> None:
    # the zero-support pin (mutation): with a min-precision target of
    # 0.0, a cell with tp=0 AND fp=0 (no predictions at all) must NOT
    # qualify — its precision is undefined, not 0.0. An implementation
    # that reads the table's rendered precision (0.0 by the eval
    # convention) would meet the 0.0 target and is red here. The
    # calibration reads tp/fp/fn and derives the ratio itself.
    examples = [_ex(0, ())]
    scores = {"0000000000000000": (0.0, 0.0)}  # no predictions anywhere
    table = eval_metrics(examples, scores, vocabulary=VOCAB)
    bands = calibrate_bands(table, min_high_precision=0.0, min_med_precision=0.0)
    for lb in bands.labels:
        assert lb.high is None
        assert lb.med is None


def test_defined_zero_precision_meets_zero_target() -> None:
    # the counter-pin: a cell with tp=0, fp>0 HAS a defined precision of
    # 0.0, which meets a 0.0 target (the math is the math — the
    # operator asked for no precision floor)
    examples = [_ex(0, ("Bills",))]
    scores = {"0000000000000000": (0.0, 0.9)}  # Bills: no TP; Shipping: FP
    table = eval_metrics(examples, scores, vocabulary=VOCAB)
    bands = calibrate_bands(table, min_high_precision=0.0, min_med_precision=0.0)
    bills = bands.labels[0]
    # Bills: every cell tp=0, fp=0 → undefined → no band
    assert bills.high is None and bills.med is None
    shipping = bands.labels[1]
    # Shipping: tp=0, fp=1 at every t in 0.1..0.9 → P=0.0 meets 0.0
    assert shipping.high == 0.9
    assert shipping.med == 0.8


def test_per_label_targets_override() -> None:
    # per-label overrides change only that label's targets (the
    # effective targets ride in the output — provenance)
    table = _golden_table()
    bands = calibrate_bands(
        table,
        overrides={"Bills": BandTargets(min_high_precision=0.5, min_med_precision=0.5)},
    )
    assert bands.labels[0].high == 0.9  # Bills: P >= 0.5 up to 0.9
    assert bands.labels[0].med == 0.8
    assert bands.labels[1].high == 0.8  # Shipping: defaults (0.9/0.7)
    assert bands.labels[1].med == 0.7
    assert bands.targets == (
        ("Bills", BandTargets(0.5, 0.5)),
        ("Shipping", BandTargets(0.9, 0.7)),
    )


# ---- determinism ----


def test_input_order_invariant() -> None:
    forward = calibrate_bands(_golden_table())
    reordered = eval_metrics(
        list(reversed(_golden_examples())), _golden_scores(), vocabulary=VOCAB
    )
    assert calibrate_bands(reordered) == forward


def test_same_table_same_bands() -> None:
    assert calibrate_bands(_golden_table()) == calibrate_bands(_golden_table())


# ---- refusals (fail-loud doctrine) ----


def _one_cell_table(
    cells: tuple[ThresholdCell, ...], thresholds: tuple[float, ...]
) -> EvalTable:
    return EvalTable(
        vocabulary=VOCAB,
        thresholds=thresholds,
        test_examples=1,
        rows=(
            LabelEvalRow(label="Bills", support=1, cells=cells),
            LabelEvalRow(label="Shipping", support=1, cells=cells),
        ),
    )


def test_empty_vocabulary_refused() -> None:
    table = EvalTable(vocabulary=(), thresholds=(0.5,), test_examples=0, rows=())
    with pytest.raises(ValueError, match="vocabulary"):
        calibrate_bands(table)


def test_unsorted_vocabulary_refused() -> None:
    table = EvalTable(
        vocabulary=("Shipping", "Bills"),
        thresholds=(0.5,),
        test_examples=1,
        rows=(
            LabelEvalRow(label="Shipping", support=1, cells=()),
            LabelEvalRow(label="Bills", support=1, cells=()),
        ),
    )
    with pytest.raises(ValueError, match="sorted"):
        calibrate_bands(table)


def test_rows_must_match_vocabulary() -> None:
    # a row for a label outside the vocabulary (or a missing one) would
    # silently miscalibrate — refuse
    table = EvalTable(
        vocabulary=("Bills",),
        thresholds=(0.5,),
        test_examples=1,
        rows=(LabelEvalRow(label="Shipping", support=1, cells=()),),
    )
    with pytest.raises(ValueError, match="vocabulary"):
        calibrate_bands(table)


def test_rows_out_of_order_refused() -> None:
    table = EvalTable(
        vocabulary=VOCAB,
        thresholds=(0.5,),
        test_examples=1,
        rows=(
            LabelEvalRow(label="Shipping", support=1, cells=()),
            LabelEvalRow(label="Bills", support=1, cells=()),
        ),
    )
    with pytest.raises(ValueError, match="order"):
        calibrate_bands(table)


def test_empty_thresholds_refused() -> None:
    table = _one_cell_table((), ())
    with pytest.raises(ValueError, match="threshold"):
        calibrate_bands(table)


def test_non_tenth_thresholds_refused() -> None:
    # the bands are chosen from the candidate columns and the report
    # labels them to one decimal — a non-tenth grid refuses (the eval
    # stance, inherited)
    cells = (ThresholdCell(tp=1, fp=0, fn=0, precision=1.0, recall=1.0),)
    table = _one_cell_table(cells, (0.11, 0.5))
    with pytest.raises(ValueError, match="tenth"):
        calibrate_bands(table)


def test_non_ascending_thresholds_refused() -> None:
    cells = (ThresholdCell(tp=1, fp=0, fn=0, precision=1.0, recall=1.0),)
    table = _one_cell_table(cells, (0.5, 0.5))
    with pytest.raises(ValueError, match="ascending"):
        calibrate_bands(table)


def test_threshold_out_of_range_refused() -> None:
    cells = (ThresholdCell(tp=1, fp=0, fn=0, precision=1.0, recall=1.0),)
    for bad in ((0.0,), (1.0,)):
        with pytest.raises(ValueError, match="threshold"):
            calibrate_bands(_one_cell_table(cells, bad))


def test_cells_must_align_to_thresholds() -> None:
    table = EvalTable(
        vocabulary=VOCAB,
        thresholds=(0.5, 0.9),
        test_examples=1,
        rows=(
            LabelEvalRow(label="Bills", support=1, cells=()),
            LabelEvalRow(label="Shipping", support=1, cells=()),
        ),
    )
    with pytest.raises(ValueError, match="align"):
        calibrate_bands(table)


def test_target_out_of_range_refused() -> None:
    table = _golden_table()
    for bad in (1.5, -0.1):
        with pytest.raises(ValueError, match="precision"):
            calibrate_bands(table, min_high_precision=bad)
        with pytest.raises(ValueError, match="precision"):
            calibrate_bands(table, min_med_precision=bad)


def test_bool_target_refused() -> None:
    # True == 1.0 would pass 0.0 <= x <= 1.0 — a bool is not a target
    table = _golden_table()
    with pytest.raises(ValueError, match="precision"):
        calibrate_bands(table, min_high_precision=True)  # type: ignore[arg-type]


def test_non_tenth_target_refused() -> None:
    # the tenth-grid stance (#20): a target between grid points would
    # render mislabeled in any report of the bands — refuse
    table = _golden_table()
    with pytest.raises(ValueError, match="tenth"):
        calibrate_bands(table, min_high_precision=0.93)
    with pytest.raises(ValueError, match="tenth"):
        calibrate_bands(table, min_med_precision=0.65)


def test_unknown_override_label_refused() -> None:
    # an override for a label the table does not carry is refused,
    # named — a typo'd label must not silently calibrate nothing
    table = _golden_table()
    with pytest.raises(ValueError, match="Ghost"):
        calibrate_bands(
            table,
            overrides={
                "Ghost": BandTargets(min_high_precision=0.5, min_med_precision=0.5)
            },
        )


def test_override_targets_get_the_same_validation() -> None:
    # an override's targets are validated EXACTLY like the run's — an
    # out-of-range, bool, or non-tenth per-label target would
    # miscalibrate silently (the config layer shields the CLI path;
    # the core is the public boundary #23's shadow run consumes; the
    # daemon and #26 ride its OUTPUT (the Bands Artifact via
    # verify_bands), never calibrate_bands)
    table = _golden_table()
    for bad in (
        BandTargets(min_high_precision=1.5, min_med_precision=0.7),
        BandTargets(min_high_precision=0.9, min_med_precision=-0.1),
        BandTargets(min_high_precision=0.93, min_med_precision=0.7),
        BandTargets(min_high_precision=0.9, min_med_precision=0.65),
        BandTargets(min_high_precision=True, min_med_precision=0.7),  # type: ignore[arg-type]
    ):
        with pytest.raises(ValueError, match="overrides"):
            calibrate_bands(table, overrides={"Bills": bad})


# ---- the Bands Artifact (versioned JSON + content hash + identity) ----


def test_artifact_golden_fields() -> None:
    bands = calibrate_bands(_golden_table())
    artifact = _artifact(bands)
    assert artifact.format_version == BANDS_FORMAT_VERSION
    # checkpoint identity (the sidecar's fields, mirrored)
    assert artifact.vocabulary_hash == vocabulary_hash(VOCAB)
    assert artifact.corpus_fingerprint == CORPUS_FP
    assert artifact.split_manifest_hash == MANIFEST_HASH
    assert artifact.encoder_id == ENCODER
    # eval identity (the measured window)
    assert artifact.split_date == SPLIT_DATE
    assert artifact.test_examples == 4
    # provenance: the candidate grid + the defaults the run used
    assert artifact.candidate_thresholds == DEFAULT_THRESHOLDS
    assert artifact.default_targets == BandTargets(0.9, 0.7)
    # the per-label entries: bands + the EFFECTIVE targets
    assert [entry.label for entry in artifact.labels] == list(VOCAB)
    bills = artifact.labels[0]
    assert isinstance(bills, BandsLabelEntry)
    assert bills.high == 0.9 and bills.med == 0.8
    assert bills.min_high_precision == 0.9 and bills.min_med_precision == 0.7
    assert bills.reasons == ()
    shipping = artifact.labels[1]
    assert shipping.high == 0.8 and shipping.med == 0.7
    assert shipping.reasons == ()


def test_artifact_json_roundtrip() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    parsed = BandsArtifact.from_json(artifact.to_json())
    assert parsed == artifact


def test_artifact_json_is_canonical_and_byte_stable() -> None:
    # canonical dump (sorted keys, ASCII, compact): the content hash
    # rides in the payload (byte-stability across runs is pinned by
    # test_artifact_byte_identical_for_same_eval)
    artifact = _artifact(calibrate_bands(_golden_table()))
    text = artifact.to_json()
    payload = json.loads(text)
    assert payload["content_hash"] == artifact.content_hash
    # the hash covers every field except itself (the SplitManifest rule)
    assert artifact.content_hash == artifact.compute_hash()


def test_artifact_byte_identical_for_same_eval() -> None:
    # AC5: same eval → byte-identical artifact — incl. across a
    # reordered test side (the table's total order absorbs it)
    a1 = _artifact(calibrate_bands(_golden_table()))
    reordered = eval_metrics(
        list(reversed(_golden_examples())), _golden_scores(), vocabulary=VOCAB
    )
    a2 = _artifact(calibrate_bands(reordered))
    assert a1.to_json() == a2.to_json()


def test_artifact_override_visible_in_json() -> None:
    # a per-label override is visible in the artifact (provenance): the
    # entry carries the effective target, the defaults stay the run's
    bands = calibrate_bands(
        _golden_table(),
        overrides={"Bills": BandTargets(min_high_precision=0.5, min_med_precision=0.5)},
    )
    payload = json.loads(_artifact(bands).to_json())
    entries = {e["label"]: e for e in payload["labels"]}
    assert entries["Bills"]["min_high_precision"] == 0.5
    assert entries["Shipping"]["min_high_precision"] == 0.9
    assert payload["default_targets"] == {
        "min_high_precision": 0.9,
        "min_med_precision": 0.7,
    }


# ---- artifact format invariants (validate) ----


def _mutated(artifact: BandsArtifact, **fields: object) -> BandsArtifact:
    return dataclasses.replace(artifact, **fields)


def test_artifact_unknown_version_refused() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    with pytest.raises(ValueError, match="format_version"):
        _mutated(artifact, format_version=BANDS_FORMAT_VERSION + 1).validate()


def test_artifact_malformed_hash_refused() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    for field in ("vocabulary_hash", "corpus_fingerprint", "split_manifest_hash"):
        with pytest.raises(ValueError, match="hash"):
            _mutated(artifact, **{field: "x" * 64}).validate()
        with pytest.raises(ValueError, match="hash"):
            _mutated(artifact, **{field: ""}).validate()


def test_artifact_naive_split_date_refused() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    with pytest.raises(ValueError, match="UTC"):
        _mutated(artifact, split_date=SPLIT_DATE.replace(tzinfo=None)).validate()


def test_artifact_non_utc_split_date_refused() -> None:
    from datetime import timedelta

    artifact = _artifact(calibrate_bands(_golden_table()))
    local = datetime.fromtimestamp(0, tz=UTC).astimezone()
    if local.utcoffset() == timedelta(0):
        pytest.skip("host tz is UTC — the non-UTC offset is untestable")
    with pytest.raises(ValueError, match="UTC"):
        _mutated(artifact, split_date=SPLIT_DATE.astimezone(local.tzinfo)).validate()


def test_artifact_med_not_below_high_refused() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    bills = artifact.labels[0]
    for bad_high, bad_med in ((0.8, 0.8), (0.8, 0.9)):  # equal / inverted
        entry = BandsLabelEntry(
            label=bills.label,
            high=bad_high,
            med=bad_med,
            min_high_precision=bills.min_high_precision,
            min_med_precision=bills.min_med_precision,
            reasons=bills.reasons,
        )
        with pytest.raises(ValueError, match="med"):
            _mutated(artifact, labels=(entry, artifact.labels[1])).validate()


def test_artifact_reasons_must_match_bands() -> None:
    # a hash-consistent artifact may not contradict itself: an earned
    # band carrying a no-band reason (or a missing band without its
    # reason) is corrupt, not a valid variant
    artifact = _artifact(calibrate_bands(_golden_table()))
    bills = artifact.labels[0]
    # earned high + a lying high-reason
    lying = BandsLabelEntry(
        label=bills.label,
        high=bills.high,
        med=bills.med,
        min_high_precision=bills.min_high_precision,
        min_med_precision=bills.min_med_precision,
        reasons=("no-candidate-meets-high-target",),
    )
    with pytest.raises(ValueError, match="reasons"):
        _mutated(artifact, labels=(lying, artifact.labels[1])).validate()
    # missing med without its reason
    missing = BandsLabelEntry(
        label=bills.label,
        high=bills.high,
        med=None,
        min_high_precision=bills.min_high_precision,
        min_med_precision=bills.min_med_precision,
        reasons=(),
    )
    with pytest.raises(ValueError, match="reasons"):
        _mutated(artifact, labels=(missing, artifact.labels[1])).validate()
    # the correct reason for a missing med below an earned high
    correct = BandsLabelEntry(
        label=bills.label,
        high=bills.high,
        med=None,
        min_high_precision=bills.min_high_precision,
        min_med_precision=bills.min_med_precision,
        reasons=("no-candidate-below-high-meets-med-target",),
    )
    fixed = _mutated(artifact, labels=(correct, artifact.labels[1]))
    fixed = _mutated(
        fixed, content_hash=fixed.compute_hash()
    )  # the hash must cover the fix
    fixed.validate()  # no refusal


def test_artifact_band_outside_grid_refused() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    bills = artifact.labels[0]
    entry = BandsLabelEntry(
        label=bills.label,
        high=0.95,  # not a candidate column
        med=bills.med,
        min_high_precision=bills.min_high_precision,
        min_med_precision=bills.min_med_precision,
        reasons=bills.reasons,
    )
    with pytest.raises(ValueError, match="candidate"):
        _mutated(artifact, labels=(entry, artifact.labels[1])).validate()


def test_artifact_empty_labels_refused() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    with pytest.raises(ValueError, match="labels"):
        _mutated(artifact, labels=()).validate()


def test_artifact_unsorted_labels_refused() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    with pytest.raises(ValueError, match="sorted"):
        _mutated(artifact, labels=(artifact.labels[1], artifact.labels[0])).validate()


def test_artifact_duplicate_labels_refused() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    with pytest.raises(ValueError, match="sorted"):
        _mutated(artifact, labels=(artifact.labels[0], artifact.labels[0])).validate()


def test_artifact_content_hash_mismatch_refused() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    with pytest.raises(ValueError, match="content_hash"):
        _mutated(artifact, content_hash="0" * 64).validate()


def test_artifact_bool_test_examples_refused() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    with pytest.raises(ValueError, match="test_examples"):
        _mutated(artifact, test_examples=True).validate()  # type: ignore[arg-type]


def test_artifact_from_json_refuses_bad_shape() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    payload = json.loads(artifact.to_json())
    payload["labels"] = "not-a-list"
    with pytest.raises(ValueError, match="labels"):
        BandsArtifact.from_json(json.dumps(payload))
    payload = json.loads(artifact.to_json())
    del payload["encoder_id"]
    with pytest.raises(ValueError, match="encoder_id"):
        BandsArtifact.from_json(json.dumps(payload))
    with pytest.raises(ValueError, match="object"):
        BandsArtifact.from_json("[1, 2]")


def test_artifact_from_json_refuses_tampered_hash() -> None:
    # a content_hash that no longer matches the fields is stale or
    # corrupt — refused, never served (the SplitManifest rule)
    artifact = _artifact(calibrate_bands(_golden_table()))
    payload = json.loads(artifact.to_json())
    payload["test_examples"] = 99
    with pytest.raises(ValueError, match="content_hash"):
        BandsArtifact.from_json(json.dumps(payload))


def test_content_hash_covers_every_meaning_bearing_field() -> None:
    # hash-coverage pin (mutation class M9/M10): flipping or removing
    # EACH meaning-bearing field must change the content hash — a
    # hash computed from a constant default_targets, or with reasons
    # dropped from the payload, dies here
    artifact = _artifact(calibrate_bands(_golden_table()))
    base = artifact.compute_hash()
    entries = artifact.labels
    mutations: dict[str, BandsArtifact] = {
        "format_version": _mutated(
            artifact, format_version=artifact.format_version + 1
        ),
        "vocabulary_hash": _mutated(artifact, vocabulary_hash="1" * 64),
        "corpus_fingerprint": _mutated(artifact, corpus_fingerprint="2" * 64),
        "split_manifest_hash": _mutated(artifact, split_manifest_hash="3" * 64),
        "encoder_id": _mutated(artifact, encoder_id="other-encoder"),
        "split_date": _mutated(artifact, split_date=datetime(2026, 2, 1, tzinfo=UTC)),
        "test_examples": _mutated(artifact, test_examples=999),
        "candidate_thresholds": _mutated(
            artifact, candidate_thresholds=artifact.candidate_thresholds[:-1]
        ),
        "default_targets": _mutated(artifact, default_targets=BandTargets(0.8, 0.7)),
        "labels.high": _mutated(
            artifact,
            labels=(
                dataclasses.replace(entries[0], high=0.8),
                *entries[1:],
            ),
        ),
        "labels.reasons": _mutated(
            artifact,
            labels=(
                dataclasses.replace(
                    entries[1], reasons=("no-candidate-meets-high-target",)
                ),
                *entries[2:],
            ),
        ),
    }
    for field, mutated in mutations.items():
        assert mutated.compute_hash() != base, field


def test_reason_order_is_canonical_in_the_hash() -> None:
    # one meaning, one identity: the hash serializes reasons SORTED,
    # so a reversed-reasons artifact is byte-identical and validates —
    # validate's sorted comparison and the hash agree (both
    # directions: the reversed order validates, and the serialization
    # is byte-stable)
    examples = [_ex(0, ())]
    scores = {"0000000000000000": (0.0, 0.0)}
    table = eval_metrics(examples, scores, vocabulary=VOCAB)
    bands = calibrate_bands(table)
    a1 = _artifact(bands)
    assert all(entry.reasons for entry in a1.labels)  # no-band state
    reversed_entries = tuple(
        dataclasses.replace(entry, reasons=entry.reasons[::-1]) for entry in a1.labels
    )
    a2 = dataclasses.replace(a1, labels=reversed_entries)
    assert a2.compute_hash() == a1.content_hash  # same meaning, same hash
    a2.validate()  # the reversed order validates (the sorted comparison)
    assert a2.to_json() == a1.to_json()  # byte-identical


def test_artifact_blank_encoder_id_refused() -> None:
    # an empty/blank encoder_id would verify against an equally empty
    # expectation — refused at validate, not "verified"
    artifact = _artifact(calibrate_bands(_golden_table()))
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="encoder_id"):
            _mutated(artifact, encoder_id=bad).validate()


def test_artifact_from_json_refuses_blank_label_key() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    payload = json.loads(artifact.to_json())
    payload["labels"][0]["label"] = "  "
    with pytest.raises(ValueError, match="label"):
        BandsArtifact.from_json(json.dumps(payload))


# ---- identity verification (the refusal contract, AC2) ----


def _verify_with(artifact: BandsArtifact, overrides: dict[str, object]) -> None:
    """verify_bands with the matching identity + the given overrides
    (the typed-kwargs stand-in — pyright strict tests)."""
    kwargs: dict[str, object] = {
        "vocabulary": VOCAB,
        "corpus_fingerprint": CORPUS_FP,
        "split_manifest_hash": MANIFEST_HASH,
        "encoder_id": ENCODER,
        "split_date": SPLIT_DATE,
        "test_examples": 4,
    }
    kwargs.update(overrides)
    verify_bands(
        artifact,
        vocabulary=cast("tuple[str, ...]", kwargs["vocabulary"]),
        corpus_fingerprint=cast("str", kwargs["corpus_fingerprint"]),
        split_manifest_hash=cast("str", kwargs["split_manifest_hash"]),
        encoder_id=cast("str", kwargs["encoder_id"]),
        split_date=cast("datetime", kwargs["split_date"]),
        test_examples=cast("int", kwargs["test_examples"]),
    )


def test_verify_bands_accepts_matching_identity() -> None:
    artifact = _artifact(calibrate_bands(_golden_table()))
    _verify_with(artifact, {})  # no refusal


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vocabulary", ("Bills",)),  # vocabulary drift
        ("corpus_fingerprint", "e" * 64),
        ("split_manifest_hash", "d" * 64),
        ("encoder_id", "other-encoder"),
        ("split_date", datetime(2026, 2, 1, tzinfo=UTC)),
        ("test_examples", 5),
    ],
)
def test_verify_bands_each_field_refused_independently(
    field: str, value: object
) -> None:
    # each identity field is checked INDEPENDENTLY: a single-field drift
    # is refused on its own, named (the sidecar doctrine)
    artifact = _artifact(calibrate_bands(_golden_table()))
    with pytest.raises(BandsRefusedError) as excinfo:
        _verify_with(artifact, {field: value})
    assert excinfo.value.fields == (field,)
    assert field in str(excinfo.value)


def test_verify_bands_naive_split_date_is_a_caller_bug() -> None:
    # a naive split_date (or a bare date) is a CALLER bug, not
    # identity drift: build_artifact's naive refusal, mirrored — a
    # plain ValueError naming it, never a BandsRefusedError with the
    # "re-run eval --bands" remedy
    artifact = _artifact(calibrate_bands(_golden_table()))
    for bad in (SPLIT_DATE.replace(tzinfo=None), SPLIT_DATE.date()):
        with pytest.raises(ValueError, match="caller bug") as ei:
            _verify_with(artifact, {"split_date": bad})
        assert not isinstance(ei.value, BandsRefusedError)


def test_verify_bands_all_disagreements_named() -> None:
    # no first-fail: every disagreeing field is collected and named in
    # one refusal
    artifact = _artifact(calibrate_bands(_golden_table()))
    with pytest.raises(BandsRefusedError) as excinfo:
        _verify_with(artifact, {"encoder_id": "other-encoder", "test_examples": 5})
    assert excinfo.value.fields == ("encoder_id", "test_examples")


def test_build_artifact_vocabulary_mismatch_refused() -> None:
    # the artifact binds to the bands' vocabulary: a different
    # vocabulary argument is a lying identity — refuse
    bands = calibrate_bands(_golden_table())
    with pytest.raises(ValueError, match="vocabulary"):
        _artifact(bands, vocabulary=("Bills",))


def test_build_artifact_requires_aware_split_date() -> None:
    bands = calibrate_bands(_golden_table())
    with pytest.raises(ValueError, match="timezone-aware"):
        build_artifact(
            bands,
            vocabulary=VOCAB,
            corpus_fingerprint=CORPUS_FP,
            split_manifest_hash=MANIFEST_HASH,
            encoder_id=ENCODER,
            split_date=SPLIT_DATE.replace(tzinfo=None),
        )


def test_build_artifact_targets_labels_misaligned_refused() -> None:
    # the entries zip bands.labels and bands.targets POSITIONALLY — a
    # swapped targets tuple would mint per-label targets for the
    # WRONG labels, and validate() cannot see it (an entry's targets
    # are never re-checked against its label) — refuse the alignment
    bands = calibrate_bands(_golden_table())
    swapped = dataclasses.replace(bands, targets=tuple(reversed(bands.targets)))
    with pytest.raises(ValueError, match="align"):
        _artifact(swapped)


def test_build_artifact_derives_provenance_from_bands() -> None:
    # the eval identity (test_examples) and the default targets are
    # DERIVED from the calibration's provenance (calibrate_bands'
    # inputs) — a caller cannot tell build_artifact a lying identity
    table = _golden_table()
    bands = calibrate_bands(table, min_high_precision=0.5)
    artifact = build_artifact(
        bands,
        vocabulary=VOCAB,
        corpus_fingerprint=CORPUS_FP,
        split_manifest_hash=MANIFEST_HASH,
        encoder_id=ENCODER,
        split_date=SPLIT_DATE,
    )
    assert artifact.test_examples == table.test_examples == 4
    assert artifact.default_targets == BandTargets(0.5, 0.7)
