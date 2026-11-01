"""Model seam (GLOSSARY: Model) + FakeModel — the offline spine.

The seam contract pinned here: batch predict → per-label sigmoid
scores, rows aligned to the input texts, columns aligned to the
checkpoint's Label-Path vocabulary. The real encoder (torch, ADR-0001)
never appears in this module — the fake mirrors FakeGmailClient's role
(invariant #9: CI stays torch-free).
"""

from __future__ import annotations

import pytest

from gmail_labeler.model import FakeModel, Model, vocabulary_hash

# ---- predict: alignment + scores ----


def test_predict_returns_programmed_scores_aligned_to_inputs() -> None:
    model = FakeModel(
        ("A", "B"),
        scores={"t1": (0.9, 0.1), "t2": (0.2, 0.8)},
    )
    rows = model.predict(["t1", "t2"])
    assert rows == ((0.9, 0.1), (0.2, 0.8))


def test_predict_rows_align_to_input_order() -> None:
    # row i corresponds to texts[i] — a column-swap regression fails this
    model = FakeModel(("A", "B"), scores={"x": (0.1, 0.9), "y": (0.7, 0.3)})
    rows = model.predict(["y", "x"])
    assert rows[0] == (0.7, 0.3)  # y first
    assert rows[1] == (0.1, 0.9)  # x second


def test_predict_unprogrammed_text_gets_default() -> None:
    model = FakeModel(("A", "B"), scores={"x": (0.9, 0.1)}, default=0.25)
    (row,) = model.predict(["unseen"])
    assert row == (0.25, 0.25)


def test_predict_empty_batch_returns_no_rows() -> None:
    model = FakeModel(("A",), scores={})
    assert model.predict([]) == ()


def test_predict_unprogrammed_text_raises_without_default() -> None:
    # fail-loud: a silent 0.5 would fake a confidence the model never
    # had — unprogrammed text raises unless the caller passed default
    model = FakeModel(("A",), scores={"x": (0.9,)})
    with pytest.raises(ValueError, match="no score programmed"):
        model.predict(["unseen"])


def test_predict_generator_input_is_materialized_once() -> None:
    # a lazy iterable consumed twice (call log + row loop) would come
    # back empty the second time — rows must stay aligned to the log
    model = FakeModel(("A",), scores={"a": (0.5,), "b": (0.25,)})
    rows = model.predict(text for text in ("a", "b"))
    assert rows == ((0.5,), (0.25,))
    assert model.calls == [("a", "b")]


def test_predict_call_log_records_batches() -> None:
    model = FakeModel(("A",), scores={"a": (0.5,), "b": (0.4,), "c": (0.3,)})
    model.predict(["a", "b"])
    model.predict(["c"])
    assert model.calls == [("a", "b"), ("c",)]


# ---- vocabulary: exposure + canonical form ----


def test_vocabulary_is_exposed_as_sorted_tuple() -> None:
    model = FakeModel(("Money", "Money/Bills"), scores={})
    assert model.vocabulary == ("Money", "Money/Bills")
    assert isinstance(model.vocabulary, tuple)


def test_vocabulary_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValueError, match="sorted"):
        FakeModel(("B", "A"), scores={})
    with pytest.raises(ValueError, match="unique"):
        FakeModel(("A", "A"), scores={})


def test_vocabulary_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        FakeModel((), scores={})


# ---- the fake refuses to lie about the wire ----


def test_programmed_scores_must_align_to_vocabulary() -> None:
    with pytest.raises(ValueError, match="align"):
        FakeModel(("A", "B"), scores={"x": (0.5,)})  # one score, two labels
    with pytest.raises(ValueError, match="align"):
        FakeModel(("A",), scores={"x": (0.5, 0.5)})  # two scores, one label


def test_programmed_scores_must_be_sigmoid_range() -> None:
    with pytest.raises(ValueError, match="range"):
        FakeModel(("A",), scores={"x": (1.5,)})
    with pytest.raises(ValueError, match="range"):
        FakeModel(("A",), scores={"x": (-0.1,)})


def test_default_must_be_sigmoid_range() -> None:
    with pytest.raises(ValueError, match="default"):
        FakeModel(("A",), scores={}, default=2.0)


# ---- Protocol conformance (structural, pyright-enforced) ----


def _via_protocol(model: Model) -> tuple[tuple[float, ...], ...]:
    # the conformance PIN: a consumer annotated ``Model``. pyright
    # (strict on gmail_labeler.model) checks FakeModel against the
    # Protocol STRUCTURALLY — a predict return-type drift on the
    # Protocol fails this file at type-check time.
    return model.predict(["x"])


def test_fake_model_conforms_to_model_protocol() -> None:
    model = FakeModel(("A",), scores={"x": (0.9,)})
    assert _via_protocol(model) == ((0.9,),)


# ---- vocabulary hash (the sidecar's identity field) ----


def test_vocabulary_hash_is_canonical_and_pinned() -> None:
    # independent worked example: sha256 of the canonical JSON
    # '["A","B"]' (sorted keys, ASCII, compact separators)
    assert vocabulary_hash(("A", "B")) == (
        "b64e3448a83a5b86466465080361c1a7e1157a27ddccd4b68069cb18caffb74a"
    )
    assert vocabulary_hash(("Money", "Money/Bills")) == (
        "a39239a21631c4e33376ad44261762e746f160537d355aa3f1177342137fc9e4"
    )


def test_vocabulary_hash_distinguishes_vocabs() -> None:
    assert vocabulary_hash(("A", "B")) != vocabulary_hash(("A", "C"))
    assert vocabulary_hash(("A",)) != vocabulary_hash(("A", "B"))


def test_vocabulary_hash_is_order_sensitive() -> None:
    # order-sensitive BY DESIGN: a reordered vocabulary must refuse —
    # the hash is the drift detector, not an order-insensitive set hash
    assert vocabulary_hash(("A", "B")) != vocabulary_hash(("B", "A"))


def test_model_module_is_torch_free() -> None:
    # AC#1's guard: importing the seam must not pull torch in (the
    # real encoder stays behind the train/ recipe's lazy import).
    # Subprocess on purpose: a torch import elsewhere in this session
    # must not mask a regression here.
    import subprocess
    import sys

    code = (
        "import sys, gmail_labeler.model; "
        "sys.exit(0 if 'torch' not in sys.modules else 1)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
