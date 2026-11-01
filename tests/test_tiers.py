"""tiers.py (ticket #23): the single would-apply decision core.

Hand-golden boundary tests pin the band operators (exactly-on-threshold
PASSES — #21's inclusive ``>=`` doctrine, mutation-pinned), the
skip-present consequence (an already-present label is not proposed and
earns NO Tier Label), the graduated-only rule (labels without a
graduated entry are never proposed), the System Label refusal, and the
deterministic total order.
"""

from __future__ import annotations

import pytest

from gmail_labeler.tiers import (
    TIER_APPLIED,
    TIER_HIGH,
    TIER_MED,
    best_tier,
    tier_for_band,
    would_apply,
)

# Golden bands (hand-computed, independent of the code): the calibrated
# per-label Threshold Bands a Graduation Table would carry.
#   Shipping:    high = 0.9, med = 0.8
#   Money/Bills: high = 0.8, med = 0.7


def test_score_exactly_at_high_earns_high_band() -> None:
    # the boundary: score EXACTLY at the high threshold PASSES (>=)
    decisions = would_apply(
        scores={"Shipping": 0.9},
        bands={"Shipping": (0.9, 0.8)},
        graduated=frozenset({"Shipping"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    assert len(decisions) == 1
    d = decisions[0]
    assert d.label == "Shipping"
    assert d.score == 0.9
    assert d.band == "high"
    assert d.tier == TIER_HIGH
    assert d.present is False
    assert d.applied is True


def test_score_exactly_at_med_earns_med_band() -> None:
    # the boundary: score EXACTLY at the med threshold PASSES (>=) —
    # just below high, so the med band, not the high band
    decisions = would_apply(
        scores={"Shipping": 0.8},
        bands={"Shipping": (0.9, 0.8)},
        graduated=frozenset({"Shipping"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    d = decisions[0]
    assert d.band == "med"
    assert d.tier == TIER_MED
    assert d.applied is True


def test_score_below_med_earns_no_band() -> None:
    # just below the med threshold: no band, no tier, not applied
    decisions = would_apply(
        scores={"Shipping": 0.79},
        bands={"Shipping": (0.9, 0.8)},
        graduated=frozenset({"Shipping"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    d = decisions[0]
    assert d.band is None
    assert d.tier is None
    assert d.applied is False


def test_high_band_outranks_med_band() -> None:
    # a score above BOTH thresholds earns the HIGH band (the higher
    # verdict), never the med
    decisions = would_apply(
        scores={"Shipping": 0.95},
        bands={"Shipping": (0.9, 0.8)},
        graduated=frozenset({"Shipping"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    assert decisions[0].band == "high"
    assert decisions[0].tier == TIER_HIGH


def test_no_high_band_falls_through_to_med() -> None:
    # a label whose band has no high threshold (the no-high-band state)
    # can still earn the med band
    decisions = would_apply(
        scores={"Money/Bills": 0.75},
        bands={"Money/Bills": (None, 0.7)},
        graduated=frozenset({"Money/Bills"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    d = decisions[0]
    assert d.band == "med"
    assert d.tier == TIER_MED


def test_skip_present_not_proposed_and_no_tier() -> None:
    # invariant #2's posture: a predicted label ALREADY PRESENT is not
    # proposed (applied=False) and earns NO Tier Label — the band
    # verdict is still visible (the report's ALREADY-PRESENT outcome)
    decisions = would_apply(
        scores={"Shipping": 0.95},
        bands={"Shipping": (0.9, 0.8)},
        graduated=frozenset({"Shipping"}),
        present=frozenset({"Shipping"}),
        system_labels=frozenset(),
    )
    d = decisions[0]
    assert d.band == "high"  # the verdict is visible
    assert d.present is True
    assert d.applied is False  # not proposed
    assert d.tier is None  # NO Tier Label


def test_labels_without_graduated_entry_never_proposed() -> None:
    # a label the Graduation Table did not graduate is never proposed —
    # it does not even appear in the decisions (no silent zero)
    decisions = would_apply(
        scores={"Shipping": 0.99, "Money/Bills": 0.99},
        bands={"Shipping": (0.9, 0.8), "Money/Bills": (0.8, 0.7)},
        graduated=frozenset({"Shipping"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    assert [d.label for d in decisions] == ["Shipping"]
    assert decisions[0].applied is True


def test_empty_graduated_proposes_nothing() -> None:
    # nothing graduated → no decisions (a legitimate state, not a
    # refusal: the run is a read-only survey)
    decisions = would_apply(
        scores={"Shipping": 0.99},
        bands={"Shipping": (0.9, 0.8)},
        graduated=frozenset(),
        present=frozenset(),
        system_labels=frozenset(),
    )
    assert decisions == ()


def test_system_labels_never_targeted() -> None:
    # a System Label that somehow carries a graduated entry is refused,
    # named — System Labels are never content labels, never touchable
    with pytest.raises(ValueError, match="INBOX"):
        would_apply(
            scores={"INBOX": 0.9},
            bands={"INBOX": (0.9, 0.8)},
            graduated=frozenset({"INBOX"}),
            present=frozenset(),
            system_labels=frozenset({"INBOX"}),
        )


def test_deterministic_total_order() -> None:
    # the decisions are in label-sorted order (a total order) — the
    # input mapping order must not leak into the output
    scores = {"Money/Bills": 0.85, "Shipping": 0.95, "Zeta": 0.99}
    bands = {
        "Money/Bills": (0.8, 0.7),
        "Shipping": (0.9, 0.8),
        "Zeta": (0.9, 0.8),
    }
    d1 = would_apply(
        scores=scores,
        bands=bands,
        graduated=frozenset(scores),
        present=frozenset(),
        system_labels=frozenset(),
    )
    d2 = would_apply(
        scores=dict(reversed(list(scores.items()))),
        bands=dict(reversed(list(bands.items()))),
        graduated=frozenset(scores),
        present=frozenset(),
        system_labels=frozenset(),
    )
    assert d1 == d2
    assert [d.label for d in d1] == ["Money/Bills", "Shipping", "Zeta"]


def test_score_out_of_range_is_refused() -> None:
    # a score outside [0, 1] (a lying model row) is refused, named
    with pytest.raises(ValueError, match="Shipping"):
        would_apply(
            scores={"Shipping": 1.5},
            bands={"Shipping": (0.9, 0.8)},
            graduated=frozenset({"Shipping"}),
            present=frozenset(),
            system_labels=frozenset(),
        )
    with pytest.raises(ValueError, match="Shipping"):
        would_apply(
            scores={"Shipping": -0.1},
            bands={"Shipping": (0.9, 0.8)},
            graduated=frozenset({"Shipping"}),
            present=frozenset(),
            system_labels=frozenset(),
        )


def test_bool_score_is_refused() -> None:
    # a bool is not a score (the config doctrine's type policing)
    with pytest.raises(ValueError, match="Shipping"):
        would_apply(
            scores={"Shipping": True},  # type: ignore[dict-item]
            bands={"Shipping": (0.9, 0.8)},
            graduated=frozenset({"Shipping"}),
            present=frozenset(),
            system_labels=frozenset(),
        )


def test_band_out_of_range_is_refused() -> None:
    # a band threshold outside (0, 1) is refused, named
    with pytest.raises(ValueError, match="Shipping"):
        would_apply(
            scores={"Shipping": 0.9},
            bands={"Shipping": (1.0, 0.8)},
            graduated=frozenset({"Shipping"}),
            present=frozenset(),
            system_labels=frozenset(),
        )
    with pytest.raises(ValueError, match="Shipping"):
        would_apply(
            scores={"Shipping": 0.9},
            bands={"Shipping": (0.9, 0.0)},
            graduated=frozenset({"Shipping"}),
            present=frozenset(),
            system_labels=frozenset(),
        )


def test_inverted_bands_are_refused() -> None:
    # med >= high is self-contradictory (the bands artifact's invariant,
    # mirrored here — a lying input is refused, not guessed at)
    with pytest.raises(ValueError, match="Shipping"):
        would_apply(
            scores={"Shipping": 0.9},
            bands={"Shipping": (0.8, 0.9)},
            graduated=frozenset({"Shipping"}),
            present=frozenset(),
            system_labels=frozenset(),
        )


def test_graduated_label_without_score_is_refused() -> None:
    # a graduated label the scores do not carry is a lying input — the
    # model scores the whole vocabulary, so its absence is a defect
    with pytest.raises(ValueError, match="Money/Bills"):
        would_apply(
            scores={"Shipping": 0.9},
            bands={"Shipping": (0.9, 0.8), "Money/Bills": (0.8, 0.7)},
            graduated=frozenset({"Shipping", "Money/Bills"}),
            present=frozenset(),
            system_labels=frozenset(),
        )


def test_graduated_label_without_bands_is_refused() -> None:
    # a graduated label the bands do not carry is a lying input (the
    # bands artifact covers the vocabulary)
    with pytest.raises(ValueError, match="Money/Bills"):
        would_apply(
            scores={"Shipping": 0.9, "Money/Bills": 0.9},
            bands={"Shipping": (0.9, 0.8)},
            graduated=frozenset({"Shipping", "Money/Bills"}),
            present=frozenset(),
            system_labels=frozenset(),
        )


def test_best_tier_prefers_high_over_med() -> None:
    # the per-message Tier Label: the best band among the APPLIED
    # decisions (high outranks med); skip-present decisions never
    # contribute (they earned no Tier Label)
    decisions = would_apply(
        scores={"Shipping": 0.95, "Money/Bills": 0.75},
        bands={"Shipping": (0.9, 0.8), "Money/Bills": (0.8, 0.7)},
        graduated=frozenset({"Shipping", "Money/Bills"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    assert best_tier(decisions) == TIER_HIGH
    med_only = would_apply(
        scores={"Money/Bills": 0.75},
        bands={"Money/Bills": (0.8, 0.7)},
        graduated=frozenset({"Money/Bills"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    assert best_tier(med_only) == TIER_MED
    # nothing applied (all skip-present) → no tier
    skip = would_apply(
        scores={"Shipping": 0.95},
        bands={"Shipping": (0.9, 0.8)},
        graduated=frozenset({"Shipping"}),
        present=frozenset({"Shipping"}),
        system_labels=frozenset(),
    )
    assert best_tier(skip) is None
    assert best_tier(()) is None


@pytest.mark.parametrize(
    ("band", "present", "expected"),
    [
        ("high", False, TIER_HIGH),
        ("med", False, TIER_MED),
        (None, False, None),
        # skip-present (invariant #2): an already-present label earns NO
        # Tier Label — the band stays visible, the tier does not
        ("high", True, None),
        ("med", True, None),
        (None, True, None),
    ],
)
def test_tier_for_band_golden(band, present, expected) -> None:
    # the band→tier mapping (the single home #24/#26 reuse): high →
    # ai-high, med → ai-med, no band → None; skip-present → None
    assert tier_for_band(band, present=present) == expected


def test_system_labels_is_required_keyword_only() -> None:
    # T1 + disposition 9: system_labels has NO fail-open default AND is
    # KEYWORD-ONLY — calling would_apply without it is a TypeError
    # (required), and passing it POSITIONALLY is also a TypeError
    # (keyword-only enforced), so no caller setup can be forgotten (the
    # module is the single safety authority for #24/#26)
    with pytest.raises(TypeError, match="system_labels"):
        would_apply(  # pyright: ignore[reportCallIssue] — deliberate
            scores={"Shipping": 0.9},
            bands={"Shipping": (0.9, 0.8)},
            graduated=frozenset({"Shipping"}),
            present=frozenset(),
        )
    # positional-pass is also a TypeError (keyword-only is literally true)
    with pytest.raises(TypeError, match="positional"):
        would_apply(  # pyright: ignore[reportCallIssue] — deliberate
            {"Shipping": 0.9},  # pyright: ignore[reportCallIssue]
            {"Shipping": (0.9, 0.8)},  # pyright: ignore[reportCallIssue]
            frozenset({"Shipping"}),  # pyright: ignore[reportCallIssue]
            frozenset(),  # pyright: ignore[reportCallIssue]
            frozenset(),  # pyright: ignore[reportCallIssue]
        )


def test_high_only_band_golden() -> None:
    # (high, None): a label whose band has no med threshold — the high
    # band is still earned at/above high
    decisions = would_apply(
        scores={"Shipping": 0.9},
        bands={"Shipping": (0.9, None)},
        graduated=frozenset({"Shipping"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    assert decisions[0].band == "high"
    assert decisions[0].tier == TIER_HIGH
    # just below high with no med → no band
    below = would_apply(
        scores={"Shipping": 0.89},
        bands={"Shipping": (0.9, None)},
        graduated=frozenset({"Shipping"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    assert below[0].band is None
    assert below[0].tier is None


def test_no_band_golden() -> None:
    # (None, None): a label with no band at all — never proposed
    decisions = would_apply(
        scores={"Shipping": 0.99},
        bands={"Shipping": (None, None)},
        graduated=frozenset({"Shipping"}),
        present=frozenset(),
        system_labels=frozenset(),
    )
    assert decisions[0].band is None
    assert decisions[0].tier is None
    assert decisions[0].applied is False


def test_med_only_skip_present_gate() -> None:
    # med-only band (high=None) + skip-present: the med band is earned
    # but the label is present → not applied, NO Tier Label (the
    # mutation case: flipping the skip-present gate would propose it)
    decisions = would_apply(
        scores={"Money/Bills": 0.75},
        bands={"Money/Bills": (None, 0.7)},
        graduated=frozenset({"Money/Bills"}),
        present=frozenset({"Money/Bills"}),
        system_labels=frozenset(),
    )
    d = decisions[0]
    assert d.band == "med"  # the verdict is visible
    assert d.present is True
    assert d.applied is False  # NOT proposed (skip-present)
    assert d.tier is None  # NO Tier Label


def test_wrong_arity_band_is_named() -> None:
    # a band entry with the wrong arity (a bare number, not a
    # (high, med) pair) is refused, named — not a silent unpack
    with pytest.raises(ValueError, match="Shipping"):
        would_apply(
            scores={"Shipping": 0.9},
            bands={"Shipping": 0.9},  # type: ignore[dict-item]
            graduated=frozenset({"Shipping"}),
            present=frozenset(),
            system_labels=frozenset(),
        )


@pytest.mark.parametrize("score", [float("nan"), None])
def test_nan_none_score_is_refused(score) -> None:
    # a NaN or None score is refused, named (a lying/absent model row)
    with pytest.raises(ValueError, match="Shipping"):
        would_apply(
            scores={"Shipping": score},  # type: ignore[dict-item]
            bands={"Shipping": (0.9, 0.8)},
            graduated=frozenset({"Shipping"}),
            present=frozenset(),
            system_labels=frozenset(),
        )


def test_tier_constants_are_pinned_to_the_glossary_spellings() -> None:
    # the GLOSSARY's Tier Label spellings (the [labels] config defaults
    # mirror these — the equality is pinned by test_config). This pin is
    # REAL: it parses GLOSSARY.md and asserts the three spellings appear
    # (a GLOSSARY drift turns this red, not just the constant check)
    from pathlib import Path

    glossary = (Path(__file__).resolve().parent.parent / "GLOSSARY.md").read_text(
        encoding="utf-8"
    )
    assert TIER_APPLIED == "ai-applied"
    assert TIER_HIGH == "ai-high"
    assert TIER_MED == "ai-med"
    assert "ai-applied" in glossary
    assert "ai-high" in glossary
    assert "ai-med" in glossary
