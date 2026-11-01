"""Tier Label decisions (ticket #23; GLOSSARY: Tier Label, Shadow Pass,
Application).

Pure core (CODING-STANDARDS): deterministic on any input — no clock,
no I/O; the output order is total and stable (the analyze discipline).

THIS IS THE SINGLE WOULD-APPLY DECISION CORE. ``shadow run`` (ticket
#23) records its verdicts, and the live Application path (tickets
#24/#26) MUST import it — "one core, no second implementation" is
#26's AC. A second band-verdict implementation anywhere in the package
is the bug class this module exists to prevent.

Band-selection rule (pinned by golden tests — the operators MIRROR
``thresholds.py``'s inclusive ``>=`` doctrine, #21):

- ``score >= high`` → the high band (a score EXACTLY at the high
  threshold earns it);
- else ``score >= med`` → the med band (a score EXACTLY at the med
  threshold earns it);
- else no band (the GLOSSARY's "none" — the explicit no-verdict state).

Consequences pinned by the same golden tests:

- GRADUATED LABELS ONLY: a label without a graduated entry is never
  proposed — it does not even appear in the decisions (no silent zero);
- skip-present (invariant #2): a predicted label that is ALREADY
  PRESENT is not proposed (``applied=False``) and earns NO Tier Label
  (``tier=None``) — the band verdict stays visible so the report can
  render the ALREADY-PRESENT outcome;
- System Labels are never targeted: a System Label carrying a
  graduated entry is refused, named (fail-loud — the graduated set is
  trusted from the Graduation Table, and the table's vocabulary is
  content labels only, so this is a lying input);
- the decisions are in label-sorted order (a total order — the input
  mapping order never leaks into the output).

The Tier Label constants are pinned to the GLOSSARY's spellings
(``ai-applied`` umbrella, ``ai-high``, ``ai-med``); the ``[labels]``
config defaults mirror them (the equality is pinned by test_config,
not by an import — config.py stays import-light).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "BAND_HIGH",
    "BAND_MED",
    "TIER_APPLIED",
    "TIER_HIGH",
    "TIER_MED",
    "TierDecision",
    "best_tier",
    "would_apply",
]

#: The Tier Label spellings (GLOSSARY: Tier Label) — pinned here so the
#: decision core and every consumer name the same labels.
TIER_APPLIED = "ai-applied"  # the umbrella: "the AI contributed something"
TIER_HIGH = "ai-high"  # the high-confidence band
TIER_MED = "ai-med"  # the review-me band

#: The band verdict names (the GLOSSARY's Threshold Band: high,
#: medium, or none — "none" is the explicit ``band is None`` state).
BAND_HIGH = "high"
BAND_MED = "med"


@dataclass(frozen=True, slots=True)
class TierDecision:
    """One graduated label's would-apply verdict for one message:

    - ``band``: the band verdict (``"high"`` / ``"med"`` / ``None`` —
      the explicit no-verdict state);
    - ``tier``: the Tier Label the verdict would carry (``TIER_HIGH`` /
      ``TIER_MED``) — ``None`` when there is no band OR when the label
      is skip-present (an already-present label earns NO Tier Label);
    - ``present``: the label is already on the message (skip-present
      input fact);
    - ``applied``: the label WOULD be applied — a band verdict AND not
      present (invariant #2: present labels are never re-applied).
    """

    label: str
    score: float
    band: str | None
    tier: str | None
    present: bool
    applied: bool


def _check_score(label: str, score: object) -> float:
    # type() policing (the config doctrine): a bool is not a score, and
    # a lying model row (out of sigmoid range) is refused, named
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(
            f"tiers for {label!r}: score must be a number in [0, 1], got {score!r}"
        )
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"tiers for {label!r}: score {score} is out of [0, 1] — a lying model row"
        )
    return float(score)


def _check_band_value(label: str, name: str, value: object) -> float | None:
    # the same shape/range doctrine as the bands artifact (mirrored —
    # a lying input is refused, not guessed at): thresholds in (0, 1)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"tiers for {label!r}: {name} band must be a number in "
            f"(0, 1) or None, got {value!r}"
        )
    if not 0.0 < value < 1.0:
        raise ValueError(f"tiers for {label!r}: {name} band {value} is not in (0, 1)")
    return float(value)


def _check_bands(label: str, high: object, med: object) -> None:
    # thresholds in (0, 1), med strictly below high when both are
    # earned (the bands artifact's invariant, mirrored)
    checked_high = _check_band_value(label, "high", high)
    checked_med = _check_band_value(label, "med", med)
    if (
        checked_high is not None
        and checked_med is not None
        and checked_med >= checked_high
    ):
        raise ValueError(
            f"tiers for {label!r}: med band {med} must be strictly below "
            f"high band {high}"
        )


def would_apply(
    *,
    scores: Mapping[str, float],
    bands: Mapping[str, tuple[float | None, float | None]],
    graduated: frozenset[str],
    present: frozenset[str],
    system_labels: frozenset[str] = frozenset(),
) -> tuple[TierDecision, ...]:
    """The per-label would-apply verdicts for one message (the Shadow
    Pass decision, GLOSSARY: Shadow Pass).

    ``scores`` maps label → sigmoid score (the model's row, aligned to
    the checkpoint vocabulary); ``bands`` maps label → (high, med)
    calibrated Threshold Bands (``None`` = the explicit no-band state);
    ``graduated`` is the Graduation Table's graduated label set;
    ``present`` is the label names already on the message;
    ``system_labels`` is the System Label set (never targeted).

    Refusals (fail-loud doctrine — the verdicts are a measured claim,
    never a guess): a score outside [0, 1] (a bool is not a score), a
    band threshold outside (0, 1) or an inverted (med >= high) pair, a
    graduated label the scores do not carry (the model scores the whole
    vocabulary — its absence is a defect), a graduated label the bands
    do not carry (the bands cover the vocabulary), and a graduated
    System Label (never targeted) all raise ValueError, named.
    Labels OUTSIDE the graduated set are ignored (the model scores the
    whole vocabulary; only graduated labels are proposed).

    Deterministic: same inputs → same decisions, in label-sorted order
    (a total order — the input mapping order never leaks out).
    """
    # the presence checks FIRST (a clearer cause than a shape check on
    # an absent value): a graduated label the scores/bands do not carry
    # is a lying input — the model scores the whole vocabulary, and the
    # bands artifact covers it
    missing_scores = sorted(label for label in graduated if label not in scores)
    if missing_scores:
        raise ValueError(
            f"tiers: graduated label(s) {missing_scores} have no score — "
            "the model scores the whole vocabulary; refusing to guess"
        )
    missing_bands = sorted(label for label in graduated if label not in bands)
    if missing_bands:
        raise ValueError(
            f"tiers: graduated label(s) {missing_bands} have no bands — "
            "the bands artifact covers the vocabulary; refusing to guess"
        )
    for label in sorted(graduated):
        if label in system_labels:
            # System Labels are never content labels, never touchable —
            # a graduated System Label is a lying input, refused named
            raise ValueError(
                f"tiers for {label!r}: a System Label is never targeted — "
                "refusing to propose it (fix the Graduation Table)"
            )
    decisions: list[TierDecision] = []
    for label in sorted(graduated):  # label-sorted — a total order
        score = _check_score(label, scores[label])
        high, med = bands[label]
        _check_bands(label, high, med)
        band: str | None = None
        if high is not None and score >= high:
            band = BAND_HIGH
        elif med is not None and score >= med:
            band = BAND_MED
        is_present = label in present
        applied = band is not None and not is_present
        # skip-present (invariant #2): an already-present label is not
        # proposed and earns NO Tier Label — the band stays visible
        tier = (
            TIER_HIGH
            if applied and band == BAND_HIGH
            else (TIER_MED if applied and band == BAND_MED else None)
        )
        decisions.append(
            TierDecision(
                label=label,
                score=score,
                band=band,
                tier=tier,
                present=is_present,
                applied=applied,
            )
        )
    return tuple(decisions)


def best_tier(decisions: Sequence[TierDecision]) -> str | None:
    """The per-message Tier Label (GLOSSARY: Application: one Tier Label
    per Application): the best band among the APPLIED decisions (high
    outranks med). Skip-present decisions never contribute — they
    earned no Tier Label (invariant #2). ``None`` when nothing is
    applied (no Application, no Tier Label)."""
    tiers = {d.tier for d in decisions if d.applied}
    if TIER_HIGH in tiers:
        return TIER_HIGH
    if TIER_MED in tiers:
        return TIER_MED
    return None
