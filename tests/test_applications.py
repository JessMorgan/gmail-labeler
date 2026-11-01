"""applications.py (ticket #24): the fail-safe write composition on the
tiers core — band edges compose to tier ids, skip-present earns nothing,
ANY compose failure means no write for that message, System Labels never
targeted (name guard AND id guard), deterministic total order, and the
plan's rows round-trip the Applications store (no bodies).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Never

import pytest

import gmail_labeler.appdb as appdb_module
from gmail_labeler.appdb import AppDB
from gmail_labeler.applications import SOURCES, WritePlan, plan_message_write
from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
from gmail_labeler.model import FakeModel
from gmail_labeler.tiers import TIER_APPLIED, TIER_HIGH, TIER_MED

TS = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)

VOCAB = ("Money/Bills", "Shipping")

# the gmail ids the fixtures resolve names to (managed + tier)
MANAGED_IDS = {"Shipping": "Label_ship", "Money/Bills": "Label_bills"}
TIER_IDS = {
    TIER_APPLIED: "Label_umbrella",
    TIER_HIGH: "Label_aihigh",
    TIER_MED: "Label_aimed",
}


def _scores(text: str, row: list[float]) -> dict[str, float]:
    """The AC1 vector path: a FakeModel hand-authored row aligned to the
    vocabulary (the shadow CLI's own zip — the scores reach the plan
    exactly as the model produces them)."""
    model = FakeModel(VOCAB, scores={text: row})
    (scores_row,) = model.predict([text])
    return dict(zip(model.vocabulary, scores_row, strict=True))


def _plan(
    scores: Mapping[str, float],
    *,
    bands: Mapping[str, tuple[float | None, float | None]] | None = None,
    graduated: frozenset[str] = frozenset(VOCAB),
    present: frozenset[str] = frozenset(),
    system_labels: frozenset[str] = frozenset(),
    managed_label_ids: Mapping[str, str] = MANAGED_IDS,
    tier_label_ids: Mapping[str, str] = TIER_IDS,
    message_id: str = "aa",
    source: str = "daemon",
) -> WritePlan:
    return plan_message_write(
        message_id,
        scores=scores,
        bands=bands if bands is not None else {label: (0.9, 0.7) for label in VOCAB},
        graduated=graduated,
        present=present,
        system_labels=system_labels,
        managed_label_ids=managed_label_ids,
        tier_label_ids=tier_label_ids,
        applied_at=TS,
        source=source,
    )


# -- AC1: band edges compose to the tier ids (exactly-on included) ----------


def test_score_exactly_at_high_composes_the_high_tier_id() -> None:
    # the COMPOSITION consequence of the #23 >= pins: a score EXACTLY at
    # the high threshold puts the high-tier id on the wire (managed id +
    # umbrella + band tier, sorted)
    plan = _plan(_scores("t", [0.5, 0.9]))
    assert plan.warning is None
    assert plan.adds == ("Label_aihigh", "Label_ship", "Label_umbrella")


def test_score_exactly_at_med_composes_the_med_tier_id() -> None:
    plan = _plan(_scores("t", [0.5, 0.7]))
    assert plan.adds == ("Label_aimed", "Label_ship", "Label_umbrella")


def test_score_below_med_is_a_legitimate_empty_plan() -> None:
    # nothing earned: EMPTY plan, NO warning (the legitimate-empty state
    # is distinguishable from the fail-safe-empty state)
    plan = _plan(_scores("t", [0.5, 0.699]))
    assert plan.adds == ()
    assert plan.rows == ()
    assert plan.warning is None


def test_umbrella_plus_best_band_only() -> None:
    # the GLOSSARY Application contract: the umbrella AND exactly ONE
    # band tier (the best among applied — high beats med); the med tier
    # id is NEVER added alongside the high one
    plan = _plan(_scores("t", [0.7, 0.95]))
    assert plan.adds == (
        "Label_aihigh",
        "Label_bills",
        "Label_ship",
        "Label_umbrella",
    )
    assert "Label_aimed" not in plan.adds


# -- AC3: skip-present consequence (no row, no tier ids) --------------------


def test_skip_present_earns_no_row_and_no_tier_ids() -> None:
    # invariant #2: the present label is not written AND the
    # consequence — nothing new added means NO Application row and NO
    # Tier Labels (empty plan → no API call at all)
    plan = _plan(_scores("t", [0.5, 0.95]), present=frozenset({"Shipping"}))
    assert plan.adds == ()
    assert plan.rows == ()
    assert plan.warning is None


def test_partial_skip_present_keeps_only_the_new_label() -> None:
    # Shipping present, Money/Bills applied: the row is the new label
    # only; the tier ids ride the APPLIED decisions (best = high)
    plan = _plan(_scores("t", [0.95, 0.95]), present=frozenset({"Shipping"}))
    assert plan.adds == ("Label_aihigh", "Label_bills", "Label_umbrella")
    assert [r.label for r in plan.rows] == ["Money/Bills"]


# -- fail-safe (invariant #1): ANY compose error → no write -----------------


class _ExplodingScores(Mapping[str, float]):
    """A lying input surface: the failure is an ARBITRARY exception
    (not a tiers ValueError) — the net must catch ANY of them."""

    def __init__(self, inner: Mapping[str, float]) -> None:
        self._inner = inner

    def __getitem__(self, key: str) -> float:
        raise RuntimeError("boom")

    def __iter__(self) -> Never:
        raise RuntimeError("boom")

    def __len__(self) -> int:
        return len(self._inner)


def test_injected_failure_means_no_write_and_id_only_warning() -> None:
    # AC fail-safe: ANY exception inside per-message compose → no write
    # for that message (empty adds — the caller's no-call-on-empty rule
    # suppresses the API call), counted via the warning; the warning
    # names the message id (the daemon logs it)
    plan = _plan(_ExplodingScores({"Shipping": 0.9, "Money/Bills": 0.9}))
    assert plan.adds == ()
    assert plan.rows == ()
    assert plan.warning is not None
    assert "aa" in plan.warning  # the id is present (counted, named)


def test_fail_safe_warning_carries_no_body(caplog: pytest.LogCaptureFixture) -> None:
    # invariant #5 at the log surface: a lying score (out of [0, 1]) is
    # refused into the warning, the log line names the message id, and
    # no body material exists anywhere near the plan — the compose
    # inputs are ids/labels/scores ONLY, so the error text structurally
    # cannot carry one
    with caplog.at_level(logging.WARNING):
        # the lying row bypasses FakeModel's programming check (a real
        # lying model row reaches the plan from a defective checkpoint)
        plan = _plan({"Money/Bills": 0.5, "Shipping": 1.5})
    assert plan.adds == ()
    assert plan.warning is not None
    # the positive shape check (the negative "no body string" assertion
    # was unfalsifiable — bodies are structurally absent from the plan's
    # inputs): the warning is exactly "{id}: {type}: {tiers message}",
    # built from ids/labels/scores ONLY
    assert plan.warning == (
        "aa: ValueError: tiers for 'Shipping': score 1.5 is out of [0, 1] "
        "— a lying model row"
    )
    logging.getLogger("gmail_labeler.test").warning("skipped message: %s", plan.warning)
    assert "aa" in caplog.text


def test_missing_managed_id_is_fail_safe() -> None:
    # a would-apply label the resolution map does not carry is a lying
    # edge (the daemon resolves once at startup) — no write, warning
    plan = _plan(
        _scores("t", [0.95, 0.5]),  # Money/Bills applied, Shipping no band
        managed_label_ids={"Shipping": "Label_ship"},  # Money/Bills absent
        graduated=frozenset({"Money/Bills"}),
    )
    assert plan.adds == ()
    assert plan.warning is not None


def test_internal_guard_assertion_raises_not_warned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # F1 (deepseek C3 MINOR): the unreachable-by-construction guards sit
    # INSIDE the fail-safe net — an internal regression (best_tier
    # contradicting the applied decisions) must surface LOUDLY, never
    # degrade to a per-message warning that turns a systematic break
    # into silent label starvation. Data-path failures stay fail-safe;
    # internal contradictions raise
    import gmail_labeler.applications as applications_module

    monkeypatch.setattr(applications_module, "best_tier", lambda decisions: None)
    with pytest.raises(AssertionError):
        applications_module.plan_message_write(
            "aa",
            scores=_scores("t", [0.5, 0.95]),
            bands={label: (0.9, 0.7) for label in VOCAB},
            graduated=frozenset(VOCAB),
            present=frozenset(),
            system_labels=frozenset(),
            managed_label_ids=MANAGED_IDS,
            tier_label_ids=TIER_IDS,
            applied_at=TS,
            source="daemon",
        )


def test_caller_bugs_raise_not_warn() -> None:
    # the pre-compose guards are CALLER bugs (exit-1 territory), not
    # per-message data errors: a bad source literal or a naive clock
    # read must crash loudly, never silently skip every message
    scores = _scores("t", [0.5, 0.95])
    with pytest.raises(ValueError, match="source"):
        _plan(scores, source="shadow")
    with pytest.raises(ValueError, match="timezone-aware"):
        plan_message_write(
            "aa",
            scores=scores,
            bands={label: (0.9, 0.7) for label in VOCAB},
            graduated=frozenset(VOCAB),
            present=frozenset(),
            system_labels=frozenset(),
            managed_label_ids=MANAGED_IDS,
            tier_label_ids=TIER_IDS,
            applied_at=TS.replace(tzinfo=None),
            source="daemon",
        )


# -- AC6: System Labels never targeted (name guard + id guard) --------------


def test_graduated_system_label_never_reaches_adds() -> None:
    # the tampered-artifact path: a System Label carrying a graduated
    # entry — tiers refuses it (the name guard) and the fail-safe net
    # turns the refusal into NO WRITE + warning (never a silent pass)
    plan = plan_message_write(
        "aa",
        scores={"INBOX": 0.95},
        bands={"INBOX": (0.9, 0.7)},
        graduated=frozenset({"INBOX"}),
        present=frozenset(),
        system_labels=frozenset({"INBOX"}),
        managed_label_ids={"INBOX": "Label_sneaky"},
        tier_label_ids=TIER_IDS,
        applied_at=TS,
        source="daemon",
    )
    assert plan.adds == ()
    assert plan.rows == ()
    assert plan.warning is not None


def test_system_label_id_in_resolution_map_is_refused() -> None:
    # the COMPOSITION-level pin on top of tiers' name guard: a lying
    # resolution map that points a would-apply label at a System Label
    # ID never reaches the wire (is_system_label_id screens every id)
    plan = _plan(
        _scores("t", [0.5, 0.95]),
        managed_label_ids={"Shipping": "INBOX", "Money/Bills": "Label_bills"},
        graduated=frozenset({"Shipping"}),
    )
    assert plan.adds == ()
    assert plan.warning is not None
    assert "INBOX" in plan.warning


def test_system_label_tier_id_is_refused() -> None:
    plan = _plan(
        _scores("t", [0.5, 0.95]),
        graduated=frozenset({"Shipping"}),
        tier_label_ids={**TIER_IDS, TIER_APPLIED: "SPAM"},
    )
    assert plan.adds == ()
    assert plan.warning is not None


# -- determinism + the wire composition -------------------------------------


def test_deterministic_total_order() -> None:
    # the input mapping order never leaks: adds is sorted (a total
    # order), rows are sorted by (message_id, label)
    scores_a = {"Shipping": 0.95, "Money/Bills": 0.8}
    reversed_scores = dict(reversed(list(scores_a.items())))
    plan_a = _plan(scores_a)
    plan_b = _plan(reversed_scores)
    assert plan_a.adds == plan_b.adds
    assert list(plan_a.adds) == sorted(plan_a.adds)
    assert [r.label for r in plan_a.rows] == ["Money/Bills", "Shipping"]


def test_empty_plan_makes_no_api_call_full_plan_makes_one() -> None:
    # the caller contract pinned at the fake: an empty plan is NO apply
    # call at all (crash-retry idempotency, invariant #2); a non-empty
    # plan is exactly ONE call whose recorded body carries addLabelIds
    # ONLY — no removeLabelIds key exists in the snapshot
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aa", eml=b"x"), FakeMessage(id="bb", eml=b"y")]
    )
    empty = _plan(_scores("t", [0.5, 0.95]), present=frozenset({"Shipping"}))
    if empty.adds:  # the daemon/apply caller pattern
        fake.apply_labels(empty.message_id, empty.adds)
    full = _plan(_scores("t2", [0.5, 0.95]), message_id="bb")
    if full.adds:
        fake.apply_labels(full.message_id, full.adds)
    assert fake.calls == ["apply_labels:bb"]
    assert fake.writes == [
        {
            "method": "apply_labels",
            "message_id": "bb",
            "body": {"addLabelIds": ["Label_aihigh", "Label_ship", "Label_umbrella"]},
        }
    ]
    body = fake.writes[0]["body"]
    assert isinstance(body, dict)
    assert "removeLabelIds" not in body


# -- the rows + the store round-trip -----------------------------------------


def test_rows_carry_the_composition_and_round_trip_the_store(
    tmp_path: Path,
) -> None:
    plan = _plan(_scores("t", [0.7, 0.95]))
    assert [r.tier for r in plan.rows] == ["ai-high", "ai-high"]  # per-message best
    assert [r.band for r in plan.rows] == ["med", "high"]
    assert [r.label_id for r in plan.rows] == ["Label_bills", "Label_ship"]
    assert all(r.applied_at == TS and r.source == "daemon" for r in plan.rows)
    db = AppDB(tmp_path / "state.db")
    db.open()
    assert db.record_applications(list(plan.rows)) == 2
    rows = db.applications_for_message("aa")
    assert [(r.label, r.band, r.tier, r.source) for r in rows] == [
        ("Money/Bills", "med", "ai-high", "daemon"),
        ("Shipping", "high", "ai-high", "daemon"),
    ]
    db.close()


def test_apply_source_is_accepted() -> None:
    plan = _plan(_scores("t", [0.5, 0.95]), source="apply")
    assert plan.rows[0].source == "apply"


def test_sources_mirror_the_store_vocabulary() -> None:
    # the literal vocabulary has ONE authority (the appdb CHECK); the
    # pure module mirrors it import-light (the _BANDS doctrine) — the
    # equality is pinned HERE, not by an import
    assert SOURCES == appdb_module._SOURCES == ("daemon", "apply")
