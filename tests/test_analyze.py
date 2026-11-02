"""dataset analyze metrics: gates, attribution honesty, determinism."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gmail_labeler.analyze import LabelMetrics, analyze_corpus, months_ago
from gmail_labeler.corpus_record import CorpusRecord, body_hash

NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)


def _rec(
    i: int,
    labels: tuple[str, ...],
    *,
    days_ago: int = 1,
    body: str | None = None,
    sender: str = "x@y.z",
    subject: str = "hello",
    recipients: tuple[str, ...] = ("me@x.y",),
) -> CorpusRecord:
    text = f"unique body {i}" if body is None else body
    return CorpusRecord(
        id=f"{i:016x}",
        date=NOW - timedelta(days=days_ago),
        subject=subject,
        sender=sender,
        recipients=recipients,
        labels=labels,
        body_hash=body_hash(text),
        source="api",
    )


def _bulk(
    n: int,
    label: str,
    *,
    days_ago: int = 1,
    sender: str = "x@y.z",
    subject: str = "hello",
    recipients: tuple[str, ...] = ("me@x.y",),
) -> list[CorpusRecord]:
    return [
        _rec(
            i,
            (label,),
            days_ago=days_ago,
            sender=sender,
            subject=subject,
            recipients=recipients,
        )
        for i in range(n)
    ]


def _only(metrics: list[LabelMetrics], name: str) -> LabelMetrics:
    return next(m for m in metrics if m.label == name)


# ---- months_ago ----


def test_months_ago_calendar_edges() -> None:
    assert months_ago(datetime(2026, 3, 31, tzinfo=UTC), 1) == datetime(
        2026, 2, 28, tzinfo=UTC
    )
    assert months_ago(datetime(2026, 1, 15, tzinfo=UTC), 1) == datetime(
        2025, 12, 15, tzinfo=UTC
    )
    assert months_ago(datetime(2026, 5, 31, tzinfo=UTC), 3) == datetime(
        2026, 2, 28, tzinfo=UTC
    )  # day clamped, never March-6th nonsense
    assert months_ago(NOW, 24) == datetime(2024, 10, 30, 12, 0, tzinfo=UTC)
    assert months_ago(NOW, 0) == NOW


def test_months_ago_rejects_naive_now() -> None:
    import pytest

    with pytest.raises(ValueError, match="timezone-aware"):
        months_ago(datetime(2026, 1, 1), 1)


# ---- summary ----


def test_empty_corpus_summary() -> None:
    summary, metrics = analyze_corpus([], filters=[], id_to_name={}, now=NOW)
    assert summary.messages == 0 and summary.first_date is None
    assert metrics == []


def test_summary_counts() -> None:
    recs = [
        _rec(1, ("A",), days_ago=100),
        _rec(2, ("A", "B")),
        _rec(3, ("B",), body="unique body 1"),  # shares body with rec 1
    ]
    recs[0] = _rec(1, ("A",), days_ago=100, body="dup")
    recs[2] = _rec(3, ("B",), body="dup")
    summary, _ = analyze_corpus(recs, filters=[], id_to_name={}, now=NOW)
    assert summary.messages == 3 and summary.distinct_labels == 2
    assert summary.duplicate_bodies == 1
    assert summary.first_date == NOW - timedelta(days=100)


# ---- gates ----


def test_happy_candidate() -> None:
    m = _only(
        analyze_corpus(_bulk(60, "Ship"), filters=[], id_to_name={}, now=NOW)[1], "Ship"
    )
    assert m.candidate and m.reasons == ()


def test_low_volume_and_low_recent() -> None:
    metrics = analyze_corpus(_bulk(40, "Tiny"), filters=[], id_to_name={}, now=NOW)[1]
    assert "low-volume" in metrics[0].reasons
    recs = _bulk(55, "Old", days_ago=900) + _bulk(5, "Old")
    metrics = analyze_corpus(recs, filters=[], id_to_name={}, now=NOW)[1]
    m = _only(metrics, "Old")
    assert "low-recent" in m.reasons and m.total == 60 and m.recent == 5


def test_template_heavy() -> None:
    recs = [_rec(i, ("Tpl",), body="SAME TEMPLATE") for i in range(60)]
    m = _only(analyze_corpus(recs, filters=[], id_to_name={}, now=NOW)[1], "Tpl")
    assert m.dup_rate == 1.0 and "template-heavy" in m.reasons


def test_filter_coverage_gate_strict_gt() -> None:
    recs = [
        _rec(i, ("Cov",), sender=f"shop{i % 2}@x.z") for i in range(60)
    ]  # half from shop1@x.z
    filt = {
        "criteria": {"from": "shop1@x.z"},
        "action": {"addLabelIds": ["L1"]},
    }
    metrics = analyze_corpus(recs, filters=[filt], id_to_name={"L1": "Cov"}, now=NOW)[1]
    m = _only(metrics, "Cov")
    assert m.attribution_ratio == 0.5 and m.candidate  # gate is STRICT > 0.5
    recs_all = _bulk(60, "Cov", sender="shop@x.z")
    m2 = _only(
        analyze_corpus(
            recs_all,
            filters=[
                {"criteria": {"from": "SHOP@X.Z"}, "action": {"addLabelIds": ["L1"]}}
            ],
            id_to_name={"L1": "Cov"},
            now=NOW,
        )[1],
        "Cov",
    )
    assert "filter-covered" in m2.reasons and m2.attribution_ratio == 1.0


def test_unverifiable_criteria_withholds_label() -> None:
    filt = {"criteria": {"query": "some ql"}, "action": {"addLabelIds": ["L1"]}}
    m = _only(
        analyze_corpus(_bulk(60, "Q"), filters=[filt], id_to_name={"L1": "Q"}, now=NOW)[
            1
        ],
        "Q",
    )
    assert m.unreliable_filters == 1 and "filter-unverifiable" in m.reasons
    assert m.filter_attributed == 0  # not guessed, excluded


def test_to_and_subject_criteria_simulated() -> None:
    recs = _bulk(
        60, "S", subject="Your ORDER 123 shipped", recipients=("orders@shop.co",)
    )
    filt = {"criteria": {"subject": "order 123"}, "action": {"addLabelIds": ["L1"]}}
    m = _only(
        analyze_corpus(recs, filters=[filt], id_to_name={"L1": "S"}, now=NOW)[1], "S"
    )
    assert m.attribution_ratio == 1.0  # case-insensitive substring
    filt_to = {"criteria": {"to": "orders@shop.co"}, "action": {"addLabelIds": ["L1"]}}
    m2 = _only(
        analyze_corpus(recs, filters=[filt_to], id_to_name={"L1": "S"}, now=NOW)[1], "S"
    )
    assert m2.attribution_ratio == 1.0  # matched against recipients


def test_mixed_criteria_any_unverifiable_marks_label() -> None:
    filt = {
        "criteria": {"from": "x@y.z", "hasAttachment": True},
        "action": {"addLabelIds": ["L1"]},
    }
    m = _only(
        analyze_corpus(_bulk(60, "M"), filters=[filt], id_to_name={"L1": "M"}, now=NOW)[
            1
        ],
        "M",
    )
    assert "filter-unverifiable" in m.reasons and m.attribution_ratio == 0.0


def test_no_snapshot_withholds_everything() -> None:
    _, metrics = analyze_corpus(_bulk(60, "P"), filters=None, id_to_name={}, now=NOW)
    m = _only(metrics, "P")
    assert not m.candidate and "no-filter-snapshot" in m.reasons


def test_unknown_filter_label_id_ignored() -> None:
    filt = {"criteria": {"from": "x@y.z"}, "action": {"addLabelIds": ["ZZ"]}}
    _, metrics = analyze_corpus(
        _bulk(60, "P"), filters=[filt], id_to_name={"ZZ": "Ghost"}, now=NOW
    )
    assert [m.label for m in metrics] == ["P"] and metrics[0].candidate


# ---- determinism / ordering ----


def test_deterministic_total_order() -> None:
    recs = _bulk(60, "Beta") + _bulk(60, "Alpha") + _bulk(10, "Tiny")
    r1 = analyze_corpus(recs, filters=[], id_to_name={}, now=NOW)[1]
    r2 = analyze_corpus(list(reversed(recs)), filters=[], id_to_name={}, now=NOW)[1]
    assert r1 == r2  # input order irrelevant
    assert [m.label for m in r1] == ["Alpha", "Beta", "Tiny"]  # candidate, -total, name


def test_months_ago_rejects_negative_and_pathological() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be >= 0"):
        months_ago(NOW, -1)
    with pytest.raises(ValueError, match="out of representable range"):
        months_ago(NOW, 10**9)  # no 10^9-iteration loop, clear message


# ---- deepseek independent-engine-pass pins (MAJOR + MINORs) ----


def _filt(criteria: object, label_id: str = "L1") -> dict[str, object]:
    f: dict[str, object] = {"action": {"addLabelIds": [label_id]}}
    if criteria is not None:
        f["criteria"] = criteria
    return f


def test_from_angle_bracket_display_name_is_unverifiable() -> None:
    # M13: name-syntax rejection pinned — the most realistic MAJOR form
    m = _only(
        analyze_corpus(
            _bulk(60, "Nike", sender="daily@news.nike.com"),
            filters=[_filt({"from": "Nike <news@nike.com>"})],
            id_to_name={"L1": "Nike"},
            now=NOW,
        )[1],
        "Nike",
    )
    assert "filter-unverifiable" in m.reasons and m.filter_attributed == 0


def test_verifiability_precedes_containment() -> None:
    # deepseek MINOR: unverifiable 'to' behind a non-matching 'from'
    m = _only(
        analyze_corpus(
            _bulk(60, "L", sender="other@x.z"),
            filters=[_filt({"from": "a@b.c", "to": "Nike"})],
            id_to_name={"L1": "L"},
            now=NOW,
        )[1],
        "L",
    )
    assert "filter-unverifiable" in m.reasons and m.unreliable_filters == 1


def test_unrounded_metrics_match_their_gates() -> None:
    # deepseek MINOR: 10001/20001 must not DISPLAY 0.5 while gated on >0.5
    n = 20001
    recs = [
        _rec(
            i,
            ("Half",),
            sender="cov@x.z" if i < 10001 else "oth@x.z",
        )
        for i in range(n)
    ]
    m = _only(
        analyze_corpus(
            recs,
            filters=[_filt({"from": "cov@x.z"})],
            id_to_name={"L1": "Half"},
            now=NOW,
        )[1],
        "Half",
    )
    assert m.attribution_ratio > 0.5 and "filter-covered" in m.reasons


def test_from_display_name_is_unverifiable_not_zero() -> None:
    # THE MAJOR: Gmail matches raw from-text against the whole header;
    # 'Nike' can never be simulated from addr-specs — it must withhold.
    recs = _bulk(60, "Nike", sender="daily@news.nike.com")
    m = _only(
        analyze_corpus(
            recs,
            filters=[_filt({"from": "Nike"})],
            id_to_name={"L1": "Nike"},
            now=NOW,
        )[1],
        "Nike",
    )
    assert "filter-unverifiable" in m.reasons and not m.candidate
    assert m.filter_attributed == 0  # exclusion, never a guessed False


def test_from_address_shaped_still_simulated() -> None:
    recs = _bulk(60, "Nike", sender="daily@news.nike.com")
    m = _only(
        analyze_corpus(
            recs,
            filters=[_filt({"from": "nike.com"})],
            id_to_name={"L1": "Nike"},
            now=NOW,
        )[1],
        "Nike",
    )
    assert m.attribution_ratio == 1.0 and "filter-covered" in m.reasons


def test_to_display_name_unverifiable() -> None:
    recs = _bulk(60, "Lbl")
    m = _only(
        analyze_corpus(
            recs,
            filters=[_filt({"to": "Orders Team"})],
            id_to_name={"L1": "Lbl"},
            now=NOW,
        )[1],
        "Lbl",
    )
    assert "filter-unverifiable" in m.reasons


def test_empty_criteria_is_match_all_filter_covered() -> None:
    recs = _bulk(60, "Lbl")
    m = _only(
        analyze_corpus(recs, filters=[_filt({})], id_to_name={"L1": "Lbl"}, now=NOW)[1],
        "Lbl",
    )
    assert m.attribution_ratio == 1.0 and "filter-covered" in m.reasons
    assert m.unreliable_filters == 0  # genuinely simulatable match-all


def test_malformed_criteria_unverifiable_not_match_all() -> None:
    recs = _bulk(60, "Lbl")
    m = _only(
        analyze_corpus(
            recs, filters=[_filt("junk")], id_to_name={"L1": "Lbl"}, now=NOW
        )[1],
        "Lbl",
    )
    assert m.unreliable_filters == 1 and m.attribution_ratio == 0.0
    assert "filter-unverifiable" in m.reasons


def test_mixed_filters_coherent_verdicts() -> None:
    # the doctrine's own mixed case: covering + unverifiable -> ratio 1.0
    # AND filter-unverifiable, exact reason ORDER pinned (M2-resistant)
    recs = _bulk(60, "Lbl")
    metrics = analyze_corpus(
        recs,
        filters=[_filt({"from": "x@y.z"}), _filt({"query": "ql"})],
        id_to_name={"L1": "Lbl"},
        now=NOW,
    )[1]
    m = _only(metrics, "Lbl")
    assert m.reasons == ("filter-covered", "filter-unverifiable")
    assert m.attribution_ratio == 1.0 and m.unreliable_filters == 1


def test_reasons_exact_order_contract() -> None:
    # total 45 < min_total 50 AND recent-window count 5 < min_recent 10
    recs = _bulk(40, "Old", days_ago=900) + _bulk(5, "Old")
    m = _only(analyze_corpus(recs, filters=[], id_to_name={}, now=NOW)[1], "Old")
    assert m.reasons == ("low-volume", "low-recent")  # contract order, exact


def test_total_order_biggest_first() -> None:
    # M3-resistant: -total must order candidates (equal-size tests can't see it)
    recs = _bulk(51, "Alpha") + _bulk(61, "Zeta")
    metrics = analyze_corpus(recs, filters=[], id_to_name={}, now=NOW)[1]
    assert [m.label for m in metrics] == ["Zeta", "Alpha"]  # -total beats name


def test_months_ago_extreme_overflow_wrapped() -> None:
    import pytest

    for huge in (10**9, 10**11, 10**14):
        with pytest.raises(ValueError, match="out of representable range"):
            months_ago(NOW, huge)


def test_nonstr_criteria_value_unverifiable() -> None:
    recs = _bulk(60, "Lbl")
    m = _only(
        analyze_corpus(
            recs, filters=[_filt({"from": 42})], id_to_name={"L1": "Lbl"}, now=NOW
        )[1],
        "Lbl",
    )
    assert "filter-unverifiable" in m.reasons
