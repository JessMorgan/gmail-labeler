from __future__ import annotations

import pytest

from gmail_labeler.labels import (
    is_system_label_id,
    label_path,
    normalize_label_names,
    path_is_at_or_under,
)


@pytest.mark.parametrize(
    "label_id",
    [
        "INBOX",
        "SENT",
        "UNREAD",
        "IMPORTANT",
        "STARRED",
        "SPAM",
        "TRASH",
        "DRAFT",
    ],
)
def test_system_ids_are_system(label_id: str) -> None:
    assert is_system_label_id(label_id)


@pytest.mark.parametrize(
    "label_id",
    ["CATEGORY_PROMOTIONS", "CATEGORY_UPDATES", "CATEGORY_FOOS"],  # prefix rule
)
def test_category_family_is_system(label_id: str) -> None:
    assert is_system_label_id(label_id)


@pytest.mark.parametrize(
    "label_id",
    ["CHAT", "SNOOZE", "UNDO", "YELLOW_STAR", "PURPLE_STAR"],
)
def test_extended_state_labels_are_system(label_id: str) -> None:
    # YELLOW_STAR observed in a real GYB corpus; CHAT/SNOOZE/UNDO per
    # community enumerations (GAM/lieer), NOT Google's guide table.
    assert is_system_label_id(label_id)


@pytest.mark.parametrize(
    "label_id",
    [
        "RED_CIRCLE",
        "RED_BANG",
        "ORANGE_GUILLEMET",
        "YELLOW_GUILLEMET",
        "YELLOW_BANG",
        "GREEN_CHECK",
        "BLUE_INFO",
        "PURPLE_QUESTION",
    ],
)
def test_superstar_format_variants_are_system(label_id: str) -> None:
    # Non-star members of Gmail's star-format namespace: users with other
    # star presets surface these ids (YELLOW_STAR was observed in a real
    # corpus; the family is closed per the ^ss_* mapping).
    assert is_system_label_id(label_id)


def test_corpus_observed_system_tokens_all_filtered() -> None:
    # Evidence tie: every system token dumped from a real GYB
    # msg-db (public Gmail tokens; user labels excluded) must be filtered.
    observed = [
        "CATEGORY_FORUMS",
        "CATEGORY_PERSONAL",
        "CATEGORY_PROMOTIONS",
        "CATEGORY_SOCIAL",
        "CATEGORY_UPDATES",
        "DRAFT",
        "IMPORTANT",
        "INBOX",
        "SENT",
        "STARRED",
        "UNREAD",
        "YELLOW_STAR",
    ]
    assert all(is_system_label_id(t) for t in observed)


@pytest.mark.parametrize(
    "label_id",
    ["Label_123", "Money", "work", "CATEGORY", "MY_CATEGORY_X", "money/star"],
)
def test_user_ids_are_not_system(label_id: str) -> None:
    assert not is_system_label_id(label_id)


def test_label_path_nesting() -> None:
    assert label_path("Money/Bills") == ("Money", "Bills")
    assert label_path("Shipping") == ("Shipping",)


def test_label_path_degenerate_is_empty() -> None:
    assert label_path("") == ()
    assert label_path("/") == ()
    assert label_path("a//b") == ("a", "b")  # empty middle segment dropped


def test_label_path_unicode_and_edges() -> None:
    assert label_path("Voiture/Électrique") == ("Voiture", "Électrique")
    assert label_path("🎄/Season's") == ("🎄", "Season's")
    assert label_path("/Money") == ("Money",)  # leading slash collapses
    assert label_path("Money/") == ("Money",)  # trailing slash collapses


def test_path_at_or_under() -> None:
    assert path_is_at_or_under(("Money", "Bills"), ("Money",))
    assert path_is_at_or_under(("Money",), ("Money",))
    assert not path_is_at_or_under(("Money",), ("Money", "Bills"))
    assert not path_is_at_or_under(("Moneymoney",), ("Money",))
    assert not path_is_at_or_under(("Money2",), ("Money",))  # sibling, not child
    assert not path_is_at_or_under(("Money", "Bills"), ())


def test_normalize_drops_system_and_maps_names() -> None:
    ids = ["INBOX", "L1", "CATEGORY_SOCIAL", "L2"]
    mapping = {"L1": "Shipping", "L2": "Money/Bills"}
    assert normalize_label_names(ids, mapping) == ["Shipping", "Money/Bills"]


def test_normalize_unknown_id_surfaces_namespaced() -> None:
    # A deleted label must remain visible to analysis, not silently vanish.
    assert normalize_label_names(["L9"], {}) == ["id:L9"]


def test_normalize_preserves_order_with_mixed_sources() -> None:
    ids = ["L2", "INBOX", "GONE", "L1"]
    names = normalize_label_names(ids, {"L1": "Music", "L2": "Shipping"})
    assert names == ["Shipping", "id:GONE", "Music"]


def test_normalize_blank_mapped_name_surfaces_unresolved() -> None:
    # "" would be silently dropped downstream (falsy) — the exact vanishing
    # the id: namespace exists to prevent.
    assert normalize_label_names(["L1"], {"L1": ""}) == ["id:L1"]


def test_accepted_tradeoff_allcaps_user_label_named_like_system_token_is_dropped() -> (
    None
):
    # Recorded decision (review): in GYB NAME space an ALL-CAPS user label
    # literally named like a community-enumerated token is filtered. We
    # accept this near-zero-probability suppression rather than risking the
    # far more likely leak; a "fix" reversing it must revisit this test.
    assert normalize_label_names(["INBOX", "SNOOZE"], {"SNOOZE": "SNOOZE"}) == []
