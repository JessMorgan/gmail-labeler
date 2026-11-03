"""attribution.parse_attribution: posture contract + invariant-#6 guards.

S3 seam (spec #65): direct function call into the pure module, prior art
test_analyze.py calling analyze_corpus directly. This suite MOVED out of the
CLI harness (ticket #67) — the shape guards and the bridge-incomplete demotion
are pure decision logic and must not require booting typer to prove. Expected
values are transcribed from the harness tests that pinned this behavior before
the extraction; the error messages are the ones the CLI echoes into exit 5.
"""

from __future__ import annotations

from typing import get_args

import pytest

from gmail_labeler.attribution import (
    SIDECAR_FILTERS,
    SIDECAR_LABELS,
    AttributionPosture,
    parse_attribution,
)

# one filter whose label id resolves through the bridge (the happy shape the
# Gmail import writer produces)
FILTERS_L1 = '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": ["L1"]}}]'
BRIDGE_L1 = '{"L1": "Shipping"}'


# ---- the on-disk contract the module owns ---------------------------------


def test_sidecar_filenames_are_the_corpus_format_contract() -> None:
    # stored corpora on disk are keyed by these names; renaming the constant
    # silently would orphan every existing snapshot
    assert SIDECAR_FILTERS == "filters.json"
    assert SIDECAR_LABELS == "labels.json"


def test_posture_literal_is_pinned_for_json_consumers() -> None:
    # summary.filters_posture is machine-read by graduation workflows
    # (spec #65 user story 6); the value set is the contract
    assert set(get_args(AttributionPosture)) == {
        "snapshot",
        "attested-none",
        "missing",
        "bridge-incomplete",
    }


# ---- postures ------------------------------------------------------------


def test_missing_posture_withholds_filters_without_attestation() -> None:
    posture, filters, id_to_name = parse_attribution(
        None, None, assume_no_filters=False
    )
    assert posture == "missing"
    assert filters is None  # candidates withheld (invariant #6)
    assert id_to_name == {}


def test_attested_none_opens_the_gate_without_a_snapshot() -> None:
    posture, filters, id_to_name = parse_attribution(None, None, assume_no_filters=True)
    assert posture == "attested-none"
    assert filters == []
    assert id_to_name == {}


def test_absent_filters_short_circuit_ignores_the_bridge() -> None:
    # filters.json absent is decided BEFORE labels.json is consulted — the
    # CLI must be able to hand over whatever the second read returned
    posture, filters, _ = parse_attribution(None, "not json", assume_no_filters=False)
    assert posture == "missing"
    assert filters is None


def test_snapshot_posture_carries_filters_and_bridge() -> None:
    posture, filters, id_to_name = parse_attribution(
        FILTERS_L1, BRIDGE_L1, assume_no_filters=False
    )
    assert posture == "snapshot"
    assert filters == [
        {"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": ["L1"]}}
    ]
    assert id_to_name == {"L1": "Shipping"}


def test_empty_snapshot_pair_is_a_snapshot() -> None:
    # a mailbox genuinely without filters writes "[]" / "{}" — that is
    # EVIDENCE, not absence; the gate opens on the real snapshot
    posture, filters, id_to_name = parse_attribution(
        "[]", "{}", assume_no_filters=False
    )
    assert posture == "snapshot"
    assert filters == []
    assert id_to_name == {}


# ---- bridge-incomplete demotion (deepseek CRITICAL) -----------------------


def test_missing_bridge_demotes_to_bridge_incomplete_and_withholds() -> None:
    # filters.json without the labels.json bridge must NEVER surface
    # filter-authored labels as CANDIDATE
    posture, filters, id_to_name = parse_attribution(
        FILTERS_L1, None, assume_no_filters=False
    )
    assert posture == "bridge-incomplete"
    assert filters is None
    assert id_to_name == {}


def test_stale_bridge_demotes_too() -> None:
    posture, filters, id_to_name = parse_attribution(
        FILTERS_L1, '{"OTHER": "NotShipping"}', assume_no_filters=False
    )
    assert posture == "bridge-incomplete"
    assert filters is None
    assert id_to_name == {}


# ---- shape guards: wrong-shape text raises ValueError (CLI maps to exit 5) -


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[42]", "filter entry is not a string-keyed object"),
        ('{"not": "a list"}', "filters snapshot is not a list"),
    ],
)
def test_filters_payload_not_a_list_of_objects_raises(
    payload: str, message: str
) -> None:
    # "[42]" is a list whose ENTRY is not an object; '{"not": ...}' is not a
    # list at all — each pins its own message (transcribed from
    # test_analyze_wrong_shape_sidecars)
    with pytest.raises(ValueError, match=message):
        parse_attribution(payload, BRIDGE_L1, assume_no_filters=False)


def test_filter_entry_not_string_keyed_object_raises() -> None:
    with pytest.raises(ValueError, match="filter entry is not a string-keyed object"):
        parse_attribution('[["L1"]]', BRIDGE_L1, assume_no_filters=False)


def test_non_object_action_raises() -> None:
    # deepseek r3: an unreadable label binding must not contribute zero
    # needed ids and let the posture claim 'snapshot'
    with pytest.raises(ValueError, match="action must be an object"):
        parse_attribution(
            '[{"criteria": {"from": "shop@x.z"}, "action": "nope"}]',
            BRIDGE_L1,
            assume_no_filters=False,
        )


def test_non_list_add_label_ids_raises() -> None:
    # qwen MINOR2: a dict-instead-of-list ids field must not attribute nothing
    with pytest.raises(ValueError, match="addLabelIds must be a list of strings"):
        parse_attribution(
            '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": {"x": 1}}}]',
            BRIDGE_L1,
            assume_no_filters=False,
        )


def test_non_string_add_label_ids_raises() -> None:
    # deepseek MINOR: [42]-as-id must not bypass the bridge check
    with pytest.raises(ValueError, match="addLabelIds entries must be strings"):
        parse_attribution(
            '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": [42]}}]',
            BRIDGE_L1,
            assume_no_filters=False,
        )


@pytest.mark.parametrize("labels_text", ['["L1"]', '{"L1": 42}'])
def test_labels_snapshot_wrong_shape_raises(labels_text: str) -> None:
    # labels map wrong-shape raises too — non-object and non-string values
    with pytest.raises(ValueError, match="labels snapshot is not an id->name object"):
        parse_attribution("[]", labels_text, assume_no_filters=False)


# ---- existing-but-empty text is WRONG SHAPE, not absence ------------------


def test_empty_filters_text_flows_into_the_wrong_shape_error() -> None:
    # an empty file on disk is present-but-unparseable: JSONDecodeError (a
    # ValueError) — the CLI's exit-5 path, NOT the missing posture
    with pytest.raises(ValueError):
        parse_attribution("", None, assume_no_filters=False)


def test_empty_labels_text_flows_into_the_wrong_shape_error() -> None:
    with pytest.raises(ValueError):
        parse_attribution("[]", "", assume_no_filters=False)
