"""Sync Cursor sidecar (ADR-0008): the pure schema — names, version, render/parse.

The cursor is sync's ledger: ``sync-cursor.json`` in the corpus root carrying
``{version, history_id, last_sync}`` with ``last_sync`` tz-aware UTC ISO-8601.
This module owns the FORMAT contract only (pure core, CODING-STANDARDS); the
durable I/O lives on CorpusStore (read_cursor/write_cursor) and birth on
gmail_import. Unparseable OR invalid state fails loud — never a silent
re-birth, because guessing here is the silent-gap class the ledger prevents.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from gmail_labeler.sync_cursor import (
    CURSOR_VERSION,
    SIDECAR_CURSOR,
    SyncCursor,
    parse_cursor,
    render_cursor,
)

# Independent literal (not recomputed from render): the byte contract consumers
# and the AC3 byte-identical pin rest on this shape.
GOOD_TEXT = (
    '{"history_id": 4242, "last_sync": "2026-11-04T12:00:00+00:00", "version": 1}'
)
GOOD = SyncCursor(
    history_id=4242, last_sync=datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)
)

_MISSING = object()


def _cursor_json(**overrides: Any) -> str:
    """A valid cursor object with the given fields replaced (or deleted via
    the _MISSING sentinel) — mutation of the known-good shape, never a
    re-derivation of render's output."""
    base: dict[str, object] = {
        "version": 1,
        "history_id": 4242,
        "last_sync": "2026-11-04T12:00:00+00:00",
    }
    for key, value in overrides.items():
        if value is _MISSING:
            del base[key]
        else:
            base[key] = value
    return json.dumps(base)


def test_sidecar_name_is_the_adr_pinned_bare_filename() -> None:
    assert SIDECAR_CURSOR == "sync-cursor.json"
    assert CURSOR_VERSION == 1


def test_cursor_refuses_naive_or_non_utc_last_sync() -> None:
    # the sidecar's first format hole would be a naive timestamp (note 07 #7);
    # CorpusRecord.date enforces the same posture
    naive = datetime(2026, 11, 4, 12, 0, 0)
    with pytest.raises(ValueError, match="tz-aware UTC"):
        SyncCursor(history_id=1, last_sync=naive)
    with pytest.raises(ValueError, match="tz-aware UTC"):
        SyncCursor(
            history_id=1,
            last_sync=datetime(
                2026, 11, 4, 12, 0, 0, tzinfo=timezone(timedelta(hours=2))
            ),
        )


def test_cursor_refuses_negative_or_bool_history_id() -> None:
    with pytest.raises(ValueError, match=r"must be non-negative"):
        SyncCursor(history_id=-1, last_sync=GOOD.last_sync)
    # bool is an int subclass; JSON true must not read as 1 anywhere
    with pytest.raises(ValueError, match=r"must be an integer"):
        SyncCursor(history_id=True, last_sync=GOOD.last_sync)


def test_render_is_deterministic_sorted_json() -> None:
    assert render_cursor(GOOD) == GOOD_TEXT


def test_parse_roundtrips_the_rendered_contract() -> None:
    parsed = parse_cursor(GOOD_TEXT)
    assert parsed == GOOD
    assert render_cursor(parsed) == GOOD_TEXT


def test_parse_names_the_naive_timestamp_as_the_failure() -> None:
    # the naive stamp must fail with the refusal-to-guess message, not the
    # generic offset one (a naive value's utcoffset() is None, so the UTC
    # check would otherwise swallow this branch and leave the explicit
    # naive-guard unpinned)
    with pytest.raises(ValueError, match="naive"):
        parse_cursor(_cursor_json(last_sync="2026-11-04T12:00:00"))


INVALID_SHAPES: list[tuple[dict[str, Any], str]] = [
    ({"version": _MISSING}, "version-missing"),
    ({"version": 2}, "wrong-version"),
    ({"version": 1.0}, "version-float"),
    ({"version": "1"}, "version-wrong-type"),
    ({"version": True}, "version-bool"),
    ({"history_id": _MISSING}, "history-id-missing"),
    ({"history_id": None}, "history-id-null"),
    ({"history_id": -5}, "history-id-negative"),
    ({"history_id": "42"}, "history-id-string"),
    ({"history_id": 1.5}, "history-id-float"),
    ({"history_id": True}, "history-id-bool"),
    ({"last_sync": _MISSING}, "last-sync-missing"),
    ({"last_sync": None}, "last-sync-null"),
    ({"last_sync": "2026-11-04T12:00:00"}, "last-sync-naive"),
    ({"last_sync": "2026-11-04T12:00:00+02:00"}, "last-sync-non-utc"),
    ({"last_sync": "yesterday"}, "last-sync-unparseable"),
    ({"last_sync": 1730000000}, "last-sync-not-a-string"),
]


@pytest.mark.parametrize(
    ("overrides", "why"),
    INVALID_SHAPES,
    ids=[why for _, why in INVALID_SHAPES],
)
def test_parse_fails_loud_on_every_invalid_shape(
    overrides: dict[str, Any], why: str
) -> None:
    # the ADR's full invalid census: unparseable, wrong/missing version, and a
    # missing/null/negative/non-integer history_id, plus the last_sync holes —
    # each must raise (the store maps it to CorpusError), never half-parse
    with pytest.raises(ValueError, match="cursor"):
        parse_cursor(_cursor_json(**overrides))


def test_parse_fails_loud_on_structural_garbage() -> None:
    for text in ("not json at all", "[1, 2]", '"a string"', "null"):
        with pytest.raises(ValueError, match="cursor"):
            parse_cursor(text)
