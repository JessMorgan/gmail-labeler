"""Sync Cursor sidecar (ADR-0008): the ledger's format contract — pure core.

``sync-cursor.json`` is a Corpus's record of how far it has consumed the
mailbox's event stream (GLOSSARY: Sync Cursor): versioned JSON
``{version, history_id, last_sync}`` living in the corpus root, with
``last_sync`` tz-aware UTC ISO-8601. This module owns the NAME and the
parse/render contract only (CODING-STANDARDS pure core: no I/O, no Gmail or
config imports); the durable I/O is ``CorpusStore.read_cursor``/
``write_cursor`` on the established snapshot discipline, and birth belongs to
Gmail Import (which seeds it) while advancement belongs exclusively to Label
Sync.

Loud-by-doctrine, like the attribution snapshot parser: a cursor that exists
but does not parse *or* validate — wrong/missing version, or a missing, null,
negative, or non-integer ``history_id``, or a missing/naive/non-UTC
``last_sync`` — raises. The store maps this to ``CorpusError``; callers must
NEVER treat invalid state as absent, because guessing here is the silent-gap
class the ledger exists to prevent (ADR-0008).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TypeGuard, cast

SIDECAR_CURSOR = "sync-cursor.json"
CURSOR_VERSION = 1


def _strict_int(value: object) -> TypeGuard[int]:
    """int, and honestly an int: bool (JSON true/false) is an int subclass
    and must not read as 1/0 anywhere in the ledger."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class SyncCursor:
    history_id: int  # the mailbox history id this Corpus has consumed up to
    last_sync: datetime  # always timezone-aware UTC (constructor enforces)

    def __post_init__(self) -> None:
        # _strict_int takes object: the guard defends against JSON/Any
        # callers the declared int type does not describe at runtime
        # (parse_cursor guards the same hole on the read side)
        if not _strict_int(self.history_id):
            raise ValueError(f"history_id must be an integer, got {self.history_id!r}")
        if self.history_id < 0:
            raise ValueError(f"history_id must be non-negative, got {self.history_id}")
        if self.last_sync.tzinfo is None or self.last_sync.utcoffset() != timedelta(0):
            raise ValueError("SyncCursor.last_sync must be tz-aware UTC")


def render_cursor(cursor: SyncCursor) -> str:
    """The on-disk JSON: sorted keys, default separators — byte-stable for a
    given cursor, so an untouched cursor is byte-identically comparable
    across runs (AC3)."""
    payload = {
        "version": CURSOR_VERSION,
        "history_id": cursor.history_id,
        "last_sync": cursor.last_sync.isoformat(),
    }
    return json.dumps(payload, sort_keys=True)


def parse_cursor(text: str) -> SyncCursor:
    """Parse + validate sidecar text; raises ValueError on every shape the
    ADR refuses (the store wraps it into CorpusError)."""
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"sync cursor is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"sync cursor is {type(raw).__name__}, expected object")
    payload = cast("dict[str, object]", raw)  # narrowed by isinstance above

    version = payload.get("version")
    # strict int posture, same as history_id: a JSON float 1.0 == 1 would
    # otherwise slip through a bare equality (deepseek review, segment 1)
    if not _strict_int(version) or version != CURSOR_VERSION:
        raise ValueError(
            f"sync cursor version must be the integer {CURSOR_VERSION}, got {version!r}"
        )

    history_id = payload.get("history_id")
    if not _strict_int(history_id):
        raise ValueError(
            f"sync cursor history_id must be an integer, got {history_id!r}"
        )
    if history_id < 0:
        raise ValueError(
            f"sync cursor history_id must be non-negative, got {history_id}"
        )

    raw_sync = payload.get("last_sync")
    if not isinstance(raw_sync, str):
        raise ValueError(
            f"sync cursor last_sync must be an ISO-8601 string, got {raw_sync!r}"
        )
    try:
        last_sync = datetime.fromisoformat(raw_sync)
    except ValueError as e:
        raise ValueError(f"sync cursor last_sync is not ISO-8601: {e}") from e
    if last_sync.tzinfo is None:
        raise ValueError("sync cursor last_sync is naive; refusing to guess a timezone")
    if last_sync.utcoffset() != timedelta(0):
        raise ValueError("sync cursor last_sync must be UTC")
    return SyncCursor(history_id=history_id, last_sync=last_sync.astimezone(UTC))
