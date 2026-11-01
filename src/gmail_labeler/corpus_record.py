"""The Corpus record format (GLOSSARY: Corpus) — pure core.

One record per message: identity, envelope essentials, normalized label
names (System Labels already filtered, GLOSSARY: Label Path), a hash of the
stored body, and the Date Provenance (GLOSSARY) — where the record's date
came from, stamped by the writer. The provenance is record data, not
derivable: temporal-split consumers filter on it, so a forged/future MIME
Date header can no longer silently move a record across the split
(invariant #7). Deliberately excludes anything derivable from the body —
this is metadata + a content fingerprint, and the jsonl line round-trips
byte-stably: the writer emits exactly one \n per record (json.dumps escapes
control chars but NOT U+2028/U+2029 under ensure_ascii=False), and the reader
splits on \n only — one line is exactly one record. Corpus files stay diffable
and resumable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

Source = Literal["api", "gyb"]

# Where the record's date came from (GLOSSARY: Date Provenance).
# Convention: writers stamp one of the first three; "unknown" is
# RESERVED for the reader — the explicit default for pre-existing lines
# written before the field existed (append-only corpus — no rewrite).
# No writer/reader type split: the constructor accepts all four, because
# the reader side (dataset build) consumes the same type.
DateProvenance = Literal["internal_date", "mime_date", "gyb_internaldate", "unknown"]
# KEEP IN SYNC with DateProvenance: the tuple is the runtime allow-list,
# the Literal is compile-time only — a value added to one without the
# other silently breaks validation.
_DATE_PROVENANCES = ("internal_date", "mime_date", "gyb_internaldate", "unknown")

# The server-authoritative trust class (GLOSSARY: Date Provenance) — the
# split-builder's correct filter; mime_date is sender-claimed, unknown legacy.
SERVER_AUTHORITATIVE_PROVENANCES: frozenset[DateProvenance] = frozenset(
    {"internal_date", "gyb_internaldate"}
)

# Gmail message ids are hex; the API/web show both cases (GYB stores lower).
_ID_RE = re.compile(r"^[0-9a-fA-F]{1,64}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def utc_datetime(dt: datetime) -> datetime:
    """Coerce a naive datetime to UTC; convert aware datetimes to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    id: str  # Gmail message id (hex); GYB uids are the same space
    date: datetime  # always timezone-aware UTC (constructor enforces)
    subject: str
    sender: str
    recipients: tuple[str, ...]
    labels: tuple[str, ...]  # normalized label *names*, no System Labels
    body_hash: str  # sha256 hex of the stored body text (UTF-8)
    source: Source
    date_provenance: DateProvenance  # where `date` came from (no default:
    # writers stamp it explicitly — "unknown" is the reader's, not a
    # writer's, choice)
    thread_id: str = ""

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.id):
            raise ValueError("CorpusRecord.id must be 1-64 hex chars")
        if self.date.tzinfo is None or self.date.utcoffset() != timedelta(0):
            raise ValueError("CorpusRecord.date must be timezone-aware UTC")
        if not _HASH_RE.fullmatch(self.body_hash):
            raise ValueError("CorpusRecord.body_hash must be 64 hex chars")
        if self.source not in ("api", "gyb"):
            raise ValueError(
                f"CorpusRecord.source must be 'api' or 'gyb', got {self.source!r}"
            )
        if self.date_provenance not in _DATE_PROVENANCES:
            raise ValueError(
                "CorpusRecord.date_provenance must be one of "
                f"{_DATE_PROVENANCES}, got {self.date_provenance!r}"
            )

    def to_json_line(self) -> str:
        payload = {
            "id": self.id,
            "date": self.date.isoformat(),
            "subject": self.subject,
            "sender": self.sender,
            "recipients": list(self.recipients),
            "labels": list(self.labels),
            "body_hash": self.body_hash,
            "source": self.source,
            "date_provenance": self.date_provenance,
            "thread_id": self.thread_id,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json_line(cls, line: str) -> CorpusRecord:
        data: object = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"record line is {type(data).__name__}, expected object")
        payload = cast("dict[str, object]", data)  # narrowed by isinstance above
        date = datetime.fromisoformat(_need(payload, "date", str))
        if date.tzinfo is None:
            raise ValueError("record date is naive; refusing to guess a timezone")
        return cls(
            id=_need(payload, "id", str),
            date=date.astimezone(UTC),
            subject=_need(payload, "subject", str),
            sender=_need(payload, "sender", str),
            recipients=_need_str_list(payload, "recipients"),
            labels=_need_str_list(payload, "labels"),
            body_hash=_need(payload, "body_hash", str),
            source=cast("Source", _need(payload, "source", str)),
            date_provenance=_opt_date_provenance(payload),
            thread_id=_need_opt_str(payload, "thread_id"),
        )


def _opt_date_provenance(payload: dict[str, object]) -> DateProvenance:
    value = payload.get("date_provenance")
    if value is None:
        # append-only corpus: a line written before the field existed stays
        # readable; the default is an EXPLICIT unknown, never a guess.
        # An explicit null is treated as absent on purpose: a fail-safe
        # read beats a hard stop — one exotic producer bug must not kill
        # iteration over the whole corpus (deliberate divergence from the
        # required fields, which reject null loudly; the optional
        # thread_id is equally tolerant).
        return "unknown"
    if not isinstance(value, str) or value not in _DATE_PROVENANCES:
        raise ValueError(
            f"field 'date_provenance' must be one of {_DATE_PROVENANCES}, got {value!r}"
        )
    return value  # narrowed to the Literal by the membership check


def _need_opt_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            f"field {key!r} must be str or absent, got {type(value).__name__}"
        )
    return value


def _need[T](payload: dict[str, object], key: str, typ: type[T]) -> T:
    value = payload.get(key)
    if not isinstance(value, typ):
        raise ValueError(
            f"field {key!r} must be {typ.__name__}, got {type(value).__name__}"
        )
    return value


def _need_str_list(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"field {key!r} must be a list of strings")
    items = cast("list[object]", value)  # narrowed by the isinstance above
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"field {key!r} must be a list of strings")
        out.append(item)
    return tuple(out)


def body_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def with_body(record: CorpusRecord, text: str) -> CorpusRecord:
    """Return a copy of `record` carrying the body fingerprint."""
    return replace(record, body_hash=body_hash(text))
