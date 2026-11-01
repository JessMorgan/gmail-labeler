"""Per-label corpus metrics for choosing Managed Labels (GLOSSARY: Managed
Label; `dataset analyze`).

Pure core (CODING-STANDARDS): deterministic on any input — ``now`` is a
required argument, never read from the clock; sorting is total and stable.

The three hazards this report exists to surface, per AGENTS.md:

* **Filter attribution (invariant #6)**: a label your own Gmail filters
  already apply is not trainable signal. Filters whose criteria are
  simulatable FROM the record fields we keep (from/to/subject substring
  containment) attribution matching messages. Approximations, all biased
  toward withholding EXCEPT one documented residual: `to` is matched
  against all kept recipients (To+Cc+Bcc — over-attribution), and from/to
  values are simulated ONLY when address-shaped (an `@`, no name syntax);
  display-name-style criteria like 'Nike' are unverifiable, not False.
  RESIDUAL toward candidacy: an address-SHAPED token Gmail matched only
  via the display name (e.g. 'nike.com' in a name but not the address)
  simulates False while Gmail matches. Bounded and rare; closing it means
  storing display names in the corpus — a schema change GYB import could
  not populate. Filters carrying criteria we CANNOT simulate
  (``query``, ``negatedQuery``, ``hasAttachment``, ``size``, ...) mark the
  label ``filter-unverifiable`` — exclusion, not guessing: graduation
  must earn itself on what filters do NOT cover.
* **Recency**: total counts over-weight old mail;
  a label must ALSO have recent-window volume to be a candidate.
* **Templates**: near-duplicate bodies fabricate accuracy; a
  template-heavy label's counts are inflated and it is flagged.

The temporal split itself (invariant #7) is downstream (eval); this module
only reports the windows it will see.

``LabelMetrics.reasons`` order is a contract for CLI/--json consumers:
no-filter-snapshot, low-volume, low-recent, template-heavy,
filter-covered, filter-unverifiable.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .corpus_record import CorpusRecord

#: Criteria keys we can simulate faithfully from kept record fields.
#: Anything else in a filter's criteria makes its attribution unverifiable.
SIMULATABLE_CRITERIA = frozenset({"from", "to", "subject"})

_DOMAINISH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+")


@dataclass(frozen=True, slots=True)
class CorpusSummary:
    messages: int
    distinct_labels: int
    first_date: datetime | None
    last_date: datetime | None
    duplicate_bodies: int  # bodies shared by 2+ messages (template signal)


@dataclass(frozen=True, slots=True)
class LabelMetrics:
    label: str
    total: int
    recent: int  # within the recency window
    dup_rate: float  # share of this label's messages on duplicate bodies
    filter_attributed: int  # messages explained by SIMULATABLE filters
    attribution_ratio: float  # filter_attributed / total (0.0 if total == 0).
    # With unreliable_filters > 0 this is a STRICT LOWER bound: unverifiable
    # filters contribute no attributed messages, only exclusion.
    unreliable_filters: int  # filters adding this label we cannot simulate
    candidate: bool
    reasons: tuple[str, ...]  # why not a candidate (empty iff candidate)


def months_ago(now: datetime, months: int) -> datetime:
    """Exact calendar month arithmetic (no 30-day approximations), UTC.
    months >= 0 only (a negative window would place the cutoff in the
    future and silently fail every label); months=0 is the identity."""
    if now.tzinfo is None:  # defensive: pure functions take aware datetimes
        raise ValueError("now must be timezone-aware")
    if months < 0:
        raise ValueError(f"months must be >= 0, got {months}")
    total = now.year * 12 + (now.month - 1) - months  # O(1), no drift loops
    year, month = divmod(total, 12)
    month += 1
    # keep the day when the target month has it (months=0 is the identity);
    # otherwise clamp to that month's last day (Mar 31 -1mo -> Feb 28)
    try:
        last_day = calendar.monthrange(year, month)[1]
        return now.replace(year=year, month=month, day=min(now.day, last_day))
    except (ValueError, OverflowError) as e:  # pathological months: name it
        raise ValueError(f"months={months} is out of representable range") from e


def _contains_ci(haystack: str, needle: object) -> bool:
    return isinstance(needle, str) and needle.lower() in haystack.lower()


def _address_shaped(value: object) -> bool:
    """A from/to criterion is honestly simulatable against stored
    addr-specs only when it can ONLY be matching the address: an '@'
    (full address or local part) or a bare domain-shaped token
    ('nike.com'), with no name-syntax (spaces, angle brackets, quotes).
    A bare word like 'Nike' is (likely) display-name matching we
    cannot see — unverifiable, never False."""
    if not isinstance(value, str) or any(c in value for c in " <>\"'"):
        return False
    return "@" in value or bool(_DOMAINISH_RE.fullmatch(value))


def filter_label_ids_strict(filter_: Mapping[str, object]) -> list[str]:
    """The label ids a filter's action adds — refusing a present-but-
    wrong-shaped action/addLabelIds with ValueError (the cli.py guard):
    unreadable bindings must not contribute zero ids and let the filter's
    labels surface un-attributed. The shape rule lives in exactly one
    place, shared by the CLI bridge check and the dataset builder."""
    action = filter_.get("action")
    if action is not None and not isinstance(action, Mapping):
        raise ValueError("action must be an object")
    # JSON payload: cheap isinstance checks + casts, strict-mode style
    ids = (
        cast(Mapping[str, object], action).get("addLabelIds")
        if isinstance(action, Mapping)
        else None
    )
    if ids is not None and not isinstance(ids, list):
        raise ValueError("addLabelIds must be a list of strings")
    if isinstance(ids, list):
        items = cast("list[object]", ids)  # narrowed by the isinstance above
        if any(not isinstance(i, str) for i in items):
            raise ValueError("addLabelIds entries must be strings")
        return cast("list[str]", items)  # narrowed by the all-str check
    return []


def filter_label_ids(filter_: Mapping[str, object]) -> list[str]:
    """The label ids a filter's action adds (lenient — EXACT base
    ``_filter_label_names`` parity: a present-but-wrong-shaped action
    contributes no ids, and list entries are str-coerced, so a type-
    corrupt id like 42 attributes as "42" rather than vanishing; the
    analyze report treats such a filter as touching that label). Strict
    consumers (the dataset builder, the CLI bridge check) use
    filter_label_ids_strict, which refuses non-str entries."""
    action = filter_.get("action")
    ids = (
        cast(Mapping[str, object], action).get("addLabelIds")
        if isinstance(action, Mapping)
        else None
    )
    if not isinstance(ids, list):
        return []
    return [str(i) for i in cast(list[object], ids)]


def simulates_filter(
    criteria: Mapping[str, object], record: CorpusRecord
) -> bool | None:
    """True/False when faithfully simulatable; None when not. Shared with
    the dataset builder: the simulation doctrine (invariant #6) lives in
    exactly one place, so attribution and exclusion can never drift."""
    if set(criteria) - SIMULATABLE_CRITERIA:
        return None
    for key in criteria:
        value = criteria[key]
        if not isinstance(value, str):
            return None  # unexpected shape: refuse to guess
    # VERIFIABILITY is checked before CONTAINMENT: whether a criterion is
    # simulatable may not depend on which record we happen to examine
    # first (deepseek MINOR: {from: 'a@b.c', to: 'Nike'} returned False
    # from the from-check and the unverifiable 'to' was never seen).
    for key in ("from", "to"):
        if key in criteria and not _address_shaped(criteria[key]):
            # Gmail matches RAW filter text against the whole header,
            # display name included; records keep only addr-specs, so a
            # name-shaped value CANNOT be simulated — calling it False
            # would under-attribute (invariant #6's unsafe direction).
            return None
    if "from" in criteria and not _contains_ci(record.sender, criteria["from"]):
        return False
    if "subject" in criteria and not _contains_ci(record.subject, criteria["subject"]):
        return False
    if "to" in criteria:
        return any(
            _contains_ci(recipient, criteria["to"]) for recipient in record.recipients
        )
    return True


def analyze_corpus(
    records: Sequence[CorpusRecord],
    *,
    filters: Sequence[Mapping[str, object]] | None,
    id_to_name: Mapping[str, str],
    now: datetime,
    recent_months: int = 24,
    min_total: int = 50,
    min_recent: int = 10,
    max_dup_rate: float = 0.5,
    max_attribution_ratio: float = 0.5,
) -> tuple[CorpusSummary, list[LabelMetrics]]:
    """Compute the report. ``filters=None`` means NO snapshot was provided:
    attribution is then UNKNOWN and no label can be a candidate (invariant
    #6 refuses to pretend; the CLI surfaces the --assume-no-filters escape
    for filterless mailboxes). Duplicate record ids would inflate the
    volume gates while attribution dedupes by id — id uniqueness is
    the importers' contract, assumed here."""
    cutoff = months_ago(now, recent_months)
    body_counts: dict[str, int] = {}
    for rec in records:
        body_counts[rec.body_hash] = body_counts.get(rec.body_hash, 0) + 1
    dates = [rec.date for rec in records]
    labels_seen = sorted({name for rec in records for name in rec.labels})
    summary = CorpusSummary(
        messages=len(records),
        distinct_labels=len(labels_seen),
        first_date=min(dates) if dates else None,
        last_date=max(dates) if dates else None,
        duplicate_bodies=sum(1 for c in body_counts.values() if c > 1),
    )

    # Per label: resolved member records, and the filter accounting.
    members: dict[str, list[CorpusRecord]] = {name: [] for name in labels_seen}
    for rec in records:
        for name in rec.labels:
            members.setdefault(name, []).append(rec)

    attributed: dict[str, set[str]] = {name: set() for name in labels_seen}
    unreliable: dict[str, int] = {name: 0 for name in labels_seen}
    if filters:
        for filter_ in filters:
            raw_criteria = filter_.get("criteria")
            # absent criteria = a real empty-criteria filter (matches all,
            # simulatable); a PRESENT non-mapping is snapshot corruption —
            # match-all would mislabel it 'filter-covered', so name it
            malformed_criteria = raw_criteria is not None and not isinstance(
                raw_criteria, Mapping
            )
            criteria: Mapping[str, object] = (
                {}
                if raw_criteria is None or not isinstance(raw_criteria, Mapping)
                else cast(Mapping[str, object], raw_criteria)
            )
            names = [
                id_to_name.get(label_id, f"id:{label_id}")
                for label_id in filter_label_ids(filter_)
            ]
            for name in names:
                if name not in members:
                    continue
                if malformed_criteria:
                    unreliable[name] += 1
                    continue
                verdicts = [simulates_filter(criteria, rec) for rec in members[name]]
                if None in verdicts:
                    # this filter unverifiably touches this label — on ANY
                    # message: the whole label becomes unverifiable
                    unreliable[name] += 1
                else:
                    for rec, hit in zip(members[name], verdicts, strict=True):
                        if hit:
                            attributed[name].add(rec.id)

    metrics: list[LabelMetrics] = []
    for name in labels_seen:
        recs = members[name]
        total = len(recs)
        recent = sum(1 for r in recs if r.date >= cutoff)
        dup = sum(1 for r in recs if body_counts[r.body_hash] > 1)
        dup_rate = dup / total if total else 0.0
        attr = len(attributed[name])
        ratio = attr / total if total else 0.0
        bad = unreliable[name]

        reasons: list[str] = []
        if filters is None:
            reasons.append("no-filter-snapshot")
        if total < min_total:
            reasons.append("low-volume")
        if recent < min_recent:
            reasons.append("low-recent")
        if dup_rate > max_dup_rate:
            reasons.append("template-heavy")
        if ratio > max_attribution_ratio:
            reasons.append("filter-covered")
        if bad:
            reasons.append("filter-unverifiable")
        metrics.append(
            LabelMetrics(
                label=name,
                total=total,
                recent=recent,
                dup_rate=dup_rate,
                filter_attributed=attr,
                attribution_ratio=ratio,
                unreliable_filters=bad,
                candidate=not reasons,
                reasons=tuple(reasons),
            )
        )

    metrics.sort(key=lambda m: (not m.candidate, -m.total, m.label))
    return summary, metrics
