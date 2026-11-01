"""Corpus → training examples + versioned split manifest (Phase 1 core).

Pure core (CODING-STANDARDS): deterministic on any input — ``now`` and
``split_date`` are required arguments, never read from the clock; sorting
is total and stable; no I/O — consumers pass already-loaded structures
(the analyze discipline).

The vocabulary is Label Paths (GLOSSARY: Label Path): classes are the full
label names as seen in the corpus — ``Money/Bills`` is a distinct class
from ``Money`` — and membership is path-aware through the labels.py
primitives (``label_path`` / ``path_is_at_or_under``), never re-parsed ad
hoc: exact membership never crosses paths, while a parent class may scope
its subtree.

The split is temporal ONLY (invariant #7): there is no other split path
in this module — no module-level rng import (pinned by
``test_no_random_in_module_namespace``) and no input-order dependence
(pinned by the shuffled-input manifest test). ``train`` and ``eval``
rebuild through ``build_dataset`` against the same Split Manifest
(invariant #7): same records + manifest → same examples, no re-split —
the manifest is authoritative over the caller's split date and trust
filter, so the rebuild is reproducible from the artifact alone.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, cast

from .analyze import filter_label_ids_strict, months_ago, simulates_filter
from .corpus_record import SERVER_AUTHORITATIVE_PROVENANCES, DateProvenance
from .labels import label_path, path_is_at_or_under

if TYPE_CHECKING:
    from .corpus_record import CorpusRecord

#: Split Manifest format version (GLOSSARY: Split Manifest). Bump on any
#: change to the manifest's field set or semantics. v3 adds the recency-
#: weighting fields (recent_months, old_weight, weight_cutoff).
FORMAT_VERSION = 3


class BuildAbortedError(ValueError):
    """The build refuses to run (invariant #6): filter attribution is
    impossible, so Filter-Authored evidence cannot be excluded.
    ``reason``: 'no-filter-snapshot' (no snapshot provided),
    'bridge-incomplete' (a filter's label ids not all resolvable through
    the id→name map — a blank mapped name is unresolvable), or
    'bridge-malformed' (a filter element is not a Mapping, or its
    action/addLabelIds is present but wrong-shaped — unreadable bindings
    must not contribute zero ids).

    Subclasses ``ValueError`` so a CLI mapping (ticket #19/#25) can catch
    the whole build-refusal family with one ``except ValueError`` — the
    same class the manifest's ``validate()``/``from_json`` raise."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One corpus record turned for training (GLOSSARY: Training
    Example). ``labels`` are the record's label names; ``weight`` is the
    per-example Recency Weight (see ``example_weight``);
    ``date_provenance`` is stamped so temporal-split consumers can filter
    fallback-sourced records at the split boundary (invariant #7)."""

    id: str
    date: datetime
    labels: tuple[str, ...]
    weight: float
    date_provenance: DateProvenance


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """The versioned, hashable artifact materializing the temporal
    train/test split (GLOSSARY: Split Manifest). ``content_hash`` is the
    sha256 of the canonical JSON of every other field — a machine-
    checkable identity the checkpoint sidecar will carry; a manifest
    whose hash does not match its fields is stale or corrupt and is
    refused, never served."""

    format_version: int
    split_date: datetime
    vocabulary: tuple[str, ...]  # sorted full label names (Label Paths)
    train_ids: tuple[str, ...]  # sorted
    test_ids: tuple[str, ...]  # sorted
    server_authoritative_only: bool  # split-boundary trust filter (invariant #7)
    bridge_hash: str  # sha256 of the canonical form of the effective bridge:
    # pinned so a drifted filters/labels snapshot refuses the rebuild loudly
    recent_months: int  # recency-weighting window (analyze's months_ago shape)
    old_weight: float  # weight outside the window (named, never ad-hoc)
    weight_cutoff: datetime  # months_ago(now, recent_months) at build time,
    # UTC-normalized: the weighting anchor — manifest-input mode rebuilds
    # weights from it, so the rebuild is reproducible from the artifact
    # alone. Scoped to fixed-offset/UTC-normalized inputs (a DST-zone
    # offset change would shift the wall-clock cutoff; callers should pass
    # `now` already in UTC for cross-host stability).
    content_hash: str

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "split_date": self.split_date.isoformat(),
            "vocabulary": list(self.vocabulary),
            "train_ids": list(self.train_ids),
            "test_ids": list(self.test_ids),
            "server_authoritative_only": self.server_authoritative_only,
            "bridge_hash": self.bridge_hash,
            "recent_months": self.recent_months,
            "old_weight": self.old_weight,
            "weight_cutoff": self.weight_cutoff.isoformat(),
        }

    def compute_hash(self) -> str:
        """The sha256 of the canonical JSON of every field except
        ``content_hash`` itself — the identity ``from_json`` verifies."""
        canonical = json.dumps(
            self._canonical_payload(),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Canonical serialization (sorted keys, ASCII, compact) —
        byte-stable for identical fields."""
        payload = self._canonical_payload()
        payload["content_hash"] = self.content_hash
        return json.dumps(
            payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, text: str) -> SplitManifest:
        """Parse + verify: wrong shape, an unknown format version, a
        naive date, a violated format invariant (sorted/unique id sets,
        disjoint train/test), or a content_hash that does not match the
        fields raises ValueError — the manifest is a machine-checkable
        identity; a hash-consistent artifact that violates the
        invariants is corrupt, not a valid variant."""
        data: object = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("manifest is not an object")
        payload = cast("dict[str, object]", data)
        try:
            manifest = cls(
                format_version=_need_int(payload, "format_version"),
                split_date=datetime.fromisoformat(_need_str(payload, "split_date")),
                vocabulary=_need_str_list(payload, "vocabulary"),
                train_ids=_need_str_list(payload, "train_ids"),
                test_ids=_need_str_list(payload, "test_ids"),
                server_authoritative_only=_need_bool(
                    payload, "server_authoritative_only"
                ),
                bridge_hash=_need_str(payload, "bridge_hash"),
                recent_months=_need_int(payload, "recent_months"),
                old_weight=_need_float(payload, "old_weight"),
                weight_cutoff=datetime.fromisoformat(
                    _need_str(payload, "weight_cutoff")
                ),
                content_hash=_need_str(payload, "content_hash"),
            )
        except ValueError as e:
            raise ValueError(f"manifest field invalid: {e}") from e
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """Every format invariant, called from BOTH entry paths —
        ``from_json`` and the in-memory manifest-input mode — so a
        manifest object passed directly is refused exactly like a parsed
        one: stale hash, unknown version, naive dates, violated
        invariants."""
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"manifest format_version {self.format_version} != {FORMAT_VERSION}"
            )
        # UTC-normalized: a non-UTC offset would make the same instant
        # hash differently on a tz-offset host (probed)
        if self.split_date.tzinfo is None or self.split_date.utcoffset() != timedelta(
            0
        ):
            raise ValueError("manifest split_date must be timezone-aware UTC")
        if self.weight_cutoff.tzinfo is None or (
            self.weight_cutoff.utcoffset() != timedelta(0)
        ):
            raise ValueError("manifest weight_cutoff must be timezone-aware UTC")
        if self.recent_months < 0:
            raise ValueError("manifest recent_months must be >= 0")
        if not 0.0 <= self.old_weight <= 1.0:
            raise ValueError(
                f"manifest old_weight must be in [0.0, 1.0], got {self.old_weight}"
            )
        if not sorted_unique(self.vocabulary):
            raise ValueError("manifest vocabulary must be sorted and unique")
        if not sorted_unique(self.train_ids):
            raise ValueError("manifest train_ids must be sorted and unique")
        if not sorted_unique(self.test_ids):
            raise ValueError("manifest test_ids must be sorted and unique")
        if set(self.train_ids) & set(self.test_ids):
            raise ValueError("manifest train_ids and test_ids must be disjoint")
        if self.content_hash != self.compute_hash():
            raise ValueError("manifest content_hash does not match its fields")


@dataclass(frozen=True, slots=True)
class BuildResult:
    """The build's output: the split manifest + examples sorted by record
    id — a total order, input-order invariant.

    Exclusion observability (NOT in the manifest/hash — these describe the
    build's exclusion decisions, not the split identity):
    ``excluded_classes`` is the sorted (label, reason) pairs for classes
    whose evidence was wholly excluded (reason 'unverifiable' or
    'malformed'), so a filter-heavy mailbox's vanished classes leave a
    trace; ``emptied_examples`` counts the records that had labels but
    came out all-negative (mass-negative targets are visible, not silent)."""

    manifest: SplitManifest
    examples: tuple[TrainingExample, ...]
    excluded_classes: tuple[tuple[str, str], ...] = ()
    emptied_examples: int = 0

    def members_of(self, class_name: str) -> tuple[str, ...]:
        """Record ids carrying ``class_name`` exactly. Its label path is
        one class: ``Money/Bills`` is NOT a member of ``Money``."""
        return tuple(
            sorted(
                example.id for example in self.examples if class_name in example.labels
            )
        )

    def subtree_members(self, class_name: str) -> tuple[str, ...]:
        """Path-aware scoping (GLOSSARY: Label Path): ids carrying
        ``class_name`` or any descendant of its label path — how a parent
        Managed Label's gate scopes its subtree. Comparison is per-segment
        (``Money2`` is not under ``Money``)."""
        ancestor = label_path(class_name)
        if not ancestor:
            return ()
        return tuple(
            sorted(
                example.id
                for example in self.examples
                if any(
                    path_is_at_or_under(label_path(name), ancestor)
                    for name in example.labels
                )
            )
        )


def example_weight(date: datetime, *, cutoff: datetime, old_weight: float) -> float:
    """Recency weight (GLOSSARY: Recency Weight): 1.0 at or after the
    recency cutoff, ``old_weight`` before it. The cutoff is analyze's
    exact calendar-month shape (``months_ago(now, recent_months)``) — no
    new decay curve — and it rides in the Split Manifest, so a 21-year
    corpus cannot be dominated by its oldest mail and a manifest-input
    rebuild is reproducible from the artifact alone."""
    return 1.0 if date >= cutoff else old_weight


def build_dataset(
    records: Sequence[CorpusRecord],
    *,
    filters: Sequence[Mapping[str, object]] | None,
    id_to_name: Mapping[str, str],
    now: datetime,
    split_date: datetime,
    manifest: SplitManifest | None = None,
    server_authoritative_only: bool = False,
    recent_months: int = 24,
    old_weight: float = 0.1,
) -> BuildResult:
    """Turn CorpusRecords + the filters/labels bridge into training
    examples + a versioned split manifest.

    Invariant #6 first: ``filters is None`` (no snapshot), a filter whose
    label ids are not all resolvable through ``id_to_name``
    (bridge-incomplete — a blank mapped name is unresolvable, the
    labels.py rule), a non-Mapping filter element, or a filter
    action/addLabelIds present but wrong-shaped (bridge-malformed, the
    cli.py guard) aborts with ``BuildAbortedError`` — the build never
    trains on Filter-Authored evidence it cannot exclude.

    Invariant #7: the split is temporal only — records with
    ``date < split_date`` are train, the rest test. ``manifest`` pins a
    prior split: membership then comes from the manifest (no re-split),
    records it does not know are excluded, and the manifest — split
    date, trust filter — is authoritative over the caller's arguments.
    The manifest's full format invariants are re-validated (an in-memory
    manifest is refused exactly like a parsed one), a bridge whose hash
    does not match the provided filters/labels snapshot refuses the
    rebuild, and a two-way drift check refuses the rebuild when a
    declared member is absent or the labels over the members no longer
    equal the declared vocabulary (a record whose date moved across the
    split is NOT refused — membership, not dates, is authoritative).

    ``server_authoritative_only`` drops fallback-sourced records
    (``mime_date``/``unknown`` Date Provenance) out of the split before
    it; the default stamps them on their examples so eval can filter.

    Recency weighting: each example carries ``example_weight`` — 1.0 at
    or after the cutoff (``months_ago(now, recent_months)``, UTC-
    normalized), ``old_weight`` before it; the cutoff rides in the
    manifest. Whether an eval applies the weight is the consumer's choice
    (the example carries it; the split identity does not depend on it).

    Determinism: examples sorted by id with sorted labels; manifest id
    sets, vocabulary, and the weighting cutoff (``weight_cutoff``)
    sorted/UTC-normalized; identical inputs give byte-identical manifests
    (``split_date`` is stored UTC-normalized). ``now`` and ``split_date``
    are required: the core never reads the clock (``now`` anchors the
    weighting cutoff). Duplicate record ids are rejected up front (the
    store guarantees uniqueness).
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if split_date.tzinfo is None:
        raise ValueError("split_date must be timezone-aware")
    split_date = split_date.astimezone(UTC)  # canonical instant in the manifest

    if filters is None:
        raise BuildAbortedError("no-filter-snapshot")
    # single pass: collect the strict ids once, shared by the bridge check
    # and the exclusion below. The input is untrusted at runtime (a
    # non-Mapping element is possible despite the annotation) — scan it as
    # object and narrow. A non-Mapping filter element, or a wrong-shaped
    # action/addLabelIds (the shape rule lives in
    # analyze.filter_label_ids_strict, shared with the CLI), is
    # bridge-malformed — unreadable bindings must not contribute zero ids.
    filter_ids: list[list[str]] = []
    needed: set[str] = set()
    for filter_ in cast("Sequence[object]", filters):
        if not isinstance(filter_, Mapping):
            raise BuildAbortedError("bridge-malformed")
        try:
            ids = filter_label_ids_strict(cast(Mapping[str, object], filter_))
        except ValueError:
            raise BuildAbortedError("bridge-malformed") from None
        filter_ids.append(ids)
        needed.update(ids)
    # a blank/whitespace mapped name is as unusable as an unresolved id
    # (labels.normalize_label_names namespaces it id: — it must never
    # enter the vocabulary)
    resolvable = {lid for lid, name in id_to_name.items() if name.strip()}
    if needed - resolvable:
        raise BuildAbortedError("bridge-incomplete")

    bridge_hash = _bridge_hash(filters, filter_ids, id_to_name)

    if manifest is not None:
        # the full format validation, not just the version — an in-memory
        # manifest is refused exactly like a parsed one
        manifest.validate()
        if manifest.bridge_hash != bridge_hash:
            raise ValueError(
                "manifest bridge_hash does not match the provided "
                "filters/labels bridge — the snapshot drifted, rebuild "
                "the manifest"
            )
        # the manifest's parameters are authoritative — same records +
        # manifest → same examples, no re-split, no re-trust-filtering,
        # no re-weighting (``now``/``recent_months``/``old_weight`` are
        # ignored here: the manifest's weight_cutoff is the anchor)
        server_authoritative_only = manifest.server_authoritative_only
        old_weight = manifest.old_weight
        cutoff = manifest.weight_cutoff
    else:
        # mint path: bounds-check the caller's weighting args (manifest
        # mode does NOT re-check — it uses the manifest's validated values)
        if isinstance(old_weight, bool):
            # True would pass 0.0 <= x <= 1.0 (True == 1.0) and mint weight
            # 1.0 — a bool is not a weight, reject it explicitly
            raise ValueError(
                f"old_weight must be a float in [0.0, 1.0], got {old_weight!r}"
            )
        if not 0.0 <= old_weight <= 1.0:
            raise ValueError(f"old_weight must be in [0.0, 1.0], got {old_weight}")
        if recent_months < 0:
            raise ValueError(f"recent_months must be >= 0, got {recent_months}")
        # coerce at mint so int-vs-float (old_weight=1 vs 1.0) and
        # int recent_months hash identically and round-trip through
        # from_json (which requires a float old_weight)
        old_weight = float(old_weight)
        recent_months = int(recent_months)
        cutoff = months_ago(now, recent_months).astimezone(UTC)

    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise ValueError(f"duplicate record id: {record.id}")
        seen.add(record.id)

    if server_authoritative_only:
        records = [
            record
            for record in records
            if record.date_provenance in SERVER_AUTHORITATIVE_PROVENANCES
        ]

    # Filter-Authored exclusion (invariant #6): per (filter, label) pair,
    # simulate against the label's member records — a simulatable filter
    # excludes the messages it covers; an unverifiable one (non-
    # simulatable criteria, malformed shape) excludes the WHOLE label's
    # evidence. Exclusion, not guessing (analyze's doctrine, shared).
    # The member index is built once (analyze's shape): O(N), not
    # O(filters x N).
    members_by_label: dict[str, list[CorpusRecord]] = {}
    for record in records:
        for name in record.labels:
            members_by_label.setdefault(name, []).append(record)

    covered: dict[str, set[str]] = {}
    unverifiable_labels: set[str] = set()  # non-simulatable criteria
    malformed_labels: set[str] = set()  # malformed criteria shape
    for filter_, ids in zip(filters, filter_ids, strict=True):
        # id_to_name via .get + explicit abort (no latent KeyError — the
        # bridge check above guarantees resolvability, this is defensive)
        names: list[str] = []
        for label_id in ids:
            name = id_to_name.get(label_id)
            if name is None:
                raise BuildAbortedError("bridge-incomplete")
            names.append(name)
        raw_criteria = filter_.get("criteria")
        malformed = raw_criteria is not None and not isinstance(raw_criteria, Mapping)
        criteria: Mapping[str, object] = (
            {}
            if raw_criteria is None or not isinstance(raw_criteria, Mapping)
            else cast(Mapping[str, object], raw_criteria)
        )
        for name in names:
            members = members_by_label.get(name, [])
            if not members:
                continue
            if malformed:
                malformed_labels.add(name)
                continue
            verdicts = [simulates_filter(criteria, record) for record in members]
            if None in verdicts:
                unverifiable_labels.add(name)
            else:
                for record, hit in zip(members, verdicts, strict=True):
                    if hit:
                        covered.setdefault(name, set()).add(record.id)

    # manifest-input mode: membership comes from the manifest (no
    # re-split) — records it does not know are excluded, not re-split
    membership: set[str] | None = (
        None if manifest is None else set(manifest.train_ids) | set(manifest.test_ids)
    )

    excluded = unverifiable_labels | malformed_labels
    examples = tuple(
        TrainingExample(
            id=record.id,
            date=record.date,
            labels=tuple(
                sorted(
                    name
                    for name in record.labels
                    if name not in excluded
                    and not (name in covered and record.id in covered[name])
                )
            ),
            weight=example_weight(record.date, cutoff=cutoff, old_weight=old_weight),
            date_provenance=record.date_provenance,
        )
        for record in sorted(records, key=lambda record: record.id)
        if membership is None or record.id in membership
    )
    vocabulary = tuple(
        sorted({name for example in examples for name in example.labels})
    )
    # exclusion observability (NOT in the manifest/hash): wholly-excluded
    # classes with their reason, and the count of records that had labels
    # but came out all-negative (mass-negative targets stay visible)
    excluded_classes = tuple(
        sorted(
            [(label, "unverifiable") for label in unverifiable_labels]
            + [(label, "malformed") for label in malformed_labels]
        )
    )
    original_labels = {record.id: record.labels for record in records}
    emptied_examples = sum(
        1
        for example in examples
        if example.labels == () and original_labels.get(example.id)
    )

    if manifest is None:
        train_ids = tuple(
            sorted(record.id for record in records if record.date < split_date)
        )
        test_ids = tuple(
            sorted(record.id for record in records if record.date >= split_date)
        )
        manifest = SplitManifest(
            format_version=FORMAT_VERSION,
            split_date=split_date,
            vocabulary=vocabulary,
            train_ids=train_ids,
            test_ids=test_ids,
            server_authoritative_only=server_authoritative_only,
            bridge_hash=bridge_hash,
            recent_months=recent_months,
            old_weight=old_weight,
            weight_cutoff=cutoff,
            content_hash="",
        )
        manifest = replace(manifest, content_hash=manifest.compute_hash())
        manifest.validate()  # symmetry: the minted manifest passes its invariants
    else:
        # two-way drift check (no-resplit): the manifest is authoritative,
        # so a rebuild is refused — not re-derived — when the corpus no
        # longer matches the declared split:
        #   (a) a declared train/test member is absent from the
        #       post-trust-filter records (dropped by the trust filter or
        #       gone from the corpus), or
        #   (b) the labels over the member examples != the declared
        #       vocabulary (a new label the manifest does not know, or a
        #       declared label that has no member evidence left).
        # A record whose DATE moved across the split is NOT refused —
        # membership comes from the manifest, dates are not re-evaluated
        # (accepted by design; pinned by test).
        present = {example.id for example in examples}
        declared = set(manifest.train_ids) | set(manifest.test_ids)
        if declared - present:
            raise ValueError(
                "declared split members absent from the (post-trust-filter) "
                f"records: {sorted(declared - present)} — rebuild the manifest"
            )
        labels_over_members = {name for example in examples for name in example.labels}
        if labels_over_members != set(manifest.vocabulary):
            raise ValueError(
                "labels over the member examples do not match the manifest "
                "vocabulary — the corpus drifted, rebuild the manifest"
            )
    return BuildResult(
        manifest=manifest,
        examples=examples,
        excluded_classes=excluded_classes,
        emptied_examples=emptied_examples,
    )


def _need_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"field {key!r} must be a string")
    return value


def _need_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"field {key!r} must be an integer")
    return value


def _need_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, float):
        raise ValueError(f"field {key!r} must be a float")
    return value


def _need_bool(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"field {key!r} must be a boolean")
    return value


def sorted_unique(values: tuple[str, ...]) -> bool:
    return all(a < b for a, b in pairwise(values))


def _need_str_list(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"field {key!r} must be a list of strings")
    items = cast("list[object]", value)  # narrowed by the isinstance above
    if not all(isinstance(v, str) for v in items):
        raise ValueError(f"field {key!r} must be a list of strings")
    return tuple(cast("list[str]", items))  # narrowed by the all() above


def _bridge_hash(
    filters: Sequence[Mapping[str, object]],
    filter_ids: Sequence[list[str]],
    id_to_name: Mapping[str, str],
) -> str:
    """The sha256 of the canonical form of the EFFECTIVE bridge — pinned
    in the manifest so a drifted filters/labels snapshot refuses the
    rebuild loudly (a stricter snapshot silently emptying a
    manifest-declared class is drift, not a valid variant).

    Canonical so it is insensitive to non-semantic variation: filters are
    hashed in sorted order (Gmail's filters.list order is not contractual,
    so an ordinary re-export must not invalidate the manifest), and only
    the id→name entries referenced by a filter action count (renaming an
    unreferenced label, or adding an unreferenced id, is not drift). A
    referenced label RENAME does change the hash — the id→name value is
    part of the effective bridge, so the hash owns that drift class.

    ``filter_ids`` is the already-collected strict id list (one per
    filter) — reused here, not re-parsed (the strict pre-pass already
    validated the shapes, so no re-parse / no latent ValueError path)."""
    canonical_filters = sorted(
        json.dumps(dict(f), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        for f in filters
    )
    referenced: set[str] = set()
    for ids in filter_ids:
        referenced.update(ids)
    canonical_id_to_name = {lid: id_to_name[lid] for lid in sorted(referenced)}
    canonical = json.dumps(
        {"filters": canonical_filters, "id_to_name": canonical_id_to_name},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
