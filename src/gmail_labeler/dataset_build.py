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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, cast

from .corpus_record import SERVER_AUTHORITATIVE_PROVENANCES, DateProvenance
from .labels import label_path, path_is_at_or_under

if TYPE_CHECKING:
    from .corpus_record import CorpusRecord

#: Split Manifest format version (GLOSSARY: Split Manifest). Bump on any
#: change to the manifest's field set or semantics.
FORMAT_VERSION = 1


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
    content_hash: str

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "split_date": self.split_date.isoformat(),
            "vocabulary": list(self.vocabulary),
            "train_ids": list(self.train_ids),
            "test_ids": list(self.test_ids),
            "server_authoritative_only": self.server_authoritative_only,
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
        if not _sorted_unique(self.vocabulary):
            raise ValueError("manifest vocabulary must be sorted and unique")
        if not _sorted_unique(self.train_ids):
            raise ValueError("manifest train_ids must be sorted and unique")
        if not _sorted_unique(self.test_ids):
            raise ValueError("manifest test_ids must be sorted and unique")
        if set(self.train_ids) & set(self.test_ids):
            raise ValueError("manifest train_ids and test_ids must be disjoint")
        if self.content_hash != self.compute_hash():
            raise ValueError("manifest content_hash does not match its fields")


@dataclass(frozen=True, slots=True)
class BuildResult:
    """The build's output: the split manifest + examples sorted by record
    id — a total order, input-order invariant."""

    manifest: SplitManifest
    examples: tuple[TrainingExample, ...]

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


def build_dataset(
    records: Sequence[CorpusRecord],
    *,
    now: datetime,
    split_date: datetime,
    manifest: SplitManifest | None = None,
    server_authoritative_only: bool = False,
) -> BuildResult:
    """Turn CorpusRecords into training examples + a versioned split
    manifest.

    Invariant #7: the split is temporal only — records with
    ``date < split_date`` are train, the rest test. ``manifest`` pins a
    prior split: membership then comes from the manifest (no re-split),
    records it does not know are excluded, and the manifest — split
    date, trust filter — is authoritative over the caller's arguments.
    The manifest's full format invariants are re-validated (an in-memory
    manifest is refused exactly like a parsed one), and a two-way drift
    check refuses the rebuild when a declared member is absent or the
    labels over the members no longer equal the declared vocabulary
    (a record whose date moved across the split is NOT refused —
    membership, not dates, is authoritative).

    ``server_authoritative_only`` drops fallback-sourced records
    (``mime_date``/``unknown`` Date Provenance) out of the split before
    it; the default stamps them on their examples so eval can filter.

    Determinism: examples sorted by id with sorted labels; manifest id
    sets and vocabulary sorted; identical inputs give byte-identical
    manifests (``split_date`` is stored UTC-normalized). ``now`` and
    ``split_date`` are required: the core never reads the clock.
    ``now`` is unused until the recency weighting lands (it anchors the
    cutoff) — required now for the analyze discipline. Duplicate record
    ids are rejected up front (the store guarantees uniqueness).
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if split_date.tzinfo is None:
        raise ValueError("split_date must be timezone-aware")
    split_date = split_date.astimezone(UTC)  # canonical instant in the manifest

    if manifest is not None:
        # the full format validation, not just the version — an in-memory
        # manifest is refused exactly like a parsed one
        manifest.validate()
        # the manifest's parameters are authoritative — same records +
        # manifest → same examples, no re-split, no re-trust-filtering
        server_authoritative_only = manifest.server_authoritative_only

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

    # manifest-input mode: membership comes from the manifest (no
    # re-split) — records it does not know are excluded, not re-split
    membership: set[str] | None = (
        None if manifest is None else set(manifest.train_ids) | set(manifest.test_ids)
    )

    examples = tuple(
        TrainingExample(
            id=record.id,
            date=record.date,
            labels=tuple(sorted(record.labels)),
            weight=1.0,
            date_provenance=record.date_provenance,
        )
        for record in sorted(records, key=lambda record: record.id)
        if membership is None or record.id in membership
    )
    vocabulary = tuple(
        sorted({name for example in examples for name in example.labels})
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
    return BuildResult(manifest=manifest, examples=examples)


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


def _need_bool(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"field {key!r} must be a boolean")
    return value


def _sorted_unique(values: tuple[str, ...]) -> bool:
    return all(a < b for a, b in pairwise(values))


def _need_str_list(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"field {key!r} must be a list of strings")
    items = cast("list[object]", value)  # narrowed by the isinstance above
    if not all(isinstance(v, str) for v in items):
        raise ValueError(f"field {key!r} must be a list of strings")
    return tuple(cast("list[str]", items))  # narrowed by the all() above
