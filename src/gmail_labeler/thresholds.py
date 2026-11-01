"""Per-label threshold calibration (ticket #21; GLOSSARY: Threshold
Band, Bands Artifact).

Pure core (CODING-STANDARDS): deterministic on any input — no clock,
no I/O; sorting is total and stable (the analyze discipline).

The input is the Eval Table (``eval.py``) — the measured per-label
precision/recall at the candidate thresholds on the temporal test
split. Calibration turns it into per-label Threshold Bands (high/med):
"confident" becomes a measured per-label claim, never a global
constant (spec #13).

Band-selection rule (pinned by golden tests):

- ``high`` = the HIGHEST candidate threshold whose measured precision
  is defined (tp+fp > 0) and meets ``min_high_precision`` — ``>=`` is
  the boundary (a cell exactly at the target earns the band);
- ``med`` = the HIGHEST candidate threshold whose measured precision
  meets ``min_med_precision`` AND is strictly below ``high`` when high
  is earned (an empty or inverted med band is meaningless);
  unconstrained when high is not earned;
- a band that is not earned is an explicit ``None`` with a named
  reason — never a silent zero (the GLOSSARY's "none").

Zero-support cells: the eval convention renders a zero denominator as
0.0, but calibration reads tp/fp/fn and derives the ratio itself — a
cell with tp=0 AND fp=0 has UNDEFINED precision (no predictions at
all) and never meets any target, not even 0.0.

The Bands Artifact (versioned JSON + content hash — the
``dataset_build.SplitManifest`` discipline) binds the calibrated bands
to the checkpoint identity (vocabulary hash, corpus fingerprint, split
manifest hash, encoder id — the Checkpoint Sidecar's fields, mirrored)
and the eval identity (split date, test example count). A consumer that
verifies the artifact against a disagreeing checkpoint/eval REFUSES
it (``BandsRefusedError`` — every field checked independently, all
disagreements named; the sidecar doctrine).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import cast

from .dataset_build import is_tenth, sorted_unique
from .eval import EvalTable, ThresholdCell
from .model import vocabulary_hash

__all__ = [
    "BANDS_FORMAT_VERSION",
    "DEFAULT_MIN_HIGH_PRECISION",
    "DEFAULT_MIN_MED_PRECISION",
    "BandTargets",
    "Bands",
    "BandsArtifact",
    "BandsLabelEntry",
    "BandsRefusedError",
    "LabelBands",
    "build_artifact",
    "calibrate_bands",
    "verify_bands",
]

#: The default min-precision targets (the config defaults mirror these
#: — the equality is pinned by test_config, not by an import: config.py
#: stays import-light). high is the silent-application tier (Phase 2
#: exit: high-band labels apply silently) — strict; med is the
#: review-me tier.
DEFAULT_MIN_HIGH_PRECISION = 0.9
DEFAULT_MIN_MED_PRECISION = 0.7

#: Bands Artifact format version (GLOSSARY: Bands Artifact). Bump on any
#: change to the artifact's field set or semantics.
BANDS_FORMAT_VERSION = 1

# Hash-shaped identity fields (the CorpusRecord._HASH_RE / model._HASH_RE
# precedent): an empty or malformed hash is refused at parse, not
# "verified" against an equally empty expectation.
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

# The explicit no-band reasons (pinned by golden tests — the no-band
# state is observable, never a silent zero, and a reason must never
# LIE: "meets" is reserved for the target itself, the strict-below-
# high constraint gets its own name):
_REASON_HIGH = "no-candidate-meets-high-target"
_REASON_MED = "no-candidate-meets-med-target"
_REASON_MED_BELOW_HIGH = "no-candidate-below-high-meets-med-target"


@dataclass(frozen=True, slots=True)
class BandTargets:
    """The per-label min-precision targets a calibration run uses: the
    floor a candidate column's measured precision must meet to earn the
    band. Both are tenths (the tenth-grid stance, #20)."""

    min_high_precision: float
    min_med_precision: float


@dataclass(frozen=True, slots=True)
class LabelBands:
    """One label's calibrated Threshold Bands: ``high``/``med`` are
    candidate thresholds, or ``None`` for the explicit no-band state
    (the GLOSSARY's "none"); ``reasons`` names every missing band."""

    label: str
    high: float | None
    med: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Bands:
    """The calibration result: per-label bands in vocabulary order
    (a total order), the candidate grid they were chosen from, and the
    per-label EFFECTIVE targets (defaults + overrides — provenance)."""

    vocabulary: tuple[str, ...]
    candidate_thresholds: tuple[float, ...]
    labels: tuple[LabelBands, ...]
    targets: tuple[tuple[str, BandTargets], ...]


@dataclass(frozen=True, slots=True)
class BandsLabelEntry:
    """One label's artifact entry: the bands + the effective targets
    the run used for it (an override is visible per label)."""

    label: str
    high: float | None
    med: float | None
    min_high_precision: float
    min_med_precision: float
    reasons: tuple[str, ...]


def _check_target(name: str, value: float) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float in [0.0, 1.0], got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")
    if not is_tenth(value):
        raise ValueError(
            f"{name} {value} is not a tenth — the bands ride the "
            "candidate grid, so a non-tenth target would render "
            "mislabeled (the eval stance, #20)"
        )


def _check_thresholds(thresholds: tuple[float, ...]) -> None:
    if not thresholds:
        raise ValueError("candidate thresholds must not be empty")
    for t in thresholds:
        if not 0.0 < t < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {t}")
        if not is_tenth(t):
            raise ValueError(
                f"threshold {t} is not a tenth — the bands are chosen "
                "from the candidate columns, which the report labels to "
                "one decimal (the eval stance, #20)"
            )
    for a, b in pairwise(thresholds):
        if a >= b:
            raise ValueError(f"thresholds must be strictly ascending, got {a} then {b}")


def _precision(cell: ThresholdCell) -> float | None:
    """The cell's measured precision, read from tp/fp/fn — NOT the
    table's rendered ratio: a zero denominator (tp=0 AND fp=0, no
    predictions at all) is UNDEFINED, never 0.0. The eval convention
    renders it 0.0, but calibration must not treat "no predictions" as
    a measured zero (it would meet a 0.0 target)."""
    if cell.tp + cell.fp == 0:
        return None
    return cell.tp / (cell.tp + cell.fp)


def calibrate_bands(
    table: EvalTable,
    *,
    min_high_precision: float = DEFAULT_MIN_HIGH_PRECISION,
    min_med_precision: float = DEFAULT_MIN_MED_PRECISION,
    overrides: Mapping[str, BandTargets] | None = None,
) -> Bands:
    """Turn the measured Eval Table into per-label Threshold Bands.

    Refusals (fail-loud doctrine): an empty/non-canonical vocabulary, a
    candidate grid that is empty, out of (0, 1), not strictly
    ascending, or not a grid of tenths (inherited from the eval), rows
    that do not align to the vocabulary in vocabulary order, cells that
    do not align to the grid, a target outside [0, 1] (a bool is not a
    target), a non-tenth target — the run's AND every override's — and
    an override for a label the table does not carry (named) all raise
    ValueError — the bands are a measured claim, never a guess.
    """
    if not table.vocabulary:
        raise ValueError("vocabulary must not be empty")
    if not sorted_unique(table.vocabulary):
        raise ValueError("vocabulary must be sorted and unique (canonical form)")
    _check_thresholds(table.thresholds)
    if len(table.rows) != len(table.vocabulary):
        raise ValueError(
            f"rows must align to the vocabulary: {len(table.rows)} rows "
            f"for {len(table.vocabulary)} labels"
        )
    for row, label in zip(table.rows, table.vocabulary, strict=True):
        if row.label != label:
            raise ValueError(
                f"rows must be in vocabulary order, got {row.label!r} before {label!r}"
            )
        if len(row.cells) != len(table.thresholds):
            raise ValueError(
                f"cells for {row.label!r} do not align to the candidate "
                f"thresholds: {len(row.cells)} cells for "
                f"{len(table.thresholds)} thresholds"
            )
    _check_target("min_high_precision", min_high_precision)
    _check_target("min_med_precision", min_med_precision)
    unknown = sorted(set(overrides or ()) - set(table.vocabulary))
    if unknown:
        raise ValueError(
            f"override(s) for label(s) {unknown} not in the vocabulary — "
            "refusing to calibrate silently (fix the override)"
        )
    # the override targets get the SAME validation as the run's — an
    # out-of-range or non-tenth per-label target would miscalibrate
    # silently (the config layer shields the CLI path; the core is the
    # public boundary #24/#25 consume)
    override_map: Mapping[str, BandTargets] = overrides if overrides is not None else {}
    for label, override in sorted(override_map.items()):
        _check_target(
            f"overrides[{label!r}].min_high_precision", override.min_high_precision
        )
        _check_target(
            f"overrides[{label!r}].min_med_precision", override.min_med_precision
        )

    labels: list[LabelBands] = []
    targets: list[tuple[str, BandTargets]] = []
    for row in table.rows:
        effective = override_map.get(
            row.label,
            BandTargets(
                min_high_precision=min_high_precision,
                min_med_precision=min_med_precision,
            ),
        )
        # two passes, each keeping the LAST (highest) qualifier — the
        # grid is strictly ascending, so "highest" is unique and the
        # selection is a total order
        high: float | None = None
        for t, cell in zip(table.thresholds, row.cells, strict=True):
            p = _precision(cell)
            if p is not None and p >= effective.min_high_precision:
                high = t
        med: float | None = None
        for t, cell in zip(table.thresholds, row.cells, strict=True):
            p = _precision(cell)
            if (
                p is not None
                and p >= effective.min_med_precision
                and (high is None or t < high)
            ):
                med = t
        reasons: list[str] = []
        if high is None:
            reasons.append(_REASON_HIGH)
        if med is None:
            # the reason must not LIE: "meets" is reserved for the
            # target itself — when high is earned, a missing med means
            # nothing BELOW high met the target (its own name)
            reasons.append(_REASON_MED if high is None else _REASON_MED_BELOW_HIGH)
        labels.append(
            LabelBands(label=row.label, high=high, med=med, reasons=tuple(reasons))
        )
        targets.append((row.label, effective))
    return Bands(
        vocabulary=table.vocabulary,
        candidate_thresholds=table.thresholds,
        labels=tuple(labels),
        targets=tuple(targets),
    )


@dataclass(frozen=True, slots=True)
class BandsArtifact:
    """The versioned, hashable Bands Artifact (GLOSSARY: Bands
    Artifact) — the calibrated per-label Threshold Bands bound to the
    identity they were measured against. ``content_hash`` is the sha256
    of the canonical JSON of every other field — a machine-checkable
    identity; an artifact whose hash does not match its fields is stale
    or corrupt and is refused, never served (the SplitManifest rule).

    Identity fields: the Checkpoint Sidecar's four (vocabulary_hash,
    corpus_fingerprint, split_manifest_hash, encoder_id — mirrored, so
    a sidecar disagreement refuses the artifact, AC2) plus the eval
    identity (split_date, test_examples — the measured window).
    Provenance: the candidate grid and the default targets the run
    used; each label entry carries its effective targets (an override
    is visible)."""

    format_version: int
    vocabulary_hash: str
    corpus_fingerprint: str
    split_manifest_hash: str
    encoder_id: str
    split_date: datetime
    test_examples: int
    candidate_thresholds: tuple[float, ...]
    default_targets: BandTargets
    labels: tuple[BandsLabelEntry, ...]
    content_hash: str

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "vocabulary_hash": self.vocabulary_hash,
            "corpus_fingerprint": self.corpus_fingerprint,
            "split_manifest_hash": self.split_manifest_hash,
            "encoder_id": self.encoder_id,
            "split_date": self.split_date.isoformat(),
            "test_examples": self.test_examples,
            "candidate_thresholds": list(self.candidate_thresholds),
            "default_targets": {
                "min_high_precision": self.default_targets.min_high_precision,
                "min_med_precision": self.default_targets.min_med_precision,
            },
            "labels": [
                {
                    "label": entry.label,
                    "high": entry.high,
                    "med": entry.med,
                    "min_high_precision": entry.min_high_precision,
                    "min_med_precision": entry.min_med_precision,
                    "reasons": list(entry.reasons),
                }
                for entry in self.labels
            ],
        }

    def compute_hash(self) -> str:
        """The sha256 of the canonical JSON of every field except
        ``content_hash`` itself — the identity ``from_json``/
        ``validate`` verify."""
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
    def from_json(cls, text: str) -> BandsArtifact:
        """Parse + verify: wrong shape, an unknown format version, a
        naive date, a violated format invariant, or a content_hash that
        does not match the fields raises ValueError — the artifact is a
        machine-checkable identity; a hash-consistent artifact that
        violates the invariants is corrupt, not a valid variant."""
        data: object = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("bands artifact is not an object")
        payload = cast("dict[str, object]", data)
        try:
            artifact = cls(
                format_version=_need_int(payload, "format_version"),
                vocabulary_hash=_need_hash(payload, "vocabulary_hash"),
                corpus_fingerprint=_need_hash(payload, "corpus_fingerprint"),
                split_manifest_hash=_need_hash(payload, "split_manifest_hash"),
                encoder_id=_need_str(payload, "encoder_id"),
                split_date=datetime.fromisoformat(_need_str(payload, "split_date")),
                test_examples=_need_int(payload, "test_examples"),
                candidate_thresholds=_need_float_list(payload, "candidate_thresholds"),
                default_targets=BandTargets(
                    min_high_precision=_need_nested_float(
                        payload, "default_targets", "min_high_precision"
                    ),
                    min_med_precision=_need_nested_float(
                        payload, "default_targets", "min_med_precision"
                    ),
                ),
                labels=_need_label_entries(payload),
                content_hash=_need_str(payload, "content_hash"),
            )
        except ValueError as e:
            raise ValueError(f"bands field invalid: {e}") from e
        artifact.validate()
        return artifact

    def validate(self) -> None:
        """Every format invariant, called from BOTH entry paths —
        ``from_json`` and ``build_artifact`` — so an artifact object
        passed directly is refused exactly like a parsed one: stale
        hash, unknown version, naive dates, violated invariants."""
        if self.format_version != BANDS_FORMAT_VERSION:
            raise ValueError(
                f"bands format_version {self.format_version} != {BANDS_FORMAT_VERSION}"
            )
        for name in (
            "vocabulary_hash",
            "corpus_fingerprint",
            "split_manifest_hash",
        ):
            if not _HASH_RE.fullmatch(getattr(self, name)):
                raise ValueError(
                    f"bands {name} must be a 64-char hex hash, got "
                    f"{getattr(self, name)!r}"
                )
        # UTC-normalized: a non-UTC offset would make the same instant
        # hash differently on a tz-offset host (the manifest rule)
        if self.split_date.tzinfo is None or self.split_date.utcoffset() != timedelta(
            0
        ):
            raise ValueError("bands split_date must be timezone-aware UTC")
        if isinstance(self.test_examples, bool) or self.test_examples < 0:
            raise ValueError(
                "bands test_examples must be a non-negative integer, got "
                f"{self.test_examples!r}"
            )
        _check_thresholds(self.candidate_thresholds)
        _check_target(
            "default_targets.min_high_precision",
            self.default_targets.min_high_precision,
        )
        _check_target(
            "default_targets.min_med_precision",
            self.default_targets.min_med_precision,
        )
        if not self.labels:
            raise ValueError("bands labels must not be empty")
        if not sorted_unique(tuple(entry.label for entry in self.labels)):
            raise ValueError("bands labels must be sorted and unique")
        grid = set(self.candidate_thresholds)
        for entry in self.labels:
            _check_target(f"{entry.label}.min_high_precision", entry.min_high_precision)
            _check_target(f"{entry.label}.min_med_precision", entry.min_med_precision)
            if entry.high is not None and entry.high not in grid:
                raise ValueError(
                    f"bands for {entry.label!r}: high {entry.high} is not "
                    "a candidate threshold"
                )
            if entry.med is not None and entry.med not in grid:
                raise ValueError(
                    f"bands for {entry.label!r}: med {entry.med} is not "
                    "a candidate threshold"
                )
            if (
                entry.high is not None
                and entry.med is not None
                and entry.med >= entry.high
            ):
                raise ValueError(
                    f"bands for {entry.label!r}: med {entry.med} must be "
                    f"strictly below high {entry.high}"
                )
            # the reasons must match the bands (a machine-checked
            # artifact may not contradict itself): each missing band
            # names itself, and no earned band may carry a reason
            expected: set[str] = set()
            if entry.high is None:
                expected.add(_REASON_HIGH)
            if entry.med is None:
                expected.add(
                    _REASON_MED if entry.high is None else _REASON_MED_BELOW_HIGH
                )
            # tuple comparison (not set): a duplicated reason would
            # pass a set equality — the machine-checked boundary must
            # not accept a lying count
            if tuple(sorted(entry.reasons)) != tuple(sorted(expected)):
                raise ValueError(
                    f"bands for {entry.label!r}: reasons {entry.reasons} "
                    f"do not match the bands (expected {sorted(expected)})"
                )
        if self.content_hash != self.compute_hash():
            raise ValueError("bands content_hash does not match its fields")


class BandsRefusedError(ValueError):
    """A consumer refuses to use a Bands Artifact: it disagrees with
    the checkpoint/eval it is served against. ``fields`` names every
    disagreeing identity field (each field is checked INDEPENDENTLY —
    a single-field drift is refused on its own; the sidecar doctrine).
    Subclasses ``ValueError`` so a CLI mapping can catch the whole
    refusal family with one ``except ValueError``."""

    def __init__(self, fields: Sequence[str], detail: str) -> None:
        super().__init__(f"bands artifact refused ({', '.join(fields)}): {detail}")
        self.fields = tuple(fields)


def build_artifact(
    bands: Bands,
    *,
    vocabulary: tuple[str, ...],
    corpus_fingerprint: str,
    split_manifest_hash: str,
    encoder_id: str,
    split_date: datetime,
    test_examples: int,
    default_targets: BandTargets,
) -> BandsArtifact:
    """Mint the Bands Artifact from a calibration run: the bands + the
    checkpoint identity (the sidecar's fields) + the eval identity
    (the measured window) + the run's default targets. Refusals: a
    vocabulary that does not match the bands' (a lying identity) and a
    naive split_date raise ValueError. The minted artifact passes its
    own invariants (the manifest symmetry)."""
    if tuple(vocabulary) != bands.vocabulary:
        raise ValueError(
            f"the artifact's vocabulary {tuple(vocabulary)} does not "
            f"match the bands' vocabulary {bands.vocabulary} — refusing "
            "to mint a lying identity"
        )
    if split_date.tzinfo is None:
        raise ValueError("split_date must be timezone-aware")
    split_date = split_date.astimezone(UTC)  # canonical instant in the artifact
    entries = tuple(
        BandsLabelEntry(
            label=label_bands.label,
            high=label_bands.high,
            med=label_bands.med,
            min_high_precision=effective.min_high_precision,
            min_med_precision=effective.min_med_precision,
            reasons=label_bands.reasons,
        )
        for label_bands, (_label, effective) in zip(
            bands.labels, bands.targets, strict=True
        )
    )
    artifact = BandsArtifact(
        format_version=BANDS_FORMAT_VERSION,
        vocabulary_hash=vocabulary_hash(vocabulary),
        corpus_fingerprint=corpus_fingerprint,
        split_manifest_hash=split_manifest_hash,
        encoder_id=encoder_id,
        split_date=split_date,
        test_examples=test_examples,
        candidate_thresholds=bands.candidate_thresholds,
        default_targets=default_targets,
        labels=entries,
        content_hash="",
    )
    artifact = replace(artifact, content_hash=artifact.compute_hash())
    artifact.validate()  # symmetry: the minted artifact passes its invariants
    return artifact


def verify_bands(
    artifact: BandsArtifact,
    *,
    vocabulary: tuple[str, ...],
    corpus_fingerprint: str,
    split_manifest_hash: str,
    encoder_id: str,
    split_date: datetime,
    test_examples: int,
) -> None:
    """Every identity field, checked INDEPENDENTLY; ALL disagreements
    are collected and reported in one ``BandsRefusedError`` (a
    single-field drift is refused on its own — no cross-field
    coupling). The vocabulary check compares the artifact's hash
    against the hash of the expected vocabulary (the manifest is the
    authority, the artifact must agree with it — the sidecar rule)."""
    mismatches: list[str] = []
    if artifact.vocabulary_hash != vocabulary_hash(vocabulary):
        mismatches.append("vocabulary")
    if artifact.corpus_fingerprint != corpus_fingerprint:
        mismatches.append("corpus_fingerprint")
    if artifact.split_manifest_hash != split_manifest_hash:
        mismatches.append("split_manifest_hash")
    if artifact.encoder_id != encoder_id:
        mismatches.append("encoder_id")
    if artifact.split_date != split_date:
        mismatches.append("split_date")
    if artifact.test_examples != test_examples:
        mismatches.append("test_examples")
    if mismatches:
        raise BandsRefusedError(
            mismatches,
            "the artifact disagrees with the checkpoint/eval it is "
            "served against — re-run `eval --bands` against this "
            "checkpoint",
        )


def _need_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"field {key!r} must be a string")
    return value


def _need_hash(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"field {key!r} must be a 64-char hex hash")
    return value


def _need_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"field {key!r} must be an integer")
    return value


def _need_float_list(payload: dict[str, object], key: str) -> tuple[float, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"field {key!r} must be a list of numbers")
    items = cast("list[object]", value)
    numbers: list[float] = []
    for v in items:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"field {key!r} must be a list of numbers")
        numbers.append(float(v))
    return tuple(numbers)


def _need_nested_float(payload: dict[str, object], table: str, key: str) -> float:
    value = payload.get(table)
    if not isinstance(value, dict):
        raise ValueError(f"field {table!r} must be an object")
    nested = cast("dict[str, object]", value)
    item = nested.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"field {table}.{key} must be a number")
    return float(item)


def _need_label_entries(payload: dict[str, object]) -> tuple[BandsLabelEntry, ...]:
    value = payload.get("labels")
    if not isinstance(value, list):
        raise ValueError("field 'labels' must be a list of objects")
    items = cast("list[object]", value)
    entries: list[BandsLabelEntry] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"labels[{i}] must be an object")
        entry = cast("dict[str, object]", item)
        try:
            entries.append(
                BandsLabelEntry(
                    label=_need_str(entry, "label"),
                    high=_need_band(entry, "high"),
                    med=_need_band(entry, "med"),
                    min_high_precision=_need_number(entry, "min_high_precision"),
                    min_med_precision=_need_number(entry, "min_med_precision"),
                    reasons=_need_str_list(entry, "reasons"),
                )
            )
        except ValueError as e:
            raise ValueError(f"labels[{i}]: {e}") from e
    return tuple(entries)


def _need_band(entry: dict[str, object], key: str) -> float | None:
    value = entry.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"field {key!r} must be a number or null")
    return float(value)


def _need_number(entry: dict[str, object], key: str) -> float:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"field {key!r} must be a number")
    return float(value)


def _need_str_list(entry: dict[str, object], key: str) -> tuple[str, ...]:
    value = entry.get(key)
    if not isinstance(value, list):
        raise ValueError(f"field {key!r} must be a list of strings")
    items = cast("list[object]", value)
    if not all(isinstance(v, str) for v in items):
        raise ValueError(f"field {key!r} must be a list of strings")
    return tuple(cast("list[str]", items))
