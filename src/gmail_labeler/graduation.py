"""Per-label Graduation gates (ticket #22; GLOSSARY: Graduation,
Graduation Table).

Pure core (CODING-STANDARDS): deterministic on any input — no clock,
no I/O; sorting is total and stable (the analyze discipline).

The input is the Eval Table (``eval.py``) — the measured per-label
support + precision at the candidate thresholds on the temporal test
split — plus the calibrated Bands Artifact (``thresholds.py``) the
gates are applied against: the precision gate reads the measured
precision AT THE LABEL'S CALIBRATED HIGH BAND THRESHOLD, and a label
whose band has no high threshold cannot graduate (the med band never
graduates anything by itself).

Gate-selection rule (pinned by golden tests — the operators MIRROR
``analyze.py``'s volume/recency gates exactly, the strict-gt
precedent):

- the examples gate is ``examples < min_examples`` → fail: a value
  EXACTLY at the gate PASSES (analyze's ``total < min_total``);
- the precision gate is ``precision_at_high < min_precision`` → fail:
  a value EXACTLY at the gate PASSES (analyze's ``dup_rate >
  max_dup_rate`` — the strict-gt side of the precedent);
- an unearned label carries NAMED reasons (``below-min-examples``,
  ``precision-below-gate``, ``no-high-band``) — never a silent zero,
  and a reason must never LIE (the no-high-band state is observable).

Gate targets are NOT on the tenth grid (documented asymmetry with
``[bands]``): a band target must match the eval's candidate grid
(thresholds are LOOKED UP in it), while a gate precision target is
COMPARED against measured values — measured precision is any rational
(tp/(tp+fp)), so a non-tenth target like 0.91 is a legitimate gate.
Range + bool refusals DO apply (the config doctrine).

The Graduation Table (versioned JSON + content hash — the
``dataset_build.SplitManifest`` discipline) binds the gate results to
the identity they were measured against: the six eval identity fields
(vocabulary hash, corpus fingerprint, split manifest hash, encoder id,
split date, test example count — the Bands Artifact's fields, mirrored)
PLUS the ``bands_content_hash`` of the Bands Artifact the gates were
computed from (AC3 — a consumer that verifies the table against a
disagreeing checkpoint/eval/bands REFUSES it, ``GraduationRefusedError``
— every field checked independently, all disagreements named; the
sidecar doctrine).

Trust boundary (stated, per the SplitManifest discipline): the
measured values (``examples``, ``precision_at_high``) are trusted from
the Eval Table + Bands Artifact at mint time. The content hash is a
CORRUPTION CHECK, not a signature (an unkeyed sha256 over the fields —
it catches torn/stale bytes, not a forger), and ``verify_graduation``
binds the table to the checkpoint/eval/bands identity it claims to be
from — it does NOT authenticate the measured values: a hand-forged
table with the correct identity verifies cleanly, so a consumer that
needs the measurement must recompute it from the Eval Table (the eval
table is the measurement authority; the bands/table cross-check in
``evaluate_graduation`` refuses — for each Managed Label with a high
band — a bands artifact whose high band has no measured predictions on
the table). The Bands Artifact itself is
computed IN-MEMORY per run (deterministically — the same eval + config
recomputes the same artifact), so the table's ``bands_content_hash``
binding is reproducible even when the bands file was never written
(``eval --graduation`` without ``--bands``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import cast

from .dataset_build import is_tenth, sorted_unique
from .eval import EvalTable
from .model import vocabulary_hash
from .thresholds import BandsArtifact

__all__ = [
    "DEFAULT_MIN_EXAMPLES",
    "DEFAULT_MIN_PRECISION",
    "GRADUATION_FORMAT_VERSION",
    "IDENTITY_FIELDS",
    "GateTargets",
    "Graduation",
    "GraduationEntry",
    "GraduationRefusedError",
    "GraduationTable",
    "build_table",
    "evaluate_graduation",
    "identity_disagreement",
    "render_graduation_diff",
    "verify_graduation",
]

#: The identity fields a consumer compares between two tables (the six
#: eval fields + the bands binding) — named in ONE place so a future
#: field addition gates every comparison site (the CLI's diff warning
#: and the write-path replace gate both read this).
IDENTITY_FIELDS: tuple[str, ...] = (
    "vocabulary_hash",
    "corpus_fingerprint",
    "split_manifest_hash",
    "encoder_id",
    "split_date",
    "test_examples",
    "bands_content_hash",
)

# The keys of verify_graduation's expected-identity mapping (one home
# for the comparison keys). A field added to IDENTITY_FIELDS without
# an expected-value builder (or vice versa) would be a KeyError/exit-1
# at a CONSUMER, not a caught refusal — so the drift is refused at
# import time, crash-by-design:
_EXPECTED_IDENTITY_KEYS: frozenset[str] = frozenset(
    {
        "vocabulary_hash",
        "corpus_fingerprint",
        "split_manifest_hash",
        "encoder_id",
        "split_date",
        "test_examples",
        "bands_content_hash",
    }
)
assert set(IDENTITY_FIELDS) == _EXPECTED_IDENTITY_KEYS, (
    "IDENTITY_FIELDS drifted from verify_graduation's expected-identity "
    "keys — add the field to BOTH (the expected-value builder included)"
)

#: The default gate targets (the config defaults mirror these — the
#: equality is pinned by test_config, not by an import: config.py stays
#: import-light).
DEFAULT_MIN_EXAMPLES = 10
DEFAULT_MIN_PRECISION = 0.9

#: Graduation Table format version (GLOSSARY: Graduation Table). Bump
#: on any change to the artifact's field set or semantics.
GRADUATION_FORMAT_VERSION = 1

# Hash-shaped identity fields (the CorpusRecord._HASH_RE / model._HASH_RE
# precedent): an empty or malformed hash is refused at parse, not
# "verified" against an equally empty expectation.
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

# The explicit no-graduation reasons (pinned by golden tests — the
# no-graduation state is observable, never a silent zero, and a reason
# must never LIE):
_REASON_BELOW_MIN_EXAMPLES = "below-min-examples"
_REASON_PRECISION_BELOW_GATE = "precision-below-gate"
_REASON_NO_HIGH_BAND = "no-high-band"


@dataclass(frozen=True, slots=True)
class GateTargets:
    """The per-label gate targets a graduation run uses: the floor the
    measured support and the measured precision at the calibrated high
    band must meet to graduate. ``min_precision`` is NOT tenth-grid
    (see the module docstring — the documented asymmetry with [bands])."""

    min_examples: int
    min_precision: float


@dataclass(frozen=True, slots=True)
class GraduationEntry:
    """One label's gate result: the measured support, the calibrated
    bands (high/med — the med is provenance, it never graduates), the
    measured precision at the high band (``None`` iff the high band is
    ``None``), the EFFECTIVE gate targets the run used for it (an
    override is visible per label), the verdict, and the named reasons
    for an unearned label (empty iff graduated)."""

    label: str
    examples: int
    precision_at_high: float | None
    high: float | None
    med: float | None
    min_examples: int
    min_precision: float
    graduated: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Graduation:
    """The gate evaluation result: per-label entries in label-sorted
    order (a total order), the candidate grid the bands ride, the
    per-label EFFECTIVE targets (defaults + overrides — provenance),
    and the measured window's provenance (test_examples, the run's
    default targets, the Bands Artifact's content hash — DERIVED from
    the evaluation inputs and carried here so build_table cannot be
    told a lying identity)."""

    vocabulary: tuple[str, ...]
    candidate_thresholds: tuple[float, ...]
    labels: tuple[GraduationEntry, ...]
    targets: tuple[tuple[str, GateTargets], ...]
    test_examples: int
    default_targets: GateTargets
    bands_content_hash: str


@dataclass(frozen=True, slots=True)
class GraduationTable:
    """The versioned, hashable Graduation Table (GLOSSARY: Graduation
    Table) — the per-label gate results bound to the identity they were
    measured against. ``content_hash`` is the sha256 of the canonical
    JSON of every other field — a machine-checkable identity; a table
    whose hash does not match its fields is stale or corrupt and is
    refused, never served (the SplitManifest rule).

    Identity fields: the Bands Artifact's six (vocabulary_hash,
    corpus_fingerprint, split_manifest_hash, encoder_id, split_date,
    test_examples — mirrored, so a sidecar disagreement refuses the
    table, AC3) plus ``bands_content_hash`` — the identity of the
    Bands Artifact the gates were computed from (the thresholds
    artifact's binding). Provenance: the candidate grid and the default
    targets the run used; each label entry carries its effective gate
    targets (an override is visible)."""

    format_version: int
    vocabulary_hash: str
    corpus_fingerprint: str
    split_manifest_hash: str
    encoder_id: str
    split_date: datetime
    test_examples: int
    bands_content_hash: str
    candidate_thresholds: tuple[float, ...]
    default_targets: GateTargets
    labels: tuple[GraduationEntry, ...]
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
            "bands_content_hash": self.bands_content_hash,
            "candidate_thresholds": list(self.candidate_thresholds),
            "default_targets": {
                "min_examples": self.default_targets.min_examples,
                "min_precision": self.default_targets.min_precision,
            },
            "labels": [
                {
                    "label": entry.label,
                    "examples": entry.examples,
                    "precision_at_high": entry.precision_at_high,
                    "high": entry.high,
                    "med": entry.med,
                    "min_examples": entry.min_examples,
                    "min_precision": entry.min_precision,
                    "graduated": entry.graduated,
                    # CANONICAL order (sorted): validate() compares
                    # reasons sorted — the hash must serialize the same
                    # meaning the same way, or one meaning would have
                    # two identities (a reversed-reasons table would
                    # validate with a different hash)
                    "reasons": sorted(entry.reasons),
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
    def from_json(cls, text: str) -> GraduationTable:
        """Parse + verify: wrong shape, an unknown format version, a
        naive date, a violated format invariant, or a content_hash that
        does not match the fields raises ValueError — the table is a
        machine-checkable identity; a hash-consistent table that
        violates the invariants is corrupt, not a valid variant."""
        data: object = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("graduation table is not an object")
        payload = cast("dict[str, object]", data)
        try:
            table = cls(
                format_version=_need_int(payload, "format_version"),
                vocabulary_hash=_need_hash(payload, "vocabulary_hash"),
                corpus_fingerprint=_need_hash(payload, "corpus_fingerprint"),
                split_manifest_hash=_need_hash(payload, "split_manifest_hash"),
                encoder_id=_need_str(payload, "encoder_id"),
                split_date=datetime.fromisoformat(_need_str(payload, "split_date")),
                test_examples=_need_int(payload, "test_examples"),
                bands_content_hash=_need_hash(payload, "bands_content_hash"),
                candidate_thresholds=_need_float_list(payload, "candidate_thresholds"),
                default_targets=GateTargets(
                    min_examples=_need_nested_int(
                        payload, "default_targets", "min_examples"
                    ),
                    min_precision=_need_nested_float(
                        payload, "default_targets", "min_precision"
                    ),
                ),
                labels=_need_label_entries(payload),
                content_hash=_need_str(payload, "content_hash"),
            )
        except ValueError as e:
            raise ValueError(f"graduation field invalid: {e}") from e
        table.validate()
        return table

    def validate(self) -> None:
        """Every format invariant, called from BOTH entry paths —
        ``from_json`` and ``build_table`` — so a table object passed
        directly is refused exactly like a parsed one: stale hash,
        unknown version, naive dates, violated invariants."""
        if self.format_version != GRADUATION_FORMAT_VERSION:
            raise ValueError(
                f"graduation format_version {self.format_version} "
                f"!= {GRADUATION_FORMAT_VERSION}"
            )
        for name in (
            "vocabulary_hash",
            "corpus_fingerprint",
            "split_manifest_hash",
            "bands_content_hash",
        ):
            if not _HASH_RE.fullmatch(getattr(self, name)):
                raise ValueError(
                    f"graduation {name} must be a 64-char hex hash, got "
                    f"{getattr(self, name)!r}"
                )
        # UTC-normalized: a non-UTC offset would make the same instant
        # hash differently on a tz-offset host (the manifest rule)
        if self.split_date.tzinfo is None or self.split_date.utcoffset() != timedelta(
            0
        ):
            raise ValueError("graduation split_date must be timezone-aware UTC")
        if isinstance(self.test_examples, bool) or self.test_examples < 0:
            raise ValueError(
                "graduation test_examples must be a non-negative integer, got "
                f"{self.test_examples!r}"
            )
        if not self.encoder_id.strip():
            # an empty/blank encoder_id would verify against an
            # equally empty expectation — refuse at parse/validate,
            # not "verify" (the hash-field doctrine)
            raise ValueError("graduation encoder_id must be non-empty")
        _check_thresholds(self.candidate_thresholds)
        _check_min_examples(
            "default_targets.min_examples", self.default_targets.min_examples
        )
        _check_min_precision(
            "default_targets.min_precision", self.default_targets.min_precision
        )
        if not self.labels:
            raise ValueError("graduation labels must not be empty")
        if not sorted_unique(tuple(entry.label for entry in self.labels)):
            raise ValueError("graduation labels must be sorted and unique")
        grid = set(self.candidate_thresholds)
        for entry in self.labels:
            if not entry.label.strip():
                raise ValueError("graduation label keys must be non-empty")
            # the two-entry-path symmetry: from_json refuses a negative
            # examples count, so validate() must too (a directly
            # constructed table is refused exactly like a parsed one)
            if isinstance(entry.examples, bool) or entry.examples < 0:
                raise ValueError(
                    f"graduation for {entry.label!r}: examples must be a "
                    f"non-negative integer, got {entry.examples!r}"
                )
            _check_min_examples(f"{entry.label}.min_examples", entry.min_examples)
            _check_min_precision(f"{entry.label}.min_precision", entry.min_precision)
            if entry.high is not None and entry.high not in grid:
                raise ValueError(
                    f"graduation for {entry.label!r}: high {entry.high} is "
                    "not a candidate threshold"
                )
            if entry.med is not None and entry.med not in grid:
                raise ValueError(
                    f"graduation for {entry.label!r}: med {entry.med} is "
                    "not a candidate threshold"
                )
            if (
                entry.high is not None
                and entry.med is not None
                and entry.med >= entry.high
            ):
                raise ValueError(
                    f"graduation for {entry.label!r}: med {entry.med} must "
                    f"be strictly below high {entry.high}"
                )
            # precision_at_high tracks the high band EXACTLY: None iff
            # the high band is None (a measured precision with no band,
            # or a band with no measurement, is self-contradictory)
            if (entry.precision_at_high is None) != (entry.high is None):
                raise ValueError(
                    f"graduation for {entry.label!r}: precision_at_high "
                    f"{entry.precision_at_high!r} does not track the high "
                    f"band {entry.high!r} (None iff the band is None)"
                )
            if entry.precision_at_high is not None and not (
                0.0 <= entry.precision_at_high <= 1.0
            ):
                raise ValueError(
                    f"graduation for {entry.label!r}: precision_at_high "
                    f"{entry.precision_at_high} is out of [0.0, 1.0]"
                )
            # the reasons must match the gates (a machine-checked
            # artifact may not contradict itself): each unearned gate
            # names itself, and a graduated label may carry no reason
            expected: set[str] = set()
            if entry.examples < entry.min_examples:
                expected.add(_REASON_BELOW_MIN_EXAMPLES)
            if entry.high is None:
                expected.add(_REASON_NO_HIGH_BAND)
            elif (
                entry.precision_at_high is not None
                and entry.precision_at_high < entry.min_precision
            ):
                expected.add(_REASON_PRECISION_BELOW_GATE)
            if entry.graduated != (not expected):
                raise ValueError(
                    f"graduation for {entry.label!r}: graduated="
                    f"{entry.graduated} does not match the gates "
                    f"(expected reasons {sorted(expected)})"
                )
            # tuple comparison (not set): a duplicated reason would
            # pass a set equality — the machine-checked boundary must
            # not accept a lying count
            if tuple(sorted(entry.reasons)) != tuple(sorted(expected)):
                raise ValueError(
                    f"graduation for {entry.label!r}: reasons {entry.reasons} "
                    f"do not match the gates (expected {sorted(expected)})"
                )
        if self.content_hash != self.compute_hash():
            raise ValueError("graduation content_hash does not match its fields")


class GraduationRefusedError(ValueError):
    """A consumer refuses to use a Graduation Table: it disagrees with
    the checkpoint/eval/bands it is served against. ``fields`` names
    every disagreeing identity field (each field is checked
    INDEPENDENTLY — a single-field drift is refused on its own; the
    sidecar doctrine). Subclasses ``ValueError`` so a CLI mapping can
    catch the whole refusal family with one ``except ValueError``."""

    def __init__(self, fields: Sequence[str], detail: str) -> None:
        super().__init__(f"graduation table refused ({', '.join(fields)}): {detail}")
        self.fields = tuple(fields)


def _check_min_examples(name: str, value: int) -> None:
    # type() (not isinstance): a bool is an int subclass, and a lying
    # caller can pass a float / numpy int past the int-typed field —
    # the exact type must be int (the from_json symmetry)
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be an integer >= 0, got {value!r}")


def _check_min_precision(name: str, value: float) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float in [0.0, 1.0], got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")
    # NOTE: no tenth-grid check — a gate precision target is compared
    # against MEASURED values (any rational tp/(tp+fp)), never looked
    # up in the candidate grid (the documented asymmetry with [bands]).


def _check_thresholds(thresholds: tuple[float, ...]) -> None:
    if not thresholds:
        raise ValueError("candidate thresholds must not be empty")
    for t in thresholds:
        if not 0.0 < t < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {t}")
        if not is_tenth(t):
            raise ValueError(
                f"threshold {t} is not a tenth — the bands ride the "
                "candidate grid, which the report labels to one decimal "
                "(the eval stance, #20)"
            )
    for a, b in pairwise(thresholds):
        if a >= b:
            raise ValueError(f"thresholds must be strictly ascending, got {a} then {b}")


def evaluate_graduation(
    table: EvalTable,
    bands: BandsArtifact,
    *,
    min_examples: int = DEFAULT_MIN_EXAMPLES,
    min_precision: float = DEFAULT_MIN_PRECISION,
    overrides: Mapping[str, GateTargets] | None = None,
    managed_labels: Sequence[str],
) -> Graduation:
    """Apply the per-label Graduation gates to the measured Eval Table,
    against the calibrated Bands Artifact.

    A Managed Label graduates when ``examples >= min_examples`` (the
    gate is ``examples < min_examples`` → fail — analyze's ``total <
    min_total`` operator, so a value exactly at the gate passes) AND
    the measured precision AT ITS CALIBRATED HIGH BAND THRESHOLD is
    ``>= min_precision`` (the gate is ``precision < min_precision`` →
    fail — analyze's strict-gt precedent, so a value exactly at the
    gate passes). A label whose band has no high threshold cannot
    graduate (``no-high-band``); the med band never graduates anything
    by itself. Unearned labels carry named reasons, never a silent
    zero.

    Refusals (fail-loud doctrine): an empty/non-canonical vocabulary,
    a candidate grid that is empty, out of (0, 1), not strictly
    ascending, or not a grid of tenths (inherited from the eval), rows
    that do not align to the vocabulary in vocabulary order, cells that
    do not align to the grid, a bands artifact whose vocabulary or
    candidate grid does not match the table (a lying identity), an
    empty managed_labels (nothing to graduate), duplicate managed
    labels, a managed label the table does not carry (named), a
    ``min_examples`` that is not an integer >= 0 (a bool is not a
    count), a ``min_precision`` outside [0, 1] (a bool is not a
    target), an override for a label the table does not carry (named),
    and an override for a label NOT in managed_labels (named — a
    silent drop: the gate would be configured but never applied) all
    raise ValueError — the gates are a measured claim, never a guess.
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
    # the artifact carries the vocabulary as its label entries (sorted
    # and unique — validated), not as a separate field
    bands_vocabulary = tuple(entry.label for entry in bands.labels)
    if bands_vocabulary != tuple(table.vocabulary):
        raise ValueError(
            f"the bands' vocabulary {bands_vocabulary} does not match "
            f"the table's vocabulary {tuple(table.vocabulary)} — refusing "
            "to gate against a lying identity"
        )
    if tuple(bands.candidate_thresholds) != tuple(table.thresholds):
        raise ValueError(
            f"the bands' candidate grid {tuple(bands.candidate_thresholds)} "
            f"does not match the table's thresholds {tuple(table.thresholds)} "
            "— the high/med thresholds would not be grid points of the "
            "measured table"
        )
    _check_min_examples("min_examples", min_examples)
    _check_min_precision("min_precision", min_precision)
    if not managed_labels:
        raise ValueError(
            "managed_labels must not be empty — nothing to graduate "
            "(populate [labels] managed first)"
        )
    if len(set(managed_labels)) != len(managed_labels):
        duplicates = sorted(
            {label for label in managed_labels if managed_labels.count(label) > 1}
        )
        raise ValueError(
            f"duplicate managed label(s) {duplicates} — refusing to gate "
            "the same label twice"
        )
    unknown = sorted(set(managed_labels) - set(table.vocabulary))
    if unknown:
        raise ValueError(
            f"managed label(s) {unknown} not in the vocabulary — refusing "
            "to gate silently (fix [labels] managed)"
        )
    override_map: Mapping[str, GateTargets] = overrides if overrides is not None else {}
    unknown_overrides = sorted(set(override_map) - set(table.vocabulary))
    if unknown_overrides:
        raise ValueError(
            f"override(s) for label(s) {unknown_overrides} not in the "
            "vocabulary — refusing to gate silently (fix the override)"
        )
    # an override for a label that is NOT managed is a silent drop:
    # the gate would be configured but never applied (refuse-never-
    # drop — the operator's intent must be visible, not absorbed)
    unmanaged_overrides = sorted(set(override_map) - set(managed_labels))
    if unmanaged_overrides:
        raise ValueError(
            f"override(s) for label(s) {unmanaged_overrides} not in "
            "managed_labels — refusing to gate silently (fix the "
            "override or [labels] managed)"
        )
    # the override targets get the SAME validation as the run's — an
    # out-of-range per-label target would miscalibrate silently (the
    # config layer shields the CLI path; the core is the public
    # boundary)
    for label, override in sorted(override_map.items()):
        _check_min_examples(f"overrides[{label!r}].min_examples", override.min_examples)
        _check_min_precision(
            f"overrides[{label!r}].min_precision", override.min_precision
        )

    band_entries = {entry.label: entry for entry in bands.labels}
    rows = {row.label: row for row in table.rows}
    labels: list[GraduationEntry] = []
    targets: list[tuple[str, GateTargets]] = []
    for label in sorted(managed_labels):  # label-sorted — a total order
        row = rows[label]
        band = band_entries[label]
        effective = override_map.get(
            label, GateTargets(min_examples=min_examples, min_precision=min_precision)
        )
        examples = row.support
        high = band.high
        precision_at_high: float | None = None
        if high is not None:
            # the high band was earned at a cell with DEFINED precision
            # (calibration reads tp/fp — a zero-denominator cell never
            # earns a band). The bands artifact is a SEPARATE input,
            # though: if it was calibrated on a DIFFERENT eval table
            # (same vocabulary + grid — the exact reason the artifact
            # exists), the cell at the high band may have NO
            # predictions on THIS table (tp=fp=0, undefined precision)
            # — a lying measurement, refused, never a division
            for t, cell in zip(table.thresholds, row.cells, strict=True):
                if t == high:
                    if cell.tp + cell.fp == 0:
                        raise ValueError(
                            f"bands for {label!r}: the high band {high} "
                            "has no measured predictions on this table "
                            "(tp=fp=0, undefined precision) — the bands "
                            "were calibrated on a different eval; re-run "
                            "the calibration against this table"
                        )
                    precision_at_high = cell.tp / (cell.tp + cell.fp)
                    break
            if precision_at_high is None:
                # unreachable (high is a grid point — checked above),
                # named in the refusal family, never an assert (which
                # -O strips and which is not a ValueError)
                raise ValueError(
                    f"bands for {label!r}: the high band {high} is not a "
                    "candidate threshold of the table"
                )
        reasons: list[str] = []
        # the operators MIRROR analyze.py exactly (the strict-gt
        # precedent): `total < min_total` → a value exactly at the gate
        # passes; `dup_rate > max_dup_rate` → a value exactly at the
        # gate passes
        if examples < effective.min_examples:
            reasons.append(_REASON_BELOW_MIN_EXAMPLES)
        if high is None:
            reasons.append(_REASON_NO_HIGH_BAND)
        elif (
            precision_at_high is not None
            and precision_at_high < effective.min_precision
        ):
            reasons.append(_REASON_PRECISION_BELOW_GATE)
        # min_precision is float-NORMALIZED at the entry boundary:
        # int 1 and 1.0 are ONE meaning, and the canonical payload
        # serializes floats ("1.0") — without the normalization one
        # meaning would have two identities (json.dumps(1) is "1")
        labels.append(
            GraduationEntry(
                label=label,
                examples=examples,
                precision_at_high=precision_at_high,
                high=high,
                med=band.med,
                min_examples=effective.min_examples,
                min_precision=float(effective.min_precision),
                graduated=not reasons,
                reasons=tuple(reasons),
            )
        )
        targets.append((label, effective))
    return Graduation(
        vocabulary=table.vocabulary,
        candidate_thresholds=table.thresholds,
        labels=tuple(labels),
        targets=tuple(targets),
        test_examples=table.test_examples,
        default_targets=GateTargets(
            min_examples=min_examples, min_precision=float(min_precision)
        ),
        bands_content_hash=bands.content_hash,
    )


def build_table(
    graduation: Graduation,
    *,
    vocabulary: tuple[str, ...],
    corpus_fingerprint: str,
    split_manifest_hash: str,
    encoder_id: str,
    split_date: datetime,
) -> GraduationTable:
    """Mint the Graduation Table from a gate evaluation: the gate
    results + the checkpoint identity (the sidecar's fields) + the eval
    identity (the measured window) + the Bands Artifact's content hash
    (the thresholds artifact's binding). The eval identity
    (test_examples), the default targets, and the bands binding are
    DERIVED from the graduation's provenance — never trusted from the
    caller (a caller-told identity is a lying identity). Refusals: a
    vocabulary that does not match the graduation's (a lying identity),
    a naive split_date, and a labels/targets alignment mismatch (the
    entries zip positionally — a swapped targets tuple would mint
    per-label targets for the wrong labels; a length mismatch fails the
    strict zip) raise ValueError. The minted table passes its own
    invariants (the manifest symmetry)."""
    if tuple(vocabulary) != graduation.vocabulary:
        raise ValueError(
            f"the table's vocabulary {tuple(vocabulary)} does not match "
            f"the graduation's vocabulary {graduation.vocabulary} — refusing "
            "to mint a lying identity"
        )
    if split_date.tzinfo is None:
        raise ValueError("split_date must be timezone-aware")
    split_date = split_date.astimezone(UTC)  # canonical instant in the table
    for entry, (target_label, _effective) in zip(
        graduation.labels, graduation.targets, strict=True
    ):
        if entry.label != target_label:
            # the entries zip labels and targets POSITIONALLY — a
            # swapped/misaligned targets tuple would mint per-label
            # targets for the WRONG labels, and validate() cannot see
            # it (the entry's targets are never re-checked against the
            # label) — refuse the lying alignment
            raise ValueError(
                f"graduation.targets for {target_label!r} do not align "
                f"with graduation.labels' {entry.label!r} — refusing to "
                "mint per-label targets for the wrong labels"
            )
    table = GraduationTable(
        format_version=GRADUATION_FORMAT_VERSION,
        vocabulary_hash=vocabulary_hash(vocabulary),
        corpus_fingerprint=corpus_fingerprint,
        split_manifest_hash=split_manifest_hash,
        encoder_id=encoder_id,
        split_date=split_date,
        test_examples=graduation.test_examples,
        bands_content_hash=graduation.bands_content_hash,
        candidate_thresholds=graduation.candidate_thresholds,
        default_targets=graduation.default_targets,
        labels=graduation.labels,
        content_hash="",
    )
    table = replace(table, content_hash=table.compute_hash())
    table.validate()  # symmetry: the minted table passes its invariants
    return table


def verify_graduation(
    table: GraduationTable,
    *,
    vocabulary: tuple[str, ...],
    corpus_fingerprint: str,
    split_manifest_hash: str,
    encoder_id: str,
    split_date: datetime | date,
    test_examples: int,
    bands_content_hash: str,
) -> None:
    """Every identity field, checked INDEPENDENTLY; ALL disagreements
    are collected and reported in one ``GraduationRefusedError`` (a
    single-field drift is refused on its own — no cross-field
    coupling). The vocabulary check compares the table's hash against
    the hash of the expected vocabulary (the manifest is the authority,
    the table must agree with it — the sidecar rule). The
    ``bands_content_hash`` check is the thresholds artifact's binding
    (AC3): a table computed from different bands is foreign.

    A naive-dated ``split_date`` is a CALLER bug, not identity drift:
    it is refused with a plain ValueError naming it (build_table's
    naive refusal, mirrored) — a GraduationRefusedError with the
    "re-run eval --graduation" remedy would misdiagnose a type error as
    a drifted checkpoint."""
    if not isinstance(split_date, datetime) or split_date.tzinfo is None:
        raise ValueError(
            "verify_graduation: split_date must be a timezone-aware "
            "datetime — a date/naive datetime is a caller bug, not "
            "identity drift"
        )
    # a corrupt table (stale hash, violated invariant) is refused
    # BEFORE the identity comparison — a corrupted table misdiagnosed
    # as identity drift would point the human at the wrong remedy
    # (re-run eval) when the bytes themselves are the problem
    table.validate()
    # the expected identity, keyed by the SAME field names
    # IDENTITY_FIELDS carries (one home — a future field addition
    # updates this mapping, and every comparison site sees it; the
    # import-time assert below makes a drift an import-time crash-by-
    # design, not a KeyError/exit-1 at a consumer): the vocabulary
    # check compares the table's hash against the hash of the expected
    # vocabulary (the manifest is the authority, the table must agree
    # with it — the sidecar rule)
    expected: dict[str, object] = {
        "vocabulary_hash": vocabulary_hash(vocabulary),
        "corpus_fingerprint": corpus_fingerprint,
        "split_manifest_hash": split_manifest_hash,
        "encoder_id": encoder_id,
        "split_date": split_date,
        "test_examples": test_examples,
        "bands_content_hash": bands_content_hash,
    }
    # the report name for the vocabulary field is "vocabulary" (the
    # verify_bands precedent — the consumer-facing name, not the
    # stored field name)
    mismatches: list[str] = [
        ("vocabulary" if field == "vocabulary_hash" else field)
        for field in IDENTITY_FIELDS
        if getattr(table, field) != expected[field]
    ]
    if mismatches:
        raise GraduationRefusedError(
            mismatches,
            "the table disagrees with the checkpoint/eval/bands it is "
            "served against — re-run `eval --graduation` against this "
            "checkpoint",
        )


def identity_disagreement(
    existing: GraduationTable, current: GraduationTable
) -> tuple[str, ...]:
    """The identity fields on which two tables disagree (each field
    checked INDEPENDENTLY — the sidecar doctrine). The CLI's human-
    confirm diff warning and the write-path replace gate both read
    this, so the field list has exactly one home."""
    return tuple(
        field
        for field in IDENTITY_FIELDS
        if getattr(existing, field) != getattr(current, field)
    )


def _state(entry: GraduationEntry | None) -> str:
    if entry is None:
        return "(absent)"
    if entry.graduated:
        return "graduated"
    # CANONICAL order (sorted): the hash serializes reasons sorted, so
    # the same identity must render the same human text — a reversed
    # reasons tuple is the same meaning, not a different one
    return f"not graduated ({', '.join(sorted(entry.reasons))})"


def render_graduation_diff(
    old: GraduationTable | None,
    new: GraduationTable,
    *,
    path: str,
    identity_disagreement: tuple[str, ...] = (),
) -> str:
    """The human-confirm diff (ADR-0005): old → new per label, in
    label-sorted order over the union of both tables' labels. A label
    missing from the old table is named ``(absent)`` (the
    everything-vs-nothing baseline), and a foreign-bound old table
    (identity drift) is named in the header — the human sees they are
    comparing a table bound to a different checkpoint/eval/bands.
    Deterministic: same inputs → byte-identical text. This is the
    surface the human confirms BEFORE anything live changes; the write
    itself is CLI-gated (``--confirm-graduation``)."""
    lines: list[str] = []
    if identity_disagreement:
        lines.append(
            f"graduation: WARNING: existing table at {path} is bound to a "
            f"different identity ({', '.join(identity_disagreement)}) — the "
            "diff compares a foreign-bound table"
        )
    if old is None:
        lines.append(f"graduation: no existing table at {path} — baseline is empty")
    old_entries = {entry.label: entry for entry in (old.labels if old else ())}
    new_entries = {entry.label: entry for entry in new.labels}
    for label in sorted(set(new_entries) | set(old_entries)):
        lines.append(
            f"graduation: {label}: {_state(old_entries.get(label))} -> "
            f"{_state(new_entries[label])}"
        )
    return "\n".join(lines) + "\n"


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


def _need_nested_int(payload: dict[str, object], table: str, key: str) -> int:
    value = payload.get(table)
    if not isinstance(value, dict):
        raise ValueError(f"field {table!r} must be an object")
    nested = cast("dict[str, object]", value)
    item = nested.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"field {table}.{key} must be an integer")
    return item


def _need_nested_float(payload: dict[str, object], table: str, key: str) -> float:
    value = payload.get(table)
    if not isinstance(value, dict):
        raise ValueError(f"field {table!r} must be an object")
    nested = cast("dict[str, object]", value)
    item = nested.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"field {table}.{key} must be a number")
    return float(item)


def _need_label_entries(payload: dict[str, object]) -> tuple[GraduationEntry, ...]:
    value = payload.get("labels")
    if not isinstance(value, list):
        raise ValueError("field 'labels' must be a list of objects")
    items = cast("list[object]", value)
    entries: list[GraduationEntry] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"labels[{i}] must be an object")
        entry = cast("dict[str, object]", item)
        try:
            label = _need_str(entry, "label")
            if not label.strip():
                raise ValueError("field 'label' must be non-empty")
            examples = _need_int(entry, "examples")
            if examples < 0:
                raise ValueError("field 'examples' must be >= 0")
            entries.append(
                GraduationEntry(
                    label=label,
                    examples=examples,
                    precision_at_high=_need_band(entry, "precision_at_high"),
                    high=_need_band(entry, "high"),
                    med=_need_band(entry, "med"),
                    min_examples=_need_int(entry, "min_examples"),
                    min_precision=_need_number(entry, "min_precision"),
                    graduated=_need_bool(entry, "graduated"),
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


def _need_bool(entry: dict[str, object], key: str) -> bool:
    value = entry.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"field {key!r} must be a boolean")
    return value


def _need_str_list(entry: dict[str, object], key: str) -> tuple[str, ...]:
    value = entry.get(key)
    if not isinstance(value, list):
        raise ValueError(f"field {key!r} must be a list of strings")
    items = cast("list[object]", value)
    if not all(isinstance(v, str) for v in items):
        raise ValueError(f"field {key!r} must be a list of strings")
    return tuple(cast("list[str]", items))
