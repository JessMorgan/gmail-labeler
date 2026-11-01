"""Model seam (GLOSSARY: Model) — the ONE new test seam (spec #13).

The ``Model`` Protocol is the public boundary between everything that
scores text (eval, shadow, the daemon) and the encoder that produces
per-label sigmoid scores (ADR-0001: small multi-label encoder with one
sigmoid head per Label Path). The real encoder (torch/transformers)
lives behind a lazy import in the ``train/`` recipe code ONLY — this
module must stay importable without torch (AC: torch is an optional
extra, CI stays torch-free).

``FakeModel`` is the offline spine mirroring ``FakeGmailClient``'s
role (invariant #9): hand-authored score vectors, wire-faithful
alignment (row i ↔ texts[i], columns aligned to the vocabulary), a
call log for exact-behavior assertions, and refusal to be programmed
with scores that a real encoder could never emit.

Checkpoint identity (GLOSSARY: Checkpoint Sidecar): a checkpoint is a
plain HF-style directory (gitignored content) plus a JSON sidecar
encoding its identity — label-vocabulary hash, thresholds source,
Corpus fingerprint, train-split manifest hash, encoder identifier.
``load_model`` REFUSES any checkpoint whose sidecar disagrees with the
Corpus/config it is asked to serve against (``ExpectedIdentity``) —
each field refused independently, so a stale or mismatched model can
never be served silently (kills vocabulary drift).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .dataset_build import SplitManifest, sorted_unique

__all__ = [
    "MANIFEST_FILENAME",
    "SIDECAR_FILENAME",
    "SIDECAR_FORMAT_VERSION",
    "THRESHOLDS_SOURCE_NONE",
    "CheckpointRefusedError",
    "CheckpointSidecar",
    "ExpectedIdentity",
    "FakeModel",
    "Model",
    "load_model",
    "verify_sidecar",
    "vocabulary_hash",
]

#: The sidecar file living inside the checkpoint directory.
SIDECAR_FILENAME = "sidecar.json"

#: The materialized Split Manifest file living inside the checkpoint
#: directory (train/eval rebuild against it; its content hash is what
#: the sidecar's ``split_manifest_hash`` must equal).
MANIFEST_FILENAME = "manifest.json"

#: Checkpoint Sidecar format version (GLOSSARY: Checkpoint Sidecar).
#: Bump on any change to the sidecar's field set or semantics.
SIDECAR_FORMAT_VERSION = 1

#: The thresholds source identity (GLOSSARY: Thresholds Source) while no
#: calibrated Threshold Bands exist yet: train writes it, the loader
#: expects it. #21's calibration artifact slots into this field (e.g.
#: ``calibrated:<hash>``) without a sidecar format change.
THRESHOLDS_SOURCE_NONE = "none"

# Hash-shaped identity fields (CorpusRecord._HASH_RE precedent): an
# empty or malformed hash must be refused at parse, not "verified"
# against an equally empty expectation.
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class Model(Protocol):
    """Batch predict → per-label sigmoid scores over the checkpoint's
    Label-Path vocabulary (GLOSSARY: Label Path). Conformance is
    structural (pyright strict) — a consumer annotated ``Model`` is the
    pin; no isinstance checks anywhere."""

    #: The checkpoint's label vocabulary — canonical sorted Label Paths;
    #: column j of every score row is the score for ``vocabulary[j]``.
    vocabulary: tuple[str, ...]

    def predict(self, texts: Iterable[str]) -> tuple[tuple[float, ...], ...]:
        """One score row per input text, aligned: row i corresponds to
        ``texts[i]``; each row carries exactly ``len(vocabulary)``
        floats in [0, 1] — column j is P(vocabulary[j] | texts[i]).
        ``texts`` is an Iterable on purpose: implementations MUST
        materialize it once (a lazy iterable consumed twice is
        misaligned)."""
        ...


class FakeModel:
    """Offline ``Model``: hand-authored score vectors.

    - ``scores`` maps text → score row (aligned to the vocabulary);
      an unprogrammed text RAISES at predict time unless the caller
      explicitly passed ``default`` — a silent 0.5 would fake a
      confidence the model never had (fail-loud doctrine);
    - the call log (``calls``) records each batch, so tests can assert
      exact predict behavior (no silent extra calls);
    - ``texts`` is materialized ONCE: a lazy iterable (generator)
      passed twice-consumed would silently misalign rows;
    - programming is validated: a score row that is misaligned with the
      vocabulary or out of sigmoid range is refused up front — a fake
      that lies about the wire is worse than no fake (the
      FakeGmailClient doctrine).
    """

    def __init__(
        self,
        vocabulary: Sequence[str],
        scores: Mapping[str, Sequence[float]] | None = None,
        default: float | None = None,
    ) -> None:
        vocab = tuple(vocabulary)
        if not vocab:
            raise ValueError("FakeModel vocabulary must not be empty")
        if not sorted_unique(vocab):
            raise ValueError(
                "FakeModel vocabulary must be sorted and unique (canonical form)"
            )
        if default is not None and not 0.0 <= default <= 1.0:
            raise ValueError(f"default must be in [0, 1], got {default}")
        self.vocabulary = vocab
        self._scores: dict[str, tuple[float, ...]] = {}
        for text, row in (scores or {}).items():
            self._scores[text] = self._check_row(text, row)
        self._default_row = None if default is None else (default,) * len(vocab)
        self.calls: list[tuple[str, ...]] = []

    def _check_row(self, text: str, row: Sequence[float]) -> tuple[float, ...]:
        if len(row) != len(self.vocabulary):
            raise ValueError(
                f"score row for {text!r} does not align to the vocabulary: "
                f"{len(row)} scores for {len(self.vocabulary)} labels"
            )
        values = tuple(row)
        if any(not 0.0 <= v <= 1.0 for v in values):
            raise ValueError(
                f"score row for {text!r} out of sigmoid range [0, 1]: {values}"
            )
        return values

    def predict(self, texts: Iterable[str]) -> tuple[tuple[float, ...], ...]:
        # materialize ONCE: a lazy iterable consumed by the call log and
        # again by the row loop would come back empty the second time
        texts = tuple(texts)
        self.calls.append(texts)
        rows: list[tuple[float, ...]] = []
        for text in texts:
            row = self._scores.get(text)
            if row is None:
                if self._default_row is None:
                    raise ValueError(
                        f"no score programmed for {text!r} and no default "
                        "given — program the text or pass default explicitly"
                    )
                row = self._default_row
            rows.append(row)
        return tuple(rows)


def vocabulary_hash(vocabulary: Sequence[str]) -> str:
    """The sha256 of the JSON of the label vocabulary — the Checkpoint
    Sidecar's vocabulary identity field. ORDER-SENSITIVE BY DESIGN: the
    vocabulary is canonically sorted Label Paths (the Split Manifest's
    vocabulary), so a REORDERED vocabulary hashes differently and must
    refuse — the hash is the drift detector. ASCII + compact
    separators make it a pure function of the content, stable across
    hosts."""
    canonical = json.dumps(list(vocabulary), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CheckpointRefusedError(ValueError):
    """The Model loader refuses to serve a checkpoint: its sidecar is
    missing, corrupt, of an unknown version, its materialized manifest
    is missing/tampered, or it disagrees with the Corpus/config it is
    asked to serve against. ``fields`` names every disagreeing sidecar
    field (each field is checked INDEPENDENTLY — a single-field drift
    is refused on its own); file-level refusals are named
    ``sidecar-missing`` / ``sidecar-corrupt`` / ``sidecar-version`` /
    ``manifest-missing`` / ``manifest-corrupt`` /
    ``split-manifest-mismatch``. Subclasses ``ValueError`` so a CLI
    mapping can catch the whole refusal family with one
    ``except ValueError`` (the same catch-set as the builder's
    manifest refusals)."""

    def __init__(self, fields: Sequence[str], detail: str) -> None:
        super().__init__(f"checkpoint refused ({', '.join(fields)}): {detail}")
        self.fields = tuple(fields)


@dataclass(frozen=True, slots=True)
class CheckpointSidecar:
    """The JSON identity file of a checkpoint (GLOSSARY: Checkpoint
    Sidecar) — five identity fields, each independently verifiable:

    - ``vocabulary_hash``: sha256 of the checkpoint's label vocabulary
      (Label Paths) — kills vocabulary drift;
    - ``thresholds_source``: where the serving thresholds come from —
      ``THRESHOLDS_SOURCE_NONE`` until calibration exists (#21 slots
      its artifact in here);
    - ``corpus_fingerprint``: identity of the TRAIN-SPLIT slice of the
      Corpus the checkpoint was trained on (GLOSSARY: Corpus
      Fingerprint) — append-only corpus growth does not move it;
    - ``split_manifest_hash``: the train-split Split Manifest's content
      hash (invariant #7 — the split is pinned, not re-derived);
    - ``encoder_id``: the encoder identifier from the config.

    Hash-shaped fields are shape-validated (64 hex) at parse — an
    empty string is refused, not "verified". A sidecar whose
    hash-consistent fields violate the format invariants is corrupt,
    not a valid variant — ``from_json`` refuses it exactly like the
    Split Manifest does."""

    format_version: int
    vocabulary_hash: str
    thresholds_source: str
    corpus_fingerprint: str
    split_manifest_hash: str
    encoder_id: str

    def to_json(self) -> str:
        """Canonical serialization (sorted keys, ASCII, compact) —
        byte-stable for identical fields."""
        payload = {
            "format_version": self.format_version,
            "vocabulary_hash": self.vocabulary_hash,
            "thresholds_source": self.thresholds_source,
            "corpus_fingerprint": self.corpus_fingerprint,
            "split_manifest_hash": self.split_manifest_hash,
            "encoder_id": self.encoder_id,
        }
        return json.dumps(
            payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, text: str) -> CheckpointSidecar:
        """Parse + shape-verify; a wrong shape, a malformed hash, or an
        unknown format version raises ValueError — a sidecar we do not
        understand is refused, never guessed at (forward-compatible:
        #21's artifact must not make an old loader serve a new
        sidecar)."""
        data: object = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("sidecar is not an object")
        payload = cast("dict[str, object]", data)
        try:
            sidecar = cls(
                format_version=_need_int(payload, "format_version"),
                vocabulary_hash=_need_hash(payload, "vocabulary_hash"),
                thresholds_source=_need_str(payload, "thresholds_source"),
                corpus_fingerprint=_need_hash(payload, "corpus_fingerprint"),
                split_manifest_hash=_need_hash(payload, "split_manifest_hash"),
                encoder_id=_need_str(payload, "encoder_id"),
            )
        except ValueError as e:
            raise ValueError(f"sidecar field invalid: {e}") from e
        if sidecar.format_version != SIDECAR_FORMAT_VERSION:
            raise ValueError(
                f"sidecar format_version {sidecar.format_version} "
                f"!= {SIDECAR_FORMAT_VERSION}"
            )
        return sidecar


@dataclass(frozen=True, slots=True)
class ExpectedIdentity:
    """The Corpus/config a checkpoint is asked to serve against — what
    the loader verifies the sidecar against. ``vocabulary`` is the
    Split Manifest's vocabulary (its hash is what the sidecar's
    ``vocabulary_hash`` must equal); ``split_manifest_hash`` is the
    manifest's ``content_hash``; ``thresholds_source`` is what the
    serving config declares (``THRESHOLDS_SOURCE_NONE`` today);
    ``encoder_id`` is the config's encoder identifier. Hash-shaped
    fields are shape-validated at construction (an empty expectation
    must not "verify" against an empty sidecar)."""

    vocabulary: tuple[str, ...]
    corpus_fingerprint: str
    split_manifest_hash: str
    thresholds_source: str
    encoder_id: str

    def __post_init__(self) -> None:
        for name in ("corpus_fingerprint", "split_manifest_hash"):
            value = getattr(self, name)
            if not _HASH_RE.fullmatch(value):
                raise ValueError(
                    f"ExpectedIdentity.{name} must be a 64-char hex hash, got {value!r}"
                )


def verify_sidecar(sidecar: CheckpointSidecar, expected: ExpectedIdentity) -> None:
    """Every identity field, checked INDEPENDENTLY; ALL disagreements
    are collected and reported in one ``CheckpointRefusedError`` (a
    single-field drift is refused on its own — no cross-field
    coupling). The vocabulary check compares the sidecar's hash against
    the hash of the expected vocabulary — the manifest is the
    authority, the sidecar must agree with it."""
    mismatches: list[str] = []
    if sidecar.vocabulary_hash != vocabulary_hash(expected.vocabulary):
        mismatches.append("vocabulary_hash")
    if sidecar.corpus_fingerprint != expected.corpus_fingerprint:
        mismatches.append("corpus_fingerprint")
    if sidecar.split_manifest_hash != expected.split_manifest_hash:
        mismatches.append("split_manifest_hash")
    if sidecar.thresholds_source != expected.thresholds_source:
        mismatches.append("thresholds_source")
    if sidecar.encoder_id != expected.encoder_id:
        mismatches.append("encoder_id")
    if mismatches:
        raise CheckpointRefusedError(
            mismatches,
            "sidecar disagrees with the Corpus/config it is asked to "
            "serve against — rebuild the checkpoint",
        )


def load_model(
    checkpoint_dir: Path,
    expected: ExpectedIdentity,
    factory: Callable[[Path, tuple[str, ...]], Model],
) -> Model:
    """Load a checkpoint for serving: read + verify the sidecar
    (refusal contract, AC#3), verify the materialized manifest against
    the sidecar (invariant #7 — a tampered manifest.json refuses),
    then instantiate the model via ``factory(checkpoint_dir,
    vocabulary)``. The factory is a required argument — the real
    encoder's factory (``train.recipe.load_hf_model``, torch) is
    supplied by the caller, so this module (and CI) never imports
    torch; tests pass a factory returning ``FakeModel``. The served
    vocabulary is the EXPECTED one (the manifest's): the sidecar's hash
    check just proved it is the checkpoint's."""
    sidecar_path = checkpoint_dir / SIDECAR_FILENAME
    if not sidecar_path.is_file():
        raise CheckpointRefusedError(
            ("sidecar-missing",),
            f"{sidecar_path} not found — a checkpoint without a sidecar "
            "has no verifiable identity",
        )
    try:
        raw = sidecar_path.read_text(encoding="utf-8")
        data: object = json.loads(raw)
    except (ValueError, OSError) as e:
        raise CheckpointRefusedError(
            ("sidecar-corrupt",), f"{sidecar_path} unreadable: {e}"
        ) from e
    # format version FIRST, with its own refusal code: an unknown
    # version is the operator's signal that a newer tool wrote this
    # checkpoint — not "corrupt"
    if isinstance(data, dict):
        payload = cast("dict[str, object]", data)
        if payload.get("format_version") != SIDECAR_FORMAT_VERSION:
            raise CheckpointRefusedError(
                ("sidecar-version",),
                f"{sidecar_path} format_version "
                f"{payload.get('format_version')!r} != {SIDECAR_FORMAT_VERSION} — "
                "a newer tool wrote this checkpoint",
            )
    try:
        sidecar = CheckpointSidecar.from_json(raw)
    except ValueError as e:
        raise CheckpointRefusedError(
            ("sidecar-corrupt",), f"{sidecar_path} unreadable: {e}"
        ) from e
    # field-level checks FIRST (each field refused independently — a
    # broken split_manifest_hash field is named as such, not masked by
    # the manifest-file check below)
    verify_sidecar(sidecar, expected)
    # the sidecar's manifest claim passed — the materialized file must
    # PROVE it: a tampered manifest.json (sidecar untouched) is a
    # split-identity lie (invariant #7)
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise CheckpointRefusedError(
            ("manifest-missing",),
            f"{manifest_path} not found — the checkpoint's split identity "
            "is not materialized",
        )
    try:
        manifest = SplitManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise CheckpointRefusedError(
            ("manifest-corrupt",), f"{manifest_path} unreadable: {e}"
        ) from e
    if manifest.content_hash != sidecar.split_manifest_hash:
        raise CheckpointRefusedError(
            ("split-manifest-mismatch",),
            f"{manifest_path} content hash != sidecar split_manifest_hash "
            "— the manifest was tampered with",
        )
    return factory(checkpoint_dir, expected.vocabulary)


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
