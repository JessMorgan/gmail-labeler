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
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from .dataset_build import sorted_unique

__all__ = ["FakeModel", "Model", "vocabulary_hash"]


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
    canonical = json.dumps(
        list(vocabulary), ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
