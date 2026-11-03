"""Checkpoint sidecar identity + loader refusal (ticket #19, AC#3).

The refusal contract pinned here, offline (tmp_path + FakeModel, no
torch): the loader REFUSES any checkpoint whose sidecar disagrees with
the Corpus/config it is asked to serve against — each field refused
INDEPENDENTLY (vocab hash, corpus fingerprint, split-manifest hash,
thresholds source, encoder id), so a stale or mismatched model can
never be served silently (spec #13: kills vocabulary drift). The
loader also verifies the materialized manifest.json against the
sidecar (invariant #7: a tampered manifest refuses), and the
Corpus Fingerprint is TRAIN-SPLIT scoped so append-only corpus growth
cannot invalidate a serving checkpoint (#25's lifeline).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gmail_labeler.corpus_record import CorpusRecord, body_hash, corpus_fingerprint
from gmail_labeler.dataset_build import SplitManifest, build_dataset
from gmail_labeler.model import (
    THRESHOLDS_SOURCE_NONE,
    CheckpointRefusedError,
    CheckpointSidecar,
    ExpectedIdentity,
    FakeModel,
    load_model,
    vocabulary_hash,
)

NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)
SPLIT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
VOCAB = ("Money", "Money/Bills", "Shipping")


def _rec(i: int, labels: tuple[str, ...]) -> CorpusRecord:
    return CorpusRecord(
        id=f"{i:016x}",
        date=NOW - timedelta(days=i),
        subject="hello",
        sender="x@y.z",
        recipients=("me@x.y",),
        labels=labels,
        body_hash=body_hash(f"unique body {i}"),
        source="api",
        date_provenance="internal_date",
    )


def _records() -> list[CorpusRecord]:
    return [_rec(1, ("Money",)), _rec(2, ("Shipping",)), _rec(3, ("Money/Bills",))]


def _manifest() -> SplitManifest:
    # a real minted manifest: record 1 train (before the split),
    # records 2-3 test — the fingerprint is scoped to the train ids
    from dataclasses import replace

    recs = [replace(_records()[0], date=SPLIT - timedelta(days=10)), *_records()[1:]]
    result = build_dataset(
        recs,
        filters=[],
        id_to_name={},
        now=NOW,
        split_date=SPLIT,
    )
    return result.manifest


def _sidecar(**overrides: object) -> CheckpointSidecar:
    manifest = _manifest()
    base: dict[str, object] = {
        "format_version": 1,
        "vocabulary_hash": vocabulary_hash(VOCAB),
        "thresholds_source": THRESHOLDS_SOURCE_NONE,
        "corpus_fingerprint": corpus_fingerprint(_records(), manifest.train_ids),
        "split_manifest_hash": manifest.content_hash,
        "encoder_id": "deberta-v3-base",
    }
    base.update(overrides)
    return CheckpointSidecar(**base)  # type: ignore[arg-type]


def _expected() -> ExpectedIdentity:
    manifest = _manifest()
    return ExpectedIdentity(
        vocabulary=VOCAB,
        corpus_fingerprint=corpus_fingerprint(_records(), manifest.train_ids),
        split_manifest_hash=manifest.content_hash,
        thresholds_source=THRESHOLDS_SOURCE_NONE,
        encoder_id="deberta-v3-base",
    )


def _write_checkpoint(
    ckpt: Path, sidecar: CheckpointSidecar, *, manifest: SplitManifest | None = None
) -> Path:
    ckpt.mkdir(parents=True)
    (ckpt / "sidecar.json").write_text(sidecar.to_json(), encoding="utf-8")
    if manifest is not None:
        (ckpt / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return ckpt


def _factory(dir: Path, vocabulary: tuple[str, ...]) -> FakeModel:
    return FakeModel(vocabulary, scores={})


# ---- sidecar serialization ----


def test_sidecar_round_trip_is_byte_stable() -> None:
    sidecar = _sidecar()
    text = sidecar.to_json()
    assert sidecar.to_json() == text  # byte-stable for identical fields
    assert CheckpointSidecar.from_json(text) == sidecar


def test_sidecar_from_json_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        CheckpointSidecar.from_json("[1, 2]")  # not an object
    payload = json.loads(_sidecar().to_json())
    del payload["encoder_id"]  # missing field
    with pytest.raises(ValueError):
        CheckpointSidecar.from_json(json.dumps(payload))
    payload["encoder_id"] = 42  # wrong type
    with pytest.raises(ValueError):
        CheckpointSidecar.from_json(json.dumps(payload))


def test_sidecar_from_json_rejects_unknown_format_version() -> None:
    # forward-compatible: a sidecar we do not understand is refused,
    # never guessed at
    sidecar = _sidecar(format_version=99)
    with pytest.raises(ValueError, match="format_version"):
        CheckpointSidecar.from_json(sidecar.to_json())


def test_sidecar_from_json_rejects_non_hex_hashes() -> None:
    # 64-hex shape validation: an empty or malformed hash is refused at
    # parse, not "verified" against an equally empty expectation
    for bad in ("", "xyz", "f" * 63, "F" * 64):
        sidecar = _sidecar(corpus_fingerprint=bad)
        with pytest.raises(ValueError, match="hex hash"):
            CheckpointSidecar.from_json(sidecar.to_json())


def test_expected_identity_rejects_non_hex_hashes() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="hex hash"):
        ExpectedIdentity(
            vocabulary=VOCAB,
            corpus_fingerprint="",
            split_manifest_hash=manifest.content_hash,
            thresholds_source=THRESHOLDS_SOURCE_NONE,
            encoder_id="deberta-v3-base",
        )


# ---- corpus fingerprint: train-split scoped ----


def test_corpus_fingerprint_is_order_invariant() -> None:
    records = _records()
    manifest = _manifest()
    assert corpus_fingerprint(records, manifest.train_ids) == corpus_fingerprint(
        list(reversed(records)), manifest.train_ids
    )


def test_corpus_fingerprint_changes_with_train_slice_content() -> None:
    from dataclasses import replace

    records = _records()
    manifest = _manifest()
    other = [replace(records[0], labels=("Other",)), *records[1:]]
    assert corpus_fingerprint(records, manifest.train_ids) != corpus_fingerprint(
        other, manifest.train_ids
    )


def test_corpus_fingerprint_ignores_test_side_growth() -> None:
    # #25's lifeline: append-only growth (a NEW record, test-side) must
    # NOT move the fingerprint — the serving identity is the train
    # slice, and the corpus only grows
    records = _records()
    manifest = _manifest()
    grown = [*records, _rec(4, ("Shipping",))]  # a new message lands
    assert corpus_fingerprint(records, manifest.train_ids) == corpus_fingerprint(
        grown, manifest.train_ids
    )


# ---- loader: happy path ----


def test_load_model_serves_matching_checkpoint(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt", _sidecar(), manifest=_manifest())
    model = load_model(ckpt, _expected(), factory=_factory)
    assert isinstance(model, FakeModel)
    assert model.vocabulary == VOCAB  # the manifest's vocabulary, verified


def test_load_model_factory_receives_dir_and_vocabulary(tmp_path: Path) -> None:
    seen: list[tuple[Path, tuple[str, ...]]] = []

    def spy(dir: Path, vocabulary: tuple[str, ...]) -> FakeModel:
        seen.append((dir, vocabulary))
        return FakeModel(vocabulary, scores={})

    ckpt = _write_checkpoint(tmp_path / "ckpt", _sidecar(), manifest=_manifest())
    load_model(ckpt, _expected(), factory=spy)
    assert seen == [(ckpt, VOCAB)]


# ---- loader: refusal contract (AC#3) ----


def test_refused_error_is_a_value_error() -> None:
    # catch-set parity: the CLI maps the refusal family with the same
    # broad catch as the builder's manifest refusals
    assert issubclass(CheckpointRefusedError, ValueError)


@pytest.mark.parametrize(
    ("field", "override"),
    [
        ("vocabulary_hash", {"vocabulary_hash": vocabulary_hash(("Money",))}),
        ("corpus_fingerprint", {"corpus_fingerprint": "0" * 64}),
        ("split_manifest_hash", {"split_manifest_hash": "e" * 64}),
        ("thresholds_source", {"thresholds_source": "calibrated:" + "d" * 64}),
        ("encoder_id", {"encoder_id": "modernbert-small"}),
    ],
)
def test_each_field_refused_independently(
    tmp_path: Path, field: str, override: dict[str, object]
) -> None:
    # exactly ONE field disagrees → the loader refuses, and names that
    # field (the others stay matching — no cross-field coupling)
    ckpt = _write_checkpoint(
        tmp_path / "ckpt", _sidecar(**override), manifest=_manifest()
    )
    with pytest.raises(CheckpointRefusedError) as exc:
        load_model(ckpt, _expected(), factory=_factory)
    assert exc.value.fields == (field,)
    assert field in str(exc.value)


def test_missing_sidecar_refused(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    with pytest.raises(CheckpointRefusedError) as exc:
        load_model(ckpt, _expected(), factory=_factory)
    assert exc.value.fields == ("sidecar-missing",)


def test_corrupt_sidecar_refused(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "sidecar.json").write_text("not json", encoding="utf-8")
    with pytest.raises(CheckpointRefusedError) as exc:
        load_model(ckpt, _expected(), factory=_factory)
    assert exc.value.fields == ("sidecar-corrupt",)


def test_unknown_sidecar_version_refused_end_to_end(tmp_path: Path) -> None:
    # the version check runs FIRST through load_model, with its own
    # refusal code — not conflated with 'sidecar-corrupt' (the
    # operator's signal that a newer tool wrote this checkpoint)
    ckpt = _write_checkpoint(
        tmp_path / "ckpt", _sidecar(format_version=99), manifest=_manifest()
    )
    with pytest.raises(CheckpointRefusedError) as exc:
        load_model(ckpt, _expected(), factory=_factory)
    assert exc.value.fields == ("sidecar-version",)


def test_missing_manifest_refused(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt", _sidecar())  # no manifest
    with pytest.raises(CheckpointRefusedError) as exc:
        load_model(ckpt, _expected(), factory=_factory)
    assert exc.value.fields == ("manifest-missing",)


def test_tampered_manifest_refused(tmp_path: Path) -> None:
    # invariant #7: the materialized manifest must agree with the
    # sidecar — a tampered manifest.json (sidecar untouched) refuses
    manifest = _manifest()
    from dataclasses import replace

    tampered = replace(manifest, test_ids=(*manifest.test_ids, "9" * 16))
    # re-hash the tampered manifest so it is internally consistent:
    # the TELL is that it no longer matches the sidecar's hash
    tampered = replace(tampered, content_hash=tampered.compute_hash())
    ckpt = _write_checkpoint(tmp_path / "ckpt", _sidecar(), manifest=tampered)
    with pytest.raises(CheckpointRefusedError) as exc:
        load_model(ckpt, _expected(), factory=_factory)
    assert exc.value.fields == ("split-manifest-mismatch",)


def test_refusal_names_every_disagreeing_field(tmp_path: Path) -> None:
    # two fields drift at once → both are named (independent checks,
    # one report)
    ckpt = _write_checkpoint(
        tmp_path / "ckpt",
        _sidecar(encoder_id="other", split_manifest_hash="e" * 64),
        manifest=_manifest(),
    )
    with pytest.raises(CheckpointRefusedError) as exc:
        load_model(ckpt, _expected(), factory=_factory)
    # deterministic field order: the sidecar's own field order
    assert exc.value.fields == ("split_manifest_hash", "encoder_id")


# ---- #25's lifeline: append-only growth keeps serving ----


def test_append_only_growth_does_not_invalidate_serving_identity(
    tmp_path: Path,
) -> None:
    # THE Phase 2 daemon scenario: train, fingerprint, then the corpus
    # grows (a new message lands). The serving identity is the
    # TRAIN-SPLIT slice — the new (test-side) record must not move it,
    # and the checkpoint must still load.
    from gmail_labeler.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    store.enforce_modes()
    records = _records()
    for i, record in enumerate(records, start=1):
        store.put(record, f"unique body {i}")  # must match body_hash
    manifest = _manifest()
    fingerprint = corpus_fingerprint(records, manifest.train_ids)
    # a new message lands (test-side: after the split)
    newcomer = _rec(4, ("Shipping",))
    store.put(newcomer, "unique body 4")  # must match body_hash
    grown = list(store.iter_records())
    assert len(grown) == len(records) + 1
    # the fingerprint is unchanged — the train slice did not move
    assert corpus_fingerprint(grown, manifest.train_ids) == fingerprint
    # and the checkpoint serves against the grown corpus
    ckpt = _write_checkpoint(
        tmp_path / "ckpt",
        _sidecar(corpus_fingerprint=fingerprint),
        manifest=manifest,
    )
    expected = ExpectedIdentity(
        vocabulary=VOCAB,
        corpus_fingerprint=fingerprint,
        split_manifest_hash=manifest.content_hash,
        thresholds_source=THRESHOLDS_SOURCE_NONE,
        encoder_id="deberta-v3-base",
    )
    model = load_model(ckpt, expected, factory=_factory)
    assert model.vocabulary == VOCAB
