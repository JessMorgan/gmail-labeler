"""train CLI: config → manifest rebuild → recipe behind the seam →
checkpoint + sidecar, exit-code contract 0/2/3/4/5 (typer.testing prior
art; the factory seam is monkeypatched to a FakeModel producer — the
offline spine, invariant #9).
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner, Result

from gmail_labeler.cli import app

runner = CliRunner()

SPLIT = "2026-01-01"  # train strictly before, test at/after (invariant #7)


def _seed_corpus(tmp_path: Path, *, n: int = 12) -> Path:
    """Records on BOTH sides of the split, with the sidecar pair the
    build needs (invariant #6: an empty filter list is a valid bridge)."""
    from gmail_labeler.corpus_record import CorpusRecord, body_hash
    from gmail_labeler.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    store.enforce_modes()
    now = datetime.now(UTC)
    for i in range(n):
        # i < n//2 → before the split (train); the rest → test
        days_ago = 400 - i if i < n // 2 else 10 + i
        rec = CorpusRecord(
            id=f"{i:016x}",
            date=now - timedelta(days=days_ago),
            subject=f"subject {i}",
            sender="shop@x.z",
            recipients=("me@x.y",),
            labels=("Shipping",) if i % 2 == 0 else ("Money/Bills",),
            body_hash=body_hash(f"body {i}"),
            source="api",
            date_provenance="internal_date",
        )
        store.put(rec, f"body {i}")
    for name, payload in (("filters.json", "[]"), ("labels.json", "{}")):
        f = store.root / name
        f.write_text(payload, encoding="utf-8")
        os.chmod(f, 0o600)
    return store.root


def _train_cfg(tmp_path: Path, *, encoder: str = "tiny-encoder") -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n'
        f'[model]\nencoder = "{encoder}"\n'
        f'checkpoint_dir = "{tmp_path / "ckpt"}"\n',
        encoding="utf-8",
    )
    return cfg


def _fake_factory():
    """The seam stand-in: records what it was asked to train on, writes
    a placeholder checkpoint file, returns a FakeModel. ``factory.seen``
    carries the recorded state for assertions."""
    from gmail_labeler.model import FakeModel

    seen: dict[str, object] = {}

    def factory(cfg, examples, texts, vocabulary, checkpoint_dir, device):
        seen["examples"] = list(examples)
        seen["texts"] = list(texts)
        seen["vocabulary"] = vocabulary
        seen["checkpoint_dir"] = checkpoint_dir
        seen["device"] = device
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "placeholder.bin").write_bytes(b"fake weights")
        return FakeModel(vocabulary, scores={})

    factory.seen = seen  # type: ignore[attr-defined]
    return factory


def _run_train(
    tmp_path: Path, *extra: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[Result, object]:
    """Returns (result, factory) so tests can assert on the recorded
    seam state."""
    import gmail_labeler.cli as cli

    factory = _fake_factory()
    monkeypatch.setattr(cli, "_model_factory", factory)
    result = runner.invoke(
        app,
        ["train", "--config", str(_train_cfg(tmp_path)), "--split-date", SPLIT, *extra],
    )
    return result, factory


# ---- happy path ----


def test_train_happy_path_writes_checkpoint_and_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_corpus(tmp_path)
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert "sidecar=ok" in result.stdout
    sidecar = json.loads(
        (tmp_path / "ckpt" / "sidecar.json").read_text(encoding="utf-8")
    )
    assert sidecar["thresholds_source"] == "none"  # no bands exist yet
    assert sidecar["encoder_id"] == "tiny-encoder"
    assert (tmp_path / "ckpt" / "placeholder.bin").is_file()
    # the identity files are secret-tight like the corpus sidecars
    # (the manifest carries the full vocabulary + every train/test id)
    for name in ("sidecar.json", "manifest.json"):
        assert (tmp_path / "ckpt" / name).stat().st_mode & 0o077 == 0
    assert (tmp_path / "ckpt").stat().st_mode & 0o077 == 0  # dir discipline


def test_train_sidecar_identity_matches_rebuilt_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC#4's offline half: the sidecar's identity fields are the
    # INDEPENDENTLY recomputed ones — the manifest is materialized with
    # the checkpoint (from_json re-verifies its own content hash), and
    # the corpus fingerprint is recomputed over the records
    from gmail_labeler.corpus_record import corpus_fingerprint
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.dataset_build import SplitManifest
    from gmail_labeler.model import vocabulary_hash

    _seed_corpus(tmp_path)
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    records = list(CorpusStore(tmp_path / "corpus").iter_records())
    manifest = SplitManifest.from_json(
        (tmp_path / "ckpt" / "manifest.json").read_text(encoding="utf-8")
    )
    sidecar = json.loads(
        (tmp_path / "ckpt" / "sidecar.json").read_text(encoding="utf-8")
    )
    assert sidecar["vocabulary_hash"] == vocabulary_hash(manifest.vocabulary)
    assert sidecar["split_manifest_hash"] == manifest.content_hash
    # TRAIN-SPLIT scoped: recomputed over the manifest's train ids only
    assert sidecar["corpus_fingerprint"] == corpus_fingerprint(
        records, manifest.train_ids
    )


def test_train_trains_only_the_train_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the recipe must never see test-split examples (invariant #7): the
    # factory receives exactly the manifest's train ids
    _seed_corpus(tmp_path, n=12)
    result, factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    seen = factory.seen  # type: ignore[attr-defined]
    examples = seen["examples"]
    assert len(examples) == 6  # n//2 train records
    # every example is strictly before the split
    assert all(
        example.date < datetime.fromisoformat(SPLIT).replace(tzinfo=UTC)
        for example in examples
    )
    assert seen["vocabulary"] == ("Money/Bills", "Shipping")


def test_train_checkpoint_round_trips_through_load_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC#4: Checkpoint → Model Protocol → batch-predict round trip,
    # offline. The run's output is loaded back through the LOADER with
    # an ExpectedIdentity rebuilt from INDEPENDENT sources (the
    # persisted manifest, the corpus store, the config) — the field
    # coupling (content_hash → split_manifest_hash, fingerprint →
    # corpus_fingerprint, vocabulary → vocabulary_hash) is what this
    # pins, plus the refusal side: a drifted fingerprint refuses.
    from gmail_labeler.corpus_record import corpus_fingerprint
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.dataset_build import SplitManifest
    from gmail_labeler.model import (
        THRESHOLDS_SOURCE_NONE,
        CheckpointRefusedError,
        ExpectedIdentity,
        FakeModel,
        load_model,
    )

    _seed_corpus(tmp_path)
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    ckpt = tmp_path / "ckpt"
    manifest = SplitManifest.from_json(
        (ckpt / "manifest.json").read_text(encoding="utf-8")
    )
    records = list(CorpusStore(tmp_path / "corpus").iter_records())
    expected = ExpectedIdentity(
        vocabulary=manifest.vocabulary,
        corpus_fingerprint=corpus_fingerprint(records, manifest.train_ids),
        split_manifest_hash=manifest.content_hash,
        thresholds_source=THRESHOLDS_SOURCE_NONE,
        encoder_id="tiny-encoder",
    )
    model = load_model(
        ckpt,
        expected,
        factory=lambda d, v: FakeModel(v, scores={"hello": (0.9, 0.1)}, default=0.25),
    )
    assert model.vocabulary == manifest.vocabulary
    rows = model.predict(["hello", "unseen"])
    assert rows[0] == (0.9, 0.1)  # aligned, programmed
    assert rows[1] == (0.25, 0.25)  # explicit default row, aligned
    # the refusal side: a drifted corpus fingerprint refuses the load
    drifted = ExpectedIdentity(
        vocabulary=manifest.vocabulary,
        corpus_fingerprint="0" * 64,
        split_manifest_hash=manifest.content_hash,
        thresholds_source=THRESHOLDS_SOURCE_NONE,
        encoder_id="tiny-encoder",
    )
    with pytest.raises(CheckpointRefusedError) as exc:
        load_model(ckpt, drifted, factory=lambda d, v: FakeModel(v, scores={}))
    assert exc.value.fields == ("corpus_fingerprint",)


# ---- exit-code contract ----


def test_train_missing_corpus_dir_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "missing or not a directory" in result.stderr


def test_train_config_error_exits_4(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(tmp_path / "nope.toml"),
            "--split-date",
            SPLIT,
        ],
    )
    assert result.exit_code == 4
    assert "config error" in result.stderr


def test_train_bad_split_date_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_corpus(tmp_path)
    import gmail_labeler.cli as cli

    monkeypatch.setattr(cli, "_model_factory", _fake_factory())
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(_train_cfg(tmp_path)),
            "--split-date",
            "not-a-date",
        ],
    )
    assert result.exit_code == 2
    assert "not an ISO date" in result.stderr


def test_train_missing_filters_snapshot_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _seed_corpus(tmp_path)
    (root / "filters.json").unlink()
    (root / "labels.json").unlink()
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "build refused" in result.stderr and "invariant #6" in result.stderr


def test_train_assume_no_filters_opens_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _seed_corpus(tmp_path)
    (root / "filters.json").unlink()
    (root / "labels.json").unlink()
    result, _factory = _run_train(
        tmp_path, "--assume-no-filters", monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr


def test_train_bridge_incomplete_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _seed_corpus(tmp_path)
    (root / "filters.json").write_text(
        '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": ["GONE"]}}]',
        encoding="utf-8",
    )
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "bridge" in result.stderr


def test_train_no_examples_before_split_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_corpus(tmp_path)
    import gmail_labeler.cli as cli

    monkeypatch.setattr(cli, "_model_factory", _fake_factory())
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(_train_cfg(tmp_path)),
            "--split-date",
            "2000-01-01",  # everything is after this
        ],
    )
    assert result.exit_code == 5
    assert "no training examples" in result.stderr
    assert "later" in result.stderr.lower()  # the remediation direction


def test_train_factory_failure_exits_5_no_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_corpus(tmp_path)
    import gmail_labeler.cli as cli

    def boom(cfg, examples, texts, vocabulary, checkpoint_dir, device):
        raise ValueError("encoder exploded")

    monkeypatch.setattr(cli, "_model_factory", boom)
    result = runner.invoke(
        app, ["train", "--config", str(_train_cfg(tmp_path)), "--split-date", SPLIT]
    )
    assert result.exit_code == 5
    assert "encoder exploded" in result.stderr
    assert not (tmp_path / "ckpt" / "sidecar.json").exists()  # no half-identity


def test_train_encoder_path_unset_refused_before_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the REAL _model_factory (no monkeypatch): the guard must refuse
    # without importing torch — the exit is 5, not a torch traceback
    _seed_corpus(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n'
        f'[model]\ncheckpoint_dir = "{tmp_path / "ckpt"}"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["train", "--config", str(cfg), "--split-date", SPLIT])
    assert result.exit_code == 5
    assert "encoder_path" in result.stderr
    assert "Traceback" not in (result.stdout + result.stderr)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="the train extra is installed — the missing-extra path is untestable",
)
def test_train_missing_train_extra_exits_5_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the REAL _model_factory with encoder_path set: the lazy import
    # fails (torch is absent from the CI venv) → exit 5 with an install
    # hint, never a traceback (the exit-code contract holds for the
    # most common first-run failure). Skipped where the train extra IS
    # installed — there the import succeeds and the failure mode is
    # different (a bad local weights dir, not a missing extra).
    _seed_corpus(tmp_path)
    (tmp_path / "weights").mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n'
        f'[model]\nencoder_path = "{tmp_path / "weights"}"\n'
        f'checkpoint_dir = "{tmp_path / "ckpt"}"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["train", "--config", str(cfg), "--split-date", SPLIT])
    assert result.exit_code == 5
    assert "extra" in result.stderr
    assert "Traceback" not in (result.stdout + result.stderr)


# ---- --force posture (ADR-0005 human gate) ----


def test_train_existing_checkpoint_refused_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch=monkeypatch)[0].exit_code == 0
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "human-gated" in result.stderr and "--force" in result.stderr


def test_train_force_replaces_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch=monkeypatch)[0].exit_code == 0
    result, _factory = _run_train(tmp_path, "--force", monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert "sidecar=ok" in result.stdout


def test_force_crash_leaves_old_checkpoint_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # staging-based --force: a crash mid-training leaves the OLD
    # checkpoint INTACT (still verifiable) and a garbage .tmp staging
    # dir — never new weights under an old, still-passing sidecar (the
    # silent-serving hazard the refusal contract exists to prevent).
    # The next --force run cleans the staging leftover.
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch=monkeypatch)[0].exit_code == 0
    old_sidecar = (tmp_path / "ckpt" / "sidecar.json").read_text(encoding="utf-8")
    import gmail_labeler.cli as cli

    def crash(cfg, examples, texts, vocabulary, checkpoint_dir, device):
        (checkpoint_dir / "placeholder.bin").write_bytes(b"new weights")
        raise ValueError("crashed mid-save")

    monkeypatch.setattr(cli, "_model_factory", crash)
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(_train_cfg(tmp_path)),
            "--split-date",
            SPLIT,
            "--force",
        ],
    )
    assert result.exit_code == 5
    # the old checkpoint is intact and still its own identity
    assert (tmp_path / "ckpt" / "sidecar.json").read_text(
        encoding="utf-8"
    ) == old_sidecar
    assert (tmp_path / "ckpt" / "manifest.json").is_file()
    # the crash residue sits in the staging dir, not the checkpoint
    assert (tmp_path / "ckpt.tmp" / "placeholder.bin").is_file()
    # the next --force run cleans the residue and succeeds
    assert _run_train(tmp_path, "--force", monkeypatch=monkeypatch)[0].exit_code == 0
    assert not (tmp_path / "ckpt.tmp").exists()


def test_force_replacement_leaves_no_stale_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # wholesale replacement: stale encoder-specific files (tokenizer,
    # weight shards) must NOT survive into the new dir — a hybrid dir
    # validated by an all-new sidecar is the silent-drift hazard
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch=monkeypatch)[0].exit_code == 0
    (tmp_path / "ckpt" / "stale-tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ckpt" / "old-shard-00001-of-00002.bin").write_bytes(b"old")
    result, _factory = _run_train(tmp_path, "--force", monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    names = {p.name for p in (tmp_path / "ckpt").iterdir()}
    assert "stale-tokenizer.json" not in names
    assert "old-shard-00001-of-00002.bin" not in names
    assert "sidecar.json" in names and "manifest.json" in names


def test_train_nonempty_checkpoint_dir_without_sidecar_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an orphaned dir (a crashed prior run) has no verifiable identity —
    # refuse it too, until --force
    _seed_corpus(tmp_path)
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "orphan.bin").write_bytes(b"crash residue")
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "not empty" in result.stderr


# ---- ticket-central invariants ----


def test_train_rebuilds_through_the_minted_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # THE ticket-central invariant (invariant #7): the second
    # build_dataset call carries the MINTED manifest (manifest mode —
    # no re-split). A naive re-split (manifest=None on the 2nd call)
    # must fail this (mutation M5).
    import gmail_labeler.dataset_build as db
    from gmail_labeler.dataset_build import SplitManifest

    calls: list[SplitManifest | None] = []
    original = db.build_dataset

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(cast("SplitManifest | None", kwargs.get("manifest")))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(db, "build_dataset", spy)
    _seed_corpus(tmp_path)
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert len(calls) == 2
    assert calls[0] is None  # first call mints
    minted = calls[1]
    assert minted is not None  # second call rebuilds THROUGH the manifest
    assert minted.bridge_hash  # the bridge pin rides on it
    sidecar = json.loads(
        (tmp_path / "ckpt" / "sidecar.json").read_text(encoding="utf-8")
    )
    assert sidecar["split_manifest_hash"] == minted.content_hash


def test_train_device_flag_is_plumbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --device reaches the recipe through the seam (no torch needed)
    _seed_corpus(tmp_path)
    result, factory = _run_train(tmp_path, "--device", "cuda", monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert factory.seen["device"] == "cuda"  # type: ignore[attr-defined]
    result, factory = _run_train(tmp_path, "--force", monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert factory.seen["device"] == "cpu"  # type: ignore[attr-defined]


def test_train_bad_device_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_corpus(tmp_path)
    import gmail_labeler.cli as cli

    monkeypatch.setattr(cli, "_model_factory", _fake_factory())
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(_train_cfg(tmp_path)),
            "--split-date",
            SPLIT,
            "--device",
            "tpu",
        ],
    )
    assert result.exit_code == 2
    assert "must be 'cpu' or 'cuda'" in result.stderr


def test_train_split_date_with_time_component_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # strict DATE: a time component is a usage error, not a split
    _seed_corpus(tmp_path)
    import gmail_labeler.cli as cli

    monkeypatch.setattr(cli, "_model_factory", _fake_factory())
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(_train_cfg(tmp_path)),
            "--split-date",
            "2026-01-01T12:00:00",
        ],
    )
    assert result.exit_code == 2
    assert "no time component" in result.stderr


def test_train_empty_corpus_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a present-but-empty corpus is an operation failure, not "no mail"
    from gmail_labeler.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    store.root.mkdir(parents=True)
    (store.root / "filters.json").write_text("[]", encoding="utf-8")
    (store.root / "labels.json").write_text("{}", encoding="utf-8")
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "corpus is empty" in result.stderr


def test_train_undecodable_body_shard_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a corrupt (undecodable) body shard is exit 5, never a traceback
    root = _seed_corpus(tmp_path, n=4)
    shard = root / "bodies" / "00" / "0000000000000000.txt"
    shard.write_bytes(b"\xff\xfe not utf-8")
    result, _factory = _run_train(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "body shard is unreadable" in result.stderr
    assert "Traceback" not in (result.stdout + result.stderr)


def test_train_factory_vocabulary_mismatch_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the sidecar must describe the TRAINER'S product: a factory whose
    # model vocabulary does not match the manifest's is refused before
    # any sidecar is written
    _seed_corpus(tmp_path)
    import gmail_labeler.cli as cli
    from gmail_labeler.model import FakeModel

    def lying_factory(cfg, examples, texts, vocabulary, checkpoint_dir, device):
        return FakeModel(("Other",), scores={})  # not the manifest's vocab

    monkeypatch.setattr(cli, "_model_factory", lying_factory)
    result = runner.invoke(
        app, ["train", "--config", str(_train_cfg(tmp_path)), "--split-date", SPLIT]
    )
    assert result.exit_code == 5
    assert "does not match" in result.stderr
    assert not (tmp_path / "ckpt" / "sidecar.json").exists()


# ---- surface ----


def test_train_is_on_the_frozen_surface() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    # the FLAG SURFACE is asserted from click's parameter metadata, not
    # the rendered help: rich wraps long option names differently per
    # terminal width (a width-80 CI runner split "--split-date" across
    # lines and this test went red with the surface unchanged).
    from typer.core import TyperGroup
    from typer.main import get_command

    root = get_command(app)
    assert isinstance(root, TyperGroup)
    train_cmd = root.commands["train"]
    declared = {opt for p in train_cmd.params for opt in getattr(p, "opts", [])}
    for flag in ("--split-date", "--assume-no-filters", "--force", "--device"):
        assert flag in declared
    # and the command still renders help successfully (exit 0 covers the
    # rendering itself; content checks stay width-independent above)
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
