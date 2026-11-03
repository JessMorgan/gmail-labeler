"""eval --bands CLI (ticket #21): the calibrated Threshold Bands
artifact — opt-in write, sidecar/manifest-bound identity, NOTICE on
replace, recompute-posture pins (ONLY eval writes the artifact),
exit-code contract 0/2/3/4/5 (typer.testing prior art; the model
factory seam is monkeypatched to a FakeModel producer — the offline
spine, invariant #9).
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from gmail_labeler.cli import app
from gmail_labeler.thresholds import BandsArtifact, BandTargets

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


def _eval_cfg(
    tmp_path: Path, *, encoder: str = "tiny-encoder", bands: str = ""
) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n'
        f"[bands]\n{bands}\n"
        f'[model]\nencoder = "{encoder}"\n'
        f'checkpoint_dir = "{tmp_path / "ckpt"}"\n',
        encoding="utf-8",
    )
    return cfg


def _train_factory():
    """The train seam stand-in (test_train_cli's pattern)."""
    from gmail_labeler.model import FakeModel

    seen: dict[str, object] = {}

    def factory(cfg, examples, texts, vocabulary, checkpoint_dir, device):
        seen["examples"] = list(examples)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "placeholder.bin").write_bytes(b"fake weights")
        return FakeModel(vocabulary, scores={})

    factory.seen = seen  # type: ignore[attr-defined]
    return factory


def _eval_factory(
    scores: dict[str, tuple[float, ...]],
    vocabulary: tuple[str, ...] | None = None,
):
    """The eval seam stand-in: returns a FakeModel programmed with
    hand-authored score vectors keyed by TEXT (the Model Protocol's
    alignment: row i ↔ texts[i])."""
    from gmail_labeler.model import FakeModel

    seen: dict[str, object] = {}

    def factory(checkpoint_dir: Path, vocab: tuple[str, ...]):
        seen["checkpoint_dir"] = checkpoint_dir
        seen["vocabulary"] = vocab
        return FakeModel(vocabulary if vocabulary is not None else vocab, scores=scores)

    factory.seen = seen  # type: ignore[attr-defined]
    return factory


def _run_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    split_date: str = SPLIT,
    force: bool = False,
) -> Result:
    import gmail_labeler.cli as cli

    monkeypatch.setattr(cli, "_model_factory", _train_factory())
    args = [
        "train",
        "--config",
        str(_eval_cfg(tmp_path)),
        "--split-date",
        split_date,
    ]
    if force:
        args.append("--force")
    return runner.invoke(app, args)


def _test_side_scores() -> dict[str, tuple[float, ...]]:
    """Hand-authored scores for the 6 test-side texts (i=6..11), in
    vocabulary order ("Money/Bills", "Shipping"):

    i=6  Shipping     (0.2, 0.9)
    i=7  Money/Bills  (0.8, 0.1)
    i=8  Shipping     (0.3, 0.7)
    i=9  Money/Bills  (0.6, 0.2)
    i=10 Shipping     (0.1, 0.4)
    i=11 Money/Bills  (0.4, 0.3)
    """
    return {
        "subject 6\nbody 6": (0.2, 0.9),
        "subject 7\nbody 7": (0.8, 0.1),
        "subject 8\nbody 8": (0.3, 0.7),
        "subject 9\nbody 9": (0.6, 0.2),
        "subject 10\nbody 10": (0.1, 0.4),
        "subject 11\nbody 11": (0.4, 0.3),
    }


def _run_eval(
    tmp_path: Path,
    *extra: str,
    monkeypatch: pytest.MonkeyPatch,
    scores: dict[str, tuple[float, ...]] | None = None,
    vocabulary: tuple[str, ...] | None = None,
    cfg: Path | None = None,
) -> tuple[Result, object]:
    import gmail_labeler.cli as cli

    factory = _eval_factory(
        scores if scores is not None else _test_side_scores(), vocabulary
    )
    monkeypatch.setattr(cli, "_eval_model_factory", factory)
    result = runner.invoke(
        app,
        [
            "eval",
            "--config",
            str(cfg if cfg is not None else _eval_cfg(tmp_path)),
            *extra,
        ],
    )
    return result, factory


# ---- the golden bands (hand-computed, independent of the code) ----
#
# Money/Bills precision (positives i=7: 0.8, i=9: 0.6, i=11: 0.4):
#   0.1: 3/6 = 0.5   0.2: 3/5 = 0.6   0.3: 3/4 = 0.75   0.4: 3/3 = 1.0
#   0.5: 2/2 = 1.0   0.6: 1.0   0.7: 1.0   0.8: 1.0   0.9: UNDEFINED
# Shipping precision (positives i=6: 0.9, i=8: 0.7, i=10: 0.4):
#   0.1: 0.5   0.2: 0.6   0.3: 0.75   0.4: 3/3 = 1.0   0.5: 1.0
#   0.6: 1.0   0.7: 1.0   0.8: 1.0   0.9: 1.0
#
# Defaults (high 0.9 / med 0.7):
#   Money/Bills: high = 0.8 (0.9 undefined), med = 0.7 (highest < 0.8
#   with P >= 0.7 — P(0.7) = 1.0)
#   Shipping:    high = 0.9, med = 0.8 (highest < 0.9 with P >= 0.7)


def test_eval_bands_writes_bound_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC2/AC5: the artifact is bound to the checkpoint + eval identity
    # and carries the measured bands (hand-computed golden)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    out = tmp_path / "bands.json"
    result, _factory = _run_eval(tmp_path, "--bands", str(out), monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert out.is_file()
    assert out.stat().st_mode & 0o077 == 0  # 0600, never world-readable
    assert not (tmp_path / "bands.json.tmp").exists()  # no staging residue
    artifact = BandsArtifact.from_json(out.read_text(encoding="utf-8"))
    # checkpoint identity (mirrored from the sidecar)
    sidecar = json.loads((tmp_path / "ckpt" / "sidecar.json").read_text())
    manifest = json.loads((tmp_path / "ckpt" / "manifest.json").read_text())
    assert artifact.vocabulary_hash == sidecar["vocabulary_hash"]
    assert artifact.corpus_fingerprint == sidecar["corpus_fingerprint"]
    assert artifact.split_manifest_hash == sidecar["split_manifest_hash"]
    assert artifact.split_manifest_hash == manifest["content_hash"]
    assert artifact.encoder_id == sidecar["encoder_id"]
    # eval identity (the measured window)
    assert artifact.split_date == datetime.fromisoformat(manifest["split_date"])
    assert artifact.test_examples == len(manifest["test_ids"]) == 6
    # provenance
    assert artifact.candidate_thresholds == (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )
    assert artifact.default_targets == BandTargets(0.9, 0.7)
    # the measured bands (the golden)
    entries = {e.label: e for e in artifact.labels}
    assert [e.label for e in artifact.labels] == ["Money/Bills", "Shipping"]
    assert entries["Money/Bills"].high == 0.8
    assert entries["Money/Bills"].med == 0.7
    assert entries["Money/Bills"].reasons == ()
    assert entries["Shipping"].high == 0.9
    assert entries["Shipping"].med == 0.8
    assert entries["Shipping"].reasons == ()
    # the report is on stdout; the bands summary is on STDERR (the
    # --out "same bytes as stdout" contract — the summary is status,
    # not report content)
    assert "test examples=6" in result.stdout
    assert (
        "bands: Money/Bills high=0.8 med=0.7; Shipping high=0.9 med=0.8"
        in result.stderr
    )


def test_eval_bands_byte_identical_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC5: same eval → byte-identical artifact (deterministic total
    # order — no clock, no I/O nondeterminism in the core)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    out1, out2 = tmp_path / "b1.json", tmp_path / "b2.json"
    r1, _f1 = _run_eval(tmp_path, "--bands", str(out1), monkeypatch=monkeypatch)
    r2, _f2 = _run_eval(tmp_path, "--bands", str(out2), monkeypatch=monkeypatch)
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert out1.read_bytes() == out2.read_bytes()


def test_eval_bands_input_order_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the corpus record order must not move the artifact bytes (the
    # total order is the vocabulary's, not the file's): the SAME
    # checkpoint + manifest, a reordered record index → identical
    # artifact
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    out1 = tmp_path / "b1.json"
    r1, _f1 = _run_eval(tmp_path, "--bands", str(out1), monkeypatch=monkeypatch)
    assert r1.exit_code == 0, r1.stderr
    # reverse the record order in the append-only index
    records_path = tmp_path / "corpus" / "records.jsonl"
    lines = records_path.read_text(encoding="utf-8").splitlines(keepends=True)
    records_path.write_text("".join(reversed(lines)), encoding="utf-8")
    out2 = tmp_path / "b2.json"
    r2, _f2 = _run_eval(tmp_path, "--bands", str(out2), monkeypatch=monkeypatch)
    assert r2.exit_code == 0, r2.stderr
    assert out1.read_bytes() == out2.read_bytes()


def test_eval_without_bands_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # recompute posture (AC4, behavioural): a bare `eval` never writes
    # the artifact — the write is the opt-in --bands act
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert not (tmp_path / "bands.json").exists()
    assert not list(tmp_path.glob("*.json.tmp"))


def test_eval_bands_replaces_existing_with_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # replacing an EXISTING artifact is announced on stderr (the #20
    # --out NOTICE — a silent clobber of a typo'd path would be a lie);
    # the file is pre-created 0644 to exercise the tighten path
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    out = tmp_path / "bands.json"
    out.write_text("old artifact", encoding="utf-8")
    os.chmod(out, 0o644)
    result, _factory = _run_eval(tmp_path, "--bands", str(out), monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert "NOTICE: replacing existing" in result.stderr
    assert out.stat().st_mode & 0o077 == 0
    BandsArtifact.from_json(out.read_text(encoding="utf-8"))  # parses


def test_eval_bands_unknown_override_label_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an override for a label the checkpoint's vocabulary does not
    # carry is refused, named — fail loud, never a silent no-op
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, bands='[bands."Ghost"]\nmin_high_precision = 0.9\n')
    out = tmp_path / "bands.json"
    result, _factory = _run_eval(
        tmp_path, "--bands", str(out), monkeypatch=monkeypatch, cfg=cfg
    )
    assert result.exit_code == 5
    assert "Ghost" in result.stderr
    assert "test examples=6" in result.stdout  # the report is never lost
    assert not out.exists()  # no partial artifact


def test_eval_bands_config_non_tenth_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a non-tenth target is a CONFIG error (the tenth-grid stance, #20)
    # — exit 4, before any corpus work
    _seed_corpus(tmp_path)
    cfg = _eval_cfg(tmp_path, bands="min_high_precision = 0.93\n")
    result, _factory = _run_eval(
        tmp_path,
        "--bands",
        str(tmp_path / "bands.json"),
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 4
    assert "config error" in result.stderr
    assert "tenth" in result.stderr


def test_eval_bands_write_failure_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --bands failure branch: an unwritable path is an operation
    # failure (exit 5), and the report still goes to stdout
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(
        tmp_path,
        "--bands",
        str(tmp_path / "no" / "such" / "dir" / "bands.json"),
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 5
    assert "cannot write --bands" in result.stderr
    assert "test examples=6" in result.stdout  # the report is not lost


def test_eval_bands_empty_path_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an empty --bands would degenerate to Path("") and fail awkwardly
    # mid-run — a usage error (exit 2) up front
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(tmp_path, "--bands", " ", monkeypatch=monkeypatch)
    assert result.exit_code == 2
    assert "non-empty" in result.stderr


def test_eval_bands_same_path_as_out_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the report and the artifact must not share a path: the artifact
    # write would silently destroy the report (exit 2 up front)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    same = tmp_path / "one.json"
    result, _factory = _run_eval(
        tmp_path,
        "--out",
        str(same),
        "--bands",
        str(same),
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 2
    assert "different files" in result.stderr
    assert not same.exists()


def test_eval_bands_same_path_as_sidecar_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a destructive collision: the artifact write would destroy the
    # checkpoint's identity sidecar (exit 2 up front, no corpus work)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    sidecar = tmp_path / "ckpt" / "sidecar.json"
    assert sidecar.is_file()
    result, _factory = _run_eval(
        tmp_path, "--bands", str(sidecar), monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "checkpoint directory" in result.stderr
    # the sidecar is intact (still the checkpoint's identity, not an
    # artifact)
    sidecar_json = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "vocabulary_hash" in sidecar_json


def test_eval_bands_inside_checkpoint_dir_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the generalized identity-file guard: ANY target inside the
    # checkpoint dir is refused — the split manifest is identity
    # material the garbage-tolerance NOTICE branch cannot see (the
    # reviewer repro: eval --bands <ckpt>/manifest.json used to exit
    # 0 and destroy it)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    manifest = tmp_path / "ckpt" / "manifest.json"
    bytes_before = manifest.read_bytes()
    result, _factory = _run_eval(
        tmp_path, "--bands", str(manifest), monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "checkpoint" in result.stderr
    assert "test examples=6" not in result.stdout  # pre-report refusal
    assert manifest.read_bytes() == bytes_before  # bytes intact


def test_eval_bands_inside_corpus_store_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the store's identity files (labels.json, filters.json,
    # records.jsonl, bodies) are equally destroyable — refused
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    labels = tmp_path / "corpus" / "labels.json"
    bytes_before = labels.read_bytes()
    result, _factory = _run_eval(
        tmp_path, "--bands", str(labels), monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "corpus store" in result.stderr
    assert labels.read_bytes() == bytes_before  # bytes intact


def test_eval_bands_symlink_into_checkpoint_dir_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a symlink whose NAME sits outside the checkpoint dir but POINTS
    # inside it: both sides are resolved, so the guard catches it (an
    # unresolved check would let the clobber through)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    sidecar = tmp_path / "ckpt" / "sidecar.json"
    bytes_before = sidecar.read_bytes()
    link = tmp_path / "bands-link.json"
    os.symlink(sidecar, link)
    result, _factory = _run_eval(
        tmp_path, "--bands", str(link), monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "checkpoint" in result.stderr
    assert sidecar.read_bytes() == bytes_before  # the target intact


def test_eval_bands_readback_is_hash_based(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # readback equality is by CONTENT HASH, not dataclass equality:
    # the canonical payload serializes reasons sorted, so a
    # reasons-order difference is the same meaning — a dataclass
    # comparison would exit 5 after a perfectly good artifact landed
    # (reachable in one reason rename)
    import gmail_labeler.cli as cli
    from gmail_labeler.dataset_build import TrainingExample
    from gmail_labeler.eval import eval_metrics
    from gmail_labeler.thresholds import build_artifact, calibrate_bands

    examples = [
        TrainingExample(
            id=f"{i:016x}",
            date=datetime.now(UTC),
            labels=(),
            weight=1.0,
            date_provenance="internal_date",
        )
        for i in range(2)
    ]
    scores = {f"{i:016x}": (0.0, 0.0) for i in range(2)}
    table = eval_metrics(examples, scores, vocabulary=("Bills", "Shipping"))
    bands = calibrate_bands(table)  # no-band state → reasons
    artifact = build_artifact(
        bands,
        vocabulary=("Bills", "Shipping"),
        corpus_fingerprint="f" * 64,
        split_manifest_hash="0" * 64,
        encoder_id="tiny-encoder",
        split_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert all(entry.reasons for entry in artifact.labels)
    reversed_entries = tuple(
        dataclasses.replace(entry, reasons=entry.reasons[::-1])
        for entry in artifact.labels
    )
    readback = dataclasses.replace(artifact, labels=reversed_entries)
    assert readback.content_hash == artifact.content_hash  # canonical
    assert readback != artifact  # dataclass equality is order-sensitive
    monkeypatch.setattr(
        BandsArtifact,
        "from_json",
        classmethod(lambda cls, text: readback),
    )
    out = tmp_path / "bands.json"
    cli._write_bands_artifact(out, artifact, artifact.to_json())  # no refusal
    assert out.is_file()


def test_eval_bands_readback_fault_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the REJECTION path: a readback fault (from_json refuses the
    # on-disk bytes — a torn/corrupt write) is a named exit-5
    # operation failure, never an uncaught ValueError traceback (exit
    # 1), and the report is preserved
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0

    def _boom(cls, text: str) -> BandsArtifact:
        raise ValueError("bands field invalid: simulated torn write")

    monkeypatch.setattr(
        BandsArtifact,
        "from_json",
        classmethod(_boom),
    )
    out = tmp_path / "bands.json"
    result, _factory = _run_eval(tmp_path, "--bands", str(out), monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "cannot write --bands" in result.stderr
    assert "readback fault" in result.stderr
    assert "test examples=6" in result.stdout  # the report is not lost


def test_eval_bands_readback_mismatch_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the true mismatch: the readback parses but its content_hash
    # differs from the minted artifact — exit 5, named
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    # any parseable artifact with a DIFFERENT stored hash
    from gmail_labeler.dataset_build import TrainingExample
    from gmail_labeler.eval import eval_metrics
    from gmail_labeler.thresholds import build_artifact, calibrate_bands

    examples = [
        TrainingExample(
            id=f"{i:016x}",
            date=datetime.now(UTC),
            labels=(),
            weight=1.0,
            date_provenance="internal_date",
        )
        for i in range(2)
    ]
    table = eval_metrics(
        examples,
        {f"{i:016x}": (0.0, 0.0) for i in range(2)},
        vocabulary=("Bills", "Shipping"),
    )
    other = build_artifact(
        calibrate_bands(table),
        vocabulary=("Bills", "Shipping"),
        corpus_fingerprint="f" * 64,
        split_manifest_hash="0" * 64,
        encoder_id="tiny-encoder",
        split_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    lying = dataclasses.replace(other, content_hash="1" * 64)
    monkeypatch.setattr(
        BandsArtifact,
        "from_json",
        classmethod(lambda cls, text: lying),
    )
    out = tmp_path / "bands.json"
    result, _factory = _run_eval(tmp_path, "--bands", str(out), monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "cannot write --bands" in result.stderr
    assert "readback mismatch" in result.stderr
    # the message names the real semantics: it DID parse
    assert "content hash differs" in result.stderr
    assert "test examples=6" in result.stdout  # the report is not lost


def test_eval_bands_same_path_as_config_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a destructive collision: the artifact write would destroy the
    # config file (exit 2 up front, no corpus work)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path)
    result, _factory = _run_eval(
        tmp_path, "--bands", str(cfg), monkeypatch=monkeypatch, cfg=cfg
    )
    assert result.exit_code == 2
    assert "config file" in result.stderr
    assert cfg.is_file() and "checkpoint_dir" in cfg.read_text()


def test_eval_bands_degraded_corpus_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # serving-gate hardening: a corpus with missing body shards is
    # scored on subject + trailing newline only — degraded scores must
    # not mint the artifact that gates the daemon's shadow path.
    # Refusal: exit 5 (the operation-failure convention the other
    # post-report eval refusals use), the report is preserved on
    # stdout, and NO artifact (or staging) is written anywhere.
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    # orphan one TEST-side body shard (i=6 → id 0000000000000006)
    shard = tmp_path / "corpus" / "bodies" / "00" / "0000000000000006.txt"
    assert shard.is_file()
    shard.unlink()
    # the degraded text for i=6 is subject + trailing newline
    scores = dict(_test_side_scores())
    scores["subject 6\n"] = scores.pop("subject 6\nbody 6")
    out = tmp_path / "bands.json"
    result, _factory = _run_eval(
        tmp_path,
        "--bands",
        str(out),
        monkeypatch=monkeypatch,
        scores=scores,
    )
    assert result.exit_code == 5
    assert "--bands refused" in result.stderr
    assert "1 test record(s) had no body shard" in result.stderr
    assert "test examples=6" in result.stdout  # the report is not lost
    assert not out.exists()  # no artifact
    assert not list(tmp_path.glob("*.json*"))  # no staging residue


def test_eval_bands_with_json_stdout_is_single_json_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json path: stdout is EXACTLY one parseable JSON object
    # (the human summary is suppressed — json.loads over the whole
    # stdout must succeed), and the artifact is still written
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    out = tmp_path / "bands.json"
    result, _factory = _run_eval(
        tmp_path, "--bands", str(out), "--json", monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)  # the whole stdout is one object
    assert payload["test_examples"] == 6
    assert [row["label"] for row in payload["rows"]] == [
        "Money/Bills",
        "Shipping",
    ]
    assert out.is_file()
    artifact = BandsArtifact.from_json(out.read_text(encoding="utf-8"))
    assert artifact.labels  # validates


def test_eval_bands_with_distinct_out_writes_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --bands plus a DISTINCT --out: both files exist complete — the
    # artifact validates, the report file is non-empty, nothing lost
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    out = tmp_path / "report.txt"
    bands = tmp_path / "bands.json"
    result, _factory = _run_eval(
        tmp_path,
        "--out",
        str(out),
        "--bands",
        str(bands),
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 0, result.stderr
    assert out.is_file() and out.read_text(encoding="utf-8").strip()
    assert "test examples=6" in out.read_text(encoding="utf-8")
    assert bands.is_file()
    artifact = BandsArtifact.from_json(bands.read_text(encoding="utf-8"))
    assert [e.label for e in artifact.labels] == ["Money/Bills", "Shipping"]


def test_eval_bands_refuses_identity_drift_on_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the serving-gate hardening: an eval of checkpoint B over a target
    # holding checkpoint A's artifact must REFUSE (exit 5, before any
    # staging) — a NOTICE-only clobber would destroy a verifiably-A
    # artifact. A's bytes stay byte-identical.
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0  # checkpoint A
    out = tmp_path / "bands.json"
    r1, _f1 = _run_eval(tmp_path, "--bands", str(out), monkeypatch=monkeypatch)
    assert r1.exit_code == 0, r1.stderr
    bytes_a = out.read_bytes()
    # checkpoint B: same corpus, a LATER split date — the manifest's
    # split identity (and the artifact's split_date) differs
    assert (
        _run_train(tmp_path, monkeypatch, split_date="2026-02-01", force=True).exit_code
        == 0
    )
    r2, _f2 = _run_eval(tmp_path, "--bands", str(out), monkeypatch=monkeypatch)
    assert r2.exit_code == 5
    assert "--bands refused" in r2.stderr
    assert "split_date" in r2.stderr  # the named disagreeing field
    assert out.read_bytes() == bytes_a  # A's bytes byte-identical


def test_eval_bands_out_file_is_the_same_bytes_as_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --out byte contract with --bands: the file is EXACTLY the
    # stdout bytes (the bands summary is status, routed to stderr —
    # the documented "same bytes" promise holds)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    out = tmp_path / "report.txt"
    bands = tmp_path / "bands.json"
    result, _factory = _run_eval(
        tmp_path,
        "--out",
        str(out),
        "--bands",
        str(bands),
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 0, result.stderr
    assert out.read_text(encoding="utf-8") == result.stdout
    assert "bands: " not in result.stdout
    assert "bands: " in result.stderr
    assert bands.is_file()


def test_eval_bands_existing_directory_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --bands naming an existing directory: refused UP FRONT as a usage
    # error (exit 2, pre-report — the old path printed the replace
    # NOTICE and then died at os.replace EISDIR, exit 5)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    d = tmp_path / "bands-dir"
    d.mkdir()
    result, _factory = _run_eval(tmp_path, "--bands", str(d), monkeypatch=monkeypatch)
    assert result.exit_code == 2
    assert "directory" in result.stderr
    assert "test examples=6" not in result.stdout  # pre-report refusal
    assert not any(d.iterdir())  # nothing written


def test_eval_bands_override_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a per-label override changes only that label's bands (and the
    # effective target is visible in the artifact — provenance)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(
        tmp_path,
        bands='[bands."Money/Bills"]\nmin_high_precision = 0.5\n',
    )
    out = tmp_path / "bands.json"
    result, _factory = _run_eval(
        tmp_path, "--bands", str(out), monkeypatch=monkeypatch, cfg=cfg
    )
    assert result.exit_code == 0, result.stderr
    artifact = BandsArtifact.from_json(out.read_text(encoding="utf-8"))
    entries = {e.label: e for e in artifact.labels}
    # Money/Bills: min_high 0.5 → highest t with P >= 0.5 → 0.8 (0.9
    # undefined); med stays the default 0.7 → 0.7 (P(0.7) = 1.0)
    assert entries["Money/Bills"].min_high_precision == 0.5
    assert entries["Money/Bills"].high == 0.8
    assert entries["Money/Bills"].med == 0.7
    # Shipping: defaults unchanged
    assert entries["Shipping"].min_high_precision == 0.9
    assert entries["Shipping"].high == 0.9
    assert artifact.default_targets == BandTargets(0.9, 0.7)


def test_eval_bands_bodies_never_in_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # invariant #5: message bodies never enter the artifact (ids,
    # probabilities, and labels only — the DB-attachable discipline)
    import logging

    marker = "SECRET-BODY-MARKER"
    from gmail_labeler.corpus_record import CorpusRecord, body_hash
    from gmail_labeler.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    store.enforce_modes()
    now = datetime.now(UTC)
    for i in range(12):
        days_ago = 400 - i if i < 6 else 10 + i
        rec = CorpusRecord(
            id=f"{i:016x}",
            date=now - timedelta(days=days_ago),
            subject=f"subject {i}",
            sender="shop@x.z",
            recipients=("me@x.y",),
            labels=("Shipping",) if i % 2 == 0 else ("Money/Bills",),
            body_hash=body_hash(f"{marker} {i}"),
            source="api",
            date_provenance="internal_date",
        )
        store.put(rec, f"{marker} {i}")
    for name, payload in (("filters.json", "[]"), ("labels.json", "{}")):
        f = store.root / name
        f.write_text(payload, encoding="utf-8")
        os.chmod(f, 0o600)

    caplog.set_level(logging.DEBUG)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    scores = {f"subject {i}\n{marker} {i}": (0.5, 0.5) for i in range(6, 12)}
    out = tmp_path / "bands.json"
    result, _factory = _run_eval(
        tmp_path,
        "--bands",
        str(out),
        monkeypatch=monkeypatch,
        scores=scores,
    )
    assert result.exit_code == 0, result.stderr
    assert marker not in out.read_text(encoding="utf-8")
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert marker not in caplog.text


# ---- recompute posture (AC4): ONLY eval writes the artifact ----


def test_only_eval_writes_the_bands_artifact() -> None:
    # NAMED-MINT TRIPWIRE (not a sandbox — honesty pass): a string
    # scan cannot prove absence; getattr-construction and a
    # hand-rolled dict json.dump evade any source scan. What this pin
    # DOES: name the sanctioned mint/write symbols and fail if a
    # SECOND module in the package (any depth — the AGENTS target
    # layout grows subpackages such as daemon/) starts naming them.
    # The behavioural weight sits in
    # test_eval_without_bands_writes_nothing.
    import pathlib

    import gmail_labeler

    package_dir = pathlib.Path(gmail_labeler.__file__).parent
    # the artifact is MINTED through build_artifact: exactly two
    # modules name it — thresholds.py (the definition) and cli.py
    # (the eval command's call)
    minters = [
        py.name
        for py in sorted(package_dir.rglob("*.py"))
        if "build_artifact" in py.read_text(encoding="utf-8")
    ]
    assert minters == ["cli.py", "thresholds.py"]
    # and no module other than thresholds.py CONSTRUCTS the class
    # directly (a mint that bypassed build_artifact would skip its
    # validation)
    constructors = [
        py.name
        for py in sorted(package_dir.rglob("*.py"))
        if "BandsArtifact(" in py.read_text(encoding="utf-8")
    ]
    assert constructors == ["thresholds.py"]
    # and the 0600 artifact write helper (the secret-tight discipline)
    # is defined and called in exactly one module — cli.py (the
    # report --out and train's manifest/sidecar writes share it)
    writers = [
        py.name
        for py in sorted(package_dir.rglob("*.py"))
        if "_write_secret_tight" in py.read_text(encoding="utf-8")
    ]
    assert writers == ["cli.py"]
    # and the Bands Artifact write helper is named in exactly one
    # module — cli.py — and the --bands write path in cli.py goes
    # THROUGH it (a second band-file write path that bypassed the
    # helper would be a second writer, and this scan would name it)
    band_writers = [
        py.name
        for py in sorted(package_dir.rglob("*.py"))
        if "_write_bands_artifact" in py.read_text(encoding="utf-8")
    ]
    assert band_writers == ["cli.py"]
    assert "_write_bands_artifact(Path(bands)" in (package_dir / "cli.py").read_text(
        encoding="utf-8"
    )


def test_eval_bands_is_on_the_frozen_surface() -> None:
    # the FLAG SURFACE is asserted from click's parameter metadata, not
    # the rendered help (the width-wrapping prior art)
    from typer.core import TyperGroup
    from typer.main import get_command

    root = get_command(app)
    assert isinstance(root, TyperGroup)
    eval_cmd = root.commands["eval"]
    declared = {opt for p in eval_cmd.params for opt in getattr(p, "opts", [])}
    for flag in ("--config", "--assume-no-filters", "--out", "--json", "--bands"):
        assert flag in declared
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
