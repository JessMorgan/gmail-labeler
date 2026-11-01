"""eval CLI: sidecar-validated checkpoint → test-side metric table,
exit-code contract 0/2/3/4/5 (typer.testing prior art; the model
factory seam is monkeypatched to a FakeModel producer — the offline
spine, invariant #9).
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


def _eval_cfg(tmp_path: Path, *, encoder: str = "tiny-encoder") -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n'
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


def _run_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Result:
    import gmail_labeler.cli as cli

    monkeypatch.setattr(cli, "_model_factory", _train_factory())
    return runner.invoke(
        app,
        ["train", "--config", str(_eval_cfg(tmp_path)), "--split-date", SPLIT],
    )


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
) -> tuple[Result, object]:
    import gmail_labeler.cli as cli

    factory = _eval_factory(
        scores if scores is not None else _test_side_scores(), vocabulary
    )
    monkeypatch.setattr(cli, "_eval_model_factory", factory)
    result = runner.invoke(app, ["eval", "--config", str(_eval_cfg(tmp_path)), *extra])
    return result, factory


# ---- happy path ----


def test_eval_happy_path_golden_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC#3: deterministic, diff-friendly — per-label rows x
    # candidate-threshold columns, hand-computed golden
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    # hand-computed: Money/Bills support 3 (i=7,9,11), Shipping support 3
    # (i=6,8,10); e.g. Money/Bills t=0.5: TP=2 (0.8, 0.6), FN=1 (0.4)
    # → P=1.0 R=0.67; t=0.9: no TP → P=0.0 R=0.0. The provenance line's
    # manifest hash is DERIVED from the materialized manifest (its
    # weight_cutoff rides on `now`, so a hard-coded hash would go
    # stale) — the FORMAT is what the golden pins
    manifest_hash = json.loads(
        (tmp_path / "ckpt" / "manifest.json").read_text(encoding="utf-8")
    )["content_hash"]
    assert result.stdout == (
        "eval: trust filter: server-authoritative-only "
        "(inherited from the split manifest)\n"
        f"eval: split 2026-01-01T00:00:00+00:00 manifest={manifest_hash[:16]}\n"
        "eval: test examples=6 labels=2 thresholds=0.1..0.9\n"
        "label        support       0.1       0.2       0.3       0.4       "
        "0.5       0.6       0.7       0.8       0.9\n"
        "Money/Bills        3 0.50/1.00 0.60/1.00 0.75/1.00 1.00/1.00 "
        "1.00/0.67 1.00/0.67 1.00/0.33 1.00/0.33 0.00/0.00\n"
        "Shipping           3 0.50/1.00 0.60/1.00 0.75/1.00 1.00/1.00 "
        "1.00/0.67 1.00/0.67 1.00/0.67 1.00/0.33 1.00/0.33\n"
        "precision/recall render 0.00 where the denominator is zero; "
        "see tp/fp/fn in --json\n"
    )


def test_eval_scores_only_the_test_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC#2 (invariant #7): the factory sees ONLY the test-side texts,
    # and the rebuild goes THROUGH the minted manifest (no re-split)
    import gmail_labeler.dataset_build as db
    from gmail_labeler.dataset_build import SplitManifest

    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    # the spy is installed AFTER train: train's own two build calls
    # (mint + rebuild) must not pollute the eval-side assertion
    calls: list[SplitManifest | None] = []
    original = db.build_dataset

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(cast("SplitManifest | None", kwargs.get("manifest")))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(db, "build_dataset", spy)
    result, factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    seen = factory.seen  # type: ignore[attr-defined]
    # exactly the 6 test-side texts (i=6..11), in canonical id order
    assert seen["checkpoint_dir"] == tmp_path / "ckpt"
    assert len(calls) == 1
    assert calls[0] is not None  # rebuilt THROUGH the manifest
    assert "Money/Bills" in seen["vocabulary"]
    assert "Shipping" in seen["vocabulary"]
    # the report's support counts are test-side only: the corpus has 6
    # Money/Bills records total, only 3 on the test side
    assert "Money/Bills        3 " in result.stdout
    assert "Shipping           3 " in result.stdout


def test_eval_post_train_arrival_is_excluded_not_resplit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # BEHAVIOURAL invariant #7 (the append-only case): the corpus gains
    # one record AFTER minting whose date falls after split_date (a
    # post-train arrival — not in manifest.test_ids). Manifest mode
    # excludes records the manifest does not know: the arrival must NOT
    # enter the scored window. A re-split (manifest=None) would pick it
    # up (its date is at/after the split) and report test examples=7 —
    # this reds that mutant BEHAVIOURALLY, not just at the spy
    from gmail_labeler.corpus_record import CorpusRecord, body_hash
    from gmail_labeler.corpus_store import CorpusStore

    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    store = CorpusStore(tmp_path / "corpus")
    arrival = CorpusRecord(
        id="f" * 16,  # hex id, not in the manifest
        date=datetime.now(UTC) - timedelta(days=1),  # after the split
        subject="subject arrival",
        sender="shop@x.z",
        recipients=("me@x.y",),
        labels=("Shipping",),  # an existing label: no vocabulary drift
        body_hash=body_hash("body arrival"),
        source="api",
        date_provenance="internal_date",
    )
    store.put(arrival, "body arrival")
    # program the arrival's score too: a re-split mutant would score it
    # and report test examples=7 — the pin must red on the COUNT, not
    # on an unprogrammed-text refusal
    scores = _test_side_scores()
    scores["subject arrival\nbody arrival"] = (0.5, 0.5)
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch, scores=scores)
    assert result.exit_code == 0, result.stderr
    # the manifest's test side is still 6 — the arrival is excluded
    assert "test examples=6" in result.stdout
    assert "subject arrival" not in result.stdout


def test_eval_rebuilds_through_the_checkpoint_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the manifest materialized WITH the checkpoint is the split's
    # authority: the rebuild reads it from the checkpoint dir, not from
    # a fresh split (invariant #7)

    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    manifest = json.loads(
        (tmp_path / "ckpt" / "manifest.json").read_text(encoding="utf-8")
    )
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    # the report's test-example count is the manifest's test side
    assert f"test examples={len(manifest['test_ids'])}" in result.stdout


# ---- sidecar refusal (no silent fallback) ----


def test_eval_sidecar_mismatch_refused_exit_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC#4: a sidecar-mismatched checkpoint refuses to load — exit 5,
    # the disagreeing field named, NO report, NO silent fallback
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    sidecar_path = tmp_path / "ckpt" / "sidecar.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["encoder_id"] = "other-encoder"
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "encoder_id" in result.stderr
    assert "refused" in result.stderr
    assert result.stdout == ""  # no report, no silent fallback


def test_eval_sidecar_split_manifest_hash_mismatch_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the split-identity field: a sidecar whose split_manifest_hash does
    # not match the materialized manifest refuses (invariant #7)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    sidecar_path = tmp_path / "ckpt" / "sidecar.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["split_manifest_hash"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "split_manifest_hash" in result.stderr
    assert result.stdout == ""


def test_eval_tampered_manifest_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a tampered manifest.json (content hash no longer matches its
    # fields) refuses — the split identity is machine-checkable
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    manifest_path = tmp_path / "ckpt" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["test_ids"] = manifest["test_ids"][:-1]  # drop a member
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "manifest" in result.stderr
    assert result.stdout == ""


def test_eval_missing_checkpoint_exit_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # no checkpoint dir at all: the named missing-manifest refusal
    _seed_corpus(tmp_path)
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "has no manifest.json" in result.stderr
    assert result.stdout == ""


# ---- exit-code contract ----


def test_eval_config_error_exits_4(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "--config", str(tmp_path / "nope.toml")])
    assert result.exit_code == 4
    assert "config error" in result.stderr


def test_eval_missing_corpus_dir_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "missing or not a directory" in result.stderr


def test_eval_empty_corpus_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gmail_labeler.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    store.root.mkdir(parents=True)
    (store.root / "filters.json").write_text("[]", encoding="utf-8")
    (store.root / "labels.json").write_text("{}", encoding="utf-8")
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "corpus is empty" in result.stderr


def test_eval_missing_filters_snapshot_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # train runs WITH the snapshot (it needs the bridge too); the
    # snapshot is removed AFTER, so only eval sees the missing bridge
    root = _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    (root / "filters.json").unlink()
    (root / "labels.json").unlink()
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "build refused" in result.stderr and "invariant #6" in result.stderr


def test_eval_assume_no_filters_opens_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    (root / "filters.json").unlink()
    (root / "labels.json").unlink()
    result, _factory = _run_eval(
        tmp_path, "--assume-no-filters", monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr


def test_eval_bridge_incomplete_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    (root / "filters.json").write_text(
        '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": ["GONE"]}}]',
        encoding="utf-8",
    )
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "bridge" in result.stderr


def test_eval_empty_test_split_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # every record before the split: the manifest's test side is empty
    from gmail_labeler.corpus_record import CorpusRecord, body_hash
    from gmail_labeler.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    store.enforce_modes()
    now = datetime.now(UTC)
    for i in range(12):
        rec = CorpusRecord(
            id=f"{i:016x}",
            date=now - timedelta(days=400 - i),  # all before the split
            subject=f"subject {i}",
            sender="shop@x.z",
            recipients=("me@x.y",),
            labels=("Shipping",),
            body_hash=body_hash(f"body {i}"),
            source="api",
            date_provenance="internal_date",
        )
        store.put(rec, f"body {i}")
    for name, payload in (("filters.json", "[]"), ("labels.json", "{}")):
        f = store.root / name
        f.write_text(payload, encoding="utf-8")
        os.chmod(f, 0o600)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(
        tmp_path,
        monkeypatch=monkeypatch,
        # 1-element rows: the vocabulary is ("Shipping",) only
        scores={f"subject {i}\nbody {i}": (0.5,) for i in range(6, 12)},
    )
    assert result.exit_code == 5
    assert "no test-side examples" in result.stderr
    assert result.stdout == ""


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="the train extra is installed — the missing-extra path is untestable",
)
def test_eval_missing_train_extra_exits_5_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the REAL _eval_model_factory (no monkeypatch): the lazy import of
    # the recipe fails (torch is absent from the CI venv) → exit 5 with
    # an install hint, never a raw ModuleNotFoundError traceback (the
    # exit-code contract: 1 is reserved for unhandled crashes).
    # CI-only: skips where the train extra is installed.
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result = runner.invoke(app, ["eval", "--config", str(_eval_cfg(tmp_path))])
    assert result.exit_code == 5
    assert "extra" in result.stderr
    assert "Traceback" not in (result.stdout + result.stderr)


def test_eval_factory_failure_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    import gmail_labeler.cli as cli

    def boom(checkpoint_dir: Path, vocabulary: tuple[str, ...]):
        raise ValueError("encoder exploded")

    monkeypatch.setattr(cli, "_eval_model_factory", boom)
    result = runner.invoke(app, ["eval", "--config", str(_eval_cfg(tmp_path))])
    assert result.exit_code == 5
    assert "encoder exploded" in result.stderr


def test_eval_wrong_row_count_named_exit_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a model that returns a wrong row count gets a NAMED operator
    # message (a checkpoint/recipe defect), not raw zip internals
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    import gmail_labeler.cli as cli
    from gmail_labeler.model import FakeModel

    def short_factory(checkpoint_dir: Path, vocab: tuple[str, ...]):
        model = FakeModel(vocab, scores=_test_side_scores())
        original = model.predict

        def short_predict(texts):
            return original(texts)[:-1]  # one row short

        model.predict = short_predict
        return model

    monkeypatch.setattr(cli, "_eval_model_factory", short_factory)
    result = runner.invoke(app, ["eval", "--config", str(_eval_cfg(tmp_path))])
    assert result.exit_code == 5
    assert "score rows for" in result.stderr
    assert "6 texts" in result.stderr
    assert "zip" not in result.stderr  # no raw internals


def test_eval_vocabulary_mismatch_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a model whose vocabulary does not match the manifest's would serve
    # misaligned columns — refuse before scoring
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(
        tmp_path,
        monkeypatch=monkeypatch,
        vocabulary=("Other",),
        # aligned to the LYING vocabulary, so the refusal is the CLI's
        # vocabulary guard, not a FakeModel programming error
        scores={f"subject {i}\nbody {i}": (0.5,) for i in range(6, 12)},
    )
    assert result.exit_code == 5
    assert "does not match" in result.stderr
    assert result.stdout == ""


def test_eval_undecodable_body_shard_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _seed_corpus(tmp_path, n=4)
    # i=2 is test-side (n=4 → i>=2 test)
    shard = root / "bodies" / "00" / "0000000000000002.txt"
    shard.write_bytes(b"\xff\xfe not utf-8")
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(
        tmp_path,
        monkeypatch=monkeypatch,
        scores={
            "subject 2\nbody 2": (0.9, 0.1),
            "subject 3\nbody 3": (0.1, 0.9),
        },
    )
    assert result.exit_code == 5
    assert "body shard is unreadable" in result.stderr
    assert "Traceback" not in (result.stdout + result.stderr)


def test_eval_missing_body_shard_warns_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an orphaned shard (crash residue): scored on subject + trailing
    # newline (invariant #4 — the same contract train warns with)
    root = _seed_corpus(tmp_path)
    (root / "bodies" / "00" / "0000000000000006.txt").unlink()
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    # i=6's text is subject + trailing newline (the missing-shard
    # contract, invariant #4) — program it accordingly
    scores = _test_side_scores()
    scores["subject 6\n"] = scores.pop("subject 6\nbody 6")
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch, scores=scores)
    assert result.exit_code == 0, result.stderr
    assert "WARNING" in result.stderr and "1 test record" in result.stderr
    assert "test examples=6" in result.stdout


# ---- report output: --out / --json ----


def test_eval_out_file_byte_identical_and_secret_tight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the report carries the user's label taxonomy: the file is 0600
    # like every other artifact this repo writes (the corpus-sidecar
    # discipline), and byte-identical to stdout. The file is
    # PRE-CREATED 0644 on purpose: the assertion must exercise the
    # tighten-a-loose-file path, not just O_CREAT's mode. Replacing an
    # EXISTING file is announced on stderr (no silent clobber)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    out = tmp_path / "report.txt"
    out.write_text("old report", encoding="utf-8")
    os.chmod(out, 0o644)
    result, _factory = _run_eval(tmp_path, "--out", str(out), monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert "NOTICE: replacing existing" in result.stderr
    assert out.read_text(encoding="utf-8") == result.stdout  # same bytes
    assert out.stat().st_mode & 0o077 == 0  # 0600, never world-readable
    assert not (tmp_path / "report.txt.tmp").exists()  # no staging residue


def test_eval_out_into_missing_dir_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --out failure branch: an unwritable path is an operation
    # failure (exit 5), and the report still goes to stdout
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(
        tmp_path,
        "--out",
        str(tmp_path / "no" / "such" / "dir" / "r.txt"),
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 5
    assert "cannot write --out" in result.stderr
    assert "test examples=6" in result.stdout  # the report is not lost


def test_eval_json_deterministic_and_shaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --json: machine-readable, byte-identical across runs, the shape
    # the graduation workflow (#22) reads
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    r1, _f1 = _run_eval(tmp_path, "--json", monkeypatch=monkeypatch)
    r2, _f2 = _run_eval(tmp_path, "--json", monkeypatch=monkeypatch)
    assert r1.exit_code == 0, r1.stderr
    assert r1.stdout == r2.stdout  # byte-identical
    payload = json.loads(r1.stdout)
    assert payload["test_examples"] == 6
    assert payload["trust"] == "server-authoritative-only"
    # split identity: two reports from different splits must be
    # tellable apart by the consumer (#22)
    assert payload["split_date"] == "2026-01-01T00:00:00+00:00"
    assert (
        payload["split_manifest_hash"]
        == json.loads(
            (tmp_path / "ckpt" / "manifest.json").read_text(encoding="utf-8")
        )["content_hash"]
    )
    assert payload["thresholds"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert [row["label"] for row in payload["rows"]] == ["Money/Bills", "Shipping"]
    bills = payload["rows"][0]
    assert bills["support"] == 3
    assert len(bills["cells"]) == 9
    cell = bills["cells"][4]  # t=0.5: TP=2 FP=0 FN=1
    assert cell == {
        "threshold": 0.5,
        "tp": 2,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 2 / 3,
    }


def test_eval_json_out_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    out = tmp_path / "report.json"
    result, _factory = _run_eval(
        tmp_path, "--json", "--out", str(out), monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(result.stdout)


# ---- invariant #5: bodies never in the report or any log path ----


def test_eval_bodies_never_in_report_or_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # distinctive body content must not appear in stdout, stderr, or
    # any log record (invariant #5 — the report carries labels, counts,
    # and probabilities only)
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
    out = tmp_path / "report.txt"
    result, _factory = _run_eval(
        tmp_path,
        "--out",
        str(out),
        monkeypatch=monkeypatch,
        scores=scores,
    )
    assert result.exit_code == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert marker not in caplog.text
    # the --out FILE and the --json payload directly (not just
    # transitively via byte-identity / stdout)
    assert marker not in out.read_text(encoding="utf-8")
    r_json, _f_json = _run_eval(
        tmp_path, "--json", monkeypatch=monkeypatch, scores=scores
    )
    assert r_json.exit_code == 0, r_json.stderr
    assert marker not in r_json.stdout


# ---- surface ----


def test_eval_is_on_the_frozen_surface() -> None:
    # the COMMAND and FLAG SURFACE are asserted from click's parameter
    # metadata, not the rendered help: rich wraps long option names
    # differently per terminal width (a width-80 CI runner split
    # "--split-date" across lines and the train surface test went red
    # with the surface unchanged)
    from typer.core import TyperGroup
    from typer.main import get_command

    root = get_command(app)
    assert isinstance(root, TyperGroup)
    assert "eval" in root.commands  # the command is on the frozen surface
    eval_cmd = root.commands["eval"]
    declared = {opt for p in eval_cmd.params for opt in getattr(p, "opts", [])}
    for flag in ("--config", "--assume-no-filters", "--out", "--json"):
        assert flag in declared
    # no --split-date: the split is pinned in the manifest (invariant #7)
    assert "--split-date" not in declared
    # and the command still renders help successfully (exit 0 covers the
    # rendering itself; content checks stay width-independent above)
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
