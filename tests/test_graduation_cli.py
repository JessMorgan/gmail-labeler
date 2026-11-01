"""eval --graduation CLI (ticket #22): the Graduation Table — opt-in
human-confirm diff, CLI-gated write (--confirm-graduation),
checkpoint/eval/bands-bound identity, recompute-posture pins (ONLY the
--confirm path writes the table), exit-code contract 0/2/3/4/5 (typer.
testing prior art; the model factory seam is monkeypatched to a
FakeModel producer — the offline spine, invariant #9).
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
from gmail_labeler.graduation import GraduationTable
from gmail_labeler.thresholds import BandsArtifact

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
    tmp_path: Path,
    *,
    encoder: str = "tiny-encoder",
    graduation: str = "",
    bands: str = "",
    labels: str = 'managed = ["Money/Bills", "Shipping"]',
) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n'
        f"[labels]\n{labels}\n"
        f"[bands]\n{bands}\n"
        f"[graduation]\n{graduation}\n"
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


# ---- the golden gates (hand-computed, independent of the code) ----
#
# Money/Bills (support 3): bands high = 0.8 (P = 1.0), med = 0.7
# Shipping (support 3): bands high = 0.9 (P = 1.0), med = 0.8
#
# min_examples = 3: both labels graduate (3 >= 3 AT the gate —
# analyze's strict-gt precedent; precision 1.0 >= 0.9)
# min_examples = 4: both labels fail (below-min-examples)


def test_eval_graduation_confirm_writes_bound_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC2/AC3: the table is bound to the checkpoint + eval identity +
    # the bands artifact it was computed from, and carries the measured
    # gates (hand-computed golden)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 0, result.stderr
    assert out.is_file()
    assert out.stat().st_mode & 0o077 == 0  # 0600, never world-readable
    assert not (tmp_path / "graduation.json.tmp").exists()  # no staging residue
    table = GraduationTable.from_json(out.read_text(encoding="utf-8"))
    # checkpoint identity (mirrored from the sidecar)
    sidecar = json.loads((tmp_path / "ckpt" / "sidecar.json").read_text())
    manifest = json.loads((tmp_path / "ckpt" / "manifest.json").read_text())
    assert table.vocabulary_hash == sidecar["vocabulary_hash"]
    assert table.corpus_fingerprint == sidecar["corpus_fingerprint"]
    assert table.split_manifest_hash == sidecar["split_manifest_hash"]
    assert table.split_manifest_hash == manifest["content_hash"]
    assert table.encoder_id == sidecar["encoder_id"]
    # eval identity (the measured window)
    assert table.split_date == datetime.fromisoformat(manifest["split_date"])
    assert table.test_examples == len(manifest["test_ids"]) == 6
    # the bands binding (AC3): the content hash of the Bands Artifact
    # the gates were computed from — recomputed independently via
    # eval --bands on the same run
    bands_out = tmp_path / "bands.json"
    r2, _f2 = _run_eval(
        tmp_path, "--bands", str(bands_out), monkeypatch=monkeypatch, cfg=cfg
    )
    assert r2.exit_code == 0, r2.stderr
    artifact = BandsArtifact.from_json(bands_out.read_text(encoding="utf-8"))
    assert table.bands_content_hash == artifact.content_hash
    # provenance
    assert table.candidate_thresholds == (
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
    # the measured gates (the golden)
    entries = {e.label: e for e in table.labels}
    assert [e.label for e in table.labels] == ["Money/Bills", "Shipping"]
    assert entries["Money/Bills"].graduated is True
    assert entries["Money/Bills"].examples == 3
    assert entries["Money/Bills"].high == 0.8
    assert entries["Money/Bills"].med == 0.7
    assert entries["Money/Bills"].precision_at_high == pytest.approx(1.0)
    assert entries["Money/Bills"].min_examples == 3
    assert entries["Money/Bills"].reasons == ()
    assert entries["Shipping"].graduated is True
    assert entries["Shipping"].high == 0.9
    assert entries["Shipping"].precision_at_high == pytest.approx(1.0)
    # the report is on stdout; the human-confirm diff is on STDERR
    # (the --out "same bytes as stdout" contract — the diff is status,
    # not report content)
    assert "test examples=6" in result.stdout
    assert "graduation: Money/Bills: (absent) -> graduated" in result.stderr
    assert "graduation: Shipping: (absent) -> graduated" in result.stderr


def test_eval_graduation_without_confirm_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC2: the human-confirm diff is the default posture — recomputation
    # shows old → new per label, NOTHING applies itself; the write is
    # the explicit --confirm-graduation act
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path, "--graduation", str(out), monkeypatch=monkeypatch, cfg=cfg
    )
    assert result.exit_code == 0, result.stderr
    assert not out.exists()  # nothing written
    assert not list(tmp_path.glob("graduation.json*"))  # no staging residue
    assert "test examples=6" in result.stdout  # the report is never lost
    assert "graduation: Money/Bills: (absent) -> graduated" in result.stderr
    assert "no existing table at" in result.stderr  # the baseline


def test_eval_graduation_byte_identical_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC5: same eval → byte-identical table (deterministic total order —
    # no clock, no I/O nondeterminism in the core)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out1, out2 = tmp_path / "g1.json", tmp_path / "g2.json"
    r1, _f1 = _run_eval(
        tmp_path,
        "--graduation",
        str(out1),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    r2, _f2 = _run_eval(
        tmp_path,
        "--graduation",
        str(out2),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert out1.read_bytes() == out2.read_bytes()


def test_eval_graduation_diff_shows_old_to_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC2: recomputation shows old → new per label — a gate change that
    # demotes a label is visible in the diff, and the file only moves
    # on --confirm
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg_loose = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    r1, _f1 = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg_loose,
    )
    assert r1.exit_code == 0, r1.stderr
    bytes_graduated = out.read_bytes()
    # tighten the gate: both labels (support 3) now fail
    cfg_strict = _eval_cfg(tmp_path, graduation="min_examples = 4\n")
    r2, _f2 = _run_eval(
        tmp_path, "--graduation", str(out), monkeypatch=monkeypatch, cfg=cfg_strict
    )
    assert r2.exit_code == 0, r2.stderr
    assert (
        "Money/Bills: graduated (examples=3, precision@high=1.0000, "
        "min_examples=3, min_precision=0.9000) -> not graduated "
        "(below-min-examples; examples=3, precision@high=1.0000, "
        "min_examples=4, min_precision=0.9000)" in r2.stderr
    )
    assert (
        "Shipping: graduated (examples=3, precision@high=1.0000, "
        "min_examples=3, min_precision=0.9000) -> not graduated "
        "(below-min-examples; examples=3, precision@high=1.0000, "
        "min_examples=4, min_precision=0.9000)" in r2.stderr
    )
    assert out.read_bytes() == bytes_graduated  # nothing applied itself
    # confirm: the file moves to the demoted table
    r3, _f3 = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg_strict,
    )
    assert r3.exit_code == 0, r3.stderr
    table = GraduationTable.from_json(out.read_text(encoding="utf-8"))
    assert all(not e.graduated for e in table.labels)
    assert all(e.reasons == ("below-min-examples",) for e in table.labels)


def test_eval_graduation_identity_drift_refuses_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC3: an eval of checkpoint B over a target holding checkpoint A's
    # table must REFUSE the confirm (exit 5, before any staging) — a
    # clobber would destroy a verifiably-A table. The diff says so
    # explicitly (the disagreeing fields named). A's bytes stay
    # byte-identical.
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0  # checkpoint A
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    r1, _f1 = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert r1.exit_code == 0, r1.stderr
    bytes_a = out.read_bytes()
    # checkpoint B: same corpus, a LATER split date — the manifest's
    # split identity (and the table's split_date) differs
    assert (
        _run_train(tmp_path, monkeypatch, split_date="2026-02-01", force=True).exit_code
        == 0
    )
    # without --confirm: the diff names the foreign-bound table
    r2, _f2 = _run_eval(
        tmp_path, "--graduation", str(out), monkeypatch=monkeypatch, cfg=cfg
    )
    assert r2.exit_code == 0, r2.stderr
    assert "bound to a different identity" in r2.stderr
    assert "split_date" in r2.stderr  # the named disagreeing field
    assert out.read_bytes() == bytes_a  # nothing applied
    # with --confirm: refused, exit 5, A's bytes intact
    r3, _f3 = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert r3.exit_code == 5
    assert "--graduation refused" in r3.stderr
    assert "split_date" in r3.stderr  # the named disagreeing field
    assert out.read_bytes() == bytes_a  # A's bytes byte-identical


def test_eval_graduation_degraded_corpus_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # serving-gate posture (shared with --bands): a corpus with missing
    # body shards is scored on subject + trailing newline only —
    # degraded scores must not mint the table that gates what goes
    # live. Refusal: exit 5 (the operation-failure convention the other
    # post-report eval refusals use), the report is preserved on
    # stdout, and NO table (or staging) is written anywhere.
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    # orphan one TEST-side body shard (i=6 → id 0000000000000006)
    shard = tmp_path / "corpus" / "bodies" / "00" / "0000000000000006.txt"
    assert shard.is_file()
    shard.unlink()
    # the degraded text for i=6 is subject + trailing newline
    scores = dict(_test_side_scores())
    scores["subject 6\n"] = scores.pop("subject 6\nbody 6")
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        scores=scores,
        cfg=cfg,
    )
    assert result.exit_code == 5
    assert "--graduation refused" in result.stderr
    assert "1 test record(s) had no body shard" in result.stderr
    assert "test examples=6" in result.stdout  # the report is not lost
    assert not out.exists()  # no table
    assert not list(tmp_path.glob("graduation.json*"))  # no staging residue


def test_eval_graduation_json_refusal_preserves_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json report-preservation doctrine: a post-report refusal
    # (here: the degraded corpus) exits 5 with the report payload on
    # stdout — exactly one parseable JSON object, the proposal simply
    # absent (the computation never ran)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    # orphan one TEST-side body shard (i=6 → id 0000000000000006)
    shard = tmp_path / "corpus" / "bodies" / "00" / "0000000000000006.txt"
    assert shard.is_file()
    shard.unlink()
    scores = dict(_test_side_scores())
    scores["subject 6\n"] = scores.pop("subject 6\nbody 6")
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        "--json",
        monkeypatch=monkeypatch,
        scores=scores,
        cfg=cfg,
    )
    assert result.exit_code == 5
    payload = json.loads(result.stdout)  # the report is on stdout
    assert payload["test_examples"] == 6
    assert "graduation" not in payload  # the proposal is absent
    assert "--graduation refused" in result.stderr
    assert not out.exists()


def test_eval_graduation_json_write_failure_preserves_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json report-preservation doctrine on a WRITE failure: the
    # payload (with the proposal — the computation succeeded) is on
    # stdout, exit 5
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(tmp_path / "no" / "such" / "dir" / "graduation.json"),
        "--confirm-graduation",
        "--json",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    payload = json.loads(result.stdout)  # the report is on stdout
    assert payload["test_examples"] == 6
    assert payload["graduation"]["test_examples"] == 6  # the proposal rides
    assert "cannot write --graduation" in result.stderr


def test_eval_graduation_json_calibration_refusal_preserves_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json report-preservation doctrine on a calibration refusal
    # (an unknown band override label): exit 5, the base payload on
    # stdout (the report), the proposal absent (the computation never
    # ran)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(
        tmp_path,
        graduation="min_examples = 3\n",
        bands='[bands."Ghost"]\nmin_high_precision = 0.9\n',
    )
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(tmp_path / "graduation.json"),
        "--confirm-graduation",
        "--json",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    payload = json.loads(result.stdout)  # the report is on stdout
    assert payload["test_examples"] == 6
    assert "graduation" not in payload
    assert "band calibration refused" in result.stderr
    assert "Ghost" in result.stderr


def test_eval_graduation_json_bands_refused_preserves_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json report-preservation doctrine on a --bands refusal
    # (an existing bands artifact bound to a different identity): exit
    # 5, the base payload on stdout
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0  # checkpoint A
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    bands_out = tmp_path / "bands.json"
    r1, _f1 = _run_eval(
        tmp_path, "--bands", str(bands_out), monkeypatch=monkeypatch, cfg=cfg
    )
    assert r1.exit_code == 0, r1.stderr
    # checkpoint B: a LATER split date — the bands identity differs
    assert (
        _run_train(tmp_path, monkeypatch, split_date="2026-02-01", force=True).exit_code
        == 0
    )
    result, _factory = _run_eval(
        tmp_path,
        "--bands",
        str(bands_out),
        "--graduation",
        str(tmp_path / "graduation.json"),
        "--confirm-graduation",
        "--json",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    payload = json.loads(result.stdout)  # the report is on stdout
    assert payload["test_examples"] == 6
    assert "graduation" not in payload
    assert "--bands refused" in result.stderr


def test_eval_graduation_json_bands_write_failure_preserves_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json report-preservation doctrine on a --bands write
    # failure: exit 5, the base payload on stdout
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    result, _factory = _run_eval(
        tmp_path,
        "--bands",
        str(tmp_path / "no" / "such" / "dir" / "bands.json"),
        "--graduation",
        str(tmp_path / "graduation.json"),
        "--confirm-graduation",
        "--json",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    payload = json.loads(result.stdout)  # the report is on stdout
    assert payload["test_examples"] == 6
    assert "graduation" not in payload
    assert "cannot write --bands" in result.stderr


def test_eval_graduation_json_gate_refusal_preserves_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json report-preservation doctrine on a gate evaluation
    # refusal (a Managed Label the vocabulary does not carry): exit 5,
    # the base payload on stdout, the proposal absent
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(
        tmp_path,
        graduation="min_examples = 3\n",
        labels='managed = ["Money/Bills", "Ghost"]',
    )
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(tmp_path / "graduation.json"),
        "--confirm-graduation",
        "--json",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    payload = json.loads(result.stdout)  # the report is on stdout
    assert payload["test_examples"] == 6
    assert "graduation" not in payload
    assert "graduation gate evaluation refused" in result.stderr
    assert "Ghost" in result.stderr


def test_eval_graduation_json_refused_confirm_preserves_report_with_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json report-preservation doctrine on a foreign-bound
    # confirm refusal: exit 5, the payload on stdout CARRIES THE
    # PROPOSAL (the computation succeeded — only the write refused)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0  # checkpoint A
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    r1, _f1 = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert r1.exit_code == 0, r1.stderr
    # checkpoint B: a LATER split date — the table identity differs
    assert (
        _run_train(tmp_path, monkeypatch, split_date="2026-02-01", force=True).exit_code
        == 0
    )
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        "--json",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    payload = json.loads(result.stdout)  # the report is on stdout
    assert payload["test_examples"] == 6
    assert payload["graduation"]["test_examples"] == 6  # the proposal rides
    assert "--graduation refused" in result.stderr


def test_eval_out_empty_path_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an empty --out would resolve to the cwd (and write a stray
    # staging file there) — a usage error (exit 2) up front, before
    # any corpus work; BOTH spellings are pinned (the literal "" and
    # whitespace)
    import os

    cwd_files = set(os.listdir(Path.cwd()))
    for empty in ("", " "):
        result, _factory = _run_eval(tmp_path, "--out", empty, monkeypatch=monkeypatch)
        assert result.exit_code == 2
        assert "non-empty" in result.stderr
        assert "test examples=6" not in result.stdout  # pre-report refusal
    assert set(os.listdir(Path.cwd())) == cwd_files  # no stray file in cwd


def test_eval_out_inside_corpus_store_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F1: --out naming a file inside the corpus store would clobber
    # identity material (labels.json) — refused up front (exit 2),
    # the bytes intact (the ds live probe at 0534077: this exited 0
    # and NOTICE-replaced the file)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    labels = tmp_path / "corpus" / "labels.json"
    bytes_before = labels.read_bytes()
    result, _factory = _run_eval(
        tmp_path, "--out", str(labels), monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "corpus store" in result.stderr
    assert "test examples=6" not in result.stdout  # pre-report refusal
    assert labels.read_bytes() == bytes_before  # bytes intact


def test_eval_out_inside_checkpoint_dir_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F1: --out naming a file inside the checkpoint dir (the split
    # manifest) — refused up front, the bytes intact
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    manifest = tmp_path / "ckpt" / "manifest.json"
    bytes_before = manifest.read_bytes()
    result, _factory = _run_eval(
        tmp_path, "--out", str(manifest), monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "checkpoint" in result.stderr
    assert "test examples=6" not in result.stdout  # pre-report refusal
    assert manifest.read_bytes() == bytes_before  # bytes intact


def test_eval_out_symlink_into_corpus_store_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F1: a symlink whose NAME sits outside the corpus store but
    # POINTS inside it — both sides are resolved, so the guard catches
    # it (an unresolved check would let the clobber through)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    labels = tmp_path / "corpus" / "labels.json"
    bytes_before = labels.read_bytes()
    link = tmp_path / "report-link.json"
    os.symlink(labels, link)
    result, _factory = _run_eval(tmp_path, "--out", str(link), monkeypatch=monkeypatch)
    assert result.exit_code == 2
    assert "corpus store" in result.stderr
    assert labels.read_bytes() == bytes_before  # the target intact


def test_eval_out_benign_target_still_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F1: the containment refusal does not over-refuse — a benign
    # target (outside the checkpoint dir and the corpus store) still
    # writes the report
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    report = tmp_path / "report.txt"
    result, _factory = _run_eval(
        tmp_path, "--out", str(report), monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr
    assert report.is_file()
    assert "test examples=6" in report.read_text(encoding="utf-8")


def test_eval_out_same_path_as_config_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --out naming the config file would destroy the identity material
    # this very run just read (the same exact-path refusal as the
    # artifact targets) — exit 2 up front, the config intact
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path)
    result, _factory = _run_eval(
        tmp_path, "--out", str(cfg), monkeypatch=monkeypatch, cfg=cfg
    )
    assert result.exit_code == 2
    assert "config file" in result.stderr
    assert cfg.is_file() and "checkpoint_dir" in cfg.read_text()


def test_eval_graduation_sibling_bands_target_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # type-confusion honesty: a target that parses as the SIBLING
    # artifact (a Bands Artifact) is a KNOWN artifact, not garbage —
    # deleting it behind the NOTICE would destroy identity material;
    # the write refuses, naming the confusion (exit 5, pre-staging)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    target = tmp_path / "artifact.json"
    r1, _f1 = _run_eval(
        tmp_path, "--bands", str(target), monkeypatch=monkeypatch, cfg=cfg
    )
    assert r1.exit_code == 0, r1.stderr
    bytes_a = target.read_bytes()
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(target),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    assert "is a Bands Artifact, not a Graduation Table" in result.stderr
    assert "refusing to delete it" in result.stderr
    assert target.read_bytes() == bytes_a  # the bands artifact intact


def test_eval_bands_sibling_graduation_target_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the symmetric case: a target that parses as a Graduation Table
    # is refused by the --bands write, naming the confusion
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    target = tmp_path / "artifact.json"
    r1, _f1 = _run_eval(
        tmp_path,
        "--graduation",
        str(target),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert r1.exit_code == 0, r1.stderr
    bytes_a = target.read_bytes()
    result, _factory = _run_eval(
        tmp_path, "--bands", str(target), monkeypatch=monkeypatch, cfg=cfg
    )
    assert result.exit_code == 5
    assert "is a Graduation Table, not a Bands Artifact" in result.stderr
    assert "refusing to delete it" in result.stderr
    assert target.read_bytes() == bytes_a  # the table intact


def test_eval_graduation_refusal_names_partial_bands_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # partial-effect honesty: when --confirm-graduation refuses AFTER a
    # successful --bands write, the exit-5 message names the partial
    # effect. The scenario: a [bands] override changes the bands
    # CONTENT (the bands_content_hash identity field) while the six
    # eval identity fields stay the same — the bands write succeeds
    # (NOTICE replace), the graduation table is foreign-bound
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg1 = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    bands_out = tmp_path / "bands.json"
    grad_out = tmp_path / "graduation.json"
    r1, _f1 = _run_eval(
        tmp_path,
        "--bands",
        str(bands_out),
        "--graduation",
        str(grad_out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg1,
    )
    assert r1.exit_code == 0, r1.stderr
    cfg2 = _eval_cfg(
        tmp_path,
        graduation="min_examples = 3\n",
        bands='[bands."Shipping"]\nmin_high_precision = 0.5\n',
    )
    r2, _f2 = _run_eval(
        tmp_path,
        "--bands",
        str(bands_out),
        "--graduation",
        str(grad_out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg2,
    )
    assert r2.exit_code == 5
    assert "--graduation refused" in r2.stderr
    assert "bands_content_hash" in r2.stderr  # the named disagreeing field
    assert "the --bands artifact was already written" in r2.stderr  # the partial effect


def test_eval_graduation_gate_refusal_does_not_publish_bands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the reorder: evaluate_graduation (pure) runs BEFORE the --bands
    # write, so a gate evaluation refusal can no longer leave the bands
    # artifact published (the partial-effect class is closed at the
    # source)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(
        tmp_path,
        graduation="min_examples = 3\n",
        labels='managed = ["Money/Bills", "Ghost"]',
    )
    bands_out = tmp_path / "bands.json"
    grad_out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--bands",
        str(bands_out),
        "--graduation",
        str(grad_out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    assert "graduation gate evaluation refused" in result.stderr
    assert "Ghost" in result.stderr
    assert not bands_out.exists()  # the bands artifact was NOT published
    assert not grad_out.exists()


def test_eval_graduation_unparseable_existing_named_in_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # diff honesty: an existing target that does not parse as a
    # Graduation Table is named as such — NOT "no existing table"
    # (the file exists, it is just not a table)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    out.write_text("not a table", encoding="utf-8")
    result, _factory = _run_eval(
        tmp_path, "--graduation", str(out), monkeypatch=monkeypatch, cfg=cfg
    )
    assert result.exit_code == 0, result.stderr
    assert (
        "existing file at" in result.stderr
        and "does not parse as a Graduation Table" in result.stderr
    )
    assert "no existing table" not in result.stderr
    assert "NOTICE" not in result.stderr  # no replace NOTICE on a read


def test_eval_graduation_unreadable_existing_named_in_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F6: an UNREADABLE existing target (I/O failure) is named as
    # unreadable — NOT "does not parse" (the cause is not
    # misdiagnosed); the diff still renders on the baseline
    import os

    if os.geteuid() == 0:
        pytest.skip("running as root — chmod 000 cannot deny reads")
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    out.write_text("not a table", encoding="utf-8")
    os.chmod(out, 0o000)
    try:
        result, _factory = _run_eval(
            tmp_path, "--graduation", str(out), monkeypatch=monkeypatch, cfg=cfg
        )
    finally:
        os.chmod(out, 0o600)  # restore for cleanup
    assert result.exit_code == 0, result.stderr
    assert "is unreadable" in result.stderr
    assert "does not parse" not in result.stderr
    assert "no existing table" not in result.stderr


def test_eval_graduation_json_out_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --json + --out + refusal: the --out file lands at the payload
    # emit (AFTER the serving-gate acts, in json mode); on a write
    # failure the proposal rides in the payload (the computation
    # succeeded); stdout is exactly one parseable JSON object; exit
    # 5; no staging residue
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    report = tmp_path / "report.json"
    result, _factory = _run_eval(
        tmp_path,
        "--out",
        str(report),
        "--graduation",
        str(tmp_path / "no" / "such" / "dir" / "graduation.json"),
        "--confirm-graduation",
        "--json",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    assert "cannot write --graduation" in result.stderr
    payload = json.loads(result.stdout)  # one parseable JSON object
    assert payload["test_examples"] == 6
    # on a WRITE failure the proposal rides (the computation
    # succeeded) — the same proposal in the --out file
    assert payload["graduation"]["test_examples"] == 6
    assert report.is_file()  # the --out file landed at the emit
    file_payload = json.loads(report.read_text(encoding="utf-8"))
    assert file_payload["test_examples"] == 6
    assert file_payload["graduation"]["test_examples"] == 6
    assert list(tmp_path.glob("report.json.*")) == []  # no staging residue


def test_eval_graduation_out_file_lands_before_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --out ordering doctrine: the report file lands BEFORE the
    # serving-gate acts — a refusal exits 5 with the --out file in
    # place (the "same bytes as stdout" promise holds even when an
    # artifact act refuses)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    report = tmp_path / "report.txt"
    result, _factory = _run_eval(
        tmp_path,
        "--out",
        str(report),
        "--graduation",
        str(tmp_path / "no" / "such" / "dir" / "graduation.json"),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    assert "cannot write --graduation" in result.stderr
    assert report.is_file()  # the report file is in place
    assert report.read_text(encoding="utf-8") == result.stdout


def test_eval_graduation_with_json_stdout_is_single_json_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json path: stdout is EXACTLY one parseable JSON object (the
    # diff lines stay off stdout — json.loads over the whole stdout must
    # succeed), the proposal rides under the "graduation" key, and the
    # table is still written on --confirm
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        "--json",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)  # the whole stdout is one object
    assert payload["test_examples"] == 6
    assert [row["label"] for row in payload["rows"]] == [
        "Money/Bills",
        "Shipping",
    ]
    proposal = payload["graduation"]
    assert proposal["test_examples"] == 6
    assert [e["label"] for e in proposal["labels"]] == ["Money/Bills", "Shipping"]
    assert all(e["graduated"] for e in proposal["labels"])
    assert "graduation: " not in result.stdout  # no diff lines on stdout
    assert out.is_file()
    table = GraduationTable.from_json(out.read_text(encoding="utf-8"))
    assert table.labels  # validates


def test_eval_graduation_json_without_confirm_carries_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --json without --confirm: the proposal is in the payload (the
    # machine-readable diff surface), nothing is written
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--json",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)  # the whole stdout is one object
    assert payload["graduation"]["bands_content_hash"]  # the binding rides
    assert not out.exists()


def test_eval_graduation_empty_path_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an empty --graduation would degenerate to Path("") and fail
    # awkwardly mid-run — a usage error (exit 2) up front
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(tmp_path, "--graduation", " ", monkeypatch=monkeypatch)
    assert result.exit_code == 2
    assert "non-empty" in result.stderr


def test_confirm_graduation_without_graduation_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --confirm-graduation requires --graduation — a usage error (exit
    # 2) up front, before any corpus work
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(
        tmp_path, "--confirm-graduation", monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "requires --graduation" in result.stderr
    assert "test examples=6" not in result.stdout  # pre-report refusal


def test_eval_graduation_same_path_as_out_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the report and the table must not share a path: the table write
    # would silently destroy the report (exit 2 up front)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    same = tmp_path / "one.json"
    result, _factory = _run_eval(
        tmp_path,
        "--out",
        str(same),
        "--graduation",
        str(same),
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 2
    assert "different files" in result.stderr
    assert not same.exists()


def test_eval_graduation_same_path_as_bands_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the two artifacts must not share a path: the second write would
    # silently destroy the first (exit 2 up front)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    same = tmp_path / "one.json"
    result, _factory = _run_eval(
        tmp_path,
        "--bands",
        str(same),
        "--graduation",
        str(same),
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 2
    assert "different files" in result.stderr
    assert not same.exists()


def test_eval_graduation_empty_path_beats_equal_path_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an empty target resolves to the cwd — the empty checks run
    # BEFORE the equal-path checks, so the message is "non-empty", not
    # a spurious "different files" collision
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(
        tmp_path, "--bands", " ", "--graduation", " ", monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "non-empty" in result.stderr


def test_eval_graduation_inside_checkpoint_dir_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the generalized identity-file guard: ANY target inside the
    # checkpoint dir is refused — the split manifest is identity
    # material the garbage-tolerance NOTICE branch cannot see
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    manifest = tmp_path / "ckpt" / "manifest.json"
    bytes_before = manifest.read_bytes()
    result, _factory = _run_eval(
        tmp_path, "--graduation", str(manifest), monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "checkpoint" in result.stderr
    assert "test examples=6" not in result.stdout  # pre-report refusal
    assert manifest.read_bytes() == bytes_before  # bytes intact


def test_eval_graduation_inside_corpus_store_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the store's identity files (labels.json, filters.json,
    # records.jsonl, bodies) are equally destroyable — refused
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    labels = tmp_path / "corpus" / "labels.json"
    bytes_before = labels.read_bytes()
    result, _factory = _run_eval(
        tmp_path, "--graduation", str(labels), monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "corpus store" in result.stderr
    assert labels.read_bytes() == bytes_before  # bytes intact


def test_eval_graduation_symlink_into_checkpoint_dir_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a symlink whose NAME sits outside the checkpoint dir but POINTS
    # inside it: both sides are resolved, so the guard catches it (an
    # unresolved check would let the clobber through)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    sidecar = tmp_path / "ckpt" / "sidecar.json"
    bytes_before = sidecar.read_bytes()
    link = tmp_path / "graduation-link.json"
    os.symlink(sidecar, link)
    result, _factory = _run_eval(
        tmp_path, "--graduation", str(link), monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "checkpoint" in result.stderr
    assert sidecar.read_bytes() == bytes_before  # the target intact


def test_eval_graduation_same_path_as_config_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a destructive collision: the table write would destroy the config
    # file (exit 2 up front, no corpus work)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path)
    result, _factory = _run_eval(
        tmp_path, "--graduation", str(cfg), monkeypatch=monkeypatch, cfg=cfg
    )
    assert result.exit_code == 2
    assert "config file" in result.stderr
    assert cfg.is_file() and "checkpoint_dir" in cfg.read_text()


def test_eval_graduation_existing_directory_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --graduation naming an existing directory: refused UP FRONT as a
    # usage error (exit 2, pre-report)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    d = tmp_path / "graduation-dir"
    d.mkdir()
    result, _factory = _run_eval(
        tmp_path, "--graduation", str(d), monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "directory" in result.stderr
    assert "test examples=6" not in result.stdout  # pre-report refusal
    assert not any(d.iterdir())  # nothing written


def test_eval_graduation_unknown_managed_label_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a Managed Label the checkpoint's vocabulary does not carry is
    # refused, named — fail loud, never a silent no-op
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(
        tmp_path,
        graduation="min_examples = 3\n",
        labels='managed = ["Money/Bills", "Ghost"]',
    )
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    assert "Ghost" in result.stderr
    assert "test examples=6" in result.stdout  # the report is never lost
    assert not out.exists()  # no partial table


def test_eval_graduation_config_out_of_range_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an out-of-range gate value is a CONFIG error — exit 4, before any
    # corpus work
    _seed_corpus(tmp_path)
    cfg = _eval_cfg(tmp_path, graduation="min_precision = 1.5\n")
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(tmp_path / "graduation.json"),
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 4
    assert "config error" in result.stderr
    assert "must be in [0.0, 1.0]" in result.stderr


def test_eval_graduation_write_failure_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --graduation failure branch: an unwritable path is an
    # operation failure (exit 5), and the report still goes to stdout
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(tmp_path / "no" / "such" / "dir" / "graduation.json"),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    assert "cannot write --graduation" in result.stderr
    assert "test examples=6" in result.stdout  # the report is not lost


def test_eval_graduation_readback_is_hash_based(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # readback equality is by CONTENT HASH, not dataclass equality:
    # the canonical payload serializes reasons sorted, so a
    # reasons-order difference is the same meaning — a dataclass
    # comparison would exit 5 after a perfectly good table landed
    import gmail_labeler.cli as cli
    from gmail_labeler.dataset_build import TrainingExample
    from gmail_labeler.eval import eval_metrics
    from gmail_labeler.graduation import build_table, evaluate_graduation
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
    bands = build_artifact(
        calibrate_bands(table),
        vocabulary=("Bills", "Shipping"),
        corpus_fingerprint="f" * 64,
        split_manifest_hash="0" * 64,
        encoder_id="tiny-encoder",
        split_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    grad = evaluate_graduation(table, bands, managed_labels=("Bills", "Shipping"))
    artifact = build_table(
        grad,
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
        GraduationTable,
        "from_json",
        classmethod(lambda cls, text: readback),
    )
    out = tmp_path / "graduation.json"
    cli._write_graduation_table(out, artifact, artifact.to_json())  # no refusal
    assert out.is_file()


def test_eval_graduation_readback_fault_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the REJECTION path: a readback fault (from_json refuses the
    # on-disk bytes — a torn/corrupt write) is a named exit-5 operation
    # failure, never an uncaught ValueError traceback (exit 1), and the
    # report is preserved
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0

    def _boom(cls, text: str) -> GraduationTable:
        raise ValueError("graduation field invalid: simulated torn write")

    monkeypatch.setattr(GraduationTable, "from_json", classmethod(_boom))
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    assert "--graduation write failed AFTER the target was replaced" in result.stderr
    assert "readback fault" in result.stderr
    assert "test examples=6" in result.stdout  # the report is not lost


def test_eval_graduation_readback_mismatch_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the true mismatch: the readback parses but its content_hash
    # differs from the minted table — exit 5, named
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    from gmail_labeler.dataset_build import TrainingExample
    from gmail_labeler.eval import eval_metrics
    from gmail_labeler.graduation import build_table, evaluate_graduation
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
    bands = build_artifact(
        calibrate_bands(table),
        vocabulary=("Bills", "Shipping"),
        corpus_fingerprint="f" * 64,
        split_manifest_hash="0" * 64,
        encoder_id="tiny-encoder",
        split_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    other = build_table(
        evaluate_graduation(table, bands, managed_labels=("Bills", "Shipping")),
        vocabulary=("Bills", "Shipping"),
        corpus_fingerprint="f" * 64,
        split_manifest_hash="0" * 64,
        encoder_id="tiny-encoder",
        split_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    lying = dataclasses.replace(other, content_hash="1" * 64)
    monkeypatch.setattr(
        GraduationTable,
        "from_json",
        classmethod(lambda cls, text: lying),
    )
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    assert "--graduation write failed AFTER the target was replaced" in result.stderr
    assert "readback mismatch" in result.stderr
    # the message names the real semantics: it DID parse
    assert "content hash differs" in result.stderr
    assert "test examples=6" in result.stdout  # the report is not lost


def test_eval_graduation_readback_read_failure_is_post_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the post-replace readback's OWN read can fail (an I/O error):
    # that is a post-replace failure, not a write failure — the
    # message must say the write failed AFTER the target was replaced
    # (the "cannot write" wording is dead; the outer except OSError
    # would print exactly that if the OSError escaped the wrap)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    orig_read_text = Path.read_text

    def _read_text_boom(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if self == out:
            raise OSError("simulated readback read failure")
        return orig_read_text(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", _read_text_boom)
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 5
    assert "--graduation write failed AFTER the target was replaced" in result.stderr
    assert "readback fault" in result.stderr
    assert "failed to read back" in result.stderr
    assert "cannot write --graduation" not in result.stderr
    assert "test examples=6" in result.stdout  # the report is not lost


def test_eval_graduation_replaces_existing_with_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # replacing an EXISTING garbage target is announced on stderr (the
    # #20 --out NOTICE — a silent clobber of a typo'd path would be a
    # lie); the file is pre-created 0644 to exercise the tighten path
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    out.write_text("old table", encoding="utf-8")
    os.chmod(out, 0o644)
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 0, result.stderr
    assert "NOTICE: replacing existing" in result.stderr
    assert out.stat().st_mode & 0o077 == 0
    GraduationTable.from_json(out.read_text(encoding="utf-8"))  # parses


def test_eval_graduation_override_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a per-label gate override changes only that label's gates (and
    # the effective target is visible in the table — provenance)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _eval_cfg(
        tmp_path,
        graduation='[graduation."Money/Bills"]\nmin_examples = 5\n',
    )
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        cfg=cfg,
    )
    assert result.exit_code == 0, result.stderr
    table = GraduationTable.from_json(out.read_text(encoding="utf-8"))
    entries = {e.label: e for e in table.labels}
    # Money/Bills: min_examples 5 → 3 < 5 → fails
    assert entries["Money/Bills"].min_examples == 5
    assert entries["Money/Bills"].graduated is False
    assert entries["Money/Bills"].reasons == ("below-min-examples",)
    # Shipping: default 10 → also fails (3 < 10)
    assert entries["Shipping"].min_examples == 10
    assert entries["Shipping"].graduated is False


def test_eval_graduation_bodies_never_in_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # invariant #5: message bodies never enter the table (ids,
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
    cfg = _eval_cfg(tmp_path, graduation="min_examples = 3\n")
    out = tmp_path / "graduation.json"
    result, _factory = _run_eval(
        tmp_path,
        "--graduation",
        str(out),
        "--confirm-graduation",
        monkeypatch=monkeypatch,
        scores=scores,
        cfg=cfg,
    )
    assert result.exit_code == 0, result.stderr
    assert marker not in out.read_text(encoding="utf-8")
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert marker not in caplog.text


# ---- recompute posture (AC4): ONLY the --confirm path writes the table ----


def test_eval_without_graduation_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # recompute posture (AC4, behavioural): a bare `eval` (and `train`)
    # never writes the table — the write is the opt-in
    # --graduation --confirm-graduation act
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    result, _factory = _run_eval(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert not (tmp_path / "graduation.json").exists()
    assert not list(tmp_path.glob("graduation.json*"))


def test_only_eval_confirm_writes_the_graduation_table() -> None:
    # NAMED-MINT TRIPWIRE (not a sandbox — honesty pass): a string
    # scan cannot prove absence; getattr-construction and a
    # hand-rolled dict json.dump evade any source scan. What this pin
    # DOES: name the sanctioned mint/write symbols and fail if a SECOND
    # module in the package (any depth — the AGENTS target layout grows
    # subpackages such as daemon/) starts naming them. The behavioural
    # weight sits in test_eval_without_graduation_writes_nothing.
    import pathlib

    import gmail_labeler

    package_dir = pathlib.Path(gmail_labeler.__file__).parent
    # the table is MINTED through build_table: exactly two modules name
    # it — graduation.py (the definition) and cli.py (the eval
    # command's call)
    minters = [
        py.name
        for py in sorted(package_dir.rglob("*.py"))
        if "build_table" in py.read_text(encoding="utf-8")
    ]
    assert minters == ["cli.py", "graduation.py"]
    # and no module other than graduation.py CONSTRUCTS the class
    # directly (a mint that bypassed build_table would skip its
    # validation)
    constructors = [
        py.name
        for py in sorted(package_dir.rglob("*.py"))
        if "GraduationTable(" in py.read_text(encoding="utf-8")
    ]
    assert constructors == ["graduation.py"]
    # and the Graduation Table write helper is named in exactly one
    # module — cli.py — and the --graduation write path in cli.py goes
    # THROUGH it (a second table-file write path that bypassed the
    # helper would be a second writer, and this scan would name it)
    table_writers = [
        py.name
        for py in sorted(package_dir.rglob("*.py"))
        if "_write_graduation_table" in py.read_text(encoding="utf-8")
    ]
    assert table_writers == ["cli.py"]
    assert "_write_graduation_table(Path(graduation)" in (
        package_dir / "cli.py"
    ).read_text(encoding="utf-8")


def test_eval_graduation_is_on_the_frozen_surface() -> None:
    # the FLAG SURFACE is asserted from click's parameter metadata, not
    # the rendered help (the width-wrapping prior art)
    from typer.core import TyperGroup
    from typer.main import get_command

    root = get_command(app)
    assert isinstance(root, TyperGroup)
    eval_cmd = root.commands["eval"]
    declared = {opt for p in eval_cmd.params for opt in getattr(p, "opts", [])}
    for flag in (
        "--config",
        "--assume-no-filters",
        "--out",
        "--json",
        "--bands",
        "--graduation",
        "--confirm-graduation",
    ):
        assert flag in declared
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
