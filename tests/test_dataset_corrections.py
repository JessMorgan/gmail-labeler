"""Corrections -> training signal through the builder's composition seam,
and the `corrections export` command (ticket #27).

Two load-bearing ACs: the export is DB-only + NEVER retrains (the human
gate), and the builder seam (apply_corrections_to_records) folds the
Corrections into the corpus WITHOUT forking build_dataset — only a
'human' 'added' folds (a 'filter_covered' or 'unverifiable' one is
excluded — invariant #6 / #22's exclusion-not-guessing), a 'removed' is a
negative signal, untouched records pass through by identity, and no bodies
move.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner, Result

from gmail_labeler.appdb import AppDB, CorrectionRow
from gmail_labeler.cli import app, corrections_export
from gmail_labeler.cli import status as status_cmd
from gmail_labeler.corpus_record import CorpusRecord, body_hash
from gmail_labeler.dataset_build import apply_corrections_to_records

runner = CliRunner()
TS = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)


def _rec(mid: str, labels: tuple[str, ...]) -> CorpusRecord:
    return CorpusRecord(
        id=mid,
        date=TS,
        subject="s",
        sender="a@x.com",
        recipients=("me@x.com",),
        labels=labels,
        body_hash=body_hash("b"),
        source="api",
        date_provenance="internal_date",
    )


def _corr(mid: str, label: str, kind: str, attribution: str = "human") -> CorrectionRow:
    return CorrectionRow(
        message_id=mid, label=label, kind=kind, detected_at=TS, attribution=attribution
    )


# -- apply_corrections_to_records (the composition seam) ----------------------


def test_removed_correction_drops_the_label() -> None:
    recs = [_rec("aa11", ("Money/Bills", "Shipping"))]
    out = apply_corrections_to_records(recs, [_corr("aa11", "Money/Bills", "removed")])
    assert out[0].labels == ("Shipping",)  # the negative signal


def test_added_correction_adds_the_label() -> None:
    recs = [_rec("aa11", ("Shipping",))]
    out = apply_corrections_to_records(recs, [_corr("aa11", "Money/Bills", "added")])
    assert out[0].labels == ("Money/Bills", "Shipping")  # sorted


def test_filter_covered_added_is_excluded() -> None:
    # invariant #6: a filter-authored add is NOT trainable signal
    recs = [_rec("aa11", ("Shipping",))]
    out = apply_corrections_to_records(
        recs, [_corr("aa11", "Money/Bills", "added", attribution="filter_covered")]
    )
    assert out[0].labels == ("Shipping",)  # unchanged — the add was excluded


def test_unverifiable_added_is_excluded() -> None:
    # F1 (invariant #6 restored): an unverifiable add is EXCLUDED from the
    # folded training records EXACTLY like filter_covered -- the seam's
    # output (the artifact the retrain consumes) does NOT gain the label.
    # This is the "ABSENT from export output" pin: the seam is the single
    # filtering point, and guessing authorship toward the human is what #22
    # forbids. (The CLI `corrections export` command is a faithful RAW dump
    # that still SHOWS the row with attribution='unverifiable' for audit;
    # the label never reaches the corpus until the seam credits it 'human'.)
    recs = [_rec("aa11", ("Shipping",))]
    out = apply_corrections_to_records(
        recs, [_corr("aa11", "Money/Bills", "added", attribution="unverifiable")]
    )
    assert out[0].labels == ("Shipping",)  # unchanged — withheld, not folded


def test_human_added_is_the_only_folded_add() -> None:
    # F1 the positive control: a 'human' add DOES fold (the exclusion is
    # not a blanket drop of every add)
    recs = [_rec("aa11", ("Shipping",))]
    out = apply_corrections_to_records(
        recs, [_corr("aa11", "Money/Bills", "added", attribution="human")]
    )
    assert out[0].labels == ("Money/Bills", "Shipping")


def test_untouched_record_passes_through_by_identity() -> None:
    recs = [_rec("aa11", ("Shipping",)), _rec("bb22", ("Money/Bills",))]
    out = apply_corrections_to_records(recs, [_corr("aa11", "Money/Bills", "added")])
    assert out[1] is recs[1]  # identity-preserving for the untouched record
    assert out[0] is not recs[0]  # corrected -> a new record


def test_removed_never_excluded_even_if_filter_matches() -> None:
    # 'removed' is human signal regardless of filters (invariant #6): a
    # removal is applied even though no filter slice exists for it
    recs = [_rec("aa11", ("Money/Bills", "Shipping"))]
    out = apply_corrections_to_records(recs, [_corr("aa11", "Shipping", "removed")])
    assert out[0].labels == ("Money/Bills",)


def test_empty_corrections_returns_records_unchanged() -> None:
    recs = [_rec("aa11", ("Shipping",))]
    out = apply_corrections_to_records(recs, [])
    assert out == recs


# -- the export command -------------------------------------------------------


def _seed_corr(tmp_path: Path, rows: list[CorrectionRow]) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    if rows:
        db.commit_page([], 999, rows)
    db.close()


def _export(tmp_path: Path, cfg: Path, extra: tuple[str, ...] = ()) -> Result:
    return runner.invoke(
        app,
        [
            "corrections",
            "export",
            "--config",
            str(cfg),
            "--db",
            str(tmp_path / "state.db"),
            *extra,
        ],
    )


def _cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n'
        f'[model]\ncheckpoint_dir = "{tmp_path / "ckpt"}"\nencoder = "fake"\n'
        f'[gmail]\ncredentials_path = "{tmp_path / "cred.json"}"\n'
        f'[store]\ndb_path = "{tmp_path / "state.db"}"\n',
        encoding="utf-8",
    )
    return cfg


def test_export_writes_rows_no_bodies(tmp_path: Path) -> None:
    _seed_corr(
        tmp_path,
        [
            _corr("aa11", "Money/Bills", "removed"),
            _corr("bb22", "Shipping", "added", attribution="filter_covered"),
            _corr("cc33", "Shipping", "added", attribution="unverifiable"),
        ],
    )
    out_file = tmp_path / "corrections.json"
    result = _export(tmp_path, _cfg(tmp_path), extra=("--out", str(out_file)))
    assert result.exit_code == 0, result.output
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload == [
        {
            "message_id": "aa11",
            "label": "Money/Bills",
            "kind": "removed",
            "attribution": "human",
        },
        {
            "message_id": "bb22",
            "label": "Shipping",
            "kind": "added",
            "attribution": "filter_covered",
        },
        # F1: the raw dump is AUDITABLE — an unverifiable add is written
        # WITH its attribution (you can see it was withheld and why); the
        # label still never reaches the corpus because the SEAM drops it
        {
            "message_id": "cc33",
            "label": "Shipping",
            "kind": "added",
            "attribution": "unverifiable",
        },
    ]
    # invariant #5: no body/subject/raw key anywhere in the export
    assert "body" not in out_file.read_text(encoding="utf-8").lower()


def test_export_to_stdout(tmp_path: Path) -> None:
    _seed_corr(tmp_path, [_corr("aa11", "Shipping", "added")])
    result = _export(tmp_path, _cfg(tmp_path))  # default --out -
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["kind"] == "added"


def test_export_makes_no_gmail_call(tmp_path: Path, monkeypatch) -> None:
    import gmail_labeler.cli as cli

    def _boom(cfg_: object) -> object:
        raise AssertionError("corrections export must not build a GmailClient")

    monkeypatch.setattr(cli, "_daemon_client", _boom)
    _seed_corr(tmp_path, [_corr("aa11", "Shipping", "removed")])
    result = _export(tmp_path, _cfg(tmp_path))
    assert result.exit_code == 0, result.output


def test_export_bad_format_is_usage_2(tmp_path: Path) -> None:
    _seed_corr(tmp_path, [_corr("aa11", "Shipping", "removed")])
    result = _export(tmp_path, _cfg(tmp_path), extra=("--format", "csv"))
    assert result.exit_code == 2


def test_export_refuses_out_equal_to_state_db(tmp_path: Path) -> None:
    # CRITICAL guard: --out naming the state DB would overwrite the SQLite
    # file with JSON and destroy the corrections + cursor. Refused (exit 2)
    # and the DB SURVIVES (re-opens clean).
    db_path = tmp_path / "state.db"
    _seed_corr(tmp_path, [_corr("aa11", "Shipping", "removed")])
    result = _export(tmp_path, _cfg(tmp_path), extra=("--out", str(db_path)))
    assert result.exit_code == 2
    assert "state DB" in result.output
    # the DB is NOT destroyed: it still opens and still has the correction
    db = AppDB(db_path)
    db.open()
    assert len(db.corrections_for_message("aa11")) == 1
    db.close()


def test_export_refuses_out_inside_corpus_store(tmp_path: Path) -> None:
    _seed_corr(tmp_path, [_corr("aa11", "Shipping", "removed")])
    inside = tmp_path / "corpus" / "leak.json"
    inside.parent.mkdir(parents=True, exist_ok=True)
    result = _export(tmp_path, _cfg(tmp_path), extra=("--out", str(inside)))
    assert result.exit_code == 2
    assert not inside.exists()


def test_export_unwritable_out_is_5_not_crash(tmp_path: Path) -> None:
    # MAJOR: an unwritable --out target is exit 5, NOT the exit-1
    # FileNotFoundError traceback the contract forbids
    _seed_corr(tmp_path, [_corr("aa11", "Shipping", "removed")])
    bad = tmp_path / "no-such-dir" / "x.json"
    result = _export(tmp_path, _cfg(tmp_path), extra=("--out", str(bad)))
    assert result.exit_code == 5, result.output


def test_export_missing_db_is_5(tmp_path: Path) -> None:
    # the export READS the store — a typo'd --db is refused, not silently
    # created-then-answered-with-[]-exit-0
    cfg = _cfg(tmp_path)
    result = runner.invoke(
        app,
        [
            "corrections",
            "export",
            "--config",
            str(cfg),
            "--db",
            str(tmp_path / "does-not-exist.db"),
        ],
    )
    assert result.exit_code == 5
    assert not (tmp_path / "does-not-exist.db").exists()  # NOT created


# -- the human gate: no call path from corrections/status to a train entry ----


def test_human_gate_no_train_call_path() -> None:
    # AC: no auto-retrain. The export + status command SOURCES never
    # reference a train entry point (train / build_dataset /
    # evaluate_graduation / calibrate_bands) — the retrain is a separate
    # human decision, and the composition seam is only DOCUMENTED here
    import inspect

    for fn in (corrections_export, status_cmd):
        src = inspect.getsource(fn)
        for forbidden in (
            "build_dataset",
            "evaluate_graduation",
            "calibrate_bands",
            "train(",
            "apply_corrections_to_records",
        ):
            assert forbidden not in src, f"{fn.__name__} references {forbidden!r}"


def test_composition_seam_is_not_wired_into_export() -> None:
    # the seam EXISTS as a pure function but the export does NOT call it
    # (export writes the artifact; a human opt-in retrain composes it)
    import inspect

    assert "apply_corrections_to_records" not in inspect.getsource(corrections_export)
