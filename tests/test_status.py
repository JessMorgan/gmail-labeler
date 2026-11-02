"""`status` — realized precision from the LOCAL store (ticket #27).

Two layers: the PURE math (compute_precision) and the CLI wiring. The
CLI pins are the AC's load-bearing ones: status makes ZERO GmailClient
calls (grep + a raising client-factory), answers on an UNAUTHENTICATED
config (no auth gate), never emits a body, and honours the exit
contract. The bands/graduation artifacts come from the shared _prepare
(train + eval --bands + eval --graduation --confirm) so the graduated
set and the content-hash basis line are REAL.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from gmail_labeler.appdb import AppDB, CorrectionRow
from gmail_labeler.applications import ApplicationRow
from gmail_labeler.cli import app
from gmail_labeler.status import compute_precision
from test_shadow_cli import _prepare, _shadow_cfg

runner = CliRunner()
TS = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)


# -- the pure math ------------------------------------------------------------


def test_precision_formula_and_no_data() -> None:
    rows = compute_precision(
        frozenset({"A", "B"}),
        {"A": 10, "B": 0},
        {"A": (2, 1, 0, 0)},  # A: removed=2, added=1, fc=0, uv=0
    )
    by = {r.label: r for r in rows}
    assert by["A"].precision == pytest.approx(0.8)  # (10-2)/10
    assert by["A"].written == 10 and by["A"].removed == 2
    assert by["A"].human_added == 1
    assert by["B"].precision is None  # written 0 -> 'no data'


def test_precision_zero_when_everything_removed() -> None:
    rows = compute_precision(frozenset({"A"}), {"A": 4}, {"A": (4, 0, 0, 0)})
    assert rows[0].precision == 0.0


def test_filter_covered_added_excluded_from_human_signal() -> None:
    # invariant #6: the filter-covered slice is DISCLOSED but EXCLUDED
    # from the net human signal
    rows = compute_precision(frozenset({"A"}), {"A": 5}, {"A": (0, 3, 2, 0)})
    r = rows[0]
    assert r.added == 3 and r.added_filter_covered == 2
    assert r.human_added == 1  # 3 - 2 - 0


def test_unverifiable_added_excluded_from_human_signal() -> None:
    # F1 (invariant #6 restored): the unverifiable slice is EXCLUDED from
    # the human signal EXACTLY like filter_covered -- an add whose
    # authorship cannot be determined is WITHHELD, not credited to the
    # human. human_added = added - filter_covered - unverifiable.
    rows = compute_precision(frozenset({"A"}), {"A": 5}, {"A": (0, 4, 1, 2)})
    r = rows[0]
    assert r.added == 4 and r.added_filter_covered == 1
    assert r.added_unverifiable == 2
    assert r.human_added == 1  # 4 - 1 - 2 (the withheld slices never count)


def test_removed_exceeding_written_is_refused_not_clamped() -> None:
    # F2 item 9: the join gate keeps removed<=written on an honest store,
    # so removed>written is a corrupt DB -- compute_precision REFUSES named
    # (PrecisionInvariantError), it does NOT emit a negative precision.
    from gmail_labeler.status import PrecisionInvariantError

    with pytest.raises(PrecisionInvariantError, match="exceed"):
        compute_precision(frozenset({"A"}), {"A": 1}, {"A": (3, 0, 0, 0)})


def test_graduated_label_with_no_corrections_is_perfect() -> None:
    rows = compute_precision(frozenset({"A"}), {"A": 7}, {})
    assert rows[0].precision == 1.0 and rows[0].removed == 0


def test_output_is_sorted_by_label_total_order() -> None:
    rows = compute_precision(frozenset({"B", "A", "C"}), {}, {})
    assert [r.label for r in rows] == ["A", "B", "C"]


# -- the CLI ------------------------------------------------------------------


def _app_row(label: str, mid: str) -> ApplicationRow:
    return ApplicationRow(
        message_id=mid,
        label=label,
        label_id="L",
        score=0.9,
        band="high",
        tier="ai-high",
        applied_at=TS,
        source="daemon",
    )


def _corr(label: str, mid: str, kind: str, attribution: str = "human") -> CorrectionRow:
    return CorrectionRow(
        message_id=mid, label=label, kind=kind, detected_at=TS, attribution=attribution
    )


def _seed(
    tmp_path: Path, apps: list[ApplicationRow], corr: list[CorrectionRow]
) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    if apps:
        db.commit_applications(apps)
    if corr:
        db.commit_page([], 999, corr)  # corrections ride a commit (cursor set)
    db.close()


def _status(tmp_path: Path, cfg: Path, extra: tuple[str, ...] = ()) -> Result:
    return runner.invoke(
        app,
        [
            "status",
            "--config",
            str(cfg),
            "--bands",
            str(tmp_path / "bands.json"),
            "--graduation",
            str(tmp_path / "graduation.json"),
            "--db",
            str(tmp_path / "state.db"),
            *extra,
        ],
    )


def test_status_reports_realized_precision(tmp_path: Path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    _seed(
        tmp_path,
        [_app_row("Money/Bills", "aa11"), _app_row("Money/Bills", "aa22")],
        [_corr("Money/Bills", "aa11", "removed")],
    )
    result = _status(tmp_path, cfg)
    assert result.exit_code == 0, result.output
    assert "Money/Bills" in result.output
    assert "written=2" in result.output and "removed=1" in result.output
    assert "precision=0.50" in result.output  # (2-1)/2


def test_status_makes_no_gmail_call(tmp_path: Path, monkeypatch) -> None:
    # the no-network pin: EVERY client factory in cli RAISES if called;
    # status exits 0 proving it never builds a GmailClient through ANY of
    # them (_daemon_client / _shadow_client / _export_client)
    _prepare(tmp_path, monkeypatch)
    import gmail_labeler.cli as cli

    def _boom(cfg_: object) -> object:
        raise AssertionError("status must not construct a GmailClient")

    for factory in ("_daemon_client", "_shadow_client", "_export_client"):
        monkeypatch.setattr(cli, factory, _boom)
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    _seed(tmp_path, [_app_row("Money/Bills", "aa11")], [])
    result = _status(tmp_path, cfg)
    assert result.exit_code == 0, result.output


def test_status_source_has_no_client_construction() -> None:
    # the grep pin: the status command's source never references the
    # client / model / corpus seams (it is DB + local artifacts only)
    import inspect

    from gmail_labeler.cli import status as status_cmd

    src = inspect.getsource(status_cmd)
    for forbidden in (
        "_daemon_client",
        "_shadow_client",
        "_export_client",
        "GmailRestClient",
        "build_credentials",
        "_daemon_model_factory",
        "load_model",
        "CorpusStore",
        "list_messages",
    ):
        assert forbidden not in src, f"status source references {forbidden!r}"


def test_status_answers_on_unauthenticated_config(tmp_path: Path, monkeypatch) -> None:
    # the no-auth pin: a config with NO usable token still answers
    # (there is no auth gate on the DB-only path)
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(
        tmp_path, store=f'db_path = "{tmp_path / "state.db"}"'
    )  # a test config: no real OAuth token exists
    _seed(tmp_path, [_app_row("Shipping", "bb11")], [])
    result = _status(tmp_path, cfg)
    assert result.exit_code == 0, result.output
    assert "Shipping" in result.output


def test_status_no_bodies_in_output(tmp_path: Path, monkeypatch) -> None:
    # invariant #5: the output carries labels/counts/rates only
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    _seed(tmp_path, [_app_row("Money/Bills", "aa11")], [])
    result = _status(tmp_path, cfg)
    assert result.exit_code == 0
    assert "body" not in result.output.lower()
    assert "Subject" not in result.output


def test_status_json_shape(tmp_path: Path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    _seed(
        tmp_path,
        [_app_row("Money/Bills", "aa11"), _app_row("Money/Bills", "aa22")],
        [_corr("Money/Bills", "aa11", "removed")],
    )
    result = _status(tmp_path, cfg, extra=("--json",))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    bills = next(x for x in payload["labels"] if x["label"] == "Money/Bills")
    assert bills["written"] == 2 and bills["removed"] == 1
    assert bills["precision"] == pytest.approx(0.5)
    # F1: the json carries the unverifiable withheld slice as its own field
    assert bills["added_unverifiable"] == 0
    assert payload["basis"]["graduation_content_hash"]


def test_status_json_nonzero_withheld_slices_pinned(
    tmp_path: Path, monkeypatch
) -> None:
    # the JSON arithmetic on the NONZERO path (the zero-only assertion above
    # would survive hardcoding added_unverifiable=0 / human_added=added):
    # one filter-covered add + one unverifiable add + one human add, so the
    # json must report added=3, filter_covered=1, unverifiable=1, human=1
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    _seed(
        tmp_path,
        [_app_row("Money/Bills", "aa00")],
        [
            _corr("Money/Bills", "aa11", "added", attribution="filter_covered"),
            _corr("Money/Bills", "aa22", "added", attribution="unverifiable"),
            _corr("Money/Bills", "aa33", "added", attribution="human"),
        ],
    )
    result = _status(tmp_path, cfg, extra=("--json",))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    bills = next(x for x in payload["labels"] if x["label"] == "Money/Bills")
    assert bills["added"] == 3
    assert bills["added_filter_covered"] == 1
    assert bills["added_unverifiable"] == 1
    assert bills["human_added"] == 1  # 3 - 1 - 1 (both withheld slices excluded)


def test_status_filter_covered_disclosed(tmp_path: Path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    _seed(
        tmp_path,
        [_app_row("Money/Bills", "aa11")],
        [_corr("Money/Bills", "aa22", "added", attribution="filter_covered")],
    )
    result = _status(tmp_path, cfg)
    assert result.exit_code == 0, result.output
    assert "filter-covered=1" in result.output
    assert "human=0" in result.output  # the covered slice is excluded


def test_status_unverifiable_disclosed_and_excluded(
    tmp_path: Path, monkeypatch
) -> None:
    # F1 (invariant #6 restored): an unverifiable 'added' is DISCLOSED as
    # its OWN withheld slice and EXCLUDED from the human count -- the
    # status line shows unverifiable=1, human=0 (the old bool credited it
    # to the human; the inversion is exactly what F1 removed)
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    _seed(
        tmp_path,
        [_app_row("Money/Bills", "aa11")],
        [_corr("Money/Bills", "aa22", "added", attribution="unverifiable")],
    )
    result = _status(tmp_path, cfg)
    assert result.exit_code == 0, result.output
    assert "unverifiable=1" in result.output
    assert "human=0" in result.output  # withheld, NOT credited to the human


def test_status_no_applications_reports_no_data(tmp_path: Path, monkeypatch) -> None:
    # a store with NO applications: every graduated label is 'no data'
    # (the graduated set is NON-empty here — _prepare graduates both
    # labels — so this pins the 'no data' state, not an empty set), and
    # status still answers 0 (having no data is not a failure). The DB
    # must EXIST (a missing --db is the separate exit-5 typo case).
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    _seed(tmp_path, [], [])  # create an empty (but real) state DB
    result = _status(tmp_path, cfg)
    assert result.exit_code == 0, result.output
    assert "no data" in result.output


# -- exit contract ------------------------------------------------------------


def test_status_empty_bands_is_usage_2(tmp_path: Path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path)
    result = runner.invoke(
        app,
        [
            "status",
            "--config",
            str(cfg),
            "--bands",
            "   ",
            "--graduation",
            str(tmp_path / "graduation.json"),
        ],
    )
    assert result.exit_code == 2


def test_status_missing_artifact_is_5(tmp_path: Path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')
    result = runner.invoke(
        app,
        [
            "status",
            "--config",
            str(cfg),
            "--bands",
            str(tmp_path / "nope.json"),
            "--graduation",
            str(tmp_path / "graduation.json"),
            "--db",
            str(tmp_path / "state.db"),
        ],
    )
    assert result.exit_code == 5


def test_status_missing_db_is_5(tmp_path: Path, monkeypatch) -> None:
    # status READS the store — a typo'd --db is refused (exit 5), not
    # silently created-then-answered-with-'no data'-exit-0
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path)
    result = runner.invoke(
        app,
        [
            "status",
            "--config",
            str(cfg),
            "--bands",
            str(tmp_path / "bands.json"),
            "--graduation",
            str(tmp_path / "graduation.json"),
            "--db",
            str(tmp_path / "does-not-exist.db"),
        ],
    )
    assert result.exit_code == 5
    assert not (tmp_path / "does-not-exist.db").exists()  # NOT created
