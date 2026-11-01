"""appdb.py (ticket #23): the state DB adapter — create-all-if-missing,
transaction-per-run (crash mid-run leaves no half rows), NO bodies
(invariant #5), deterministic reads, the #24 applications seam.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gmail_labeler.appdb import (
    SCHEMA_VERSION,
    AppDB,
    AppDBError,
    ShadowRow,
)

TS = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)


def _row(
    message_id: str = "aa",
    label: str = "Shipping",
    probability: float = 0.9,
    band: str | None = "high",
) -> ShadowRow:
    return ShadowRow(
        message_id=message_id,
        label=label,
        probability=probability,
        band=band,
        run_ts=TS,
    )


def test_creates_schema_if_missing(tmp_path: Path) -> None:
    # create-all-if-missing: opening a fresh path creates the schema
    # (schema_version row + the shadow_results table)
    db = AppDB(tmp_path / "state.db")
    db.open()
    conn = sqlite3.connect(str(db.path))
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchone() == (
            SCHEMA_VERSION,
        )
        cols = {
            r[1] for r in conn.execute("PRAGMA table_info(shadow_results)").fetchall()
        }
        assert cols == {
            "run_id",
            "message_id",
            "label",
            "probability",
            "band",
            "run_ts",
        }
    finally:
        conn.close()
    db.close()


def test_db_file_is_0600_from_creation(tmp_path: Path) -> None:
    # the secret-tight discipline (the rows are attachable-public, but
    # the file is created 0600 — no 0644 window)
    db = AppDB(tmp_path / "state.db")
    db.open()
    assert db.path.stat().st_mode & 0o077 == 0
    db.close()


def test_record_run_and_load_roundtrip(tmp_path: Path) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    count = db.record_run(
        "run-1",
        TS,
        [
            _row("bb", "Shipping", 0.9, "high"),
            _row("aa", "Money/Bills", 0.75, "med"),
            _row("aa", "Shipping", 0.4, None),  # the no-band state
        ],
    )
    assert count == 3
    run_ts, rows = db.load_run("run-1")
    assert run_ts == TS
    # deterministic (message_id, label) order — the report's total order
    assert [(r.message_id, r.label) for r in rows] == [
        ("aa", "Money/Bills"),
        ("aa", "Shipping"),
        ("bb", "Shipping"),
    ]
    assert rows[0].probability == 0.75
    assert rows[0].band == "med"
    assert rows[1].band is None  # the explicit no-verdict state
    assert db.run_exists("run-1")
    assert not db.run_exists("run-2")
    db.close()


def test_crash_mid_run_leaves_no_half_rows(tmp_path: Path) -> None:
    # the transaction-per-run contract: a mid-run refusal (an invalid
    # row) rolls back the whole run — no half rows
    db = AppDB(tmp_path / "state.db")
    db.open()
    with pytest.raises(AppDBError, match="out of \\[0, 1\\]"):
        db.record_run(
            "run-1",
            TS,
            [_row("aa", "Shipping", 0.9, "high"), _row("bb", "Shipping", 1.5, "high")],
        )
    assert db.latest_run() is None  # nothing committed
    conn = sqlite3.connect(str(db.path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM shadow_results").fetchone() == (0,)
    finally:
        conn.close()
    db.close()


def test_latest_run_ordering(tmp_path: Path) -> None:
    # latest run = the most recent run_ts (then run_id — a total order)
    db = AppDB(tmp_path / "state.db")
    db.open()
    db.record_run("run-a", TS, [_row("aa", "Shipping", 0.9, "high")])
    later = TS.replace(hour=13)
    db.record_run("run-b", later, [_row("aa", "Shipping", 0.9, "high")])
    assert db.latest_run() == "run-b"
    db.close()


def test_load_run_not_found_is_named(tmp_path: Path) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    with pytest.raises(AppDBError, match="run-1"):
        db.load_run("run-1")
    db.close()


def test_row_validation_refusals(tmp_path: Path) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    with pytest.raises(AppDBError, match="hex"):
        db.record_run("run-1", TS, [_row("not-hex", "Shipping", 0.9, "high")])
    with pytest.raises(AppDBError, match="label must be non-empty"):
        db.record_run("run-1", TS, [_row("aa", "  ", 0.9, "high")])
    with pytest.raises(AppDBError, match="must be a number"):
        db.record_run(
            "run-1",
            TS,
            [_row("aa", "Shipping", True, "high")],  # type: ignore[arg-type]
        )
    with pytest.raises(AppDBError, match="band"):
        db.record_run("run-1", TS, [_row("aa", "Shipping", 0.9, "urgent")])
    with pytest.raises(AppDBError, match="duplicate"):
        db.record_run(
            "run-1",
            TS,
            [_row("aa", "Shipping", 0.9, "high"), _row("aa", "Shipping", 0.8, "med")],
        )
    with pytest.raises(AppDBError, match="non-empty"):
        db.record_run("  ", TS, [_row()])
    with pytest.raises(AppDBError, match="timezone-aware"):
        db.record_run("run-1", datetime(2026, 11, 1, 12, 0, 0), [_row()])
    db.close()


def test_foreign_schema_version_is_refused(tmp_path: Path) -> None:
    # a newer tool's DB is refused, named (the SplitManifest discipline)
    # — and NOT written before the refusal (no (1) row appended)
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version (version) VALUES (99)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="schema_version \\[99\\]"):
        db.open()
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(99,)]
    finally:
        conn.close()


def test_polluted_schema_version_is_refused(tmp_path: Path) -> None:
    # a DB polluted with BOTH (99) and (1) (the write-before-refusal
    # interim state) is refused — the version set must be EXACTLY
    # {SCHEMA_VERSION}, not a min/max that a second row would bypass
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version (version) VALUES (99)")
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="schema_version"):
        db.open()


def test_schema_version_row_is_stable_across_opens(tmp_path: Path) -> None:
    # the schema_version row is NOT appended on every open (the PK makes
    # re-stamps a no-op) — repeated opens leave exactly one row
    db_path = tmp_path / "state.db"
    for _ in range(3):
        db = AppDB(db_path)
        db.open()
        db.close()
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(1,)]
    finally:
        conn.close()


def test_corrupt_file_is_refused(tmp_path: Path) -> None:
    # a non-SQLite file is a corrupt DB, named — the cause phrase is
    # path-INDEPENDENT (the old match="corrupt" passed vacuously because
    # tmp_path embeds the test name)
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"not a sqlite file")
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="not a SQLite file"):
        db.open()


def test_not_open_is_refused(tmp_path: Path) -> None:
    db = AppDB(tmp_path / "state.db")
    with pytest.raises(AppDBError, match="not open"):
        db.record_run("run-1", TS, [_row()])
    with pytest.raises(AppDBError, match="not open"):
        db.latest_run()


def test_applications_table_seam_is_documented() -> None:
    # the #24 seam: the applications table lands in the same file, under
    # the same versioning story — the seam is obvious and tested
    import gmail_labeler.appdb as appdb_module

    source = Path(appdb_module.__file__).read_text(encoding="utf-8")
    assert "ticket #24" in source
    assert "applications" in source
