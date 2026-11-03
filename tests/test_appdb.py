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
    _SCHEMA,
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
    # a foreign-version schema_version (99) is refused by the version
    # HARD gate (fold-4 item 1) with the version-named message
    with pytest.raises(AppDBError, match=r"schema_version \[99\] != 1"):
        db.open()
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(99,)]
    finally:
        conn.close()


def test_polluted_schema_version_is_refused(tmp_path: Path) -> None:
    # a DB polluted with BOTH (99) and (1) (the write-before-refusal
    # interim state) is refused — the version set must be EXACTLY
    # {SCHEMA_VERSION}, not a min/max that a second row would bypass.
    # fold-4 item 1: UN-RETARGETED to assert its docstring property —
    # the exact-set refusal with the VERSION-named message (the old
    # match="schema_version" substring was satisfied by the table name
    # in a foreign-tables listing, certifying the opposite of the
    # docstring — the retargeting is recorded in the commit attribution)
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version (version) VALUES (99)")
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match=r"schema_version \[1, 99\] != 1"):
        db.open()


def test_our_shape_shadow_results_with_foreign_version_is_refused(
    tmp_path: Path,
) -> None:
    # fold-4 item 1 (the data-integrity regression ds reproduced): a DB
    # with our-shape shadow_results PLUS a foreign schema_version ({99})
    # is REFUSED — the version is a HARD gate, not advisory (a v2 DB is
    # not consumed by a v1 tool). The old OR-gate OPENED and WROTE this.
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE shadow_results ("
        "run_id TEXT, message_id TEXT, label TEXT, probability REAL, "
        "band TEXT, run_ts TEXT, "
        "PRIMARY KEY (run_id, message_id, label))"
    )
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version (version) VALUES (99)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match=r"schema_version \[99\] != 1"):
        db.open()
    # NOT opened/written: no new rows, no restamp
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(99,)]
        # the foreign shadow_results table is untouched (no rows written)
        assert conn.execute("SELECT COUNT(*) FROM shadow_results").fetchone() == (0,)
    finally:
        conn.close()
    # fold-5 A6: NOT opened, literal — a write attempt is refused (the
    # connection was never established, so record_run raises "not open")
    with pytest.raises(AppDBError, match="not open"):
        db.record_run("run-1", TS, [_row()])


def test_our_shape_empty_schema_version_is_corrupt(tmp_path: Path) -> None:
    # fold-4 item 1 + fold-7 item 1: an our-shape schema_version that is
    # EMPTY (no version row) is named corrupt — NOT silently adopted —
    # and this is the EXECUTABLE proof of the appdb "corrupt" comment's
    # claims: the exact crash state (executescript done, both tables,
    # version row never inserted) keeps REFUSING on every re-open (the
    # ownership gate raises before open()'s re-stamp branch), and
    # recovery is NOT re-running: delete the file (a fresh open
    # re-creates + stamps) or stamp the row manually.
    def _build_crash_state(path: Path) -> None:
        # what a crash between executescript and the stamp commit leaves
        conn = sqlite3.connect(str(path))
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

    db_path = tmp_path / "state.db"
    _build_crash_state(db_path)
    # THREE open() attempts all refuse with the SAME named message
    messages: list[str] = []
    for _attempt in range(3):
        with pytest.raises(AppDBError, match="schema_version is empty") as exc:
            AppDB(db_path).open()
        messages.append(str(exc.value))
    assert len(set(messages)) == 1  # repeat refusal, same named cause
    # the refusal never writes: the state survives exactly as crashed
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == []
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "shadow_results" in tables  # from the executescript, not a re-stamp
    finally:
        conn.close()
    # recovery A: DELETE the file — a fresh open re-creates + stamps
    db_path.unlink()
    db = AppDB(db_path)
    db.open()
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [
            (SCHEMA_VERSION,)
        ]
    finally:
        conn.close()
        db.close()
    # recovery B: rebuild the crash state (from a clean file — _SCHEMA's
    # CREATE TABLE IF NOT EXISTS would otherwise keep recovery A's stamp),
    # stamp the row MANUALLY — open OK (the gate passes on the stamp)
    db_path.unlink()
    _build_crash_state(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    db.open()  # no refusal
    db.close()


def test_wrong_shape_shadow_results_is_refused(tmp_path: Path) -> None:
    # disposition 2: a same-named shadow_results with the WRONG column
    # set is refused, named (shape, not name) — BEFORE any write
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE shadow_results (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="shadow_results has the wrong shape"):
        db.open()


def test_wrong_shape_schema_version_is_refused(tmp_path: Path) -> None:
    # disposition 2: a same-named schema_version that is NOT the
    # single-column `version` table (e.g. two columns) is refused, named
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER, note TEXT)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="schema_version has the wrong shape"):
        db.open()


def test_valid_ours_coeexists_with_foreign_tables(tmp_path: Path) -> None:
    # disposition 2 (documented decision): a valid ours-DB (our
    # shadow_results) PLUS foreign extra tables COEXISTS — our tables are
    # the authority, the extras are left alone
    db_path = tmp_path / "state.db"
    db = AppDB(db_path)
    db.open()
    db.record_run("run-1", TS, [_row("aa", "Shipping", 0.9, "high")])
    db.close()
    # add a foreign extra table
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE other_app_data (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO other_app_data (id) VALUES (1)")
    conn.commit()
    conn.close()
    # re-open: coexists (no refusal)
    db2 = AppDB(db_path)
    db2.open()
    _run_ts, rows = db2.load_run("run-1")
    assert len(rows) == 1
    db2.close()


def test_versioned_foreign_db_gains_no_table_pre_refusal(tmp_path: Path) -> None:
    # disposition 2 (ds-C5 probe): a versioned-foreign DB (our
    # schema_version shape + a foreign version) is refused BEFORE
    # executescript — it does NOT gain a shadow_results table pre-refusal
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (99)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match=r"schema_version \[99\] != 1"):
        db.open()
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "shadow_results" not in tables  # NOT created pre-refusal
        assert tables == {"schema_version"}
    finally:
        conn.close()
    # fold-5 A6: NOT opened, literal — a write attempt is refused (the
    # connection was never established, so record_run raises "not open")
    with pytest.raises(AppDBError, match="not open"):
        db.record_run("run-1", TS, [_row()])


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
    # the #24 seam: the applications table lands in the same file under a
    # v2 restamp (the v1->v2 restamp/migration is UNSOLVED pending #24 —
    # see the appdb module docstring's UPGRADE STORY) — the seam is
    # obvious and tested
    import gmail_labeler.appdb as appdb_module

    source = Path(appdb_module.__file__).read_text(encoding="utf-8")
    assert "ticket #24" in source
    assert "applications" in source


def test_directory_is_io_failure_not_corrupt(tmp_path: Path) -> None:
    # S2 cause split: a DIRECTORY at the db path is "cannot be opened
    # (I/O failure)" — NOT "corrupt (not a SQLite file)" (the two causes
    # are not conflated, the #22 doctrine)
    db_path = tmp_path / "state.db"
    db_path.mkdir()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="cannot be opened \\(I/O failure\\)"):
        db.open()


def test_foreign_sqlite_db_is_refused_by_ownership(tmp_path: Path) -> None:
    # S1: a SQLite file that carries foreign tables and NONE of ours is
    # refused, named (the foreign table names are listed) — a foreign DB
    # is not silently adopted with our schema stamped on it
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE other_app_data (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO other_app_data (id) VALUES (1)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="not a gmail-labeler state DB"):
        db.open()
    # the foreign table is untouched (no schema_version stamped on it)
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "schema_version" not in tables
        assert "shadow_results" not in tables
        assert "other_app_data" in tables
    finally:
        conn.close()


def test_record_run_empty_is_refused(tmp_path: Path) -> None:
    # S2 pin: record_run refuses an empty row list at the only writer
    # (load_run's docstring assumes an empty run was never committed)
    db = AppDB(tmp_path / "state.db")
    db.open()
    with pytest.raises(AppDBError, match="no rows"):
        db.record_run("run-1", TS, [])
    db.close()


def _tampered_db(db_path: Path) -> None:
    # a hand-edited DB: the shadow_results table WITHOUT the shipped
    # NOT NULL constraints (so NULL rows can exist) + a valid
    # schema_version (so the ownership/version checks pass and the
    # read-back validation is the gate that fires)
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.execute(
        "CREATE TABLE shadow_results ("
        "run_id TEXT, message_id TEXT, label TEXT, probability REAL, "
        "band TEXT, run_ts TEXT, "
        "PRIMARY KEY (run_id, message_id, label))"
    )
    conn.commit()
    conn.close()


def test_load_run_tampered_rows_are_refused(tmp_path: Path) -> None:
    # S2 pin: a tampered DB (hand-edited rows) is refused, named — not
    # an exit-1 traceback. Covers: NULL label, NULL message_id, and a
    # mixed run_ts stamp (one run, one stamp)
    db_path = tmp_path / "state.db"
    _tampered_db(db_path)
    cols = "run_id, message_id, label, probability, band, run_ts"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        f"INSERT INTO shadow_results ({cols}) "
        "VALUES ('run-1', 'aa', NULL, 0.9, 'high', '2026-11-01T12:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    db.open()
    with pytest.raises(AppDBError, match="fails validation"):
        db.load_run("run-1")
    db.close()
    # NULL message_id
    _tampered_db(tmp_path / "state2.db")
    conn = sqlite3.connect(str(tmp_path / "state2.db"))
    conn.execute(
        f"INSERT INTO shadow_results ({cols}) "
        "VALUES ('run-1', NULL, 'Shipping', 0.9, 'high', '2026-11-01T12:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    db2 = AppDB(tmp_path / "state2.db")
    db2.open()
    with pytest.raises(AppDBError, match="fails validation"):
        db2.load_run("run-1")
    db2.close()
    # mixed run_ts stamps (one run, one stamp)
    _tampered_db(tmp_path / "state3.db")
    conn = sqlite3.connect(str(tmp_path / "state3.db"))
    conn.execute(
        f"INSERT INTO shadow_results ({cols}) VALUES ("
        "'run-2', 'bb', 'Shipping', 0.8, 'med', '2026-11-01T12:00:00+00:00'),"
        "('run-2', 'cc', 'Money/Bills', 0.7, 'med', '2026-11-02T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    db3 = AppDB(tmp_path / "state3.db")
    db3.open()
    with pytest.raises(AppDBError, match="distinct run_ts"):
        db3.load_run("run-2")
    db3.close()
