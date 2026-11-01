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
from gmail_labeler.applications import ApplicationRow

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
    with pytest.raises(AppDBError, match=rf"schema_version \[99\] != {SCHEMA_VERSION}"):
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
    with pytest.raises(
        AppDBError, match=rf"schema_version \[1, 99\] != {SCHEMA_VERSION}"
    ):
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
    with pytest.raises(AppDBError, match=rf"schema_version \[99\] != {SCHEMA_VERSION}"):
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
    # and this is the EXECUTABLE proof of the tamper guard: the state
    # (both tables, version row never inserted) is TAMPER-ONLY since the
    # F1 atomic transition (a crash cannot leave it; only hand-editing
    # or a pre-F1 residue file can) and it keeps REFUSING on every
    # re-open (the ownership gate raises before _apply_schema), with the
    # recovery NAMED IN THE MESSAGE: stamp the row manually or delete
    # the file (a fresh open re-creates + stamps).
    def _build_crash_state(path: Path) -> None:
        # the tamper state (pre-F1, this was the crash state between
        # executescript and the stamp commit — now unreachable by crash)
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
    with pytest.raises(AppDBError, match=rf"schema_version \[99\] != {SCHEMA_VERSION}"):
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
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [
            (SCHEMA_VERSION,)
        ]
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
    # the #24 seam PAID: the applications table landed in the same file
    # under the v2 single-row restamp (the migration story is the appdb
    # module docstring's UPGRADE STORY) — the seam marker stays tested
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
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
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


# -- applications table + the v1->v2 migration (ticket #24) ------------------

_APPLICATIONS_COLUMNS = {
    "message_id",
    "label",
    "label_id",
    "score",
    "band",
    "tier",
    "applied_at",
    "source",
}


def _app_row(
    message_id: str = "aa",
    label: str = "Shipping",
    label_id: str = "Label_1",
    score: float = 0.9,
    band: str | None = "high",
    tier: str = "ai-high",
    applied_at: datetime = TS,
    source: str = "daemon",
) -> ApplicationRow:
    return ApplicationRow(
        message_id=message_id,
        label=label,
        label_id=label_id,
        score=score,
        band=band,  # pyright: ignore[reportArgumentType] — the lying-row fixture
        tier=tier,
        applied_at=applied_at,
        source=source,
    )


def _v1_db(db_path: Path, *, with_run: bool = True) -> None:
    """A DB exactly as ticket #23 shipped it: schema_version {1} +
    shadow_results (our shape), NO applications table."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.execute(
        "CREATE TABLE shadow_results ("
        "run_id TEXT NOT NULL, message_id TEXT NOT NULL, label TEXT NOT NULL, "
        "probability REAL NOT NULL, band TEXT, run_ts TEXT NOT NULL, "
        "PRIMARY KEY (run_id, message_id, label))"
    )
    if with_run:
        conn.execute(
            "INSERT INTO shadow_results VALUES "
            "('run-1', 'aa', 'Shipping', 0.9, 'high', '2026-11-01T12:00:00+00:00')"
        )
    conn.commit()
    conn.close()


def _versions(db_path: Path) -> list[int]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in conn.execute("SELECT version FROM schema_version")]
    finally:
        conn.close()


def test_fresh_db_is_v3_with_daemon_state(tmp_path: Path) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    assert _versions(db.path) == [SCHEMA_VERSION] == [3]
    conn = sqlite3.connect(str(db.path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(applications)")}
        assert cols == _APPLICATIONS_COLUMNS
        # v3 (ticket #25): the daemon_state single-row cursor table
        state_cols = {r[1] for r in conn.execute("PRAGMA table_info(daemon_state)")}
        assert state_cols == {"history_cursor"}
        # the corrections/status join (#27): indexes on message_id AND
        # label (the #23 NIT)
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'applications'"
            )
        }
        assert {"idx_applications_message_id", "idx_applications_label"} <= indexes
    finally:
        conn.close()
    db.close()


def test_v1_db_migrates_to_a_single_v3_row(tmp_path: Path) -> None:
    # the migration story (#23's recorded debt, carried through v3):
    # opening a live v1 DB with the v3 tool CREATES the applications and
    # daemon_state tables and RESTAMPS the version row to 3 — single-row
    # model: UPDATE, never a second row
    db_path = tmp_path / "state.db"
    _v1_db(db_path)
    db = AppDB(db_path)
    db.open()
    assert _versions(db_path) == [3]  # exactly one row, restamped
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(applications)")}
        assert cols == _APPLICATIONS_COLUMNS
        state_cols = {r[1] for r in conn.execute("PRAGMA table_info(daemon_state)")}
        assert state_cols == {"history_cursor"}
    finally:
        conn.close()
    # the recorded shadow run SURVIVES the migration (the data is the
    # operator's, the restamp is metadata)
    _run_ts, rows = db.load_run("run-1")
    assert len(rows) == 1
    db.close()
    # re-opening in-process is v3-clean (the tool that just migrated IS
    # the v3 tool)
    db2 = AppDB(db_path)
    db2.open()
    db2.close()


def test_migrated_db_refused_by_the_v1_gate(tmp_path: Path, monkeypatch) -> None:
    # the OTHER direction of the pin: the SAME migrated file opened by
    # the version-gate logic as a v1 tool refuses it — {3} != {1},
    # named. The forward-compat story IS the hard gate (documented
    # honestly: our own v3 is the tool that just migrated it)
    db_path = tmp_path / "state.db"
    _v1_db(db_path)
    migrator = AppDB(db_path)
    migrator.open()
    migrator.close()
    import gmail_labeler.appdb as appdb_module

    monkeypatch.setattr(appdb_module, "SCHEMA_VERSION", 1)
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match=r"schema_version \[3\] != 1"):
        db.open()


def test_v1_stamp_with_applications_table_is_refused(tmp_path: Path) -> None:
    # the single-row model's inconsistent state: a v1 tool never writes
    # an applications table — {1} + applications is corrupt, refused
    # named. TAMPER pin: unreachable by crash since the F1 atomic
    # transition (pre-F1 this was the unsafe "crash direction"; the safe
    # direction was {2}-without-tables — both are now hand-made states,
    # the self-heal of that one is pinned separately below)
    db_path = tmp_path / "state.db"
    _v1_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE applications (message_id TEXT, label TEXT, label_id TEXT, "
        "score REAL, band TEXT, tier TEXT, applied_at TEXT, source TEXT)"
    )
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match=r"schema_version \[1\]"):
        db.open()


def test_v2_stamp_without_applications_migrates_to_v3(tmp_path: Path) -> None:
    # {2} stamped but the applications table never created: unreachable
    # by crash since the F1 atomic transition (a tamper or pre-F1
    # residue) — under v3 this is the MIGRATABLE v2 stamp: the next open
    # runs the one-transaction transition (create-all-if-missing for
    # BOTH missing tables + the restamp), no refusal
    db_path = tmp_path / "state.db"
    _v1_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE schema_version SET version = 2")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    db.open()
    assert _versions(db_path) == [3]  # the migratable v2 stamp restamps
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(applications)")}
        assert cols == _APPLICATIONS_COLUMNS
        state_cols = {r[1] for r in conn.execute("PRAGMA table_info(daemon_state)")}
        assert state_cols == {"history_cursor"}
    finally:
        conn.close()
    db.close()


def test_wrong_shape_applications_is_refused(tmp_path: Path) -> None:
    # the ownership gate EXTENDED: applications, if present, must match
    # ITS shape too (the column-set equality pattern of shadow_results)
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL PRIMARY KEY)")
    conn.execute(
        "INSERT INTO schema_version (version) VALUES (2)",
    )
    conn.execute("CREATE TABLE applications (message_id TEXT, note TEXT)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="applications has the wrong shape"):
        db.open()


def test_v2_db_with_only_applications_is_ours(tmp_path: Path) -> None:
    # a v2 DB carrying ONLY the applications table (no shadow_results) is
    # ours by the shape gate — the applications validity stands alone
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (2)")
    conn.execute(
        "CREATE TABLE applications (message_id TEXT NOT NULL, label TEXT NOT NULL, "
        "label_id TEXT NOT NULL, score REAL NOT NULL, band TEXT NOT NULL, "
        "tier TEXT NOT NULL, applied_at TEXT NOT NULL, source TEXT NOT NULL "
        "CHECK (source IN ('daemon', 'apply')))"
    )
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    db.open()  # no refusal; shadow_results is created if-missing
    db.record_applications([_app_row()])
    assert len(db.applications_for_message("aa")) == 1
    db.close()


def test_applications_coeexists_with_unknown_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    db = AppDB(db_path)
    db.open()
    db.record_applications([_app_row()])
    db.close()
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE other_app_data (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    db2 = AppDB(db_path)
    db2.open()
    assert len(db2.applications_for_message("aa")) == 1
    db2.close()


def test_record_applications_roundtrip(tmp_path: Path) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    count = db.record_applications(
        [
            _app_row("bb", "Money/Bills", "Label_2", 0.7, "med", "ai-med"),
            _app_row("aa", "Shipping", "Label_1", 0.9, "high", "ai-high"),
            _app_row("aa", "Invoices", "Label_3", 0.8, "high", "ai-high"),
        ]
    )
    assert count == 3
    rows = db.applications_for_message("aa")
    assert [r.label for r in rows] == ["Invoices", "Shipping"]  # label-sorted tie
    assert rows[0].applied_at == TS  # aware UTC round-trip
    assert rows[0].source == "daemon"
    assert rows[0].score == 0.8
    by_label = db.applications_for_label("Money/Bills")
    assert [r.message_id for r in by_label] == ["bb"]
    assert db.applications_for_message("deadbeef") == ()  # empty is legitimate
    db.close()
    # the join read survives re-open (durability)
    db2 = AppDB(tmp_path / "state.db")
    db2.open()
    assert len(db2.applications_for_message("aa")) == 2  # the bb row is other's
    db2.close()


def test_record_applications_empty_is_refused(tmp_path: Path) -> None:
    # like record_run: an empty batch is refused at the writer
    db = AppDB(tmp_path / "state.db")
    db.open()
    with pytest.raises(AppDBError, match="no rows"):
        db.record_applications([])
    db.close()


def test_record_applications_allows_reapplication_history(tmp_path: Path) -> None:
    # NO PK on (message_id, label): a label manually removed and later
    # re-earned earns a NEW Application — the audit history is the
    # operator's, not a uniqueness constraint
    db = AppDB(tmp_path / "state.db")
    db.open()
    db.record_applications([_app_row()])
    db.record_applications([_app_row(applied_at=TS.replace(hour=13))])
    assert len(db.applications_for_message("aa")) == 2
    db.close()


def test_record_applications_row_validation(tmp_path: Path) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    bad: list[tuple[ApplicationRow, str]] = [
        (_app_row(message_id="zz"), "hex"),
        (_app_row(label="  "), "non-empty"),
        (_app_row(label_id=""), "non-empty"),
        (_app_row(score=1.5), r"out of \[0, 1\]"),
        (_app_row(score=True), "must be a number"),
        (_app_row(band="low"), "band"),
        (_app_row(band=None), "band"),
        (_app_row(tier=""), "non-empty"),
        (_app_row(source="shadow"), "source"),
        (_app_row(source="DAEMON"), "source"),
        (_app_row(applied_at=TS.replace(tzinfo=None)), "timezone-aware"),
        (
            ApplicationRow(  # a str applied_at: named refusal, not AttributeError
                message_id="aa",
                label="Shipping",
                label_id="Label_1",
                score=0.9,
                band="high",
                tier="ai-high",
                applied_at="yesterday",  # pyright: ignore[reportArgumentType]
                source="daemon",
            ),
            "must be a datetime",
        ),
    ]
    for row, cause in bad:
        with pytest.raises(AppDBError, match=cause):
            db.record_applications([row])
    # duplicate (message_id, label) WITHIN one batch is a composition bug
    with pytest.raises(AppDBError, match="duplicate"):
        db.record_applications([_app_row(), _app_row(label_id="Label_2")])
    assert db.applications_for_message("aa") == ()  # nothing half-written
    db.close()


def test_record_applications_one_transaction(tmp_path: Path) -> None:
    # the crash-safety contract (record_run's): a mid-batch INSERT
    # failure rolls back the whole batch — no half rows
    db = AppDB(tmp_path / "state.db")
    db.open()
    db.record_applications([_app_row("aa")])

    class _FlakyConn:
        # delegating proxy: the next executemany raises (a simulated
        # mid-write crash — sqlite3.Connection itself refuses attribute
        # injection)
        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real

        def executemany(self, *_a: object, **_k: object) -> None:
            raise sqlite3.OperationalError("disk full (simulated)")

        def __enter__(self) -> _FlakyConn:
            self._real.__enter__()
            return self

        def __exit__(self, *exc: object) -> object:
            return self._real.__exit__(*exc)  # type: ignore[arg-type,no-any-return]

        def __getattr__(self, name: str) -> object:
            return getattr(self._real, name)

    real_conn = db._conn
    assert real_conn is not None
    db._conn = _FlakyConn(real_conn)  # type: ignore[assignment]
    with pytest.raises(AppDBError, match="disk full"):
        db.record_applications([_app_row("bb"), _app_row("cc")])
    db._conn = real_conn
    assert len(db.applications_for_message("aa")) == 1  # prior batch intact
    assert db.applications_for_message("bb") == ()  # rolled back
    assert db.applications_for_message("cc") == ()
    db.close()


def test_applications_schema_has_no_body_columns(tmp_path: Path) -> None:
    # invariant #5, structural: the table carries the EIGHT metadata
    # columns and nothing else — no body/subject/raw column exists to
    # put a body in (the caplog side of the proof lands with the plan's
    # id-only warnings in applications.py)
    db = AppDB(tmp_path / "state.db")
    db.open()
    conn = sqlite3.connect(str(db.path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(applications)")}
        assert cols == _APPLICATIONS_COLUMNS
        assert not {"body", "subject", "raw", "snippet"} & cols
    finally:
        conn.close()
    db.close()


def test_applications_read_revalidates_tampered_rows(tmp_path: Path) -> None:
    # a hand-edited applications row (bad band / lying source) is
    # refused, named — not rendered (the load_run doctrine)
    db_path = tmp_path / "state.db"
    db = AppDB(db_path)
    db.open()
    db.record_applications([_app_row()])
    db.close()
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE applications SET band = 'low'")
    conn.commit()
    conn.close()
    db2 = AppDB(db_path)
    db2.open()
    with pytest.raises(AppDBError, match="band"):
        db2.applications_for_message("aa")
    db2.close()


def test_applications_tampered_applied_at_is_corrupt(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    db = AppDB(db_path)
    db.open()
    db.record_applications([_app_row()])
    db.close()
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE applications SET applied_at = 'not-a-date'")
    conn.commit()
    conn.close()
    db2 = AppDB(db_path)
    db2.open()
    with pytest.raises(AppDBError, match="not-a-date"):
        db2.applications_for_message("aa")
    db2.close()


# -- v3: the daemon_state cursor + the v2->v3 migration (ticket #25) ---------


def _v2_db(
    db_path: Path, *, with_run: bool = True, with_application: bool = True
) -> None:
    """A DB exactly as ticket #24 shipped it: schema_version {2} +
    shadow_results + applications (our shapes), NO daemon_state."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (2)")
    conn.execute(
        "CREATE TABLE shadow_results ("
        "run_id TEXT NOT NULL, message_id TEXT NOT NULL, label TEXT NOT NULL, "
        "probability REAL NOT NULL, band TEXT, run_ts TEXT NOT NULL, "
        "PRIMARY KEY (run_id, message_id, label))"
    )
    conn.execute(
        "CREATE TABLE applications (message_id TEXT NOT NULL, label TEXT NOT NULL, "
        "label_id TEXT NOT NULL, score REAL NOT NULL, band TEXT NOT NULL, "
        "tier TEXT NOT NULL, applied_at TEXT NOT NULL, source TEXT NOT NULL "
        "CHECK (source IN ('daemon', 'apply')))"
    )
    if with_run:
        conn.execute(
            "INSERT INTO shadow_results VALUES "
            "('run-1', 'aa', 'Shipping', 0.9, 'high', '2026-11-01T12:00:00+00:00')"
        )
    if with_application:
        conn.execute(
            "INSERT INTO applications VALUES ('aa', 'Shipping', 'Label_1', 0.9, "
            "'high', 'ai-high', '2026-11-01T12:00:00+00:00', 'daemon')"
        )
    conn.commit()
    conn.close()


def test_v2_db_migrates_to_a_single_v3_row(tmp_path: Path) -> None:
    # the v2→v3 transition (ticket #25): opening a live v2 DB with the
    # v3 tool CREATES daemon_state and RESTAMPS {2} → {3} in ONE
    # explicit transaction (the #24 machinery rides); recorded
    # applications + shadow rows SURVIVE (the restamp is metadata, the
    # data is the operator's)
    db_path = tmp_path / "state.db"
    _v2_db(db_path)
    db = AppDB(db_path)
    db.open()
    assert _versions(db_path) == [3]  # exactly one row, restamped
    conn = sqlite3.connect(str(db_path))
    try:
        state_cols = {r[1] for r in conn.execute("PRAGMA table_info(daemon_state)")}
        assert state_cols == {"history_cursor"}
    finally:
        conn.close()
    _run_ts, rows = db.load_run("run-1")
    assert len(rows) == 1
    assert len(db.applications_for_message("aa")) == 1
    assert db.get_history_cursor() is None  # the never-established state
    db.close()


def test_migrated_db_refused_by_the_v2_gate(tmp_path: Path, monkeypatch) -> None:
    # the OTHER direction, pinned exactly as v1→v2 was: the SAME
    # migrated file opened by the version-gate logic as a v2 tool
    # refuses it — {3} != {2}, named (the forward-compat hard gate)
    db_path = tmp_path / "state.db"
    _v2_db(db_path)
    migrator = AppDB(db_path)
    migrator.open()
    migrator.close()
    import gmail_labeler.appdb as appdb_module

    monkeypatch.setattr(appdb_module, "SCHEMA_VERSION", 2)
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match=r"schema_version \[3\] != 2"):
        db.open()


def test_v2_stamp_with_daemon_state_table_is_refused(tmp_path: Path) -> None:
    # the {2}-WITH-state dead-end is TAMPER-ONLY (the crash pin below
    # proves no crash reaches it — the atomic transition never commits
    # the table without the restamp): a v2 tool never wrote one
    db_path = tmp_path / "state.db"
    _v2_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE daemon_state (history_cursor INTEGER NOT NULL)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match=r"schema_version \[2\]") as exc:
        db.open()
    assert "UPDATE schema_version SET version = 3" in str(exc.value)


def test_v3_stamp_without_daemon_state_self_heals(tmp_path: Path) -> None:
    # {3} stamped but daemon_state never created (tamper-only since the
    # atomic transition — the v2-era self-heal's successor): the next
    # open re-runs create-all-if-missing and HEALS, NO restamp (the
    # version is already ours)
    db_path = tmp_path / "state.db"
    db = AppDB(db_path)
    db.open()
    db.close()
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE daemon_state")
    conn.commit()
    conn.close()
    db2 = AppDB(db_path)
    db2.open()
    assert _versions(db_path) == [3]  # healed, not restamped (it never moved)
    conn = sqlite3.connect(str(db_path))
    try:
        state_cols = {r[1] for r in conn.execute("PRAGMA table_info(daemon_state)")}
        assert state_cols == {"history_cursor"}
    finally:
        conn.close()
    db2.close()


def test_wrong_shape_daemon_state_is_refused(tmp_path: Path) -> None:
    # the ownership gate EXTENDED to the v3 table (the applications
    # pattern): a same-named daemon_state with the wrong column set is
    # refused, named
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (3)")
    conn.execute("CREATE TABLE daemon_state (id INTEGER, note TEXT)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="daemon_state has the wrong shape"):
        db.open()


def test_daemon_state_has_no_body_columns(tmp_path: Path) -> None:
    # invariant #5, structural: the cursor table carries the ONE
    # metadata column and nothing else — no body/subject/raw column
    # exists to put a body in
    db = AppDB(tmp_path / "state.db")
    db.open()
    conn = sqlite3.connect(str(db.path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(daemon_state)")}
        assert cols == {"history_cursor"}
        assert not {"body", "subject", "raw", "snippet"} & cols
    finally:
        conn.close()
    db.close()


def test_get_history_cursor_none_when_unset(tmp_path: Path) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    assert db.get_history_cursor() is None  # the never-established state
    db.close()


def test_set_history_cursor_roundtrip_is_single_row(tmp_path: Path) -> None:
    # the single-row model AT THE WRITER: every set UPSERTS rowid 1 —
    # three sets, one row (the table never grows; a lower value is a
    # caller decision, the store keeps the last write)
    db = AppDB(tmp_path / "state.db")
    db.open()
    for cursor in (10, 20, 15):
        db.set_history_cursor(cursor)
    assert db.get_history_cursor() == 15
    conn = sqlite3.connect(str(db.path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM daemon_state").fetchone() == (1,)
    finally:
        conn.close()
    db.close()
    # durability across re-open
    db2 = AppDB(tmp_path / "state.db")
    db2.open()
    assert db2.get_history_cursor() == 15
    db2.close()


def test_set_history_cursor_refusals_write_nothing(tmp_path: Path) -> None:
    # the cursor is an unsigned int64 per the wire doctrine (a negative
    # historyId must never silently advance anything); a bool is not a
    # number (the config discipline); refusals land BEFORE any write
    db = AppDB(tmp_path / "state.db")
    db.open()
    for bad in (0, -1, True, 1.5, "5", None):
        with pytest.raises(AppDBError, match="history_cursor"):
            db.set_history_cursor(bad)  # pyright: ignore[reportArgumentType]
    assert db.get_history_cursor() is None
    db.close()


def test_daemon_state_multi_row_tamper_is_refused_on_read(tmp_path: Path) -> None:
    # the single-row model's READ gate: the writer upserts rowid 1, so
    # a second row is hand-made — refused, named (never silently picked)
    db_path = tmp_path / "state.db"
    db = AppDB(db_path)
    db.open()
    db.set_history_cursor(10)
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO daemon_state (rowid, history_cursor) VALUES (2, 20)")
    conn.commit()
    conn.close()
    with pytest.raises(AppDBError, match="single-row") as exc:
        db.get_history_cursor()
    # F1 MINOR 3: the module doctrine is recovery in every refusal
    assert "DELETE FROM daemon_state" in str(exc.value)
    db.close()


def test_daemon_state_tampered_cursor_is_refused_on_read(tmp_path: Path) -> None:
    # the tamper doctrine (the load_run precedent): a hand-edited
    # cursor value is refused, named — not served to the poll loop
    db_path = tmp_path / "state.db"
    db = AppDB(db_path)
    db.open()
    db.set_history_cursor(10)
    db.close()
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE daemon_state SET history_cursor = 0")
    conn.commit()
    conn.close()
    db2 = AppDB(db_path)
    db2.open()
    with pytest.raises(AppDBError, match="history_cursor"):
        db2.get_history_cursor()
    db2.close()


def test_set_history_cursor_ceiling_refused(tmp_path: Path) -> None:
    # the wire type (uint64) is wider than SQLite's signed INTEGER: an
    # unbounded value would escape as a raw OverflowError at bind time,
    # past every named-error wrapper (review finding) — refused here
    db = AppDB(tmp_path / "state.db")
    db.open()
    with pytest.raises(AppDBError, match="SQLite INTEGER range"):
        db.set_history_cursor(2**63)
    assert db.get_history_cursor() is None
    # the top of the supported range still lands
    db.set_history_cursor(2**63 - 1)
    assert db.get_history_cursor() == 2**63 - 1
    db.close()


def test_daemon_state_off_rowid_tamper_is_refused_on_read(tmp_path: Path) -> None:
    # the single-row model's rowid gate: the writer only ever touches
    # rowid 1, so a lone row elsewhere is hand-made (and the next
    # legitimate write would make the table two rows) — refused,
    # named, recovery given (review finding)
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (3)")
    conn.execute("CREATE TABLE daemon_state (history_cursor INTEGER NOT NULL)")
    conn.execute("INSERT INTO daemon_state (rowid, history_cursor) VALUES (7, 42)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    db.open()
    with pytest.raises(AppDBError, match="rowid 7") as exc:
        db.get_history_cursor()
    assert "DELETE FROM daemon_state WHERE rowid != 1" in str(exc.value)
    db.close()


def test_commit_page_refuses_mixed_applied_at(tmp_path: Path) -> None:
    # F1 MINOR 2 (both families): the #24 batch-homogeneity contract is
    # now a STORE invariant — a commit_page batch carrying >1 distinct
    # applied_at is refused NAMED before the transaction, so #26's
    # future caller inherits the guard instead of rediscovering it
    db = AppDB(tmp_path / "state.db")
    db.open()
    db.set_history_cursor(10)
    rows = [
        _app_row(message_id="a1", applied_at=TS),
        _app_row(
            message_id="b2",
            label="Other",
            applied_at=datetime(2026, 11, 1, 12, 0, 1, tzinfo=UTC),
        ),
    ]
    with pytest.raises(AppDBError, match="distinct applied_at") as exc:
        db.commit_page(rows, 20)
    assert "ONE clock reading" in str(exc.value)
    assert db.get_history_cursor() == 10  # refused BEFORE the transaction
    assert db.applications_for_message("a1") == ()
    db.close()


def test_read_path_cursor_tamper_names_recovery(tmp_path: Path) -> None:
    # F1 MINOR 3: a hand-set invalid cursor value surfacing on the READ
    # is the operator's to repair — the refusal names the recovery (the
    # write-path refusals carry none: there the value came from the
    # caller, not from state damage)
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (3)")
    conn.execute("CREATE TABLE daemon_state (history_cursor INTEGER NOT NULL)")
    conn.execute("INSERT INTO daemon_state (rowid, history_cursor) VALUES (1, 0)")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    db.open()
    with pytest.raises(AppDBError, match="not a valid cursor") as exc:
        db.get_history_cursor()
    assert "recovery" in str(exc.value)
    db.close()


def test_unknown_stamp_set_refusal_names_the_honest_diagnosis(tmp_path: Path) -> None:
    # F1 MINOR 3: the blanket "a newer tool wrote this DB" diagnosed
    # states NO tool can write as if a tool had. A single FUTURE stamp
    # keeps the newer-tool wording (the sealed [99] pins hold); an
    # unwriteable set gets the honest diagnosis + recovery
    def _stamped(name: str, stamps: list[int]) -> Path:
        path = tmp_path / name
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.executemany(
            "INSERT INTO schema_version (version) VALUES (?)", [(v,) for v in stamps]
        )
        conn.commit()
        conn.close()
        return path

    for stamps in ([2, 3], [0], [-1]):
        db = AppDB(_stamped("db-" + "-".join(map(str, stamps)) + ".db", stamps))
        with pytest.raises(AppDBError, match="unknown stamp set"):
            db.open()
    db = AppDB(_stamped("db-future.db", [99]))
    with pytest.raises(AppDBError, match="a newer tool wrote") as exc:
        db.open()
    assert "upgrade this tool" in str(exc.value)


def test_commit_page_rows_and_cursor_land_together(tmp_path: Path) -> None:
    # the durable-handling commit point (#25, invariant #3): ONE
    # transaction carries the page's Application rows AND the cursor
    # advance — COMMIT = handled
    db = AppDB(tmp_path / "state.db")
    db.open()
    count = db.commit_page(
        [
            _app_row("aa"),
            _app_row("bb", "Money/Bills", "Label_2", 0.7, "med", "ai-med"),
        ],
        12345,
    )
    assert count == 2
    assert db.get_history_cursor() == 12345
    assert len(db.applications_for_message("aa")) == 1
    assert len(db.applications_for_message("bb")) == 1
    db.close()
    db2 = AppDB(tmp_path / "state.db")
    db2.open()
    assert db2.get_history_cursor() == 12345  # durability
    db2.close()


def test_commit_page_empty_rows_still_advances_cursor(tmp_path: Path) -> None:
    # a page whose messages all composed EMPTY plans (skip-present /
    # no-band) still advances the cursor — record_applications'
    # empty-batch refusal is deliberately NOT inherited here: the page
    # WAS handled, the cursor must move
    db = AppDB(tmp_path / "state.db")
    db.open()
    assert db.commit_page([], 999) == 0
    assert db.get_history_cursor() == 999
    db.close()


def test_commit_page_rolls_back_rows_and_cursor_together(tmp_path: Path) -> None:
    # the ATOMICITY pin (the crash matrix's mid-transaction row): the
    # cursor UPSERT fails AFTER the rows inserted — the ONE transaction
    # rolls BOTH back. A split-transaction mutant leaves the rows
    # committed with the cursor stale — the replay then DOUBLE-WRITES
    # the rows (the exact failure this pin exists to kill)
    db = AppDB(tmp_path / "state.db")
    db.open()

    class _FlakyConn:
        # delegating proxy: the daemon_state UPSERT raises (a simulated
        # mid-write crash — the _FlakyConn precedent, marker-side)
        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real

        def execute(self, sql: str, *a: object) -> object:
            if "daemon_state" in sql:
                raise sqlite3.OperationalError("disk full (simulated)")
            return self._real.execute(sql, *a)  # pyright: ignore[reportArgumentType]

        def __enter__(self) -> _FlakyConn:
            self._real.__enter__()
            return self

        def __exit__(self, *exc: object) -> object:
            return self._real.__exit__(*exc)  # type: ignore[arg-type,no-any-return]

        def __getattr__(self, name: str) -> object:
            return getattr(self._real, name)

    real_conn = db._conn
    assert real_conn is not None
    db._conn = _FlakyConn(real_conn)  # type: ignore[assignment]
    with pytest.raises(AppDBError, match="disk full"):
        db.commit_page([_app_row("aa")], 12345)
    db._conn = real_conn
    assert db.applications_for_message("aa") == ()  # rows rolled back
    assert db.get_history_cursor() is None  # cursor NOT advanced
    db.close()


def test_commit_page_refusals_write_nothing(tmp_path: Path) -> None:
    # refusals BEFORE the transaction (the record_applications doctrine
    # + the cursor refusals): a lying row, a within-batch duplicate, a
    # bad cursor — nothing lands, the cursor untouched
    db = AppDB(tmp_path / "state.db")
    db.open()
    with pytest.raises(AppDBError, match="hex"):
        db.commit_page([_app_row(message_id="zz")], 5)
    with pytest.raises(AppDBError, match="duplicate"):
        db.commit_page([_app_row("aa"), _app_row("aa", label_id="Label_2")], 5)
    for bad in (0, -1, True):
        with pytest.raises(AppDBError, match="history_cursor"):
            db.commit_page([_app_row("aa")], bad)  # pyright: ignore[reportArgumentType]
    assert db.applications_for_message("aa") == ()
    assert db.get_history_cursor() is None
    db.close()


# -- F1: atomic schema transitions (the crash windows closed) ---------------


class _CrashOn:
    """Delegating proxy: the execute()/executescript() whose SQL carries
    `marker` raises (a simulated process death mid-transaction — the
    journal, not a ROLLBACK, is what a real crash leaves behind; the
    close-without-commit below reproduces the same file state)."""

    def __init__(self, real: sqlite3.Connection, marker: str) -> None:
        self._real = real
        self._marker = marker

    def execute(self, sql: str, *a: object) -> object:
        if self._marker in sql:
            raise sqlite3.OperationalError("simulated crash mid-transaction")
        return self._real.execute(sql, *a)  # pyright: ignore[reportArgumentType]

    def executescript(self, script: str) -> object:
        if self._marker in script:
            raise sqlite3.OperationalError("simulated crash mid-transaction")
        return self._real.executescript(script)

    def __enter__(self) -> object:
        return self._real.__enter__()

    def __exit__(self, *exc: object) -> object:
        return self._real.__exit__(*exc)  # type: ignore[arg-type,no-any-return]

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def _arm_crash_on(monkeypatch: pytest.MonkeyPatch, marker: str) -> None:
    # the seam: arm the proxy right after the ownership gate (the last
    # point before the schema transition begins writing)
    real_check = AppDB._check_ownership

    def check_then_arm(self: AppDB) -> int | None:
        needs = real_check(self)
        assert self._conn is not None
        self._conn = _CrashOn(self._conn, marker)  # type: ignore[assignment]
        return needs

    monkeypatch.setattr(AppDB, "_check_ownership", check_then_arm)


def _table_set(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


def test_fresh_db_crash_mid_transition_leaves_clean_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F1 MAJOR pin (1): kill anywhere inside the fresh transition (here:
    # the stamp INSERT after the table DDL) — ONE transaction means the
    # file is fully PRIOR-STATE (no tables), never the empty-version
    # corrupt trap the old executescript-then-stamp order shipped. The
    # next open succeeds CLEANLY as fresh: no deletion, no refusal.
    _arm_crash_on(monkeypatch, "INSERT INTO schema_version")
    db_path = tmp_path / "state.db"
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="simulated crash"):
        db.open()
    assert _table_set(db_path) == set()  # fully prior-state
    monkeypatch.undo()  # the crash seam is dead — the retry opens for real
    db2 = AppDB(db_path)
    db2.open()  # clean fresh — the crash left nothing to recover from
    db2.record_run("run-1", TS, [_row()])  # and it is fully usable
    db2.close()


def test_v1_migration_crash_mid_transition_leaves_clean_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F1 MAJOR pin (2): kill mid-migration, on the applications INDEX —
    # AFTER the applications CREATE TABLE has been attempted but BEFORE
    # the restamp UPDATE, so a non-transactional transition has already
    # committed the table here -> {1}+applications (dead-end refusal).
    # A leaked/early-committed restamp (the pre-F1 shape) shows up in
    # the VERSION assertion below instead. ONE transaction: the file
    # stays CLEAN v1 and the retry migrates successfully — shadow
    # history survives, no delete-recovery, no manual stamp. (Marker
    # position matters: on the applications CREATE itself the test
    # stays green under the split-transaction mutant — the table has
    # not been attempted yet at that point — F1 review MAJOR; a COMMIT
    # marker misses the leaked-restamp shape — the index placement
    # discriminates both, per the round-2 probes.)
    _arm_crash_on(monkeypatch, "CREATE INDEX IF NOT EXISTS idx_applications_message_id")
    db_path = tmp_path / "state.db"
    _v1_db(db_path)
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="simulated crash"):
        db.open()
    assert _versions(db_path) == [1]  # fully prior-state: the restamp rolled back
    assert "applications" not in _table_set(db_path)
    monkeypatch.undo()  # the crash seam is dead — the retry opens for real
    db2 = AppDB(db_path)
    db2.open()  # the retry sees clean v1 and migrates
    assert _versions(db_path) == [3]
    _run_ts, rows = db2.load_run("run-1")  # shadow history SURVIVES
    assert len(rows) == 1
    db2.record_applications([_app_row()])  # fully usable v3
    db2.close()


def test_v2_migration_crash_mid_transition_leaves_clean_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the v2→v3 crash pin (the F1 precedent, marker discipline LEARNED
    # from the v1→v2 round-2 probe): kill on the restamp UPDATE —
    # daemon_state is the transition's LAST DDL, so a non-transactional
    # transition (split BEGIN, or the pre-F1 executescript shape) has
    # ALREADY committed the table here → {2}+daemon_state (the tamper
    # dead-end above) and the table-set assertion reddens. ONE
    # transaction: the file stays CLEAN v2 and the retry migrates —
    # applications + shadow history survive, no delete-recovery, no
    # manual stamp. (Mutant probe on record: a marker on the
    # daemon_state CREATE itself SURVIVED the split-BEGIN mutant — the
    # crash fires before the statement executes, so the table is never
    # attempted; the UPDATE marker is what discriminates. The v1→v2 pin
    # needed the INDEX marker for the opposite reason: its new table
    # was followed by indexes. Both placements are statement-order
    # facts, executed — not reasoned.)
    _arm_crash_on(monkeypatch, "UPDATE schema_version SET version")
    db_path = tmp_path / "state.db"
    _v2_db(db_path)
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match="simulated crash"):
        db.open()
    assert _versions(db_path) == [2]  # fully prior-state: the restamp rolled back
    assert "daemon_state" not in _table_set(db_path)
    monkeypatch.undo()  # the crash seam is dead — the retry opens for real
    db2 = AppDB(db_path)
    db2.open()  # the retry sees clean v2 and migrates
    assert _versions(db_path) == [3]
    _run_ts, rows = db2.load_run("run-1")  # shadow history SURVIVES
    assert len(rows) == 1
    assert len(db2.applications_for_message("aa")) == 1  # applications SURVIVE
    db2.close()


def test_executescript_is_banned_from_appdb_production_source() -> None:
    # F2 NIT: the executescript ban (the crash-safety premise of the
    # atomic transition) was pinned only INDIRECTLY by the crash pins;
    # this is the DIRECT assertion — no CALL of executescript exists in
    # the module's source (the prose mentions in docstrings/comments
    # carry no call parens; the joined _SCHEMA text is fixture-only)
    import gmail_labeler.appdb as appdb_module

    source = Path(appdb_module.__file__).read_text(encoding="utf-8")
    assert "executescript(" not in source


def test_connection_isolation_mode_is_pinned(tmp_path: Path) -> None:
    # the atomic transition depends on the connect DEFAULT (legacy
    # isolation mode: implicit BEGIN before DML only — DDL is NEVER
    # auto-wrapped, which is exactly why _apply_schema issues an
    # explicit BEGIN and why executescript (implicit COMMIT first) is
    # banned from it). If a future edit flips isolation/autocommit,
    # this pin reddens before the crash tests silently stop meaning
    # what they say.
    db = AppDB(tmp_path / "state.db")
    db.open()
    assert db._conn is not None
    assert db._conn.isolation_level == ""
    # the journal side of the crash story (F2 NIT, mode verified at
    # runtime BEFORE the pin, not a pasted guess): DELETE journal mode
    # is what rolls the uncommitted transition back for a dead process
    assert db._conn.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    db.close()


def test_tamper_refusals_name_their_recovery(tmp_path: Path) -> None:
    # F1 (qwen MINOR): the two guards that STAY are tamper guards (no
    # crash reaches them once the transitions are atomic) — and each
    # message names the recovery operator-visibly, not just the module
    # docstring.
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    with pytest.raises(AppDBError, match="schema_version is empty") as exc:
        AppDB(db_path).open()
    # the recovery is RUNNABLE SQL, not prose-about-SQL (F2)
    assert "INSERT INTO schema_version (version) VALUES (3)" in str(exc.value)
    db_path.unlink()
    _v1_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE applications (message_id TEXT, label TEXT, label_id TEXT, "
        "score REAL, band TEXT, tier TEXT, applied_at TEXT, source TEXT)"
    )
    conn.commit()
    conn.close()
    with pytest.raises(AppDBError, match=r"schema_version \[1\]") as exc:
        AppDB(db_path).open()
    assert "UPDATE schema_version SET version = 3" in str(exc.value)
