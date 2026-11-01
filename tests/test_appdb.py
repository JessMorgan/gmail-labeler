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


def test_fresh_db_is_v2_with_applications(tmp_path: Path) -> None:
    db = AppDB(tmp_path / "state.db")
    db.open()
    assert _versions(db.path) == [SCHEMA_VERSION] == [2]
    conn = sqlite3.connect(str(db.path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(applications)")}
        assert cols == _APPLICATIONS_COLUMNS
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


def test_v1_db_migrates_to_a_single_v2_row(tmp_path: Path) -> None:
    # the migration story (#23's recorded debt): opening a live v1 DB
    # with the v2 tool CREATES the applications table and RESTAMPS the
    # version row to 2 — single-row model: UPDATE, never a second row
    db_path = tmp_path / "state.db"
    _v1_db(db_path)
    db = AppDB(db_path)
    db.open()
    assert _versions(db_path) == [2]  # exactly one row, restamped
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(applications)")}
        assert cols == _APPLICATIONS_COLUMNS
    finally:
        conn.close()
    # the recorded shadow run SURVIVES the migration (the data is the
    # operator's, the restamp is metadata)
    _run_ts, rows = db.load_run("run-1")
    assert len(rows) == 1
    db.close()
    # re-opening in-process is v2-clean (the tool that just migrated IS
    # the v2 tool)
    db2 = AppDB(db_path)
    db2.open()
    db2.close()


def test_migrated_db_refused_by_the_v1_gate(tmp_path: Path, monkeypatch) -> None:
    # the OTHER direction of the pin: the SAME migrated file opened by
    # the version-gate logic as a v1 tool refuses it — {2} != {1},
    # named. The forward-compat story IS the hard gate (documented
    # honestly: our own v2 is the tool that just migrated it)
    db_path = tmp_path / "state.db"
    _v1_db(db_path)
    migrator = AppDB(db_path)
    migrator.open()
    migrator.close()
    import gmail_labeler.appdb as appdb_module

    monkeypatch.setattr(appdb_module, "SCHEMA_VERSION", 1)
    db = AppDB(db_path)
    with pytest.raises(AppDBError, match=r"schema_version \[2\] != 1"):
        db.open()


def test_v1_stamp_with_applications_table_is_refused(tmp_path: Path) -> None:
    # the single-row model's inconsistent state: a v1 tool never writes
    # an applications table — {1} + applications is corrupt, refused
    # named (the crash between the restamp UPDATE and the table creation
    # is self-healing; THIS direction is not, and refuses)
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


def test_v2_stamp_without_applications_self_heals(tmp_path: Path) -> None:
    # the crash-window direction that IS safe: {2} stamped but the
    # applications table never created (crash between the UPDATE and the
    # schema script) — the next open re-runs create-all-if-missing and
    # heals, no refusal
    db_path = tmp_path / "state.db"
    _v1_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE schema_version SET version = 2")
    conn.commit()
    conn.close()
    db = AppDB(db_path)
    db.open()
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(applications)")}
        assert cols == _APPLICATIONS_COLUMNS
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
