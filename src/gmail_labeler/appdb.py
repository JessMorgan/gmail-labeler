"""State DB adapter (ticket #23; GLOSSARY: Shadow Pass).

The SQLite file behind ``[store] db_path``: the shadow_results table
(the Shadow Pass's recorded would-apply verdicts) and, with the daemon
phase, the applications table (#24 — same file; the seam is the
schema_version row + the comment below).

UPGRADE STORY (stated honestly, fold-4 item 6): ``SCHEMA_VERSION`` stays
1 this ticket. The v1→v2 restamp/migration path (adding the applications
table) is DELIBERATELY UNSOLVED pending #24's schema design — no sentence
here implies the versioning story is complete. Meanwhile the HARD version
gate (the ownership check refuses a foreign/future schema_version, named)
is what protects cross-version reads: a v2 DB is not consumed by a v1 tool.

Invariants:

- NO BODIES (invariant #5): the rows carry message ids, label names,
  probabilities, band verdicts, and run timestamps ONLY — the DB must
  be publicly attachable (the corpus-sidecar discipline);
- TRANSACTION PER RUN: ``record_run`` inserts a run's rows in ONE
  transaction — a crash mid-run (or a mid-run refusal) rolls back,
  leaving no half rows; a committed run is atomic;
- create-all-if-missing: opening a fresh path creates the schema;
  opening an existing DB verifies the schema_version (a foreign
  version is refused, named — the SplitManifest discipline);
- deterministic reads: ``load_run`` returns rows in (message_id,
  label) order — the report's total order.
"""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "SCHEMA_VERSION",
    "AppDB",
    "AppDBError",
    "RunNotFoundError",
    "ShadowRow",
]

#: The schema version (bump on any change to the tables' field set or
#: semantics). The #24 applications table lands under a v2 restamp —
#: the v1→v2 restamp/migration path is UNSOLVED pending #24's schema
#: design (see the module docstring's UPGRADE STORY); the HARD version
#: gate below is what protects cross-version reads meanwhile.
SCHEMA_VERSION = 1

# Gmail message ids are hex (the CorpusRecord._ID_RE precedent): a
# non-hex id is a lying row, refused at the adapter boundary.
_ID_RE = re.compile(r"^[0-9a-fA-F]{1,64}$")

# The band verdict names (tiers.BAND_HIGH/BAND_MED — mirrored, not
# imported: appdb stays import-light, and the equality is pinned by
# test_appdb).
_BANDS = ("high", "med")

# The table SHAPES the ownership gate requires (disposition 2 —
# ownership is by shape, not by name): the exact column-name sets.
_SHADOW_RESULTS_COLUMNS = frozenset(
    {"run_id", "message_id", "label", "probability", "band", "run_ts"}
)
_SCHEMA_VERSION_COLUMNS = frozenset({"version"})


class AppDBError(Exception):
    """State DB failure: a foreign schema version, a corrupt file, a
    validation refusal, or an I/O failure. The CLI maps it to exit 5
    (operation failed) — never a raw sqlite3 traceback."""


class RunNotFoundError(AppDBError):
    """A named NOT-FOUND refusal: the requested run id has no rows (the
    run was never committed, or the id is wrong). Distinct from an I/O
    failure so the CLI can name the cause directly (not wrap it in
    "state DB read failed" — the cause-naming doctrine)."""


@dataclass(frozen=True, slots=True)
class ShadowRow:
    """One would-apply verdict row (NO bodies — invariant #5): the
    message id, the label, the sigmoid probability, and the band verdict
    (``"high"`` / ``"med"`` / ``None`` — the explicit no-verdict state).
    ``run_ts`` is the run's UTC timestamp — ``None`` on the write path
    (``record_run`` stamps the run-level timestamp on every row) and
    populated on the read path (``load_run``)."""

    message_id: str
    label: str
    probability: float
    band: str | None
    run_ts: datetime | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS shadow_results (
    run_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    label TEXT NOT NULL,
    probability REAL NOT NULL,
    band TEXT,
    run_ts TEXT NOT NULL,
    PRIMARY KEY (run_id, message_id, label)
);
-- Seam (ticket #24): the applications table lands in THIS file under a
-- v2 restamp — the v1->v2 restamp/migration is UNSOLVED pending #24's
-- schema design (the HARD version gate protects cross-version reads
-- meanwhile). No write surface exists here until #24/#25 (the Shadow
-- Pass is read-only).
"""


class AppDB:
    """The state DB adapter (bounded I/O, CODING-STANDARDS). One
    connection per instance (single-process contract, ADR-0002);
    ``open`` creates the schema if missing and verifies the version;
    ``record_run`` is the ONLY writer (transaction-per-run); the
    ``latest_run``/``load_run`` reads serve the report."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ------------------------------------------------------

    def open(self) -> None:
        """Connect + create-all-if-missing. A fresh path creates the
        schema (0600 from creation — the secret-tight discipline, even
        though the rows are attachable-public); an existing file is
        verified BEFORE any write (the ownership gate: a SQLite file
        that carries foreign tables and NONE of ours is refused, named
        — a foreign DB is not silently adopted; a foreign
        schema_version is refused, named). Cause naming (the #22
        doctrine): a directory / unwritable / unreadable path is "state
        DB cannot be opened (I/O failure)"; genuinely non-SQLite bytes
        are "state DB corrupt (not a SQLite file)" — the two are not
        conflated."""
        if self._conn is not None:
            return
        if not self.path.parent.is_dir():
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise AppDBError(
                    f"state DB cannot be opened (I/O failure): "
                    f"dir unusable {self.path.parent}: {e}"
                ) from e
        existed = self.path.exists()
        try:
            self._conn = sqlite3.connect(str(self.path))
            # connect is LAZY: a non-SQLite file is not read until the
            # first query, so force it here — the failure is named at
            # the site that actually raises (not a generic "unusable"
            # later). A directory / unwritable path fails at connect
            # ("unable to open database file" → I/O failure); non-SQLite
            # bytes fail at the forced read ("file is not a database" →
            # corrupt).
            self._conn.execute("SELECT 1").fetchone()
        except sqlite3.DatabaseError as e:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            if "file is not a database" in str(e):
                raise AppDBError(
                    f"state DB corrupt (not a SQLite file): {self.path}: {e}"
                ) from e
            raise AppDBError(
                f"state DB cannot be opened (I/O failure): {self.path}: {e}"
            ) from e
        if not existed:
            with contextlib.suppress(OSError):
                os.chmod(self.path, 0o600)  # foreign fs: best-effort
        try:
            # the OWNERSHIP gate (shape + version) BEFORE any write: a
            # SQLite file that carries foreign tables (or a
            # wrong-shape / foreign-version same-named table) is
            # refused, named — the "verified before any write" claim is
            # literally true (a foreign DB is not adopted, and a
            # versioned-foreign DB does not gain a shadow_results table
            # pre-refusal). An empty file (no tables) is fresh.
            self._check_ownership()
            self._conn.executescript(_SCHEMA)
            # a fresh DB (no version row) gets stamped; an existing
            # ours-DB is NOT re-stamped (the version is already
            # {SCHEMA_VERSION} — verified in the ownership gate above)
            versions = self._conn.execute(
                "SELECT DISTINCT version FROM schema_version"
            ).fetchall()
            if not versions:
                self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
                self._conn.commit()
        except (sqlite3.DatabaseError, AppDBError) as e:
            # BOTH the I/O failure and the ownership/version refusal
            # close the connection (open()'s lifecycle contract — a
            # refused open leaves no open conn behind)
            self._conn.close()
            self._conn = None
            if isinstance(e, AppDBError):
                raise
            raise AppDBError(
                f"state DB cannot be opened (I/O failure): {self.path}: {e}"
            ) from e

    def _check_ownership(self) -> None:
        # the ownership gate (disposition 2 + fold-4 item 1): ownership
        # is by SHAPE, not by table NAME, and the version is a HARD gate
        # (not advisory). Everything is read BEFORE any write (the
        # "verified before any write" claim is literally true — a
        # foreign DB is not adopted, and a versioned-foreign DB does not
        # gain a shadow_results table pre-refusal):
        #
        # - shadow_results present → its PRAGMA column set must EQUAL
        #   ours (a same-named wrong-shape table is refused, named);
        # - schema_version present → it must be the single-column
        #   `version` table (a generic same-named table is NOT ours —
        #   the adoption hole ds reproduced), and the version HARD gate
        #   applies: the version set must EQUAL {SCHEMA_VERSION} — a
        #   foreign version ({99}), a polluted set ({1,99}), or an
        #   our-shape-but-EMPTY table is refused, named (the version is
        #   NOT advisory: a v2 DB is not consumed by a v1 tool);
        # - a DB with a valid ours-table PLUS UNKNOWN extra tables
        #   COEXISTS (documented decision — our tables are the
        #   authority; the unknowns are left alone). Coexist is for
        #   UNKNOWN TABLES only — a version disagreement NEVER
        #   coexists (it is refused above);
        # - a DB with tables but NONE valid-ours is FOREIGN — refused,
        #   named (the foreign table names are listed);
        # - an empty file (no tables) is fresh.
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        shadow_results_valid = False
        schema_version_present = False
        if "shadow_results" in tables:
            cols = self._column_set("shadow_results")
            if cols != _SHADOW_RESULTS_COLUMNS:
                raise AppDBError(
                    f"state DB shadow_results has the wrong shape "
                    f"(columns {sorted(cols)} != {sorted(_SHADOW_RESULTS_COLUMNS)}) "
                    f"— refusing to adopt it: {self.path}"
                )
            shadow_results_valid = True
        if "schema_version" in tables:
            schema_version_present = True
            cols = self._column_set("schema_version")
            if cols != _SCHEMA_VERSION_COLUMNS:
                # a generic same-named table (not our single-column
                # `version` table) is NOT ours — the adoption hole
                raise AppDBError(
                    f"state DB schema_version has the wrong shape "
                    f"(columns {sorted(cols)} != {sorted(_SCHEMA_VERSION_COLUMNS)}) "
                    f"— a generic table, not a gmail-labeler state DB: "
                    f"{self.path}"
                )
            # the version HARD gate (fold-4 item 1): our-shape
            # schema_version must be stamped with EXACTLY our version —
            # the version is NOT advisory (a v2 DB is not consumed by a
            # v1 tool; the parent's _verify_version refused exactly this)
            versions = {
                row[0]
                for row in self._conn.execute(
                    "SELECT DISTINCT version FROM schema_version"
                ).fetchall()
            }
            if versions != {SCHEMA_VERSION}:
                # key=repr: a hostile mixed-affinity table must not raise
                # a raw TypeError out of open() (the shape check does not
                # constrain column types)
                if not versions:
                    # our-shape but EMPTY (no version row) — corrupt, named
                    raise AppDBError(
                        f"state DB schema_version is empty (our shape, no "
                        f"version row) — corrupt: {self.path}"
                    )
                # a foreign version ({99}) or a polluted set ({1,99}) —
                # the version cause is named (restored from the parent's
                # _verify_version)
                raise AppDBError(
                    f"state DB schema_version {sorted(versions, key=repr)} != "
                    f"{SCHEMA_VERSION}: {self.path} — a newer tool wrote "
                    f"this DB"
                )
        if shadow_results_valid:
            return  # ours (coexist with any UNKNOWN extra tables)
        if schema_version_present:
            return  # ours (our-shape schema_version stamped with our version)
        if tables:
            raise AppDBError(
                f"state DB is not a gmail-labeler state DB (foreign "
                f"tables: {sorted(tables)}) — refusing to adopt it: "
                f"{self.path}"
            )
        # no tables → fresh (proceed to create the schema)

    def _column_set(self, table: str) -> set[str]:
        # the PRAGMA column-name set for a table (shape, not values)
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        return {
            row[1]
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- writing (transaction per run) ----------------------------------

    def record_run(
        self, run_id: str, run_ts: datetime, rows: Sequence[ShadowRow]
    ) -> int:
        """Record one run's rows in ONE transaction (the crash-safety
        contract: a mid-run refusal or crash rolls back — no half
        rows). ``run_ts`` is the run's UTC timestamp (aware; normalized
        to UTC — the split_date doctrine). Refusals (a blank run_id, a
        naive run_ts, a row failing validation, a duplicate
        (message_id, label) within the run) raise AppDBError BEFORE the
        transaction — nothing is written. Returns the row count."""
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        if not run_id.strip():
            raise AppDBError("run_id must be non-empty")
        if not rows:
            # an empty run is refused at the ONLY writer (load_run's
            # docstring assumes "an empty run was never committed" — a
            # future caller #24/#26 cannot create the state the reader
            # assumes impossible)
            raise AppDBError(
                "record_run: no rows — an empty run is never committed "
                "(the CLI refuses a zero-verdict run before the store)"
            )
        if run_ts.tzinfo is None:
            raise AppDBError(
                "run_ts must be timezone-aware (a naive timestamp is a "
                "caller bug — the split_date doctrine)"
            )
        run_ts = run_ts.astimezone(UTC)
        for row in rows:
            _check_row(row)
        seen: set[tuple[str, str]] = set()
        for row in rows:
            key = (row.message_id, row.label)
            if key in seen:
                raise AppDBError(
                    f"duplicate row for {row.message_id} / {row.label} in "
                    "one run — a message has one verdict per label"
                )
            seen.add(key)
        ts = run_ts.isoformat()
        try:
            # ONE transaction: the context manager commits on success
            # and rolls back on ANY exception — a crash mid-run leaves
            # no half rows (the atomicity contract, pinned by test)
            with self._conn:
                self._conn.executemany(
                    "INSERT INTO shadow_results "
                    "(run_id, message_id, label, probability, band, run_ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            run_id,
                            row.message_id,
                            row.label,
                            row.probability,
                            row.band,
                            ts,
                        )
                        for row in rows
                    ],
                )
        except sqlite3.DatabaseError as e:
            raise AppDBError(f"state DB write failed: {self.path}: {e}") from e
        return len(rows)

    # -- reading ---------------------------------------------------------

    def latest_run(self) -> str | None:
        """The most recent run id (by run_ts, then run_id — a total
        order), or ``None`` when no runs are recorded."""
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        try:
            row = self._conn.execute(
                "SELECT run_id FROM shadow_results "
                "GROUP BY run_id ORDER BY MAX(run_ts) DESC, run_id DESC "
                "LIMIT 1"
            ).fetchone()
        except sqlite3.DatabaseError as e:
            raise AppDBError(f"state DB read failed: {self.path}: {e}") from e
        return row[0] if row is not None else None

    def run_exists(self, run_id: str) -> bool:
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        try:
            row = self._conn.execute(
                "SELECT 1 FROM shadow_results WHERE run_id = ? LIMIT 1", (run_id,)
            ).fetchone()
        except sqlite3.DatabaseError as e:
            raise AppDBError(f"state DB read failed: {self.path}: {e}") from e
        return row is not None

    def load_run(self, run_id: str) -> tuple[datetime, list[ShadowRow]]:
        """One run's rows in (message_id, label) order (the report's
        total order) + the run's timestamp. A run with no rows is
        refused, named (an empty run was never committed — a run with
        zero scored messages is exit 5 at the CLI, never stored)."""
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        try:
            rows = self._conn.execute(
                "SELECT message_id, label, probability, band, run_ts "
                "FROM shadow_results WHERE run_id = ? "
                "ORDER BY message_id, label",
                (run_id,),
            ).fetchall()
        except sqlite3.DatabaseError as e:
            raise AppDBError(f"state DB read failed: {self.path}: {e}") from e
        if not rows:
            raise RunNotFoundError(
                f"no shadow run {run_id!r} recorded — run `shadow run` first "
                "(or the run id is wrong)"
            )
        try:
            run_ts = datetime.fromisoformat(rows[0][4])
        except AppDBError:
            raise  # already a named refusal — never double-wrap
        except (ValueError, TypeError) as e:
            # a tampered/hand-edited DB with a non-ISO (or NULL) run_ts is
            # a corrupt row, named — not a bare ValueError/TypeError
            # traceback (exit 1)
            raise AppDBError(
                f"state DB corrupt (run_ts {rows[0][4]!r} is not ISO-8601): "
                f"{self.path}: {e}"
            ) from e
        # one run, one stamp: every row in a run carries the SAME run_ts
        # (record_run stamps the run-level timestamp on every row). A
        # tampered DB with mixed stamps is corrupt, named.
        distinct_ts = {r[4] for r in rows}
        if len(distinct_ts) != 1:
            raise AppDBError(
                f"state DB corrupt (run {run_id!r} carries {len(distinct_ts)} "
                f"distinct run_ts stamps {sorted(distinct_ts)} — one run, "
                f"one stamp): {self.path}"
            )
        out: list[ShadowRow] = []
        for r in rows:
            row = ShadowRow(
                message_id=r[0],
                label=r[1],
                probability=r[2],
                band=r[3],
                run_ts=run_ts,
            )
            try:
                # re-validate the read-back row (a tampered DB can carry an
                # arbitrary band string / out-of-range probability / a NULL
                # column — refused, named, not rendered)
                _check_row(row)
            except AppDBError:
                raise  # already a named refusal — never double-wrap
            except (ValueError, TypeError, AttributeError) as e:
                # a NULL column (a hand-edited DB without the shipped
                # NOT NULL constraints) raises AttributeError/TypeError in
                # _check_row — a corrupt row, named, not an exit-1 traceback
                raise AppDBError(
                    f"state DB corrupt (row {r!r} fails validation): {self.path}: {e}"
                ) from e
            out.append(row)
        return run_ts, out


def _check_row(row: ShadowRow) -> None:
    # the row validation (fail-loud — a lying row is refused, named,
    # BEFORE the transaction): the id shape, the label, the probability
    # range (a bool is not a probability), and the band verdict name
    if not _ID_RE.fullmatch(row.message_id):
        raise AppDBError(
            f"shadow row message_id {row.message_id!r} must be 1-64 hex chars"
        )
    if not row.label.strip():
        raise AppDBError(f"shadow row for {row.message_id}: label must be non-empty")
    if isinstance(row.probability, bool) or not isinstance(
        row.probability, (int, float)
    ):
        raise AppDBError(
            f"shadow row for {row.message_id} / {row.label}: probability "
            f"must be a number in [0, 1], got {row.probability!r}"
        )
    if not 0.0 <= row.probability <= 1.0:
        raise AppDBError(
            f"shadow row for {row.message_id} / {row.label}: probability "
            f"{row.probability} is out of [0, 1] — a lying model row"
        )
    if row.band is not None and row.band not in _BANDS:
        raise AppDBError(
            f"shadow row for {row.message_id} / {row.label}: band "
            f"{row.band!r} must be 'high', 'med', or None"
        )
