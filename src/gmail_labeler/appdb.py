"""State DB adapter (tickets #23/#24/#25; GLOSSARY: Shadow Pass,
Application, Durable Handling).

The SQLite file behind ``[store] db_path``: the shadow_results table
(the Shadow Pass's recorded would-apply verdicts), the applications
table (the Application path's recorded writes — ticket #24; the join
corrections and ``status`` read later), and the daemon_state table
(the daemon's persisted history cursor — ticket #25; GLOSSARY: History
Cursor).

UPGRADE STORY (paid #23's recorded debt): ``SCHEMA_VERSION`` is 3. The
v1→v2 migration (applications table) and the v2→v3 migration
(daemon_state) are SINGLE-ROW restamps, and EVERY schema transition —
fresh create+stamp, v1→vN / v2→vN restamp+append, self-heal of a
missing table — runs in ONE explicit transaction (``_apply_schema``;
the F1 fix). A crash leaves the file fully prior-state or fully
new-state, NEVER the intermediate: the old executescript-then-stamp
order could strand a fresh DB empty-version-corrupt and a migration
half-restamped, both needing hand-recovery. ``executescript`` is banned
from the transition (it commits any pending transaction first) and
python-sqlite3's legacy isolation mode does NOT auto-wrap DDL — hence
the explicit BEGIN (the mode is pinned by test). The tamper guards
(``{1}`` WITH a later-version table; ``{2}`` WITH daemon_state; our-
shape-but-EMPTY version) stay — unreachable by crash now, reachable
only by hand-editing — and each refusal message names the recovery
(manual stamp to the right version, or delete). The forward-compat
story is the HARD version gate, stated honestly: a v2 tool opening the
migrated file refuses ``{3} != {2}`` (our own v3 is the tool that just
migrated it — re-opening in-process is v3-clean).

Invariants:

- NO BODIES (invariant #5): the tables carry message ids, label
  names/ids, probabilities, band/tier verdicts, timestamps, and the
  history cursor ONLY — the DB must be publicly attachable (the
  corpus-sidecar discipline);
- TRANSACTION PER WRITE: ``record_run``, ``record_applications`` and
  ``commit_page`` insert their batch in ONE transaction — a crash
  mid-write (or a mid-write refusal) rolls back, leaving no half rows;
  a committed batch is atomic (``commit_page`` carries the page's rows
  AND the cursor advance in the SAME transaction — the Durable Handling
  commit point, invariant #3);
- create-all-if-missing: opening a fresh path creates the schema;
  opening an existing DB verifies the schema BEFORE any write (the
  ownership gate: shape equality per table, the version a HARD gate —
  a foreign/future/inconsistent DB is refused, named, and gains no
  table pre-refusal);
- deterministic reads: ``load_run`` returns rows in (message_id,
  label) order (the report's total order); the applications join reads
  return rows in (applied_at, label)/(applied_at, message_id) order.
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
from typing import Any, NoReturn

from .applications import ApplicationRow

__all__ = [
    "SCHEMA_VERSION",
    "AppDB",
    "AppDBError",
    "RunNotFoundError",
    "ShadowRow",
]

#: The schema version (bump on any change to the tables' field set or
#: semantics). v2 added the applications table (ticket #24); v3 adds
#: the daemon_state cursor table (ticket #25); the v1→v2 and v2→v3
#: migrations are single-row restamps documented in the module
#: docstring's UPGRADE STORY.
SCHEMA_VERSION = 3

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
_APPLICATIONS_COLUMNS = frozenset(
    {
        "message_id",
        "label",
        "label_id",
        "score",
        "band",
        "tier",
        "applied_at",
        "source",
    }
)
_DAEMON_STATE_COLUMNS = frozenset({"history_cursor"})
_SCHEMA_VERSION_COLUMNS = frozenset({"version"})

# The Application source literals (the CHECK constraint's vocabulary —
# mirrored in the Python validation so a lying row is refused NAMED at
# the writer, not as a raw sqlite3 CHECK traceback).
_SOURCES = ("daemon", "apply")


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


# The schema as INDIVIDUAL statements: _apply_schema runs them inside ONE
# explicit transaction (executescript is banned there — it commits any
# pending transaction first, which is what made the old transition
# crash-unsafe). The inline comments live with the applications and
# daemon_state DDL.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL PRIMARY KEY
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS shadow_results (
        run_id TEXT NOT NULL,
        message_id TEXT NOT NULL,
        label TEXT NOT NULL,
        probability REAL NOT NULL,
        band TEXT,
        run_ts TEXT NOT NULL,
        PRIMARY KEY (run_id, message_id, label)
    )
    """,
    """
    -- The Applications table (ticket #24): one row per applied label of an
    -- Application (GLOSSARY: Application). NO PK on (message_id, label): a
    -- manually-removed label re-earned later is a NEW Application — the
    -- audit history is the operator's. The two indexes serve the
    -- corrections/status join (#27) on message_id AND label. The source
    -- CHECK is the vocabulary's authority at the DB level; the writer
    -- validates the same set in Python (named refusals, not raw CHECK
    -- tracebacks). NO BODIES (invariant #5).
    CREATE TABLE IF NOT EXISTS applications (
        message_id TEXT NOT NULL,
        label TEXT NOT NULL,
        label_id TEXT NOT NULL,
        score REAL NOT NULL,
        band TEXT NOT NULL,
        tier TEXT NOT NULL,
        applied_at TEXT NOT NULL,
        source TEXT NOT NULL CHECK (source IN ('daemon', 'apply'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_applications_message_id "
    "ON applications(message_id)",
    "CREATE INDEX IF NOT EXISTS idx_applications_label ON applications(label)",
    """
    -- The daemon's history cursor (ticket #25; GLOSSARY: History
    -- Cursor, Durable Handling): the mailbox historyId persisted ONLY
    -- after the page's events are durably handled (invariant #3) — the
    -- advance rides the SAME transaction as the page's Application rows
    -- (``commit_page``; COMMIT = handled). SINGLE-ROW model: the writer
    -- UPSERTS rowid 1 (below), so the table never grows; a second row
    -- is hand-made and the read refuses it, named (the schema_version
    -- single-row discipline). NO BODIES (invariant #5): one integer.
    CREATE TABLE IF NOT EXISTS daemon_state (
        history_cursor INTEGER NOT NULL
    )
    """,
)

#: SQLite's signed INTEGER ceiling: the store's cursor bound (the wire's
#: uint64 is WIDER — gmail_client.UINT64_MAX; F1 named each bound once
#: instead of repeating literals in code and message).
_MAX_SQLITE_INTEGER = 2**63 - 1

# The cursor write (the single-row model at the writer): always rowid
# 1 — set replaces, never appends. INSERT OR REPLACE with the explicit
# rowid is the DOCUMENTED form (the upsert conflict-target is specified
# for UNIQUE/PRIMARY-Key constraints; the implicit rowid is outside
# that wording — the review round's finding). A replace at rowid 1
# leaves any hand-made extra row untouched, and the read gate refuses
# the multi-row state either way.
_CURSOR_UPSERT = (
    "INSERT OR REPLACE INTO daemon_state (rowid, history_cursor) VALUES (1, ?)"
)

#: The joined script text — kept ONLY for the tamper-state test fixtures
#: (their executescript is a fixture convenience, never a production
#: path); the production single source is _SCHEMA_STATEMENTS above.
_SCHEMA = ";\n".join(_SCHEMA_STATEMENTS) + ";"


class AppDB:
    """The state DB adapter (bounded I/O, CODING-STANDARDS). One
    connection per instance (single-process contract, ADR-0002);
    ``open`` creates the schema if missing and verifies the version;
    ``record_run``, ``record_applications``, ``set_history_cursor`` and
    ``commit_page`` are the only writers (transaction-per-write); the
    ``latest_run``/``load_run`` reads serve the report, the
    ``applications_for_*`` reads serve the join, and
    ``get_history_cursor`` serves the daemon's poll loop."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ------------------------------------------------------

    def open(self) -> None:
        """Connect + create-all-if-missing (the schema transition is
        ONE explicit transaction — module docstring's UPGRADE STORY). A
        fresh path creates the schema (0600 from creation — the
        secret-tight discipline, even though the rows are
        attachable-public); an existing file is
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
            from_version = self._check_ownership()
            self._apply_schema(from_version)
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

    def _apply_schema(self, from_version: int | None) -> None:
        """create-all-if-missing + the version stamp / legacy restamp in
        ONE explicit transaction (the F1 atomicity fix): a crash leaves
        the file fully prior-state (SQLite journal recovery) or fully
        new-state — never the intermediate the old executescript-then-
        stamp order exposed (fresh DBs stranded empty-version-corrupt,
        migrations stranded half-restamped). Legacy isolation mode does
        NOT auto-wrap DDL in the implicit transaction and executescript
        commits any pending transaction first — hence the explicit
        BEGIN/COMMIT over individual execute() calls (the mode is
        pinned by test_connection_isolation_mode_is_pinned; the ban by
        test_executescript_is_banned_from_appdb_production_source — the
        DIRECT assertion, F2). The rowcount guard keeps the
        single-row model honest INSIDE the transaction: a disagreement
        rolls the whole transition back."""
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        try:
            self._conn.execute("BEGIN")
            for statement in _SCHEMA_STATEMENTS:
                self._conn.execute(statement)
            if from_version is not None:
                cur = self._conn.execute(
                    "UPDATE schema_version SET version = ? WHERE version = ?",
                    (SCHEMA_VERSION, from_version),
                )
                if cur.rowcount != 1:
                    # the gate saw exactly {from_version}; a different
                    # row count is a concurrent tamper — the ROLLBACK
                    # below undoes the whole transition, the file stays
                    # at the prior version
                    raise AppDBError(
                        f"state DB v{from_version}→v{SCHEMA_VERSION} restamp "
                        f"touched {cur.rowcount} rows, expected 1 "
                        f"(single-row model) — refusing: {self.path}"
                    )
            else:
                # a fresh DB (no version row) gets stamped; an existing
                # ours-DB is NOT re-stamped (the version is already
                # {SCHEMA_VERSION} — verified in the ownership gate)
                versions = self._conn.execute(
                    "SELECT DISTINCT version FROM schema_version"
                ).fetchall()
                if not versions:
                    self._conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (SCHEMA_VERSION,),
                    )
            self._conn.execute("COMMIT")
        except BaseException:
            # the guarantee is the UNCOMMITTED transaction: a dead
            # process recovers via the journal, a live one via the
            # connection close in open()'s handler (the crash tests pin
            # the FILE state, which is prior-state either way — mutant
            # probe: suppressing this ROLLBACK alone stays green because
            # the close carries it). The explicit ROLLBACK is hygiene
            # for any future caller that survives the failure with the
            # connection open
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK")
            raise

    def _check_ownership(self) -> int | None:
        """Verify the file is ours (shape + version) BEFORE any write.
        Returns the LEGACY version to restamp from when the DB carries a
        migratable stamp ({1} pre-applications, {2} pre-daemon_state);
        ``None`` when it is current-or-fresh. Everything is read BEFORE
        any write (the "verified before any write" claim is literally
        true — a foreign DB is not adopted, and a versioned-foreign DB
        does not gain an applications table pre-refusal):

        - shadow_results present → its PRAGMA column set must EQUAL
          ours (a same-named wrong-shape table is refused, named);
        - applications present → its column set must EQUAL ours too
          (the ownership gate EXTENDED with the v2 table — the same
          shape-equality pattern; a same-named wrong-shape table is
          refused, named);
        - daemon_state present → its column set must EQUAL ours as well
          (the gate EXTENDED with the v3 table — same pattern);
        - schema_version present → it must be the single-column
          `version` table (a generic same-named table is NOT ours —
          the adoption hole ds reproduced), and the version HARD gate
          applies: the version set must EQUAL {SCHEMA_VERSION} — a
          foreign version ({99}), a polluted set ({1,99}), or an
          our-shape-but-EMPTY table is refused, named (the version is
          NOT advisory: a v3 DB is not consumed by a v2 tool). The
          tolerated MIGRATABLE exceptions: exactly {1} WITHOUT any of
          the later-version tables — the state #23 shipped — and
          exactly {2} WITHOUT daemon_state — the state #24 shipped —
          each returns its version for the single-row restamp. The
          TAMPER dead-ends stay: {1} WITH a later table (no v1 tool
          ever wrote one) and {2} WITH daemon_state (no v2 tool ever
          wrote one) are inconsistent, refused, named — unreachable by
          crash since the atomic transition, hand-editing only;
        - a DB with a valid ours-table PLUS UNKNOWN extra tables
          COEXISTS (documented decision — our tables are the
          authority; the unknowns are left alone). Coexist is for
          UNKNOWN TABLES only — a version disagreement NEVER
          coexists (it is refused above);
        - a DB with tables but NONE valid-ours is FOREIGN — refused,
          named (the foreign table names are listed);
        - an empty file (no tables) is fresh.
        """
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        shadow_results_valid = False
        applications_valid = False
        daemon_state_valid = False
        schema_version_present = False
        from_version: int | None = None
        if "shadow_results" in tables:
            cols = self._column_set("shadow_results")
            if cols != _SHADOW_RESULTS_COLUMNS:
                raise AppDBError(
                    f"state DB shadow_results has the wrong shape "
                    f"(columns {sorted(cols)} != {sorted(_SHADOW_RESULTS_COLUMNS)}) "
                    f"— refusing to adopt it: {self.path}"
                )
            shadow_results_valid = True
        if "applications" in tables:
            cols = self._column_set("applications")
            if cols != _APPLICATIONS_COLUMNS:
                raise AppDBError(
                    f"state DB applications has the wrong shape "
                    f"(columns {sorted(cols)} != {sorted(_APPLICATIONS_COLUMNS)}) "
                    f"— refusing to adopt it: {self.path}"
                )
            applications_valid = True
        if "daemon_state" in tables:
            cols = self._column_set("daemon_state")
            if cols != _DAEMON_STATE_COLUMNS:
                raise AppDBError(
                    f"state DB daemon_state has the wrong shape "
                    f"(columns {sorted(cols)} != {sorted(_DAEMON_STATE_COLUMNS)}) "
                    f"— refusing to adopt it: {self.path}"
                )
            daemon_state_valid = True
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
            # the version is NOT advisory (a v3 DB is not consumed by a
            # v2 tool; the parent's _verify_version refused exactly this)
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
                    # our-shape but EMPTY (no version row) — corrupt,
                    # named. UNREACHABLE by crash since the atomic
                    # transition (F1 — the stamp and the DDL commit as
                    # one); reachable only by hand-editing (or the
                    # pre-F1 crash state a live file may still carry).
                    # Re-opening KEEPS refusing — the ownership gate
                    # raises HERE, before _apply_schema — and the
                    # refusal message names the recovery: stamp the row
                    # manually or DELETE the state DB file (a fresh open
                    # then re-creates + stamps).
                    # Deleting discards recorded shadow runs — none can
                    # exist in the crash state (the run that crashed
                    # never committed), though the branch is also reached
                    # by a hand-emptied populated DB, where they would
                    # be lost. The fail-safe direction is correct
                    # (refuse, never adopt). All of the above is pinned
                    # by test_our_shape_empty_schema_version_is_corrupt
                    # (3x refusal + both recoveries).
                    raise AppDBError(
                        f"state DB schema_version is empty (our shape, no "
                        f"version row) — corrupt: {self.path} — recovery: "
                        f"run INSERT INTO schema_version (version) VALUES "
                        f"({SCHEMA_VERSION}) manually, or delete the file "
                        f"(recorded runs are lost)"
                    )
                if (
                    versions == {1}
                    and SCHEMA_VERSION >= 2
                    and not {"applications", "daemon_state"} & tables
                ):
                    # the legacy v1 stamp (the state #23 shipped): the
                    # migratable exception to the hard gate — the caller
                    # runs the single-row restamp INSIDE the one
                    # transition transaction (the crash story, module
                    # docstring); v3 creates applications AND
                    # daemon_state on the way
                    from_version = 1
                elif versions == {1} and SCHEMA_VERSION >= 2:
                    # {1} WITH a later-version table is inconsistent: no
                    # v1 tool ever wrote one, and no crash reaches it
                    # since the atomic transition (F1) — this is a
                    # TAMPER guard. The refusal message names the
                    # offending table(s) and the recovery (manual stamp
                    # to the current version, or delete).
                    offenders = sorted(
                        t for t in ("applications", "daemon_state") if t in tables
                    )
                    raise AppDBError(
                        f"state DB schema_version [1] is stamped v1 but "
                        f"carries {' and '.join(offenders)} (a v1 tool never "
                        f"wrote one) — inconsistent, refusing: {self.path} "
                        f"— recovery: run UPDATE schema_version SET version "
                        f"= {SCHEMA_VERSION} manually, or delete the file "
                        f"(shadow history is lost)"
                    )
                elif (
                    versions == {2}
                    and SCHEMA_VERSION >= 3
                    and "daemon_state" not in tables
                ):
                    # the legacy v2 stamp (the state #24 shipped): the
                    # migratable exception — the caller runs the
                    # single-row restamp inside the one transition
                    # transaction (daemon_state rides the create-all)
                    from_version = 2
                elif versions == {2} and SCHEMA_VERSION >= 3:
                    # {2} WITH a daemon_state table is inconsistent: no
                    # v2 tool ever wrote one, and no crash reaches it
                    # since the atomic transition (the v2→v3 crash pin)
                    # — TAMPER guard, recovery named
                    raise AppDBError(
                        f"state DB schema_version [2] is stamped v2 but "
                        f"carries a daemon_state table (a v2 tool never "
                        f"wrote one) — inconsistent, refusing: {self.path} "
                        f"— recovery: run UPDATE schema_version SET version "
                        f"= {SCHEMA_VERSION} manually, or delete the file "
                        f"(shadow and application history is lost)"
                    )
                elif (
                    len(versions) == 1
                    and isinstance(only := next(iter(versions)), int)
                    and not isinstance(only, bool)
                    and only > SCHEMA_VERSION
                ):
                    # a SINGLE future INT stamp: the only state where "a
                    # newer tool wrote this DB" is honest (the sealed [99]
                    # pins keep their prefix). The type guard is the F1
                    # round-2 fix: the shape check does not constrain
                    # column TYPES (hostile-affinity doctrine above), so a
                    # TEXT/NULL stamp must route to the unknown-set branch
                    # — comparing it here escaped open() as a raw
                    # TypeError, past the CLI's exit-5 mapping
                    raise AppDBError(
                        f"state DB schema_version {sorted(versions, key=repr)} != "
                        f"{SCHEMA_VERSION}: {self.path} — a newer tool wrote "
                        f"this DB; upgrade this tool, do not downgrade the DB"
                    )
                else:
                    # a foreign polluted set ({1,99}), a zero/negative
                    # stamp ({0}, {-1}), or a mixed legacy set ({1,2},
                    # {2,3}): NO released tool ever writes these — the
                    # F1 honesty fix retires the blanket "a newer tool"
                    # diagnosis for states no tool can produce
                    raise AppDBError(
                        f"state DB schema_version {sorted(versions, key=repr)} != "
                        f"{SCHEMA_VERSION}: {self.path} — unknown stamp set "
                        f"(this tool writes exactly [{SCHEMA_VERSION}]; the "
                        f"migratable legacy sets are [1] and [2] with no "
                        f"later-version tables) — hand-edit or corruption: "
                        f"recovery: inspect the tables by hand, or delete "
                        f"the file (all local history is lost)"
                    )
        if shadow_results_valid or applications_valid or daemon_state_valid:
            # ours (coexist with any UNKNOWN extra tables); a DB with
            # ONLY the applications or daemon_state table (no
            # shadow_results) is ours by the shape gate too
            return from_version
        if schema_version_present:
            # ours (our-shape schema_version stamped with our version,
            # or a migratable legacy stamp with no tables to adopt).
            # DOCUMENTED PRE-EXISTING POSTURE (#23's disposition-2
            # branch, unchanged in shape by v2): a file carrying ONLY
            # our-stamped version evidence is adopted as ours and
            # WRITTEN TO (the schema script appends the missing tables;
            # a {1} stamp also restamps). Our tool never produces a
            # version row without its tables, so this admits only
            # hand-made or hand-emptied files — the same surface #23
            # accepted, now including the restamp. The branch is
            # UNREACHABLE for any self-produced state (the atomic
            # transition creates the tables and the stamp together), so
            # requiring one ours-shaped table would change no
            # legitimate file — but that tightening is a #23 decision
            # change, not this ticket's scope. The atomic transition
            # did NOT narrow this posture (F1 note, per the manager
            # ruling): a stamp-only file is still adopted and written
            # to (the transition appends the missing tables).
            return from_version
        if tables:
            raise AppDBError(
                f"state DB is not a gmail-labeler state DB (foreign "
                f"tables: {sorted(tables)}) — refusing to adopt it: "
                f"{self.path}"
            )
        # no tables → fresh (proceed to create the schema)
        return None

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
            # future caller #25/#26 cannot create the state the reader
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

    def record_applications(self, rows: Sequence[ApplicationRow]) -> int:
        """Record one Application's rows in ONE transaction (the same
        crash-safety contract as ``record_run``: a mid-batch failure
        rolls back — no half rows). Refusals raise BEFORE the
        transaction — nothing is written: an empty batch (like
        ``record_run`` — the plan composes no-call-on-empty, an empty
        batch here is a caller bug), a row failing validation (the id
        shape, the score range, the band/tier/source vocabularies, a
        naive applied_at), or a duplicate (message_id, label) WITHIN one
        batch, or MORE THAN ONE distinct applied_at stamp in the batch (the
        #24 homogeneity contract as a store invariant, F1) (one
        Application carries one row per label — the same
        (message_id, label) across SEPARATE batches is legitimate
        re-application history, so there is no PK). Returns the row
        count."""
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        if not rows:
            raise AppDBError(
                "record_applications: no rows — an empty write batch is "
                "never committed (an empty plan is no Application)"
            )
        _check_application_batch(rows)
        try:
            # ONE transaction (the record_run precedent): the context
            # manager commits on success and rolls back on ANY exception
            with self._conn:
                _insert_applications(self._conn, rows)
        except sqlite3.DatabaseError as e:
            raise AppDBError(f"state DB write failed: {self.path}: {e}") from e
        return len(rows)

    # -- the daemon's history cursor (ticket #25) -------------------------

    def get_history_cursor(self) -> int | None:
        """The persisted history cursor (GLOSSARY: History Cursor), or
        ``None`` when never established (fresh daemon — the caller
        bootstraps via the export prior-art path, see daemon.py). The
        single-row model's READ gate: the writer UPSERTS rowid 1, so a
        second row is hand-made — refused, named (never silently
        picked); a hand-edited cursor VALUE is refused by the same
        re-validate-on-read doctrine (the load_run precedent)."""
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        try:
            rows = self._conn.execute(
                "SELECT rowid, history_cursor FROM daemon_state"
            ).fetchall()
        except sqlite3.DatabaseError as e:
            raise AppDBError(f"state DB read failed: {self.path}: {e}") from e
        if not rows:
            return None
        if len(rows) != 1:
            raise AppDBError(
                f"state DB daemon_state carries {len(rows)} rows (the "
                f"single-row model: the writer UPSERTS rowid 1) — corrupt: "
                f"{self.path} — recovery: run DELETE FROM daemon_state "
                f"manually (the cursor is re-established by the daemon's "
                f"bootstrap), or delete the file"
            )
        rowid, value = rows[0]
        if rowid != 1:
            # the writer only ever touches rowid 1: a lone row elsewhere
            # is hand-made (and the next legitimate write would make the
            # table two rows) — refused NAMED with the recovery, not
            # silently adopted (review finding)
            raise AppDBError(
                f"state DB daemon_state row carries rowid {rowid!r} (the "
                f"single-row model: the writer UPSERTS rowid 1) — corrupt: "
                f"{self.path} — recovery: run DELETE FROM daemon_state "
                f"WHERE rowid != 1 manually, or delete the file (the "
                f"cursor is re-established by bootstrap)"
            )
        return _check_cursor_value(
            value,
            recovery=(
                f"run UPDATE daemon_state SET history_cursor = <known-good "
                f"Gmail historyId> manually, or delete the file (the cursor "
                f"is re-established by the daemon's bootstrap): {self.path}"
            ),
        )

    def set_history_cursor(self, history_cursor: int) -> None:
        """Establish/replace the history cursor OUTSIDE a page commit
        (bootstrap and re-bootstrap — no Application rows ride it). The
        PAGE advance goes through ``commit_page``: the cursor must ride
        the same transaction as the page's rows (invariant #3). A
        refusal (a non-cursor value) raises BEFORE any write."""
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        _check_cursor_value(history_cursor)
        try:
            with self._conn:
                self._conn.execute(_CURSOR_UPSERT, (history_cursor,))
        except sqlite3.DatabaseError as e:
            raise AppDBError(f"state DB write failed: {self.path}: {e}") from e

    def commit_page(self, rows: Sequence[ApplicationRow], history_cursor: int) -> int:
        """The daemon's DURABLE HANDLING commit point (ticket #25,
        GLOSSARY: Durable Handling; invariant #3): ONE transaction
        carrying the page's Application rows AND the cursor advance —
        COMMIT = handled. ``rows`` may be EMPTY (a page whose messages
        all composed empty plans was still handled — the cursor must
        move; record_applications' empty-batch refusal is deliberately
        NOT inherited here). Refusals raise BEFORE the transaction: a
        row failing validation, a within-batch (message_id, label)
        duplicate, or a non-cursor value. A crash before the COMMIT
        replays the page; skip-present makes the replay idempotent —
        the safety proof is the daemon loop's write-first ordering
        (daemon.py, the pinned comment there). Returns the row count."""
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        _check_application_batch(rows)
        stamps = {row.applied_at.astimezone(UTC).isoformat() for row in rows}
        if len(stamps) > 1:
            # the #24 batch-homogeneity contract AS A STORE INVARIANT
            # (F1, both families): one Application — one clock reading,
            # so one delta's batch carries ONE applied_at. Enforced here,
            # not only at the daemon call site, so #26's future caller
            # inherits the invariant instead of rediscovering it
            raise AppDBError(
                f"commit_page batch carries {len(stamps)} distinct applied_at "
                f"stamps {sorted(stamps)} — one delta, ONE clock reading; "
                f"the caller must stamp its whole batch from a single "
                f"clock() call"
            )
        _check_cursor_value(history_cursor)
        try:
            # ONE transaction (the record_run precedent): the context
            # manager commits rows AND cursor together, and rolls back
            # BOTH on any failure — a half-committed page (rows without
            # the cursor) would double-write the rows on replay (the
            # atomicity pin: test_commit_page_rolls_back_rows_and_cursor)
            with self._conn:
                if rows:
                    # the ONE applications-insert authority (shared with
                    # record_applications — a column added to the table
                    # changes in exactly one place)
                    _insert_applications(self._conn, rows)
                self._conn.execute(_CURSOR_UPSERT, (history_cursor,))
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

    # -- applications join reads (the #27 corrections/status join) --------

    def applications_for_message(self, message_id: str) -> tuple[ApplicationRow, ...]:
        """The Applications recorded for one message, in (applied_at,
        label) order (a total order). An empty result is LEGITIMATE (a
        message with no Application — skip-present, no verdict, or a
        fail-safe skip), unlike ``load_run``'s named not-found: this is
        a join read, not a run lookup. Read-back rows are re-validated
        (the tamper doctrine: a hand-edited row is refused, named)."""
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        try:
            rows = self._conn.execute(
                "SELECT message_id, label, label_id, score, band, tier, "
                "applied_at, source FROM applications "
                "WHERE message_id = ? ORDER BY applied_at, label",
                (message_id,),
            ).fetchall()
        except sqlite3.DatabaseError as e:
            raise AppDBError(f"state DB read failed: {self.path}: {e}") from e
        return self._application_rows(rows)

    def applications_for_label(self, label: str) -> tuple[ApplicationRow, ...]:
        """The Applications recorded for one label (the status/audit
        join), in (applied_at, message_id) order (a total order). Empty
        is legitimate; rows are re-validated (the tamper doctrine)."""
        if self._conn is None:
            raise AppDBError("state DB not open — call open() first")
        try:
            rows = self._conn.execute(
                "SELECT message_id, label, label_id, score, band, tier, "
                "applied_at, source FROM applications "
                "WHERE label = ? ORDER BY applied_at, message_id",
                (label,),
            ).fetchall()
        except sqlite3.DatabaseError as e:
            raise AppDBError(f"state DB read failed: {self.path}: {e}") from e
        return self._application_rows(rows)

    def _application_rows(self, rows: list[Any]) -> tuple[ApplicationRow, ...]:
        # the shared read-back path (the fetchall rows are the DB
        # boundary — shape unknown by nature, the _svc precedent):
        # parse the applied_at stamp (a
        # non-ISO value is corrupt, named — the load_run doctrine) and
        # re-validate every row (a tampered band/source/score is
        # refused, named, not rendered)
        out: list[ApplicationRow] = []
        for r in rows:
            try:
                applied_at = datetime.fromisoformat(str(r[6]))
            except ValueError as e:
                raise AppDBError(
                    f"state DB corrupt (applied_at {r[6]!r} is not "
                    f"ISO-8601): {self.path}: {e}"
                ) from e
            row = ApplicationRow(
                message_id=r[0],
                label=r[1],
                label_id=r[2],
                score=r[3],
                band=r[4],
                tier=r[5],
                applied_at=applied_at,
                source=r[7],
            )
            try:
                _check_application_row(row)
            except AppDBError:
                raise  # already a named refusal — never double-wrap
            except (ValueError, TypeError, AttributeError) as e:
                raise AppDBError(
                    f"state DB corrupt (row {r!r} fails validation): {self.path}: {e}"
                ) from e
            out.append(row)
        return tuple(out)


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


def _check_application_batch(rows: Sequence[ApplicationRow]) -> None:
    # the ONE batch-validation authority (record_applications and
    # commit_page share it — the dup message is pinned verbatim): every
    # row validated, then the within-batch (message_id, label)
    # uniqueness (one Application carries one row per label)
    for row in rows:
        _check_application_row(row)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.message_id, row.label)
        if key in seen:
            raise AppDBError(
                f"duplicate row for {row.message_id} / {row.label} in "
                "one batch — one Application carries one row per label"
            )
        seen.add(key)


def _insert_applications(
    conn: sqlite3.Connection, rows: Sequence[ApplicationRow]
) -> None:
    # the ONE applications-INSERT column authority (the caller owns the
    # transaction — record_applications and commit_page both ride it)
    conn.executemany(
        "INSERT INTO applications "
        "(message_id, label, label_id, score, band, tier, "
        "applied_at, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row.message_id,
                row.label,
                row.label_id,
                row.score,
                row.band,
                row.tier,
                row.applied_at.astimezone(UTC).isoformat(),
                row.source,
            )
            for row in rows
        ],
    )


def _check_cursor_value(value: object, recovery: str | None = None) -> int:
    # The cursor vocabulary (the _opt_int64 wire doctrine: unsigned
    # int64 — a negative historyId must never silently advance
    # anything; the fake's start_history_id < 1 refusal is the prior
    # art): an int (a bool is not one), >= 1. The write paths refuse
    # it NAMED before the transaction; the read re-validates (the
    # tamper doctrine). The read path passes ``recovery`` (a tampered
    # row is the OPERATOR's to repair); the write paths carry no
    # recovery clause — there the value came from the caller, and
    # "delete the file" would misdiagnose a caller bug as state damage.
    def refuse(message: str) -> NoReturn:
        raise AppDBError(
            message if recovery is None else f"{message} — recovery: {recovery}"
        )

    if isinstance(value, bool) or not isinstance(value, int):
        refuse(f"history_cursor must be an integer, got {value!r}")
    if value < 1:
        refuse(f"history_cursor {value} is not a valid cursor (>= 1, unsigned)")
    if value > _MAX_SQLITE_INTEGER:
        # the wire type (uint64) is WIDER than SQLite's signed INTEGER:
        # an unbounded value would escape as a raw sqlite3 OverflowError
        # at bind time (past the DatabaseError wrapper, past the CLI's
        # exit-5 mapping) — refused NAMED here instead (review finding)
        refuse(
            f"history_cursor {value} exceeds the SQLite INTEGER range "
            f"({_MAX_SQLITE_INTEGER}) — no real historyId reaches this bound; "
            f"refusing"
        )
    return value


def _check_application_row(row: ApplicationRow) -> None:
    # the row validation (fail-loud — a lying row is refused, named,
    # BEFORE the transaction): the id shape, the label/label_id/tier
    # names, the score range (a bool is not a score), the band verdict
    # (an APPLIED row always has one — never None), the source literal
    # (the CHECK constraint's vocabulary, mirrored so the refusal is
    # named here, not a raw sqlite3 traceback), and the aware
    # applied_at (the split_date doctrine)
    if not _ID_RE.fullmatch(row.message_id):
        raise AppDBError(
            f"application row message_id {row.message_id!r} must be 1-64 hex chars"
        )
    if not row.label.strip():
        raise AppDBError(
            f"application row for {row.message_id}: label must be non-empty"
        )
    if not row.label_id.strip():
        raise AppDBError(
            f"application row for {row.message_id} / {row.label}: label_id "
            "must be non-empty"
        )
    if isinstance(row.score, bool) or not isinstance(row.score, (int, float)):
        raise AppDBError(
            f"application row for {row.message_id} / {row.label}: score must "
            f"be a number in [0, 1], got {row.score!r}"
        )
    if not 0.0 <= row.score <= 1.0:
        raise AppDBError(
            f"application row for {row.message_id} / {row.label}: score "
            f"{row.score} is out of [0, 1] — a lying model row"
        )
    if row.band not in _BANDS:
        raise AppDBError(
            f"application row for {row.message_id} / {row.label}: band "
            f"{row.band!r} must be 'high' or 'med' (an applied row always "
            "has a band)"
        )
    if not row.tier.strip():
        raise AppDBError(
            f"application row for {row.message_id} / {row.label}: tier must "
            "be non-empty"
        )
    if not isinstance(row.applied_at, datetime):
        # a str/None applied_at is a caller bug — refused NAMED here, not
        # as a bare AttributeError from the tzinfo check (the read path
        # wraps, the write path must too)
        raise AppDBError(
            f"application row for {row.message_id} / {row.label}: applied_at "
            f"must be a datetime, got {row.applied_at!r}"
        )
    if row.applied_at.tzinfo is None:
        raise AppDBError(
            f"application row for {row.message_id} / {row.label}: applied_at "
            "must be timezone-aware (a naive timestamp is a caller bug — the "
            "split_date doctrine)"
        )
    if row.source not in _SOURCES:
        raise AppDBError(
            f"application row for {row.message_id} / {row.label}: source "
            f"{row.source!r} must be one of {_SOURCES}"
        )
