"""State DB adapter (ticket #23; GLOSSARY: Shadow Pass).

The SQLite file behind ``[store] db_path``: the shadow_results table
(the Shadow Pass's recorded would-apply verdicts) and, with the daemon
phase, the applications table (#24 — same file, same versioning story;
the seam is the schema_version row + the comment below).

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
    "ShadowRow",
]

#: The schema version (bump on any change to the tables' field set or
#: semantics — the #24 applications table lands under the SAME
#: versioning story).
SCHEMA_VERSION = 1

# Gmail message ids are hex (the CorpusRecord._ID_RE precedent): a
# non-hex id is a lying row, refused at the adapter boundary.
_ID_RE = re.compile(r"^[0-9a-fA-F]{1,64}$")

# The band verdict names (tiers.BAND_HIGH/BAND_MED — mirrored, not
# imported: appdb stays import-light, and the equality is pinned by
# test_appdb).
_BANDS = ("high", "med")


class AppDBError(Exception):
    """State DB failure: a foreign schema version, a corrupt file, a
    validation refusal, or an I/O failure. The CLI maps it to exit 5
    (operation failed) — never a raw sqlite3 traceback."""


@dataclass(frozen=True, slots=True)
class ShadowRow:
    """One would-apply verdict row (NO bodies — invariant #5): the
    message id, the label, the sigmoid probability, the band verdict
    (``"high"`` / ``"med"`` / ``None`` — the explicit no-verdict state),
    and the run's UTC timestamp (stamped by the run, not the row)."""

    message_id: str
    label: str
    probability: float
    band: str | None
    run_ts: datetime


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
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
-- Seam (ticket #24): the applications table lands in THIS file, under
-- this schema_version — the same versioning story. No write surface
-- exists here until #24/#25 (the Shadow Pass is read-only).
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
        verified (a non-SQLite file is a corrupt DB, named; a foreign
        schema_version is refused, named)."""
        if self._conn is not None:
            return
        if not self.path.parent.is_dir():
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise AppDBError(
                    f"state DB dir unusable: {self.path.parent}: {e}"
                ) from e
        existed = self.path.exists()
        try:
            self._conn = sqlite3.connect(str(self.path))
        except sqlite3.DatabaseError as e:
            raise AppDBError(
                f"state DB corrupt (not a SQLite file): {self.path}: {e}"
            ) from e
        if not existed:
            with contextlib.suppress(OSError):
                os.chmod(self.path, 0o600)  # foreign fs: best-effort
        try:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            self._conn.commit()
            self._verify_version()
        except sqlite3.DatabaseError as e:
            self._conn.close()
            self._conn = None
            raise AppDBError(f"state DB unusable: {self.path}: {e}") from e

    def _verify_version(self) -> None:
        # a foreign version is refused, named (the SplitManifest
        # discipline — a newer tool's DB is not served by an older one)
        assert self._conn is not None
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            raise AppDBError(
                f"state DB has no schema_version row: {self.path} — corrupt"
            )
        if row[0] != SCHEMA_VERSION:
            raise AppDBError(
                f"state DB schema_version {row[0]} != {SCHEMA_VERSION}: "
                f"{self.path} — a newer tool wrote this DB"
            )

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
            raise AppDBError(
                f"no shadow run {run_id!r} recorded — run `shadow run` first "
                "(or the run id is wrong)"
            )
        run_ts = datetime.fromisoformat(rows[0][4])
        return run_ts, [
            ShadowRow(
                message_id=r[0],
                label=r[1],
                probability=r[2],
                band=r[3],
                run_ts=run_ts,
            )
            for r in rows
        ]


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
