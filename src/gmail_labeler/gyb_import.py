"""GYB backup import (GLOSSARY: GYB Import; corpus source ``gyb``).

Reads a Got Your Back backup — msg-db.sqlite (READ-ONLY URI: a misconfigured
import can never mutate someone's only backup) joined as
``uids.uid`` = Gmail message id (verified against a 21-year, 394k-message
corpus), ``messages.message_filename`` = the ``.eml`` path relative to the
backup root, ``messages.message_internaldate`` = Gmail timestamp AS WRITTEN BY GYB:
a naive 'YYYY-MM-DD HH:MM:SS' string in the backup machine's LOCAL time
(GYB's folder tree, by contrast, is UTC). This module therefore treats the
FOLDER DATE (year/month/day in message_filename) as authoritative and
corrects the stamp's date to it — the day is what temporal splits
(invariant #7) depend on; the H:M:S may be off by the backup machine's
zone. Bodies go through the SAME shared MIME path as the API export
(invariant #4: one parser, no per-source drift).

Label names arrive in Gmail's NAME space; verified corpus shows system
labels exactly as the canonical uppercase tokens (the id==name equality
documented in labels.py holds for this data path — GYB's backup path
stores system labels by id; its 73-locale labellang affects RESTORE, not
the backup DB). ``system_labels_extra`` exists for hand-edited or derived
exports where localized tokens CAN appear; a token list you maintain is
safer than this module guessing.

Backups are read as POSIX trees (Windows-made backslash filenames
are out of scope; split('/') finds no date shape and read_bytes finds no file).
A full-run read transaction (BEGIN) keeps counts and stream on ONE
snapshot; the price is a SHARED LOCK held for the whole import: run it
after GYB has finished, or GYB's own writes will hit 'database is locked'
(release comes when the read stream closes, not per row).

Resume is structural (store id-set skip-present), crash-retry harmless,
and a corrupt index STOPS the run — identical posture to export.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from .corpus_record import CorpusRecord, body_hash
from .corpus_store import CorpusError, CorpusStore
from .labels import is_system_label_id
from .mime_extract import parse_mime

log = logging.getLogger("gmail_labeler.gyb")

DB_FILENAME = "msg-db.sqlite"
_LABEL_SEP = "\x1f"

_QUERY = (
    "SELECT u.uid, m.message_filename, m.message_internaldate,"
    " group_concat(l.label, ?)"
    " FROM messages m"
    " JOIN uids u ON u.message_num = m.message_num"
    " LEFT JOIN labels l ON l.message_num = m.message_num"
    " GROUP BY m.message_num, u.uid, m.message_filename, m.message_internaldate"
    " ORDER BY m.message_internaldate, u.uid"
)


@dataclass(frozen=True, slots=True)
class GybRow:
    id: str
    eml_relpath: str
    when: datetime  # tz-aware UTC (folder-date-corrected; see module docstring)
    labels: tuple[str, ...]  # content labels only (system/extra filtered)
    date_corrected: bool  # stamp's date fixed to the folder (zone skew <= 1 day)
    folder_unparsed: bool  # filename lacks the year/month/day shape:
    # NO UTC anchor exists for this row; the machine-local day is kept
    date_conflict: bool  # folder/stamp disagree by MORE than a day:
    # provenance broken (corruption, not time zones) — stamp kept, loud


@dataclass
class GybImportStats:
    scanned: int = 0
    written: int = 0
    skipped_present: int = 0
    date_corrected: int = 0  # rows whose stamp was fixed to the folder (<= 1 day)
    folder_unparsed: int = 0  # rows imported with NO UTC anchor (odd filename)
    date_conflict: int = 0  # folder/stamp disagree > 1 day: provenance broken
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def failed_ids(self) -> list[str]:
        return [i for i, _ in self.failed]


class GybImportError(Exception):
    """The backup is unreadable or its schema drifted — a loud whole-run
    stop (exit 5), never a silent partial import. Separate posture: rows
    whose UTC anchor is UNCONFIRMED are imported anyway, loudly (stats +
    CLI NOTE): 'folder_unparsed' has no folder anchor at all, while
    'date_conflict' has two anchors (the gmtime folder and the local-clock
    stamp) pointing at different days — a hand-moved file makes the folder
    the liar, a corrupt clock makes the stamp the liar, and neither side
    can be trusted blindly, so the raw stamp is kept and flagged.
    Distinct from incidental
    ValueErrors (real defects) which must NOT be laundered into it.

    ``stats`` is attached when a run aborts mid-loop, so the CLI can report
    what was durably imported before the stop (deepseek partial-state NIT).
    """

    stats: GybImportStats | None = None


def _parse_internaldate(raw: object) -> datetime:
    """GYB writes message_internaldate as 'YYYY-MM-DD HH:MM:SS'; some GYB
    versions/tools leave a microsecond tail ('.ffffff') whenever epoch-ms
    are non-zero (0 occurrences in the verified 394k corpus — tolerated as
    version drift, not observed behavior). The string is the BACKUP
    MACHINE'S LOCAL time; callers must reconcile it against the UTC folder
    date. Anything unparseable is schema drift -> raise."""
    if not isinstance(raw, str):
        raise GybImportError(f"unexpected message_internaldate {type(raw).__name__}")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise GybImportError(f"unparseable message_internaldate {raw!r}")


def _folder_utc_date(relpath: str) -> date | None:
    """The GYB tree is '<year>/<month>/<day>/<hexid>.eml' in UTC (gmtime).
    A 4-digit, plausible (>= 1900) year is REQUIRED: '1/1/1/x.eml' is
    corruption or hand-editing, not a folder to correct a stamp toward."""
    parts = relpath.split("/")
    if len(parts) < 4 or len(parts[0]) != 4:
        return None
    try:
        parsed = date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None
    return parsed if parsed.year >= 1900 else None


def _keep_label(name: str, system_labels_extra: frozenset[str]) -> bool:
    return (
        bool(name) and name not in system_labels_extra and not is_system_label_id(name)
    )


def read_gyb_index(
    db_path: Path, *, system_labels_extra: frozenset[str] = frozenset()
) -> Iterator[GybRow]:
    """Stream the backup index ordered by the RAW stamp (on non-UTC backup
    machines that ordering is the machine's local clock, and rows can
    cross corrected-day boundaries out of order — harmless: resume is
    id-set based and corpus order is not a training feature). The DB is
    opened read-only; a malformed schema is
    a loud abort, not a silent partial read. Labels are returned as a
    SORTED, DEDUPED set (byte-stable records; GYB's own storage carries no
    meaningful label ordering, and SQLite aggregate order is arbitrary)."""
    if not db_path.is_file():
        # typed like every other unreadable backup (a DIRECTORY named
        # msg-db.sqlite says 'not a file' honestly; the CLI pre-check stays)
        raise GybImportError(f"backup database missing or not a file: {db_path}")
    try:
        # as_uri(): exact percent-encoding — raw interpolation let a '?' or
        # '#' in the backup path silently open a DIFFERENT file (deepseek)
        con = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as e:
        # honest observation only: a -wal file can coexist with a fine
        # read (deepseek retracted the 'live WAL breaks reads' claim — it
        # was their fixture's missing column); do NOT diagnose with it
        hint = (
            " [note: a -wal file exists beside the backup]"
            if (db_path.parent / (db_path.name + "-wal")).exists()
            else ""
        )
        raise GybImportError(
            f"database unreadable ({type(e).__name__}){hint}"
        ) from None
    try:
        # ONE read snapshot: counts and stream must see the same database
        # (a mid-run writer commit used to fake a 'corrupt index' abort)
        con.isolation_level = None  # autocommit; we drive the transaction
        con.execute("BEGIN")
        total = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        joined = con.execute(
            "SELECT COUNT(DISTINCT m.message_num) FROM messages m"
            " JOIN uids u ON u.message_num = m.message_num"
        ).fetchone()[0]
        dup_uids = con.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT message_num) FROM uids"
        ).fetchone()[0]
        if dup_uids:
            # duplicate uid rows pass the gap check yet import one .eml
            # under two ids
            raise GybImportError(
                f"uids table has {dup_uids} duplicate message_num rows"
            )
        # integrity is checked one direction only: ORPHAN uid rows
        # (message_num absent from messages) are silently invisible to the
        # INNER JOIN — harmless (nothing to import), deliberately not an
        # abort (deepseek NIT)
        if joined != total:
            # index integrity: silent INNER-JOIN drops would undercount the
            # corpus with no signal — this posture STOPS on a corrupt index
            raise GybImportError(
                f"uids/messages gap: {total - joined} of {total} messages"
                " have no uid row (corrupt backup index)"
            )
        cur = con.execute(_QUERY, (_LABEL_SEP,))
        for uid, fname, idate, labels_raw in cur:
            if not isinstance(uid, str) or not isinstance(fname, str):
                raise GybImportError("schema drift: uid/filename must be TEXT")
            when = _parse_internaldate(idate)
            corrected = False
            conflict = False
            folder = _folder_utc_date(fname)
            unparsed = folder is None
            if folder is not None and when.date() != folder:
                # Zone skew between a machine-local stamp and the UTC
                # folder is AT MOST ±1 day (|offset| < 24 h). Wider
                # disagreement is broken provenance: correcting across it
                # could move a record YEARS (deepseek engine pass: a 1999
                # folder vs 2026 mail crossing the temporal split with a
                # 'zone skew' label). Keep the stamp; make it loud below.
                if abs((when.date() - folder).days) <= 1:
                    when = when.replace(
                        year=folder.year, month=folder.month, day=folder.day
                    )
                    corrected = True
                else:
                    conflict = True
            raw_labels = (
                labels_raw.split(_LABEL_SEP) if isinstance(labels_raw, str) else []
            )
            # label SET semantics made explicit: sorted + deduped, so the
            # stored line is byte-stable despite SQLite's arbitrary
            # aggregate order and any duplicate rows (deepseek MINOR)
            labels = tuple(
                sorted({n for n in raw_labels if _keep_label(n, system_labels_extra)})
            )
            yield GybRow(
                id=uid,
                eml_relpath=fname,
                when=when,
                labels=labels,
                date_corrected=corrected,
                folder_unparsed=unparsed,
                date_conflict=conflict,
            )
    except sqlite3.Error as e:
        # corrupt file / foreign schema / locked: typed, not a raw traceback
        raise GybImportError(f"database read failed ({type(e).__name__})") from None
    finally:
        with contextlib.suppress(sqlite3.Error):  # release the read snapshot
            con.execute("COMMIT")
        con.close()


def import_gyb(
    gyb_root: Path,
    store: CorpusStore,
    *,
    max_messages: int | None = None,
    system_labels_extra: frozenset[str] = frozenset(),
) -> GybImportStats:
    """Import the backup into the corpus store. Returns counts + failures."""
    stats = GybImportStats()
    if max_messages is not None and max_messages <= 0:
        return stats  # explicit no-op, same boundary as export
    db_path = gyb_root / DB_FILENAME
    root_resolved = gyb_root.resolve()
    try:
        for row in read_gyb_index(db_path, system_labels_extra=system_labels_extra):
            stats.scanned += 1
            mid = row.id.lower()  # ids normalize to lowercase (export does too)
            if store.has(mid):
                stats.skipped_present += 1
                if max_messages is not None and stats.scanned >= max_messages:
                    # the bound holds on EVERY path (deepseek engine pass:
                    # skipped rows used to bypass it, scanning unbounded)
                    return stats
                continue
            eml = (gyb_root / row.eml_relpath).resolve()
            if not eml.is_relative_to(root_resolved):
                # FAIL-CLOSED whole-run (vs per-row failed for unreadable
                # .eml): the DB's paths are untrusted input, the file
                # contents are the user's own — deliberate asymmetry
                # the DB is untrusted input: '../' or absolute filenames must
                # not read arbitrary files into the corpus (precedent:
                # CorpusStore._body_path containment)
                raise GybImportError(
                    f"message_filename escapes the backup root: {row.eml_relpath!r}"
                )
            try:
                data = eml.read_bytes()
            except OSError as e:
                stats.failed.append((mid, f"eml unreadable ({type(e).__name__})"))
                continue
            parsed = parse_mime(data)  # never raises; same path as the API export
            try:
                record = CorpusRecord(
                    id=mid,
                    date=row.when,
                    subject=parsed.subject,
                    sender=parsed.sender,
                    recipients=parsed.recipients,
                    labels=row.labels,
                    body_hash=body_hash(parsed.body_text),
                    source="gyb",
                    thread_id="",  # msg-db carries no thread map; never guessed
                )
            except ValueError as e:
                stats.failed.append((mid, f"record shape: {e}"))
                continue
            # CorpusError (corrupt index etc.) PROPAGATES and stops the run.
            added = store.put(record, parsed.body_text)
            if added:
                stats.written += 1
                if row.folder_unparsed:
                    stats.folder_unparsed += 1
                if row.date_corrected:
                    # count rows THIS run corrected into the corpus, not
                    # scans of already-present mail (deepseek NIT)
                    stats.date_corrected += 1
                if row.date_conflict:
                    stats.date_conflict += 1
            else:
                stats.skipped_present += 1
            if max_messages is not None and stats.scanned >= max_messages:
                return stats
    except CorpusError as e:
        # store-side abort: re-surface as the typed gyb exit-5 class with
        # counts attached, so both abort paths print alike (deepseek NIT);
        # CorpusError itself stays untouched — it's shared with export
        wrapped = GybImportError(f"corpus store aborted: {e}")
        wrapped.stats = stats
        raise wrapped from e
    except GybImportError as e:
        if e.stats is None:
            e.stats = stats  # partial state becomes visible to the CLI
        raise
    return stats
