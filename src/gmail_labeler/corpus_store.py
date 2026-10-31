"""Filesystem corpus store (bounded I/O adapter per CODING-STANDARDS).

Layout under a corpus directory:
  records.jsonl        one CorpusRecord per line, appended, keyed by id
  bodies/<id[:2]>/<id>.txt   body text sharded by id prefix

Durability and integrity contract:
  - files are created 0600 (and fchmod'd on rewrite, closing the
    pre-0600-revision migration hole); corpus dirs 0700 on the write path.
    Opening a store NEVER mutates permissions (a read-only command pointed
    at a misconfigured corpus.dir must not clamp someone's $HOME): call
    enforce_modes() from write-side commands (export/apply/daemon startup)
    to harden the tree; verify_modes() warns (content-free) on read paths;
  - body is written+fsynced BEFORE its record line is appended+fsynced;
    directory fsyncs are best-effort (NFS/FUSE tolerate) — a crash leaves at
    worst an orphaned body (rewritten idempotently), never a record
    pointing at nothing;
  - crash tails: an unparseable final fragment is truncated by the next
    put(); a COMPLETE line that merely lost its newline is sealed, never
    dropped; an unterminated tail beyond the scan bound is refused loudly
    (CorpusError) rather than guessed — whole-file destruction is never an
    acceptable "repair";
  - put() verifies the record's body_hash against the body actually stored
    (preprocessing-integrity tripwire);
  - re-putting an existing id is a no-op (append-only; corrections happen
    at dataset build, not by mutation).

Single-process contract (ADR-0002): one writer per corpus directory at a
time; no cross-process locking. Do not iterate and write concurrently.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

from .corpus_record import CorpusRecord
from .corpus_record import body_hash as _body_hash

log = logging.getLogger("gmail_labeler.corpus")


class CorpusError(Exception):
    """Corpus integrity violation or unreadable/corrupt data we refuse to guess at."""


_DIR_MODE = 0o700
_FILE_MODE = 0o600
_TAIL_CHUNK = 8192
_MAX_TAIL_SCAN = 1 << 20


class CorpusStore:
    def __init__(self, root: Path) -> None:
        # Read-only by contract: no mkdir, no chmod here — see module docstring.
        # The _known_ids cache loads once per instance and does NOT observe
        # external edits to records.jsonl (single-process contract).
        self.root = root
        self._records_path = root / "records.jsonl"
        self._bodies_dir = root / "bodies"
        self._known_ids: set[str] | None = None

    # -- reading ----------------------------------------------------------

    def _ids(self) -> set[str]:
        # Strict by construction: the resume gate IS the reader's validation
        # (a corpus _ids() accepts, iter_records can read). Streams; only the
        # id set is retained.
        if self._known_ids is None:
            self._known_ids = {rec.id for rec in self.iter_records()}
        return self._known_ids

    def _iter_lines(self) -> Iterator[tuple[str, bool]]:
        """Yield (line, is_torn_tail). Only a final line lacking its
        terminating newline is torn; everything else must parse."""
        if not self._records_path.is_file():
            return
        with self._records_path.open("r", encoding="utf-8", newline="\n") as fh:
            pending: tuple[str, bool] | None = None
            for raw in fh:  # iterates on \\n/\\r\\n/\\r; JSON escapes raw \\r
                item = (raw[:-1], False) if raw.endswith("\n") else (raw, True)
                if pending is not None:
                    yield pending
                pending = item
            if pending is not None:
                yield pending

    def iter_records(self) -> Iterator[CorpusRecord]:
        for lineno, (line, torn_tail) in enumerate(self._iter_lines(), start=1):
            if not line.strip():
                continue
            try:
                yield CorpusRecord.from_json_line(line)
            except Exception as e:
                if torn_tail:
                    return  # torn crash tail: skipped, store resumes
                raise CorpusError(
                    f"{self._records_path}:{lineno}: corrupt record: {e}"
                ) from e

    def has(self, message_id: str) -> bool:
        return message_id in self._ids()

    def get_body(self, message_id: str) -> str | None:
        path = self._body_path(message_id)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    # -- writing ----------------------------------------------------------

    def put(self, record: CorpusRecord, body: str) -> bool:
        """Store record+body; True if newly added, False if id already present.

        Raises CorpusError if the record's body_hash does not describe `body`.
        """
        if record.id in self._ids():
            return False
        if _body_hash(body) != record.body_hash:
            raise CorpusError(
                f"corpus: body does not match record {record.id} fingerprint"
            )
        body_path = self._body_path(record.id)
        self._ensure_written_dirs(body_path.parent)
        try:
            _write_fsync(body_path, body)  # body bytes first...
        except OSError as e:
            raise CorpusError(
                f"corpus: body write failed ({type(e).__name__})"
            ) from None
        _fsync_dir(body_path.parent, body_path)  # ...then its dirent
        self._repair_tail()
        # ...then the index line, created 0600 (fchmod'd if pre-existing).
        existed = self._records_path.exists()
        fd = self._open_append_checked()
        with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(record.to_json_line() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if not existed:
            _fsync_dir(self.root, self._records_path)
        self._ids().add(record.id)
        return True

    def _ensure_written_dirs(self, shard: Path) -> None:
        """Create the chain root -> bodies/ -> shard (parents allowed — a
        config'd path like ~/mail/corpus may be nested), chmod each corpus-
        owned dir EXACTLY (mkdir mode is leaf-only + umask'd), and fsync the
        parent of every dir freshly created (existence checked BEFORE the
        mkdir — checking after would skip the root's own dirent)."""
        for d in (self.root, self._bodies_dir, shard):
            try:
                existed = d.is_dir()
                d.mkdir(parents=True, exist_ok=True)
                os.chmod(d, _DIR_MODE)  # corpus-owned dirs only
                if not existed:
                    _fsync_dir(d.parent, d)
            except OSError as e:
                raise CorpusError(f"corpus dir unusable: {d}: {e}") from e

    def _open_append_checked(self) -> int:
        try:
            fd = os.open(
                self._records_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                _FILE_MODE,
            )
            try:
                os.fchmod(fd, _FILE_MODE)  # pre-existing loose file tightened
            except OSError:
                os.close(fd)
                raise
            return fd
        except OSError as e:
            raise CorpusError(
                f"records index unusable: {self._records_path}: {e}"
            ) from e

    def _repair_tail(self) -> None:
        """Make the file append-ready (put only) without re-reading it.

        A tail after the last newline is either a COMPLETE record line that
        lost its terminator (sealed, kept — it parses) or torn bytes that can
        never parse (truncated; the message re-puts cleanly). A tail whose
        START we cannot even locate within _MAX_TAIL_SCAN is pathological
        corruption: we REFUSE (CorpusError) rather than truncate-to-zero the
        whole corpus on a guess.
        """
        try:
            with self._records_path.open("r+b") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                if size == 0:
                    return
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) == b"\n":
                    return
                back, start = 0, None
                while back < size and back < _MAX_TAIL_SCAN:
                    step = min(_TAIL_CHUNK, size - back)
                    fh.seek(size - back - step)
                    idx = fh.read(step).rfind(b"\n")
                    if idx != -1:
                        start = size - back - step + idx + 1
                        break
                    back += step
                if start is None:
                    if size > _MAX_TAIL_SCAN:
                        raise CorpusError(
                            f"{self._records_path}: unterminated tail beyond "
                            f"{_MAX_TAIL_SCAN} B — refusing to truncate blindly; "
                            "inspect/repair manually"
                        )
                    start = 0  # whole file is one small unterminated line
                fh.seek(start)
                tail = fh.read(size - start)
                fh.seek(0, os.SEEK_END)
                if _parses(tail):
                    fh.write(b"\n")
                else:
                    fh.truncate(start)
                fh.flush()
                os.fsync(fh.fileno())
        except FileNotFoundError:
            return
        except CorpusError:
            raise
        except OSError as e:
            raise CorpusError(f"{self._records_path}: tail repair failed: {e}") from e

    def _body_path(self, message_id: str) -> Path:
        # CorpusRecord validates id shape, but defense in depth: containment.
        path = (self._bodies_dir / message_id[:2] / f"{message_id}.txt").resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise CorpusError(
                f"corpus: refusing path outside corpus root for id {message_id}"
            )
        return path

    # -- permissions (SECURITY.md: corpus material is secret) --------------

    def enforce_modes(self) -> None:
        """Harden the corpus tree: dirs 0700, every file 0600 (one bounded
        walk of ~256 shard dirs). Call from WRITE-side commands at startup
        (export/apply/daemon) — not from read-only tools."""
        if not self.root.is_dir():
            return
        _chmod_if(self.root, _DIR_MODE)
        records = self._records_path
        if records.is_file():
            _chmod_if(records, _FILE_MODE)
        if self._bodies_dir.is_dir():
            _chmod_if(self._bodies_dir, _DIR_MODE)
            for shard in self._bodies_dir.iterdir():
                if not shard.is_dir():
                    continue
                _chmod_if(shard, _DIR_MODE)
                for body in shard.iterdir():
                    if body.is_file():
                        _chmod_if(body, _FILE_MODE)

    def verify_modes(self) -> list[str]:
        """Non-destructive check for group/world-readable STRUCTURAL paths
        (root, bodies/, records index) — not a full-tree scan; call
        enforce_modes() from write-side commands for the whole tree."""
        offenders: list[str] = []
        for p in (self.root, self._bodies_dir, self._records_path):
            try:
                if p.exists() and p.stat().st_mode & 0o077:
                    offenders.append(str(p))
            except OSError:
                continue
        return offenders


def _parses(raw: bytes) -> bool:
    try:
        CorpusRecord.from_json_line(raw.decode("utf-8"))
    except Exception:
        return False
    return True


def _write_fsync(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
    try:
        os.fchmod(fd, _FILE_MODE)  # tighten pre-existing files (0644 era)
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as fh:  # owns fd from here
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


_warned_dir_fsync = False


def _fsync_dir(dirpath: Path, subject: Path | None = None) -> None:
    """Best-effort (some NFS/FUSE mounts refuse dir fsync); a missing
    dirent is a durability downgrade, not a reason to kill an export."""
    global _warned_dir_fsync
    try:
        fd = os.open(dirpath, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as e:
        if not _warned_dir_fsync:
            log.warning("corpus: directory fsync unsupported here (%s)", e)
            _warned_dir_fsync = True


def _chmod_if(path: Path, mode: int) -> None:
    try:
        if path.stat().st_mode & 0o777 != mode:
            os.chmod(path, mode)
    except OSError:
        pass  # foreign files on odd filesystems: best-effort hardening
