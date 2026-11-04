from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from gmail_labeler.corpus_record import (
    CorpusRecord,
    body_hash,
    utc_datetime,
    with_body,
)
from gmail_labeler.corpus_store import CorpusError, CorpusStore

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
DUMMY_HASH = body_hash("placeholder")


def rec(id_: str = "abc123", **kw: object) -> CorpusRecord:
    base: dict[str, object] = {
        "id": id_,
        "date": NOW,
        "subject": "hi",
        "sender": "a@b.c",
        "recipients": ("me@x.y",),
        "labels": ("Shipping",),
        "body_hash": DUMMY_HASH,
        "source": "api",
    }
    base.update(kw)
    return CorpusRecord(**base)  # type: ignore[arg-type]


def test_record_json_round_trip() -> None:
    r = with_body(rec(), "the body")
    assert CorpusRecord.from_json_line(r.to_json_line()) == r


def test_record_line_is_stable_json() -> None:
    assert '"labels": ["Shipping"]' in rec().to_json_line()


def test_record_rejects_bad_ids() -> None:
    for bad in ("", "../escaped", "with/slash", "zzz", "a" * 65, "id with space"):
        with pytest.raises(ValueError, match="hex"):
            rec(id_=bad)


def test_record_accepts_uppercase_hex_id() -> None:
    assert rec(id_="19F682EEE4FA68B6").id == "19F682EEE4FA68B6"


def test_record_rejects_empty_hash() -> None:
    # An unhashed record must be indistinguishable-from-impossible, not a
    # silent wildcard (deepseek/qwen both flagged "" as the path of least
    # resistance).
    with pytest.raises(ValueError, match="body_hash"):
        rec(body_hash="")


def test_record_rejects_bad_source_and_dates() -> None:
    with pytest.raises(ValueError, match="source"):
        rec(source="weird")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timezone-aware"):
        rec(date=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="timezone-aware"):
        rec(date=datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=-5))))


def test_utc_datetime_coercion() -> None:
    assert utc_datetime(datetime(2026, 1, 1)).tzinfo is UTC
    # fixed -05:00 == US/Eastern on 2026-01-01 (EST); runners lack a tz db
    eastern = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=-5)))
    assert utc_datetime(eastern).utcoffset() == timedelta(0)


def test_from_json_line_rejects_naive_date_and_bad_shapes() -> None:
    line = rec().to_json_line()
    naive = line.replace(NOW.isoformat(), "2026-01-01T00:00:00")
    with pytest.raises(ValueError, match="refusing to guess"):
        CorpusRecord.from_json_line(naive)
    not_list = line.replace('"recipients": ["me@x.y"]', '"recipients": "me@x.y"')
    with pytest.raises(ValueError, match="list of strings"):
        CorpusRecord.from_json_line(not_list)


def test_unicode_line_separators_round_trip(tmp_path: Path) -> None:
    # U+2028/U+2029 inside JSON strings must NOT split records on reload
    # (the splitlines() brick: both reviewers reproduced it).
    r = with_body(rec(subject="line\u2028sep\u2029end"), "body")
    store = CorpusStore(tmp_path / "c")
    store.put(r, "body")
    reopened = CorpusStore(tmp_path / "c")
    assert list(reopened.iter_records()) == [r]
    assert reopened.has("abc123")


def test_store_put_get_round_trip(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    r = with_body(rec(), "hello body")
    assert store.put(r, "hello body") is True
    assert store.get_body("abc123") == "hello body"
    assert list(store.iter_records()) == [r]


def test_store_put_is_idempotent_on_id(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    assert store.put(with_body(rec(), "v1"), "v1") is True
    assert store.put(with_body(rec(), "v2"), "v2") is False
    assert store.get_body("abc123") == "v1"  # append-only: no mutation
    assert len(list(store.iter_records())) == 1


def test_store_rejects_body_hash_mismatch(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    with pytest.raises(CorpusError, match="fingerprint"):
        store.put(with_body(rec(), "REAL BODY"), "DIFFERENT BODY")


def test_store_file_permissions(tmp_path: Path) -> None:
    # The corpus holds secret mail: nothing group-/world-readable.
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec("deadbeef"), "x"), "x")
    records = tmp_path / "corpus" / "records.jsonl"
    body = tmp_path / "corpus" / "bodies" / "de" / "deadbeef.txt"
    assert records.stat().st_mode & 0o077 == 0
    assert body.stat().st_mode & 0o077 == 0
    assert (tmp_path / "corpus").stat().st_mode & 0o077 == 0


def test_store_survives_reload(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec(), "body"), "body")
    reopened = CorpusStore(tmp_path / "corpus")
    assert reopened.has("abc123")
    assert reopened.put(with_body(rec(), "again"), "again") is False


def test_corrupt_middle_line_names_lineno(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec("aaaa"), "a"), "a")
    store.put(with_body(rec("bbbb"), "b"), "b")
    p = tmp_path / "corpus" / "records.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[0] = "{not json}"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match=":1: corrupt record"):
        list(CorpusStore(tmp_path / "corpus").iter_records())


def test_torn_tail_resumes(tmp_path: Path) -> None:
    # A crash mid-append leaves a final line without '\n': iter skips it,
    # has() still works, and the NEXT put seals the tail and appends cleanly.
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec("aaaa"), "a"), "a")
    with (tmp_path / "corpus" / "records.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"id": "partial", "date": ')  # torn, no newline
    reopened = CorpusStore(tmp_path / "corpus")
    assert reopened.has("aaaa")
    r = with_body(rec("cccc"), "c")
    assert reopened.put(r, "c") is True
    assert len(list(reopened.iter_records())) == 2  # torn line gone, both real


def test_parseable_unterminated_tail_is_sealed_not_dropped(tmp_path: Path) -> None:
    # A COMPLETE record that merely lost its trailing newline must survive
    # the next put (deepseek/qwen both reproduced the silent deletion).
    store = CorpusStore(tmp_path / "corpus")
    r1 = with_body(rec("aaaa"), "a")
    store.put(r1, "a")
    rp = tmp_path / "corpus" / "records.jsonl"
    rp.write_text(rp.read_text(encoding="utf-8").removesuffix("\n"), encoding="utf-8")
    reopened = CorpusStore(tmp_path / "corpus")
    reopened.put(with_body(rec("bbbb"), "b"), "b")
    fresh = CorpusStore(tmp_path / "corpus")
    assert [r.id for r in fresh.iter_records()] == ["aaaa", "bbbb"]


def test_bodies_dir_is_0700(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec("deadbeef"), "x"), "x")
    bodies = tmp_path / "corpus" / "bodies"
    assert bodies.stat().st_mode & 0o077 == 0  # mkdir(mode=) is leaf-only


def test_constructor_is_side_effect_free_and_verify_warns(tmp_path: Path) -> None:
    # A read-only command pointed at ~/ must not clamp it (review minor):
    corpus = tmp_path / "corpus"
    corpus.mkdir(mode=0o755, parents=True, exist_ok=True)
    CorpusStore(corpus)
    assert corpus.stat().st_mode & 0o077  # UNCHANGED
    assert str(corpus) in CorpusStore(corpus).verify_modes()


def test_enforce_modes_hardens_whole_tree(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    store = CorpusStore(corpus)
    store.put(with_body(rec("deadbeef"), "x"), "x")
    body = corpus / "bodies" / "de" / "deadbeef.txt"
    corpus.chmod(0o755)
    body.chmod(0o644)
    store.enforce_modes()  # write-side startup call
    assert corpus.stat().st_mode & 0o077 == 0
    assert body.stat().st_mode & 0o077 == 0


def test_oversized_unterminatable_tail_refuses_destructively(tmp_path: Path) -> None:
    # Blocker regression (deepseek): scan exhaustion must NEVER truncate(0).
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec("aaaa"), "a"), "a")
    with (tmp_path / "corpus" / "records.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("x" * ((1 << 20) + 7))  # junk tail with no newline, over cap
    with pytest.raises(CorpusError, match="refusing to truncate"):
        CorpusStore(tmp_path / "corpus").put(with_body(rec("bbbb"), "b"), "b")
    # the valid prefix survives untouched; READ resumes (torn tail skipped)
    # while WRITE keeps refusing until a human repairs the file.
    fresh = CorpusStore(tmp_path / "corpus")
    assert [r.id for r in fresh.iter_records()] == ["aaaa"]
    assert (tmp_path / "corpus" / "records.jsonl").stat().st_size > (1 << 20)


def test_nested_corpus_path_creates_parents(tmp_path: Path) -> None:
    # major regression: ~/mail/corpus-style nested roots must work again.
    store = CorpusStore(tmp_path / "a" / "b" / "corpus")
    assert store.put(with_body(rec("cafe"), "x"), "x") is True


def test_root_existing_file_is_typed_error(tmp_path: Path) -> None:
    blocker = tmp_path / "corpus"
    blocker.write_text("not a dir", encoding="utf-8")
    store = CorpusStore(blocker)
    with pytest.raises(CorpusError, match="unusable"):
        store.put(with_body(rec("cafe"), "x"), "x")


def test_rewritten_body_gets_0600_even_if_previously_loose(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec("cafe"), "v1"), "v1")
    body = tmp_path / "corpus" / "bodies" / "ca" / "cafe.txt"
    body.chmod(0o644)
    fresh = CorpusStore(tmp_path / "corpus")
    # id present -> no rewrite happens (append-only); enforce_modes covers it...
    assert fresh.put(with_body(rec("cafe"), "v1"), "v1") is False
    fresh.enforce_modes()
    assert body.stat().st_mode & 0o077 == 0


def test_has_refuses_corpus_iter_cannot_read(tmp_path: Path) -> None:
    # The resume gate must be exactly as strict as the reader: a line that
    # iter_records rejects must make has() loud, not report "already done".
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec("aaaa"), "a"), "a")
    rp = tmp_path / "corpus" / "records.jsonl"
    rp.write_text(rp.read_text(encoding="utf-8") + '{"id": "half"}\n', encoding="utf-8")
    reopened = CorpusStore(tmp_path / "corpus")
    with pytest.raises(CorpusError):
        reopened.has("anything")


def test_iter_raises_on_non_dict_line(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec("aaaa"), "a"), "a")
    p = tmp_path / "corpus" / "records.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "5\n", encoding="utf-8")
    with pytest.raises(CorpusError, match=":2: corrupt record"):
        list(store.iter_records())


# ---- snapshot I/O (ticket #66: the store owns corpus-root snapshot files) ----


def test_read_snapshot_absent_returns_none(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    assert store.read_snapshot("filters.json") is None


def test_read_snapshot_present_returns_exact_text(tmp_path: Path) -> None:
    # exact text, byte-for-byte: unicode and trailing whitespace included
    payload = '{"a": "é"}  \n'
    store = CorpusStore(tmp_path / "corpus")
    store.write_snapshot("labels.json", payload)
    reopened = CorpusStore(tmp_path / "corpus")
    assert reopened.read_snapshot("labels.json") == payload


def test_write_snapshot_creates_root_0700_and_file_0600(tmp_path: Path) -> None:
    # secret material from the FIRST byte: root 0700, snapshot 0600-at-fd
    store = CorpusStore(tmp_path / "deep" / "corpus")
    store.write_snapshot("filters.json", "[]")
    assert (tmp_path / "deep" / "corpus").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "deep" / "corpus" / "filters.json").stat().st_mode & 0o777 == (
        0o600
    )


def test_write_snapshot_fsyncs_file_and_dir(tmp_path: Path, monkeypatch) -> None:
    # the rename-durability dir fsync must stay pinned at the store seam
    import os as _os

    synced: list[str] = []
    real_fsync = _os.fsync

    def spy(fno: int) -> None:
        synced.append("fsync")
        real_fsync(fno)

    monkeypatch.setattr(_os, "fsync", spy)
    CorpusStore(tmp_path / "c").write_snapshot("filters.json", "[]")
    assert synced.count("fsync") >= 2  # file + directory (tokens.py parity)


def test_write_snapshot_midwrite_failure_leaves_no_tmp(
    tmp_path: Path, monkeypatch
) -> None:
    import os as _os

    store = CorpusStore(tmp_path / "c")

    def boom(*a: object, **k: object) -> None:
        raise OSError("ENOSPC-ish")

    monkeypatch.setattr(_os, "fdopen", boom)
    with pytest.raises(CorpusError):
        store.write_snapshot("filters.json", "[]")
    assert not (store.root / "filters.json.tmp").exists()


def test_write_snapshot_fchmod_failure_cleans_up(tmp_path: Path, monkeypatch) -> None:
    # the fd-ownership finally: no leaked fd, no tmp litter
    import os as _os

    store = CorpusStore(tmp_path / "c")
    monkeypatch.setattr(
        _os, "fchmod", lambda *a, **k: (_ for _ in ()).throw(OSError("EPERM"))
    )
    fd_cand = next(
        (c for c in ("/proc/self/fd", "/dev/fd") if _os.path.isdir(c)), None
    )  # linux + macos legs both mutation-pin the fd fold; the count is
    # process-global, so it assumes no other thread/plugin opens fds mid-call
    # (single-process contract — same prior art as test_gmail_import.py)
    fd_before = len(_os.listdir(fd_cand)) if fd_cand else 0
    with pytest.raises(CorpusError):
        store.write_snapshot("filters.json", "[]")
    assert not (store.root / "filters.json.tmp").exists()
    if not fd_cand:
        pytest.skip("no fd table to count — fd-fold invisible on this host")
    assert len(_os.listdir(fd_cand)) == fd_before


def test_write_snapshot_replace_failure_tightens_tmp_and_is_resumable(
    tmp_path: Path, monkeypatch
) -> None:
    # never-read-a-partial-state: the final path only ever appears via
    # os.replace; a failed replace leaves the COMPLETE tmp, tightened to
    # 0600 even when it pre-existed loose (the O_CREAT fchmod trap), and
    # the next write re-writes-and-renames it away (resumable).
    import os as _os

    store = CorpusStore(tmp_path / "c")
    store.write_snapshot("filters.json", '["v1"]')
    tmp = store.root / "filters.json.tmp"
    tmp.write_text("stale loose litter", encoding="utf-8")
    tmp.chmod(0o666)
    real_replace = _os.replace

    def boom(a: object, b: object, **k: object) -> None:
        raise OSError("EXDEV-ish")

    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(CorpusError):
        store.write_snapshot("filters.json", '["v2"]')
    assert tmp.is_file()  # complete tmp kept, never deleted
    assert tmp.stat().st_mode & 0o777 == 0o600  # pre-existing tmp tightened
    assert (store.root / "filters.json").read_text(encoding="utf-8") == '["v1"]'
    monkeypatch.setattr(_os, "replace", real_replace)
    store.write_snapshot("filters.json", '["v2"]')  # the next run heals it
    assert (store.root / "filters.json").read_text(encoding="utf-8") == '["v2"]'
    assert not tmp.exists()


def test_enforce_modes_hardens_sidecars_and_tmp_remnants(tmp_path: Path) -> None:
    # SECURITY promise made literal (ticket #66): root-level sidecars and
    # stray *.tmp remnants are chmod'd 0600 by the write-side walk.
    corpus = tmp_path / "corpus"
    store = CorpusStore(corpus)
    store.put(with_body(rec("deadbeef"), "x"), "x")
    sidecars = [corpus / "filters.json", corpus / "labels.json"]
    remnants = [corpus / "filters.json.tmp", corpus / "labels.json.tmp"]
    for p in sidecars + remnants:
        p.write_text("{}", encoding="utf-8")
        p.chmod(0o644)
    store.enforce_modes()
    for p in sidecars + remnants:
        assert p.stat().st_mode & 0o077 == 0


def test_verify_modes_lists_loose_sidecar_offender(tmp_path: Path) -> None:
    # read paths warn symmetrically with the enforce walk
    corpus = tmp_path / "corpus"
    store = CorpusStore(corpus)
    store.put(with_body(rec("deadbeef"), "x"), "x")
    filters = corpus / "filters.json"
    labels = corpus / "labels.json"
    filters.write_text("[]", encoding="utf-8")
    labels.write_text("{}", encoding="utf-8")
    filters.chmod(0o600)
    labels.chmod(0o600)
    assert store.verify_modes() == []  # 0600 sidecars: clean
    filters.chmod(0o644)
    offenders = store.verify_modes()
    assert str(filters) in offenders
    assert str(labels) not in offenders


def test_snapshot_names_are_bare_filenames(tmp_path: Path) -> None:
    # containment, mirroring _body_path: a traversal name is a caller bug
    # refused before any filesystem touch (ticket #66 review MINOR)
    secret = tmp_path / "secret.txt"
    secret.write_text("outside", encoding="utf-8")
    store = CorpusStore(tmp_path / "corpus")
    for bad in (
        "../secret.txt",
        "nested/x.json",
        "/etc/passwd",
        "",
        ".",
        "..",
        "a\x00b",
    ):
        with pytest.raises(CorpusError, match="bare filename"):
            store.read_snapshot(bad)
        with pytest.raises(CorpusError, match="bare filename"):
            store.write_snapshot(bad, "x")
    assert secret.read_text(encoding="utf-8") == "outside"  # untouched
    assert not (tmp_path / "corpus").exists()  # refused before mkdir


# ---- revision log: resolved view + append_revision (ADR-0007, ticket #70) --


def test_resolved_view_is_identity_on_duplicate_free_log(tmp_path: Path) -> None:
    # Zero-migration promise (ADR-0007): a log written only through put()
    # folds to itself — old corpora read identically through the new seam.
    store = CorpusStore(tmp_path / "corpus")
    for i in ("aaaa", "bbbb", "cccc"):
        store.put(with_body(rec(i), f"body {i}"), f"body {i}")
    assert list(store.iter_resolved()) == list(store.iter_records())


def test_resolved_view_folds_last_occurrence_wins(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    r1 = with_body(rec("aaaa"), "a")
    store.put(r1, "a")
    r2 = replace(r1, labels=("Shipping", "Bills"))
    assert store.append_revision(r2) is True
    assert list(store.iter_resolved()) == [r2]  # the newer label set wins
    assert len(list(store.iter_records())) == 2  # the raw log keeps both


def test_resolved_view_order_is_first_appearance(tmp_path: Path) -> None:
    # Log position is the version, but the fold's sequence contract is
    # first-appearance order carrying each id's LAST line (pinned so a
    # refactor cannot silently flip it).
    store = CorpusStore(tmp_path / "corpus")
    a = with_body(rec("aaaa"), "a")
    b = with_body(rec("bbbb"), "b")
    store.put(a, "a")
    store.put(b, "b")
    store.append_revision(replace(a, labels=("Z",)))
    assert [r.id for r in store.iter_resolved()] == ["aaaa", "bbbb"]


def test_resolved_view_keeps_the_strict_reader(tmp_path: Path) -> None:
    # The fold inherits iter_records' posture: corrupt middle line raises
    # with :lineno:, never a silent guess.
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec("aaaa"), "a"), "a")
    store.put(with_body(rec("bbbb"), "b"), "b")
    p = tmp_path / "corpus" / "records.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[0] = "{not json}"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match=":1: corrupt record"):
        list(CorpusStore(tmp_path / "corpus").iter_resolved())


def test_resolved_view_skips_torn_tail(tmp_path: Path) -> None:
    # A crash tail is skipped by the resolved view exactly as iter_records
    # skips it — the fold never resurrects or refuses it differently.
    store = CorpusStore(tmp_path / "corpus")
    a = with_body(rec("aaaa"), "a")
    store.put(a, "a")
    with (tmp_path / "corpus" / "records.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"id": "partial", "date": ')  # torn, no newline
    reopened = CorpusStore(tmp_path / "corpus")
    assert list(reopened.iter_resolved()) == [a]


def test_resolved_view_survives_reload(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    r = with_body(rec("aaaa"), "a")
    store.put(r, "a")
    rev = replace(r, labels=("New",))
    store.append_revision(rev)
    reopened = CorpusStore(tmp_path / "corpus")
    assert list(reopened.iter_resolved()) == [rev]


def test_append_revision_refuses_absent_id(tmp_path: Path) -> None:
    # A revision for an id with no resident record would append a line
    # pointing at a body that does not exist — the durability contract
    # forbids it, and sync only ever revises corpus-resident ids.
    store = CorpusStore(tmp_path / "corpus")
    store.put(with_body(rec("aaaa"), "a"), "a")
    with pytest.raises(CorpusError, match="cannot revise absent id"):
        store.append_revision(with_body(rec("ffff"), "ghost"))
    assert len(list(store.iter_records())) == 1  # nothing was appended


def test_append_revision_writes_present_id_without_touching_body(
    tmp_path: Path,
) -> None:
    store = CorpusStore(tmp_path / "corpus")
    r = with_body(rec("aaaa"), "original body")
    store.put(r, "original body")
    body = tmp_path / "corpus" / "bodies" / "aa" / "aaaa.txt"
    before = body.read_bytes()
    rev = replace(r, labels=("New",))
    assert store.append_revision(rev) is True
    assert body.read_bytes() == before  # the store neither rewrote nor read it
    assert store.get_body("aaaa") == "original body"
    assert list(store.iter_resolved()) == [rev]


def test_append_revision_never_reads_the_body(tmp_path: Path) -> None:
    # ADR-0007: the revision path does no body I/O — sync never has the
    # body at hand, and the hash-vs-body tripwire stays on the body-READ
    # path. Proven structurally: with the body file removed from disk the
    # revision still appends (any read or hash-vs-body check would fail
    # or diverge). Deliberately does NOT pin the absence of a future
    # in-memory field-parity check (round-1 review finding).
    store = CorpusStore(tmp_path / "corpus")
    r = with_body(rec("aaaa"), "a")
    store.put(r, "a")
    body = tmp_path / "corpus" / "bodies" / "aa" / "aaaa.txt"
    body.unlink()
    assert store.append_revision(replace(r, labels=("New",))) is True
    assert not body.exists()  # no read, no resurrection


def test_append_revision_line_is_newline_terminated(tmp_path: Path) -> None:
    # The terminator is part of the durability contract: without it the
    # next append glues two records into one unparseable line (mutant-
    # verified: dropping the newline terminator here was invisible to
    # every other test).
    store = CorpusStore(tmp_path / "corpus")
    r = with_body(rec("aaaa"), "a")
    store.put(r, "a")
    store.append_revision(replace(r, labels=("New",)))
    raw = (tmp_path / "corpus" / "records.jsonl").read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert len(raw.splitlines()) == 2


def test_append_revision_replay_is_harmless(tmp_path: Path) -> None:
    # Crash-retry of a Label Sync re-appends the identical line; the
    # last-wins fold absorbs replays (ADR-0007).
    store = CorpusStore(tmp_path / "corpus")
    r = with_body(rec("aaaa"), "a")
    store.put(r, "a")
    rev = replace(r, labels=("New",))
    assert store.append_revision(rev) is True
    assert store.append_revision(rev) is True
    assert len(list(store.iter_records())) == 3  # the log is honest
    assert list(store.iter_resolved()) == [rev]  # the view is stable


def test_append_revision_repairs_torn_tail_like_put(tmp_path: Path) -> None:
    # AC4: the crash-tail contract is put's, verbatim — an unparseable
    # fragment is truncated, then the revision appends cleanly.
    store = CorpusStore(tmp_path / "corpus")
    a = with_body(rec("aaaa"), "a")
    store.put(a, "a")
    with (tmp_path / "corpus" / "records.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"id": "partial", "date": ')
    reopened = CorpusStore(tmp_path / "corpus")
    assert reopened.append_revision(replace(a, labels=("New",))) is True
    assert len(list(reopened.iter_records())) == 2  # torn line gone, both real


def test_append_revision_seals_parseable_tail(tmp_path: Path) -> None:
    # A COMPLETE line that merely lost its newline is sealed and kept,
    # mirroring test_parseable_unterminated_tail_is_sealed_not_dropped.
    store = CorpusStore(tmp_path / "corpus")
    a = with_body(rec("aaaa"), "a")
    store.put(a, "a")
    rp = tmp_path / "corpus" / "records.jsonl"
    rp.write_text(rp.read_text(encoding="utf-8").removesuffix("\n"), encoding="utf-8")
    reopened = CorpusStore(tmp_path / "corpus")
    assert reopened.append_revision(replace(a, labels=("New",))) is True
    assert len(list(reopened.iter_records())) == 2


def test_append_revision_tightens_loose_records_file(tmp_path: Path) -> None:
    # AC4: permission behavior identical — 0600 at the fd plus fchmod
    # tightening of a pre-existing loose file (put's _open_append_checked).
    store = CorpusStore(tmp_path / "corpus")
    r = with_body(rec("deadbeef"), "x")
    store.put(r, "x")
    records = tmp_path / "corpus" / "records.jsonl"
    records.chmod(0o644)
    store.append_revision(replace(r, labels=("New",)))
    assert records.stat().st_mode & 0o077 == 0


def test_put_still_refuses_present_ids_after_revisions(tmp_path: Path) -> None:
    # AC2: put()'s never-duplicate contract survives the revision log —
    # the acquisition commands stay structurally incapable of revising.
    store = CorpusStore(tmp_path / "corpus")
    r = with_body(rec("aaaa"), "a")
    store.put(r, "a")
    store.append_revision(replace(r, labels=("New",)))
    assert store.put(with_body(rec("aaaa"), "other"), "other") is False
    assert store.get_body("aaaa") == "a"  # body untouched by the refusal
    assert len(list(store.iter_records())) == 2  # only the revision line
