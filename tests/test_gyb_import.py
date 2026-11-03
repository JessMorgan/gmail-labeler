from __future__ import annotations

import sqlite3
from datetime import UTC
from pathlib import Path

import pytest

from gmail_labeler.corpus_store import CorpusStore
from gmail_labeler.gyb_import import import_gyb, read_gyb_index


def make_gyb(root: Path, rows: list[tuple[str, str, list[str]]]) -> Path:
    """rows: (uid, 'YYYY-MM-DD HH:MM:SS', labels) -> msg-db + .eml files."""
    root.mkdir(parents=True, exist_ok=True)
    db = root / "msg-db.sqlite"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE messages (
            message_num INTEGER PRIMARY KEY,
            message_filename TEXT,
            message_internaldate TEXT
        );
        CREATE TABLE labels (
            message_num INTEGER, label TEXT, UNIQUE(message_num, label)
        );
        CREATE TABLE uids (message_num INTEGER PRIMARY KEY, uid TEXT);
        """
    )
    for i, (uid, date, labels) in enumerate(rows, start=1):
        rel = f"2026/7/15/{uid}.eml"
        eml = root / rel
        eml.parent.mkdir(parents=True, exist_ok=True)
        eml.write_bytes(
            b"Subject: mail " + uid.encode() + b"\r\nFrom: s@x.com\r\n"
            b"To: me@x.com\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n"
            b"Content-Type: text/plain\r\n\r\nbody " + uid.encode()
        )
        con.execute("INSERT INTO messages VALUES (?,?,?)", (i, rel, date))
        con.execute("INSERT INTO uids VALUES (?,?)", (i, uid))
        con.executemany("INSERT INTO labels VALUES (?,?)", [(i, lab) for lab in labels])
    con.commit()
    con.close()
    return db


@pytest.fixture
def gyb(tmp_path: Path) -> Path:
    make_gyb(
        tmp_path,
        [
            ("aaaa", "2026-07-15 23:48:51", ["INBOX", "UNREAD", "Work/Woodworking"]),
            ("bbbb", "2026-07-16 10:00:00", ["CATEGORY_PROMOTIONS", "Money/Bills"]),
            # localized system name (non-English GYB export) passes unless
            # declared — pinned below via system_labels_extra
            ("cccc", "2026-07-17 09:00:00", ["INKASSIE", "Friends"]),
        ],
    )
    return tmp_path


def test_index_reads_joined_rows_in_date_order(gyb: Path) -> None:
    rows = list(read_gyb_index(gyb / "msg-db.sqlite"))
    assert [r.id for r in rows] == ["aaaa", "bbbb", "cccc"]
    a = rows[0]
    assert a.labels == ("Work/Woodworking",)  # system tokens filtered in name
    # space (canonical uppercase equality verified against a real corpus)
    assert a.when.year == 2026 and a.when.tzinfo is UTC
    assert a.eml_relpath.endswith(".eml")


def test_import_writes_normalized_records(gyb: Path, tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    stats = import_gyb(gyb, store)
    assert stats.written == 3
    recs = {r.id: r for r in store.iter_records()}
    assert recs["bbbb"].labels == ("Money/Bills",)
    # group_concat order is SQL grouping order, not insertion order (see
    # read_gyb_index docstring) — compare as sets
    assert set(recs["cccc"].labels) == {"INKASSIE", "Friends"}  # undeclared:
    # localized system name passes through — documented leak posture, pinned
    assert store.get_body("aaaa") == "body aaaa"
    assert all(r.source == "gyb" for r in recs.values())
    assert all(r.thread_id == "" for r in recs.values())  # never guessed
    # GYB stamps its own source on the record contract (the folder-
    # corrected internal-date) — distinct from the two API provenances
    assert all(r.date_provenance == "gyb_internaldate" for r in recs.values())


def test_system_labels_extra_localizes_the_filter(gyb: Path, tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    import_gyb(gyb, store, system_labels_extra=frozenset({"INKASSIE"}))
    recs = {r.id: r for r in store.iter_records()}
    assert recs["cccc"].labels == ("Friends",)


def test_resume_skips_present_without_touching_disk(gyb: Path, tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "corpus")
    import_gyb(gyb, store, max_messages=2)
    stats = import_gyb(gyb, store)
    assert stats.written == 1  # the two already-imported were skip-present
    assert stats.skipped_present == 2


def test_missing_eml_fails_that_message_only(gyb: Path, tmp_path: Path) -> None:
    (gyb / "2026/7/15/bbbb.eml").unlink()
    store = CorpusStore(tmp_path / "corpus")
    stats = import_gyb(gyb, store)
    assert [i for i, _ in stats.failed] == ["bbbb"]
    assert stats.written == 2


def test_db_is_never_written(tmp_path: Path) -> None:
    db = make_gyb(tmp_path / "b", [("aaaa", "2026-07-15 00:00:00", ["INBOX"])])
    before = db.read_bytes()
    store = CorpusStore(tmp_path / "c")
    import_gyb(tmp_path / "b", store)
    assert db.read_bytes() == before  # mode=ro is not optional


def test_schema_drift_is_a_loud_abort(tmp_path: Path) -> None:
    db = tmp_path / "msg-db.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE messages (message_num INTEGER, message_filename TEXT,"
        " message_internaldate INTEGER)"
    )
    con.execute("CREATE TABLE labels (message_num INTEGER, label TEXT)")
    con.execute("CREATE TABLE uids (message_num INTEGER, uid TEXT)")
    con.execute("INSERT INTO messages VALUES (1, 'x.eml', 1790000000)")
    con.execute("INSERT INTO uids VALUES (1, 'aa')")
    con.commit()
    con.close()
    from gmail_labeler.gyb_import import GybImportError

    with pytest.raises(GybImportError, match="message_internaldate"):
        list(read_gyb_index(db))


def test_bad_uid_fails_only_that_record(gyb: Path, tmp_path: Path) -> None:
    # corrupt one uid to a path-escaping non-hex value IN THE DB
    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.execute("UPDATE uids SET uid = '../escape' WHERE uid = 'cccc'")
    con.commit()
    con.close()
    store = CorpusStore(tmp_path / "corpus")
    stats = import_gyb(gyb, store)
    assert dict(stats.failed)["../escape"].startswith("record shape")
    assert stats.written == 2


def test_corrupt_db_is_typed_not_raw_traceback(tmp_path: Path) -> None:
    from gmail_labeler.gyb_import import GybImportError, read_gyb_index

    (tmp_path / "msg-db.sqlite").write_bytes(b"not a database at all")
    with pytest.raises(GybImportError, match="database"):
        list(read_gyb_index(tmp_path / "msg-db.sqlite"))


def test_foreign_schema_is_typed(tmp_path: Path) -> None:
    import sqlite3

    from gmail_labeler.gyb_import import GybImportError, read_gyb_index

    con = sqlite3.connect(tmp_path / "msg-db.sqlite")
    con.execute("CREATE TABLE something_else (x INTEGER)")
    con.commit()
    con.close()
    with pytest.raises(GybImportError, match="database read failed"):
        list(read_gyb_index(tmp_path / "msg-db.sqlite"))


def test_fractional_seconds_tolerated(tmp_path: Path) -> None:
    import sqlite3

    from gmail_labeler.gyb_import import read_gyb_index

    con = sqlite3.connect(tmp_path / "msg-db.sqlite")
    con.executescript(
        "CREATE TABLE messages (message_num INTEGER PRIMARY KEY,"
        " message_filename TEXT, message_internaldate TEXT);"
        "CREATE TABLE labels (message_num INTEGER, label TEXT);"
        "CREATE TABLE uids (message_num INTEGER PRIMARY KEY, uid TEXT);"
        "INSERT INTO messages VALUES (1, '2026/7/15/a.eml',"
        " '2026-07-15 23:48:51.123456');"
        "INSERT INTO uids VALUES (1, 'a');"
    )
    con.commit()
    con.close()
    rows = list(read_gyb_index(tmp_path / "msg-db.sqlite"))
    assert rows[0].when.microsecond == 123456 and not rows[0].date_corrected


def test_folder_date_authority_corrects_local_stamps(tmp_path: Path) -> None:
    # backup box at UTC+10: stamp says next-day 02:00, folder (UTC) rules
    import sqlite3

    from gmail_labeler.gyb_import import read_gyb_index

    con = sqlite3.connect(tmp_path / "msg-db.sqlite")
    con.executescript(
        "CREATE TABLE messages (message_num INTEGER PRIMARY KEY,"
        " message_filename TEXT, message_internaldate TEXT);"
        "CREATE TABLE labels (message_num INTEGER, label TEXT);"
        "CREATE TABLE uids (message_num INTEGER PRIMARY KEY, uid TEXT);"
        "INSERT INTO messages VALUES (1, '2026/7/15/a.eml',"
        " '2026-07-16 02:00:00');"
        "INSERT INTO uids VALUES (1, 'a');"
    )
    con.commit()
    con.close()
    rows = list(read_gyb_index(tmp_path / "msg-db.sqlite"))
    assert rows[0].when.day == 15 and rows[0].when.hour == 2
    assert rows[0].date_corrected


def test_uid_gap_stops_the_run(tmp_path: Path) -> None:
    import sqlite3

    import pytest

    from gmail_labeler.gyb_import import GybImportError, read_gyb_index

    con = sqlite3.connect(tmp_path / "msg-db.sqlite")
    con.executescript(
        "CREATE TABLE messages (message_num INTEGER PRIMARY KEY,"
        " message_filename TEXT, message_internaldate TEXT);"
        "CREATE TABLE labels (message_num INTEGER, label TEXT);"
        "CREATE TABLE uids (message_num INTEGER PRIMARY KEY, uid TEXT);"
        "INSERT INTO messages VALUES (1, '2026/7/15/a.eml',"
        " '2026-07-15 10:00:00');"
        "INSERT INTO uids VALUES (1, 'a');"
        "INSERT INTO messages VALUES (2, '2026/7/15/b.eml',"
        " '2026-07-15 11:00:00');"  # no uid row: corrupt index
    )
    con.commit()
    con.close()
    with pytest.raises(GybImportError, match="gap"):
        list(read_gyb_index(tmp_path / "msg-db.sqlite"))


def test_filename_escape_stops_the_run(tmp_path: Path) -> None:
    import sqlite3

    import pytest

    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.gyb_import import GybImportError, import_gyb

    gyb = tmp_path / "gyb"
    (gyb / "2026/7/15").mkdir(parents=True)
    (tmp_path / "secret.eml").write_bytes(b"Subject: s\r\n\r\nsteal me")
    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.executescript(
        "CREATE TABLE messages (message_num INTEGER PRIMARY KEY,"
        " message_filename TEXT, message_internaldate TEXT);"
        "CREATE TABLE labels (message_num INTEGER, label TEXT);"
        "CREATE TABLE uids (message_num INTEGER PRIMARY KEY, uid TEXT);"
        "INSERT INTO messages VALUES (1, '../../../secret.eml',"
        " '2026-07-15 10:00:00');"
        "INSERT INTO uids VALUES (1, 'aa');"
    )
    con.commit()
    con.close()
    store = CorpusStore(tmp_path / "corpus")
    with pytest.raises(GybImportError, match="escapes") as exc:
        import_gyb(gyb, store)
    assert exc.value.stats is not None  # deepseek NIT: partial state attached
    assert not (tmp_path / "corpus" / "records.jsonl").exists()  # nothing read in


def test_engine_max_messages_zero_is_noop(tmp_path: Path) -> None:
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.gyb_import import import_gyb

    stats = import_gyb(tmp_path, CorpusStore(tmp_path / "c"), max_messages=0)
    assert stats.scanned == 0 and stats.written == 0


def test_corrected_rows_already_present_do_not_trigger_note(tmp_path: Path) -> None:
    # deepseek NIT: the count means "corrected INTO the corpus", not
    # "seen while scanning" — a pure-resume run must stay quiet
    import sqlite3

    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.gyb_import import import_gyb

    gyb = tmp_path / "gyb"
    (gyb / "2026/7/15").mkdir(parents=True)
    (gyb / "2026/7/15" / "a.eml").write_bytes(
        b"Subject: x\r\nDate: Wed, 15 Jul 2026 02:00:00 +0000\r\n\r\nhi"
    )
    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.executescript(
        "CREATE TABLE messages (message_num INTEGER PRIMARY KEY,"
        " message_filename TEXT, message_internaldate TEXT);"
        "CREATE TABLE labels (message_num INTEGER, label TEXT);"
        "CREATE TABLE uids (message_num INTEGER PRIMARY KEY, uid TEXT);"
        "INSERT INTO messages VALUES (1, '2026/7/15/a.eml',"
        " '2026-07-16 02:00:00');"  # disagrees with folder date
        "INSERT INTO uids VALUES (1, 'aa');"
        "INSERT INTO labels VALUES (1, 'Work');"
    )
    con.commit()
    con.close()
    store = CorpusStore(tmp_path / "corpus")
    first = import_gyb(gyb, store)
    assert first.written == 1 and first.date_corrected == 1
    store2 = CorpusStore(tmp_path / "corpus")  # fresh resume cache
    second = import_gyb(gyb, store2)
    assert second.skipped_present == 1 and second.date_corrected == 0


def test_store_abort_carries_partial_counts(tmp_path: Path) -> None:
    # deepseek NIT: a store-side abort (CorpusError under the hood) must
    # read like every other exit 5 — wrapped typed, counts attached
    import sqlite3

    import pytest

    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.gyb_import import GybImportError, import_gyb

    gyb = tmp_path / "gyb"
    (gyb / "2026/7/15").mkdir(parents=True)
    for name in ("a", "b"):
        (gyb / "2026/7/15" / f"{name}.eml").write_bytes(
            b"Subject: x\r\nDate: Wed, 15 Jul 2026 10:00:00 +0000\r\n\r\nhi"
        )
    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.executescript(
        "CREATE TABLE messages (message_num INTEGER PRIMARY KEY,"
        " message_filename TEXT, message_internaldate TEXT);"
        "CREATE TABLE labels (message_num INTEGER, label TEXT);"
        "CREATE TABLE uids (message_num INTEGER PRIMARY KEY, uid TEXT);"
        "INSERT INTO messages VALUES (1, '2026/7/15/a.eml',"
        " '2026-07-15 10:00:00');"
        "INSERT INTO messages VALUES (2, '2026/7/15/b.eml',"
        " '2026-07-15 11:00:00');"
        "INSERT INTO uids VALUES (1, 'aa'); INSERT INTO uids VALUES (2, 'bb');"
    )
    con.commit()
    con.close()
    store = CorpusStore(tmp_path / "corpus")
    store.root.mkdir(parents=True)  # enforce_modes chmods, it never creates
    store.enforce_modes()
    (store.root / "records.jsonl").write_text(
        '{"broken": true, "mid": 1}\n', encoding="utf-8"
    )
    with pytest.raises(GybImportError, match="corpus store aborted") as exc:
        import_gyb(gyb, store)
    assert exc.value.stats is not None


def _mini_db(tmp_path: Path, rows: list[tuple[int, str, str]], uid_fmt: str = "%04x"):
    """rows: (num, relpath, stamp) — labels omitted; uids present for all."""
    import sqlite3

    gyb = tmp_path / "gyb"
    gyb.mkdir(exist_ok=True)
    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.executescript(
        "CREATE TABLE messages (message_num INTEGER PRIMARY KEY,"
        " message_filename TEXT, message_internaldate TEXT);"
        "CREATE TABLE labels (message_num INTEGER, label TEXT);"
        "CREATE TABLE uids (message_num INTEGER PRIMARY KEY, uid TEXT);"
    )
    for num, rel, stamp in rows:
        con.execute("INSERT INTO messages VALUES (?, ?, ?)", (num, rel, stamp))
        con.execute("INSERT INTO uids VALUES (?, ?)", (num, uid_fmt % num))
    con.commit()
    con.close()
    return gyb


def test_utc_minus8_earlier_day_corrected(tmp_path: Path) -> None:
    # qwen: backup box at UTC-8 — previous-DAY 18:00 stamp, folder rules
    from gmail_labeler.gyb_import import read_gyb_index

    _mini_db(tmp_path, [(1, "2026/7/15/a.eml", "2026-07-14 18:00:00")])
    (tmp_path / "gyb" / "2026/7/15").mkdir(parents=True)
    (tmp_path / "gyb" / "2026/7/15" / "a.eml").write_bytes(b"Subject: x\r\n\r\nb")
    rows = list(read_gyb_index(tmp_path / "gyb" / "msg-db.sqlite"))
    assert rows[0].when.day == 15 and rows[0].when.hour == 18
    assert rows[0].date_corrected and not rows[0].folder_unparsed


def test_unparsed_folder_counted_not_silent(tmp_path: Path) -> None:
    # qwen MINOR: the no-UTC-anchor row must be VISIBLE, never silent
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.gyb_import import import_gyb

    gyb = _mini_db(tmp_path, [(1, "a.eml", "2026-07-14 18:00:00")])
    (tmp_path / "gyb" / "a.eml").write_bytes(b"Subject: x\r\n\r\nb")
    stats = import_gyb(gyb, CorpusStore(tmp_path / "corpus"))
    assert stats.written == 1 and stats.folder_unparsed == 1
    assert stats.date_corrected == 0


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    # qwen: resolve() follows links out of root -> must be refused
    import pytest

    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.gyb_import import GybImportError, import_gyb

    outside = tmp_path / "out.eml"
    outside.write_bytes(b"Subject: x\r\n\r\nlink me")
    gyb = _mini_db(tmp_path, [(1, "2026/7/15/a.eml", "2026-07-15 10:00:00")])
    day = gyb / "2026/7/15"
    day.mkdir(parents=True)
    (day / "a.eml").symlink_to(outside)
    with pytest.raises(GybImportError, match="escapes"):
        import_gyb(gyb, CorpusStore(tmp_path / "corpus"))


# ---- deepseek independent-engine-pass pins ----


def test_correction_bounded_to_one_day(tmp_path: Path) -> None:
    # THE MAJOR: a 1999 folder must never move 2026 mail across years
    from gmail_labeler.gyb_import import read_gyb_index

    gyb = _mini_db(tmp_path, [(1, "1999/01/01/a.eml", "2026-07-15 10:00:00")])
    rows = list(read_gyb_index(gyb / "msg-db.sqlite"))
    r = rows[0]
    assert r.when.year == 2026 and not r.date_corrected and r.date_conflict


def test_short_year_folder_is_unparsed_not_corrected(tmp_path: Path) -> None:
    # '1/1/1/x.eml' is corruption, not a target date
    from gmail_labeler.gyb_import import read_gyb_index

    gyb = _mini_db(tmp_path, [(1, "1/1/1/a.eml", "2026-07-15 10:00:00")])
    r = next(iter(read_gyb_index(gyb / "msg-db.sqlite")))
    assert r.folder_unparsed and not r.date_corrected and r.when.year == 2026


def test_duplicate_uid_rows_stop_the_run(tmp_path: Path) -> None:
    import sqlite3

    import pytest

    from gmail_labeler.gyb_import import GybImportError, read_gyb_index

    # the stock schema's PK makes this unrepresentable; a hand-edited or
    # rebuilt msg-db (PK dropped) is exactly the corruption the check guards
    gyb = tmp_path / "gyb"
    gyb.mkdir()
    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.executescript(
        "CREATE TABLE messages (message_num INTEGER PRIMARY KEY,"
        " message_filename TEXT, message_internaldate TEXT);"
        "CREATE TABLE labels (message_num INTEGER, label TEXT);"
        "CREATE TABLE uids (message_num INTEGER, uid TEXT);"  # no PK: damaged
        "INSERT INTO messages VALUES (1, '2026/7/15/a.eml',"
        " '2026-07-15 10:00:00');"
        "INSERT INTO uids VALUES (1, 'aa'); INSERT INTO uids VALUES (1, 'cc');"
    )
    con.commit()
    con.close()
    with pytest.raises(GybImportError, match="duplicate message_num"):
        list(read_gyb_index(gyb / "msg-db.sqlite"))


def test_missing_db_is_typed_gyberror(tmp_path: Path) -> None:
    import pytest

    from gmail_labeler.gyb_import import GybImportError, read_gyb_index

    (tmp_path / "msg-db.sqlite").mkdir()  # a DIRECTORY named like the db
    with pytest.raises(GybImportError, match="not a file"):
        list(read_gyb_index(tmp_path / "msg-db.sqlite"))


def test_query_string_in_backup_path_opens_the_right_file(tmp_path: Path) -> None:
    # URI interpolation used to let '?' reroute the open (deepseek MINOR)
    from gmail_labeler.gyb_import import read_gyb_index

    (tmp_path / "q?x").mkdir()
    gyb = _mini_db(tmp_path / "q?x", [(1, "2026/7/15/a.eml", "2026-07-15 10:00:00")])
    (gyb / "2026/7/15").mkdir(parents=True)
    rows = list(read_gyb_index(gyb / "msg-db.sqlite"))
    assert len(rows) == 1


def test_labels_are_a_sorted_deduped_set(tmp_path: Path) -> None:
    import sqlite3

    from gmail_labeler.gyb_import import read_gyb_index

    gyb = _mini_db(tmp_path, [(1, "2026/7/15/a.eml", "2026-07-15 10:00:00")])
    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.executescript(
        "INSERT INTO labels VALUES (1, 'Work');"
        "INSERT INTO labels VALUES (1, 'Billing');"
        "INSERT INTO labels VALUES (1, 'Work');"  # duplicate row
    )
    con.commit()
    con.close()
    r = next(iter(read_gyb_index(gyb / "msg-db.sqlite")))
    assert r.labels == ("Billing", "Work")  # sorted + deduped, byte-stable


def test_max_messages_holds_on_skipped_rows(tmp_path: Path) -> None:
    # every row present: the skip-present continue used to bypass the bound
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.gyb_import import import_gyb

    rows = [
        (n, f"2026/7/15/{n:04x}.eml", f"2026-07-15 {n % 24:02d}:00:00")
        for n in range(1, 11)
    ]
    gyb = _mini_db(tmp_path, rows)
    for _n, rel, _ in rows:
        f = gyb / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"Subject: x\r\n\r\nbody")
    store = CorpusStore(tmp_path / "corpus")
    import_gyb(gyb, store)  # first pass populates
    fresh = CorpusStore(tmp_path / "corpus")
    stats = import_gyb(gyb, fresh, max_messages=3)
    assert stats.scanned == 3 and stats.skipped_present == 3
