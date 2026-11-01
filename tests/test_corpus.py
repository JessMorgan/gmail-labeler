from __future__ import annotations

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
    # fixed -05:00 == US/Eastern on 2026-01-01 (EST); needs no tz database
    eastern = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=-5)))
    # the exact instant, not just the offset: an astimezone→replace
    # mutation (a 5 h wrong instant) must fail this pin
    assert utc_datetime(eastern) == datetime(2026, 1, 1, 5, tzinfo=UTC)
    assert utc_datetime(eastern).tzinfo is UTC


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
