"""gmail_import import loop: resume gate, retry taxonomy, snapshot writers."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gmail_labeler.corpus_record import CorpusRecord, body_hash
from gmail_labeler.corpus_store import CorpusError, CorpusStore
from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
from gmail_labeler.gmail_client import LabelInfo
from gmail_labeler.gmail_errors import GmailPermanentError, classify_status
from gmail_labeler.gmail_import import import_gmail
from gmail_labeler.sync_cursor import SIDECAR_CURSOR, SyncCursor


def _eml(
    subject: str = "Hi", date: str | None = "Sun, 30 Oct 2026 10:00:00 +0000"
) -> bytes:
    lines = [f"Subject: {subject}", "From: a@x.com", "To: me@x.com"]
    if date is not None:
        lines.append(f"Date: {date}")
    lines += ["Content-Type: text/plain", "", f"body of {subject}"]
    return "\r\n".join(lines).encode()


NAMES = {"L1": "Shipping", "L2": "Money/Bills"}


def store_at(tmp_path: Path) -> CorpusStore:
    return CorpusStore(tmp_path / "corpus")


def test_import_gmail_writes_records_with_normalized_fields(tmp_path: Path) -> None:
    fake = FakeGmailClient(
        messages=[
            FakeMessage(
                id="aaa",
                eml=_eml("One"),
                label_ids=("INBOX", "L1", "GONE"),
                thread_id="t1",
                internal_date_ms=1761818400000,  # authoritative
            ),
            FakeMessage(id="bbb", eml=_eml("Two"), label_ids=("L2",), thread_id="t2"),
        ],
        labels=[LabelInfo("L1", "Shipping", "user")],
    )
    store = store_at(tmp_path)
    stats = import_gmail(fake, store, id_to_name=NAMES)
    assert stats.written == 2 and not stats.failed
    records = {r.id: r for r in store.iter_records()}
    one = records["aaa"]
    assert one.labels == ("Shipping", "id:GONE")  # System Labels filtered,
    # deleted labels surfaced (invariant #6 posture)
    assert one.date.isoformat().startswith("2025-10-30")  # from internalDate
    two = records["bbb"]
    assert two.date.isoformat().startswith("2026-10-30")  # from MIME Date
    body = store.get_body("aaa")
    assert body == "body of One"  # stored body == hashed body


def test_import_gmail_resume_skips_present(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    store = store_at(tmp_path)
    import_gmail(fake, store, id_to_name=NAMES)
    calls_before = len(fake.calls)
    stats = import_gmail(fake, store, id_to_name=NAMES)
    assert stats.written == 0 and stats.skipped_present == 1
    # skip happened without even fetching the message (id-set cursor)
    fetches = [c for c in fake.calls[calls_before:] if c.startswith("get_message")]
    assert fetches == []


def test_transient_retries_then_succeeds(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    fake.fail("get_message", classify_status(503))
    slept: list[float] = []
    stats = import_gmail(
        fake, store_at(tmp_path), id_to_name=NAMES, sleeper=slept.append
    )
    assert stats.written == 1 and stats.retried == 1 and len(slept) == 1


def test_rate_limit_backoff_honors_retry_after(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    fake.fail("get_message", classify_status(429, retry_after_s=7.0))
    slept: list[float] = []
    stats = import_gmail(
        fake, store_at(tmp_path), id_to_name=NAMES, sleeper=slept.append
    )
    assert stats.written == 1
    assert slept[0] == 7.0  # server-said delay, not the exponential guess


def test_permanent_failure_records_and_continues(tmp_path: Path) -> None:
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="aaa", eml=_eml()),
            FakeMessage(id="bbb", eml=_eml("B")),
        ]
    )
    fake.fail("get_message", classify_status(404))
    stats = import_gmail(
        fake, store_at(tmp_path), id_to_name=NAMES, sleeper=lambda s: None
    )
    # permanent = do NOT retry: get_message:aaa issued exactly once, aaa
    # failed, and the run continued to bbb
    assert stats.failed == [("aaa", "fetch failed (permanent/retries)")]
    assert fake.calls.count("get_message:aaa") == 1
    assert stats.written == 1


def test_unrecoverable_retries_exhaust(tmp_path: Path) -> None:
    class AlwaysDown(FakeGmailClient):
        def get_message(self, message_id: str):  # type: ignore[no-untyped-def]
            raise classify_status(503)

    down = AlwaysDown(messages=[FakeMessage(id="aaa", eml=_eml())])
    stats = import_gmail(
        down,
        store_at(tmp_path),
        id_to_name=NAMES,
        max_attempts=2,
        sleeper=lambda s: None,
    )
    assert stats.failed and stats.failed[0][0] == "aaa"


def test_corrupt_raw_and_missing_dates_are_failed_not_fabricated(
    tmp_path: Path,
) -> None:
    class BadRaw(FakeGmailClient):
        def get_message(self, message_id: str):  # type: ignore[no-untyped-def]
            from gmail_labeler.gmail_client import RawMessage

            if message_id == "ccc":
                # non-ASCII raw: the one input class non-strict base64
                # cannot silently swallow (ascii encode explodes)
                return RawMessage("ccc", "t", (), None, "\u20ac" * 8)
            return super().get_message(message_id)

    fake = BadRaw(
        messages=[
            FakeMessage(id="ccc", eml=_eml()),
            FakeMessage(id="ddd", eml=_eml("NoDate", date=None)),
            FakeMessage(id="not-hex!", eml=_eml("Z")),
        ]
    )
    stats = import_gmail(
        fake, store_at(tmp_path), id_to_name=NAMES, sleeper=lambda s: None
    )
    reasons = dict(stats.failed)
    assert reasons["ccc"].startswith("raw decode")  # loud decode, documented
    assert reasons["ddd"] == "no authoritative timestamp"  # never now()
    assert reasons["not-hex!"].startswith("record shape")  # wire lies -> fail 1
    assert stats.written == 0


def test_pagination_walks_all_pages(tmp_path: Path) -> None:
    fake = FakeGmailClient(
        messages=[FakeMessage(id=f"5{i:02x}", eml=_eml(f"M{i}")) for i in range(5)],
        page_size=2,
    )
    stats = import_gmail(fake, store_at(tmp_path), id_to_name=NAMES)
    assert stats.scanned == 5 and stats.written == 5


def test_page_level_failure_stops_run_loudly(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    fake.fail("list_messages", classify_status(503))
    stats = import_gmail(
        fake,
        store_at(tmp_path),
        id_to_name=NAMES,
        max_attempts=1,
        sleeper=lambda s: None,
    )
    assert stats.failed == [
        ("<page>", "list_messages stopped (permanent or exhausted)")
    ]


def test_max_messages_cap_and_case_insensitive_ids(tmp_path: Path) -> None:
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="19F6ABC", eml=_eml("Up")),
            FakeMessage(id="b2", eml=_eml("B")),
        ]
    )
    store = store_at(tmp_path)
    stats = import_gmail(
        fake, store, id_to_name=NAMES, max_messages=1, sleeper=lambda s: None
    )
    assert stats.scanned == 1
    ids = [r.id for r in store.iter_records()]
    assert ids == ["19f6abc"]  # ids normalized to lowercase
    # rerun: the UPPERCASE stub id still matches the stored lowercase id
    stats2 = import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    assert stats2.skipped_present >= 1


def test_max_messages_bound_holds_on_warm_resume(tmp_path: Path) -> None:
    # M7 CLOSED: every row already present — the skip-present continue path
    # used to bypass the cap (deepseek engine pass)
    ids = ["aa", "bb", "cc", "dd"]
    store = store_at(tmp_path)
    for i, mid in enumerate(ids):
        store.put(
            CorpusRecord(
                id=mid,
                date=datetime(2026, 7, 15, tzinfo=UTC),
                subject="x",
                sender="a@b.c",
                recipients=("m@x.y",),
                labels=("Shipping",),
                body_hash=body_hash(f"b{i}"),
                source="api",
            ),
            f"b{i}",
        )
    fake = FakeGmailClient(messages=[FakeMessage(id=mid, eml=_eml()) for mid in ids])
    fresh = CorpusStore(tmp_path / "corpus")
    stats = import_gmail(
        fake, fresh, id_to_name=NAMES, sleeper=lambda s: None, max_messages=2
    )
    assert stats.scanned == 2 and stats.skipped_present == 2
    assert sum(1 for c in fake.calls if c.startswith("list_messages")) == 1
    assert not [c for c in fake.calls if c.startswith("get_message")]


def test_store_corruption_stops_the_run(tmp_path: Path) -> None:
    from gmail_labeler.corpus_store import CorpusError

    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="aaa", eml=_eml()),
            FakeMessage(id="bbb", eml=_eml("B")),
        ]
    )
    store = store_at(tmp_path)
    import_gmail(fake, store, id_to_name=NAMES)
    # corrupt the index (middle line) so the strict resume gate refuses
    rp = tmp_path / "corpus" / "records.jsonl"
    lines = rp.read_text(encoding="utf-8").splitlines()
    lines[0] = "{broken"
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="corrupt record"):
        import_gmail(fake, store_at(tmp_path), id_to_name=NAMES)


def test_records_round_trip_through_the_store(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml("RT"))])
    store = store_at(tmp_path)
    import_gmail(fake, store, id_to_name=NAMES)
    reopened = CorpusStore(tmp_path / "corpus")
    rec = next(iter(reopened.iter_records()))
    assert isinstance(rec, CorpusRecord)
    assert reopened.get_body("aaa") == "body of RT"


def test_max_messages_zero_is_a_noop(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    stats = import_gmail(fake, store_at(tmp_path), id_to_name=NAMES, max_messages=0)
    assert stats.scanned == stats.written == stats.skipped_present == 0


def test_auth_errors_propagate_typed_not_swallowed(tmp_path: Path) -> None:
    from gmail_labeler.gmail_errors import GmailAuthError

    class Revoked(FakeGmailClient):
        def get_message(self, message_id: str):  # type: ignore[no-untyped-def]
            raise GmailAuthError("revoked", status=401)

    fake = Revoked(messages=[FakeMessage(id="aaa", eml=_eml())])
    with pytest.raises(GmailAuthError):
        import_gmail(fake, store_at(tmp_path), id_to_name=NAMES, sleeper=lambda s: None)


def test_settings_fetch_propagates_auth_too(tmp_path: Path) -> None:
    from gmail_labeler.gmail_errors import GmailAuthError
    from gmail_labeler.gmail_import import fetch_settings

    class DeadAuth(FakeGmailClient):
        def list_labels(self):  # type: ignore[no-untyped-def]
            raise GmailAuthError("consent withdrawn", status=403)

    with pytest.raises(GmailAuthError):
        fetch_settings(DeadAuth(), sleeper=lambda s: None)


def test_snapshot_death_leaves_secret_tight_root(tmp_path: Path) -> None:
    # deepseek MINOR: root created by the snapshot path must be 0700 even
    # when the run dies before the first put()
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.gmail_import import write_filters_snapshot

    store = CorpusStore(tmp_path / "deep" / "corpus")
    write_filters_snapshot(store, [])
    assert (tmp_path / "deep" / "corpus").stat().st_mode & 0o777 == 0o700


def test_snapshot_fsyncs_directory_after_replace(tmp_path: Path, monkeypatch) -> None:
    # deepseek MINOR: the rename-durability dir fsync must stay pinned
    import os as _os

    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.gmail_import import write_filters_snapshot

    synced: list[str] = []
    real_fsync = _os.fsync

    def spy(fno: int) -> None:
        synced.append("fsync")
        real_fsync(fno)

    monkeypatch.setattr(_os, "fsync", spy)
    write_filters_snapshot(CorpusStore(tmp_path / "c"), [])
    assert synced.count("fsync") >= 2  # file + directory (tokens.py parity)


def test_snapshot_failure_leaves_no_tmp(tmp_path: Path, monkeypatch) -> None:
    # deepseek NIT: mid-write OSError must unlink the predictable-named tmp
    import os as _os

    from gmail_labeler.corpus_store import CorpusError, CorpusStore
    from gmail_labeler.gmail_import import write_filters_snapshot

    store = CorpusStore(tmp_path / "c")

    def boom(*a: object, **k: object) -> None:
        raise OSError("ENOSPC-ish")

    monkeypatch.setattr(_os, "fdopen", boom)
    import pytest as _pt

    with _pt.raises(CorpusError):
        write_filters_snapshot(store, [])
    assert not (store.root / "filters.json.tmp").exists()


def test_snapshot_fchmod_failure_cleans_up(tmp_path: Path, monkeypatch) -> None:
    # deepseek r3 NIT: the fd-ownership finally is probe-only otherwise
    import os as _os

    import pytest as _pt

    from gmail_labeler.corpus_store import CorpusError, CorpusStore
    from gmail_labeler.gmail_import import write_filters_snapshot

    store = CorpusStore(tmp_path / "c")
    monkeypatch.setattr(
        _os, "fchmod", lambda *a, **k: (_ for _ in ()).throw(OSError("EPERM"))
    )
    fd_cand = next(
        (c for c in ("/proc/self/fd", "/dev/fd") if _os.path.isdir(c)), None
    )  # linux + macos legs both mutation-pin the fd fold (deepseek r4 NIT)
    fd_before = len(_os.listdir(fd_cand)) if fd_cand else 0
    with _pt.raises(CorpusError):
        write_filters_snapshot(store, [])
    assert not (store.root / "filters.json.tmp").exists()
    if not fd_cand:
        pytest.skip("no fd table to count — fd-fold invisible on this host")
    assert len(_os.listdir(fd_cand)) == fd_before


def test_snapshot_replace_failure_leaves_healable_tmp(
    tmp_path: Path, monkeypatch
) -> None:
    # deepseek r3 NIT: the documented contract — failed replace keeps the
    # complete 0600 tmp; readers saw nothing; next run heals it
    import os as _os

    import pytest as _pt

    from gmail_labeler.attribution import SIDECAR_FILTERS
    from gmail_labeler.corpus_store import CorpusError, CorpusStore
    from gmail_labeler.gmail_import import write_filters_snapshot

    store = CorpusStore(tmp_path / "c")
    real_replace = _os.replace

    def boom(a: object, b: object, **k: object) -> None:
        raise OSError("EXDEV-ish")

    monkeypatch.setattr(_os, "replace", boom)
    with _pt.raises(CorpusError):
        write_filters_snapshot(store, [])
    tmp = store.root / (SIDECAR_FILTERS + ".tmp")
    assert tmp.is_file() and not (store.root / SIDECAR_FILTERS).exists()
    assert tmp.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(_os, "replace", real_replace)
    write_filters_snapshot(store, [])  # the next run heals it
    assert (store.root / SIDECAR_FILTERS).is_file() and not tmp.exists()


# ---- deepseek independent-engine-pass pins ----


def test_resume_gate_is_case_insensitive(tmp_path: Path) -> None:
    # M1-resistant: dropping .lower() in the resume gate refetches every
    # present id on resume (hundreds of thousands of wasted fetches per
    # crash-retry, invisible to CI)

    from gmail_labeler.corpus_record import CorpusRecord

    fake = FakeGmailClient(messages=[FakeMessage(id="19F6ABC", eml=_eml())])
    store = store_at(tmp_path)
    store.put(
        CorpusRecord(
            id="19f6abc",
            date=datetime(2026, 7, 15, tzinfo=UTC),
            subject="x",
            sender="a@b.c",
            recipients=("m@x.y",),
            labels=("Shipping",),
            body_hash=body_hash("whatever"),
            source="api",
        ),
        "whatever",
    )
    fresh = CorpusStore(tmp_path / "corpus")  # fresh resume cache
    stats = import_gmail(fake, fresh, id_to_name=NAMES, sleeper=lambda s: None)
    assert stats.skipped_present == 1
    assert not [c for c in fake.calls if c.startswith("get_message")]


def test_retry_after_hint_ceiling_pinned(tmp_path: Path) -> None:
    # the 60 s CEILING on server hints is a documented decision — pinned
    from gmail_labeler.gmail_errors import GmailRateLimitedError

    slept: list[float] = []
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    fake.fail("get_message", GmailRateLimitedError("slow down", retry_after_s=300.0))
    import_gmail(
        fake,
        store_at(tmp_path),
        id_to_name=NAMES,
        sleeper=slept.append,
        max_attempts=2,
    )
    assert slept and max(slept) <= 60.0 and 60.0 in slept


def test_max_attempts_below_one_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        import_gmail(
            FakeGmailClient(),
            store_at(tmp_path),
            id_to_name=NAMES,
            max_attempts=0,
            sleeper=lambda s: None,
        )


def test_logs_never_contain_body_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # invariant #5 as a TEST: this loop is the first bulk log path
    import logging as _logging

    marker = "SECRET-BODY-MARKER-8831"
    eml = (
        b"Subject: x\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\n" + marker.encode()
    )
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=eml)])
    with caplog.at_level(_logging.DEBUG):
        import_gmail(fake, store_at(tmp_path), id_to_name=NAMES, sleeper=lambda s: None)
    assert marker not in caplog.text


def test_empty_body_is_a_deliberate_record(tmp_path: Path) -> None:
    # deepseek claimed a contradiction with _raw_nonempty; DISPOSED as a
    # false positive: empty BODY text is legitimate attachment-only mail
    # (the wire guard rejects empty RAW — a different thing). Pinned intent.
    eml = b"Subject: x\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\n"
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=eml)])
    stats = import_gmail(
        fake, store_at(tmp_path), id_to_name=NAMES, sleeper=lambda s: None
    )
    assert stats.written == 1  # label signal survives a body-less mail


# ---- Sync Cursor birth-once (ADR-0008, ticket #72) --------------------------


def _cursor_path(store: CorpusStore) -> Path:
    return store.root / SIDECAR_CURSOR


def test_import_births_cursor_from_the_newest_first_stub(tmp_path: Path) -> None:
    # the sync-guide seed: the FIRST (newest) stub's historyId. Distinct
    # per-stub values make a last-stub read observable — the uniform default
    # would hide it (fake docstring).
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="aaa", eml=_eml("One"), history_id=900),
            FakeMessage(id="bbb", eml=_eml("Two"), history_id=100),
        ],
        current_history_id=900,
    )
    store = store_at(tmp_path)
    import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    born = store.read_cursor()
    assert born is not None and born.history_id == 900  # the FIRST stub wins
    # the page seed saves the quota unit: no profile call at all
    assert "get_profile" not in fake.calls


def test_import_profile_fallback_precedes_first_ingestion(tmp_path: Path) -> None:
    # the CRITICAL ordering (ADR-0008): every seed — page OR profile — is
    # captured BEFORE any message is ingested. A profile fallback taken after
    # page processing would skip label changes made in between; this pins the
    # call sequence on the fake's log (stub_history_id=False is the
    # absent-tolerant quadrant that forces the fallback).
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aaa", eml=_eml("One"))],
        current_history_id=777,
        stub_history_id=False,
    )
    store = store_at(tmp_path)
    import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    profile_at = fake.calls.index("get_profile")
    first_ingest_at = next(
        i for i, c in enumerate(fake.calls) if c.startswith("get_message:")
    )
    assert profile_at < first_ingest_at
    assert fake.calls.count("get_profile") == 1  # ONE call, not a per-page leak
    born = store.read_cursor()
    assert born is not None and born.history_id == 777


def test_import_negative_wire_seed_falls_back_then_refuses(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # a negative historyId is a wire lie the ledger refuses: an unusable
    # PAGE seed takes the guide-ordered profile fallback, and only a profile
    # seed that is ALSO unusable degrades loudly (never an untyped
    # ValueError crash at the CLI). The import stays healthy; the corpus
    # stays birth-eligible.
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aaa", eml=_eml("One"), history_id=-5)],
        current_history_id=-5,
    )
    store = store_at(tmp_path)
    with caplog.at_level(logging.WARNING, logger="gmail_labeler.gmail_import"):
        stats = import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    assert stats.written == 1 and not stats.failed
    assert "get_profile" in fake.calls  # the fallback got its turn
    assert store.read_cursor() is None
    assert any(
        r.name == "gmail_labeler.gmail_import"
        and "no usable historyId seed" in r.getMessage()
        for r in caplog.records
    )

    # a healthy profile rescues an unusable page seed
    fake2 = FakeGmailClient(
        messages=[FakeMessage(id="bbb", eml=_eml("Two"), history_id=-5)],
        current_history_id=800,
    )
    store2 = CorpusStore(tmp_path / "corpus2")
    import_gmail(fake2, store2, id_to_name=NAMES, sleeper=lambda s: None)
    born = store2.read_cursor()
    assert born is not None and born.history_id == 800

    # 0 is as unusable as a negative: history ids count from 1, and a
    # startHistoryId=0 ledger would make the first sync die 400
    fake3 = FakeGmailClient(
        messages=[FakeMessage(id="ccc", eml=_eml("Three"), history_id=0)],
        current_history_id=0,
    )
    store3 = CorpusStore(tmp_path / "corpus3")
    import_gmail(fake3, store3, id_to_name=NAMES, sleeper=lambda s: None)
    assert store3.read_cursor() is None


def test_import_seed_capture_attempted_once_across_pages(tmp_path: Path) -> None:
    # the "ONE call, not a per-page leak" claim, pinned: with the fallback
    # forced on EVERY page (stubs lack historyId) and the profile permanently
    # down, a per-page seed attempt would spend a quota unit per page — the
    # seeding_done latch must hold it to exactly one
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="aaa", eml=_eml("One")),
            FakeMessage(id="bbb", eml=_eml("Two")),
        ],
        current_history_id=777,
        stub_history_id=False,
        page_size=1,
    )
    fake.fail("get_profile", GmailPermanentError("profile down", status=400))
    store = store_at(tmp_path)
    stats = import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    assert stats.written == 2  # both pages ingested
    assert fake.calls.count("get_profile") == 1
    assert store.read_cursor() is None  # no seed, no cursor


def test_import_empty_mailbox_still_births_from_profile(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[], current_history_id=42)
    store = store_at(tmp_path)
    import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    cur = store.read_cursor()
    assert cur is not None and cur.history_id == 42


def test_import_relabel_import_window_never_advances_the_cursor(
    tmp_path: Path,
) -> None:
    # THE ADR-0008 TRAP, executable (AC #3/#4): import → Backfill → import
    # with no sync between. Import #1 births the lower bound L; the mailbox
    # moves on (relabels + new mail); import #2 must leave the cursor
    # BYTE-IDENTICAL so the next sync's window still covers the Backfill.
    # The naive "refresh every run" version passes every other test — this
    # is the pin that kills it.
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aaa", eml=_eml("One"), history_id=900)],
        current_history_id=900,
    )
    store = store_at(tmp_path)
    import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    born = _cursor_path(store).read_bytes()
    assert b'"history_id": 900' in born  # the lower bound is really on disk

    # the Backfill window: mailbox activity between the two imports
    fake.messages["aaa"] = FakeMessage(
        id="aaa", eml=_eml("One"), label_ids=("L1",), history_id=900
    )
    fake.messages["ccc"] = FakeMessage(id="ccc", eml=_eml("Three"), history_id=950)
    fake.order.append("ccc")
    fake.current_history_id = 950

    stats = import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    assert stats.written == 1  # the new mail still lands...
    assert _cursor_path(store).read_bytes() == born  # untouched, byte-for-byte


def test_import_with_live_cursor_skips_the_profile_call(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # birth-when-absent means the whole seed capture is skipped when a cursor
    # exists — even in the fallback quadrant, so a warm import spends zero
    # quota on cursor-adjacent state (it never advances, never re-reads) and
    # stays silent (no spurious birth warning over a healthy ledger)
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aaa", eml=_eml("One"))],
        current_history_id=777,
        stub_history_id=False,
    )
    store = store_at(tmp_path)
    import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    assert fake.calls.count("get_profile") == 1
    fake.messages["ccc"] = FakeMessage(id="ccc", eml=_eml("Three"))
    fake.order.append("ccc")
    fake.current_history_id = 999
    with caplog.at_level(logging.WARNING, logger="gmail_labeler.gmail_import"):
        import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    assert fake.calls.count("get_profile") == 1  # no second seed capture
    assert not [
        r for r in caplog.records if r.name == "gmail_labeler.gmail_import"
    ]  # a warm import has nothing to warn about
    born = store.read_cursor()
    assert born is not None and born.history_id == 777  # never advanced


def test_import_invalid_cursor_fails_loud_never_rebirths(tmp_path: Path) -> None:
    # an existing cursor that does not parse is corruption, not absence: the
    # run dies typed (CorpusError → CLI exit 5) BEFORE ingesting, and the
    # garbage stays on disk for a human — never silently overwritten by a
    # fresh birth (the ADR's never-silent-re-birth rule at the import seam)
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aaa", eml=_eml("One"), history_id=900)],
        current_history_id=900,
    )
    store = store_at(tmp_path)
    store.root.mkdir(parents=True)
    garbage = "{ not json"
    _cursor_path(store).write_text(garbage, encoding="utf-8")
    with pytest.raises(CorpusError, match="sync cursor"):
        import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    assert _cursor_path(store).read_text(encoding="utf-8") == garbage
    # died BEFORE ingesting anything (the seed gate sits above the stub loop)
    assert list(store.iter_records()) == []
    assert not store.get_body("aaa")


def test_import_profile_failure_degrades_loudly_not_fatally(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # the cursor is ADDITIVE: a permanently failing profile fallback must not
    # kill an otherwise-healthy import (existing import behavior unchanged) —
    # it warns, leaves no cursor, and the corpus stays birth-eligible
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aaa", eml=_eml("One"))],
        current_history_id=777,
        stub_history_id=False,
    )
    fake.fail("get_profile", GmailPermanentError("profile unavailable", status=400))
    store = store_at(tmp_path)
    with caplog.at_level(logging.WARNING, logger="gmail_labeler.gmail_import"):
        stats = import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    assert stats.written == 1 and not stats.failed  # the import itself is fine
    assert store.read_cursor() is None  # no cursor, no fabricated seed
    assert any(
        r.name == "gmail_labeler.gmail_import"
        and "no usable historyId seed" in r.getMessage()
        for r in caplog.records
    )  # the birth-refused warning itself, not any cursor-shaped log line

    # and the NEXT import still gets its birth chance (birth-when-absent is
    # not consumed by a failed attempt)
    import_gmail(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
    assert store.read_cursor() is not None


def test_import_clock_stamps_the_born_cursor(tmp_path: Path) -> None:
    # last_sync is birth time, tz-aware UTC, from the injected clock (the
    # sleeper's sibling seam — no wall-clock reads in tests)
    born_at = datetime(2026, 11, 4, 9, 30, tzinfo=UTC)
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aaa", eml=_eml("One"), history_id=50)],
        current_history_id=50,
    )
    store = store_at(tmp_path)
    import_gmail(
        fake,
        store,
        id_to_name=NAMES,
        sleeper=lambda s: None,
        clock=lambda: born_at,
    )
    assert store.read_cursor() == SyncCursor(history_id=50, last_sync=born_at)
