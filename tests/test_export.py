from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gmail_labeler.corpus_record import CorpusRecord, body_hash
from gmail_labeler.corpus_store import CorpusStore
from gmail_labeler.export import DEFAULT_QUERY, export_corpus
from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
from gmail_labeler.gmail_client import LabelInfo
from gmail_labeler.gmail_errors import classify_status


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


def test_export_writes_records_with_normalized_fields(tmp_path: Path) -> None:
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
    stats = export_corpus(fake, store, id_to_name=NAMES)
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


def test_date_provenance_stamps_fallback_distinctly_from_internal_date(
    tmp_path: Path,
) -> None:
    # mutation pin (invariant #7): the MIME-Date fallback path must stamp
    # DIFFERENTLY from the server-internalDate path, and split-boundary
    # consumers must be able to filter on the flag as round-tripped record
    # data — not derive it. A forged/future Date header can no longer
    # silently move a record across the temporal split.
    fake = FakeGmailClient(
        messages=[
            FakeMessage(
                id="aaa",
                eml=_eml("Server"),
                internal_date_ms=1761818400000,  # authoritative
            ),
            FakeMessage(id="bbb", eml=_eml("Header")),  # no internalDate
        ],
    )
    store = store_at(tmp_path)
    export_corpus(fake, store, id_to_name=NAMES)
    records = {r.id: r for r in store.iter_records()}
    assert records["aaa"].date_provenance == "internal_date"
    assert records["bbb"].date_provenance == "mime_date"
    fallback = [r.id for r in records.values() if r.date_provenance == "mime_date"]
    assert fallback == ["bbb"]  # the flag is filterable, per-record


def test_unconvertible_internal_date_falls_back_to_header_stamp(
    tmp_path: Path,
) -> None:
    # present-but-unconvertible internalDate (epoch overflow): the record
    # keeps the header date AND stamps mime_date — the stamp follows the
    # date actually used, not the field that was merely present (mutation
    # pin: _ms_to_utc's try/except — without it the OverflowError escapes)
    eml = _eml("Overflow", date="Sun, 30 Oct 2026 10:00:00 +0000")
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aaa", eml=eml, internal_date_ms=10**30)]
    )
    store = store_at(tmp_path)
    export_corpus(fake, store, id_to_name=NAMES)
    rec = next(iter(store.iter_records()))
    assert rec.date.isoformat().startswith("2026-10-30")  # header date kept
    assert rec.date_provenance == "mime_date"  # stamped by the actual source


def test_forged_future_date_header_cannot_shift_provenance(tmp_path: Path) -> None:
    # internalDate present + a forged FUTURE Date header: the record keeps
    # the server date AND stamps internal_date — the header is not even
    # consulted, so it cannot move the record across the split
    eml = _eml("Forged", date="Sun, 30 Oct 2099 10:00:00 +0000")
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aaa", eml=eml, internal_date_ms=1761818400000)]
    )
    store = store_at(tmp_path)
    export_corpus(fake, store, id_to_name=NAMES)
    rec = next(iter(store.iter_records()))
    assert rec.date.isoformat().startswith("2025-10-30")  # server date kept
    assert rec.date_provenance == "internal_date"


def test_export_scope_pins_both_layers(tmp_path: Path) -> None:
    # Two layers (maintainer decision 2026-11-01, ticket #17):
    # QUERY layer — documented tokens only: -in:draft is Google-
    # documented; NO documented chat token exists (is:chat retired;
    # -in:chat undocumented + word-degradation hazard), so chat is NOT in
    # the query. Pinned through the FakeGmailClient's recorded request
    # params as an EXACT token list (a -in:drafts/-in:chats plural
    # variant would silently word-degrade), with every page carrying the
    # same scope.
    # INGEST layer — raw-wire DRAFT/CHAT labelIds rejection, the
    # load-bearing layer (see test_ingest_rejects_draft_and_chat_label_ids).
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="aaa", eml=_eml("A")),
            FakeMessage(id="bbb", eml=_eml("B")),
        ],
        page_size=1,  # two pages: EVERY page must carry the same scope
    )
    export_corpus(fake, store_at(tmp_path), id_to_name=NAMES)
    list_calls = [c for c in fake.calls if c.startswith("list_messages")]
    assert len(list_calls) == 2  # the two-page claim is self-verifying
    queries = {c.removeprefix("list_messages:").rsplit(":", 1)[0] for c in list_calls}
    assert queries == {DEFAULT_QUERY}  # every page carries the same scope
    assert DEFAULT_QUERY.split() == [
        "in:anywhere",
        "-in:spam",
        "-in:trash",
        "-in:draft",
    ]


def test_ingest_rejects_draft_and_chat_label_ids(tmp_path: Path) -> None:
    # ingest layer (load-bearing): RAW wire labelIds exact-id rejection,
    # BEFORE labels.py normalization strips System Labels from the record
    # vocabulary. Skipped messages are COUNTED (no silent vanish), never
    # stored, and are not failures.
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="aaa", eml=_eml("D"), label_ids=("INBOX", "DRAFT")),
            FakeMessage(id="bbb", eml=_eml("C"), label_ids=("CHAT",)),
            FakeMessage(id="ccc", eml=_eml("Ok"), label_ids=("INBOX",)),
        ]
    )
    store = store_at(tmp_path)
    stats = export_corpus(fake, store, id_to_name=NAMES)
    assert [r.id for r in store.iter_records()] == ["ccc"]
    assert stats.skipped_draft_chat == 2
    assert stats.written == 1
    assert not stats.failed  # a policy skip is a decision, not a defect


def test_ingest_rejection_is_id_exact_not_name_match(tmp_path: Path) -> None:
    # a USER label named "chat" (id "L3") and a path "Chat/Work" (id "L4")
    # must NOT be skipped: the check is exact on the raw wire id — never
    # name-matched, never path-matched
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="aaa", eml=_eml("U1"), label_ids=("L3",)),
            FakeMessage(id="bbb", eml=_eml("U2"), label_ids=("L4",)),
        ]
    )
    store = store_at(tmp_path)
    stats = export_corpus(fake, store, id_to_name={"L3": "chat", "L4": "Chat/Work"})
    assert stats.skipped_draft_chat == 0
    assert stats.written == 2
    records = {r.id: r for r in store.iter_records()}
    assert records["aaa"].labels == ("chat",)
    assert records["bbb"].labels == ("Chat/Work",)


def test_export_resume_skips_present(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    store = store_at(tmp_path)
    export_corpus(fake, store, id_to_name=NAMES)
    calls_before = len(fake.calls)
    stats = export_corpus(fake, store, id_to_name=NAMES)
    assert stats.written == 0 and stats.skipped_present == 1
    # skip happened without even fetching the message (id-set cursor)
    fetches = [c for c in fake.calls[calls_before:] if c.startswith("get_message")]
    assert fetches == []


def test_transient_retries_then_succeeds(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    fake.fail("get_message", classify_status(503))
    slept: list[float] = []
    stats = export_corpus(
        fake, store_at(tmp_path), id_to_name=NAMES, sleeper=slept.append
    )
    assert stats.written == 1 and stats.retried == 1 and len(slept) == 1


def test_rate_limit_backoff_honors_retry_after(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    fake.fail("get_message", classify_status(429, retry_after_s=7.0))
    slept: list[float] = []
    stats = export_corpus(
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
    stats = export_corpus(
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
    stats = export_corpus(
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
    stats = export_corpus(
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
    stats = export_corpus(fake, store_at(tmp_path), id_to_name=NAMES)
    assert stats.scanned == 5 and stats.written == 5


def test_page_level_failure_stops_run_loudly(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    fake.fail("list_messages", classify_status(503))
    stats = export_corpus(
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
    stats = export_corpus(
        fake, store, id_to_name=NAMES, max_messages=1, sleeper=lambda s: None
    )
    assert stats.scanned == 1
    ids = [r.id for r in store.iter_records()]
    assert ids == ["19f6abc"]  # ids normalized to lowercase
    # rerun: the UPPERCASE stub id still matches the stored lowercase id
    stats2 = export_corpus(fake, store, id_to_name=NAMES, sleeper=lambda s: None)
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
                date_provenance="internal_date",
            ),
            f"b{i}",
        )
    fake = FakeGmailClient(messages=[FakeMessage(id=mid, eml=_eml()) for mid in ids])
    fresh = CorpusStore(tmp_path / "corpus")
    stats = export_corpus(
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
    export_corpus(fake, store, id_to_name=NAMES)
    # corrupt the index (middle line) so the strict resume gate refuses
    rp = tmp_path / "corpus" / "records.jsonl"
    lines = rp.read_text(encoding="utf-8").splitlines()
    lines[0] = "{broken"
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="corrupt record"):
        export_corpus(fake, store_at(tmp_path), id_to_name=NAMES)


def test_records_round_trip_through_the_store(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml("RT"))])
    store = store_at(tmp_path)
    export_corpus(fake, store, id_to_name=NAMES)
    reopened = CorpusStore(tmp_path / "corpus")
    rec = next(iter(reopened.iter_records()))
    assert isinstance(rec, CorpusRecord)
    assert reopened.get_body("aaa") == "body of RT"


def test_max_messages_zero_is_a_noop(tmp_path: Path) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    stats = export_corpus(fake, store_at(tmp_path), id_to_name=NAMES, max_messages=0)
    assert stats.scanned == stats.written == stats.skipped_present == 0


def test_auth_errors_propagate_typed_not_swallowed(tmp_path: Path) -> None:
    from gmail_labeler.gmail_errors import GmailAuthError

    class Revoked(FakeGmailClient):
        def get_message(self, message_id: str):  # type: ignore[no-untyped-def]
            raise GmailAuthError("revoked", status=401)

    fake = Revoked(messages=[FakeMessage(id="aaa", eml=_eml())])
    with pytest.raises(GmailAuthError):
        export_corpus(
            fake, store_at(tmp_path), id_to_name=NAMES, sleeper=lambda s: None
        )


def test_settings_fetch_propagates_auth_too(tmp_path: Path) -> None:
    from gmail_labeler.export import fetch_settings
    from gmail_labeler.gmail_errors import GmailAuthError

    class DeadAuth(FakeGmailClient):
        def list_labels(self):  # type: ignore[no-untyped-def]
            raise GmailAuthError("consent withdrawn", status=403)

    with pytest.raises(GmailAuthError):
        fetch_settings(DeadAuth(), sleeper=lambda s: None)


def test_snapshot_death_leaves_secret_tight_root(tmp_path: Path) -> None:
    # deepseek MINOR: root created by the snapshot path must be 0700 even
    # when the run dies before the first put()
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.export import write_filters_snapshot

    store = CorpusStore(tmp_path / "deep" / "corpus")
    write_filters_snapshot(store, [])
    assert (tmp_path / "deep" / "corpus").stat().st_mode & 0o777 == 0o700


def test_snapshot_fsyncs_directory_after_replace(tmp_path: Path, monkeypatch) -> None:
    # deepseek MINOR: the rename-durability dir fsync must stay pinned
    import os as _os

    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.export import write_filters_snapshot

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
    from gmail_labeler.export import write_filters_snapshot

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
    from gmail_labeler.export import write_filters_snapshot

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

    from gmail_labeler.corpus_store import CorpusError, CorpusStore
    from gmail_labeler.export import SIDECAR_FILTERS, write_filters_snapshot

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
            date_provenance="internal_date",
        ),
        "whatever",
    )
    fresh = CorpusStore(tmp_path / "corpus")  # fresh resume cache
    stats = export_corpus(fake, fresh, id_to_name=NAMES, sleeper=lambda s: None)
    assert stats.skipped_present == 1
    assert not [c for c in fake.calls if c.startswith("get_message")]


def test_retry_after_hint_ceiling_pinned(tmp_path: Path) -> None:
    # the 60 s CEILING on server hints is a documented decision — pinned
    from gmail_labeler.gmail_errors import GmailRateLimitedError

    slept: list[float] = []
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=_eml())])
    fake.fail("get_message", GmailRateLimitedError("slow down", retry_after_s=300.0))
    export_corpus(
        fake,
        store_at(tmp_path),
        id_to_name=NAMES,
        sleeper=slept.append,
        max_attempts=2,
    )
    assert slept and max(slept) <= 60.0 and 60.0 in slept


def test_max_attempts_below_one_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        export_corpus(
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
        export_corpus(
            fake, store_at(tmp_path), id_to_name=NAMES, sleeper=lambda s: None
        )
    assert marker not in caplog.text


def test_empty_body_is_a_deliberate_record(tmp_path: Path) -> None:
    # deepseek claimed a contradiction with _raw_nonempty; DISPOSED as a
    # false positive: empty BODY text is legitimate attachment-only mail
    # (the wire guard rejects empty RAW — a different thing). Pinned intent.
    eml = b"Subject: x\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\n"
    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=eml)])
    stats = export_corpus(
        fake, store_at(tmp_path), id_to_name=NAMES, sleeper=lambda s: None
    )
    assert stats.written == 1  # label signal survives a body-less mail
