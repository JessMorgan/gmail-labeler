from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from gmail_labeler.fake_gmail import (
    FakeGmailClient,
    FakeMessage,
    LabelsAdded,
    LabelsRemoved,
    MessageAdded,
    MessageDeleted,
    history_records,
)
from gmail_labeler.gmail_client import (
    FilterInfo,
    GmailClient,
    GmailRestClient,
    HistoryPage,
    LabelInfo,
    MessageMetadata,
    MessagePage,
    MessageStub,
    ProfileInfo,
    RawMessage,
)
from gmail_labeler.gmail_errors import (
    GmailAuthError,
    GmailError,
    GmailPermanentError,
    GmailRateLimitedError,
    GmailResyncError,
    GmailTransientError,
    classify_status,
)

# Static Protocol conformance: drift between Protocol and fake fails
# typecheck, not test-time.
_CHECK_PROTOCOL: GmailClient = FakeGmailClient()
_CHECK_REST: GmailClient = GmailRestClient(credentials=object())
assert _CHECK_PROTOCOL is not None and _CHECK_REST is not None  # silence F841


class _Req:
    def __init__(self, payload: Any = None, error: Exception | None = None) -> None:
        self.payload, self.error = payload, error

    def execute(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.payload


class _RespShim(dict):
    """httplib2.Response behaves as mapping AND attribute bag."""

    @property
    def status(self) -> int:
        return int(self["status"])

    @property
    def reason(self) -> str:
        return str(self.get("reason", ""))


class _HttpError(Exception):
    def __init__(
        self,
        status: int,
        reason: str | None = None,
        content: bytes | None = None,
        resp: object | None = None,
    ) -> None:
        self.resp: object = (
            resp
            if resp is not None
            else type("R", (), {"status": status, "reason": ""})()
        )
        if reason is not None:
            self.reason = reason
        if content is not None:
            self.content = content


def _rest(payload: Any = None, error: Exception | None = None) -> GmailRestClient:
    """REST client wired to a service returning/raising one canned result."""

    def build(_creds: object) -> Any:
        class _Leaf:
            def list(self, **_kw: object) -> _Req:
                return _Req(payload, error)

            def get(self, **_kw: object) -> _Req:
                return _Req(payload, error)

        # tree mirrors the REAL discovery doc (#59): labels live under
        # users, never at top level — a fake shaped like the bug cannot
        # catch the bug. getProfile is a users-level METHOD (users.getProfile,
        # gmail/v1/users/{userId}/profile), not a sub-resource — #71 grows
        # the leaf exactly where the doc puts it
        class _Mid(_Leaf):
            def labels(self) -> _Mid:
                return self

            def filters(self) -> _Mid:
                return self

            def settings(self) -> _Mid:
                return self

            def history(self) -> _Mid:
                return self

            def messages(self) -> _Mid:
                return self

            def getProfile(self, **_kw: object) -> _Req:
                return _Req(payload, error)

        class _Top:
            def users(self) -> _Mid:
                return _Mid()

        return _Top()

    return GmailRestClient(credentials=object(), build_service=build)


# -- taxonomy (pure) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, GmailRateLimitedError),
        (401, GmailAuthError),
        (403, GmailAuthError),
        (408, GmailTransientError),
        (425, GmailTransientError),
        (400, GmailPermanentError),
        (404, GmailPermanentError),
        (410, GmailPermanentError),
        (500, GmailTransientError),
        (502, GmailTransientError),
        (503, GmailTransientError),
        (418, GmailPermanentError),
    ],
)
def test_status_classification(status: int, expected: type[GmailError]) -> None:
    assert isinstance(classify_status(status), expected)


@pytest.mark.parametrize(
    "reason",
    [
        "userRateLimitExceeded",
        "rateLimitExceeded",
        "dailyLimitExceeded",
        "quotaExceeded",
    ],
)
def test_403_rate_limit_reasons_backoff(reason: str) -> None:
    err = classify_status(403, reason=reason)
    assert isinstance(err, GmailRateLimitedError), "the import must back off, not abort"


@pytest.mark.parametrize("reason", ["permissionDenied", "insufficientPermissions"])
def test_403_permission_reasons_are_human_gated(reason: str) -> None:
    assert isinstance(classify_status(403, reason=reason), GmailAuthError)


def test_kinds_are_stable_strings() -> None:
    assert classify_status(500).kind == "retry"
    assert classify_status(429).kind == "backoff"
    assert classify_status(403).kind == "stop"


def test_rate_limited_carries_retry_after() -> None:
    err = classify_status(429, retry_after_s=30.5)
    assert isinstance(err, GmailRateLimitedError)
    assert err.retry_after_s == 30.5


# -- REST parsing (wire shapes from the discovery schema) ---------------------


def test_rest_labels_parse() -> None:
    client = _rest({"labels": [{"id": "L1", "name": "Shipping", "type": "user"}]})
    got = client.list_labels()
    assert got[0].id == "L1" and got[0].name == "Shipping"


def test_rest_message_page_and_get() -> None:
    client = _rest({"messages": [{"id": "a1", "threadId": "t1"}], "nextPageToken": "p"})
    page = client.list_messages("in:inbox")
    assert page == MessagePage(
        stubs=(MessageStub(id="a1", thread_id="t1"),), next_page_token="p"
    )


def test_rest_empty_next_page_token_means_end() -> None:
    client = _rest({"messages": [{"id": "a1", "threadId": "t1"}], "nextPageToken": ""})
    assert client.list_messages("q").next_page_token is None


def test_rest_internal_date_is_string_on_the_wire() -> None:
    # discovery: int64 fields arrive as STRINGS (deepseek CRITICAL: an
    # isinstance-int check was permanently None in production)
    client = _rest(
        {
            "id": "a1",
            "threadId": "t",
            "labelIds": ["INBOX", "L1"],
            "raw": "abc",
            "internalDate": "1790000000000",
        }
    )
    assert client.get_message("a1").internal_date_ms == 1790000000000


def test_rest_raw_format_may_omit_internal_date() -> None:
    # format=raw documented fields only (qwen MAJOR: presence was faked)
    client = _rest({"id": "a1", "threadId": "t", "labelIds": [], "raw": "abc"})
    assert client.get_message("a1").internal_date_ms is None


def test_rest_missing_raw_is_loud() -> None:
    client = _rest({"id": "a1", "threadId": "t", "labelIds": []})
    with pytest.raises(GmailPermanentError, match="'raw'"):
        client.get_message("a1")


def test_rest_bad_int64_is_loud() -> None:
    client = _rest(
        {"id": "a", "threadId": "t", "labelIds": [], "raw": "x", "internalDate": "x"}
    )
    with pytest.raises(GmailPermanentError, match=r"not an (unsigned )?integer"):
        client.get_message("a")


def test_rest_get_message_int_shape_too() -> None:
    client = _rest(
        {
            "id": "a1",
            "threadId": "t",
            "labelIds": ["INBOX", "L1"],
            "raw": "abc",
            "internalDate": 1790000000000,
        }
    )
    m = client.get_message("a1")
    assert m == RawMessage("a1", "t", ("INBOX", "L1"), 1790000000000, "abc")


def test_rest_history_page_history_id_is_string_on_wire() -> None:
    client = _rest({"historyId": "42", "history": [{"messagesAdded": []}]})
    h = client.history(7)
    assert h.history_id == 42 and len(h.records) == 1


def test_rest_filters_keep_bool_and_int_criteria() -> None:
    # invariant #6: dropping hasAttachment/size would over-match filters
    client = _rest(
        {
            # wire-truth: ListFiltersResponse carries SINGULAR "filter"
            "filter": [
                {
                    "id": "f1",
                    "criteria": {
                        "from": "a@b.c",
                        "query": "ship",
                        "negatedQuery": "unsub",
                        "hasAttachment": True,
                        "excludeChats": True,
                        "size": 1048576,
                    },
                    "action": {"addLabelIds": ["L1"], "removeLabelIds": ["INBOX"]},
                }
            ]
        }
    )
    f = client.list_filters()[0]
    assert f.criteria["hasAttachment"] is True
    assert f.criteria["size"] == 1048576
    assert f.criteria["query"] == "ship"
    assert f.action["addLabelIds"] == ["L1"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [(503, GmailTransientError), (429, GmailRateLimitedError), (403, GmailAuthError)],
)
def test_rest_maps_http_errors_to_taxonomy(status: int, expected: type) -> None:
    client = _rest(error=_HttpError(status))
    with pytest.raises(expected):
        client.list_labels()


def test_rest_403_rate_limit_reason_from_body() -> None:
    err = _HttpError(
        403, content=b'{"error": {"errors": [{"reason": "userRateLimitExceeded"}]}}'
    )
    client = _rest(error=err)
    with pytest.raises(GmailRateLimitedError):
        client.list_labels()


@pytest.mark.parametrize(
    "exc", [TimeoutError("slow"), ConnectionResetError("reset"), ConnectionAbortedError]
)
def test_rest_network_blips_are_transient(exc: Exception) -> None:
    client = _rest(error=exc)
    with pytest.raises(GmailTransientError):
        client.list_labels()


def test_rest_malformed_fields_are_loud_permanent() -> None:
    client = _rest({"labels": "not-a-list"})
    with pytest.raises(GmailPermanentError, match="not a list"):
        client.list_labels()
    client2 = _rest({"labels": [{"id": 5, "name": "x", "type": "user"}]})
    with pytest.raises(GmailPermanentError, match="missing/not-str"):
        client2.list_labels()


def test_rest_non_http_error_is_permanent() -> None:
    client = _rest(error=RuntimeError("programmer error"))
    with pytest.raises(GmailPermanentError, match="unexpected error"):
        client.list_labels()


# -- #71: metadata get, profile, per-stub historyId, label totals -----------
# (Label Sync client spine. Wire facts, verified against the bundled
# discovery doc: ListMessagesResponse has NO page-level historyId — the
# cursor rides PER MESSAGE (Message.historyId, uint64-as-string); Label.
# messagesTotal is int32 (a JSON number, not an int64 string); Profile.
# historyId is uint64-as-string; the metadata format carries no raw/payload
# field at all.)


def test_rest_message_stub_history_id_string_on_wire() -> None:
    client = _rest(
        {
            "messages": [{"id": "a1", "threadId": "t1", "historyId": "77"}],
            "nextPageToken": "p",
        }
    )
    assert client.list_messages("q").stubs[0].history_id == 77


def test_rest_message_stub_history_id_absent_is_none() -> None:
    # absent-tolerant by contract: a schema-legal absent must never become
    # a loud permanent error on an import (the profile fallback covers it)
    client = _rest({"messages": [{"id": "a1", "threadId": "t1"}]})
    assert client.list_messages("q").stubs[0].history_id is None


def test_rest_message_stub_bad_history_id_is_loud() -> None:
    client = _rest({"messages": [{"id": "a1", "threadId": "t1", "historyId": "-1"}]})
    with pytest.raises(GmailPermanentError, match="unsigned"):
        client.list_messages("q")


def test_rest_label_messages_total_int32_on_wire() -> None:
    # int32 arrives as a real JSON number — _opt_int64 tolerates real ints;
    # absent => None (never strict-required); malformed => loud permanent
    client = _rest(
        {"labels": [{"id": "L1", "name": "S", "type": "user", "messagesTotal": 42}]}
    )
    assert client.list_labels()[0].messages_total == 42
    client2 = _rest({"labels": [{"id": "L1", "name": "S", "type": "user"}]})
    assert client2.list_labels()[0].messages_total is None
    client3 = _rest(
        {"labels": [{"id": "L1", "name": "S", "type": "user", "messagesTotal": "x"}]}
    )
    with pytest.raises(GmailPermanentError, match="integer"):
        client3.list_labels()


def test_rest_get_message_metadata_parses() -> None:
    client = _rest(
        {
            "id": "m-17",
            "labelIds": ["INBOX", "L1"],
            "internalDate": "1790000000000",
        }
    )
    assert client.get_message_metadata("m-17") == MessageMetadata(
        id="m-17", label_ids=("INBOX", "L1"), internal_date_ms=1790000000000
    )


def test_rest_get_message_metadata_omitted_internal_date_is_none() -> None:
    # format=metadata commonly reports internalDate alongside id/labelIds,
    # but presence is not contractually pinned upstream (the bundled
    # discovery enum text names id/labels/headers only) — tolerate absence,
    # same discipline as the raw path (never trust presence)
    client = _rest({"id": "m-17", "labelIds": ["L1"]})
    assert client.get_message_metadata("m-17").internal_date_ms is None


def test_rest_get_message_metadata_malformed_is_loud() -> None:
    client = _rest({"labelIds": ["L1"]})  # id missing
    with pytest.raises(GmailPermanentError, match="'id' missing/not-str"):
        client.get_message_metadata("m")
    client2 = _rest({"id": "m", "labelIds": "L1"})  # not a list
    with pytest.raises(GmailPermanentError, match="'labelIds'"):
        client2.get_message_metadata("m")
    client3 = _rest({"id": "m", "labelIds": ["L1"], "internalDate": "x"})
    with pytest.raises(GmailPermanentError, match="integer"):
        client3.get_message_metadata("m")


def test_rest_get_message_metadata_needs_no_raw() -> None:
    # the raw-only helper _raw_nonempty must NOT be on this path: a
    # metadata response without 'raw' parses fine — and the SAME payload
    # is a loud permanent on the raw path, pinning the asymmetry both ways
    client = _rest({"id": "m", "threadId": "t", "labelIds": ["L1"]})
    assert client.get_message_metadata("m").id == "m"
    with pytest.raises(GmailPermanentError, match="'raw'"):
        client.get_message("m")


def test_rest_get_profile_history_id_string_on_wire() -> None:
    client = _rest(
        {
            "emailAddress": "a@b.c",
            "messagesTotal": 9,
            "threadsTotal": 3,
            "historyId": "42",
        }
    )
    assert client.get_profile() == ProfileInfo(history_id=42)


def test_rest_get_profile_history_id_absent_is_none() -> None:
    client = _rest({"emailAddress": "a@b.c", "messagesTotal": 9})
    assert client.get_profile().history_id is None


def test_rest_get_profile_bad_history_id_is_loud() -> None:
    client = _rest({"emailAddress": "a@b.c", "historyId": "+42"})
    with pytest.raises(GmailPermanentError, match="unsigned"):
        client.get_profile()


# -- fake behavior ------------------------------------------------------------


def test_fake_pagination_matches_protocol() -> None:
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id=f"m{i}", eml=b"x", thread_id=f"t{i}") for i in range(5)
        ],
        page_size=2,
    )
    p1 = fake.list_messages("in:inbox")
    p2 = fake.list_messages("in:inbox", p1.next_page_token)
    p3 = fake.list_messages("in:inbox", p2.next_page_token)
    assert [s.id for s in p1.stubs] == ["m0", "m1"]
    assert [s.id for s in p2.stubs] == ["m2", "m3"]
    assert p3.next_page_token is None and len(p3.stubs) == 1


def test_fake_bad_page_token_is_typed_not_valueerror() -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="m", eml=b"x")])
    with pytest.raises(GmailPermanentError, match="page token"):
        fake.list_messages("q", "not-a-token")


def test_fake_unknown_message_is_typed_404() -> None:
    fake = FakeGmailClient()
    with pytest.raises(GmailPermanentError, match="not found"):
        fake.get_message("ghost")


def test_fake_wire_faithful_defaults_and_opt_in_internal_date() -> None:
    plain = FakeGmailClient(
        messages=[FakeMessage(id="m", eml=b"Subject: hi\r\n\r\nbody")]
    )
    assert plain.get_message("m").internal_date_ms is None  # raw-shape truth
    dated = FakeGmailClient(
        messages=[FakeMessage(id="m", eml=b"x", internal_date_ms=1790000000000)]
    )
    assert dated.get_message("m").internal_date_ms == 1790000000000


def test_fake_serves_real_mime_and_per_method_failures() -> None:
    from gmail_labeler.mime_extract import decode_gmail_raw, parse_mime

    fake = FakeGmailClient(
        messages=[FakeMessage(id="m", eml=b"Subject: hi\r\n\r\nbody")]
    )
    fake.fail("list_messages", classify_status(503))
    with pytest.raises(GmailTransientError):
        fake.list_messages("q")
    got = fake.get_message("m")  # unrelated method NOT poisoned by the queue
    parsed = parse_mime(decode_gmail_raw(got.raw_b64url))
    assert parsed.subject == "hi" and parsed.body_text == "body"
    assert fake.calls == ["list_messages:q:", "get_message:m"]


def test_list_messages_label_ids_repeats_on_the_wire_and_refuses_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review minors (both families, commit-A round 1), pinned standalone
    because the shared surface table compares dict(parse_qsl(...)) — which
    collapses a repeated key to its LAST value and structurally cannot
    carry the two-id form. The real adapter must emit labelIds REPEATED
    (discovery: repeated:true), not comma-joined: a comma-joined drift
    would read as one bogus label id upstream and silently return an
    empty inventory."""
    import urllib.parse

    import httplib2
    from google.oauth2.credentials import Credentials
    from googleapiclient.http import HttpRequest

    monkeypatch.setattr(
        httplib2.Http,
        "request",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("offline: no network in tests (invariant #9)")
        ),
    )
    creds = Credentials(
        "fake",
        client_id="cid",
        client_secret="sec",  # fake, offline, never refreshed
        token_uri="https://token.invalid",
    )
    client = GmailRestClient(creds)  # DEFAULT real-build path
    captured: list[HttpRequest] = []
    monkeypatch.setattr(
        GmailRestClient,
        "_call",
        staticmethod(lambda request: captured.append(request) or {"messages": []}),
    )
    client.list_messages("", label_ids=["L1", "L2"])
    assert len(captured) == 1
    _, _, query = captured[0].uri.partition("?")
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    assert [v for k, v in pairs if k == "labelIds"] == ["L1", "L2"]
    assert "q" not in dict(pairs)  # empty query rides absent

    # an EMPTY label_ids list is a caller bug on both spines, never a
    # silent whole-mailbox sweep (doctrine: loud, like _label_list)
    with pytest.raises(ValueError, match="whole mailbox"):
        client.list_messages("", label_ids=[])
    fake = FakeGmailClient()
    with pytest.raises(ValueError, match="whole mailbox"):
        fake.list_messages("", label_ids=[])


def test_fake_list_messages_label_ids_filters_and_logs() -> None:
    """#74 protocol extension: the full-resync inventory scan enumerates a
    label's current members by label ID (ids, not search names — dodges
    slash-path quoting). The fake must filter like the wire (AND across
    ids) and log the ids so call-sequence tests can tell an inventory call
    from a plain list."""
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="m1", eml=b"x", label_ids=("L1", "L2")),
            FakeMessage(id="m2", eml=b"x", label_ids=("L1",)),
            FakeMessage(id="m3", eml=b"x", label_ids=("L2",)),
        ]
    )
    one = fake.list_messages("", label_ids=["L1"])
    assert [s.id for s in one.stubs] == ["m1", "m2"]
    both = fake.list_messages("", label_ids=["L1", "L2"])
    assert [s.id for s in both.stubs] == ["m1"]  # Gmail AND semantics
    # no label_ids => the call log is byte-identical to the landed shape
    # (import call-sequence pins survive the extension)
    plain = fake.list_messages("q")
    assert [s.id for s in plain.stubs] == ["m1", "m2", "m3"]
    assert fake.calls == [
        "list_messages:::L1",
        "list_messages:::L1,L2",
        "list_messages:q:",
    ]


def test_fake_list_messages_label_ids_pages_the_filtered_set() -> None:
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id=f"m{i}", eml=b"x", label_ids=("L1",) if i % 2 else ("L2",))
            for i in range(6)
        ],
        page_size=2,
    )
    p1 = fake.list_messages("", label_ids=["L1"])
    assert [s.id for s in p1.stubs] == ["m1", "m3"]
    assert p1.next_page_token == "2"
    p2 = fake.list_messages("", p1.next_page_token, label_ids=["L1"])
    assert [s.id for s in p2.stubs] == ["m5"] and p2.next_page_token is None


def test_fake_fail_once_then_succeeds() -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="m", eml=b"x")])
    fake.fail("get_message", classify_status(503))
    with pytest.raises(GmailTransientError):
        fake.get_message("m")
    assert fake.get_message("m").id == "m"  # consumed, call now succeeds


def test_fake_filters_are_typed_and_preserved() -> None:
    f = FilterInfo(id="f1", criteria={"hasAttachment": True, "size": 10}, action={})
    fake = FakeGmailClient(filters=[f])
    assert fake.list_filters() == [f]


def test_fake_history_token_chain_and_current_id() -> None:
    fake = FakeGmailClient(
        history_pages=[
            HistoryPage(history_id=5, next_page_token=None, records=({"a": 1},)),
            HistoryPage(history_id=9, next_page_token=None, records=({"b": 2},)),
        ],
        current_history_id=42,
    )
    p1 = fake.history(1)
    p2 = fake.history(1, p1.next_page_token)
    assert [r for p in (p1, p2) for r in p.records] == [{"a": 1}, {"b": 2}]
    # wire truth (discovery doc): EVERY page carries the mailbox's CURRENT
    # id, never a per-page progress id — persisting p1's id mid-delta would
    # skip page 2's events (invariant #3); the fake cannot bless that bug
    assert p1.history_id == p2.history_id == 42
    idle = fake.history(42, p2.next_page_token)
    assert idle.records == () and idle.history_id == 42  # real API: current id
    with pytest.raises(GmailPermanentError, match="page token"):
        fake.history(42, "junk")


def test_fake_get_message_metadata_serves_label_state_without_body() -> None:
    fake = FakeGmailClient(
        messages=[
            FakeMessage(
                id="m",
                eml=b"Subject: hi\r\n\r\nsecret body",
                label_ids=("L1",),
                internal_date_ms=1790000000000,
            )
        ]
    )
    got = fake.get_message_metadata("m")
    assert got == MessageMetadata(
        id="m", label_ids=("L1",), internal_date_ms=1790000000000
    )
    # the body must be nowhere in the returned value (invariant #5)
    assert "secret body" not in repr(got)
    assert fake.calls == ["get_message_metadata:m"]


def test_fake_get_message_metadata_unknown_id_is_typed_404() -> None:
    fake = FakeGmailClient()
    with pytest.raises(GmailPermanentError, match="not found") as exc:
        fake.get_message_metadata("ghost")
    assert exc.value.status == 404


def test_fake_get_profile_serves_current_history_id() -> None:
    # explicit current id (not the history_pages default-derivation) so the
    # assertion isolates "profile echoes the CURRENT id"
    fake = FakeGmailClient(current_history_id=98765)
    assert fake.get_profile() == ProfileInfo(history_id=98765)
    assert fake.calls == ["get_profile"]


def test_fake_new_methods_are_failure_injectable() -> None:
    # _KNOWN_METHODS must accept the new names or the taxonomy is
    # unprovable on their call paths (AC #4)
    fake = FakeGmailClient(messages=[FakeMessage(id="m", eml=b"x")])
    fake.fail("get_message_metadata", classify_status(503))
    with pytest.raises(GmailTransientError):
        fake.get_message_metadata("m")
    assert fake.get_message_metadata("m").id == "m"  # fail-once, then serves
    fake.fail("get_profile", classify_status(403))
    with pytest.raises(GmailAuthError):
        fake.get_profile()
    # the failed calls still ride the log (call-log proofs depend on it)
    assert fake.calls == [
        "get_message_metadata:m",
        "get_message_metadata:m",
        "get_profile",
    ]


# -- #71: fake history-spine parity + deterministic history events -----------


def test_fake_list_messages_stubs_carry_current_history_id() -> None:
    # wire truth: list stubs carry historyId per message; the fake's
    # default reproduces the real shape — PRESENT, at the mailbox's
    # current id (ADR-0008 birth reads the FIRST stub)
    fake = FakeGmailClient(
        messages=[FakeMessage(id="m1", eml=b"x", thread_id="t1")],
        current_history_id=4242,
    )
    assert fake.list_messages("q").stubs[0].history_id == 4242


def test_fake_stub_history_id_per_message_override() -> None:
    # per-message override lets downstream birth tests prove they read the
    # FIRST stub's id, not the last — uniform values would bless that bug
    # (the newest message carries the current id; older mail carries less)
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="m1", eml=b"x", thread_id="t1"),  # newest: current
            FakeMessage(id="m2", eml=b"x", thread_id="t2", history_id=60),
        ],
        current_history_id=100,
    )
    assert [s.history_id for s in fake.list_messages("q").stubs] == [100, 60]


def test_fake_stub_history_id_absence_is_opt_in() -> None:
    # the absent-tolerant quadrant ticket #72's profile-fallback branch
    # must be able to program (the real wire may omit the field)
    fake = FakeGmailClient(
        messages=[FakeMessage(id="m1", eml=b"x")],
        stub_history_id=False,
        current_history_id=7,
    )
    assert fake.list_messages("q").stubs[0].history_id is None


def test_fake_expired_cursor_raises_resync_like_the_adapter() -> None:
    # taxonomy alignment (AC #4): the fake's out-of-window cursor raises
    # the SAME type the real adapter converts a history 404 into —
    # GmailResyncError — so sync code detects resync through ONE type on
    # both spines, and it stays a GmailPermanentError(status=404) for
    # natural handlers
    fake = FakeGmailClient(current_history_id=10)
    with pytest.raises(GmailResyncError) as exc:
        fake.history(11)
    assert isinstance(exc.value, GmailPermanentError)
    assert exc.value.status == 404
    assert "outside the retained" in str(exc.value)


def test_history_records_emit_wire_faithful_events() -> None:
    # shapes verified against the discovery schemas: HistoryMessageAdded/
    # Deleted carry only `message`; HistoryLabelAdded/Removed carry
    # `message` + `labelIds` (list on the wire); the flat `messages[]`
    # field duplicates the typed entries — Google recommends the typed
    # fields, so the generator emits typed only
    records = history_records(
        [
            MessageAdded(message_id="m1", thread_id="t1"),
            LabelsAdded(message_id="m1", label_ids=("L2",)),
            LabelsRemoved(message_id="m1", label_ids=("INBOX",)),
            MessageDeleted(message_id="m2"),
        ]
    )
    assert records == (
        {"id": "1", "messagesAdded": [{"message": {"id": "m1", "threadId": "t1"}}]},
        {"id": "2", "labelsAdded": [{"message": {"id": "m1"}, "labelIds": ["L2"]}]},
        {
            "id": "3",
            "labelsRemoved": [{"message": {"id": "m1"}, "labelIds": ["INBOX"]}],
        },
        {"id": "4", "messagesDeleted": [{"message": {"id": "m2"}}]},
    )


def test_history_records_are_deterministic_with_seeded_ids() -> None:
    # literal expected render (independent of the generator), not a
    # self-equality — determinism means THIS shape, every time
    events = [MessageAdded(message_id="m1", thread_id="t1"), LabelsAdded("m1", ("L1",))]
    assert history_records(events) == (
        {"id": "1", "messagesAdded": [{"message": {"id": "m1", "threadId": "t1"}}]},
        {"id": "2", "labelsAdded": [{"message": {"id": "m1"}, "labelIds": ["L1"]}]},
    )
    ids = [r["id"] for r in history_records(events, start_id=50)]
    assert ids == ["50", "51"]


def test_history_records_reject_loud_inputs() -> None:
    # the generator's doctrine is loud-on-nonsense: a start_id that is not
    # an integer >= 1 is not a wire-valid id (the fake's own history()
    # rejects such cursors), an event outside the union must never render a
    # silent id-only record, and an Any-typed caller's str label_ids must
    # not character-split into a wire lie
    with pytest.raises(ValueError, match="start_id"):
        history_records([], start_id=0)
    with pytest.raises(ValueError, match="start_id"):
        history_records([], start_id=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start_id"):
        # bool passes isinstance(x, int), so this pins the dedicated arm
        history_records([], start_id=True)
    with pytest.raises(TypeError, match="unsupported history event"):
        history_records([{"message_id": "m"}])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="sequence of strings"):
        history_records([LabelsAdded("m1", "INBOX")])  # type: ignore[arg-type]


def test_history_records_feed_the_fakes_history_pages() -> None:
    # the generator's output is exactly what HistoryPage.records carries,
    # so downstream sync tests program deltas declaratively
    fake = FakeGmailClient(
        history_pages=[
            HistoryPage(
                history_id=None,
                next_page_token=None,
                records=history_records([LabelsAdded("m1", ("L2",))]),
            )
        ],
        current_history_id=9,
    )
    page = fake.history(1)
    assert page.records[0]["labelsAdded"] == [
        {"message": {"id": "m1"}, "labelIds": ["L2"]}
    ]


# -- #71 AC3/AC4: call-log & taxonomy proofs (invariant #5 executable) -------


def test_message_metadata_type_has_no_body_bearing_field() -> None:
    # structural proof, not behavioral: the sync direction's message view
    # has NO raw/payload/snippet field to populate — a metadata get is
    # incapable of returning a body even if the wire carried one
    assert {f.name for f in dataclasses.fields(MessageMetadata)} == {
        "id",
        "label_ids",
        "internal_date_ms",
    }


def test_rest_get_message_metadata_never_surfaces_raw_even_if_wire_carries_it() -> None:
    # adversarial canned wire: raw/snippet present (a metadata-format server
    # response never has them — if one ever did, the parse must not carry
    # them out; the not-REQUIRE-ing-them half is pinned by
    # test_rest_get_message_metadata_needs_no_raw)
    client = _rest(
        {
            "id": "m",
            "labelIds": ["L1"],
            "internalDate": "5",
            "raw": "SECRET-RAW",
            "snippet": "SECRET-SNIPPET",
        }
    )
    got = client.get_message_metadata("m")
    assert not isinstance(got, RawMessage)
    assert "SECRET" not in repr(got)


def test_message_get_404_is_plain_permanent_not_resync() -> None:
    # the resync reading belongs to history.list ONLY: a 404 on a message
    # get means the message is gone, and must stay a natural permanent
    client = _rest(error=_HttpError(404))
    with pytest.raises(GmailPermanentError) as exc:
        client.get_message_metadata("m")
    assert not isinstance(exc.value, GmailResyncError)


def test_resync_404_is_one_type_on_both_spines() -> None:
    # AC #4 cross-spine pin: sync code detects the expired cursor through
    # ONE type whether it runs against the fake (function-seam tests) or
    # the real adapter (production)
    client = _rest(error=_HttpError(404))
    with pytest.raises(GmailResyncError) as real:
        client.history(5)
    fake = FakeGmailClient(current_history_id=10)
    with pytest.raises(GmailResyncError) as scripted:
        fake.history(11)
    assert type(real.value) is type(scripted.value)
    assert real.value.status == scripted.value.status == 404


@pytest.mark.parametrize("method", ["get_message_metadata", "get_profile"])
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (503, GmailTransientError),
        (429, GmailRateLimitedError),
        (403, GmailAuthError),
        (400, GmailPermanentError),
    ],
)
def test_taxonomy_holds_on_every_new_real_call_path(
    method: str, status: int, expected: type
) -> None:
    # _call is the single chokepoint — the taxonomy rides the new methods
    # by construction; this is the pin that keeps that construction honest.
    # Dict dispatch (not an if/else catch-all): a method added to the
    # parametrize list without a lambda here fails LOUD on KeyError
    client = _rest(error=_HttpError(status))
    calls = {
        "get_message_metadata": lambda: client.get_message_metadata("m"),
        "get_profile": lambda: client.get_profile(),
    }
    with pytest.raises(expected):
        calls[method]()


@pytest.mark.parametrize("method", ["get_message_metadata", "get_profile"])
def test_taxonomy_holds_on_every_new_fake_call_path(method: str) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="m", eml=b"x")])
    fake.fail(method, classify_status(503))
    calls = {
        "get_message_metadata": lambda: fake.get_message_metadata("m"),
        "get_profile": lambda: fake.get_profile(),
    }
    with pytest.raises(GmailTransientError):
        calls[method]()


def test_sync_relevant_sequence_never_requires_raw_get() -> None:
    # AC3: the whole freshness path — birth cursor from a list stub, event
    # walk, refresh-on-any-touch metadata fetch, a profile read (the
    # ADR-0008 fallback's branch input; the branch itself is #72's), label
    # totals for the estimator — runs on the fake with NO raw get in the
    # call log and no body anywhere near it (invariant #5, executable)
    fake = FakeGmailClient(
        messages=[
            FakeMessage(
                id="m1",
                eml=b"Subject: x\r\n\r\nBODY-SECRET",
                label_ids=("L1",),
                thread_id="t1",
            )
        ],
        labels=[LabelInfo("L1", "Shipping", "user", messages_total=42)],
        history_pages=[
            HistoryPage(
                history_id=None,
                next_page_token=None,
                records=history_records([LabelsAdded("m1", ("L2",))]),
            )
        ],
        current_history_id=100,
    )
    page = fake.list_messages("in:anywhere")
    assert page.stubs[0].history_id == 100  # ADR-0008 birth value
    delta = fake.history(page.stubs[0].history_id)
    assert "m1" in str(delta.records[0])  # the event names the corpus id
    current = fake.get_message_metadata("m1")
    # label state as programmed: the fake's history() SERVES records and
    # never mutates self.messages, so the post-event truth is what the test
    # set on the FakeMessage (see the history() docstring)
    assert current.label_ids == ("L1",)  # body-free
    # get_profile reachable on the sync path; the ADR-0008 FALLBACK branch
    # itself is #72's, programmed via stub_history_id=False
    assert fake.get_profile().history_id == 100
    totals = {la.id: la.messages_total for la in fake.list_labels()}
    assert totals == {"L1": 42}  # estimator basis, no extra call needed
    # the log distinguishes the metadata path from a raw get — and shows
    # the raw path was never taken
    assert fake.calls == [
        "list_messages:in:anywhere:",
        "history:100:",
        "get_message_metadata:m1",
        "get_profile",
        "list_labels",
    ]
    assert [c for c in fake.calls if c.startswith("get_message:")] == []
    assert "BODY-SECRET" not in repr(fake.calls)


# -- wire-truth pins for the reviewer-found production classes ---------------


def test_real_httperror_body_reason_code_wins_over_human_message() -> None:
    # googleapiclient sets HttpError.reason = error.message (prose); the
    # structured errors[].reason code must be what reaches the classifier
    # (deepseek+qwen MAJOR: prose-first shadowed the code in production).
    from googleapiclient.errors import HttpError

    body = (
        b'{"error": {"message": "The user has exceeded a limit", '
        b'"errors": [{"reason": "userRateLimitExceeded"}]}}'
    )
    err = HttpError(
        resp=type("R", (), {"status": 403, "reason": "Forbidden"})(), content=body
    )
    client = _rest(error=err)
    with pytest.raises(GmailRateLimitedError):
        client.list_labels()


def test_human_message_prose_still_classifies_when_code_absent() -> None:
    # normalization must survive spaces/hyphens AND not false-fire on
    # insufficientPermissions mentioning a quota PROJECT (both reproduced)
    from gmail_labeler.gmail_errors import classify_status

    assert classify_status(403, reason="User-rate limit exceeded").kind == "backoff"
    assert classify_status(403, reason="Rate Limit Exceeded").kind == "backoff"
    assert (
        classify_status(
            403, reason="insufficientPermissions: quota project mismatch"
        ).kind
        == "stop"
    )


@pytest.mark.parametrize(
    "exc",
    [
        __import__("ssl").SSLError("handshake fail"),
        __import__("socket").gaierror("dns down"),
        OSError("raw os blip"),
    ],
)
def test_tls_dns_and_bare_os_blips_are_transient(exc: Exception) -> None:
    client = _rest(error=exc)
    with pytest.raises(GmailTransientError):
        client.list_labels()


def test_real_httplib2_errors_are_transient() -> None:
    from httplib2 import ServerNotFoundError  # real package (api dep chain)

    client = _rest(error=ServerNotFoundError("no route"))
    with pytest.raises(GmailTransientError):
        client.list_labels()


def test_lookalike_class_name_stays_programmer_error() -> None:
    # name-based matching would retry this forever (deepseek MINOR)
    class HttpLib2Error(ValueError):  # deliberate collision, non-httplib2 module
        pass

    client = _rest(error=HttpLib2Error("bug, not a blip"))
    with pytest.raises(GmailPermanentError):
        client.list_labels()


def test_reason_falls_back_to_own_reason_when_errors_lacks_one() -> None:
    from gmail_labeler.gmail_client import _error_reason

    err = _HttpError(403)
    err.content = (
        b'{"error": {"reason": "userRateLimitExceeded", '
        b'"errors": [{"message": "no code here"}]}}'
    )
    assert _error_reason(err) == "userRateLimitExceeded"  # type: ignore[union-attr]


def test_fake_history_idle_never_regresses_cursor() -> None:
    fake = FakeGmailClient(
        history_pages=[HistoryPage(history_id=98765, next_page_token=None, records=())]
    )
    fake.history(1)
    idle = fake.history(98765)
    assert idle.history_id is not None and idle.history_id >= 98765


@pytest.mark.parametrize(
    "bad",
    [True, -1],
)
def test_int64_rejects_bool_and_negative(bad: object) -> None:
    client = _rest(
        {"id": "a", "threadId": "t", "labelIds": [], "raw": "x", "internalDate": bad}
    )
    with pytest.raises(GmailPermanentError, match="integer"):
        client.get_message("a")


def test_error_reason_tolerates_non_json_content() -> None:
    from gmail_labeler.gmail_client import _error_reason

    err = _HttpError(403, reason="Forbidden")
    err.content = b"<html>gateway blah</html>"
    assert _error_reason(err) == "Forbidden"  # type: ignore[union-attr]


def test_retry_after_non_numeric_header_tolerated() -> None:
    err = _HttpError(429, resp=_RespShim({"status": 429, "retry-after": "Wed, 21 Oct"}))
    client = _rest(error=err)
    with pytest.raises(GmailRateLimitedError) as ei:
        client.list_labels()
    assert ei.value.retry_after_s is None


def test_signed_or_padded_int64_cursors_are_rejected() -> None:
    # a corrupt historyId must never silently become a plausible cursor
    for bad in (" 42 ", "+42", "-42"):
        client = _rest({"historyId": bad, "history": []})
        with pytest.raises(GmailPermanentError, match="unsigned"):
            client.history(1)


def test_retry_after_header_becomes_backoff_seconds() -> None:
    err = _HttpError(429, resp=_RespShim({"status": 429, "retry-after": "30"}))
    client = _rest(error=err)
    with pytest.raises(GmailRateLimitedError) as ei:
        client.list_labels()
    assert ei.value.retry_after_s == 30.0


def test_gmail_error_passthrough_untouched() -> None:
    queued = GmailRateLimitedError("already classified", retry_after_s=2.0)
    client = _rest(error=queued)
    with pytest.raises(GmailRateLimitedError) as ei:
        client.list_labels()
    assert ei.value is queued


def test_non_object_body_is_loud() -> None:
    client = _rest([1, 2, 3])
    with pytest.raises(GmailPermanentError, match="not an object"):
        client.list_labels()


def test_fake_history_last_page_has_no_phantom_token() -> None:
    # explicit CURRENT id differing from the programmed page id: the echo
    # must win (no vacuous equality against a default)
    fake = FakeGmailClient(
        history_pages=[HistoryPage(history_id=9, next_page_token=None, records=())],
        current_history_id=99,
    )
    p = fake.history(1)
    assert p.next_page_token is None  # last programmed page ENDS
    assert p.history_id == 99  # the CURRENT id echo beats the page id


@pytest.mark.parametrize("token", ["\u0663", "\u00b2"])
def test_fake_rejects_non_ascii_digit_tokens(token: str) -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aa", eml=b"x")])
    with pytest.raises(GmailPermanentError):
        fake.list_messages("q", token)


def test_fake_token_past_end_is_empty_not_error() -> None:
    # a well-formed token beyond the last page is an empty window, exactly
    # like the real API (only MALFORMED tokens are 400-class)
    fake = FakeGmailClient(messages=[FakeMessage(id="aa", eml=b"x")], page_size=1)
    page = fake.list_messages("q", "12")
    assert page.stubs == () and page.next_page_token is None


# -- deepseek review-fold pins ----------------------------------------------


def test_google_auth_errors_are_classified_not_stopped() -> None:
    from google.auth import exceptions as ga

    from gmail_labeler.gmail_client import _classify_google_auth_error

    assert isinstance(
        _classify_google_auth_error(ga.TransportError("socket blast")),
        GmailTransientError,
    )
    revoked = _classify_google_auth_error(
        ga.RefreshError("RefreshError: invalid_grant: Token has been revoked")
    )
    assert isinstance(revoked, GmailAuthError)
    assert "auth login" in str(revoked)
    assert isinstance(
        _classify_google_auth_error(ga.RefreshError("metadata server 503")),
        GmailTransientError,
    )

    # module-pin: a user exception merely NAMED RefreshError is NOT auth-lib
    class RefreshError(Exception):
        pass

    assert _classify_google_auth_error(RefreshError()) is None


def test_http_client_blips_are_transient() -> None:
    from http.client import IncompleteRead

    from gmail_labeler.gmail_client import _is_http_client_error

    assert _is_http_client_error(IncompleteRead(b"", 5))


def test_history_404_surfaces_as_resync_required() -> None:
    from gmail_labeler.gmail_errors import GmailResyncError

    client = _rest(error=_HttpError(404))
    with pytest.raises(GmailResyncError, match="full re-sync"):
        client.history(5)
    # a 403 on the same call is NOT resync: it stays auth
    client = _rest(error=_HttpError(403))
    with pytest.raises(GmailAuthError):
        client.history(5)


def test_nonstring_page_token_is_loud() -> None:
    client = _rest(payload={"messages": [], "nextPageToken": 123})
    with pytest.raises(GmailPermanentError, match="malformed nextPageToken"):
        client.list_messages("in:anywhere")


def test_present_but_empty_raw_is_loud() -> None:
    client = _rest(
        payload={
            "id": "a",
            "threadId": "t",
            "labelIds": [],
            "internalDate": "1",
            "raw": "",
        }
    )
    with pytest.raises(GmailPermanentError, match="raw"):
        client.get_message("a")


def test_fake_fail_typo_raises() -> None:
    fake = FakeGmailClient()
    with pytest.raises(ValueError, match="unknown fake failure target"):
        fake.fail("get_messsage", classify_status(500))


def test_known_failure_targets_cover_the_whole_protocol() -> None:
    # the same drift class as the surface table (#59): a Protocol method
    # missing from _KNOWN_METHODS cannot be failure-tested, and the AC
    # "taxonomy on every call path" would quietly stop covering it
    public = {n for n in dir(GmailClient) if not n.startswith("_")}
    assert public | {"any"} == FakeGmailClient._KNOWN_METHODS


def test_fake_history_cursor_guards() -> None:
    fake = FakeGmailClient(current_history_id=10)
    with pytest.raises(GmailPermanentError, match="not a valid cursor"):
        fake.history(-999)
    # #71 taxonomy alignment: the out-of-window 404 is the SAME typed
    # resync the real adapter raises (see
    # test_fake_expired_cursor_raises_resync_like_the_adapter)
    with pytest.raises(GmailResyncError, match="outside the retained") as exc:
        fake.history(11)
    assert exc.value.status == 404  # the resync-shaped status


def test_rest_client_transport_is_timeout_bounded(monkeypatch) -> None:
    import googleapiclient.discovery as discovery
    import httplib2

    seen: dict[str, object] = {}

    class _Http:  # sentinel transport
        pass

    sentinel = _Http()

    def fake_build(service: str, version: str, **kwargs: object) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(discovery, "build", fake_build)
    client = GmailRestClient(object(), http=sentinel)
    client._svc()
    # seam contract (#56): an injected http is the caller's authorized
    # transport — it rides to build() VERBATIM, never re-wrapped
    assert seen["http"] is sentinel

    seen.clear()
    creds = object()
    GmailRestClient(creds)._svc()  # default path
    default_http = seen["http"]
    assert getattr(default_http, "timeout", None) == 30.0
    # #56: build() forbids http+credentials, so the default transport is
    # the AuthorizedHttp WE compose — the bound rides its INNER httplib2
    # (the getattr above only passes via AuthorizedHttp.timeout's proxy;
    # without the isinstance pin a bare un-wrapped Http would too), and it
    # carries THIS client's credentials (a wrong-auth wiring is the same
    # failure class #56 came from: only caught live)
    from google_auth_httplib2 import AuthorizedHttp

    assert isinstance(default_http, AuthorizedHttp)
    assert default_http.credentials is creds
    assert isinstance(default_http.http, httplib2.Http)
    assert default_http.http.timeout == 30.0


# -- #56: default build path, REAL construction, offline ---------------------


def test_default_build_path_constructs_offline(monkeypatch) -> None:
    """#56 RED pin: build(http=…, credentials=…) is mutually exclusive in
    google-api-python-client 2.x — the FIRST live `dataset import-gmail` run died in
    _svc() with a raw ValueError, bypassing the typed taxonomy. This runs
    the real build with no spy and no injection: static discovery +
    cache_discovery=False keep construction offline, and the patched
    httplib2 transport turns any network attempt into a loud test failure
    (invariant #9)."""
    import httplib2
    from google.oauth2.credentials import Credentials
    from googleapiclient.http import HttpRequest

    def _no_network(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("offline: no network in tests (invariant #9)")

    monkeypatch.setattr(httplib2.Http, "request", _no_network)

    creds = Credentials(
        "fake",
        client_id="cid",
        client_secret="sec",  # fake, offline, never refreshed
        token_uri="https://token.invalid",
    )
    client = GmailRestClient(creds)
    svc = client._svc()
    # request-object construction is pure (discovery-schema bound method):
    # an HttpRequest comes back without a single byte leaving the process
    assert isinstance(svc.users().messages().list(userId="me"), HttpRequest)


# -- #59: EVERY surface path walks the REAL discovery tree, offline ----------

# (call, canned payload, expected URI path, EXACT query-param set, expected
# parsed result); the case label rides pytest.param's id.
# Paths read from the bundled discovery doc (source of truth, offline):
# rootUrl https://gmail.googleapis.com/ + gmail/v1/users/{userId}/... —
# the v1 tree has ONLY `users` at top level; labels = users.labels. The
# param set is EXACT (alt=json is attached by googleapiclient's JSON
# model, not the doc): a superset check would bless a drift that ADDS a
# param, and the no-token cases pin pageToken ABSENCE for free. Each case
# also asserts the canned payload PARSES into the expected value object —
# the table is self-contained, a wire-key drift fails HERE, not only in
# the pre-existing parse tests.
_SURFACE_CASES = [
    pytest.param(
        lambda c: c.list_labels(),
        {
            "labels": [
                {
                    "id": "L1",
                    "name": "Shipping",
                    "type": "user",
                    "messagesTotal": 12,
                }
            ]
        },
        "https://gmail.googleapis.com/gmail/v1/users/me/labels",
        {"alt": "json"},
        [LabelInfo(id="L1", name="Shipping", type="user", messages_total=12)],
        id="list_labels",
    ),
    pytest.param(
        lambda c: c.list_filters(),
        # non-empty values everywhere in this table: empty containers
        # EQUAL the helpers' absent-key defaults, so a key-drift mutant
        # (filter->filters, history->histories, labelIds->label_ids…)
        # would sail through the parse assert un-challenged
        {
            "filter": [
                {
                    "id": "f1",
                    "criteria": {"from": "a@b.c"},
                    "action": {"addLabelIds": ["L1"]},
                }
            ]
        },
        "https://gmail.googleapis.com/gmail/v1/users/me/settings/filters",
        {"alt": "json"},
        [
            FilterInfo(
                id="f1", criteria={"from": "a@b.c"}, action={"addLabelIds": ["L1"]}
            )
        ],
        id="list_filters",
    ),
    pytest.param(
        lambda c: c.list_messages("in:inbox"),
        {
            "messages": [{"id": "a1", "threadId": "t1", "historyId": "77"}],
            "nextPageToken": "n2",
        },
        "https://gmail.googleapis.com/gmail/v1/users/me/messages",
        {"q": "in:inbox", "maxResults": "500", "alt": "json"},
        MessagePage(
            stubs=(MessageStub(id="a1", thread_id="t1", history_id=77),),
            next_page_token="n2",
        ),
        id="list_messages",
    ),
    pytest.param(
        # page_token forwarding is import-termination load-bearing
        lambda c: c.list_messages("in:inbox", "p2"),
        {"messages": [{"id": "a1", "threadId": "t1"}]},
        "https://gmail.googleapis.com/gmail/v1/users/me/messages",
        {"q": "in:inbox", "maxResults": "500", "pageToken": "p2", "alt": "json"},
        MessagePage(
            stubs=(MessageStub(id="a1", thread_id="t1"),), next_page_token=None
        ),
        id="list_messages_page_token",
    ),
    pytest.param(
        # #74 inventory scan: members enumerated by label ID, not by search
        # name (dodges slash-path quoting). The empty query must NOT ride
        # the wire as an empty q= param — the EXACT param set pins its
        # absence; labelIds is `repeated: true` in the discovery doc, so a
        # single id keeps the pin readable (parse_qsl would collapse a
        # repeated key to its last value)
        lambda c: c.list_messages("", None, label_ids=["L9"]),
        {"messages": [{"id": "a1", "threadId": "t1"}]},
        "https://gmail.googleapis.com/gmail/v1/users/me/messages",
        {"labelIds": "L9", "maxResults": "500", "alt": "json"},
        MessagePage(
            stubs=(MessageStub(id="a1", thread_id="t1"),), next_page_token=None
        ),
        id="list_messages_label_ids",
    ),
    pytest.param(
        # distinct id: a path built from the WRONG argument is unforgeable
        lambda c: c.get_message("m-17"),
        {
            "id": "m-17",
            "threadId": "t1",
            "labelIds": ["INBOX", "L1"],
            "internalDate": "1790000000000",
            "raw": "abc",
        },
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m-17",
        {"format": "raw", "alt": "json"},
        RawMessage(
            id="m-17",
            thread_id="t1",
            label_ids=("INBOX", "L1"),
            internal_date_ms=1790000000000,
            raw_b64url="abc",
        ),
        id="get_message",
    ),
    pytest.param(
        lambda c: c.history(7),
        {
            "historyId": "42",
            "history": [{"messagesAdded": []}],
            "nextPageToken": "n2",
        },
        "https://gmail.googleapis.com/gmail/v1/users/me/history",
        {"startHistoryId": "7", "alt": "json"},
        HistoryPage(
            history_id=42, next_page_token="n2", records=({"messagesAdded": []},)
        ),
        id="history",
    ),
    pytest.param(
        lambda c: c.history(7, "hp"),
        {"historyId": "42", "history": []},
        "https://gmail.googleapis.com/gmail/v1/users/me/history",
        {"startHistoryId": "7", "pageToken": "hp", "alt": "json"},
        HistoryPage(history_id=42, next_page_token=None, records=()),
        id="history_page_token",
    ),
    pytest.param(
        # distinct id again: a path built from the WRONG argument is
        # unforgeable; format=metadata is the body-free sync path — the
        # canned payload carries NO raw key (that key is what a raw get
        # would have to have non-empty; a metadata parse must not care)
        lambda c: c.get_message_metadata("m-42"),
        {
            "id": "m-42",
            "labelIds": ["INBOX", "L1"],
            "internalDate": "1790000000000",
        },
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m-42",
        {"format": "metadata", "alt": "json"},
        MessageMetadata(
            id="m-42", label_ids=("INBOX", "L1"), internal_date_ms=1790000000000
        ),
        id="get_message_metadata",
    ),
    pytest.param(
        # discovery: getProfile is a users-level method — path
        # gmail/v1/users/{userId}/profile, params {userId} only (the fake
        # REST tree grows a getProfile leaf under users(), mirroring the
        # REAL doc: a fake shaped like the bug cannot catch the bug, #59)
        lambda c: c.get_profile(),
        {
            "emailAddress": "a@b.c",
            "messagesTotal": 9,
            "threadsTotal": 3,
            "historyId": "42",
        },
        "https://gmail.googleapis.com/gmail/v1/users/me/profile",
        {"alt": "json"},
        ProfileInfo(history_id=42),
        id="get_profile",
    ),
]


@pytest.mark.parametrize(
    ("call", "payload", "expected_path", "expected_params", "expected"),
    _SURFACE_CASES,
)
def test_surface_paths_walk_the_real_discovery_tree(
    monkeypatch: pytest.MonkeyPatch,
    call: Any,
    payload: dict[str, object],
    expected_path: str,
    expected_params: dict[str, str],
    expected: object,
) -> None:
    """#59 RED pin: the FIRST live `dataset import-gmail` after #56 died with
    AttributeError: 'Resource' object has no attribute 'labels' — the gmail
    v1 tree has ONLY `users` at top level (labels = users.labels), the
    build_service fakes mirrored the code's wrong assumption, and #57's
    real-build test constructed only the users().messages().list path.
    Request construction is pure against the static bundled discovery doc
    (only .execute() is live), so capturing the HttpRequest each surface
    method builds pins EVERY path+param drift offline — params are EXACT-
    set equality (a superset check blesses drift that ADDS a param) — and
    each case asserts the canned payload parses to the expected value, so
    wire-key drift fails here too (invariant #9 stays honored — no byte
    leaves the process)."""
    import urllib.parse

    import httplib2
    from google.oauth2.credentials import Credentials
    from googleapiclient.http import HttpRequest

    def _no_network(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("offline: no network in tests (invariant #9)")

    monkeypatch.setattr(httplib2.Http, "request", _no_network)

    creds = Credentials(
        "fake",
        client_id="cid",
        client_secret="sec",  # fake, offline, never refreshed
        token_uri="https://token.invalid",
    )
    client = GmailRestClient(creds)  # DEFAULT real-build path, no injection

    captured: list[object] = []

    def _record(request: object) -> dict[str, object]:
        captured.append(request)
        return payload

    monkeypatch.setattr(GmailRestClient, "_call", staticmethod(_record))
    result = call(client)

    assert len(captured) == 1
    request = captured[0]
    assert isinstance(request, HttpRequest)
    assert request.method == "GET"
    # the uri carries query params — pin the PATH portion exactly against
    # the doc's rootUrl + path pattern
    path, _, query = request.uri.partition("?")
    assert path == expected_path
    assert dict(urllib.parse.parse_qsl(query, keep_blank_values=True)) == (
        expected_params
    )
    # parse assert LAST: a wire-key failure then lands on top of the
    # shape diagnostics instead of masking them
    assert result == expected  # the parse path is part of the pin


def test_surface_cases_cover_the_whole_protocol() -> None:
    # the pin list going stale as the Protocol grows is the SAME drift
    # class #59 came from — a sixth surface method must land here too
    public = {n for n in dir(GmailClient) if not n.startswith("_")}
    ids = set()
    for case in _SURFACE_CASES:
        assert isinstance(case.id, str)  # every param carries an id
        ids.add(case.id)
    # plain ids must equal the Protocol EXACTLY: a variant case is an
    # extra, never a stand-in for its base case (deleting the base would
    # silently drop that case's pageToken-ABSENCE or labelIds-ABSENCE pin).
    # The suffixes are exemption-by-convention: a Protocol method actually
    # named *_page_token / *_label_ids would slip through — contrived
    # today; rename it if it ever ships
    variant_suffixes = ("_page_token", "_label_ids")
    assert {
        i for i in ids if not any(i.endswith(suffix) for suffix in variant_suffixes)
    } == public


def test_build_service_precedence_over_http_seam() -> None:
    # __init__ contract: build_service replaces the WHOLE default build and
    # ignores the http seam (#56: precedence unchanged by the fix)
    seen: list[object] = []
    service = object()

    def build(creds: object) -> Any:
        seen.append(creds)
        return service

    creds = object()
    client = GmailRestClient(creds, build_service=build, http=object())
    assert client._svc() is service
    assert seen == [creds]


def test_filterinfo_isolates_the_source_dict() -> None:
    src: dict[str, object] = {"from": "shipping@shop.co"}
    f = FilterInfo(id="f1", criteria=src, action={})
    src["from"] = "evil@x.z"  # caller mutation after construction
    assert f.criteria["from"] == "shipping@shop.co"  # invariant #6 intact


def test_non_transport_google_auth_errors_are_not_blips() -> None:
    from google.auth import exceptions as ga

    from gmail_labeler.gmail_client import _classify_google_auth_error

    # credential-CONFIG errors must not ride the transient path forever
    assert _classify_google_auth_error(ga.MalformedError("bad creds json")) is None
    assert _classify_google_auth_error(ga.DefaultCredentialsError("no ADC")) is None


def test_refresh_error_oauth_client_codes_are_loud() -> None:
    from google.auth import exceptions as ga

    from gmail_labeler.gmail_client import _classify_google_auth_error

    wrong_client = _classify_google_auth_error(
        ga.RefreshError("('invalid_client', {'error': 'invalid_client'})")
    )
    assert isinstance(wrong_client, GmailAuthError)
    assert "GCP project" in str(wrong_client)


def test_resync_error_is_a_permanent_with_status() -> None:
    from gmail_labeler.gmail_errors import GmailPermanentError, GmailResyncError

    client = _rest(error=_HttpError(404))
    with pytest.raises(GmailResyncError) as exc:
        client.history(5)
    assert isinstance(exc.value, GmailPermanentError)  # natural handler catches
    assert exc.value.status == 404


def test_refresh_error_credential_state_is_auth() -> None:
    # qwen MINOR: a token store missing its refresh token must not retry
    from google.auth import exceptions as ga

    from gmail_labeler.gmail_client import _classify_google_auth_error

    broken_store = _classify_google_auth_error(
        ga.RefreshError(
            "The credentials do not contain the necessary fields need to "
            "refresh the access token."
        )
    )
    assert isinstance(broken_store, GmailAuthError)
