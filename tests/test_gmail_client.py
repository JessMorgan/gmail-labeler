from __future__ import annotations

from typing import Any

import pytest

from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
from gmail_labeler.gmail_client import (
    FilterInfo,
    GmailClient,
    GmailRestClient,
    HistoryPage,
    LabelInfo,
    MessagePage,
    MessageStub,
    RawMessage,
)
from gmail_labeler.gmail_errors import (
    GmailAuthError,
    GmailError,
    GmailPermanentError,
    GmailRateLimitedError,
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

        class _Mid(_Leaf):
            def filters(self) -> _Mid:
                return self

            def settings(self) -> _Mid:
                return self

            def history(self) -> _Mid:
                return self

            def messages(self) -> _Mid:
                return self

        class _Top:
            def labels(self) -> _Mid:
                return _Mid()

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
    assert isinstance(err, GmailRateLimitedError), "export must back off, not abort"


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
    # id, never a per-page watermark — persisting p1's id mid-delta would
    # skip page 2's events (invariant #3); the fake cannot bless that bug
    assert p1.history_id == p2.history_id == 42
    idle = fake.history(42, p2.next_page_token)
    assert idle.records == () and idle.history_id == 42  # real API: current id
    with pytest.raises(GmailPermanentError, match="page token"):
        fake.history(42, "junk")


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


def test_fake_history_cursor_guards() -> None:
    fake = FakeGmailClient(current_history_id=10)
    with pytest.raises(GmailPermanentError, match="not a valid cursor"):
        fake.history(-999)
    with pytest.raises(GmailPermanentError, match="outside the retained") as exc:
        fake.history(11)
    assert exc.value.status == 404  # the resync-shaped status


def test_rest_client_transport_is_timeout_bounded(monkeypatch) -> None:
    import googleapiclient.discovery as discovery

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
    assert seen["http"] is sentinel

    seen.clear()
    GmailRestClient(object())._svc()  # default path
    default_http = seen["http"]
    assert getattr(default_http, "timeout", None) == 30.0


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


# -- write surface (ticket #24): apply_labels / ensure_label -----------------


class _WriteService:
    """Transport shim for the write methods: captures the modify/create
    kwargs (the FULL body as the adapter composed it) and serves canned
    payloads. The chain shape mirrors googleapiclient:
    ``users().messages().modify`` and ``labels().create``."""

    def __init__(
        self,
        *,
        labels_payload: Any = None,
        create_payload: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.labels_payload = labels_payload if labels_payload is not None else {}
        self.create_payload = (
            create_payload
            if create_payload is not None
            else {"id": "Label_new", "name": "x", "type": "user"}
        )
        self.error = error
        self.modify_calls: list[dict[str, object]] = []
        self.create_calls: list[dict[str, object]] = []

    def users(self) -> _WriteService:
        return self

    def messages(self) -> _WriteService:
        return self

    def labels(self) -> _WriteService:
        return self

    def modify(self, **kw: object) -> _Req:
        self.modify_calls.append(kw)
        return _Req({}, self.error)

    def create(self, **kw: object) -> _Req:
        self.create_calls.append(kw)
        return _Req(self.create_payload, self.error)

    def list(self, **kw: object) -> _Req:
        return _Req(self.labels_payload, self.error)


def _write_rest(
    *,
    labels: Any = None,
    create: Any = None,
    error: Exception | None = None,
) -> tuple[GmailRestClient, _WriteService]:
    service = _WriteService(labels_payload=labels, create_payload=create, error=error)
    return (
        GmailRestClient(credentials=object(), build_service=lambda _c: service),
        service,
    )


def test_apply_labels_body_is_add_label_ids_only() -> None:
    # AC2 write-shape pin: the composed wire body is modify with
    # addLabelIds ONLY — the body snapshot is asserted EXACTLY, so any
    # extra key reddens it (mutation pin: a removeLabelIds presence in
    # the body kills this test — invariant #1's structural guard)
    client, service = _write_rest()
    client.apply_labels("abcd", ["Label_1", "ai-high"])
    assert service.modify_calls == [
        {"userId": "me", "id": "abcd", "body": {"addLabelIds": ["Label_1", "ai-high"]}}
    ]


def test_apply_labels_wire_errors_use_the_taxonomy() -> None:
    # the write rides the SAME _call taxonomy as the reads: 5xx is a
    # transient blip, a permission 403 is auth (human-gated) — no new
    # error class is invented for the write path
    client, _ = _write_rest(error=_HttpError(500))
    with pytest.raises(GmailTransientError):
        client.apply_labels("abcd", ["Label_1"])
    client, _ = _write_rest(error=_HttpError(403, reason="permissionDenied"))
    with pytest.raises(GmailAuthError):
        client.apply_labels("abcd", ["Label_1"])


def test_ensure_label_returns_the_existing_user_label_id_without_create() -> None:
    # get-or-create: present means NO create call hits the wire
    # (idempotent startup resolution — the daemon calls this once per
    # configured tier name)
    client, service = _write_rest(
        labels={
            "labels": [
                {"id": "Label_9", "name": "ai-high", "type": "user"},
                {"id": "INBOX", "name": "INBOX", "type": "system"},
            ]
        }
    )
    assert client.ensure_label("ai-high") == "Label_9"
    assert service.create_calls == []


def test_ensure_label_creates_when_absent() -> None:
    client, service = _write_rest(
        labels={"labels": [{"id": "Label_9", "name": "other", "type": "user"}]},
        create={"id": "Label_new", "name": "ai-applied", "type": "user"},
    )
    assert client.ensure_label("ai-applied") == "Label_new"
    assert service.create_calls == [{"userId": "me", "body": {"name": "ai-applied"}}]


def test_ensure_label_refuses_a_system_label_name_collision() -> None:
    # F1 (qwen MINOR): a configured tier name that name-sakes a System
    # Label is REFUSED, named — minting a USER label named "INBOX" would
    # make the tier visually indistinguishable from system state (the
    # plan's id screen cannot catch it: the fresh id is a Label_* id).
    # The remedy is renaming [labels], and it fires at startup, not
    # silently on the wire
    client, service = _write_rest(
        labels={"labels": [{"id": "INBOX", "name": "INBOX", "type": "system"}]},
        create={"id": "Label_new", "name": "INBOX", "type": "user"},
    )
    with pytest.raises(ValueError, match="collides with a System Label name"):
        client.ensure_label("INBOX")
    assert service.create_calls == []  # no duplicate-named user label minted


def test_ensure_label_casefold_catches_lowercase_collision() -> None:
    # F2 (deepseek MINOR): the collision guard casefolds — a configured
    # lowercase variant (high = "inbox") must NOT slip past == and mint
    # the visually-system user label. The Gmail v1 REST reference
    # documents the 409 duplicate-name refusal but NOT a case rule, and
    # CI cannot ask the live API (invariant #9) — casefold is the
    # belt-and-braces posture: an "inbox"-named user label beside the
    # system INBOX is the same visual trap the guard exists to refuse
    client, service = _write_rest(
        labels={"labels": [{"id": "INBOX", "name": "INBOX", "type": "system"}]},
        create={"id": "Label_new", "name": "inbox", "type": "user"},
    )
    with pytest.raises(ValueError, match="collides with a System Label name"):
        client.ensure_label("inbox")
    assert service.create_calls == []


def test_ensure_label_user_hit_wins_regardless_of_list_order() -> None:
    # F2 review MINOR: deterministic precedence — the exact-case user
    # hit is pass 1, the casefolded collision guard is pass 2. The old
    # single loop RAISED when the system entry preceded the user entry
    # in labels.list (same mailbox, order-dependent outcome); an
    # EXISTING exact-case user label is always reused — the guard only
    # prevents MINTING a look-alike
    client, service = _write_rest(
        labels={
            "labels": [
                {"id": "INBOX", "name": "INBOX", "type": "system"},
                {"id": "Label_user", "name": "inbox", "type": "user"},
            ]
        }
    )
    assert client.ensure_label("inbox") == "Label_user"
    assert service.create_calls == []


def test_fake_ensure_label_user_hit_wins_regardless_of_list_order() -> None:
    # the mirrored pin (F2 review NIT): #25's exact-call tests ride the
    # FAKE, so its two-pass split needs its own order-determinism pin —
    # a single-pass revert in the fake must not stay green
    fake = FakeGmailClient(
        labels=[
            LabelInfo(id="INBOX", name="INBOX", type="system"),
            LabelInfo(id="Label_user", name="inbox", type="user"),
        ]
    )
    assert fake.ensure_label("inbox") == "Label_user"
    assert fake.writes == []


def test_ensure_label_case_variant_of_a_user_label_creates_not_refuses() -> None:
    # F2 review NIT: the guard's negative direction — a case variant of
    # a USER label (no system collision) must proceed to create, not
    # refuse: the refusal is for System-Label look-alikes only (what
    # Gmail's own uniqueness rule then answers on the wire is the 409
    # self-heal's story, not the guard's)
    fake = FakeGmailClient(
        labels=[LabelInfo(id="Label_9", name="ai-high", type="user")]
    )
    created = fake.ensure_label("AI-HIGH")
    assert created != "Label_9"
    assert fake.writes == [{"method": "ensure_label", "body": {"name": "AI-HIGH"}}]


def test_ensure_label_409_race_self_heals_by_relisting() -> None:
    # F1 (qwen MINOR): a concurrent create (another client won the name
    # between our list and our create — Gmail answers 409) is NOT a hard
    # failure: re-list ONCE and adopt the winner's id (the get-or-create
    # promise survives the race). If the re-list still finds nothing,
    # the original 409 refusal stands (the name is taken by something we
    # cannot see — no silent retry loop)

    class _RaceService(_WriteService):
        def __init__(self) -> None:
            super().__init__()
            self.lists = 0

        def list(self, **kw: object) -> _Req:
            self.lists += 1
            if self.lists == 1:
                return _Req({"labels": []})
            return _Req(
                {"labels": [{"id": "Label_won", "name": "ai-high", "type": "user"}]}
            )

        def create(self, **kw: object) -> _Req:
            return _Req(None, _HttpError(409, reason="labelName: already exists"))

    service = _RaceService()
    client = GmailRestClient(credentials=object(), build_service=lambda _c: service)
    assert client.ensure_label("ai-high") == "Label_won"

    class _StillAbsentService(_RaceService):
        def list(self, **kw: object) -> _Req:
            self.lists += 1
            return _Req({"labels": []})

    service2 = _StillAbsentService()
    client2 = GmailRestClient(credentials=object(), build_service=lambda _c: service2)
    with pytest.raises(GmailPermanentError, match="409"):
        client2.ensure_label("ai-high")
    assert service2.lists == 2  # exactly ONE re-list — no retry loop


def test_ensure_label_create_response_without_id_is_loud() -> None:
    client, _ = _write_rest(create={"name": "ai-med"})
    with pytest.raises(GmailPermanentError, match="field 'id'"):
        client.ensure_label("ai-med")


def test_apply_labels_rejects_a_bare_string() -> None:
    # F1 (deepseek MINOR): Sequence[str] admits a bare str at the TYPE
    # level (a str is a Sequence of its characters) — the wire would
    # carry one-char label ids. Runtime guard, both surfaces, BEFORE any
    # call/writes land (the REST guard fires before the transport, so
    # the fake raises before even the call-log entry — mirroring
    # "no wire call happened")
    client, service = _write_rest()
    with pytest.raises(ValueError, match="bare string"):
        client.apply_labels("abcd", "Label_1")
    assert service.modify_calls == []
    fake = FakeGmailClient(messages=[FakeMessage(id="aa", eml=b"x")])
    with pytest.raises(ValueError, match="bare string"):
        fake.apply_labels("aa", "Label_1")
    assert fake.calls == []
    assert fake.writes == []


def test_fake_apply_labels_records_snapshot_and_updates_state() -> None:
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aa", eml=b"x", label_ids=("INBOX",))]
    )
    fake.apply_labels("aa", ["Label_1", "ai-high"])
    assert fake.calls == ["apply_labels:aa"]
    # full body snapshot — the wire-shape pin at the fake level: the
    # recorded body carries addLabelIds ONLY (no removeLabelIds key)
    assert fake.writes == [
        {
            "method": "apply_labels",
            "message_id": "aa",
            "body": {"addLabelIds": ["Label_1", "ai-high"]},
        }
    ]
    assert fake.messages["aa"].label_ids == ("INBOX", "Label_1", "ai-high")


def test_fake_apply_labels_dedupes_like_the_wire() -> None:
    # the real API stores a label SET: re-adding a present id is a
    # no-op, not a duplicate — the fake must not lie about it, ACROSS
    # calls AND WITHIN one call (F1 deepseek MINOR: the intra-call
    # duplicate used to land twice)
    fake = FakeGmailClient(
        messages=[FakeMessage(id="aa", eml=b"x", label_ids=("Label_1",))]
    )
    fake.apply_labels("aa", ["Label_1", "ai-high", "ai-high"])
    assert fake.messages["aa"].label_ids == ("Label_1", "ai-high")
    # the body SNAPSHOT keeps the raw caller input (wire evidence — a
    # composition bug that sends duplicates stays VISIBLE here even
    # though the state dedupes like Gmail)
    assert fake.writes[0]["body"] == {"addLabelIds": ["Label_1", "ai-high", "ai-high"]}


def test_fake_apply_labels_unknown_message_is_typed_404() -> None:
    fake = FakeGmailClient()
    with pytest.raises(GmailPermanentError, match="not found"):
        fake.apply_labels("deadbeef", ["Label_1"])
    # the attempt is visible in the call log (the zero-write pin must
    # see it), but nothing landed: no write recorded
    assert fake.calls == ["apply_labels:deadbeef"]
    assert fake.writes == []


def test_fake_ensure_label_is_idempotent_and_creates() -> None:
    fake = FakeGmailClient(
        labels=[LabelInfo(id="Label_9", name="ai-high", type="user")]
    )
    assert fake.ensure_label("ai-high") == "Label_9"
    assert fake.writes == []  # present: no create hit the wire
    created = fake.ensure_label("ai-applied")
    assert created != "Label_9"
    assert fake.writes == [{"method": "ensure_label", "body": {"name": "ai-applied"}}]
    # the created label is now present — the second call is a pure hit
    assert fake.ensure_label("ai-applied") == created
    assert len(fake.writes) == 1


def test_fake_list_labels_returns_a_copy() -> None:
    # F3 (deepseek): list_labels() returned the LIVE internal list —
    # the ctor aliasing trap, OTHER direction: a caller mutating the
    # returned list corrupted the fake's mailbox behind its back (and
    # #25's exact-call tests ride this fake). The REST adapter builds
    # fresh objects per call; the fake now returns a copy
    fake = FakeGmailClient(
        labels=[LabelInfo(id="Label_9", name="ai-high", type="user")]
    )
    returned = fake.list_labels()
    returned.append(LabelInfo(id="Label_x", name="sneaked", type="user"))
    returned[0] = LabelInfo(id="INBOX", name="INBOX", type="system")
    assert fake.labels_list == [LabelInfo(id="Label_9", name="ai-high", type="user")]
    # the mailbox the fake READS from is untouched: ensure_label still
    # hits the real entry, not the corrupted view
    assert fake.ensure_label("ai-high") == "Label_9"


def test_fake_list_filters_returns_a_copy() -> None:
    # F3 review MINOR: the same trap, ONE METHOD BELOW — list_filters
    # handed out the live filters_list (the REST adapter builds fresh
    # FilterInfo per call; the copy rationale holds word-for-word).
    # #25/#26 ride this fake; the read surfaces must be symmetric
    fake = FakeGmailClient(filters=[FilterInfo(id="f1", criteria={"from": "x"})])
    returned = fake.list_filters()
    returned.clear()
    assert fake.filters_list == [FilterInfo(id="f1", criteria={"from": "x"})]


def test_fake_ensure_label_refuses_system_namesakes() -> None:
    # the fake mirrors the REST refusal (F1): no silent duplicate-named
    # user label, in CI or on the wire
    fake = FakeGmailClient(labels=[LabelInfo(id="INBOX", name="INBOX", type="system")])
    with pytest.raises(ValueError, match="collides with a System Label name"):
        fake.ensure_label("INBOX")
    assert fake.writes == []


def test_fake_ensure_label_casefold_catches_lowercase_collision() -> None:
    # the fake mirrors the casefold guard (F1/F2 fidelity)
    fake = FakeGmailClient(labels=[LabelInfo(id="INBOX", name="INBOX", type="system")])
    with pytest.raises(ValueError, match="collides with a System Label name"):
        fake.ensure_label("inbox")
    assert fake.writes == []


def test_fake_ensure_label_logs_the_list_labels_wire_call() -> None:
    # F1 (qwen MINOR): the REST adapter does list_labels + maybe create
    # — the fake records the SAME call sequence so #25's exact-call
    # tests do not diverge between faked and real wiring. list_labels is
    # a READ (READ_SURFACE untouched — the shadow zero-write pin is
    # unaffected)
    fake = FakeGmailClient(
        labels=[LabelInfo(id="Label_9", name="ai-high", type="user")]
    )
    assert fake.ensure_label("ai-high") == "Label_9"
    assert fake.calls == ["ensure_label:ai-high", "list_labels"]


def test_fake_write_failures_are_programmable() -> None:
    fake = FakeGmailClient(messages=[FakeMessage(id="aa", eml=b"x")])
    fake.fail("apply_labels", GmailTransientError("blip"))
    with pytest.raises(GmailTransientError):
        fake.apply_labels("aa", ["Label_1"])
    assert fake.writes == []  # the programmed failure landed BEFORE the record
    fake.apply_labels("aa", ["Label_1"])  # fail-once semantics
    assert len(fake.writes) == 1
    fake.fail("ensure_label", GmailAuthError("revoked"))
    with pytest.raises(GmailAuthError):
        fake.ensure_label("ai-high")
    assert fake.writes == [
        {
            "method": "apply_labels",
            "message_id": "aa",
            "body": {"addLabelIds": ["Label_1"]},
        }
    ]
