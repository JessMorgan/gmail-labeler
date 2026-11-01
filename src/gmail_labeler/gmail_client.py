"""Gmail API surface — Protocol + REST adapter.

The Phase 0 read surface (labels, filters, message list/get, history) plus
the application path's FIRST write methods (ticket #24): ``apply_labels``
(the ONLY label write the product ever performs — addLabelIds ONLY,
structurally incapable of carrying a removal — invariant #1) and
``ensure_label`` (get-or-create for the configured Tier Label names —
exact-case user hits win, casefolded System Label name-sakes are refused
named — invoked from the daemon/apply startup only, never the Shadow
Pass).
Both live behind the same Protocol (FakeGmailClient is the test spine —
invariant #9). Errors surface via gmail_errors taxonomy only.

Wire-shape rule (verified against the Gmail v1 discovery schema): int64 /
uint64 fields (internalDate, historyId) arrive as STRINGS — googleapiclient
deserializes with bare json.loads, no int coercion — so the adapter parses
them explicitly instead of isinstance-sniffing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .gmail_errors import (
    GmailAuthError,
    GmailError,
    GmailPermanentError,
    GmailResyncError,
    GmailTransientError,
    classify_status,
)

# Adapter bound (CODING-STANDARDS: bounded transports): google_auth_httplib2
# ignores per-request timeouts, so the bound lives on the httplib2 transport.
_REQUEST_TIMEOUT_S = 30.0

#: The bare-str/bare-bytes refusal for apply_labels — Sequence[str]
#: admits a str at the type level (a str is a Sequence of its
#: characters, the wire would carry one-char label ids). ONE message
#: for both surfaces (F2: FakeGmailClient imports this, never a copy).
BARE_ADD_IDS_REFUSAL = (
    "apply_labels: add_label_ids must be a sequence of label ids, not a bare string"
)


@dataclass(frozen=True, slots=True)
class LabelInfo:
    id: str
    name: str
    type: str  # "system" | "user" (labels.list; System Label filtering is
    # labels.py's job — this carries the raw facts)


@dataclass(frozen=True, slots=True)
class FilterInfo:
    id: str
    # FilterCriteria mixes str (from/to/query/negatedQuery), bool
    # (hasAttachment/excludeChats) and int (size); FilterAction mixes
    # list[str] (add/removeLabelIds), str (forward) and bool. Everything is
    # PRESERVED — dropping a field would corrupt filter attribution
    # (invariant #6).
    criteria: dict[str, object] = field(default_factory=dict)
    action: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # frozen is shallow; copy at construction so the wire-parse source
        # dicts can no longer alias into the value (invariant #6 data must
        # stay intact). Nested containers inside remain shared — consumers
        # MUST NOT mutate them (documented, not runtime-enforced).
        object.__setattr__(self, "criteria", dict(self.criteria))
        object.__setattr__(self, "action", dict(self.action))


@dataclass(frozen=True, slots=True)
class MessageStub:
    id: str
    thread_id: str


@dataclass(frozen=True, slots=True)
class RawMessage:
    id: str
    thread_id: str
    label_ids: tuple[str, ...]
    # Authoritative timestamp when present. Presence is NOT guaranteed on
    # the raw path (Google's format docs emphasize id/threadId/labelIds/raw;
    # whether internalDate rides along is not contractually pinned) — so
    # consumers MUST tolerate absence and fall back to the MIME Date header
    # (export does); never trust presence.
    internal_date_ms: int | None
    raw_b64url: str


@dataclass(frozen=True, slots=True)
class MessagePage:
    stubs: tuple[MessageStub, ...]
    next_page_token: str | None


@dataclass(frozen=True, slots=True)
class HistoryPage:
    """history_id is the MAILBOX'S CURRENT id on EVERY page of a delta
    (wire-verified via the discovery doc), NOT a per-page watermark:
    persist it only after the last page of the delta (invariant #3)."""

    history_id: int | None
    next_page_token: str | None
    records: tuple[dict[str, object], ...] = ()


class GmailClient(Protocol):
    def list_labels(self) -> list[LabelInfo]: ...

    def list_filters(self) -> list[FilterInfo]: ...

    def list_messages(
        self, query: str, page_token: str | None = None
    ) -> MessagePage: ...

    def get_message(self, message_id: str) -> RawMessage: ...

    def history(
        self, start_history_id: int, page_token: str | None = None
    ) -> HistoryPage: ...

    def apply_labels(self, message_id: str, add_label_ids: Sequence[str]) -> None:
        """Add labels to one message — the ONLY label write the product
        performs. The wire body carries addLabelIds ONLY (invariant #1:
        structurally incapable of carrying a removal); the caller's
        no-call-on-empty rule keeps an empty plan off the wire."""
        ...

    def ensure_label(self, name: str) -> str:
        """Get-or-create a USER label by exact name (daemon/apply
        startup only — never the Shadow Pass). Idempotent when present;
        a System Label name collision is refused, named; a concurrent-
        create 409 self-heals by re-listing once."""
        ...


class GmailRestClient:
    """googleapiclient-backed adapter. The transport is injected
    (build_service) so every response-shape parse is testable offline; no
    live calls in CI (invariant #9)."""

    def __init__(
        self,
        credentials: object,
        *,
        build_service: Callable[[object], Any] | None = None,
        http: object | None = None,
    ) -> None:
        self._credentials = credentials
        self._service: Any | None = None
        self._build_service = build_service
        self._http = http  # injection seam; default is timeout-bounded.
        # Precedence: build_service (if given) replaces the whole default
        # build and therefore ignores http.

    # -- plumbing ---------------------------------------------------------

    def _svc(self) -> Any:  # transport boundary: shape unknown by nature
        if self._service is None:
            if self._build_service is None:
                # CODING-STANDARDS: adapters are bounded. google_auth_httplib2
                # ignores per-request timeouts, so the bound goes on the
                # httplib2 transport itself — a half-open TLS read fails as
                # a socket.timeout (transient) instead of wedging the poll.
                import httplib2
                from googleapiclient.discovery import build

                http = self._http or httplib2.Http(timeout=_REQUEST_TIMEOUT_S)
                self._service = build(
                    "gmail",
                    "v1",
                    credentials=self._credentials,
                    cache_discovery=False,
                    http=http,
                )
            else:
                self._service = self._build_service(self._credentials)
        return self._service

    @staticmethod
    def _call(request: object) -> dict[str, object]:
        try:
            result = request.execute()  # type: ignore[attr-defined]
        except GmailError:
            raise
        except (TimeoutError, ConnectionError, OSError) as e:
            # socket.timeout/ConnectionReset/ssl.SSLError/socket.gaierror
            # (an OSError): THE common failure class for a polling daemon —
            # transient, per the taxonomy contract.
            raise GmailTransientError(f"network error ({type(e).__name__})") from e
        except Exception as e:
            if _is_httplib2_error(e) or _is_http_client_error(e):
                # httplib2 failures (ServerNotFoundError etc.) and raw
                # http.client blips (IncompleteRead/BadStatusLine) are
                # THE common poll-loop failure class: transient.
                raise GmailTransientError(
                    f"transport error ({type(e).__name__})"
                ) from e
            auth_err = _classify_google_auth_error(e)
            if auth_err is not None:
                raise auth_err from e
            status = getattr(getattr(e, "resp", None), "status", None)
            if isinstance(status, int):
                retry_after = _retry_after(e)
                raise classify_status(
                    status, retry_after_s=retry_after, reason=_error_reason(e)
                ) from e
            raise GmailPermanentError(f"unexpected error: {type(e).__name__}") from e
        if not isinstance(result, dict):
            raise GmailPermanentError("malformed response (not an object)")
        return result

    # -- surface ----------------------------------------------------------

    def list_labels(self) -> list[LabelInfo]:
        data = self._call(self._svc().labels().list(userId="me"))
        return [
            LabelInfo(id=_s(item, "id"), name=_s(item, "name"), type=_s(item, "type"))
            for item in _obj_list(data, "labels")
        ]

    def list_filters(self) -> list[FilterInfo]:
        data = self._call(self._svc().users().settings().filters().list(userId="me"))
        return [
            FilterInfo(
                id=_s(item, "id"),
                criteria=dict(_obj(item, "criteria")),
                action=dict(_obj(item, "action")),
            )
            # wire field is SINGULAR "filter" (ListFiltersResponse; verified
            # against the REST reference — "filters" silently yields [])
            for item in _obj_list(data, "filter")
        ]

    def list_messages(self, query: str, page_token: str | None = None) -> MessagePage:
        req = (
            self._svc()
            .users()
            .messages()
            .list(userId="me", q=query, maxResults=500, pageToken=page_token)
        )
        data = self._call(req)
        stubs = tuple(
            MessageStub(id=_s(m, "id"), thread_id=_s(m, "threadId"))
            for m in _obj_list(data, "messages")
        )
        return MessagePage(stubs=stubs, next_page_token=_token(data))

    def get_message(self, message_id: str) -> RawMessage:
        data = self._call(
            self._svc().users().messages().get(userId="me", id=message_id, format="raw")
        )
        return RawMessage(
            id=_s(data, "id"),
            thread_id=_s(data, "threadId"),
            label_ids=_str_list(data, "labelIds"),
            internal_date_ms=_opt_int64(data, "internalDate"),
            raw_b64url=_raw_nonempty(data),  # absent OR empty: loud
        )

    def history(
        self, start_history_id: int, page_token: str | None = None
    ) -> HistoryPage:
        try:
            data = self._call(
                self._svc()
                .users()
                .history()
                .list(
                    userId="me", startHistoryId=start_history_id, pageToken=page_token
                )
            )
        except GmailError as e:
            # the sync guide is explicit: an expired/out-of-window
            # startHistoryId arrives as HTTP 404 HERE — the one error whose
            # only remedy is a full re-sync, so it gets its own type
            if getattr(e, "status", None) == 404:
                raise GmailResyncError(
                    "history.list 404: startHistoryId or pageToken no "
                    "longer retained — full re-sync required",
                    status=404,
                ) from e
            raise
        records = data.get("history", [])
        if not isinstance(records, list) or not all(
            isinstance(r, dict) for r in records
        ):
            raise GmailPermanentError("response field 'history' not a list of objects")
        return HistoryPage(
            history_id=_opt_int64(data, "historyId"),
            next_page_token=_token(data),
            records=tuple(records),
        )

    # -- write surface (ticket #24 — the ONLY label write is addLabelIds) --

    def apply_labels(self, message_id: str, add_label_ids: Sequence[str]) -> None:
        """Add labels to one message (GLOSSARY: Application). The body is
        a literal with the ONE key ``addLabelIds`` — structurally
        incapable of carrying a removal (invariant #1: additive-only, no
        code path may remove labels). The composition layer (tickets
        #24/#26) never calls this with an empty list — an empty plan is
        no API call at all (pinned at the plan level, not here). Wire
        errors surface through the SAME taxonomy as every read (the
        ``_call`` path: transient blips retry, auth is human-gated).

        ``Sequence[str]`` admits a bare str at the TYPE level (a str is
        a Sequence of its characters — the wire would carry one-char
        label ids), so the guard is RUNTIME: a str (or bytes) is refused
        as a caller bug before anything reaches the transport."""
        if isinstance(add_label_ids, (str, bytes)):
            raise ValueError(f"{BARE_ADD_IDS_REFUSAL} (got {add_label_ids!r})")
        body: dict[str, object] = {"addLabelIds": list(add_label_ids)}
        self._call(
            self._svc().users().messages().modify(userId="me", id=message_id, body=body)
        )

    def ensure_label(self, name: str) -> str:
        """Get-or-create a USER label by exact name (the daemon/apply
        startup resolves the configured ``[labels] applied/high/med``
        names to Gmail label ids ONCE — pure thereafter; never called
        from the Shadow Pass, which is zero-write). Idempotent when the
        name is present (EXACT-case match, pass 1 — an existing user
        label is always reused): NO create call is made. A System Label
        whose name casefold-matches the requested name is REFUSED,
        named (F1/F2, pass 2): minting a USER label named like a System
        Label would make the tier visually indistinguishable from
        system state, and the plan's id screen cannot catch a fresh
        Label_* id — the operator must rename [labels]. A concurrent
        create (another client won the
        name between our list and our create; Gmail answers 409) is
        self-healed: re-list ONCE and adopt the winner's id — if the
        name is STILL not ours to see, the original refusal stands (no
        retry loop)."""
        labels = self.list_labels()
        wanted = name.casefold()
        for label in labels:  # pass 1: the exact-case get-or-create hit
            if label.name == name and label.type == "user":
                return label.id
        for label in labels:  # pass 2 (F2 review): the collision guard
            # runs AFTER the user-hit pass — an EXISTING exact-case user
            # label is always reused (deterministic regardless of the
            # list order); the guard only prevents MINTING a look-alike
            if label.type == "system" and label.name.casefold() == wanted:
                # CASEFOLDED (F2): the Gmail REST reference documents the
                # 409 duplicate-name refusal but not a case rule, and CI
                # cannot ask the live API — belt-and-braces: the visual
                # trap is the same whichever way Gmail's rule falls (the
                # hit/refuse contract, incl. the exact-case HIT, is the
                # method docstring — no restatement here)
                raise ValueError(
                    f"ensure_label: {name!r} collides with a System Label "
                    "name in the mailbox — rename [labels] applied/high/med "
                    "(System Labels are never write targets)"
                )
        try:
            data = self._call(
                self._svc().labels().create(userId="me", body={"name": name})
            )
        except GmailPermanentError as e:
            if e.status != 409:
                raise
            for label in self.list_labels():  # the ONE re-list
                if label.name == name and label.type == "user":
                    return label.id
            raise  # still absent after the race: the original refusal
        return _s(data, "id")


# -- strict shape helpers (a malformed response is a loud permanent error,
#    never a silent None-propagation) --------------------------------------


def _s(obj: dict[str, object], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise GmailPermanentError(f"response field {key!r} missing/not-str")
    return value


def _str_list(obj: dict[str, object], key: str) -> tuple[str, ...]:
    value = obj.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise GmailPermanentError(f"response field {key!r} not a list of strings")
    return tuple(value)


def _obj(obj: dict[str, object], key: str) -> dict[str, object]:
    value = obj.get(key, {})
    if not isinstance(value, dict):
        raise GmailPermanentError(f"response field {key!r} not an object")
    return value


def _obj_list(obj: dict[str, object], key: str) -> list[dict[str, object]]:
    value = obj.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise GmailPermanentError(f"response field {key!r} not a list of objects")
    return list(value)


def _raw_nonempty(data: dict[str, object]) -> str:
    raw = _s(data, "raw")
    if not raw:
        # present-but-empty decodes to b"" and would store a body-less
        # corpus record silently — reject it at the wire boundary
        raise GmailPermanentError("message raw is present but empty")
    return raw


def _token(obj: dict[str, object]) -> str | None:
    """nextPageToken: absent or empty-string means 'no more' (Google's list
    APIs do emit empty tokens). A NON-STRING token is malformed wire — loud
    permanent, never a silent end-of-list that truncates an export."""
    value = obj.get("nextPageToken")
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    raise GmailPermanentError(f"malformed nextPageToken ({type(value).__name__})")


_UINT_RE = re.compile(r"[0-9]{1,20}")


def _opt_int64(obj: dict[str, object], key: str) -> int | None:
    """int64/uint64 wire fields arrive as strings (discovery schema:
    {"type":"string","format":"int64"}; googleapiclient does a bare
    json.loads). ABSENT => None (format-dependent presence is legitimate);
    PRESENT-and-malformed => loud permanent error."""
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise GmailPermanentError(f"response field {key!r} not an integer")
    if isinstance(value, int):  # tolerate servers that send real ints
        if value < 0:
            raise GmailPermanentError(f"response field {key!r} not an unsigned integer")
        return value
    if isinstance(value, str):
        # STRICT unsigned decimal only: " 42 ", "+42", "-42" are corrupt
        # cursor values (a negative historyId must never silently advance
        # anything), not parseable-but-weird inputs.
        if not _UINT_RE.fullmatch(value):
            raise GmailPermanentError(f"response field {key!r} not an unsigned integer")
        return int(value)
    raise GmailPermanentError(f"response field {key!r} not an integer")


def _is_http_client_error(e: BaseException) -> bool:
    import http.client

    return isinstance(e, http.client.HTTPException)


def _classify_google_auth_error(e: BaseException) -> GmailError | None:
    """google-auth errors surface during Credentials.before_request refresh,
    wrapped by google_auth_httplib2 (TransportError) or raised directly
    (RefreshError). Module-pinned: only google.auth.exceptions counts —
    a user exception merely named 'RefreshError' is not."""
    if not type(e).__module__.startswith("google.auth.exceptions"):
        return None
    name = type(e).__name__
    if name == "TransportError":
        # ALLOWLIST: TransportError is what the refresh round-trip wraps
        # network blips in. Other google.auth classes (MalformedError,
        # DefaultCredentialsError) are credential-CONFIG errors that
        # retrying can never fix — they go to the generic permanent path.
        return GmailTransientError(f"auth transport blip ({name})")
    if name == "RefreshError":
        # google-auth flattens the structured error into the message
        # ("{error}: {error_description}"); substring matching can read a
        # code inside a 5xx response body as permanent (liveness miss,
        # never a correctness one). e.args[0] keeps the structured form if
        # a tighter match is ever worth it.
        msg = str(e).lower()
        if "invalid_grant" in msg or "invalid grant" in msg:
            # the refresh token itself is dead: only a human can fix it
            return GmailAuthError(
                f"refresh token rejected ({type(e).__name__}) — run "
                "`gmail-labeler auth login`"
            )
        # NOTE UPSTREAM SOURCE: 'do not contain the necessary fields' is
        # prose from google.auth.external_account_authorized_user (raise
        # sites ~:303/:356); if upstream rewordings drop this case back to
        # the transient branch, CI stays green while a corrupt token store
        # retries forever — grep that phrase here when bumping google-auth.
        if "do not contain the necessary fields" in msg:
            # token store lacks its refresh token — not a client-config
            # problem: re-authenticate (deepseek r3 NIT: distinct remedy)
            return GmailAuthError(
                "stored credentials cannot refresh (no refresh token in "
                "store) — re-run `gmail-labeler auth login`"
            )
        if any(
            code in msg
            for code in (
                "invalid_client",
                "unauthorized_client",
                "invalid_scope",
                "access_denied",
            )
        ):
            # wrong/deleted client, refused consent: corrected config + a
            # fresh auth login is the remedy — exit-4 territory, not a blip
            return GmailAuthError(
                f"OAuth client config rejected ({type(e).__name__}) — check "
                "the GCP project / credentials file, then `gmail-labeler "
                "auth login`"
            )
        return GmailTransientError(  # e.g. auth-endpoint 5xx: retryable
            f"token refresh blip ({name})"
        )
    return None


def _is_httplib2_error(e: Exception) -> bool:
    """httplib2 failures are blips — identified by DEFINING MODULE, not
    class name: an app exception merely named HttpLib2Error must NOT become
    an infinite retry loop (there, a name match is a programmer bug)."""
    return any(
        getattr(k, "__module__", "").split(".")[0] == "httplib2"
        for k in type(e).__mro__
    )


def _error_reason(e: Exception) -> str | None:
    """Gmail error REASON CODE for classification.

    Priority is machine truth over prose: error_details/body codes first —
    googleapiclient's HttpError.reason is the human ``error.message`` (e.g.
    "User Rate Limit Exceeded"), which normalization handles, but only as
    the LAST hint (verified: trusting .reason first shadowed the code)."""
    details = getattr(e, "error_details", None)
    code = _reason_from(details)
    if code:
        return code
    content = getattr(e, "content", None)
    if isinstance(content, (bytes, str)):
        try:
            parsed: object = json.loads(content)
        except ValueError:  # includes UnicodeDecodeError
            code = None
        else:
            code = _reason_from(
                parsed.get("error") if isinstance(parsed, dict) else None
            )
        if code:
            return code
    reason = getattr(e, "reason", None)
    return reason if isinstance(reason, str) and reason else None


def _reason_from(node: object) -> str | None:
    """First ``reason`` code out of {"errors":[{...}]} / [{...}] shapes."""
    if isinstance(node, dict):
        errors = node.get("errors")
        code = _reason_from(errors[0]) if isinstance(errors, list) and errors else None
        if code:
            return code
        reason = node.get("reason")
        return reason if isinstance(reason, str) and reason else None
    if isinstance(node, list) and node:
        return _reason_from(node[0])
    return None


def _retry_after(e: Exception) -> float | None:
    resp = getattr(e, "resp", None)
    # httplib2.Response subclasses dict; anything else has nothing to give
    value = resp.get("retry-after") if isinstance(resp, dict) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
