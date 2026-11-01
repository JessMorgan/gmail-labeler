"""FakeGmailClient — the test spine (invariant #9: the real API is never hit
in CI). Lives in src/ deliberately: every sub-branch (export, gyb-import,
analyze, and later the daemon) programs it, so one fake keeps them honest
against the same Protocol.

Wire fidelity is the point: a fake that lies about the wire is worse than no
fake (both faked internalDate-as-int and the tests pinned the lie). Defaults
here reproduce the REAL shapes — e.g. format=raw without internalDate — and
the richer shape is opt-in per message.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass

from .gmail_client import (
    FilterInfo,
    HistoryPage,
    LabelInfo,
    MessagePage,
    MessageStub,
    RawMessage,
)
from .gmail_errors import GmailError, GmailPermanentError


@dataclass
class FakeMessage:
    id: str
    eml: bytes  # raw RFC 2822 — parsed exactly like a GYB file by consumers
    label_ids: tuple[str, ...] = ()
    thread_id: str = ""
    # Presence of internalDate on the raw path is not contractually
    # guaranteed upstream: serve exactly what is set. None => the field is
    # omitted (consumer tolerance branch); an int => served (authority
    # branch). Export tests exercise BOTH.
    internal_date_ms: int | None = None


class FakeGmailClient:
    """Scriptable GmailClient.

    - list_messages page-splits exactly like the real API. page_token is
      OPAQUE to callers (internally an offset string; a bad token raises the
      typed 400-class error the real API produces, not ValueError);
    - query is NOT evaluated — tests size the message set itself (documented
      divergence: query-construction bugs are caught at the live opt-in
      seam, never in CI);
    - failures are programmed PER METHOD (method name, or "any"); an armed
      "any" fails WHICHEVER method is called next (documented opportunism),
      method queues cannot be stolen by other methods; each error is
      consumed on first use (then that call succeeds — fail-once semantics);
    - history serves an explicit page list honoring page_token, then a
      configurable current id (the real API echoes its CURRENT historyId, it
      does not repeat your start id);
    - call log enables exact-behavior assertions (no silent extra fetches);
    - write calls (ticket #24: apply_labels, ensure_label) append to the
      SAME call log — the shadow zero-write pin (calls ⊆ read surface)
      catches a write from a read-only path by NAME — and successful
      writes are recorded in ``writes`` with the FULL body snapshot, so
      the wire-shape assertions (addLabelIds ONLY) pin the composition,
      not just the call;
    - documented divergence: apply_labels ACCEPTS label ids the mailbox
      never listed (the real messages.modify answers 400 on an unknown
      id). Enforcement here would force every write test to program the
      label list; a stale resolution that the REST path would hard-fail
      stays green in CI — the live opt-in seam is where that class of
      bug dies (the query-not-evaluated precedent above).
    """

    def __init__(
        self,
        *,
        messages: list[FakeMessage] | None = None,
        labels: list[LabelInfo] | None = None,
        filters: list[FilterInfo] | None = None,
        page_size: int = 500,
        history_pages: list[HistoryPage] | None = None,
        current_history_id: int | None = None,
    ) -> None:
        self.messages = {m.id: m for m in messages or []}
        self.order = [m.id for m in messages or []]
        self.labels_list = labels or []
        self.filters_list = filters or []
        self.page_size = page_size
        # pages follow the token chain: index by page_token ("p0","p1",...).
        self._history_pages = list(history_pages or [])
        if current_history_id is None:
            # NEVER regress a cursor: default to the last programmed page's
            # id (a fixed default of 1 would move 98765 -> 1 — something
            # the real API cannot do, and invariant #3 tests must not be
            # able to lie about).
            ids = [
                p.history_id for p in self._history_pages if p.history_id is not None
            ]
            current_history_id = max(ids, default=1)
        self.current_history_id = current_history_id
        self._history_index = 0
        self._history_expected: str | None = None
        self._failures: dict[str, list[GmailError]] = {}
        self.calls: list[str] = []
        # successful write calls with their full body snapshots (the
        # wire-shape evidence for the application path, ticket #24)
        self.writes: list[dict[str, object]] = []
        self._created_label_count = 0

    # -- programming --------------------------------------------------------

    _KNOWN_METHODS = frozenset(
        {
            "list_labels",
            "list_messages",
            "get_message",
            "list_filters",
            "history",
            "apply_labels",
            "ensure_label",
            "any",
        }
    )

    def fail(self, method: str, error: GmailError) -> None:
        """Queue one failure for `method` (or "any"). Consumed on first hit.

        A method-name typo would silently disarm the failure path — the
        exact lie this fake exists to prevent — so unknown names raise."""
        if method not in self._KNOWN_METHODS:
            raise ValueError(
                f"unknown fake failure target {method!r} "
                f"(known: {sorted(self._KNOWN_METHODS)})"
            )
        self._failures.setdefault(method, []).append(error)

    def _maybe_fail(self, method: str) -> None:
        for key in (method, "any"):
            queue = self._failures.get(key)
            if queue:
                raise queue.pop(0)

    # -- Protocol surface ---------------------------------------------------

    def list_labels(self) -> list[LabelInfo]:
        self.calls.append("list_labels")
        self._maybe_fail("list_labels")
        return self.labels_list

    def list_filters(self) -> list[FilterInfo]:
        self.calls.append("list_filters")
        self._maybe_fail("list_filters")
        return self.filters_list

    def list_messages(self, query: str, page_token: str | None = None) -> MessagePage:
        self.calls.append(f"list_messages:{query}:{page_token or ''}")
        self._maybe_fail("list_messages")
        if page_token is None:
            start = 0
        elif _is_ascii_digit(page_token):
            start = int(page_token)
        else:  # the real API answers 400 on a corrupt token; so do we
            raise GmailPermanentError(f"invalid page token {page_token!r}", status=400)
        window = self.order[start : start + self.page_size]
        nxt = start + self.page_size
        return MessagePage(
            stubs=tuple(
                MessageStub(id=i, thread_id=self.messages[i].thread_id) for i in window
            ),
            next_page_token=str(nxt) if nxt < len(self.order) else None,
        )

    def get_message(self, message_id: str) -> RawMessage:
        self.calls.append(f"get_message:{message_id}")
        self._maybe_fail("get_message")
        try:
            m = self.messages[message_id]
        except KeyError as e:
            raise GmailPermanentError(
                f"message {message_id} not found", status=404
            ) from e
        return RawMessage(
            id=m.id,
            thread_id=m.thread_id,
            label_ids=m.label_ids,
            internal_date_ms=m.internal_date_ms,  # None = wire-faithful raw shape
            raw_b64url=base64.urlsafe_b64encode(m.eml).decode("ascii").rstrip("="),
        )

    def history(
        self, start_history_id: int, page_token: str | None = None
    ) -> HistoryPage:
        """Programmed RESPONSE SEQUENCE for consecutive calls (each poll is
        one entry; multi-page polls carry their own next_page_token).

        A passed page_token must match what the previous response promised
        (a skip/replay mismatch is a typed error — invariant #3 tests drive
        exactly this). Once the sequence is exhausted, the real API returns
        an empty delta carrying its CURRENT historyId (cursor advance), and
        so do we — never an echo of the caller's start id."""
        self.calls.append(f"history:{start_history_id}:{page_token or ''}")
        self._maybe_fail("history")
        if start_history_id < 1:
            raise GmailPermanentError(
                f"startHistoryId {start_history_id} is not a valid cursor",
                status=400,
            )
        if start_history_id > self.current_history_id:
            # a cursor ahead of the mailbox = outside the retained window:
            # the fake serves the SAME typed 404 the real sync guide
            # describes (GmailResyncError is the adapter-level reading of it)
            raise GmailPermanentError(
                f"startHistoryId {start_history_id} outside the retained "
                f"window (current: {self.current_history_id})",
                status=404,
            )
        if page_token != self._history_expected:
            raise GmailPermanentError(
                f"history page token {page_token!r} != expected "
                f"{self._history_expected!r} (skip/replay misuse)",
                status=400,
            )
        if self._history_index < len(self._history_pages):
            page = self._history_pages[self._history_index]
            nxt = page.next_page_token
            if nxt is None and self._history_index + 1 < len(self._history_pages):
                nxt = str(self._history_index + 1)  # chain while pages remain
            self._history_index += 1
            self._history_expected = nxt
            # wire truth: every page echoes the MAILBOX'S CURRENT id, never
            # a per-page watermark — a test must not be able to bless the
            # page-1-cursor bug (invariant #3)
            return HistoryPage(
                history_id=self.current_history_id,
                next_page_token=nxt,
                records=tuple(page.records),
            )
        self._history_expected = None
        return HistoryPage(history_id=self.current_history_id, next_page_token=None)

    # -- write surface (ticket #24 — mirrors the REST adapter's behavior) --

    def apply_labels(self, message_id: str, add_label_ids: Sequence[str]) -> None:
        """Add labels to a message (the ONLY label write, invariant #1).
        The recorded body snapshot carries ``addLabelIds`` ONLY — the
        same shape the REST adapter composes; a removal is not even a
        representable argument here. An unknown message id is the typed
        404 the real API answers (the attempt stays visible in the call
        log; nothing lands, so no write is recorded). The bare-str guard
        mirrors the REST adapter (a caller bug raises BEFORE even the
        call-log entry — no wire call happened)."""
        if isinstance(add_label_ids, (str, bytes)):
            raise ValueError(
                "apply_labels: add_label_ids must be a sequence of label "
                f"ids, not a bare string (got {add_label_ids!r})"
            )
        self.calls.append(f"apply_labels:{message_id}")
        self._maybe_fail("apply_labels")
        try:
            message = self.messages[message_id]
        except KeyError as e:
            raise GmailPermanentError(
                f"message {message_id} not found", status=404
            ) from e
        adds = list(add_label_ids)
        # the real API stores a label SET: re-adding a present id is a
        # no-op, not a duplicate — ACROSS calls AND WITHIN one call (the
        # fake must not lie about it). The body SNAPSHOT keeps the raw
        # caller input: wire evidence (a composition bug that sends
        # duplicates stays visible, like the modify body the wire saw)
        present = set(message.label_ids)
        deduped: list[str] = []
        for label_id in adds:
            if label_id not in present and label_id not in deduped:
                deduped.append(label_id)
        message.label_ids = message.label_ids + tuple(deduped)
        self.writes.append(
            {
                "method": "apply_labels",
                "message_id": message_id,
                "body": {"addLabelIds": adds},
            }
        )

    def ensure_label(self, name: str) -> str:
        """Get-or-create a USER label by exact name (idempotent when
        present — no create recorded). A System Label name-sake is
        REFUSED, named (the REST mirror, F1). The fake performs the
        same WIRE calls as the adapter — list_labels first (recorded in
        the call log; a READ, so the shadow zero-write pin is
        unaffected), create only when the name is absent — so #25's
        exact-call tests read one sequence, not two. (The
        ``ensure_label:<name>`` entry itself is this fake's own
        convention — the adapter logs nothing; the fidelity claim is
        about the wire calls.) The 409 concurrent-create self-heal is
        REST-only: the fake's single-threaded list+create has no race
        window (documented divergence)."""
        self.calls.append(f"ensure_label:{name}")
        self._maybe_fail("ensure_label")
        labels = self.list_labels()  # the wire call the adapter makes
        for label in labels:
            if label.name == name:
                if label.type == "user":
                    return label.id
                raise ValueError(
                    f"ensure_label: {name!r} collides with a System Label "
                    "name in the mailbox — rename [labels] applied/high/med "
                    "(System Labels are never write targets)"
                )
        self._created_label_count += 1
        new_id = f"Label_fake_{self._created_label_count}"
        self.labels_list.append(LabelInfo(id=new_id, name=name, type="user"))
        self.writes.append({"method": "ensure_label", "body": {"name": name}})
        return new_id


def _is_ascii_digit(token: str) -> bool:
    # .isdigit() is Unicode-aware ("٣" and "²" would page the window); the
    # fake only ever issues ASCII offset tokens, so everything else is
    # rejected outright — exactly the typed-400 class the real API gives
    # for a token it never handed out.
    return bool(token) and all(c in "0123456789" for c in token)
