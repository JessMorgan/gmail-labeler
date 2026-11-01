"""FakeGmailClient — the test spine (invariant #9: the real API is never hit
in CI). Lives in src/ deliberately: every sub-branch (gmail-import, gyb-import,
analyze, the Label Sync module, and later the daemon) programs it, so one fake
keeps them honest against the same Protocol.

Wire fidelity is the point: a fake that lies about the wire is worse than no
fake (both faked internalDate-as-int and the tests pinned the lie). Defaults
here reproduce the REAL shapes — e.g. format=raw without internalDate, list
stubs WITH historyId — and the richer or rarer shape is opt-in per message or
per fake.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass

from .gmail_client import (
    BARE_ADD_IDS_REFUSAL,
    FilterInfo,
    HistoryPage,
    LabelInfo,
    MessageMetadata,
    MessagePage,
    MessageStub,
    ProfileInfo,
    RawMessage,
)
from .gmail_errors import GmailError, GmailPermanentError, GmailResyncError


@dataclass
class FakeMessage:
    id: str
    eml: bytes  # raw RFC 2822 — parsed exactly like a GYB file by consumers
    label_ids: tuple[str, ...] = ()
    thread_id: str = ""
    # Presence of internalDate on the raw path is not contractually
    # guaranteed upstream: serve exactly what is set. None => the field is
    # omitted (consumer tolerance branch); an int => served (authority
    # branch). Gmail-import tests exercise BOTH.
    internal_date_ms: int | None = None
    # Per-message list-stub historyId override (Message.historyId = the
    # mailbox record that last modified THIS message). None => the fake
    # stamps the mailbox's current id (the real shape for the newest mail);
    # an int => served as-is, letting tests give older stubs smaller ids so
    # a birth that reads the WRONG stub (last, not first) cannot hide.
    history_id: int | None = None


# -- deterministic history-event generator (Label Sync test spine, #71) ------


@dataclass(frozen=True, slots=True)
class MessageAdded:
    message_id: str
    # required, not defaulted: real messagesAdded records always carry the
    # threadId, and an "" default would fabricate a value that survives any
    # downstream presence check (a wire lie is worse than no fake)
    thread_id: str


@dataclass(frozen=True, slots=True)
class LabelsAdded:
    message_id: str
    label_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LabelsRemoved:
    message_id: str
    label_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MessageDeleted:
    message_id: str


HistoryEvent = MessageAdded | LabelsAdded | LabelsRemoved | MessageDeleted


def history_records(
    events: Sequence[HistoryEvent], *, start_id: int = 1
) -> tuple[dict[str, object], ...]:
    """Wire-faithful History records for explicit events — deterministic:
    the same event list always yields the same records, ids are sequential
    strings from `start_id` (history ids are uint64-as-string on the wire).

    Shapes verified against the discovery schemas: HistoryMessageAdded/
    Deleted carry only `message` (with threadId for added); HistoryLabel-
    Added/Removed carry `message` + `labelIds` (JSON list). The flat
    `messages[]` field duplicates the change-type entries and Google
    recommends the typed fields — so typed fields only, never the flat
    one. Downstream sync tests program deltas declaratively through this
    and hand the output to HistoryPage(records=...).

    Loud by doctrine: a start_id that is not an integer >= 1 is not a
    wire-valid id, and an event outside the union (a raw dict, or a future
    HistoryEvent member forgotten here) raises instead of silently
    rendering an id-only record that names no change."""
    # bool is a subclass of int, so the negated int check never fires for
    # a bool: isinstance(True, int) is True, and without this arm
    # start_id=True would pass every clause and silently render ids from
    # "1"
    if not isinstance(start_id, int) or isinstance(start_id, bool) or start_id < 1:
        raise ValueError(f"history start_id must be an integer >= 1, got {start_id!r}")
    records: list[dict[str, object]] = []
    for offset, event in enumerate(events):
        record: dict[str, object] = {"id": str(start_id + offset)}
        match event:
            case MessageAdded(message_id=m, thread_id=t):
                record["messagesAdded"] = [{"message": {"id": m, "threadId": t}}]
            case LabelsAdded(message_id=m, label_ids=labels):
                record["labelsAdded"] = [
                    {"message": {"id": m}, "labelIds": _label_list(labels)}
                ]
            case LabelsRemoved(message_id=m, label_ids=labels):
                record["labelsRemoved"] = [
                    {"message": {"id": m}, "labelIds": _label_list(labels)}
                ]
            case MessageDeleted(message_id=m):
                record["messagesDeleted"] = [{"message": {"id": m}}]
            case _:
                # unreachable for a type-checked HistoryEvent, live for
                # Any-typed callers and forgotten union members — the
                # silent id-only record is the wire lie this arm exists to
                # refuse
                raise TypeError(f"unsupported history event: {event!r}")
        records.append(record)
    return tuple(records)


def _label_list(labels: object) -> list[str]:
    # the same loud doctrine at the field level: an Any-typed caller's
    # plain str would list() into character-split labelIds — a wire lie
    # that survives every downstream presence check. The positive str test
    # is load-bearing: str IS a Sequence, so the negative Sequence check
    # below would let it pass, and all() can't catch it either (every
    # character of a string is itself a str)
    if (
        isinstance(labels, str)
        or not isinstance(labels, Sequence)
        or not all(isinstance(x, str) for x in labels)
    ):
        raise TypeError(
            f"history labelIds must be a sequence of strings, got {labels!r}"
        )
    return list(labels)


class FakeGmailClient:
    """Scriptable GmailClient.

    - list_messages page-splits exactly like the real API. page_token is
      OPAQUE to callers (internally an offset string; a bad token raises the
      typed 400-class error the real API produces, not ValueError); stubs
      carry historyId (the real list shape): an explicit FakeMessage.
      history_id wins, else the mailbox's current id — or the field is
      omitted on every stub that carries no explicit value when
      stub_history_id=False. Order is insertion order: model Gmail's
      newest-first list by listing the newest FakeMessage first (ADR-0008
      birth reads the FIRST stub), and give stubs DISTINCT history_ids via
      the per-message field whenever a birth test must discriminate
      first-vs-last — the uniform default hides a last-stub read;
    - query is NOT evaluated — tests size the message set itself (documented
      divergence: query-construction bugs are caught at the live opt-in
      seam, never in CI); labelIds IS evaluated (AND across ids, like the
      wire) because the full-resync inventory scan (#74) enumerates label
      members through it — paging indexes the filtered set;
    - failures are programmed PER METHOD (method name, or "any"); an armed
      "any" fails WHICHEVER method is called next (documented opportunism),
      method queues cannot be stolen by other methods; each error is
      consumed on first use (then that call succeeds — fail-once semantics);
    - history serves an explicit page list honoring page_token, then a
      configurable current id (the real API echoes its CURRENT historyId, it
      does not repeat your start id); an out-of-window cursor raises
      GmailResyncError — the SAME type the real adapter converts the history
      404 into, so resync detection has one type on both spines (#71);
    - get_message_metadata never reads the eml (invariant #5: the sync path
      is structurally body-free), and its call-log entry is distinct from a
      raw get so "no raw fetch" proofs can assert on the log;
    - with ``retention_floor`` set, a start id BELOW the floor answers the
      typed 404 of an out-of-date cursor (the real expiry direction, F1:
      daemon down past retention — the history.list reference documents 404
      when startHistoryId is "invalid or out of date");
    - list_messages serves SCRIPTED order by default; with
      ``documented_order=True`` it sorts by ``internal_date_ms`` DESC,
      modeling the documented reverse-chronological order (every message
      must carry the field; see the method docstring);
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
        stub_history_id: bool = True,
        retention_floor: int | None = None,
        documented_order: bool = False,
    ) -> None:
        self.messages = {m.id: m for m in messages or []}
        self.retention_floor = retention_floor
        if documented_order:
            # model the documented reverse-chronological order (F1):
            # newest internalDate first. A fake that silently sorted
            # None timestamps to the top would be inventing recency
            # — refuse instead.
            missing = sorted(m.id for m in messages or [] if m.internal_date_ms is None)
            if missing:
                raise ValueError(
                    f"documented_order needs internal_date_ms on every "
                    f"message (missing on {missing})"
                )
            self.order = [
                m.id
                for m in sorted(
                    messages or [],
                    key=lambda m: (
                        m.internal_date_ms if m.internal_date_ms is not None else 0
                    ),  # None refused above; explicit guard, not truthiness
                    reverse=True,
                )
            ]
        else:
            self.order = [m.id for m in messages or []]
        # list() copy (F2 review NIT, pre-existing trap): ensure_label
        # CREATES append here — an aliased caller list would mutate a
        # shared #25 fixture behind its back
        self.labels_list = list(labels or [])
        self.filters_list = filters or []
        self.page_size = page_size
        # True (default): list stubs carry historyId, the real shape. False:
        # stubs omit historyId UNLESS a FakeMessage carries an explicit
        # per-message value (specific beats global) — the absent-tolerant
        # quadrant the ADR-0008 profile-fallback branch must be able to
        # program (with no explicit values at all, every stub is None).
        self.stub_history_id = stub_history_id
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
        # the re-walk tracker (#25): which start id owns the current walk
        # and where in the script that walk began
        self._walk_start: int | None = None
        self._walk_start_index = 0
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
            "get_message_metadata",
            "get_profile",
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
        # a COPY (F3): the REST adapter builds fresh objects per call;
        # handing out the live list let a caller corrupt the fake's
        # mailbox behind its back (the ctor aliasing trap, other
        # direction — #25's exact-call tests ride this fake)
        return list(self.labels_list)

    def list_filters(self) -> list[FilterInfo]:
        self.calls.append("list_filters")
        self._maybe_fail("list_filters")
        # a LIST-LEVEL COPY too (F3 review): the read surfaces stay
        # list-level-copy safe — the REST adapter builds fresh
        # FilterInfo per call; #25/#26 ride this fake. The NESTED
        # criteria/action dicts REMAIN SHARED (REST rebuilds them per
        # call; safety rests on the documented nested-mutation ban —
        # FilterInfo.__post_init__)
        return list(self.filters_list)

    def list_messages(
        self,
        query: str,
        page_token: str | None = None,
        *,
        label_ids: Sequence[str] | None = None,
    ) -> MessagePage:
        """The real API's documented order (List Gmail messages guide,
        https://developers.google.com/workspace/gmail/api/guides/list-messages):
        "The returned ``messages`` objects are listed in reverse
        chronological order (newest first)." The fake serves SCRIPTED
        order by default (import fixtures pin it); pass
        ``documented_order=True`` to have the fake APPLY that order
        (sort by internal_date_ms desc) — the #25 bootstrap pins ride
        it to prove the seed anchors on the newest message."""
        # labelIds filter (#74): AND across the listed ids, like the wire.
        # The filter applies BEFORE paging — the token indexes the filtered
        # window, so a paged inventory scan sees exactly the label's
        # members (the fake's token is an offset into that window; the real
        # token is opaque and bound to its originating query — a divergence
        # the fake cannot model and the core never exploits: it always
        # chains its own tokens). The call log carries the ids (comma-
        # joined) only when given, keeping the landed no-labels shape
        # byte-identical for the import-path call-sequence pins. An EMPTY
        # list is refused loud, matching the adapter.
        if label_ids is not None and not label_ids:
            raise ValueError(
                "label_ids=[] would enumerate the whole mailbox — "
                "pass None for no label filter"
            )
        log = f"list_messages:{query}:{page_token or ''}"
        if label_ids:
            log += f":{','.join(label_ids)}"
        self.calls.append(log)
        self._maybe_fail("list_messages")
        order = self.order
        if label_ids:
            wanted = set(label_ids)
            order = [i for i in order if wanted <= set(self.messages[i].label_ids)]
        if page_token is None:
            start = 0
        elif _is_ascii_digit(page_token):
            start = int(page_token)
        else:  # the real API answers 400 on a corrupt token; so do we
            raise GmailPermanentError(f"invalid page token {page_token!r}", status=400)
        window = order[start : start + self.page_size]
        nxt = start + self.page_size
        return MessagePage(
            stubs=tuple(
                MessageStub(
                    id=i,
                    thread_id=self.messages[i].thread_id,
                    history_id=self._stub_history_id(self.messages[i]),
                )
                for i in window
            ),
            next_page_token=str(nxt) if nxt < len(order) else None,
        )

    def _stub_history_id(self, m: FakeMessage) -> int | None:
        if m.history_id is not None:
            return m.history_id
        return self.current_history_id if self.stub_history_id else None

    def get_message(self, message_id: str) -> RawMessage:
        self.calls.append(f"get_message:{message_id}")
        self._maybe_fail("get_message")
        m = self._require_message(message_id)
        return RawMessage(
            id=m.id,
            thread_id=m.thread_id,
            label_ids=m.label_ids,
            internal_date_ms=m.internal_date_ms,  # None = wire-faithful raw shape
            raw_b64url=base64.urlsafe_b64encode(m.eml).decode("ascii").rstrip("="),
            history_id=m.history_id,  # None = field omitted (same doctrine)
        )

    def _require_message(self, message_id: str) -> FakeMessage:
        # one 404 shape shared by the raw and metadata gets, so the two
        # cannot drift apart
        try:
            return self.messages[message_id]
        except KeyError as e:
            raise GmailPermanentError(
                f"message {message_id} not found", status=404
            ) from e

    def get_message_metadata(self, message_id: str) -> MessageMetadata:
        # metadata format: id/labelIds/internalDate — the eml is NEVER read
        # (invariant #5: the sync direction is structurally body-free).
        # internalDate is commonly reported on the metadata format, but
        # presence is not contractually pinned upstream — the fake serves
        # exactly what the FakeMessage sets (None => omitted), the same
        # honest rule as the raw path.
        self.calls.append(f"get_message_metadata:{message_id}")
        self._maybe_fail("get_message_metadata")
        m = self._require_message(message_id)
        return MessageMetadata(
            id=m.id,
            label_ids=m.label_ids,
            internal_date_ms=m.internal_date_ms,
        )

    def get_profile(self) -> ProfileInfo:
        self.calls.append("get_profile")
        self._maybe_fail("get_profile")
        # the mailbox's CURRENT history id — same state the history() echo
        # uses, so the ADR-0008 birth fallback needs no new fake state
        return ProfileInfo(history_id=self.current_history_id)

    def history(
        self, start_history_id: int, page_token: str | None = None
    ) -> HistoryPage:
        """Programmed RESPONSE SEQUENCE for consecutive calls (each poll is
        one entry; multi-page polls carry their own next_page_token).

        A passed page_token must match what the previous response promised
        (a skip/replay mismatch is a typed error — invariant #3 tests drive
        exactly this). Once the sequence is exhausted, the real API returns
        an empty delta carrying its CURRENT historyId (cursor advance), and
        so do we — never an echo of the caller's start id.

        Records are SERVED, never applied: history() does not mutate
        self.messages, so the mailbox truth a metadata get must return is
        whatever the test set on the FakeMessages — sync tests program the
        post-event label state directly, not via event replay.
        RE-WALK (crash-recovery replay, #25): a token-less call is a
        walk — with the SAME start id as the previous walk it REPLAYS
        from that walk's first page (the real API is stateless per
        request: polling again from a HELD cursor re-serves the retained
        window, ADR-0003's "the next poll replays the gap"); with a
        DIFFERENT start id it is a FRESH poll from a different cursor
        (a re-bootstrap) and the script continues from its current
        position. The typed 400 guard stays for every TOKEN mismatch
        (a stale or wrong pageToken, a chain resumed under a different
        token) — the skip/replay catch the guard exists for."""
        self.calls.append(f"history:{start_history_id}:{page_token or ''}")
        self._maybe_fail("history")
        if start_history_id < 1:
            raise GmailPermanentError(
                f"startHistoryId {start_history_id} is not a valid cursor",
                status=400,
            )
        if start_history_id > self.current_history_id:
            # a cursor ahead of the mailbox = outside the retained window:
            # the fake raises the SAME type the real adapter converts the
            # history 404 into (GmailResyncError, a GmailPermanentError
            # carrying status=404) — resync detection has ONE type on both
            # spines (#71 taxonomy alignment, AC #4)
            raise GmailResyncError(
                f"startHistoryId {start_history_id} outside the retained "
                f"window (current: {self.current_history_id})",
                status=404,
            )
        if self.retention_floor is not None and start_history_id < self.retention_floor:
            # THE REAL EXPIRY DIRECTION (F1, both families): the daemon was
            # down past Gmail's retention and the cursor is BEHIND the
            # window. Wire basis: the history.list reference documents the
            # 404 for a startHistoryId that is "invalid or out of date"
            # (https://developers.google.com/workspace/gmail/api/reference/
            # rest/v1/users.history/list) — the out-of-date half is
            # exactly this arm
            raise GmailPermanentError(
                f"startHistoryId {start_history_id} is invalid or out of "
                f"date (retention floor: {self.retention_floor})",
                status=404,
            )
        if page_token is None:
            if (
                start_history_id == self._walk_start
                and self._history_index > self._walk_start_index
            ):
                # re-walk from the held cursor: the window re-serves from
                # the walk's first page (see the docstring)
                self._history_index = self._walk_start_index
                self._history_expected = None
        elif page_token != self._history_expected:
            raise GmailPermanentError(
                f"history page token {page_token!r} != expected "
                f"{self._history_expected!r} (skip/replay misuse)",
                status=400,
            )
        if page_token is None:
            self._walk_start = start_history_id
            self._walk_start_index = self._history_index
        if self._history_index < len(self._history_pages):
            page = self._history_pages[self._history_index]
            nxt = page.next_page_token
            if nxt is None and self._history_index + 1 < len(self._history_pages):
                nxt = str(self._history_index + 1)  # chain while pages remain
            self._history_index += 1
            self._history_expected = nxt
            # wire truth: every page echoes the MAILBOX'S CURRENT id, never
            # a per-page progress id — a test must not be able to bless the
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
            raise ValueError(f"{BARE_ADD_IDS_REFUSAL} (got {add_label_ids!r})")
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
        present — exact-case match, no create recorded). A System
        Label name-sake (casefolded) is REFUSED, named (the REST
        mirror, F1/F2) — unless a USER label of exactly that name
        already exists: it is used as-is (the refusal guards minting
        only). The fake performs the same WIRE calls as the adapter —
        list_labels first (recorded in the call log; a READ, so the
        shadow zero-write pin is unaffected), create only when the name
        is absent — so #25's
        exact-call tests read one sequence, not two. (The
        ``ensure_label:<name>`` entry itself is this fake's own
        convention — the adapter logs nothing; the fidelity claim is
        about the wire calls.) The 409 concurrent-create self-heal is
        REST-only: the fake's single-threaded list+create has no race
        window (documented divergence)."""
        self.calls.append(f"ensure_label:{name}")
        self._maybe_fail("ensure_label")
        labels = self.list_labels()  # the wire call the adapter makes
        wanted = name.casefold()
        for label in labels:  # pass 1: the exact-case get-or-create hit
            if label.name == name and label.type == "user":
                return label.id
        for label in labels:  # pass 2: the casefolded collision guard
            # AFTER the user-hit pass (deterministic precedence, F2
            # review) — the guard only prevents MINTING a look-alike
            if label.type == "system" and label.name.casefold() == wanted:
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
