"""Shared retry/pacing helper — the established error-taxonomy loop (was
private to gmail_import; generalized for Label Sync, ticket #73).

Semantics (unchanged from gmail_import's ``_retrying``, plus one fix):
RETRY/BACKOFF retry with honored ``retry_after_s`` or exponential pacing
under a 60 s ceiling; exhaustion and non-signal permanents warn and return
``None`` (the caller records the stop). ``GmailAuthError`` re-raises — the
fix is human (CLI exit 4). ``GmailResyncError`` NOW ALSO RE-RAISES: kind
STOP must not reduce "the history window is gone" to a logged warning +
None — silent success over a blind window is exactly the failure Label
Sync forbids (ticket #73 / manager note 3). For Gmail Import the added
re-raise is inert: no import call path produces a resync error (only
``history`` does).

The stats coupling is a duck-typed ``retried`` counter (small Protocol),
so GmailImportStats and SyncStats share the loop without a base class
(the two sync spines share a private base among themselves — it stays
inside ``label_sync`` and never reaches across to the import stats).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from .gmail_errors import BACKOFF, RETRY, GmailAuthError, GmailError, GmailResyncError

log = logging.getLogger("gmail_labeler.retry")

MAX_ATTEMPTS = 5
_BASE_BACKOFF_S = 1.0


class RetryStats(Protocol):
    """Anything with an honest ``retried`` counter (GmailImportStats,
    SyncStats) — the loop's only stats need."""

    retried: int


def retrying[T](
    fn: Callable[[], T],
    *,
    max_attempts: int,
    stats: RetryStats,
    sleeper: Callable[[float], None],
    what: str,
) -> T | None:
    for attempt in range(max_attempts):
        try:
            return fn()
        except GmailError as e:
            if e.kind in (RETRY, BACKOFF):
                stats.retried += 1
                if attempt == max_attempts - 1:
                    log.warning("retry: %s retries exhausted (%s)", what, e.kind)
                    return None
                delay = getattr(e, "retry_after_s", None)
                if delay is None:  # an explicit server 0.0 stays 0.0
                    delay = _BASE_BACKOFF_S * (2**attempt)
                # 60 s CEILING on server hints: a dailyLimitExceeded-scale
                # hint is honest-but-bounded here per message; a global
                # stop-vs-backoff policy is a daemon-phase decision (the
                # ceiling keeps a poisoned hint from sleeping an hour)
                sleeper(min(delay, 60.0))
            else:
                if isinstance(e, (GmailAuthError, GmailResyncError)):
                    # every later call would fail identically (auth), or the
                    # retained window is gone (resync) — neither is a fault
                    # to swallow: surface it typed, never a silent None
                    raise
                log.warning("retry: %s permanent error: %s", what, e)
                return None
    return None
