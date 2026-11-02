"""Typed Gmail error taxonomy — pure (retry decisions are deterministic, never
string-matched at call sites; CODING-STANDARDS).

transient: retry the same request later (5xx, timeouts, connection drops).
rate_limited: back off (honor retry_after when known) then retry.
permanent: do NOT retry — includes auth (re-auth needed) and policy/shape.

403 is NOT one thing on Gmail: userRateLimitExceeded/rateLimitExceeded/
dailyLimitExceeded/quotaExceeded are backoff-then-retry, while
permissionDenied/insufficient are human-gated. The reason string (from the
HttpError) therefore participates in classification — at this single site.
"""

from __future__ import annotations

import re
from typing import Literal

RETRY = "retry"
BACKOFF = "backoff"
STOP = "stop"


class GmailError(Exception):
    kind: Literal["retry", "backoff", "stop"] = STOP

    def __init__(self, message: str = "", *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class GmailTransientError(GmailError):
    kind = RETRY


class GmailRateLimitedError(GmailError):
    kind = BACKOFF

    def __init__(
        self,
        message: str = "",
        *,
        status: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message, status=status)
        self.retry_after_s = retry_after_s


class GmailPermanentError(GmailError):
    kind = STOP


class GmailAuthError(GmailPermanentError):
    """401 or a non-rate-limit 403: token revoked, consent withdrawn, or
    permission denied — the fix is human, never a retry."""


#: 403 reasons that mean "slow down", not "go away". Matched on NORMALIZED
#: text (lowercase, non-alpha stripped) so both reason CODES and Google's
#: human messages hit, while "insufficientPermissions mentioning a quota
#: project" cannot (a bare "quota" substring false-fires there — verified).
_RATE_LIMIT_MARKERS = ("ratelimit", "limitexceeded", "quotaexceeded")


def _is_rate_limit_reason(reason: str | None) -> bool:
    if not reason:
        return False
    normalized = re.sub(r"[^a-z]", "", reason.lower())
    return any(tok in normalized for tok in _RATE_LIMIT_MARKERS)


def classify_status(
    status: int,
    *,
    retry_after_s: float | None = None,
    reason: str | None = None,
) -> GmailError:
    """Single mapping site from HTTP status (+ Gmail error reason) to the
    taxonomy (pure)."""
    if status == 429 or (status == 403 and _is_rate_limit_reason(reason)):
        return GmailRateLimitedError(
            f"rate limited{f' ({reason})' if reason else ''}",
            status=status,
            retry_after_s=retry_after_s,
        )
    if status in (401, 403):
        return GmailAuthError(
            f"auth rejected{f' ({reason})' if reason else ''} (token revoked / "
            "consent withdrawn / permission denied) — run "
            "`gmail-labeler auth login` or check the GCP project",
            status=status,
        )
    if status in (408, 425):  # Request Timeout / Too Early: transient
        return GmailTransientError(f"timeout ({status})", status=status)
    if status in (400, 404):
        return GmailPermanentError(f"request rejected ({status})", status=status)
    if status == 410:  # Gone: not documented for these endpoints — plain stop
        return GmailPermanentError("resource gone (410)", status=status)
    if 500 <= status <= 599:
        return GmailTransientError(f"server error ({status})", status=status)
    return GmailPermanentError(f"unexpected status {status}", status=status)


class GmailResyncError(GmailPermanentError):
    """history.list rejected the startHistoryId: Gmail returns HTTP 404
    when the cursor falls outside the retained window (official sync
    guide). The only remedy is a full re-sync — never a retry.

    Subclasses GmailPermanentError so natural `except GmailPermanentError`
    call sites catch it; carries status=404 — ENFORCED by the __init__
    default (the docstring is true by construction, not by every caller
    remembering to pass it: a GmailResyncError() with no status would
    otherwise read status=None and slip past the ``is_404`` arm)."""

    def __init__(self, message: str = "", *, status: int = 404) -> None:
        super().__init__(message, status=status)


def is_404(e: GmailError) -> bool:
    """The ONE 404 predicate for the whole taxonomy (ticket #26 F1): a
    message is GONE (deleted mid-window / between fetch and write) or the
    cursor has EXPIRED (outside the retained window) exactly when a
    GmailPermanentError carries status=404. GmailResyncError IS such an
    error (the documented subclass relationship), so the adapter's reading
    and the fake's raw 404 route to the SAME handling. Checked, never
    string-matched (CODING-STANDARDS). Every CONSUMER of a 404 — the
    daemon's history/get/apply arms and the apply path's read/write arms —
    rides THIS predicate; a fourth inlined consumer-copy is the taxonomy
    drift this repo names as its worst bug class. (The adapter's
    ``history()`` wrapper is the 404 PRODUCER — it reads the raw status to
    mint a GmailResyncError, so it is not a consumer that would ride this
    predicate.)"""
    return isinstance(e, GmailPermanentError) and e.status == 404
