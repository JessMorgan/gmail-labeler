"""Shared MIME parsing — the preprocessing-integrity core (invariant #4).

Gmail's API (format=raw) and a GYB backup (``.eml`` files) both yield RFC 2822
bytes. Parsing them through this one function is what guarantees the classifier
sees a message the same way regardless of which corpus source it came from —
the divergence auto-github needed a dedicated gotcha to police is structurally
impossible here because there is no second code path.

Pure core: stdlib ``email`` only, no I/O, DETERMINISTIC, and panic-free on
malformed input — any failure (broken headers, pathological nesting, bad
charsets) degrades to best-effort fields and a content-free log warning; a
crash here would abort the whole import over a 20-byte garbage header.

The ``date`` is the MESSAGE's Date header or None — never "now": this module
is deterministic, and both callers hold an authoritative timestamp (Gmail
``internalDate`` epoch ms, GYB ``message_internaldate``) which belongs on
CorpusRecord. Silently fabricating a date would move messages across the
temporal split (invariant #7) nondeterministically.
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email import message_from_bytes
from email.message import EmailMessage, Message, MIMEPart
from email.policy import default as default_policy
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser

log = logging.getLogger("gmail_labeler.mime")

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\u00a0]+")
_MULTI_NL = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class ParsedMail:
    subject: str
    sender: str
    recipients: tuple[str, ...]  # deduped, order-preserving
    date: datetime | None  # message Date header, tz-aware UTC; None if absent
    body_text: str  # canonicalized: THIS is what gets stored and hashed

    def is_parseable_body(self) -> bool:
        return bool(self.body_text)


def _scrub(text: str) -> str:
    """Replace undecodable/surrogate chars — headers with raw 8-bit bytes
    survive the stdlib parser as surrogate escapes that would detonate later
    at UTF-8 encode time (corpus write); neutralize them HERE, at the seam."""
    try:
        text.encode("utf-8")  # fast path: already clean
        return text
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


class _HtmlText(HTMLParser):
    """Strip HTML to text: drop script/style, keep visible text."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1
        elif tag in ("br", "p", "div", "tr"):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def _canonicalize(text: str) -> str:
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _body_text(msg: EmailMessage) -> str:
    """Non-blank text/plain > stripped text/html > first INLINE part > "".

    A blank/whitespace-only text/plain part does NOT shadow real HTML content
    (the dominant newsletter shape: empty plain alternative + full HTML).
    Every selection failure degrades one step and logs — never raises.
    """
    plain = _content_of(msg, "plain")
    if plain is not None and plain.strip():
        return _canonicalize(plain)
    html = _content_of(msg, "html")
    if html is not None and html.strip():
        return _canonicalize(_strip_html(html))
    raw = _first_inline_text(msg)
    if raw.strip():
        return _canonicalize(raw)
    if plain is None and html is None:
        log.warning("mime: no text part extractable (structure degraded)")
    return _canonicalize(raw)


def _content_of(msg: EmailMessage, subtype: str) -> str | None:
    try:
        part = msg.get_body(preferencelist=(subtype,))
        if part is None or _is_attachmentish(part):
            return None
        content = part.get_content()
        if isinstance(content, bytes):  # pragma: no cover - defensive
            content = content.decode("utf-8", errors="replace")
        return content
    except Exception:
        log.warning("mime: %s part unreadable (charset/transfer degraded)", subtype)
        return None


def _is_attachmentish(part: EmailMessage | MIMEPart) -> bool:
    """Attachment signals per stdlib normalization: disposition enum (covers
    any-case "ATTACHMENT"), a filename/name parameter, or non-text content.
    get_body selects by STRUCTURE alone — name-only attachments (classic
    RFC 2045 mailer output) sail past it, so every selection site needs this
    guard. Accepted trade-off: a genuine body part mislabeled with name=
    degrades to empty (fail-closed) rather than risking attachment contents
    in training data."""
    return (
        part.get_content_disposition() == "attachment"
        or bool((part.get_filename() or "").strip())
        or part.get_content_maintype() != "text"
    )


def _leaf_text_parts(
    part: EmailMessage | MIMEPart,
) -> Iterator[EmailMessage | MIMEPart]:
    """Text leaves reachable WITHOUT descending into message/* parts.

    email.walk() recurses into message/rfc822 (and friends), which would
    expose the inside of an attached .eml — forwarded mail as an attachment
    is a common real shape and was a verified body leak. Only genuine
    multipart containers are descended into; a single-part (non-message)
    leaf yields itself (iter_parts() alone forgets non-multipart roots —
    verified 3.12 regression that blanked every plain-text body)."""
    maintype = part.get_content_maintype()
    if maintype == "multipart":
        for child in part.iter_parts():
            yield from _leaf_text_parts(child)
    elif maintype != "message":  # message/rfc822 etc: never a body source
        yield part


def _first_inline_text(msg: EmailMessage) -> str:
    """First NON-attachment text payload; attachments can never be the body
    (an invoice's contents leaking into the classifier corpus is the failure
    class this whole path exists to prevent)."""
    try:
        for part in _leaf_text_parts(msg):
            if _is_attachmentish(part):
                continue
            try:
                payload = part.get_payload(decode=True)
            except Exception:
                continue
            if isinstance(payload, bytes):
                charset = part.get_content_charset() or "utf-8"
                try:
                    return payload.decode(charset, errors="replace")
                except LookupError:
                    return payload.decode("utf-8", errors="replace")
    except Exception:  # RecursionError on pathological nesting, parser bugs
        log.warning("mime: inline scan failed (structure degraded)")
    return ""


def _strip_html(html: str) -> str:
    parser = _HtmlText()
    try:
        parser.feed(html)
    except Exception:  # malformed HTML — fall back to regex tag-strip
        return _TAG.sub(" ", html)
    return parser.text()


def _parse_date(msg: Message) -> datetime | None:
    raw = msg.get("Date")
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(str(raw))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _single(msg: Message, header: str) -> str:
    return _scrub(str(msg.get(header) or "").strip())


def _addresses(msg: Message, header: str, data: bytes) -> list[str]:
    """Addr-specs from an address header; per-header hardening because the
    stdlib address parser raises on garbage like 'a: b; c <' (verified)."""
    out: list[str] = []
    try:
        # On 3.12 the lazy parser raises HERE (get_all materializes the
        # value); on 3.13 inside .addresses iteration. Both paths fall back
        # to raw-string extraction so one junk header costs that header's
        # addresses only — never subject/body/date via the top-level
        # catcher (interpreter-dependent CI regression, verified on both).
        hdrs = list(msg.get_all(header) or [])
    except Exception:
        log.warning("mime: %s header unparseable (raw fallback)", header)
        # Every parsed view (get/items) re-runs the same broken parse on
        # 3.12; a compat30 twin over the raw BYTES has no structured
        # parsing at all — its headers are plain strings and never raise.
        return _fallback_addrs(_raw_header(data, header))
    for hdr in hdrs:
        acc: list[str] = []
        try:
            for addr in getattr(hdr, "addresses", ()):
                spec = getattr(addr, "addr_spec", "")
                # Bias: specs without "@" (local-only like "root", or
                # parser junk like "<>") are not useful features — drop.
                if spec and "@" in (scrubbed := _scrub(spec)):
                    acc.append(scrubbed)
        except Exception:
            # Lazy-parse bomb mid-iteration: recover from raw header bytes.
            log.warning("mime: %s fragment unparseable (raw fallback)", header)
            acc = _fallback_addrs(_raw_header(data, header))
        out.extend(acc)
    return out


def _raw_header(data: bytes, header: str) -> str:
    """Raw (possibly RFC2047-encoded) header values from a compat30 twin —
    string headers, no structured parse. Guarded anyway: a fallback must
    never itself become the failure (e.g. RecursionError on deep nesting)."""
    try:
        twin = message_from_bytes(data)
        lower = header.lower()
        return ", ".join(v for k, v in twin.items() if k.lower() == lower)
    except Exception:
        return ""


_QUOTED_RE = re.compile(r'"[^"]*"')
_COMMENT_RE = re.compile(r"\([^)]*\)")
# Non-empty domain labels: rejects dotty junk like x..com.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}")


def _fallback_addrs(raw: str) -> list[str]:
    """Regex addr-spec extraction for unparseable headers.

    Quoted strings and comments are stripped FIRST: display-name decoys
    ('"attack@evil.example" <boss@corp.example') must not out-rank the
    real spec — this also converges the 3.12 fallback with 3.13's
    structured answer for the common shapes."""
    clean = _COMMENT_RE.sub(" ", _QUOTED_RE.sub(" ", raw))
    return [_scrub(m) for m in _EMAIL_RE.findall(clean)]


def _all_recipients(msg: Message, data: bytes) -> tuple[str, ...]:
    seen: dict[str, None] = {}  # order-preserving dedup
    for header in ("To", "Cc", "Bcc"):
        for addr in _addresses(msg, header, data):
            seen.setdefault(addr, None)
    return tuple(seen)


def parse_mime(data: bytes) -> ParsedMail:
    """Parse RFC 2822 bytes into normalized envelope + body. NEVER raises."""
    try:
        msg = message_from_bytes(data, policy=default_policy)
        return ParsedMail(
            subject=_single(msg, "Subject"),
            sender=(_addresses(msg, "From", data) or [""])[0],
            recipients=_all_recipients(msg, data),
            date=_parse_date(msg),
            body_text=_body_text(msg),
        )
    except Exception as e:  # includes RecursionError: never-raises contract
        log.warning(
            "mime: unparseable message (%s); degraded to empty parse",
            type(e).__name__,
        )
        return ParsedMail(subject="", sender="", recipients=(), date=None, body_text="")


def decode_gmail_raw(raw_b64url: str) -> bytes:
    """Decode Gmail's ``messages...raw`` field (base64url, un-padded).

    Raises ValueError/UnicodeEncodeError on malformed base64url — unlike
    parse_mime, NOT under a never-raises contract: its input is Gmail's own
    API response, and a corrupted field should fail that one message loudly
    in the export loop, not degrade silently.
    """
    padded = raw_b64url + "=" * (-len(raw_b64url) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))
