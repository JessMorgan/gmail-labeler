from __future__ import annotations

import base64
import binascii

import pytest

from gmail_labeler.mime_extract import decode_gmail_raw, parse_mime


def _msg(
    *,
    subject: str = "Hello",
    from_: str = "Alice <alice@example.com>",
    to: str = "me@example.com",
    date: str = "Sun, 30 Oct 2026 10:00:00 -0700",
    body: str = "Plain body",
    html: str | None = None,
    attachment: str | None = None,
    plain_override: str | None = None,
) -> bytes:
    plain = body if plain_override is None else plain_override
    if html is not None or attachment is not None:
        boundary = "BOUND123"
        parts = [f"--{boundary}\r\nContent-Type: text/plain\r\n\r\n{plain}"]
        if html is not None:
            parts.append(f"--{boundary}\r\nContent-Type: text/html\r\n\r\n{html}")
        if attachment is not None:
            parts.append(
                f'--{boundary}\r\nContent-Type: text/plain; name="invoice.txt"\r\n'
                f'Content-Disposition: attachment; filename="invoice.txt"\r\n\r\n'
                f"{attachment}"
            )
        parts.append(f"--{boundary}--\r\n")
        payload = "\r\n".join(parts)
        ctype = f'multipart/alternative; boundary="{boundary}"'
    else:
        ctype = "text/plain"
        payload = plain
    return (
        f"Subject: {subject}\r\n"
        f"From: {from_}\r\n"
        f"To: {to}\r\n"
        f"Date: {date}\r\n"
        f"Content-Type: {ctype}\r\n"
        f"\r\n{payload}"
    ).encode()


def test_parses_envelope_and_date() -> None:
    parsed = parse_mime(_msg())
    assert parsed.subject == "Hello"
    assert parsed.sender == "alice@example.com"
    assert parsed.recipients == ("me@example.com",)
    assert parsed.date is not None and parsed.date.hour == 17  # -0700 -> UTC


def test_html_body_stripped_when_no_plain() -> None:
    raw = (
        b"Content-Type: text/html; charset=utf-8\r\n\r\n"
        b"<div><style>p{color:red}</style><p>Hi <b>there</b></p></div>"
    )
    assert parse_mime(raw).body_text == "Hi there"


def test_script_dropped_from_html() -> None:
    raw = b"Content-Type: text/html\r\n\r\n<p>keep</p><script>alert('x')</script>"
    assert parse_mime(raw).body_text == "keep"


def test_multipart_prefers_nonblank_plain() -> None:
    parsed = parse_mime(_msg(body="The real text", html="<p>ignored</p>"))
    assert parsed.body_text == "The real text"


def test_empty_plain_does_not_shadow_html() -> None:
    # The dominant newsletter shape (verified regression: empty plain
    # alternative + real HTML content).
    parsed = parse_mime(_msg(plain_override="", html="<p>Real content here</p>"))
    assert parsed.body_text == "Real content here"


def test_attachment_part_is_not_the_body_even_alone() -> None:
    parsed = parse_mime(_msg(body="intro", attachment="SECRET-INVOICE-DATA"))
    assert "SECRET-INVOICE-DATA" not in parsed.body_text
    # Attachment-ONLY message: no inline text at all -> empty body, never
    # the attachment's content (verified CRITICAL from review).
    only = _msg(html=None, body="", attachment="ATTACHMENT-SECRET-BODY").replace(
        b"Content-Type: multipart/alternative",
        b"Content-Type: multipart/mixed",
    )
    assert parse_mime(only).body_text == ""


def test_recipients_dedup_and_parse() -> None:
    raw = (
        b"To: Bob <bob@x.com>, carol@x.com, Bob <bob@x.com>\r\n"
        b"Cc: me@x.com\r\n"
        b"Content-Type: text/plain\r\n\r\nbody"
    )
    assert parse_mime(raw).recipients == ("bob@x.com", "carol@x.com", "me@x.com")


def test_whitespace_canonicalized_in_body_text() -> None:
    raw = b"Content-Type: text/plain\r\n\r\nlots    of\n\n\n\nspace \t here"
    assert parse_mime(raw).body_text == "lots of\n\nspace here"


def test_garbage_headers_never_raise() -> None:
    # Verified stdlib address-parser bombs (review CRITICAL):
    for raw in (
        b"From: g1: g2: a@b;;\r\n\r\nx",
        b"From: <a@b>, <\r\nTo: <x@y>, <\r\n\r\nx",
        b"Subject: =?bogus?Q?broken?=\r\n\r\nx",
        b"this is not \x00\x01 a valid message at all",
        b"",
        b"\r\n\r\n",
        b"From: \xff\xfe raw <\xff\xfe@x>\r\n\r\nbody",
    ):
        parsed = parse_mime(raw)  # must not raise
        # and its outputs must survive UTF-8 encoding (no deferred bombs)
        parsed.subject.encode("utf-8")
        parsed.sender.encode("utf-8")
        for addr in parsed.recipients:
            addr.encode("utf-8")
        parsed.body_text.encode("utf-8")


def test_8bit_from_has_no_surrogates() -> None:
    parsed = parse_mime(b"From: \xff\xfe <a@x>\r\n\r\nbody")
    assert "\udcff" not in parsed.sender
    assert parsed.sender  # scrubbed to replacement chars, still present


def test_deeply_nested_multipart_does_not_raise() -> None:
    depth = 1500  # stdlib recursion limit is 1000: RecursionError path
    parts = "".join(
        f"--b{i}\r\nContent-Type: multipart/mixed; boundary=b{i + 1}\r\n\r\n"
        for i in range(depth)
    )
    raw = f"Content-Type: multipart/mixed; boundary=b0\r\n\r\n{parts}tail".encode()
    assert parse_mime(raw).body_text == ""


def test_missing_or_bad_date_is_none_not_now() -> None:
    # Determinism (invariant #7): callers supply authoritative timestamps.
    assert parse_mime(b"Content-Type: text/plain\r\n\r\nx").date is None
    assert parse_mime(b"Date: not-a-date\r\n\r\nx").date is None


def test_unknown_charset_degrades_to_replacement() -> None:
    raw = b"Content-Type: text/plain; charset=not-a-real-charset\r\n\r\ncaf\xe9"
    assert "\ufffd" in parse_mime(raw).body_text or "caf" in parse_mime(raw).body_text


def test_decode_gmail_raw_round_trips() -> None:
    original = _msg(body="round trip")
    enc = base64.urlsafe_b64encode(original).decode("ascii").rstrip("=")
    assert decode_gmail_raw(enc) == original


def test_decode_gmail_raw_raises_on_garbage() -> None:
    with pytest.raises(binascii.Error):
        decode_gmail_raw("!!!not base64!!!")


def test_name_only_attachment_never_becomes_body() -> None:
    # get_body selects by STRUCTURE: a text/plain part with name= and NO
    # Content-Disposition (classic mailer output) must still be excluded
    # (deepseek CRITICAL on the primary path).
    raw = (
        b"Content-Type: multipart/mixed; boundary=b\r\n\r\n"
        b'--b\r\nContent-Type: text/plain; name="inv.txt"\r\n\r\nSECRETINVOICE\r\n'
        b"--b--\r\n"
    )
    assert parse_mime(raw).body_text == ""


def test_uppercase_attachment_disposition_excluded() -> None:
    raw = (
        b"Content-Type: multipart/mixed; boundary=b\r\n\r\n"
        b"--b\r\nContent-Type: text/plain\r\n"
        b"Content-Disposition: ATTACHMENT\r\n\r\nCAPS-ATTACH\r\n"
        b"--b--\r\n"
    )
    assert parse_mime(raw).body_text == ""


def test_inline_binary_cannot_be_body() -> None:
    raw = (
        b"Content-Type: multipart/related; boundary=b\r\n\r\n"
        b"--b\r\nContent-Type: image/png\r\nContent-ID: <img1>\r\n"
        b"\r\n\x89PNGDATA\r\n"
        b"--b--\r\n"
    )
    assert parse_mime(raw).body_text == ""


def test_junk_address_does_not_eat_valid_ones() -> None:
    # per-address tolerance (deepseek minor): one bomb must not zero the list.
    raw = b"To: a@x.com, b@x.com, <\r\n\r\nbody"
    got = parse_mime(raw).recipients
    assert "a@x.com" in got and "b@x.com" in got


def test_attached_eml_cannot_supply_body() -> None:
    # email.walk() descends into message/rfc822 — the inner parts of an
    # attached .eml (forwarded mail: a COMMON shape) must never reach the
    # body (qwen-verified leak).
    eml = (
        b"Content-Type: multipart/mixed; boundary=outer\r\n\r\n"
        b"--outer\r\nContent-Type: text/plain\r\n\r\nvisible intro\r\n"
        b"--outer\r\nContent-Type: message/rfc822\r\n"
        b'Content-Disposition: attachment; filename="f.eml"\r\n\r\n'
        b"Subject: inner\r\nContent-Type: text/plain\r\n\r\n"
        b"INNER-SECRET-BODY\r\n"
        b"--outer--\r\n"
    )
    parsed = parse_mime(eml)
    assert parsed.body_text == "visible intro"
    assert "INNER-SECRET-BODY" not in parsed.body_text


def test_attachment_only_eml_yields_no_body() -> None:
    eml = (
        b"Content-Type: multipart/mixed; boundary=outer\r\n\r\n"
        b'--outer\r\nContent-Type: message/rfc822; name="f.eml"\r\n\r\n'
        b"Subject: inner\r\nContent-Type: text/plain\r\n\r\n"
        b"INNER-SECRET-BODY\r\n"
        b"--outer--\r\n"
    )
    parsed = parse_mime(eml)
    assert parsed.body_text == ""
    assert not parsed.is_parseable_body()


def test_junk_to_header_keeps_subject_and_body() -> None:
    # 3.12's eager header parse used to blank the WHOLE message over one
    # junk address (105/4500 fuzz cases): the cost must stay in To.
    raw = (
        b"Subject: Keep me\r\n"
        b"To: a@x.com, b@x.com, <\r\n"
        b"Date: Sun, 30 Oct 2026 10:00:00 +0000\r\n"
        b"Content-Type: text/plain\r\n\r\nbody stays"
    )
    parsed = parse_mime(raw)
    assert parsed.subject == "Keep me"
    assert parsed.body_text == "body stays"
    assert parsed.date is not None
    assert "a@x.com" in parsed.recipients and "b@x.com" in parsed.recipients


def test_fallback_decoy_cannot_outrank_real_sender() -> None:
    # 3.12 falls back to regex for this junk header; quote-stripping makes
    # it pick the REAL spec, converging with 3.13's structured parse.
    raw = b'From: "attack@evil.example" <boss@corp.example, <\r\n\r\nx'
    assert parse_mime(raw).sender == "boss@corp.example"


def test_fallback_rejects_dotty_domain_junk() -> None:
    raw = b"To: ..a..b@x..com, ok@good.example, <\r\n\r\nx"
    got = parse_mime(raw).recipients
    assert "ok@good.example" in got
    assert all(".." not in r for r in got)
