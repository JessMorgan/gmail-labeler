"""apply CLI (ticket #26, C3): the `apply --since [--apply]` command —
the ONLY bulk path. The pins here are the CLI-layer contract (the core's
behavioral matrix lives in tests/test_apply.py): the exit-code contract
(0/2/4/5), the wire query (after:<epoch> WITHOUT in:inbox), the preview's
zero-write discipline through the FULL startup (the name-based
READ_SURFACE idiom on the fake call log — ensure_label LIST round-trips
are legitimate reads, the refusal is only at MINT), the preview mint
guard (a Tier Label the mailbox lacks fails named, never silently
creates), the AC5 provenance warning (ONE counted line at the --since
boundary), and the human-only grep pin (daemon.py carries no reference
to the apply machinery — the AC's "never automatic").
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from gmail_labeler.appdb import AppDB
from gmail_labeler.cli import app
from gmail_labeler.fake_gmail import FakeGmailClient
from gmail_labeler.gmail_client import LabelInfo
from gmail_labeler.gmail_errors import GmailAuthError
from test_shadow_cli import (
    _backlog_messages,
    _backlog_scores,
    _prepare,
    _shadow_cfg,
    _shadow_factory,
    _shadow_labels,
)

runner = CliRunner()

# the read surface the preview zero-write pin allows (the shadow idiom,
# minus history — apply never polls history)
READ_SURFACE = frozenset(
    {"list_messages", "get_message", "list_labels", "list_filters"}
)


def _apply_labels() -> list[LabelInfo]:
    """The mailbox WITH the three Tier Labels present — a dry preview
    resolves them by LOOKUP only (it never mints), so the happy preview
    fixture carries them."""
    return [
        *_shadow_labels(),
        LabelInfo("LA", "ai-applied", "user"),
        LabelInfo("LH", "ai-high", "user"),
        LabelInfo("LM", "ai-med", "user"),
    ]


def _apply_cfg(tmp_path: Path) -> Path:
    return _shadow_cfg(tmp_path, store=f'db_path = "{tmp_path / "state.db"}"')


def _run_apply_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake: FakeGmailClient | None = None,
    extra: tuple[str, ...] = ("--since", "2026-10-01"),
    cfg: Path | None = None,
    client_factory: object | None = None,
    model_factory: object | None = None,
) -> tuple[Result, FakeGmailClient]:
    import gmail_labeler.cli as cli

    fake = fake or FakeGmailClient(messages=_backlog_messages(), labels=_apply_labels())
    monkeypatch.setattr(
        cli,
        "_daemon_client",
        client_factory if client_factory is not None else (lambda cfg_: fake),
    )
    monkeypatch.setattr(
        cli,
        "_daemon_model_factory",
        model_factory
        if model_factory is not None
        else _shadow_factory(_backlog_scores()),
    )
    monkeypatch.setattr(
        cli, "_now", lambda: datetime(2026, 11, 1, 12, 30, 0, tzinfo=UTC)
    )
    cfg = cfg or _apply_cfg(tmp_path)
    result = runner.invoke(
        app,
        [
            "apply",
            "--config",
            str(cfg),
            "--bands",
            str(tmp_path / "bands.json"),
            "--graduation",
            str(tmp_path / "graduation.json"),
            *extra,
        ],
    )
    return result, fake


# -- AC1: the preview writes NOTHING through the full startup -----------------


def test_preview_exit_0_zero_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC1: without --apply the preview exits 0 and performs ZERO Gmail
    # writes — the recorded calls are a subset of the READ surface (the
    # startup's ensure_label LIST round-trips are legitimate reads; the
    # refusal is only at apply_labels / ensure_label MINT), and the fake's
    # write ledger is EMPTY
    _prepare(tmp_path, monkeypatch)
    result, fake = _run_apply_cli(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.stderr
    methods = {call.split(":", 1)[0] for call in fake.calls}
    assert methods <= READ_SURFACE, f"write call recorded: {methods - READ_SURFACE}"
    assert fake.writes == []  # no apply_labels, no ensure_label create
    assert "list_messages" in methods and "get_message" in methods
    # the write candidate is REPORTED would-apply; the mailbox is untouched
    assert "aa22 would-apply Money/Bills" in result.stderr
    assert fake.messages["aa22"].label_ids == ("INBOX",)


# -- the wire query: after:<epoch> WITHOUT in:inbox ---------------------------


def test_backfill_query_is_after_epoch_without_inbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --since 2026-10-01 → UTC midnight of that day → after:<epoch>,
    # deliberately WITHOUT in:inbox (backfill targets historical mail,
    # mostly archived). The epoch form is the unambiguous one (#23).
    _prepare(tmp_path, monkeypatch)
    result, fake = _run_apply_cli(
        tmp_path, monkeypatch, extra=("--since", "2026-10-01")
    )
    assert result.exit_code == 0, result.stderr
    expected_epoch = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp())
    # the fake logs "list_messages:<query>:<token>" — the query itself
    # carries a colon (after:<epoch>), so match the full entry
    list_calls = [c for c in fake.calls if c.startswith("list_messages:")]
    assert list_calls, "the backfill never paged list_messages"
    assert list_calls[0] == f"list_messages:after:{expected_epoch}:"
    assert "in:inbox" not in list_calls[0]


# -- AC3: --apply records apply-sourced Applications --------------------------


def test_apply_backfill_records_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC3: with --apply the backfill records Applications identically to
    # daemon-applied ones (source="apply"); the write lands on the mail
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
    result, _ = _run_apply_cli(
        tmp_path, monkeypatch, fake=fake, extra=("--since", "2026-10-01", "--apply")
    )
    assert result.exit_code == 0, result.stderr
    db = AppDB(tmp_path / "state.db")
    db.open()
    rows = db.applications_for_message("aa22")
    assert [(r.label, r.source) for r in rows] == [("Money/Bills", "apply")]
    # the daemon's cursor was NEVER established by apply (no daemon_state write)
    assert db.get_history_cursor() is None
    db.close()


# -- the preview mint guard: a missing Tier Label fails named, never creates --


def test_preview_missing_tier_label_fails_named_not_mints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the preview's read-only discipline (AC1): a dry run whose Tier
    # Labels are ABSENT must fail NAMED (exit 5), never silently create
    # them (a create is a wire write). The daemon/backfill path mints via
    # ensure_label; the preview does not
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
    result, _ = _run_apply_cli(tmp_path, monkeypatch, fake=fake)
    assert result.exit_code == 5
    assert "does not create Tier Labels" in result.stderr
    # the mint guard held: NO ensure_label create was recorded
    assert fake.writes == []


# -- AC5: the provenance warning at the --since boundary ----------------------


def test_provenance_warning_one_counted_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC5: a corpus record dated at/after the --since boundary whose date
    # lacks a server-authoritative provenance stamp (a pre-#17 "unknown")
    # surfaces ONE named warning with the count (never per-message spam)
    _prepare(tmp_path, monkeypatch)
    # rewrite ONE in-window corpus record to lack provenance (append-only
    # doctrine: a pre-#17 line simply has no date_provenance field)
    from gmail_labeler.corpus_record import CorpusRecord, body_hash
    from gmail_labeler.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    rec = CorpusRecord(
        id="ffff000000000001",
        date=datetime(2026, 10, 15, tzinfo=UTC),  # inside the --since window
        subject="old",
        sender="x@y.z",
        recipients=("me@x.y",),
        labels=("Shipping",),
        body_hash=body_hash("b"),
        source="api",
        date_provenance="unknown",  # the pre-#17 state
    )
    store.put(rec, "b")
    result, _ = _run_apply_cli(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert "lack a" in result.stderr and "Date Provenance" in result.stderr
    assert "1 corpus record" in result.stderr  # the count, ONE line
    # not spam: exactly ONE warning line (the phrase appears twice inside
    # the single line, so count the line's distinctive prefix)
    assert result.stderr.count("apply: WARNING") == 1


# -- the exit-code contract ---------------------------------------------------


def test_apply_since_required_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --since is REQUIRED (the only bulk path never guesses its window):
    # its absence is a usage error (exit 2), before any work
    _prepare(tmp_path, monkeypatch)
    result, _ = _run_apply_cli(tmp_path, monkeypatch, extra=())
    assert result.exit_code == 2


def test_apply_since_non_iso_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path, monkeypatch)
    result, _ = _run_apply_cli(tmp_path, monkeypatch, extra=("--since", "20261001"))
    assert result.exit_code == 2
    assert "not an ISO date" in result.stderr


def test_apply_auth_gate_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path, monkeypatch)
    from gmail_labeler.auth import AuthError

    def _boom(cfg_: object) -> None:
        raise AuthError("not signed in")

    result, _ = _run_apply_cli(tmp_path, monkeypatch, client_factory=_boom)
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_apply_midrun_auth_error_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the propagating human-gated arm lands on the contract: auth on a
    # wire call → 4, never a traceback
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(messages=_backlog_messages(), labels=_apply_labels())
    fake.fail("get_message", GmailAuthError("token revoked"))
    result, _ = _run_apply_cli(
        tmp_path, monkeypatch, fake=fake, extra=("--since", "2026-10-01", "--apply")
    )
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_apply_bodies_absent_cli_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # invariant #5 end-to-end: the marker rides a real scored message;
    # the preview line carries ids/labels/scores only
    _prepare(tmp_path, monkeypatch)
    from test_shadow_cli import _eml

    marker = "SECRET-BODY-MARKER"
    messages = _backlog_messages()
    for m in messages:
        if m.id == "aa22":
            m.eml = _eml("backlog two", f"{marker} body two")
    scores = dict(_backlog_scores())
    del scores["backlog two\nbody two"]
    scores[f"backlog two\n{marker} body two"] = (0.85, 0.3)
    fake = FakeGmailClient(messages=messages, labels=_apply_labels())
    result, _ = _run_apply_cli(
        tmp_path, monkeypatch, fake=fake, model_factory=_shadow_factory(scores)
    )
    assert result.exit_code == 0, result.stderr
    assert "aa22 would-apply" in result.stderr  # non-vacuity: the line landed
    assert marker not in result.stdout
    assert marker not in result.stderr


# -- the human-only grep pin (AC: "never automatic") --------------------------


def test_daemon_never_references_the_apply_machinery() -> None:
    # the AC's "never automatic", pinned structurally: daemon.py carries
    # NO reference to the apply module or its machinery — the bulk path
    # is human-only, the daemon can never invoke it. (apply_labels /
    # plan_message_write are the daemon's OWN write path, not the apply
    # machinery — the pin is on the apply module's names specifically)
    from pathlib import Path as _P

    import gmail_labeler.daemon as daemon_module

    src = _P(daemon_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "run_apply",
        "commit_applications",
        "from .apply",
        "gmail_labeler.apply",
        "apply_cmd",
        "ApplyOutcome",
    ):
        assert forbidden not in src, (
            f"daemon.py references apply machinery: {forbidden}"
        )
