"""shadow run CLI (ticket #23, C2): the Shadow Pass — score the recent
Backlog through the sidecar-validated checkpoint + Graduation Table +
Threshold Bands and APPLY NOTHING. Exit-code contract 0/2/3/4/5; the
zero-Gmail-write pin (recorded calls ⊆ the read surface); the
verify_bands/verify_graduation FIRST-consumer refusals (identity
refusal exit 5, naming fields; unreadable-vs-unparseable-vs-absent
cause naming); the degraded per-message skip posture; bodies-absent
across DB rows + stdout/stderr + caplog (invariant #5).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from gmail_labeler.cli import app
from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
from gmail_labeler.gmail_client import LabelInfo
from gmail_labeler.gmail_errors import GmailTransientError

runner = CliRunner()

SPLIT = "2026-01-01"  # train strictly before, test at/after (invariant #7)

# the read surface the zero-write pin allows (the Protocol's read
# methods minus history — shadow never polls history)
READ_SURFACE = frozenset(
    {"list_messages", "get_message", "list_labels", "list_filters"}
)


def _seed_corpus(tmp_path: Path, *, n: int = 12) -> Path:
    """Records on BOTH sides of the split, with the sidecar pair the
    build needs (invariant #6: an empty filter list is a valid bridge)."""
    from gmail_labeler.corpus_record import CorpusRecord, body_hash
    from gmail_labeler.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    store.enforce_modes()
    now = datetime.now(UTC)
    for i in range(n):
        days_ago = 400 - i if i < n // 2 else 10 + i
        rec = CorpusRecord(
            id=f"{i:016x}",
            date=now - timedelta(days=days_ago),
            subject=f"subject {i}",
            sender="shop@x.z",
            recipients=("me@x.y",),
            labels=("Shipping",) if i % 2 == 0 else ("Money/Bills",),
            body_hash=body_hash(f"body {i}"),
            source="api",
            date_provenance="internal_date",
        )
        store.put(rec, f"body {i}")
    for name, payload in (("filters.json", "[]"), ("labels.json", "{}")):
        f = store.root / name
        f.write_text(payload, encoding="utf-8")
        os.chmod(f, 0o600)
    return store.root


def _shadow_cfg(
    tmp_path: Path,
    *,
    encoder: str = "tiny-encoder",
    labels: str = 'managed = ["Money/Bills", "Shipping"]',
    store: str = "",
) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n'
        f"[labels]\n{labels}\n"
        f"[store]\n{store}\n"
        f'[model]\nencoder = "{encoder}"\n'
        f'checkpoint_dir = "{tmp_path / "ckpt"}"\n',
        encoding="utf-8",
    )
    return cfg


def _train_factory():
    """The train seam stand-in (test_train_cli's pattern)."""
    from gmail_labeler.model import FakeModel

    def factory(cfg, examples, texts, vocabulary, checkpoint_dir, device):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "placeholder.bin").write_bytes(b"fake weights")
        return FakeModel(vocabulary, scores={})

    return factory


def _eval_factory(
    scores: dict[str, tuple[float, ...]],
    vocabulary: tuple[str, ...] | None = None,
):
    """The eval seam stand-in (test_bands_cli's pattern)."""
    from gmail_labeler.model import FakeModel

    seen: dict[str, object] = {}

    def factory(checkpoint_dir: Path, vocab: tuple[str, ...]):
        seen["checkpoint_dir"] = checkpoint_dir
        seen["vocabulary"] = vocab
        return FakeModel(vocabulary if vocabulary is not None else vocab, scores=scores)

    factory.seen = seen  # type: ignore[attr-defined]
    return factory


def _shadow_factory(
    scores: dict[str, tuple[float, ...]],
    vocabulary: tuple[str, ...] | None = None,
):
    """The shadow seam stand-in: a FakeModel producer keyed by TEXT
    (the Model Protocol's alignment: row i ↔ texts[i])."""
    from gmail_labeler.model import FakeModel

    seen: dict[str, object] = {}

    def factory(checkpoint_dir: Path, vocab: tuple[str, ...]):
        seen["checkpoint_dir"] = checkpoint_dir
        seen["vocabulary"] = vocab
        return FakeModel(vocabulary if vocabulary is not None else vocab, scores=scores)

    factory.seen = seen  # type: ignore[attr-defined]
    return factory


def _run_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    split_date: str = SPLIT,
    force: bool = False,
    cfg: Path | None = None,
) -> Result:
    import gmail_labeler.cli as cli

    monkeypatch.setattr(cli, "_model_factory", _train_factory())
    args = [
        "train",
        "--config",
        str(cfg if cfg is not None else _shadow_cfg(tmp_path)),
        "--split-date",
        split_date,
    ]
    if force:
        args.append("--force")
    return runner.invoke(app, args)


def _test_side_scores() -> dict[str, tuple[float, ...]]:
    """Hand-authored scores for the 6 test-side texts (i=6..11), in
    vocabulary order ("Money/Bills", "Shipping")."""
    return {
        "subject 6\nbody 6": (0.2, 0.9),
        "subject 7\nbody 7": (0.8, 0.1),
        "subject 8\nbody 8": (0.3, 0.7),
        "subject 9\nbody 9": (0.6, 0.2),
        "subject 10\nbody 10": (0.1, 0.4),
        "subject 11\nbody 11": (0.4, 0.3),
    }


def _run_eval(
    tmp_path: Path,
    *extra: str,
    monkeypatch: pytest.MonkeyPatch,
    cfg: Path | None = None,
) -> Result:
    import gmail_labeler.cli as cli

    monkeypatch.setattr(cli, "_eval_model_factory", _eval_factory(_test_side_scores()))
    return runner.invoke(
        app,
        [
            "eval",
            "--config",
            str(cfg if cfg is not None else _shadow_cfg(tmp_path)),
            *extra,
        ],
    )


def _eml(subject: str, body: str) -> bytes:
    return (
        f"Subject: {subject}\r\nFrom: a@x.com\r\nTo: me@x.com\r\n"
        "Content-Type: text/plain\r\n\r\n" + body
    ).encode()


def _backlog_messages() -> list[FakeMessage]:
    """The Backlog the shadow run scores: 4 messages, ids hex.
    m1: Shipping present (skip-present candidate), m2: nothing present,
    m3: Money/Bills present, m4: nothing present."""
    return [
        FakeMessage(
            id="aa11",
            eml=_eml("backlog one", "body one"),
            label_ids=("INBOX", "L1"),
            thread_id="t1",
        ),
        FakeMessage(
            id="aa22",
            eml=_eml("backlog two", "body two"),
            label_ids=("INBOX",),
            thread_id="t2",
        ),
        FakeMessage(
            id="aa33",
            eml=_eml("backlog three", "body three"),
            label_ids=("INBOX", "L2"),
            thread_id="t3",
        ),
        FakeMessage(
            id="aa44",
            eml=_eml("backlog four", "body four"),
            label_ids=("INBOX",),
            thread_id="t4",
        ),
    ]


def _shadow_labels() -> list[LabelInfo]:
    return [
        LabelInfo("INBOX", "INBOX", "system"),
        LabelInfo("L1", "Shipping", "user"),
        LabelInfo("L2", "Money/Bills", "user"),
    ]


def _backlog_scores() -> dict[str, tuple[float, ...]]:
    """Hand-authored scores for the 4 backlog texts, in vocabulary
    order ("Money/Bills", "Shipping"):

    m1 (aa11): (0.5, 0.95)  → Shipping high (present → skip-present)
    m2 (aa22): (0.85, 0.3)  → Money/Bills high (absent → ADDED)
    m3 (aa33): (0.75, 0.2)  → Money/Bills med (present → skip-present)
    m4 (aa44): (0.4, 0.85)  → Shipping med (absent → ADDED)
    """
    return {
        "backlog one\nbody one": (0.5, 0.95),
        "backlog two\nbody two": (0.85, 0.3),
        "backlog three\nbody three": (0.75, 0.2),
        "backlog four\nbody four": (0.4, 0.85),
    }


def _db_has_rows(db_path: Path) -> bool:
    import sqlite3

    if not db_path.is_file():
        return False
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM shadow_results").fetchone()[0] > 0
    finally:
        conn.close()


def _run_shadow(
    tmp_path: Path,
    *extra: str,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeGmailClient | None = None,
    scores: dict[str, tuple[float, ...]] | None = None,
    cfg: Path | None = None,
) -> tuple[Result, FakeGmailClient]:
    import gmail_labeler.cli as cli

    fake = (
        fake
        if fake is not None
        else FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
    )
    monkeypatch.setattr(cli, "_shadow_client", lambda cfg_: fake)
    monkeypatch.setattr(
        cli,
        "_shadow_model_factory",
        _shadow_factory(scores if scores is not None else _backlog_scores()),
    )
    bands = tmp_path / "bands.json"
    graduation = tmp_path / "graduation.json"
    args = [
        "shadow",
        "run",
        "--config",
        str(cfg if cfg is not None else _shadow_cfg(tmp_path)),
        "--bands",
        str(bands),
        "--graduation",
        str(graduation),
        *extra,
    ]
    return runner.invoke(app, args), fake


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The full prep: corpus + train + eval --bands + eval
    --graduation --confirm (the golden gates: min_examples = 3 → both
    labels graduate)."""
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _shadow_cfg(tmp_path)
    bands = tmp_path / "bands.json"
    assert (
        _run_eval(
            tmp_path, "--bands", str(bands), monkeypatch=monkeypatch, cfg=cfg
        ).exit_code
        == 0
    )
    graduation = tmp_path / "graduation.json"
    # min_examples = 3: both labels graduate (3 >= 3 AT the gate)
    grad_cfg = _shadow_cfg(tmp_path, labels='managed = ["Money/Bills", "Shipping"]')
    grad_cfg.write_text(
        grad_cfg.read_text(encoding="utf-8").replace(
            "[store]", "[graduation]\nmin_examples = 3\n[store]"
        ),
        encoding="utf-8",
    )
    assert (
        _run_eval(
            tmp_path,
            "--graduation",
            str(graduation),
            "--confirm-graduation",
            monkeypatch=monkeypatch,
            cfg=grad_cfg,
        ).exit_code
        == 0
    )


def test_shadow_run_happy_path_records_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC5/AC2: the run records the would-apply verdicts in the state DB
    # (one transaction per run) and applies NOTHING
    _prepare(tmp_path, monkeypatch)
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    db_path = tmp_path / "gmail-labeler.db"  # the [store] default
    assert db_path.is_file()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT run_id, message_id, label, probability, band "
            "FROM shadow_results ORDER BY message_id, label"
        ).fetchall()
    finally:
        conn.close()
    # 4 messages x 2 graduated labels = 8 rows
    assert len(rows) == 8
    by = {(r[1], r[2]): (r[3], r[4]) for r in rows}
    # m1: Shipping 0.95 high (present → skip-present), Money/Bills 0.5 none
    assert by[("aa11", "Shipping")] == (0.95, "high")
    assert by[("aa11", "Money/Bills")] == (0.5, None)
    # m2: Money/Bills 0.85 high (absent → ADDED), Shipping 0.3 none
    assert by[("aa22", "Money/Bills")] == (0.85, "high")
    assert by[("aa22", "Shipping")] == (0.3, None)
    # m3: Money/Bills 0.75 med (present → skip-present), Shipping 0.2 none
    assert by[("aa33", "Money/Bills")] == (0.75, "med")
    assert by[("aa33", "Shipping")] == (0.2, None)
    # m4: Shipping 0.85 med (absent → ADDED), Money/Bills 0.4 none
    assert by[("aa44", "Shipping")] == (0.85, "med")
    assert by[("aa44", "Money/Bills")] == (0.4, None)
    # the summary is on STDERR (status, not report content); stdout is
    # empty in text mode
    assert "shadow run: run=" in result.stderr
    assert "scored=4" in result.stderr
    assert "proposed=2" in result.stderr  # m2 Money/Bills + m4 Shipping
    assert "already_present=2" in result.stderr  # m1 Shipping + m3 Money/Bills
    assert "not_applied=4" in result.stderr
    assert result.stdout == ""


def test_shadow_run_zero_gmail_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AC1: zero Gmail writes — the recorded calls are a subset of the
    # READ surface (list_messages, get_message, list_labels,
    # list_filters); no write method of any shape (the Protocol has
    # none in this ticket — the write surface lands #24/#25)
    _prepare(tmp_path, monkeypatch)
    result, fake = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    methods = {call.split(":", 1)[0] for call in fake.calls}
    assert methods <= READ_SURFACE, f"write call recorded: {methods - READ_SURFACE}"
    # and the read path was actually used (not a no-op run)
    assert "list_messages" in methods
    assert "get_message" in methods
    assert "list_labels" in methods


def test_shadow_run_since_query_is_inbox_after_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S4: the --since Backlog query is in:inbox after:<epoch-seconds> —
    # the EPOCH form (unambiguous; calendar dates in API queries are
    # interpreted as MIDNIGHT PST, and newer_than: is relative-duration
    # only). --since 2026-10-01 → UTC midnight of that date → epoch.
    # (The REAL default-window pin is
    # test_shadow_run_default_window_cli_pin — fold-4 item 2.)
    from datetime import UTC, datetime

    _prepare(tmp_path, monkeypatch)
    result, fake = _run_shadow(
        tmp_path, "--since", "2026-10-01", monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr
    expected_epoch = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp())
    list_calls = [c for c in fake.calls if c.startswith("list_messages")]
    assert list_calls[0] == f"list_messages:in:inbox after:{expected_epoch}:"


def test_backlog_cutoff_epoch_math_is_pinned() -> None:
    # S4: the epoch math is pinned with a FIXED UTC now (no clock in the
    # test): --since → UTC midnight of that date; default → now - 90d
    # (the time component kept)
    from gmail_labeler.cli import _backlog_cutoff

    fixed_now = datetime(2026, 11, 1, 12, 30, 0, tzinfo=UTC)
    # --since 2026-10-01 → UTC midnight of that date
    cutoff = _backlog_cutoff(date(2026, 10, 1), fixed_now)
    assert cutoff == datetime(2026, 10, 1, tzinfo=UTC)
    assert int(cutoff.timestamp()) == int(datetime(2026, 10, 1, tzinfo=UTC).timestamp())
    # default → now - 90 days (time component kept: 12:30)
    default_cutoff = _backlog_cutoff(None, fixed_now)
    assert default_cutoff == datetime(2026, 8, 3, 12, 30, 0, tzinfo=UTC)


def test_shadow_run_db_path_containment_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S3: a [store] db_path inside the checkpoint dir OR the corpus store
    # is a usage refusal (exit 2) — a state DB there is destroyable by
    # train --force / a corpus re-export (both sides resolved)
    _prepare(tmp_path, monkeypatch)
    # inside the checkpoint dir
    cfg_ckpt = _shadow_cfg(tmp_path, store='db_path = "ckpt/state.db"\n')
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch, cfg=cfg_ckpt)
    assert result.exit_code == 2
    assert "lives inside the checkpoint directory" in result.stderr
    # inside the corpus store
    cfg_corpus = _shadow_cfg(tmp_path, store='db_path = "corpus/state.db"\n')
    result2, _fake2 = _run_shadow(tmp_path, monkeypatch=monkeypatch, cfg=cfg_corpus)
    assert result2.exit_code == 2
    assert "lives inside the corpus store" in result2.stderr


def test_shadow_run_since_strict_iso(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S5: --since accepts ISO STRICTLY (YYYY-MM-DD) — the lenient
    # "20261001" (accepted by 3.11+ date.fromisoformat) is refused
    _prepare(tmp_path, monkeypatch)
    result, _fake = _run_shadow(
        tmp_path, "--since", "20261001", monkeypatch=monkeypatch
    )
    assert result.exit_code == 2
    assert "not an ISO date" in result.stderr


def test_shadow_run_query_escape_hatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --query replaces the whole backlog window — the default and
    # --since alike (the README's --query scope, #23)
    _prepare(tmp_path, monkeypatch)
    result, fake = _run_shadow(
        tmp_path, "--query", "is:starred", monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr
    list_calls = [c for c in fake.calls if c.startswith("list_messages")]
    assert list_calls[0] == "list_messages:is:starred:"


def test_shadow_run_limit_honored_across_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --limit is honored ACROSS pages (the pageToken loop): 4 messages,
    # page size 1, limit 2 → exactly 2 list_messages pages + 2
    # get_message calls
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(
        messages=_backlog_messages(), labels=_shadow_labels(), page_size=1
    )
    result, _fake = _run_shadow(
        tmp_path, "--limit", "2", monkeypatch=monkeypatch, fake=fake
    )
    assert result.exit_code == 0, result.stderr
    get_calls = [c for c in fake.calls if c.startswith("get_message")]
    assert len(get_calls) == 2
    list_calls = [c for c in fake.calls if c.startswith("list_messages")]
    assert len(list_calls) == 2  # paged across the 3-message window


def test_shadow_run_repeated_page_is_deduped_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the pageToken loop dedupes ids and breaks on a no-progress page
    # (a server repeating the same ids with a token would loop forever
    # without the break): a bounded fake that repeats page 1 on page 2
    # then yields new ids on page 3 → the run scores the deduped set and
    # stops (no hang, no duplicate rows)
    from gmail_labeler.gmail_client import MessagePage, MessageStub

    _prepare(tmp_path, monkeypatch)

    class RepeatingClient(FakeGmailClient):
        def __init__(self) -> None:
            super().__init__(messages=_backlog_messages(), labels=_shadow_labels())
            self._page = 0

        def list_messages(self, query, page_token=None):
            self.calls.append(f"list_messages:{query}:{page_token or ''}")
            self._page += 1
            if self._page == 1:
                # page 1: aa11, aa22 (token 1)
                return MessagePage(
                    stubs=(
                        MessageStub(id="aa11", thread_id="t1"),
                        MessageStub(id="aa22", thread_id="t2"),
                    ),
                    next_page_token="1",
                )
            if self._page == 2:
                # page 2: the SAME ids again (no progress, token 2)
                return MessagePage(
                    stubs=(
                        MessageStub(id="aa11", thread_id="t1"),
                        MessageStub(id="aa22", thread_id="t2"),
                    ),
                    next_page_token="2",
                )
            # page 3: new ids, no more pages
            return MessagePage(
                stubs=(
                    MessageStub(id="aa33", thread_id="t3"),
                    MessageStub(id="aa44", thread_id="t4"),
                ),
                next_page_token=None,
            )

    fake = RepeatingClient()
    import gmail_labeler.cli as cli

    monkeypatch.setattr(cli, "_shadow_client", lambda cfg_: fake)
    monkeypatch.setattr(
        cli, "_shadow_model_factory", _shadow_factory(_backlog_scores())
    )
    result = runner.invoke(
        app,
        [
            "shadow",
            "run",
            "--config",
            str(_shadow_cfg(tmp_path)),
            "--bands",
            str(tmp_path / "bands.json"),
            "--graduation",
            str(tmp_path / "graduation.json"),
        ],
    )
    assert result.exit_code == 0, result.stderr
    # the no-progress break stops at page 2 (page 2 repeats page 1's
    # ids — continuing would loop forever), so 2 list_messages calls and
    # the 2 deduped messages scored (page 3 is never reached)
    list_calls = [c for c in fake.calls if c.startswith("list_messages")]
    assert len(list_calls) == 2
    assert "scored=2" in result.stderr
    # no duplicate rows (aa11/aa22 appear once each — 2 x 2 labels)
    assert _db_row_count(tmp_path / "gmail-labeler.db") == 4


def _db_row_count(db_path: Path) -> int:
    import sqlite3

    if not db_path.is_file():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM shadow_results").fetchone()[0]
    finally:
        conn.close()


def test_shadow_run_usage_refusals_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # usage refusals up front (exit 2, before any corpus-scale work):
    # empty --bands/--graduation, bad --since, --limit <= 0, empty
    # --query
    _prepare(tmp_path, monkeypatch)
    bands = tmp_path / "bands.json"
    graduation = tmp_path / "graduation.json"
    import gmail_labeler.cli as cli

    def _invoke(*extra: str) -> Result:
        fake = FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
        monkeypatch.setattr(cli, "_shadow_client", lambda cfg_: fake)
        monkeypatch.setattr(
            cli, "_shadow_model_factory", _shadow_factory(_backlog_scores())
        )
        return runner.invoke(
            app,
            [
                "shadow",
                "run",
                "--config",
                str(_shadow_cfg(tmp_path)),
                "--bands",
                str(bands),
                "--graduation",
                str(graduation),
                *extra,
            ],
        )

    r = _invoke("--bands", " ")
    assert r.exit_code == 2
    assert "non-empty" in r.stderr
    r = _invoke("--graduation", "   ")
    assert r.exit_code == 2
    assert "non-empty" in r.stderr
    r = _invoke("--since", "not-a-date")
    assert r.exit_code == 2
    assert "ISO date" in r.stderr
    r = _invoke("--limit", "0")
    assert r.exit_code == 2
    r = _invoke("--query", "  ")
    assert r.exit_code == 2
    assert "non-empty" in r.stderr


def test_shadow_run_auth_gate_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the auth gate (exit 4 precedent): a not-signed-in run refuses
    # BEFORE any corpus-scale work
    _prepare(tmp_path, monkeypatch)
    import gmail_labeler.cli as cli
    from gmail_labeler.auth import AuthError

    def _boom(cfg):
        raise AuthError("no credentials_path set")

    monkeypatch.setattr(cli, "_shadow_client", _boom)
    result = runner.invoke(
        app,
        [
            "shadow",
            "run",
            "--config",
            str(_shadow_cfg(tmp_path)),
            "--bands",
            str(tmp_path / "bands.json"),
            "--graduation",
            str(tmp_path / "graduation.json"),
        ],
    )
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_shadow_run_bands_refused_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the FIRST real consumer of verify_bands: a bands artifact bound
    # to a DIFFERENT identity is refused, exit 5, naming the field
    _prepare(tmp_path, monkeypatch)
    # retrain with a LATER split date → the manifest's split identity
    # (and the artifact's split_date) differs
    assert (
        _run_train(tmp_path, monkeypatch, split_date="2026-02-01", force=True).exit_code
        == 0
    )
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "bands artifact refused" in result.stderr
    assert "split_date" in result.stderr  # the named disagreeing field


def test_shadow_run_graduation_refused_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the FIRST real consumer of verify_graduation: a graduation table
    # whose bands_content_hash binding disagrees is refused, exit 5,
    # naming the field. The bands artifact is kept VALID (matching the
    # checkpoint) so ONLY the table drifts — the bands gate passes and
    # the graduation gate is the one that fires (deleting the
    # verify_graduation call would leave this test red).
    from dataclasses import replace

    from gmail_labeler.graduation import GraduationTable

    _prepare(tmp_path, monkeypatch)
    table_path = tmp_path / "graduation.json"
    table = GraduationTable.from_json(table_path.read_text(encoding="utf-8"))
    # a lying bands binding: the table claims a bands artifact it was
    # not computed from (the bands_content_hash field is the binding)
    lying = replace(table, bands_content_hash="f" * 64)
    lying = replace(lying, content_hash=lying.compute_hash())
    table_path.write_text(lying.to_json(), encoding="utf-8")
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "graduation table refused" in result.stderr
    assert "bands_content_hash" in result.stderr  # the named field


def test_shadow_run_bands_absent_unreadable_unparseable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cause naming per #22's doctrine: absent vs unreadable (I/O
    # failure) vs unparseable — each named by its cause
    _prepare(tmp_path, monkeypatch)
    import gmail_labeler.cli as cli

    def _invoke(bands_path: Path) -> Result:
        fake = FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
        monkeypatch.setattr(cli, "_shadow_client", lambda cfg_: fake)
        monkeypatch.setattr(
            cli, "_shadow_model_factory", _shadow_factory(_backlog_scores())
        )
        return runner.invoke(
            app,
            [
                "shadow",
                "run",
                "--config",
                str(_shadow_cfg(tmp_path)),
                "--bands",
                str(bands_path),
                "--graduation",
                str(tmp_path / "graduation.json"),
            ],
        )

    # absent
    r = _invoke(tmp_path / "nope.json")
    assert r.exit_code == 5
    assert "no bands artifact" in r.stderr
    # unparseable (garbage bytes)
    garbage = tmp_path / "garbage.json"
    garbage.write_text("not json", encoding="utf-8")
    r = _invoke(garbage)
    assert r.exit_code == 5
    assert "does not parse as a Bands Artifact" in r.stderr
    # unreadable (I/O failure — the read itself fails): target ONLY the
    # bands path (a global read_text monkeypatch would break the
    # manifest read earlier in the run)
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{}", encoding="utf-8")
    orig_read_text = Path.read_text

    def _read_boom(self, *args, **kwargs):
        if self.name == "unreadable.json":
            raise OSError("simulated I/O failure")
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _read_boom)
    r = _invoke(unreadable)
    assert r.exit_code == 5
    assert "unreadable (I/O failure)" in r.stderr


def test_shadow_run_degraded_body_fetch_skips_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the documented posture choice: a message whose body can't be
    # fetched is a per-message SKIP (warning + count, the run
    # continues) — vs eval's whole-run refusal (shadow is a read-only
    # survey)
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
    fake.fail("get_message", GmailTransientError("network blip"))
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch, fake=fake)
    assert result.exit_code == 0, result.stderr
    assert "WARNING: skipped message aa11" in result.stderr
    assert "scored=3" in result.stderr
    assert "skipped=1" in result.stderr
    # the DB has 3 messages x 2 labels = 6 rows (aa11 absent)
    conn = sqlite3.connect(str(tmp_path / "gmail-labeler.db"))
    try:
        ids = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT message_id FROM shadow_results"
            ).fetchall()
        }
    finally:
        conn.close()
    assert ids == {"aa22", "aa33", "aa44"}


def test_shadow_run_transport_failure_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a transport failure on the backlog fetch is exit 5 (the
    # gmail_errors precedent — failure after retries)
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
    fake.fail("list_messages", GmailTransientError("network blip"))
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch, fake=fake)
    assert result.exit_code == 5
    assert "backlog fetch failed" in result.stderr


def test_shadow_run_bodies_absent_everywhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # invariant #5: message bodies never enter the DB rows, the
    # summary (stdout/stderr), or any log path — bodies are fetched
    # for scoring (in-memory only) and discarded
    import logging

    marker = "SECRET-BODY-MARKER"
    _prepare(tmp_path, monkeypatch)
    messages = [
        FakeMessage(
            id="aa11",
            eml=_eml("backlog one", f"{marker} body one"),
            label_ids=("INBOX", "L1"),
            thread_id="t1",
        ),
        FakeMessage(
            id="aa22",
            eml=_eml("backlog two", f"{marker} body two"),
            label_ids=("INBOX",),
            thread_id="t2",
        ),
    ]
    scores = {
        f"backlog one\n{marker} body one": (0.5, 0.95),
        f"backlog two\n{marker} body two": (0.85, 0.3),
    }
    caplog.set_level(logging.DEBUG)
    fake = FakeGmailClient(messages=messages, labels=_shadow_labels())
    result, _fake = _run_shadow(
        tmp_path, monkeypatch=monkeypatch, fake=fake, scores=scores
    )
    assert result.exit_code == 0, result.stderr
    # DB rows: ids, probabilities, labels, bands only
    conn = sqlite3.connect(str(tmp_path / "gmail-labeler.db"))
    try:
        rows = conn.execute(
            "SELECT message_id, label, probability, band FROM shadow_results"
        ).fetchall()
    finally:
        conn.close()
    for row in rows:
        assert marker not in str(row)
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert marker not in caplog.text


def test_shadow_run_json_single_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the --json contract: stdout is EXACTLY one parseable JSON object
    # (the human summary is suppressed — on stderr)
    import re
    from datetime import UTC, datetime

    import gmail_labeler.cli as cli

    _prepare(tmp_path, monkeypatch)
    # fold-4 item 2/3 pins: a fixed injectable clock so the payload's
    # run_ts and cutoff are assertable against the SAME clock read
    fixed_now = datetime(2026, 11, 1, 12, 30, 0, tzinfo=UTC)
    monkeypatch.setattr(cli, "_now", lambda: fixed_now)
    result, _fake = _run_shadow(tmp_path, "--json", monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)  # the whole stdout is one object
    assert payload["scored"] == 4
    assert payload["skipped"] == 0
    assert payload["proposed"] == 2
    assert payload["already_present"] == 2
    assert payload["not_applied"] == 4
    assert len(payload["rows"]) == 8
    assert set(payload["rows"][0]) == {
        "message_id",
        "label",
        "probability",
        "band",
        "present",
        "applied",
    }
    # in --json mode the payload IS the output — the human summary is
    # suppressed (stdout stays exactly one parseable JSON object)
    assert "shadow run: run=" not in result.stdout
    # fold-4 item 3 pin: the payload's cutoff is derived from the SAME
    # cutoff_dt that built the query — the payload window == the query
    # window (a mutant binding the payload to run_ts survives no suite
    # that asserts this)
    m = re.search(r"after:(\d+)$", payload["query"])
    assert m is not None
    query_epoch = int(m.group(1))
    assert payload["cutoff_epoch"] == query_epoch
    assert payload["cutoff"] == datetime.fromtimestamp(query_epoch, tz=UTC).isoformat()
    # fold-4 item 2 pin: run_ts goes through the SAME injectable clock
    # read (a mutant restoring datetime.now(UTC) for run_ts survives no
    # suite that asserts this)
    assert payload["run_ts"] == fixed_now.isoformat()


def test_shadow_run_json_query_override_omits_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fold-5 A1: when --query overrides the default window wholesale, the
    # --json payload must NOT emit cutoff_epoch/cutoff (they would
    # describe a window the query never used — no phantom window). The
    # default case (test_shadow_run_json_single_object) pins their
    # PRESENCE; this pin their ABSENCE under the override.
    # Mutant: payload emits cutoff_epoch/cutoff despite the query
    # override -> this test goes red.
    _prepare(tmp_path, monkeypatch)
    result, _fake = _run_shadow(
        tmp_path,
        "--json",
        "--query",
        "in:inbox from:example.com",
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    # the override query is what was used
    assert payload["query"] == "in:inbox from:example.com"
    # the window fields are ABSENT (the query replaced the window)
    assert "cutoff_epoch" not in payload
    assert "cutoff" not in payload


def test_shadow_run_json_since_window_keeps_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fold-6 item 4 (the pin deepseek proved missing): the guard is
    # query-is-None, NOT "default window only" — with --since (no
    # --query), the cutoff fields are PRESENT and describe the --since
    # window. Mutant: drop the fields when since is not None → this
    # test is the one that reddens.
    from datetime import UTC, datetime

    _prepare(tmp_path, monkeypatch)
    result, _fake = _run_shadow(
        tmp_path, "--json", "--since", "2026-10-01", monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    # the --since window was used: UTC midnight of the named day
    expected_epoch = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp())
    assert payload["query"] == f"in:inbox after:{expected_epoch}"
    # the cutoff fields are PRESENT and describe the --since cutoff
    assert payload["cutoff_epoch"] == expected_epoch
    assert payload["cutoff"] == datetime(2026, 10, 1, tzinfo=UTC).isoformat()


def test_shadow_run_zero_graduated_labels_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a run whose Graduation Table graduated NO labels produces zero
    # verdicts — it is refused (exit 5, nothing recorded), not reported
    # as a success that records nothing (a `shadow report` on an empty
    # DB would say "no runs recorded" — a silent loss of the run)
    _seed_corpus(tmp_path)
    assert _run_train(tmp_path, monkeypatch).exit_code == 0
    cfg = _shadow_cfg(tmp_path)
    bands = tmp_path / "bands.json"
    assert (
        _run_eval(
            tmp_path, "--bands", str(bands), monkeypatch=monkeypatch, cfg=cfg
        ).exit_code
        == 0
    )
    # min_examples = 999 → NO label graduates (a realistic early state)
    grad_cfg = _shadow_cfg(tmp_path)
    grad_cfg.write_text(
        grad_cfg.read_text(encoding="utf-8").replace(
            "[store]", "[graduation]\nmin_examples = 999\n[store]"
        ),
        encoding="utf-8",
    )
    graduation = tmp_path / "graduation.json"
    assert (
        _run_eval(
            tmp_path,
            "--graduation",
            str(graduation),
            "--confirm-graduation",
            monkeypatch=monkeypatch,
            cfg=grad_cfg,
        ).exit_code
        == 0
    )
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch, cfg=grad_cfg)
    assert result.exit_code == 5
    assert "no verdicts to record" in result.stderr
    # nothing was recorded
    assert not _db_has_rows(tmp_path / "gmail-labeler.db")


def test_shadow_run_no_checkpoint_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # no checkpoint (no manifest) → exit 5, named (run train first)
    _seed_corpus(tmp_path)
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    assert result.exit_code == 5
    assert "no verifiable split identity" in result.stderr


def test_shadow_run_empty_backlog_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an empty backlog (no messages in the window) → exit 5, named
    # (nothing recorded — a run that scores nothing is incomplete)
    _prepare(tmp_path, monkeypatch)
    fake = FakeGmailClient(messages=[], labels=_shadow_labels())
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch, fake=fake)
    assert result.exit_code == 5
    assert "no messages to score" in result.stderr


def test_shadow_run_graduated_system_label_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # disposition 7: a Graduation Table that GRADUATES a System Label
    # (INBOX — the fake declares LabelInfo("INBOX","INBOX","system"))
    # → shadow run exit 5 "would-apply refused". This pins the CALL
    # SITE passes a real NON-EMPTY system set to would_apply (kwarg
    # presence alone never proves non-emptiness — ds).
    from datetime import UTC, datetime

    from gmail_labeler.corpus_record import CorpusRecord, body_hash
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.dataset_build import TrainingExample
    from gmail_labeler.eval import eval_metrics
    from gmail_labeler.graduation import (
        GateTargets,
        Graduation,
        GraduationEntry,
        build_table,
    )
    from gmail_labeler.model import MANIFEST_FILENAME
    from gmail_labeler.thresholds import BandsArtifact

    # a corpus whose vocabulary includes the System Label NAME "INBOX"
    # (a user label named INBOX — the checkpoint vocabulary carries it)
    store = CorpusStore(tmp_path / "corpus")
    store.enforce_modes()
    now = datetime.now(UTC)
    for i in range(12):
        days_ago = 400 - i if i < 6 else 10 + i
        rec = CorpusRecord(
            id=f"{i:016x}",
            date=now - timedelta(days=days_ago),
            subject=f"subject {i}",
            sender="shop@x.z",
            recipients=("me@x.y",),
            labels=("INBOX", "Shipping") if i % 2 == 0 else ("Shipping",),
            body_hash=body_hash(f"body {i}"),
            source="api",
            date_provenance="internal_date",
        )
        store.put(rec, f"body {i}")
    for name, payload in (("filters.json", "[]"), ("labels.json", "{}")):
        f = store.root / name
        f.write_text(payload, encoding="utf-8")
        os.chmod(f, 0o600)

    # train the checkpoint (vocabulary = ("INBOX", "Shipping"))
    cfg = _shadow_cfg(tmp_path, labels='managed = ["INBOX", "Shipping"]')
    assert _run_train(tmp_path, monkeypatch, cfg=cfg).exit_code == 0
    # calibrate the bands
    bands_path = tmp_path / "bands.json"
    assert (
        _run_eval(
            tmp_path, "--bands", str(bands_path), monkeypatch=monkeypatch, cfg=cfg
        ).exit_code
        == 0
    )
    artifact = BandsArtifact.from_json(bands_path.read_text(encoding="utf-8"))
    # the checkpoint identity (for the graduation table binding): the
    # manifest carries split_date + content_hash; the sidecar carries
    # corpus_fingerprint
    manifest = json.loads(
        (tmp_path / "ckpt" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    sidecar = json.loads(
        (tmp_path / "ckpt" / "sidecar.json").read_text(encoding="utf-8")
    )
    split_date = datetime.fromisoformat(manifest["split_date"])
    # construct a Graduation that GRADUATES INBOX (a System Label)
    examples = [
        TrainingExample(
            id=f"{i:016x}",
            date=now,
            labels=(),
            weight=1.0,
            date_provenance="internal_date",
        )
        for i in range(6)
    ]
    table = eval_metrics(
        examples,
        {f"{i:016x}": (0.9, 0.9) for i in range(6)},
        vocabulary=("INBOX", "Shipping"),
    )
    grad = Graduation(
        vocabulary=("INBOX", "Shipping"),
        candidate_thresholds=table.thresholds,
        labels=(
            GraduationEntry(
                label="INBOX",
                examples=6,
                precision_at_high=1.0,
                high=0.9,
                med=0.8,
                min_examples=1,
                min_precision=0.9,
                graduated=True,
                reasons=(),
            ),
            GraduationEntry(
                label="Shipping",
                examples=6,
                precision_at_high=1.0,
                high=0.9,
                med=0.8,
                min_examples=1,
                min_precision=0.9,
                graduated=True,
                reasons=(),
            ),
        ),
        targets=(
            ("INBOX", GateTargets(min_examples=1, min_precision=0.9)),
            ("Shipping", GateTargets(min_examples=1, min_precision=0.9)),
        ),
        test_examples=6,
        default_targets=GateTargets(min_examples=1, min_precision=0.9),
        bands_content_hash=artifact.content_hash,
    )
    graduation_table = build_table(
        grad,
        vocabulary=("INBOX", "Shipping"),
        corpus_fingerprint=sidecar["corpus_fingerprint"],
        split_manifest_hash=manifest["content_hash"],
        encoder_id="tiny-encoder",
        split_date=split_date,
    )
    graduation_path = tmp_path / "graduation.json"
    graduation_path.write_text(graduation_table.to_json(), encoding="utf-8")
    # shadow run → would_apply refuses the graduated System Label (INBOX)
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch, cfg=cfg)
    assert result.exit_code == 5
    assert "would-apply refused" in result.stderr
    assert "INBOX" in result.stderr


def test_shadow_run_default_window_cli_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fold-4 item 2: the CLI-level pin for the REAL shipped default —
    # monkeypatch cli._now to a fixed aware datetime, run shadow run
    # WITHOUT --since, and assert the FakeGmailClient-recorded
    # list_messages q is EXACTLY in:inbox after:<int((fixed-90d).
    # timestamp())>. This kills the fold-2 rewire mutant (the dead-
    # branch bug where the default passed a non-None since_date, so the
    # cutoff was UTC-midnight(today-90d) instead of now-90d time-kept).
    from datetime import UTC, datetime

    import gmail_labeler.cli as cli

    _prepare(tmp_path, monkeypatch)
    fixed_now = datetime(2026, 11, 1, 12, 30, 0, tzinfo=UTC)
    # fold-5 A4: counting pin for the ONE-read claim — cli._now is called
    # EXACTLY once per shadow run (window math + run_ts share the read)
    now_calls = 0

    def _counting_now() -> datetime:
        nonlocal now_calls
        now_calls += 1
        return fixed_now

    monkeypatch.setattr(cli, "_now", _counting_now)
    fake = FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
    monkeypatch.setattr(cli, "_shadow_client", lambda cfg_: fake)
    monkeypatch.setattr(
        cli, "_shadow_model_factory", _shadow_factory(_backlog_scores())
    )
    result = runner.invoke(
        app,
        [
            "shadow",
            "run",
            "--config",
            str(_shadow_cfg(tmp_path)),
            "--bands",
            str(tmp_path / "bands.json"),
            "--graduation",
            str(tmp_path / "graduation.json"),
        ],
    )
    assert result.exit_code == 0, result.stderr
    expected_epoch = int((fixed_now - timedelta(days=90)).timestamp())
    list_calls = [c for c in fake.calls if c.startswith("list_messages")]
    assert list_calls[0] == f"list_messages:in:inbox after:{expected_epoch}:"
    # fold-5 A4: the ONE-read claim — cli._now is called EXACTLY once
    # (a mutant reading the clock twice, e.g. run_ts = _now(), goes red)
    assert now_calls == 1


def test_shadow_run_fail_fast_db_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # disposition 4: the state DB is validated FAIL-FAST — a corrupt /
    # foreign / unreadable DB is exit 5 BEFORE any network work (the
    # fake's call log is EMPTY — no list_labels / list_messages /
    # get_message was paid for)
    _prepare(tmp_path, monkeypatch)
    # a corrupt (non-SQLite) DB at the --db path
    corrupt_db = tmp_path / "corrupt.db"
    corrupt_db.write_bytes(b"not a sqlite file")
    fake = FakeGmailClient(messages=_backlog_messages(), labels=_shadow_labels())
    import gmail_labeler.cli as cli

    monkeypatch.setattr(cli, "_shadow_client", lambda cfg_: fake)
    monkeypatch.setattr(
        cli, "_shadow_model_factory", _shadow_factory(_backlog_scores())
    )
    result = runner.invoke(
        app,
        [
            "shadow",
            "run",
            "--config",
            str(_shadow_cfg(tmp_path)),
            "--bands",
            str(tmp_path / "bands.json"),
            "--graduation",
            str(tmp_path / "graduation.json"),
            "--db",
            str(corrupt_db),
        ],
    )
    assert result.exit_code == 5
    assert "state DB unusable" in result.stderr
    assert "corrupt (not a SQLite file)" in result.stderr
    assert fake.calls == []  # NO network work was done


def test_shadow_run_db_flag_overrides_store_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # D3: the --db flag is a usage-level override of [store] db_path —
    # the rows land at the --db path, not the config default
    _prepare(tmp_path, monkeypatch)
    custom_db = tmp_path / "custom" / "state.db"
    result, _fake = _run_shadow(
        tmp_path, "--db", str(custom_db), monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr
    assert custom_db.is_file()
    assert _db_has_rows(custom_db)
    # the config default was NOT written
    assert not (tmp_path / "gmail-labeler.db").exists()


def test_shadow_run_db_flag_trailing_space_is_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fold-4 item 4: --db "<path>.db " (trailing space) → the DB lands at
    # the CLEAN stripped path (no space-suffixed file) — proves the strip
    _prepare(tmp_path, monkeypatch)
    clean_db = tmp_path / "state.db"
    result, _fake = _run_shadow(
        tmp_path, "--db", str(clean_db) + " ", monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr
    assert clean_db.is_file()
    assert _db_has_rows(clean_db)
    # no space-suffixed file
    assert not (tmp_path / "state.db ").exists()


def test_shadow_run_db_flag_expanduser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fold-4 item 4: --db '~/state.db' with HOME=tmp_path → lands under
    # tmp (the ~ is expanded, matching the config path fields)
    _prepare(tmp_path, monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    result, _fake = _run_shadow(tmp_path, "--db", "~/state.db", monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.stderr
    assert (tmp_path / "state.db").is_file()
    assert _db_has_rows(tmp_path / "state.db")


def test_shadow_run_db_flag_containment_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # disposition 4: --db containment inheritance — a --db path inside
    # the checkpoint dir OR the corpus store is a usage refusal (exit 2),
    # and the refusal names the SOURCE flag (--db, not [store] db_path)
    _prepare(tmp_path, monkeypatch)
    result, _fake = _run_shadow(
        tmp_path,
        "--db",
        str(tmp_path / "ckpt" / "state.db"),
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 2
    assert "--db PATH" in result.stderr
    assert "lives inside the checkpoint directory" in result.stderr
    result2, _fake2 = _run_shadow(
        tmp_path,
        "--db",
        str(tmp_path / "corpus" / "state.db"),
        monkeypatch=monkeypatch,
    )
    assert result2.exit_code == 2
    assert "--db PATH" in result2.stderr
    assert "lives inside the corpus store" in result2.stderr


def test_shadow_run_db_flag_empty_is_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an empty --db is a usage refusal (exit 2)
    _prepare(tmp_path, monkeypatch)
    result, _fake = _run_shadow(tmp_path, "--db", "  ", monkeypatch=monkeypatch)
    assert result.exit_code == 2
    assert "--db PATH must be non-empty" in result.stderr


def test_shadow_run_store_db_path_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the [store] db_path config: a relative path anchors to the config
    # dir (the corpus/model dir rule)
    _prepare(tmp_path, monkeypatch)
    cfg = _shadow_cfg(tmp_path, store='db_path = "state/custom.db"\n')
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch, cfg=cfg)
    assert result.exit_code == 0, result.stderr
    assert (tmp_path / "state" / "custom.db").is_file()
    assert not (tmp_path / "gmail-labeler.db").exists()


@pytest.mark.live
@pytest.mark.smoke
def test_live_shadow_run_real_checkpoint(tmp_path: Path) -> None:
    """The Phase 1 exit demo (opt-in, NEVER run in CI): a READ-ONLY
    real-checkpoint shadow run against the live mailbox.

    This test NEVER trains and NEVER confirms a Graduation Table (those
    stay README-documented HUMAN actions — a test must not run
    ``train --force`` against the operator's real checkpoint, nor pass
    ``--confirm-graduation``). It requires the operator to have ALREADY
    done the human steps (train, ``eval --bands``, ``eval --graduation
    --confirm-graduation``) and then runs ``shadow run`` + ``shadow
    report`` against the EXISTING checkpoint + artifacts.

    It is READ-ONLY against Gmail (list_labels / list_messages /
    get_message only — no write), costs API quota, and writes ONLY the
    tmp-redirected state DB (``--db <tmp>``). It skips (with a reason)
    when the config / corpus / checkpoint / bands / graduation artifacts
    are absent — it never fails for "not set up".

    Opt-in:

        uv run pytest -m "live and smoke" \\
            tests/test_shadow_cli.py::test_live_shadow_run_real_checkpoint

    (needs the ``train`` extra for the real encoder, a signed-in Gmail
    account, a corpus, and the human-step artifacts in the cwd).
    """
    from gmail_labeler.config import ConfigError, find_config, load_config
    from gmail_labeler.model import MANIFEST_FILENAME

    # the config must be discoverable (the operator's real install)
    try:
        config_path = find_config()
    except ConfigError as e:
        pytest.skip(f"no config.toml discoverable: {e}")
    try:
        cfg = load_config(str(config_path))
    except ConfigError as e:
        pytest.skip(f"config invalid: {e}")
    # the corpus must exist and have records
    corpus_root = Path(cfg.corpus.dir)
    if not (corpus_root / "records.jsonl").is_file():
        pytest.skip(f"corpus dir {corpus_root} has no records.jsonl")
    # the checkpoint manifest must exist (the human train step was done)
    manifest = Path(cfg.model.checkpoint_dir) / MANIFEST_FILENAME
    if not manifest.is_file():
        pytest.skip(f"checkpoint manifest {manifest} absent (run train first)")
    # the bands + graduation artifacts must exist (the human eval steps)
    cwd = Path.cwd()
    bands = cwd / "bands.json"
    graduation = cwd / "graduation.json"
    if not bands.is_file() or not graduation.is_file():
        pytest.skip(
            "bands.json / graduation.json absent in cwd (run the human "
            "eval --bands / eval --graduation --confirm-graduation steps)"
        )
    # the state DB is redirected to a tmp path (the test writes ONLY here)
    db = tmp_path / "state.db"
    # the shadow run over a SMALL Backlog window (the demo — not a full
    # mailbox scan). READ-ONLY against Gmail; the real encoder loads the
    # existing checkpoint (no train).
    run = runner.invoke(
        app,
        [
            "shadow",
            "run",
            "--config",
            str(config_path),
            "--bands",
            str(bands),
            "--graduation",
            str(graduation),
            "--db",
            str(db),
            "--limit",
            "5",
        ],
    )
    # disposition 6-cli: the test SKIPS (not fails) on setup-state
    # stderr per its docstring promise — a mailbox whose Graduation
    # Table graduated no labels ("no verdicts to record"), or whose
    # bands/graduation artifacts don't match the checkpoint (an
    # identity refusal), is a "not set up for this mailbox" state, not
    # a test failure
    if run.exit_code != 0:
        if "no verdicts to record" in run.stderr:
            pytest.skip(
                "the Graduation Table graduated no labels for this "
                "mailbox (no verdicts to record)"
            )
        if "refused" in run.stderr:
            pytest.skip(
                "the bands/graduation artifacts do not match this "
                f"checkpoint (identity refusal): {run.stderr.strip()}"
            )
        pytest.fail(f"shadow run failed unexpectedly: {run.stderr}")
    assert db.is_file()  # the tmp DB was written (the only write)
    # the report (the Phase 1 exit review) — READ-ONLY
    report = runner.invoke(
        app,
        [
            "shadow",
            "report",
            "--config",
            str(config_path),
            "--db",
            str(db),
        ],
    )
    assert report.exit_code == 0, report.stderr
    # no vacuous assert: the report must have real content (a header +
    # at least one outcome line or the per-label block)
    assert "shadow report: run=" in report.stdout
    assert any(
        token in report.stdout
        for token in ("ADDED", "ALREADY-PRESENT", "NOT-APPLIED", "per-label:")
    )
