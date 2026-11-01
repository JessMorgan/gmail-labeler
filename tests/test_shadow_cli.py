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
from datetime import UTC, datetime, timedelta
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


def test_shadow_run_default_query_is_inbox_newer_than(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the default Backlog query: in:inbox newer_than:YYYY/MM/DD (the
    # --since window, default 90 days)
    _prepare(tmp_path, monkeypatch)
    result, fake = _run_shadow(
        tmp_path, "--since", "2026-10-01", monkeypatch=monkeypatch
    )
    assert result.exit_code == 0, result.stderr
    list_calls = [c for c in fake.calls if c.startswith("list_messages")]
    assert list_calls[0] == "list_messages:in:inbox newer_than:2026/10/01:"


def test_shadow_run_query_escape_hatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --query replaces the default wholesale (the operator's query)
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
    # bound to a DIFFERENT identity is refused, exit 5, naming the
    # field
    _prepare(tmp_path, monkeypatch)
    # retrain with a LATER split date → the table's split identity
    # differs
    assert (
        _run_train(tmp_path, monkeypatch, split_date="2026-02-01", force=True).exit_code
        == 0
    )
    result, _fake = _run_shadow(tmp_path, monkeypatch=monkeypatch)
    # the bands artifact is checked FIRST (it also drifts) — the
    # refusal names the bands field
    assert result.exit_code == 5
    assert "refused" in result.stderr


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
    _prepare(tmp_path, monkeypatch)
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
