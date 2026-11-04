from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from gmail_labeler.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("auth", "dataset"):
        assert cmd in result.output


def test_dataset_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["dataset", "--help"])
    assert result.exit_code == 0
    for cmd in ("import-gmail", "import-gyb", "analyze"):
        assert cmd in result.output


def test_usage_error_stays_exit_2() -> None:
    """The 2/3 split is the contract: unknown commands keep exit 2."""
    result = runner.invoke(app, ["bogus-command"])
    assert result.exit_code == 2


def test_old_export_command_is_usage_error() -> None:
    """Hard cut (ticket #68): `dataset export` is gone with no compat alias —
    the old name stays a plain usage error (exit 2), never a hidden command."""
    result = runner.invoke(app, ["dataset", "export"])
    assert result.exit_code == 2


def test_auth_status_reports_and_exits_4_when_unsigned(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[gmail]\ntoken_backend = "file"\ntoken_file = "{tmp_path / "tok.json"}"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["auth", "--config", str(cfg)])
    assert result.exit_code == 4
    assert "NOT SIGNED IN" in result.output
    assert "auth login" in result.output


def test_auth_config_error_exits_4(tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    result = runner.invoke(app, ["auth", "--config", str(missing)])
    assert result.exit_code == 4
    assert "config error" in result.stderr


def test_auth_login_failures_exit_4_no_traceback(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[gmail]\ntoken_backend = "file"\ntoken_file = "{tmp_path / "tok.json"}"\n',
        encoding="utf-8",
    )
    # (credentials_path unset → AuthError → 4, and NO Rich traceback which
    # would exit 1 and echo source lines)
    result = runner.invoke(app, ["auth", "login", "--config", str(cfg)])
    assert result.exit_code == 4
    assert "Traceback" not in result.output
    assert "auth:" in result.stderr or "credentials_path" in (
        result.stderr + result.output
    )


def _import_gmail_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[gmail]\ntoken_backend = "file"\n'
        f'token_file = "{tmp_path / "tok.json"}"\n'
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n',
        encoding="utf-8",
    )
    return cfg


def test_dataset_import_gmail_writes_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
    from gmail_labeler.gmail_client import FilterInfo, LabelInfo

    fake = FakeGmailClient(
        messages=[
            FakeMessage(
                id="aaa",
                eml=b"Subject: Hi\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nbody",
                label_ids=("INBOX", "L1"),
                thread_id="t",
                internal_date_ms=1761818400000,
            )
        ],
        labels=[LabelInfo("L1", "Shipping", "user")],
        filters=[FilterInfo(id="f1", criteria={"from": "shipper@x.com"}, action={})],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "import-gmail:" in result.stdout  # success-line prefix follows the name
    assert "written=1" in result.output
    assert (tmp_path / "corpus" / "records.jsonl").is_file()
    import json as _json

    sidecar = _json.loads(
        (tmp_path / "corpus" / "filters.json").read_text(encoding="utf-8")
    )
    assert sidecar[0]["id"] == "f1"  # offline attribution snapshot
    assert sidecar[0]["criteria"] == {"from": "shipper@x.com"}


def test_dataset_import_gmail_incomplete_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gmail_labeler import cli
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage

    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=b"no-date-here")])
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "failed=1" in result.stdout


def test_dataset_import_gmail_auth_failure_exits_4(tmp_path: Path) -> None:
    # token_file has no token -> credentials() AuthError -> 4 (not 5, not 1)
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 4
    assert "Traceback" not in result.output


def test_import_gmail_sidecar_is_0600_and_query_flows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
    from gmail_labeler.gmail_client import FilterInfo, LabelInfo

    fake = FakeGmailClient(
        messages=[
            FakeMessage(
                id="aaa",
                eml=b"Subject: x\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nb",
            )
        ],
        labels=[LabelInfo("L1", "Shipping", "user")],
        filters=[FilterInfo(id="f1", criteria={"from": "a@b.c"}, action={})],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    result = runner.invoke(
        app,
        [
            "dataset",
            "import-gmail",
            "--config",
            str(_import_gmail_cfg(tmp_path)),
            "--query",
            "has:attachment",
        ],
    )
    assert result.exit_code == 0
    sidecar = tmp_path / "corpus" / "filters.json"
    assert sidecar.stat().st_mode & 0o077 == 0  # 0600 at creation, not after
    bridge = tmp_path / "corpus" / "labels.json"  # id->name for attribution
    assert bridge.is_file() and bridge.stat().st_mode & 0o077 == 0
    import json as _json2

    assert _json2.loads(bridge.read_text(encoding="utf-8")) == {"L1": "Shipping"}
    assert any("list_messages:has:attachment:" in c for c in fake.calls)


def test_import_gmail_births_cursor_0600_and_second_run_preserves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the ticket's end-to-end demonstration against the fake: import, import
    # again, and the first lower bound survives untouched — at the CLI seam,
    # with the secret-handling mode asserted right after the run
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage

    fake = FakeGmailClient(
        messages=[
            FakeMessage(
                id="aaa",
                eml=b"Subject: x\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nb",
                history_id=900,
            )
        ],
        current_history_id=900,
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = str(_import_gmail_cfg(tmp_path))
    result = runner.invoke(app, ["dataset", "import-gmail", "--config", cfg])
    assert result.exit_code == 0
    cursor = tmp_path / "corpus" / "sync-cursor.json"
    assert cursor.is_file()
    assert cursor.stat().st_mode & 0o077 == 0  # protected state from byte one
    born = cursor.read_bytes()
    assert b'"history_id": 900' in born

    fake.current_history_id = 950  # the mailbox moves on between imports
    result = runner.invoke(app, ["dataset", "import-gmail", "--config", cfg])
    assert result.exit_code == 0
    assert cursor.read_bytes() == born  # birth-once, byte-for-byte (AC3)


def test_import_gmail_invalid_cursor_exits_5_not_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the typed failure surface at the CLI seam: a corrupt ledger is an
    # operation failure (exit 5, "import failed"), never a raw traceback
    # (exit 1), and the garbage stays on disk untouched for a human
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage

    fake = FakeGmailClient(
        messages=[FakeMessage(id="aaa", eml=b"Subject: x\r\n\r\nb", history_id=900)]
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    garbage = "{ not json"
    (corpus / "sync-cursor.json").write_text(garbage, encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "import failed" in result.output
    assert "Traceback" not in result.output
    assert (corpus / "sync-cursor.json").read_text(encoding="utf-8") == garbage


def test_import_gmail_settings_blip_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
    from gmail_labeler.gmail_errors import classify_status

    fake = FakeGmailClient(
        messages=[
            FakeMessage(
                id="aaa",
                eml=b"Subject: x\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nb",
            )
        ]
    )
    fake.fail("list_labels", classify_status(503))  # one transient on bootstrap
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 0  # settings are not a single-blip kill switch
    assert "written=1" in result.output


def test_import_gmail_auth_failure_exits_4_not_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.auth import AuthError

    def boom(cfg: object) -> object:
        raise AuthError("not signed in — run `gmail-labeler auth login`")

    monkeypatch.setattr(cli, "_import_gmail_client", boom)
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 4
    assert "auth" in (result.stdout + result.stderr)


def test_import_gmail_bootstrap_auth_failure_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # deepseek MINOR: the now-live `except GmailAuthError` clauses need
    # CLI-level pins (engine pins cannot catch a routing regression)
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient
    from gmail_labeler.gmail_errors import classify_status

    fake = FakeGmailClient()
    fake.fail("list_labels", classify_status(401))
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_import_gmail_settings_exhaustion_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient
    from gmail_labeler.gmail_errors import classify_status

    fake = FakeGmailClient()
    for _ in range(8):  # exhaust every attempt (MAX_ATTEMPTS is smaller)
        fake.fail("list_labels", classify_status(503))
    monkeypatch.setattr(
        "gmail_labeler.gmail_import.time.sleep", lambda _s: None
    )  # 5x503 exhaustion would pay ~15 s real backoff
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "label map unavailable after retries" in result.stderr
    assert "import failed:" in result.stderr  # reworded prefix (ticket #68)


def test_import_gmail_retried_counts_settings_blips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # deepseek NIT: the retried= merge (cli.py) must be asserted, not just
    # implied by the exit-0 blip test
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
    from gmail_labeler.gmail_errors import classify_status

    fake = FakeGmailClient(
        messages=[
            FakeMessage(
                id="aaa",
                eml=b"Subject: x\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nb",
            )
        ]
    )
    fake.fail("list_filters", classify_status(503))
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "retried=1" in result.stdout


def test_import_gmail_midrun_auth_failure_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # deepseek seal MINOR: the mid-run `except GmailAuthError` clause is the
    # last unpinned link of the exit-code contract (bootstrap half is
    # pinned above); deleting it today silently regresses 4 -> 5
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
    from gmail_labeler.gmail_errors import classify_status

    fake = FakeGmailClient(
        messages=[
            FakeMessage(
                id="aaa",
                eml=b"Subject: x\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nb",
            )
        ]
    )
    fake.fail("get_message", classify_status(401))
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_import_gmail_filters_exhaustion_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # deepseek seal NIT: the list_filters exhaustion branch is its own
    # message and its own line of code — pin it separately
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient
    from gmail_labeler.gmail_errors import classify_status

    fake = FakeGmailClient()
    for _ in range(8):
        fake.fail("list_filters", classify_status(503))
    monkeypatch.setattr(
        "gmail_labeler.gmail_import.time.sleep", lambda _s: None
    )  # 5x503 exhaustion would pay ~15 s real backoff
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "import-gmail", "--config", str(_import_gmail_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "filter snapshot unavailable after retries" in result.stderr


def _gyb_fixture(tmp_path: Path) -> Path:
    import sqlite3

    gyb = tmp_path / "gyb"
    gyb.mkdir()
    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.executescript(
        "CREATE TABLE messages (message_num INTEGER PRIMARY KEY,"
        " message_filename TEXT, message_internaldate TEXT);"
        "CREATE TABLE labels (message_num INTEGER, label TEXT);"
        "CREATE TABLE uids (message_num INTEGER PRIMARY KEY, uid TEXT);"
        "INSERT INTO messages VALUES (1, '2026/7/15/abcd.eml',"
        " '2026-07-15 10:00:00');"
        "INSERT INTO uids VALUES (1, 'abcd');"
        "INSERT INTO labels VALUES (1, 'INBOX');"
        "INSERT INTO labels VALUES (1, 'Work');"
    )
    con.commit()
    con.close()
    (gyb / "2026/7/15").mkdir(parents=True)
    (gyb / "2026/7/15" / "abcd.eml").write_bytes(
        b"Subject: x\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nhi"
    )
    return gyb


def test_import_gyb_requires_path(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[corpus]\ndir = "{tmp_path / "corpus"}"\n', encoding="utf-8")
    result = runner.invoke(app, ["dataset", "import-gyb", "--config", str(cfg)])
    assert result.exit_code == 5
    assert "no backup path" in result.stderr


def test_import_gyb_happy_path(tmp_path: Path) -> None:
    gyb = _gyb_fixture(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n[gyb]\npath = "{gyb}"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["dataset", "import-gyb", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "written=1" in result.stdout
    records = (tmp_path / "corpus" / "records.jsonl").read_text(encoding="utf-8")
    assert '"labels": ["Work"]' in records and "INBOX" not in records
    # corpus files created secret-tight (SECURITY.md posture)
    assert (tmp_path / "corpus" / "records.jsonl").stat().st_mode & 0o077 == 0


def test_import_gyb_missing_db_exits_5(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n'
        f'[gyb]\npath = "{tmp_path / "nothing"}"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["dataset", "import-gyb", "--config", str(cfg)])
    assert result.exit_code == 5
    assert "msg-db.sqlite not found" in result.stderr


def test_import_gyb_corrupt_db_exits_5_not_traceback(tmp_path: Path) -> None:
    gyb = tmp_path / "gyb"
    gyb.mkdir()
    (gyb / "msg-db.sqlite").write_bytes(b"garbage not sqlite")
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n[gyb]\npath = "{gyb}"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["dataset", "import-gyb", "--config", str(cfg)])
    assert result.exit_code == 5
    assert "import failed" in result.stderr
    assert "resumes" in result.stderr  # honest partial-state echo
    assert "Traceback" not in (result.stdout + result.stderr)


def test_import_gyb_flags_flow_through(tmp_path: Path) -> None:
    gyb = _gyb_fixture(tmp_path)
    (gyb / "2026/7/15" / "beef.eml").write_bytes(
        b"Subject: y\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nsecond"
    )
    import sqlite3

    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.execute(
        "INSERT INTO messages VALUES (2, '2026/7/15/beef.eml', '2026-07-16 10:00:00')"
    )
    con.execute("INSERT INTO uids VALUES (2, 'beef')")
    con.execute("INSERT INTO labels VALUES (2, 'INKASSIE')")
    con.execute("INSERT INTO labels VALUES (2, 'Friends')")
    con.commit()
    con.close()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[corpus]\ndir = "{tmp_path / "corpus"}"\n', encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "dataset",
            "import-gyb",
            "--config",
            str(cfg),
            "--path",
            str(gyb),
            "--system-labels",
            "INKASSIE, ,Friends",
            "--max-messages",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "scanned=2" in result.stdout  # max-messages honored
    records = (tmp_path / "corpus" / "records.jsonl").read_text(encoding="utf-8")
    assert "INKASSIE" not in records and "Friends" not in records


def test_import_gyb_empty_path_flag_is_error(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[corpus]\ndir = "{tmp_path / "corpus"}"\n', encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "import-gyb", "--config", str(cfg), "--path", " "]
    )
    assert result.exit_code == 5
    assert "empty" in result.stderr


def test_import_gyb_abort_reports_partial_counts(tmp_path: Path) -> None:
    gyb = _gyb_fixture(tmp_path)
    import sqlite3

    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.execute("INSERT INTO messages VALUES (2, '2026/7/16/beef.eml', 'garbage-date')")
    con.execute("INSERT INTO uids VALUES (2, 'beef')")
    con.commit()
    con.close()
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n[gyb]\npath = "{gyb}"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["dataset", "import-gyb", "--config", str(cfg)])
    assert result.exit_code == 5
    # good row (sorts before 'garbage') was durably imported and IS reported
    assert "1 record(s) durably imported so far" in result.stderr


def test_import_gyb_help_documents_comma_limit(tmp_path: Path) -> None:
    # the rendered help truncates at panel width, so pin the DECLARED help
    # string on the OptionInfo (discoverability of the comma limitation)
    import gmail_labeler.cli as cli

    opt = inspect.signature(cli.dataset_import_gyb).parameters["system_labels"]
    assert "containing commas cannot be expressed" in opt.default.help


def test_import_gyb_date_conflict_warning(tmp_path: Path) -> None:
    # the loud surface for >1-day folder/stamp disagreement (deepseek MAJOR)
    gyb = _gyb_fixture(tmp_path)
    import sqlite3

    con = sqlite3.connect(gyb / "msg-db.sqlite")
    con.execute(
        "UPDATE messages SET message_internaldate = '2027-01-01 10:00:00'"
        " WHERE message_num = 1"  # folder says 2026/7/15: > 1 day apart
    )
    con.commit()
    con.close()
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n[gyb]\npath = "{gyb}"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["dataset", "import-gyb", "--config", str(cfg)])
    assert result.exit_code == 0  # imported with the raw stamp — not silently
    assert "MORE than a day" in result.stderr


# ---- dataset analyze ----


def _seed_corpus(
    tmp_path: Path, *, n: int = 60, label: str = "Shipping", sender: str = "shop@x.z"
) -> Path:
    from datetime import UTC, datetime, timedelta

    from gmail_labeler.corpus_record import CorpusRecord, body_hash
    from gmail_labeler.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    store.enforce_modes()
    now = datetime.now(UTC)
    base = sum(1 for _ in store.iter_records())
    for i in range(n):
        rec = CorpusRecord(
            id=f"{base + i:08x}",
            date=now - timedelta(days=1),
            subject="hi",
            sender=sender,
            recipients=("me@x.y",),
            labels=(label,),
            body_hash=body_hash(f"body {base + i}"),
            source="api",
        )
        store.put(rec, f"body {base + i}")
    return store.root


def _analyze_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[corpus]\ndir = "{tmp_path / "corpus"}"\n', encoding="utf-8")
    return cfg


def test_analyze_no_snapshot_withholds_candidates(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "Shipping" in result.stdout
    assert "no-filter-snapshot" in result.stdout
    assert "CANDIDATE" not in result.stdout
    assert "no filters.json snapshot" in result.stderr
    # the NOTE must advise the renamed acquisition command (ticket #68)
    assert "dataset import-gmail" in result.stderr


def test_analyze_assume_no_filters_opens_gate(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    result = runner.invoke(
        app,
        [
            "dataset",
            "analyze",
            "--config",
            str(_analyze_cfg(tmp_path)),
            "--assume-no-filters",
        ],
    )
    assert result.exit_code == 0
    assert "CANDIDATE" in result.stdout
    assert "Managed-Label candidates: Shipping" in result.stdout


def test_analyze_json_shape(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    result = runner.invoke(
        app,
        [
            "dataset",
            "analyze",
            "--config",
            str(_analyze_cfg(tmp_path)),
            "--assume-no-filters",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["summary"]["messages"] == 60
    shipping = next(m for m in payload["labels"] if m["label"] == "Shipping")
    assert shipping["candidate"] is True


def test_analyze_filter_covered_blocks(tmp_path: Path) -> None:
    root = _seed_corpus(tmp_path)
    import os

    for name, payload in (
        (
            "filters.json",
            '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": ["L1"]}}]',
        ),
        ("labels.json", '{"L1": "Shipping"}'),
    ):
        f = root / name
        f.write_text(payload, encoding="utf-8")
        os.chmod(f, 0o600)
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "filter-covered" in result.stdout
    assert "CANDIDATE" not in result.stdout


def test_analyze_corrupt_snapshot_exits_5(tmp_path: Path) -> None:
    root = _seed_corpus(tmp_path)
    (root / "filters.json").write_text("not json", encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "snapshot unreadable" in result.stderr


def test_analyze_corrupt_labels_without_filters_stay_missing(tmp_path: Path) -> None:
    # the short-circuit is load-bearing: with no filters.json the CLI must
    # not read labels.json at all — a corrupt bridge file beside an absent
    # snapshot is still an honest 'missing' posture (exit 0), not exit 5
    root = _seed_corpus(tmp_path, n=2)
    (root / "labels.json").write_bytes(b"\xff\xfe not utf-8")
    import os

    os.chmod(root / "labels.json", 0o600)  # representative healthy-corpus mode
    result = runner.invoke(
        app,
        ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path)), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["summary"]["filters_posture"] == "missing"


def test_analyze_empty_corpus_is_friendly(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()  # present but empty = friendly guidance
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "corpus is empty" in result.stdout


def test_analyze_missing_corpus_dir_exits_5(tmp_path: Path) -> None:
    # deepseek MINOR: a config typo must NOT read as an empty mailbox
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "missing or not a directory" in result.stderr


def test_analyze_corpus_dir_is_a_file_exits_5(tmp_path: Path) -> None:
    (tmp_path / "corpus").write_text("not a dir", encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 5


def test_analyze_records_undecodable_exits_5(tmp_path: Path) -> None:
    root = _seed_corpus(tmp_path, n=2)
    with (root / "records.jsonl").open("ab") as fh:
        fh.write(b"\xff\xfe bad bytes\n")
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "Traceback" not in (result.stdout + result.stderr)


def test_analyze_contradictory_attestation_is_ignored_loudly(tmp_path: Path) -> None:
    # deepseek MAJOR: evidence beats attestation, and the ignore is visible
    root = _seed_corpus(tmp_path)
    (root / "filters.json").write_text("[]", encoding="utf-8")
    (root / "labels.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "dataset",
            "analyze",
            "--config",
            str(_analyze_cfg(tmp_path)),
            "--assume-no-filters",
        ],
    )
    assert result.exit_code == 0
    assert "ignored" in result.stderr and "evidence beats attestation" in result.stderr


def test_analyze_json_carries_posture_and_iso_dates(tmp_path: Path) -> None:
    # deepseek MINOR: machine consumers must see WHY and parseable timestamps
    root = _seed_corpus(tmp_path)
    (root / "filters.json").write_text("[]", encoding="utf-8")
    (root / "labels.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app,
        ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path)), "--json"],
    )
    payload = json.loads(result.stdout)
    assert payload["summary"]["filters_posture"] == "snapshot"
    assert "T" in payload["summary"]["first_date"]  # ISO-8601


def test_analyze_long_labels_are_not_truncated(tmp_path: Path) -> None:
    # deepseek MINOR: distinct long names must not collide on screen
    long_a = "A" * 40 + "-one"
    long_b = "A" * 40 + "-two"
    for name in (long_a, long_b):
        _seed_corpus(tmp_path, n=3, label=name)
    result = runner.invoke(
        app,
        [
            "dataset",
            "analyze",
            "--config",
            str(_analyze_cfg(tmp_path)),
            "--assume-no-filters",
            "--min-total",
            "1",
            "--min-recent",
            "1",
        ],
    )
    assert long_a in result.stdout and long_b in result.stdout


def test_stub_helper_keeps_the_reserved_exit_3() -> None:
    # deepseek MINOR: the exit-3 contract stays unit-pinned even with an
    # empty stub surface (train/eval land on it in Phase 1)
    import gmail_labeler.cli as cli

    with pytest.raises(typer.Exit) as exc:
        cli._stub("train")
    assert exc.value.exit_code == 3


def test_help_surface_is_clean() -> None:
    for argv in (["--help"], ["dataset", "--help"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0


def test_analyze_corrupt_records_exit_5(tmp_path: Path) -> None:
    root = _seed_corpus(tmp_path, n=2)
    with (root / "records.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{{{ corrupt mid-line\n")
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 5


def test_analyze_attested_posture_is_visible_in_text_mode(tmp_path: Path) -> None:
    # deepseek MINOR: --assume-no-filters rests the gate on the attestation,
    # and the human report must say so
    _seed_corpus(tmp_path)
    result = runner.invoke(
        app,
        [
            "dataset",
            "analyze",
            "--config",
            str(_analyze_cfg(tmp_path)),
            "--assume-no-filters",
        ],
    )
    assert result.exit_code == 0
    assert "filters: attested-none" in result.stdout


def test_analyze_bridge_incomplete_reason_not_misnamed(tmp_path: Path) -> None:
    # deepseek NIT: with a snapshot present, the flag must name the BRIDGE
    root = _seed_corpus(tmp_path, n=2)
    (root / "filters.json").write_text(
        '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": ["GONE"]}}]',
        encoding="utf-8",
    )
    (root / "labels.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert "no-filter-snapshot" not in result.stdout
    assert "bridge-incomplete" in result.stdout


def test_analyze_json_keeps_engine_reasons_alongside_posture(tmp_path: Path) -> None:
    # qwen MINOR1: the asymmetry is INTENTIONAL — engine vocabulary in
    # reasons, real cause in filters_posture; pinning keeps it a decision,
    # not an accident
    root = _seed_corpus(tmp_path, n=2)
    (root / "filters.json").write_text(
        '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": ["GONE"]}}]',
        encoding="utf-8",
    )
    (root / "labels.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path)), "--json"]
    )
    payload = json.loads(result.stdout)
    assert payload["summary"]["filters_posture"] == "bridge-incomplete"
    assert all("no-filter-snapshot" in m["reasons"] for m in payload["labels"])
    assert all("bridge-incomplete" not in m["reasons"] for m in payload["labels"])
    assert "stale bridge" in result.stderr  # the NOTE names the real cause


def test_analyze_empty_corpus_gets_no_filter_note(tmp_path: Path) -> None:
    # deepseek NIT: the empty-corpus advice is the friendly line, not
    # 'run dataset import-gmail to capture filters'
    (tmp_path / "corpus").mkdir()
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "no filters.json snapshot" not in result.stderr
    assert "corpus is empty" in result.stdout
    # the --json quadrant: no human advice line exists to bury, and the
    # NOTEs stay suppressed too — the payload's filters_posture field IS
    # the machine-readable posture ('missing': no sidecar in this fixture)
    json_result = runner.invoke(
        app,
        ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path)), "--json"],
    )
    assert json_result.exit_code == 0
    assert "corpus is empty" not in json_result.stdout
    assert "no filters.json snapshot" not in json_result.stderr
    payload = json.loads(json_result.stdout)
    assert payload["summary"]["filters_posture"] == "missing"


def test_analyze_cli_consumes_resolved_view(tmp_path: Path) -> None:
    # Producer switch (ticket #70): dataset analyze reads the store's
    # resolved view, not the raw log — a revised message counts once,
    # under its newest labels only.
    from gmail_labeler.corpus_store import CorpusStore

    root = _seed_corpus(tmp_path, n=60, label="Shipping")
    store = CorpusStore(root)
    for r in list(store.iter_records())[:10]:
        store.append_revision(replace(r, labels=("Money",)))
    result = runner.invoke(
        app,
        ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path)), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["summary"]["messages"] == 60  # raw log would say 70
    by_label = {m["label"]: m for m in payload["labels"]}
    assert by_label["Shipping"]["total"] == 50  # raw log would say 60
    assert by_label["Money"]["total"] == 10
