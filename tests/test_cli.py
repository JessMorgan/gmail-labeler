from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import typer
from typer.testing import CliRunner

from gmail_labeler.cli import app
from gmail_labeler.corpus_store import CorpusStore

if TYPE_CHECKING:
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage
    from gmail_labeler.gmail_client import HistoryPage

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("auth", "dataset"):
        assert cmd in result.output


def test_dataset_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["dataset", "--help"])
    assert result.exit_code == 0
    for cmd in ("import-gmail", "import-gyb", "analyze", "sync-labels"):
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


# ---- dataset sync-labels (ticket #73) --------------------------------------


def _sync_seed_corpus(
    tmp_path: Path, labels: tuple[str, ...] = ("Shipping",)
) -> CorpusStore:
    from datetime import UTC, datetime

    from gmail_labeler.corpus_record import CorpusRecord, body_hash

    store = CorpusStore(tmp_path / "corpus")
    store.put(
        CorpusRecord(
            id="a1",
            date=datetime(2026, 10, 30, 10, 0, tzinfo=UTC),
            subject="hi",
            sender="a@x.com",
            recipients=("me@x.com",),
            labels=labels,
            body_hash=body_hash("body a1"),
            source="api",
            thread_id="ta1",
        ),
        "body a1",
    )
    store.enforce_modes()
    return store


def _sync_fake(
    *,
    messages: list[FakeMessage] | None = None,
    history_pages: list[HistoryPage] | None = None,
    current_history_id: int | None = None,
) -> FakeGmailClient:
    from gmail_labeler.fake_gmail import FakeGmailClient
    from gmail_labeler.gmail_client import LabelInfo

    return FakeGmailClient(
        labels=[LabelInfo("L1", "Shipping", "user")],
        filters=[],
        messages=messages,
        history_pages=history_pages,
        current_history_id=current_history_id,
    )


def _write_sync_cursor(store: CorpusStore, history_id: int = 100) -> None:
    from datetime import UTC, datetime

    from gmail_labeler.sync_cursor import SyncCursor

    store.write_cursor(
        SyncCursor(history_id=history_id, last_sync=datetime(2026, 11, 1, tzinfo=UTC))
    )


def test_sync_labels_reports_gains_losses_and_analyze_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage, LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    fake = _sync_fake(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([LabelsAdded("a1", ("L9",))]),
            )
        ],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert (
        "sync-labels: events=1 stale=1 revised=1 deleted_kept=0 "
        "new_match=0 retried=0 failed=0" in result.stdout
    )  # full summary shape (retried= survives only here)
    table_rows = [
        line.split() for line in result.stdout.splitlines() if line.startswith("id:L9")
    ]
    assert table_rows == [["id:L9", "1", "0"]]  # per-label gains/losses numbers
    assert "dataset analyze" in result.stdout  # the closing pointer
    assert "Traceback" not in (result.stdout + result.stderr)
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "id:L9")
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500


def test_sync_labels_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage, LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    records_bytes = (store.root / "records.jsonl").read_bytes()
    cursor_bytes = (store.root / "sync-cursor.json").read_bytes()
    fake = _sync_fake(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([LabelsAdded("a1", ("L9",))]),
            )
        ],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app, ["dataset", "sync-labels", "--config", str(cfg), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "(dry-run)" in result.stdout and "revised=1" in result.stdout
    assert (store.root / "records.jsonl").read_bytes() == records_bytes
    assert (store.root / "sync-cursor.json").read_bytes() == cursor_bytes


def test_sync_labels_expired_cursor_prints_guidance_and_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #5/#6: non-zero exit + ADR-0008's FINAL guidance naming the exact
    recovery command. The #73 vaporware-absence pin ("--full-resync" must
    NOT appear while the flag was unbuilt) flips to a presence pin now that
    ticket #74 made the flag real."""
    import gmail_labeler.cli as cli

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store, history_id=900)  # ahead of the fake's window
    fake = _sync_fake(current_history_id=500)
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 5
    assert "retained history" in result.stderr
    assert "dataset sync-labels --full-resync" in (result.stdout + result.stderr)
    assert "not built yet" not in (result.stdout + result.stderr)
    assert "Traceback" not in (result.stdout + result.stderr)
    assert store.read_cursor() is not None  # ledger untouched


def test_sync_labels_auth_failure_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.gmail_errors import GmailAuthError

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    fake = _sync_fake()
    fake.fail("list_labels", GmailAuthError("revoked"))
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_sync_labels_midrun_auth_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.gmail_errors import GmailAuthError

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    fake = _sync_fake(history_pages=[])
    fake.fail("history", GmailAuthError("revoked mid-run"))
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_sync_labels_partial_failure_exits_5_and_holds_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage, LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage
    from gmail_labeler.gmail_errors import GmailPermanentError

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    fake = _sync_fake(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([LabelsAdded("a1", ("L9",))]),
            )
        ],
    )
    fake.fail("get_message_metadata", GmailPermanentError("boom", status=400))
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 5  # never silently partial
    assert "failed a1" in result.stderr
    assert "dataset analyze" not in result.stdout  # no pointer over a half-sync
    assert store.read_cursor() == _cursor_at(100)


def _cursor_at(history_id: int):
    from datetime import UTC, datetime

    from gmail_labeler.sync_cursor import SyncCursor

    return SyncCursor(
        history_id=history_id, last_sync=datetime(2026, 11, 1, tzinfo=UTC)
    )


def test_sync_labels_births_cursorless_corpus_and_reports_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli

    store = _sync_seed_corpus(tmp_path)  # no cursor: GYB-shaped corpus
    fake = _sync_fake(current_history_id=777)
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "no Sync Cursor" in result.stdout  # provenance disclosed
    assert "applies nothing" in result.stdout.lower()
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 777


def test_sync_labels_counts_new_match_and_deleted_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import LabelsAdded, MessageDeleted, history_records
    from gmail_labeler.gmail_client import HistoryPage

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    records_bytes = (store.root / "records.jsonl").read_bytes()
    fake = _sync_fake(
        messages=[],  # a1 gone upstream; d4 never existed locally
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records(
                    [LabelsAdded("d4", ("L1",)), MessageDeleted("a1")]
                ),
            )
        ],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "new_match=1" in result.stdout
    assert "deleted_kept=1" in result.stdout
    assert "not in the corpus" in result.stderr  # NOTE goes to stderr
    assert (store.root / "records.jsonl").read_bytes() == records_bytes


def test_sync_labels_missing_corpus_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli

    fake = _sync_fake()
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)  # corpus dir never created
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 5
    assert "missing or not a directory" in result.stderr


def test_sync_labels_help_documents_dry_run() -> None:
    # rendered help truncates at panel width — pin the DECLARED help string
    import gmail_labeler.cli as cli

    param = inspect.signature(cli.dataset_sync_labels).parameters["dry_run"]
    assert "plan" in param.default.help


def test_sync_labels_local_auth_error_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The likeliest real failure — local credential/store trouble before
    any API call — keeps the exit-4 mapping (not 5)."""
    import gmail_labeler.cli as cli
    from gmail_labeler.auth import AuthError

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)

    def boom(cfg: object) -> FakeGmailClient:
        raise AuthError("no credentials")

    monkeypatch.setattr(cli, "_import_gmail_client", boom)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 4
    assert "auth:" in result.stderr
    assert "Traceback" not in (result.stdout + result.stderr)


def test_sync_labels_unreadable_records_exit_5_no_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """analyze's catch tuple mirrored: an undecodable records index is a
    typed exit 5, never the contract-breaking exit 1 traceback."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    (store.root / "records.jsonl").write_bytes(b"\xff\xfe not utf-8 at all")
    fake = _sync_fake(
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([LabelsAdded("a1", ("L1",))]),
            )
        ]
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 5
    assert "Traceback" not in (result.stdout + result.stderr)


# ---- dataset sync-labels --full-resync (ticket #74) -------------------------


def _resync_fake(
    *,
    messages: list[FakeMessage] | None = None,
    labels: list[tuple[str, str, int]] | None = None,
    current_history_id: int = 500,
) -> FakeGmailClient:
    from gmail_labeler.fake_gmail import FakeGmailClient
    from gmail_labeler.gmail_client import LabelInfo

    return FakeGmailClient(
        labels=[
            LabelInfo(id, name, "user", messages_total=total)
            for id, name, total in (
                labels
                if labels is not None
                else [("L1", "Shipping", 4), ("L2", "Money/Bills", 2)]
            )
        ],
        filters=[],
        messages=messages,
        current_history_id=current_history_id,
    )


def test_sync_labels_full_resync_scan_prints_estimate_choice_and_revises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #1/#6 at the CLI seam: the estimate + strategy lines print (with
    numbers), the flag-rate rides the report, the revision lands, and the
    cursor reseeds to the up-front profile value."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage

    store = _sync_seed_corpus(tmp_path)  # a1 with ("Shipping",)
    _write_sync_cursor(store)
    fake = _resync_fake(messages=[FakeMessage(id="a1", eml=b"x", label_ids=())])
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app, ["dataset", "sync-labels", "--full-resync", "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "full-resync estimate:" in out and "walk ≈ 20 units" in out
    assert "full-resync strategy: scan" in out  # 10-unit floor < 20-unit walk
    assert "flag-rate 100%" in out  # the flood guard's number is printed
    assert "revised=1" in out
    assert "sync-labels --full-resync:" in out  # the report names its mode
    assert "dataset analyze" in out  # the closing pointer
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ()
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # the seed
    # the CLI bootstrap (fetch_settings) precedes the core; WITHIN the core
    # the seed rides first — get_profile before any list/metadata spend
    spend = [
        i
        for i, c in enumerate(fake.calls)
        if c.startswith(("list_messages", "get_message_metadata"))
    ]
    assert fake.calls.index("get_profile") < min(spend)


def test_sync_labels_full_resync_walk_chosen_for_small_corpus_big_mailbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    fake = _resync_fake(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1",))],
        labels=[("L1", "Shipping", 50_000), ("L2", "Money/Bills", 30_000)],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app, ["dataset", "sync-labels", "--full-resync", "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    assert "full-resync strategy: walk" in result.stdout
    assert "scan enumeration floor ≈ 800 units" in result.stdout
    # the walk fetches every id BY CONSTRUCTION — quoting a 100%
    # flag-rate there would read as drift that never happened
    assert "(flag-rate" not in result.stdout
    assert "corpus-wide drift" not in result.stdout


def test_sync_labels_full_resync_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #4 (dry-run half): estimate + choice + plan print; zero writes;
    cursor byte-identical."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    records_bytes = (store.root / "records.jsonl").read_bytes()
    cursor_bytes = (store.root / "sync-cursor.json").read_bytes()
    fake = _resync_fake(messages=[FakeMessage(id="a1", eml=b"x", label_ids=())])
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        [
            "dataset",
            "sync-labels",
            "--full-resync",
            "--dry-run",
            "--config",
            str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "full-resync estimate:" in result.stdout
    assert "(dry-run)" in result.stdout and "revised=1" in result.stdout
    assert (store.root / "records.jsonl").read_bytes() == records_bytes
    assert (store.root / "sync-cursor.json").read_bytes() == cursor_bytes


def test_sync_labels_full_resync_gyb_corpus_births_cursor_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #7 at the CLI seam: a cursorless (GYB-shaped) corpus reconciles
    end-to-end and births the ledger with the up-front seed."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage

    store = _sync_seed_corpus(tmp_path)  # no cursor written
    fake = _resync_fake(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=())],
        current_history_id=777,
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app, ["dataset", "sync-labels", "--full-resync", "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    assert "no Sync Cursor" in result.stdout  # provenance disclosed
    assert "birthed it from the profile seed" in result.stdout  # NOT "WOULD"
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 777
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ()


def test_sync_labels_full_resync_dry_run_cursorless_says_would_birth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The honesty branch (deepseek review): under --dry-run nothing is
    written, so the birth NOTE must say WOULD birth — and the cursor file
    must stay absent while the plan still counts the revision."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage

    store = _sync_seed_corpus(tmp_path)  # no cursor: birth-eligible
    records_bytes = (store.root / "records.jsonl").read_bytes()
    fake = _resync_fake(messages=[FakeMessage(id="a1", eml=b"x", label_ids=())])
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        [
            "dataset",
            "sync-labels",
            "--full-resync",
            "--dry-run",
            "--config",
            str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "WOULD birth" in result.stdout
    assert "birthed it from" not in result.stdout
    assert "revised=1" in result.stdout  # the plan says what WOULD be written
    assert not (store.root / "sync-cursor.json").exists()
    assert (store.root / "records.jsonl").read_bytes() == records_bytes


def test_sync_labels_full_resync_failure_exits_5_and_holds_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage
    from gmail_labeler.gmail_errors import GmailPermanentError

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    fake = _resync_fake(messages=[FakeMessage(id="a1", eml=b"x", label_ids=())])
    fake.fail("get_message_metadata", GmailPermanentError("boom", status=400))
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app, ["dataset", "sync-labels", "--full-resync", "--config", str(cfg)]
    )
    assert result.exit_code == 5  # never silently partial
    assert "failed a1" in result.stderr
    assert "dataset analyze" not in result.stdout
    assert store.read_cursor() == _cursor_at(100)


def test_sync_labels_birth_note_recommends_the_real_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cursorless-birth NOTE (event walk) now points at the real
    --full-resync command instead of the interim rebuild advice."""
    import gmail_labeler.cli as cli

    _sync_seed_corpus(tmp_path)  # no cursor: the event-walk birth path
    fake = _sync_fake(current_history_id=777)
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    out = result.stdout + result.stderr
    assert "dataset sync-labels --full-resync" in out
    assert "not built yet" not in out


def test_sync_labels_help_documents_full_resync() -> None:
    import gmail_labeler.cli as cli

    param = inspect.signature(cli.dataset_sync_labels).parameters["full_resync"]
    assert "no history" in param.default.help.lower()


# ---- composition flags (ticket #75) ----------------------------------------

# shared fixture bodies for the composition tests (kept as constants: the
# inline byte literals exceed the line budget at this nesting depth)
_COMPOSITE_NEW_EML = (
    b"Subject: New\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nbody b2"
)
_COMPOSITE_OLD_EML = (
    b"Subject: Hi\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nbody a1"
)


def test_sync_labels_also_import_gmail_reports_both_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #1 (sync→import): after the sync delta is durable (cursor
    advanced), the FULL Gmail Import pass runs and ONE report carries both
    passes' stats. The new-match interaction is the honest story: sync
    counts b2 as new_match and writes nothing for it; the import pass then
    ingests it — the resume gate (skip-present) absorbs only the already
    resident a1, never fighting sync (spec #64 user story 14)."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import (
        FakeMessage,
        LabelsAdded,
        MessageAdded,
        history_records,
    )
    from gmail_labeler.gmail_client import HistoryPage

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)  # history_id=100
    fake = _sync_fake(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9")),
            FakeMessage(
                id="b2",
                eml=_COMPOSITE_NEW_EML,
                label_ids=("L1",),
                internal_date_ms=1761818400000,
            ),
        ],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records(
                    [LabelsAdded("a1", ("L9",)), MessageAdded("b2", "tb2")]
                ),
            )
        ],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "sync-labels", "--config", str(cfg), "--also-import-gmail"],
    )
    assert result.exit_code == 0, result.output
    # first pass: the standalone sync summary, unchanged shape
    assert (
        "sync-labels: events=2 stale=1 revised=1 deleted_kept=0 "
        "new_match=1 retried=0 failed=0" in result.stdout
    )
    # second pass: the import summary carries its own stats, clearly marked
    assert (
        "import-gmail (also): scanned=2 written=1 skipped_present=1 "
        "retried=0 failed=0" in result.stdout
    )
    # ONE report: a single analyze pointer, closing AFTER both passes
    assert result.stdout.count("dataset analyze") == 1
    assert result.stdout.index("import-gmail (also):") < result.stdout.index(
        "Next: run `dataset analyze`"
    )
    # post-state through the resolved view: sync's revision AND the
    # imported record are both durable
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "id:L9")
    assert resolved["b2"].labels == ("Shipping",)
    # the import pass never moved the ledger sync advanced (ADR-0008)
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500
    assert "Traceback" not in (result.stdout + result.stderr)


def test_sync_labels_also_import_gmail_second_pass_failure_keeps_sync_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #2 (sync→import): the second pass's failure is reported non-zero
    per the pinned taxonomy, and the first pass's durable records AND the
    advanced cursor stay byte-intact — no rollback (spec #64: the second
    pass's failure does not invalidate the first)."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import (
        FakeMessage,
        LabelsAdded,
        MessageAdded,
        history_records,
    )
    from gmail_labeler.gmail_client import HistoryPage
    from gmail_labeler.gmail_errors import GmailPermanentError

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    fake = _sync_fake(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9")),
            FakeMessage(
                id="b2",
                eml=_COMPOSITE_NEW_EML,
                label_ids=("L1",),
                internal_date_ms=1761818400000,
            ),
        ],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records(
                    [LabelsAdded("a1", ("L9",)), MessageAdded("b2", "tb2")]
                ),
            )
        ],
    )
    # the import pass's first wire call fails permanently — sync never
    # touches list_messages, so only the second pass can consume this
    from typing import Any

    fake.fail("list_messages", GmailPermanentError("quota blown", status=400))
    captured: dict[str, bytes] = {}
    orig = fake.list_messages

    def spy(query: str, page_token: str | None = None, **kw: Any) -> Any:
        captured.setdefault("cursor", (store.root / "sync-cursor.json").read_bytes())
        captured.setdefault("records", (store.root / "records.jsonl").read_bytes())
        return orig(query, page_token, **kw)

    monkeypatch.setattr(fake, "list_messages", spy)
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "sync-labels", "--config", str(cfg), "--also-import-gmail"],
    )
    assert result.exit_code == 5
    assert "import-gmail (also):" in result.stdout
    assert "failed=1" in result.stdout
    assert "failed <page>" in result.stderr  # the sentinel, reported
    assert "Next: run `dataset analyze`" not in result.stdout  # incomplete
    # the failed second pass rolled nothing back: the cursor bytes the sync
    # pass wrote are still the bytes on disk, and sync's revision resolves
    assert (store.root / "sync-cursor.json").read_bytes() == captured["cursor"]
    assert (store.root / "records.jsonl").read_bytes() == captured["records"]
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "id:L9")
    assert "b2" not in resolved  # the import died before ingesting it
    assert "Traceback" not in (result.stdout + result.stderr)


def test_sync_labels_also_import_gmail_second_pass_auth_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #2 taxonomy: a revoked token mid-import-pass is exit 4 (the fix is
    human), never a raw traceback — and the sync pass's durable work stands
    untouched."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage, LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage
    from gmail_labeler.gmail_errors import GmailAuthError

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    fake = _sync_fake(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([LabelsAdded("a1", ("L9",))]),
            )
        ],
    )
    fake.fail("list_messages", GmailAuthError("revoked mid-run"))
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "sync-labels", "--config", str(cfg), "--also-import-gmail"],
    )
    assert result.exit_code == 4
    assert "auth:" in result.stderr
    assert "sync-labels: events=1" in result.stdout  # first pass reported
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # stands
    assert "Traceback" not in (result.stdout + result.stderr)


def test_import_gmail_also_sync_labels_reports_both_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #1 (import→sync): after the import completes durably, the Label
    Sync delta runs and ONE report carries both passes' stats. The cursor
    story is ADR-0008's designed loop: import births the ledger from the
    FIRST stub's historyId captured before paging (100), the sync walk
    starts there and advances to the mailbox current id (500). The walk
    re-touches the freshly imported a1 (refresh-on-any-touch): the metadata
    matches what import wrote, so the Revision is a self-healing append —
    revised=1 with an empty gains/losses table is the honest stats story."""
    import gmail_labeler.cli as cli
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.fake_gmail import FakeMessage, LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage

    fake = _sync_fake(
        messages=[
            FakeMessage(
                id="a1",
                eml=_COMPOSITE_OLD_EML,
                label_ids=("L1", "L9"),
                thread_id="ta1",
                internal_date_ms=1761818400000,
                history_id=100,  # the birth seed the import must read first
            )
        ],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([LabelsAdded("a1", ("L9",))]),
            )
        ],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "import-gmail", "--config", str(cfg), "--also-sync-labels"],
    )
    assert result.exit_code == 0, result.output
    assert (
        "import-gmail: scanned=1 written=1 skipped_present=0 "
        "retried=0 failed=0" in result.stdout
    )
    assert (
        "sync-labels (also): events=1 stale=1 revised=1 deleted_kept=0 "
        "new_match=0 retried=0 failed=0" in result.stdout
    )
    # ONE report: the analyze pointer closes after both passes, once
    assert result.stdout.count("dataset analyze") == 1
    assert result.stdout.index("sync-labels (also):") < result.stdout.index(
        "Next: run `dataset analyze`"
    )
    store = CorpusStore(cfg.parent / "corpus")
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # sync advanced
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "id:L9")
    assert "Traceback" not in (result.stdout + result.stderr)


def test_import_gmail_also_sync_labels_second_pass_failure_keeps_import_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #2 (import→sync): the sync pass's failure is reported non-zero;
    the import's durable records and its birthed cursor stay byte-intact —
    no rollback, no analyze pointer over a half-finished composite."""
    import gmail_labeler.cli as cli
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.fake_gmail import FakeMessage, history_records
    from gmail_labeler.gmail_client import HistoryPage
    from gmail_labeler.gmail_errors import GmailPermanentError

    fake = _sync_fake(
        messages=[
            FakeMessage(
                id="a1",
                eml=_COMPOSITE_OLD_EML,
                label_ids=("L1",),
                thread_id="ta1",
                internal_date_ms=1761818400000,
                history_id=100,
            )
        ],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([]),
            )
        ],
    )
    # the sync pass's first wire call fails permanently — import never
    # touches history(), so only the second pass can consume this
    fake.fail("history", GmailPermanentError("service blown", status=400))
    captured: dict[str, bytes] = {}
    orig = fake.history

    def spy(start_history_id: int, page_token: str | None = None) -> object:
        captured.setdefault(
            "cursor", (tmp_path / "corpus" / "sync-cursor.json").read_bytes()
        )
        captured.setdefault(
            "records", (tmp_path / "corpus" / "records.jsonl").read_bytes()
        )
        return orig(start_history_id, page_token)

    monkeypatch.setattr(fake, "history", spy)
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "import-gmail", "--config", str(cfg), "--also-sync-labels"],
    )
    assert result.exit_code == 5
    assert "sync-labels (also):" in result.stdout
    assert "failed=1" in result.stdout
    assert "failed <page>" in result.stderr
    assert "Next: run `dataset analyze`" not in result.stdout
    # the failed second pass left the import's durable state byte-identical
    assert (tmp_path / "corpus" / "sync-cursor.json").read_bytes() == (
        captured["cursor"]
    )
    assert (tmp_path / "corpus" / "records.jsonl").read_bytes() == captured["records"]
    store = CorpusStore(tmp_path / "corpus")
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping",)  # import's record stands
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 100  # the birth, held
    assert "Traceback" not in (result.stdout + result.stderr)


def _also_sync_raise_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error):
    """Shared fixture for the import→sync raise-path tests: import runs
    clean, the sync pass's first wire call raises `error` (consumed once —
    these types are never retried, so one arm is the whole story)."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage

    fake = _sync_fake(
        messages=[
            FakeMessage(
                id="a1",
                eml=_COMPOSITE_OLD_EML,
                label_ids=("L1",),
                thread_id="ta1",
                internal_date_ms=1761818400000,
                history_id=100,
            )
        ],
    )
    fake.fail("history", error)
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    return fake


def test_import_gmail_also_sync_labels_second_pass_auth_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #2 taxonomy: a revoked token mid-sync-pass is exit 4 (the fix is
    human), the import's durable work standing."""
    from gmail_labeler.gmail_errors import GmailAuthError

    _also_sync_raise_fake(tmp_path, monkeypatch, GmailAuthError("revoked"))
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "import-gmail", "--config", str(cfg), "--also-sync-labels"],
    )
    assert result.exit_code == 4
    assert "auth:" in result.stderr
    assert "import-gmail: scanned=1 written=1" in result.stdout  # first pass
    from gmail_labeler.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 100  # the birth, held
    assert "Traceback" not in (result.stdout + result.stderr)


def test_import_gmail_also_sync_labels_expired_cursor_prints_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sync pass keeps the 404 recovery contract (ADR-0008): the exact
    `dataset sync-labels --full-resync` command prints, exit 5, the import's
    birthed ledger untouched — recovery stays a human decision even when
    sync arrives as a composite's second pass."""
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.gmail_errors import GmailPermanentError

    _also_sync_raise_fake(
        tmp_path, monkeypatch, GmailPermanentError("history gone", status=404)
    )
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "import-gmail", "--config", str(cfg), "--also-sync-labels"],
    )
    assert result.exit_code == 5
    assert "retained history" in result.stderr
    assert "dataset sync-labels --full-resync" in (result.stdout + result.stderr)
    store = CorpusStore(tmp_path / "corpus")
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 100  # untouched
    assert "Traceback" not in (result.stdout + result.stderr)


def test_sync_labels_also_import_gmail_import_starts_after_cursor_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordering pin (sync→import): the import pass's first wire call lands
    strictly AFTER the Sync Cursor write — the sync pass's final durability
    point — and the exact call log shows no silent extra fetches between
    the passes."""
    import gmail_labeler.cli as cli
    import gmail_labeler.corpus_store as corpus_store
    from gmail_labeler.fake_gmail import FakeMessage, LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage
    from gmail_labeler.gmail_import import DEFAULT_QUERY

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    fake = _sync_fake(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9")),
            FakeMessage(
                id="b2",
                eml=_COMPOSITE_NEW_EML,
                label_ids=("L1",),
                internal_date_ms=1761818400000,
            ),
        ],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([LabelsAdded("a1", ("L9",))]),
            )
        ],
    )
    from gmail_labeler.sync_cursor import SyncCursor

    cursor_writes: list[list[str]] = []
    orig_write = corpus_store.CorpusStore.write_cursor

    def spy_write(self: corpus_store.CorpusStore, cursor: SyncCursor) -> None:
        cursor_writes.append(list(fake.calls))  # the call log AT the write
        orig_write(self, cursor)

    monkeypatch.setattr(corpus_store.CorpusStore, "write_cursor", spy_write)
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "sync-labels", "--config", str(cfg), "--also-import-gmail"],
    )
    assert result.exit_code == 0, result.output
    # import never re-writes a live cursor (birth-when-absent): the single
    # write is the sync pass's own advancement
    assert len(cursor_writes) == 1
    # at the durability point, NO import wire call has happened yet — the
    # second pass starts strictly after the first pass's final write
    assert not any(c.startswith("list_messages") for c in cursor_writes[0])
    assert fake.calls == [
        "list_labels",
        "list_filters",
        "history:100:",
        "get_message_metadata:a1",
        f"list_messages:{DEFAULT_QUERY}:",
        "get_message:b2",
    ]


def test_import_gmail_also_sync_labels_sync_starts_after_import_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordering pin (import→sync): the sync walk's first wire call lands
    after the import's last one, and the walk marking the freshly imported
    id stale (stale=1) is the functional proof the import's writes were
    durable and readable before the walk began — the resume gate and the
    event walk never race over the same message."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage, LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage
    from gmail_labeler.gmail_import import DEFAULT_QUERY

    fake = _sync_fake(
        messages=[
            FakeMessage(
                id="a1",
                eml=_COMPOSITE_OLD_EML,
                label_ids=("L1", "L9"),
                thread_id="ta1",
                internal_date_ms=1761818400000,
                history_id=100,
            )
        ],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([LabelsAdded("a1", ("L9",))]),
            )
        ],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "import-gmail", "--config", str(cfg), "--also-sync-labels"],
    )
    assert result.exit_code == 0, result.output
    assert "stale=1" in result.stdout  # the walk SAW the imported record
    assert fake.calls == [
        "list_labels",
        "list_filters",
        f"list_messages:{DEFAULT_QUERY}:",
        "get_message:a1",
        "history:100:",  # from the birth seed — captured before paging
        "get_message_metadata:a1",
    ]


def _composition_sync_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root_name: str
) -> tuple[Path, FakeGmailClient]:
    """One mailbox/corpus fixture built twice (sync side of the
    byte-identical pin): returns the config path for a fresh run."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import (
        FakeMessage,
        LabelsAdded,
        MessageAdded,
        history_records,
    )
    from gmail_labeler.gmail_client import HistoryPage

    root = tmp_path / root_name
    root.mkdir(parents=True)
    store = _sync_seed_corpus(root)
    _write_sync_cursor(store)
    fake = _sync_fake(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9")),
            FakeMessage(
                id="b2",
                eml=_COMPOSITE_NEW_EML,
                label_ids=("L1",),
                internal_date_ms=1761818400000,
            ),
        ],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records(
                    [LabelsAdded("a1", ("L9",)), MessageAdded("b2", "tb2")]
                ),
            )
        ],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    return _import_gmail_cfg(root), fake


def test_sync_labels_composite_body_is_byte_identical_to_standalone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #3: without the flag the report is exactly the standalone
    contract; with it the sync pass's report body is BYTE-IDENTICAL — the
    composition only appends the import pass and moves the closing
    pointer."""
    cfg_a, _fake_a = _composition_sync_fixture(tmp_path, monkeypatch, "a")
    res_a = runner.invoke(app, ["dataset", "sync-labels", "--config", str(cfg_a)])
    cfg_b, _fake_b = _composition_sync_fixture(tmp_path, monkeypatch, "b")
    res_b = runner.invoke(
        app,
        ["dataset", "sync-labels", "--config", str(cfg_b), "--also-import-gmail"],
    )
    assert res_a.exit_code == 0, res_a.output
    assert res_b.exit_code == 0, res_b.output
    marker = "\nNext: run `dataset analyze`"
    tail = (
        " for the updated candidate gates (ADR-0005 keeps retraining "
        "and applying human-gated)\n"
    )
    body_a, found_a, tail_a = res_a.stdout.partition(marker)
    body_b, found_b, tail_b = res_b.stdout.partition(marker)
    assert found_a and found_b and tail_a == tail_b == tail
    assert body_b.startswith(body_a)  # the sync report body did not move
    assert body_b == (
        body_a + "import-gmail (also): scanned=2 written=1 skipped_present=1 "
        "retried=0 failed=0\n"
    )


def test_import_gmail_composite_body_is_byte_identical_to_standalone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #3 (import side): the standalone import report is untouched, and
    the composite appends the sync pass after it verbatim."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage, LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage

    def build(root_name: str) -> Path:
        root = tmp_path / root_name
        root.mkdir(parents=True)
        fake = _sync_fake(
            messages=[
                FakeMessage(
                    id="a1",
                    eml=_COMPOSITE_OLD_EML,
                    label_ids=("L1", "L9"),
                    thread_id="ta1",
                    internal_date_ms=1761818400000,
                    history_id=100,
                )
            ],
            history_pages=[
                HistoryPage(
                    history_id=500,
                    next_page_token=None,
                    records=history_records([LabelsAdded("a1", ("L9",))]),
                )
            ],
        )
        monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
        return _import_gmail_cfg(root)

    cfg_a = build("a")
    res_a = runner.invoke(app, ["dataset", "import-gmail", "--config", str(cfg_a)])
    cfg_b = build("b")
    res_b = runner.invoke(
        app,
        ["dataset", "import-gmail", "--config", str(cfg_b), "--also-sync-labels"],
    )
    assert res_a.exit_code == 0, res_a.output
    assert res_b.exit_code == 0, res_b.output
    assert res_a.stdout == (
        "import-gmail: scanned=1 written=1 skipped_present=0 retried=0 failed=0\n"
    )
    assert res_b.stdout.startswith(res_a.stdout)  # the import report did not move
    marker = "\nNext: run `dataset analyze`"
    tail = (
        " for the updated candidate gates (ADR-0005 keeps retraining "
        "and applying human-gated)\n"
    )
    # the appended sync body is pinned exactly, tail included (round 1 nit:
    # startswith alone left the pointer placement unpinned on this side)
    assert res_b.stdout[len(res_a.stdout) :] == (
        "sync-labels (also): events=1 stale=1 revised=1 deleted_kept=0 "
        "new_match=0 retried=0 failed=0\n" + marker + tail
    )


def test_sync_labels_dry_run_with_also_import_gmail_is_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contradictory quadrant: --dry-run promises zero writes, and the
    import pass writes records — so the combination is refused as a usage
    error (exit 2) BEFORE any quota spend, not silently degraded into a
    one-pass run."""
    cfg, _fake = _composition_sync_fixture(tmp_path, monkeypatch, "x")
    root = tmp_path / "x" / "corpus"
    records_bytes = (root / "records.jsonl").read_bytes()
    cursor_bytes = (root / "sync-cursor.json").read_bytes()
    result = runner.invoke(
        app,
        [
            "dataset",
            "sync-labels",
            "--config",
            str(cfg),
            "--dry-run",
            "--also-import-gmail",
        ],
    )
    assert result.exit_code == 2
    assert "--dry-run" in result.stderr and "--also-import-gmail" in result.stderr
    assert (root / "records.jsonl").read_bytes() == records_bytes
    assert (root / "sync-cursor.json").read_bytes() == cursor_bytes
    # the refusal precedes the config read: a bogus --config still exits 2
    # with the refusal — never the config-error path (round 1 nit: without
    # this quadrant the test could not tell "before config read" from
    # "after config read, before quota spend")
    bogus = runner.invoke(
        app,
        [
            "dataset",
            "sync-labels",
            "--config",
            str(tmp_path / "nope.toml"),
            "--dry-run",
            "--also-import-gmail",
        ],
    )
    assert bogus.exit_code == 2
    assert "--also-import-gmail" in bogus.stderr
    assert "config error" not in bogus.stderr


def test_composite_flags_documented_in_help() -> None:
    """The flags' help names the contract they exist to keep: order
    (after durability) and the no-rollback failure story."""
    import gmail_labeler.cli as cli

    sync_flag = inspect.signature(cli.dataset_sync_labels).parameters[
        "also_import_gmail"
    ]
    help_sync = sync_flag.default.help
    assert "durable" in help_sync and "never invalidates the sync" in help_sync
    import_flag = inspect.signature(cli.dataset_import_gmail).parameters[
        "also_sync_labels"
    ]
    help_import = import_flag.default.help
    assert "durably" in help_import and "never invalidates the import" in help_import


def test_sync_labels_full_resync_also_import_gmail_composes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The composite flag rides BOTH sync branches: after the
    --full-resync reconciliation is durable (cursor reseeded), the import
    pass runs and one report carries both passes' stats."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage

    store = _sync_seed_corpus(tmp_path)  # a1 with ("Shipping",)
    _write_sync_cursor(store)
    fake = _resync_fake(
        messages=[
            FakeMessage(id="a1", eml=b"x", label_ids=()),
            FakeMessage(
                id="b2",
                eml=_COMPOSITE_NEW_EML,
                label_ids=("L1",),
                internal_date_ms=1761818400000,
            ),
        ]
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        [
            "dataset",
            "sync-labels",
            "--config",
            str(cfg),
            "--full-resync",
            "--also-import-gmail",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "sync-labels --full-resync:" in result.stdout
    assert (
        "import-gmail (also): scanned=2 written=1 skipped_present=1 "
        "retried=0 failed=0" in result.stdout
    )
    assert result.stdout.count("dataset analyze") == 1
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ()  # the resync's revision resolves
    assert resolved["b2"].labels == ("Shipping",)  # the import's record
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # the resync seed


def test_sync_labels_failed_first_pass_never_reaches_import_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate pin (review round 1, both families): a sync pass that FAILED
    never reached its durability point, so the composite import pass must
    not run at all — no import wire call, no (also) line, no pointer — and
    the cursor stays behind the unconsumed delta (invariant #3)."""
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeMessage, LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage
    from gmail_labeler.gmail_errors import GmailPermanentError

    store = _sync_seed_corpus(tmp_path)
    _write_sync_cursor(store)
    fake = _sync_fake(
        messages=[FakeMessage(id="a1", eml=b"x", label_ids=("L1", "L9"))],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([LabelsAdded("a1", ("L9",))]),
            )
        ],
    )
    fake.fail(
        "get_message_metadata", GmailPermanentError("boom", status=400)
    )  # permanent, non-404: a recorded per-message failure
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "sync-labels", "--config", str(cfg), "--also-import-gmail"],
    )
    assert result.exit_code == 5
    assert "failed=1" in result.stdout
    assert "import-gmail (also):" not in result.stdout
    assert "Next: run `dataset analyze`" not in result.stdout
    assert not any(c.startswith("list_messages") for c in fake.calls)
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 100  # held back


def test_import_gmail_failed_first_pass_never_reaches_sync_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate pin, import side: a failed import (a permanently unfetchable
    message after the page and its birth) gets no sync pass — no history
    wire call, no (also) line — while the birth-once cursor the import
    already wrote stands (durable work, no rollback)."""
    import gmail_labeler.cli as cli
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.fake_gmail import FakeMessage
    from gmail_labeler.gmail_errors import GmailPermanentError

    fake = _sync_fake(
        messages=[
            FakeMessage(
                id="a1",
                eml=_COMPOSITE_OLD_EML,
                label_ids=("L1",),
                thread_id="ta1",
                internal_date_ms=1761818400000,
                history_id=100,
            )
        ],
    )
    fake.fail("get_message", GmailPermanentError("boom", status=400))
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "import-gmail", "--config", str(cfg), "--also-sync-labels"],
    )
    assert result.exit_code == 5
    assert "sync-labels (also):" not in result.stdout
    assert "Next: run `dataset analyze`" not in result.stdout
    assert not any(c.startswith("history") for c in fake.calls)
    # the page fetch succeeded (so the birth ran) and only the body get
    # failed: the birth-once cursor is durable work the failure keeps
    cursor = CorpusStore(tmp_path / "corpus").read_cursor()
    assert cursor is not None and cursor.history_id == 100  # the birth stands


def test_composite_report_counts_bootstrap_retried_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No-double-count pin (review round 1): the shared settings blip rides
    the FIRST pass's report only — the composite's second line keeps its
    own retried honest at zero."""
    from gmail_labeler.gmail_errors import GmailTransientError

    cfg, fake = _composition_sync_fixture(tmp_path, monkeypatch, "a")
    fake.fail("list_labels", GmailTransientError("blip"))
    result = runner.invoke(
        app,
        ["dataset", "sync-labels", "--config", str(cfg), "--also-import-gmail"],
    )
    assert result.exit_code == 0, result.output
    assert (
        "sync-labels: events=2 stale=1 revised=1 deleted_kept=0 "
        "new_match=1 retried=1 failed=0" in result.stdout
    )
    assert "import-gmail (also):" in result.stdout
    assert "retried=0" in result.stdout.split("import-gmail (also):")[1]


def test_import_gmail_also_sync_labels_store_fault_exits_5_keeps_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second-pass RAISED-fault coverage (round 2 minor): a structural
    store fault inside the sync walk — not the stats.failed sentinel path —
    maps through the standalone sync's tuple (exit 5, typed message), and
    the import's durable records and birthed cursor stand."""
    import gmail_labeler.cli as cli
    import gmail_labeler.corpus_store as corpus_store
    from gmail_labeler.corpus_store import CorpusStore
    from gmail_labeler.fake_gmail import FakeMessage, LabelsAdded, history_records
    from gmail_labeler.gmail_client import HistoryPage

    fake = _sync_fake(
        messages=[
            FakeMessage(
                id="a1",
                eml=_COMPOSITE_OLD_EML,
                label_ids=("L1", "L9"),
                thread_id="ta1",
                internal_date_ms=1761818400000,
                history_id=100,
            )
        ],
        history_pages=[
            HistoryPage(
                history_id=500,
                next_page_token=None,
                records=history_records([LabelsAdded("a1", ("L9",))]),
            )
        ],
    )
    monkeypatch.setattr(cli, "_import_gmail_client", lambda cfg: fake)

    def boom(self: corpus_store.CorpusStore, record: object) -> None:
        raise corpus_store.CorpusError("disk gone mid-revision")

    monkeypatch.setattr(corpus_store.CorpusStore, "append_revision", boom)
    cfg = _import_gmail_cfg(tmp_path)
    result = runner.invoke(
        app,
        ["dataset", "import-gmail", "--config", str(cfg), "--also-sync-labels"],
    )
    assert result.exit_code == 5
    assert "sync-labels failed:" in result.stderr
    assert "Next: run `dataset analyze`" not in result.stdout
    store = CorpusStore(tmp_path / "corpus")
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "id:L9")  # import's record
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 100  # birth held
    assert "Traceback" not in (result.stdout + result.stderr)


def test_sync_labels_also_import_gmail_store_fault_exits_5_keeps_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The import pass's raised-fault twin: a structural store fault inside
    the import's record write (put — a path the sync pass never takes;
    write_snapshot would arm the WRONG pass, since write_cursor reuses it)
    maps to exit 5 with the standalone import's message, and the sync
    pass's durable revision and advanced cursor stand — no rollback."""
    import gmail_labeler.corpus_store as corpus_store

    cfg, _fake = _composition_sync_fixture(tmp_path, monkeypatch, "a")

    def boom(self: corpus_store.CorpusStore, record: object, body: object) -> bool:
        raise corpus_store.CorpusError("disk gone mid-put")

    monkeypatch.setattr(corpus_store.CorpusStore, "put", boom)
    result = runner.invoke(
        app,
        ["dataset", "sync-labels", "--config", str(cfg), "--also-import-gmail"],
    )
    assert result.exit_code == 5
    assert "import failed:" in result.stderr
    assert "Next: run `dataset analyze`" not in result.stdout
    store = corpus_store.CorpusStore(tmp_path / "a" / "corpus")
    resolved = {r.id: r for r in store.iter_resolved()}
    assert resolved["a1"].labels == ("Shipping", "id:L9")  # sync's revision
    cursor = store.read_cursor()
    assert cursor is not None and cursor.history_id == 500  # advance held
    assert "Traceback" not in (result.stdout + result.stderr)
