from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
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
    for cmd in ("export", "import-gyb", "analyze"):
        assert cmd in result.output


def test_usage_error_stays_exit_2() -> None:
    """The 2/3 split is the contract: unknown commands keep exit 2."""
    result = runner.invoke(app, ["bogus-command"])
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


def _export_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[gmail]\ntoken_backend = "file"\n'
        f'token_file = "{tmp_path / "tok.json"}"\n'
        f'[corpus]\ndir = "{tmp_path / "corpus"}"\n',
        encoding="utf-8",
    )
    return cfg


def test_dataset_export_writes_and_reports(
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
    monkeypatch.setattr(cli, "_export_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "written=1" in result.output
    assert (tmp_path / "corpus" / "records.jsonl").is_file()
    import json as _json

    sidecar = _json.loads(
        (tmp_path / "corpus" / "filters.json").read_text(encoding="utf-8")
    )
    assert sidecar[0]["id"] == "f1"  # offline attribution snapshot
    assert sidecar[0]["criteria"] == {"from": "shipper@x.com"}


def test_export_stats_echo_includes_skipped_draft_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # user-visible stats line: the ingest policy-skip count must not
    # regress silently (deepseek nit)
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage

    eml = b"Subject: x\r\nDate: Sun, 30 Oct 2026 10:00:00 +0000\r\n\r\nb"
    fake = FakeGmailClient(
        messages=[
            FakeMessage(id="aaa", eml=eml, label_ids=("INBOX",)),
            FakeMessage(id="bbb", eml=eml, label_ids=("DRAFT",)),
        ]
    )
    monkeypatch.setattr(cli, "_export_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert (
        "scanned=2 written=1 skipped_present=0 skipped_draft_chat=1 "
        "retried=0 failed=0" in result.stdout
    )


def test_dataset_export_incomplete_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gmail_labeler import cli
    from gmail_labeler.fake_gmail import FakeGmailClient, FakeMessage

    fake = FakeGmailClient(messages=[FakeMessage(id="aaa", eml=b"no-date-here")])
    monkeypatch.setattr(cli, "_export_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "failed=1" in result.stdout


def test_dataset_export_auth_failure_exits_4(tmp_path: Path) -> None:
    # token_file has no token -> credentials() AuthError -> 4 (not 5, not 1)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
    )
    assert result.exit_code == 4
    assert "Traceback" not in result.output


def test_export_sidecar_is_0600_and_query_flows(
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
    monkeypatch.setattr(cli, "_export_client", lambda cfg: fake)
    result = runner.invoke(
        app,
        [
            "dataset",
            "export",
            "--config",
            str(_export_cfg(tmp_path)),
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


def test_export_settings_blip_retries_then_succeeds(
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
    monkeypatch.setattr(cli, "_export_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
    )
    assert result.exit_code == 0  # settings are not a single-blip kill switch
    assert "written=1" in result.output


def test_export_auth_failure_exits_4_not_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.auth import AuthError

    def boom(cfg: object) -> object:
        raise AuthError("not signed in — run `gmail-labeler auth login`")

    monkeypatch.setattr(cli, "_export_client", boom)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
    )
    assert result.exit_code == 4
    assert "auth" in (result.stdout + result.stderr)


def test_export_bootstrap_auth_failure_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # deepseek MINOR: the now-live `except GmailAuthError` clauses need
    # CLI-level pins (engine pins cannot catch a routing regression)
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient
    from gmail_labeler.gmail_errors import classify_status

    fake = FakeGmailClient()
    fake.fail("list_labels", classify_status(401))
    monkeypatch.setattr(cli, "_export_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
    )
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_export_settings_exhaustion_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gmail_labeler.cli as cli
    from gmail_labeler.fake_gmail import FakeGmailClient
    from gmail_labeler.gmail_errors import classify_status

    fake = FakeGmailClient()
    for _ in range(8):  # exhaust every attempt (MAX_ATTEMPTS is smaller)
        fake.fail("list_labels", classify_status(503))
    monkeypatch.setattr(
        "gmail_labeler.export.time.sleep", lambda _s: None
    )  # 5x503 exhaustion would pay ~15 s real backoff
    monkeypatch.setattr(cli, "_export_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "label map unavailable after retries" in result.stderr


def test_export_retried_counts_settings_blips(
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
    monkeypatch.setattr(cli, "_export_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "retried=1" in result.stdout


def test_export_midrun_auth_failure_exits_4(
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
    monkeypatch.setattr(cli, "_export_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
    )
    assert result.exit_code == 4
    assert "auth:" in result.stderr


def test_export_filters_exhaustion_message(
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
        "gmail_labeler.export.time.sleep", lambda _s: None
    )  # 5x503 exhaustion would pay ~15 s real backoff
    monkeypatch.setattr(cli, "_export_client", lambda cfg: fake)
    result = runner.invoke(
        app, ["dataset", "export", "--config", str(_export_cfg(tmp_path))]
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
            date_provenance="internal_date",
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


def test_analyze_missing_bridge_withholds_candidates(tmp_path: Path) -> None:
    # deepseek CRITICAL: filters.json without the labels.json bridge must
    # NEVER surface filter-authored labels as CANDIDATE
    root = _seed_corpus(tmp_path)
    (root / "filters.json").write_text(
        '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": ["L1"]}}]',
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "CANDIDATE" not in result.stdout
    assert "bridge" in result.stderr


def test_analyze_stale_bridge_withholds_too(tmp_path: Path) -> None:
    root = _seed_corpus(tmp_path)
    (root / "filters.json").write_text(
        '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": ["GONE"]}}]',
        encoding="utf-8",
    )
    (root / "labels.json").write_text('{"L1": "Shipping"}', encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "CANDIDATE" not in result.stdout and "bridge" in result.stderr


def test_analyze_wrong_shape_sidecars_exit_5(tmp_path: Path) -> None:
    # deepseek MAJOR: parseable-but-wrong-shape is exit 5, not a traceback
    for payload in ("[42]", '{"not": "a list"}'):
        root = _seed_corpus(tmp_path, n=2)
        (root / "filters.json").write_text(payload, encoding="utf-8")
        result = runner.invoke(
            app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
        )
        assert result.exit_code == 5 and "snapshot unreadable" in result.stderr
        (root / "filters.json").unlink(missing_ok=True)
        (root / "labels.json").unlink(missing_ok=True)
    root = _seed_corpus(tmp_path, n=2)
    (root / "filters.json").write_text("[]", encoding="utf-8")
    (root / "labels.json").write_text('["L1"]', encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 5  # labels map wrong-shape raises too


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


def test_analyze_nonstring_label_ids_rejected(tmp_path: Path) -> None:
    # deepseek MINOR: [42]-as-id must not bypass the bridge check
    root = _seed_corpus(tmp_path, n=2)
    (root / "filters.json").write_text(
        '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": [42]}}]',
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "must be strings" in result.stderr


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


def test_analyze_nonlist_addlabelids_rejected(tmp_path: Path) -> None:
    # qwen MINOR2: a dict-instead-of-list ids field must not attribute nothing
    root = _seed_corpus(tmp_path, n=2)
    (root / "filters.json").write_text(
        '[{"criteria": {"from": "shop@x.z"}, "action": {"addLabelIds": {"x": 1}}}]',
        encoding="utf-8",
    )
    (root / "labels.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "addLabelIds must be a list" in result.stderr


def test_analyze_empty_corpus_gets_no_filter_note(tmp_path: Path) -> None:
    # deepseek NIT: the empty-corpus advice is the friendly line, not
    # 'run dataset export to capture filters'
    (tmp_path / "corpus").mkdir()
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 0
    assert "no filters.json snapshot" not in result.stderr
    assert "corpus is empty" in result.stdout


def test_analyze_nonobject_action_rejected(tmp_path: Path) -> None:
    # deepseek r4 MINOR: a filter whose action is unreadable must not
    # contribute zero needed-ids and let the posture claim 'snapshot'
    root = _seed_corpus(tmp_path, n=2)
    (root / "filters.json").write_text(
        '[{"criteria": {"from": "shop@x.z"}, "action": "nope"}]',
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["dataset", "analyze", "--config", str(_analyze_cfg(tmp_path))]
    )
    assert result.exit_code == 5
    assert "action must be an object" in result.stderr
