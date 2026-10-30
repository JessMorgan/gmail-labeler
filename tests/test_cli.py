from __future__ import annotations

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


@pytest.mark.parametrize(
    ("argv", "name"),
    [
        (["dataset", "import-gyb"], "dataset import-gyb"),
        (["dataset", "analyze"], "dataset analyze"),
    ],
)
def test_every_stub_exits_3_with_marker(argv: list[str], name: str) -> None:
    """No stub may silently succeed: exit 3 (not the usage-error 2) and the
    per-command marker on stderr."""
    result = runner.invoke(app, argv)
    assert result.exit_code == 3
    assert f"{name}: not implemented yet" in result.stderr


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
