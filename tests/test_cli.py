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
        (["dataset", "export"], "dataset export"),
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
