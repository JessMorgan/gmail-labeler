from __future__ import annotations

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
        (["auth"], "auth"),
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
