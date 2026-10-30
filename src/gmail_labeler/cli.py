"""gmail-labeler command-line interface.

Command surface is frozen by the Phase 0 design (AGENTS.md "Target layout");
later phases add train/eval/shadow/apply/daemon/status (absent until wired,
which surfaces as a usage error). Phase 0 subcommands are wired as they are
implemented; unimplemented ones exit 3 with a clear "not implemented" marker
— distinct from typer's usage-error code 2, so scripts can tell "not built
yet" from "typo" — and never silently succeed.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    help="Strictly-local Gmail auto-labeling, trained on your own inbox.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Corpus acquisition and analysis.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")

_NOT_IMPLEMENTED = "not implemented yet (Phase 0 lands this command)"


_NOT_IMPLEMENTED_EXIT = 3


def _stub(name: str) -> None:
    typer.echo(f"{name}: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(code=_NOT_IMPLEMENTED_EXIT)


@app.command()
def auth() -> None:
    """Set up and check the Gmail OAuth credential (ADR-0004)."""
    _stub("auth")


@dataset_app.command("export")
def dataset_export() -> None:
    """Export labeled mail from the Gmail API into the corpus."""
    _stub("dataset export")


@dataset_app.command("import-gyb")
def dataset_import_gyb() -> None:
    """Import a Got Your Back backup directory into the corpus."""
    _stub("dataset import-gyb")


@dataset_app.command("analyze")
def dataset_analyze() -> None:
    """Per-label report: counts, filter attribution, dup rate, baselines."""
    _stub("dataset analyze")


if __name__ == "__main__":
    app()
