"""gmail-labeler command-line interface.

Command surface is frozen by the Phase 0 design (AGENTS.md "Target layout");
later phases add train/eval/shadow/apply/daemon/status (absent until wired,
which surfaces as a usage error). Phase 0 subcommands are wired as they are
implemented; unimplemented ones exit 3 with a clear "not implemented" marker
— distinct from typer's usage-error code 2, so scripts can tell "not built
yet" from "typo" — and never silently succeed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from .export import DEFAULT_QUERY

if TYPE_CHECKING:
    from .config import Config
    from .gmail_client import GmailClient

app = typer.Typer(
    help="Strictly-local Gmail auto-labeling, trained on your own inbox.",
    epilog=(
        "Exit codes: 0 ok · 2 usage error · 3 not implemented · "
        "4 auth/config unusable · 5 export failed/incomplete"
    ),
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Corpus acquisition and analysis.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")

_NOT_IMPLEMENTED = "not implemented yet (Phase 0 lands this command)"


_NOT_IMPLEMENTED_EXIT = 3


def _stub(name: str) -> None:
    typer.echo(f"{name}: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(code=_NOT_IMPLEMENTED_EXIT)


auth_app = typer.Typer(help="Gmail OAuth credential (ADR-0004).", no_args_is_help=False)
app.add_typer(auth_app, name="auth")

_AUTH_ERROR_EXIT = 4  # 2 = usage, 3 = not implemented, 4 = auth/config unusable
# (typer requires concrete annotations: str | None, not Any)


@auth_app.callback(invoke_without_command=True)
def auth_status(
    ctx: typer.Context,
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
) -> None:
    """Report credential status; `auth login` runs the consent flow."""
    if ctx.invoked_subcommand is not None:
        return
    from .auth import status as _status

    cfg = _load_config(config)
    st = _status(cfg)
    state = "SIGNED IN" if st.signed_in else "NOT SIGNED IN"
    typer.echo(f"auth: {state} ({st.backend}) — {st.detail}")
    if not st.signed_in:
        raise typer.Exit(code=_AUTH_ERROR_EXIT)


@auth_app.command("login")
def auth_login(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
) -> None:
    """Run the one-time desktop OAuth consent flow."""
    from .auth import AuthError
    from .auth import login as _login
    from .tokens import TokenStoreError

    cfg = _load_config(config)
    try:
        _login(cfg)
    except (AuthError, TokenStoreError) as e:
        # Exit-code contract: login failures are 4 (auth-config unusable),
        # never raw tracebacks that could echo third-party exception text.
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    typer.echo("auth: signed in, refresh token stored")


def _load_config(config: str | None):
    from .config import ConfigError, load_config

    try:
        return load_config(config)
    except ConfigError as e:  # load_config raises nothing else (store errors
        # belong to auth status/login, which catch them where they arise)
        typer.echo(f"config error: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from e


_EXPORT_ERROR_EXIT = 5  # transport/corpus failure or incomplete run


def _export_client(config: Config) -> GmailClient:
    """Client factory seam: tests inject FakeGmailClient here."""
    from .auth import credentials as build_credentials
    from .gmail_client import GmailRestClient

    return GmailRestClient(build_credentials(config))


@dataset_app.command("export")
def dataset_export(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    query: str = typer.Option(
        DEFAULT_QUERY, "--query", help="Gmail search query (single-sourced)"
    ),
    max_messages: int | None = typer.Option(
        None,
        "--max-messages",
        min=1,
        help="Stop after scanning N messages (smoke runs)",
    ),
) -> None:
    """Export labeled mail from the Gmail API into the corpus."""
    from .auth import AuthError
    from .corpus_store import CorpusError, CorpusStore
    from .export import (
        export_corpus,
        fetch_settings,
        write_filters_snapshot,
        write_labels_snapshot,
    )
    from .gmail_errors import GmailAuthError, GmailError
    from .tokens import TokenStoreError

    cfg = _load_config(config)
    # Auth gate FIRST: a not-signed-in run must not create or re-mode a
    # corpus tree (a typo'd corpus.dir especially).
    try:
        client = _export_client(cfg)
        id_to_name, filters, bootstrap = fetch_settings(client)
    except (AuthError, TokenStoreError) as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailAuthError as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailError as e:
        typer.echo(f"export failed: {e}", err=True)
        raise typer.Exit(code=_EXPORT_ERROR_EXIT) from None
    store = CorpusStore(cfg.corpus.dir)
    store.enforce_modes()
    for offender in store.verify_modes():
        typer.echo(f"WARNING: {offender} is group/world-readable", err=True)
    try:
        write_filters_snapshot(store, filters)
        write_labels_snapshot(store, id_to_name)
        stats = export_corpus(
            client,
            store,
            id_to_name=id_to_name,
            query=query,
            max_messages=max_messages,
        )
    except GmailAuthError as e:  # revoked token mid-run: the fix is human
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except (GmailError, CorpusError) as e:
        typer.echo(f"export failed: {e}", err=True)
        raise typer.Exit(code=_EXPORT_ERROR_EXIT) from None
    stats.retried += bootstrap.retried  # settings blips count too
    typer.echo(
        f"export: scanned={stats.scanned} written={stats.written} "
        f"skipped_present={stats.skipped_present} retried={stats.retried} "
        f"failed={len(stats.failed)}"
    )
    if stats.failed:
        for message_id, reason in stats.failed[:20]:
            typer.echo(f"  failed {message_id}: {reason}", err=True)
        if len(stats.failed) > 20:
            typer.echo(f"  ... and {len(stats.failed) - 20} more", err=True)
        raise typer.Exit(code=_EXPORT_ERROR_EXIT)  # never silently partial


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
