"""gmail-labeler command-line interface.

Command surface is frozen by the Phase 0 design (AGENTS.md "Target layout");
later phases add train/eval/shadow/apply/daemon/status (absent until wired,
which surfaces as a usage error). Phase 0 subcommands are wired as they are
implemented; unimplemented ones exit 3 with a clear "not implemented" marker
— distinct from typer's usage-error code 2, so scripts can tell "not built
yet" from "typo" — and never silently succeed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from .attribution import SIDECAR_FILTERS, SIDECAR_LABELS, parse_attribution
from .gmail_import import DEFAULT_QUERY

if TYPE_CHECKING:
    from .config import Config
    from .gmail_client import GmailClient

app = typer.Typer(
    help="Strictly-local Gmail auto-labeling, trained on your own inbox.",
    epilog=(
        "Exit codes: 0 ok · 2 usage error · 3 not implemented · "
        "4 auth/config unusable · 5 operation failed or incomplete"
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


_OPERATION_ERROR_EXIT = 5  # operation failure (transport/corpus) or incomplete
# run — shared by all dataset commands (import-gmail, import-gyb,
# sync-labels, analyze)

# ADR-0008's FINAL recovery wording — the swap site ticket #73 documented
# and ticket #74 consumed: --full-resync is real, so the 404 guidance
# prints the exact recovery command and the birth NOTE recommends it. The
# CLI test pins PRESENCE of `dataset sync-labels --full-resync` (the #73
# vaporware-absence pin flipped alongside these constants).
_RESYNC_GUIDANCE = (
    "guidance: the Sync Cursor is outside Gmail's retained history \u2014 "
    "recovery is never automatic (ADR-0008). Run "
    "`dataset sync-labels --full-resync`: it reconciles the corpus's label "
    "state against the mailbox directly (no history needed), prints a cost "
    "estimate before any enumeration spend, and reseeds the cursor only "
    "after the reconciliation's revisions are durable."
)
_RESYNC_GUIDANCE_NOTE = (
    "Run `dataset sync-labels --full-resync` to reconcile label drift older "
    "than the retained history (cost-estimated up front)."
)


def _import_gmail_client(config: Config) -> GmailClient:
    """Client factory seam: tests inject FakeGmailClient here."""
    from .auth import credentials as build_credentials
    from .gmail_client import GmailRestClient

    return GmailRestClient(build_credentials(config))


@dataset_app.command("import-gmail")
def dataset_import_gmail(
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
    """Import labeled mail from the Gmail API into the corpus."""
    from .auth import AuthError
    from .corpus_store import CorpusError, CorpusStore
    from .gmail_errors import GmailAuthError, GmailError
    from .gmail_import import (
        fetch_settings,
        import_gmail,
        write_filters_snapshot,
        write_labels_snapshot,
    )
    from .tokens import TokenStoreError

    cfg = _load_config(config)
    # Auth gate FIRST: a not-signed-in run must not create or re-mode a
    # corpus tree (a typo'd corpus.dir especially).
    try:
        client = _import_gmail_client(cfg)
        id_to_name, filters, bootstrap = fetch_settings(client)
    except (AuthError, TokenStoreError) as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailAuthError as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailError as e:
        typer.echo(f"import failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    store = CorpusStore(cfg.corpus.dir)
    store.enforce_modes()
    for offender in store.verify_modes():
        typer.echo(f"WARNING: {offender} is group/world-readable", err=True)
    try:
        write_filters_snapshot(store, filters)
        write_labels_snapshot(store, id_to_name)
        stats = import_gmail(
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
        typer.echo(f"import failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    stats.retried += bootstrap.retried  # settings blips count too
    typer.echo(
        f"import-gmail: scanned={stats.scanned} written={stats.written} "
        f"skipped_present={stats.skipped_present} retried={stats.retried} "
        f"failed={len(stats.failed)}"
    )
    if stats.failed:
        for message_id, reason in stats.failed[:20]:
            typer.echo(f"  failed {message_id}: {reason}", err=True)
        if len(stats.failed) > 20:
            typer.echo(f"  ... and {len(stats.failed) - 20} more", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT)  # never silently partial


@dataset_app.command("import-gyb")
def dataset_import_gyb(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    path: str | None = typer.Option(
        None, "--path", help="GYB backup root (default: [gyb] path from config)"
    ),
    max_messages: int | None = typer.Option(
        None,
        "--max-messages",
        min=1,
        help="Stop after scanning N messages (smoke runs)",
    ),
    system_labels: str = typer.Option(
        "",
        "--system-labels",
        help=(
            "Comma-separated extra label names to treat as System Labels "
            "(e.g. 'INKASSIE'); the delimiter is a comma, so label names "
            "containing commas cannot be expressed via this flag"
        ),
    ),
) -> None:
    """Import a Got Your Back backup directory into the corpus.

    Holds a shared read lock on msg-db.sqlite for the whole run — finish
    an in-flight GYB backup first, or its writes will block.

    Fully offline — msg-db.sqlite is opened read-only (your backup is
    never mutated) and bodies go through the same MIME parser as import-gmail.
    """
    from .corpus_store import CorpusError, CorpusStore
    from .gyb_import import DB_FILENAME, GybImportError, import_gyb

    cfg = _load_config(config)
    if path is not None and not path.strip():
        typer.echo("import-gyb: --path is empty (not falling back silently)", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    root = Path(path).expanduser() if path else cfg.gyb.path
    if root is None:
        typer.echo(
            "import-gyb: no backup path — set [gyb] path in config or pass --path",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    if not (root / DB_FILENAME).is_file():
        typer.echo(f"import-gyb: {DB_FILENAME} not found under {root}", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    store = CorpusStore(cfg.corpus.dir)
    store.enforce_modes()
    for offender in store.verify_modes():
        typer.echo(f"WARNING: {offender} is group/world-readable", err=True)
    try:
        stats = import_gyb(
            root,
            store,
            max_messages=max_messages,
            system_labels_extra=frozenset(
                part.strip() for part in system_labels.split(",") if part.strip()
            ),
        )
    except (CorpusError, GybImportError) as e:
        # GybImportError: corrupt DB, foreign schema, locked file, index
        # gap, escaping filename, date drift — a loud whole-run stop,
        # never a silent partial import. Anything else (plain ValueError
        # etc.) is a REAL defect and must surface as one, not be laundered
        # into an operator message.
        partial = getattr(e, "stats", None)
        state = (
            f" — {partial.written} record(s) durably imported so far are "
            "kept; re-running resumes"
            if partial is not None
            else " — records durably imported so far are kept; re-running resumes"
        )
        typer.echo(f"import failed: {e}{state}", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    typer.echo(
        f"import-gyb: scanned={stats.scanned} written={stats.written} "
        f"skipped_present={stats.skipped_present} failed={len(stats.failed)}"
    )
    if stats.date_conflict:
        typer.echo(
            f"WARNING: {stats.date_conflict} imported row(s) whose folder "
            "date and stamp disagree by MORE than a day — the folder was "
            "NOT trusted and the raw stamp was kept; the backup index is "
            "inconsistent (investigate before trusting temporal splits)",
            err=True,
        )
    if stats.folder_unparsed:
        typer.echo(
            f"NOTE: {stats.folder_unparsed} row(s) had no year/month/day "
            "folder path — imported with the raw stamp and NO UTC anchor "
            "(their dates may be machine-local)",
            err=True,
        )
    if stats.date_corrected:
        typer.echo(
            f"NOTE: {stats.date_corrected} imported stamp(s) disagreed with "
            "the UTC folder date and were corrected to it (times may remain "
            "zone-skewed; a non-UTC backup machine is one possible cause)",
            err=True,
        )
    if stats.failed:
        for message_id, reason in stats.failed[:20]:
            typer.echo(f"  failed {message_id}: {reason}", err=True)
        if len(stats.failed) > 20:
            typer.echo(f"  ... and {len(stats.failed) - 20} more", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT)  # never silently partial


@dataset_app.command("sync-labels")
def dataset_sync_labels(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the change plan: identical report, no records or cursor written",
    ),
    full_resync: bool = typer.Option(
        False,
        "--full-resync",
        help=(
            "Ground-truth reconciliation for an expired Sync Cursor or a "
            "cursorless (GYB-shaped) corpus: estimates both strategies from "
            "one labels.list, prints the choice, runs the cheaper one (ties "
            "and unpriceable scans go to the exact-cost walk) — no history "
            "needed"
        ),
    ),
) -> None:
    """Refresh the corpus's label state from mailbox changes (Label Sync).

    Default path: walks the Gmail history from the corpus's Sync Cursor and
    appends a label Revision for every touched message — metadata only,
    bodies never (invariant #5). New mail is counted and reported, never
    ingested here; deleted mail keeps its record.

    --full-resync (recovery path, ticket #74): reconciles the corpus's
    label state against the mailbox directly when the history window is
    gone or was never consumed — cost-estimated up front from one
    labels.list plus local resolved-view counts, cheaper of the
    label-inventory scan and the corpus-metadata walk picked with a printed
    rationale, the cursor reseeded (from the seed captured BEFORE the scan)
    only after the reconciliation's revisions are durable (ADR-0008).
    """
    from .auth import AuthError
    from .corpus_store import CorpusError, CorpusStore
    from .gmail_errors import GmailAuthError, GmailError, GmailResyncError
    from .gmail_import import fetch_settings
    from .label_sync import FullResyncStats, SyncStats, sync_labels
    from .label_sync import full_resync as run_full_resync
    from .tokens import TokenStoreError

    cfg = _load_config(config)
    # Auth gate FIRST (import-gmail parity): a not-signed-in run must not
    # touch the corpus tree.
    try:
        client = _import_gmail_client(cfg)
        # Deliberate reuse of the landed bootstrap (spec #64 seam 3): the
        # filters fetch is unused here — a filters-only outage therefore
        # blocks sync too (loudly, exit 5). Accepted: one cheap shared call
        # keeps every dataset command on the SAME auth/retry spine; a
        # labels-only bootstrap is a refactor for the daemon phase.
        id_to_name, _filters, bootstrap = fetch_settings(client)
    except (AuthError, TokenStoreError) as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailAuthError as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailError as e:
        typer.echo(f"sync-labels failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    store = CorpusStore(cfg.corpus.dir)
    if not store.root.is_dir():
        # analyze's gate, same reasoning: a config typo is an operation
        # failure, NOT an "empty mailbox" — and never a birth target
        typer.echo(
            f"sync-labels failed: corpus dir {store.root} is missing or not a "
            "directory \u2014 run dataset import-gmail/import-gyb first or fix "
            "[corpus] dir",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    store.enforce_modes()  # sync-labels is a WRITE command (revisions + cursor)
    for offender in store.verify_modes():
        typer.echo(f"WARNING: {offender} is group/world-readable", err=True)
    try:
        if full_resync:
            # progress lines (estimate + strategy BEFORE any enumeration
            # spend; the scan's flag-rate BEFORE the metadata pass) print
            # as they happen — the flood guard is only a guard if it
            # lands early
            stats: SyncStats | FullResyncStats = run_full_resync(
                client,
                store,
                id_to_name=id_to_name,
                dry_run=dry_run,
                progress=typer.echo,
            )
        else:
            stats = sync_labels(client, store, id_to_name=id_to_name, dry_run=dry_run)
    except GmailResyncError as e:
        # the ledger is untouched (advancement is sync's act and this run
        # never reached it); recovery is never automatic (ADR-0008)
        typer.echo(f"sync-labels: {e}", err=True)
        typer.echo(_RESYNC_GUIDANCE, err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    except GmailAuthError as e:  # revoked token mid-run: the fix is human
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except (GmailError, CorpusError, OSError, UnicodeDecodeError) as e:
        # analyze's tuple: an undecodable/unreadable records index is exit
        # 5, never a raw traceback (a `1` would mean an unhandled crash)
        typer.echo(f"sync-labels failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    stats.retried += bootstrap.retried  # settings blips count too
    if full_resync:
        assert isinstance(stats, FullResyncStats)  # branch-shaped narrowing
        prefix = (
            "sync-labels --full-resync (dry-run):"
            if dry_run
            else "sync-labels --full-resync:"
        )
        # the flag-rate is the scan's drift signal; on the walk every id
        # is fetched by construction, so quoting a 100% "flag-rate" there
        # would read as drift that never happened (deepseek review)
        rate = f" (flag-rate {stats.flag_rate:.0%})" if stats.strategy == "scan" else ""
        typer.echo(
            f"{prefix} strategy={stats.strategy} "
            f"corpus={stats.corpus_messages} "
            f"flagged={stats.flagged}{rate} "
            f"revised={stats.revised} unchanged={stats.unchanged} "
            f"deleted_kept={stats.deleted_kept} new_match={stats.new_match} "
            f"retried={stats.retried} failed={len(stats.failed)}"
        )
        if stats.birthed:
            # honesty under dry_run: the plan says WOULD birth — nothing was
            # written (the core sets birthed from the read, not the write)
            typer.echo(
                "NOTE: this corpus had no Sync Cursor \u2014 full-resync "
                + (
                    "WOULD birth it from the profile seed captured BEFORE the scan"
                    if dry_run
                    else "birthed it from the profile seed captured BEFORE the scan"
                )
                + " (ADR-0008: changes made during the scan carry higher "
                "ids and land in the next sync's window)"
            )
    else:
        assert isinstance(stats, SyncStats)  # branch-shaped narrowing
        prefix = "sync-labels (dry-run):" if dry_run else "sync-labels:"
        typer.echo(
            f"{prefix} events={stats.events} stale={stats.stale} "
            f"revised={stats.revised} deleted_kept={stats.deleted_kept} "
            f"new_match={stats.new_match} retried={stats.retried} "
            f"failed={len(stats.failed)}"
        )
        if stats.birthed:
            # ADR-0008 consequence 1: the provenance disclosure — a fresh
            # cursor over a corpus whose label state may be months old
            typer.echo(
                "NOTE: this corpus had no Sync Cursor \u2014 it was birthed "
                "from the mailbox profile and applies NOTHING (no consumed "
                "baseline to diff against yet; the corpus's label state may "
                "predate the cursor). " + _RESYNC_GUIDANCE_NOTE
            )
    _render_gains_losses(stats.gains, stats.losses)
    if stats.new_match:
        typer.echo(
            f"NOTE: {stats.new_match} mailbox id(s) are not in the corpus \u2014 "
            "new mail is ingested by dataset import-gmail, never by sync "
            "(spec #64: ingest stays the acquisition paths' alone)",
            err=True,
        )
    if not stats.failed:
        # an incomplete run does not recommend analyzing a half-synced
        # corpus — it exits 5 and says re-run instead
        typer.echo(
            "\nNext: run `dataset analyze` for the updated candidate gates "
            "(ADR-0005 keeps retraining and applying human-gated)"
        )
    if stats.failed:
        for message_id, reason in stats.failed[:20]:
            typer.echo(f"  failed {message_id}: {reason}", err=True)
        if len(stats.failed) > 20:
            typer.echo(f"  ... and {len(stats.failed) - 20} more", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT)  # never silently partial


def _render_gains_losses(gains: dict[str, int], losses: dict[str, int]) -> None:
    """Per-label gain/loss table, shared by the event-walk and full-resync
    reports (the sync report closes with these counts, spec #64)."""
    if not (gains or losses):
        return
    labels = sorted(set(gains) | set(losses))
    width = min(max((len(name) for name in labels), default=5) + 1, 64)
    typer.echo(f"\n{'label':<{width}} {'gains':>6} {'losses':>6}")
    for name in labels:
        # NO truncation: slash-nested label paths must stay readable
        typer.echo(f"{name:<{width}} {gains.get(name, 0):>6} {losses.get(name, 0):>6}")


@dataset_app.command("analyze")
def dataset_analyze(
    config_file: str | None = typer.Option(
        None, "--config", help="Path to config.toml"
    ),
    recent_months: int = typer.Option(
        24, "--recent-months", min=1, help="Recency window for candidate gates"
    ),
    min_total: int = typer.Option(
        50, "--min-total", min=1, help="Minimum label volume"
    ),
    min_recent: int = typer.Option(
        10, "--min-recent", min=1, help="Minimum recent-window volume"
    ),
    assume_no_filters: bool = typer.Option(
        False,
        "--assume-no-filters",
        help="Attest the mailbox has no labeling filters (GYB-only users)",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Machine-readable output (graduation workflows)"
    ),
) -> None:
    """Per-label report: counts, recency, dup rate, filter attribution."""
    import dataclasses
    from datetime import UTC, datetime

    from .analyze import analyze_corpus
    from .corpus_store import CorpusError, CorpusStore

    cfg = _load_config(config_file)
    store = CorpusStore(cfg.corpus.dir)
    if not store.root.is_dir():
        # a config typo is an operation failure, NOT an "empty mailbox"
        typer.echo(
            f"analyze failed: corpus dir {store.root} is missing or not a "
            "directory \u2014 run dataset import-gmail/import-gyb first or fix "
            "[corpus] dir",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    for offender in store.verify_modes():
        typer.echo(f"WARNING: {offender} is group/world-readable", err=True)
    try:
        # The resolved view, not the raw log (ADR-0007): a revised message
        # counts once, under its newest label set — duplicate-id inflation
        # is eliminated by the fold, not guarded against here.
        records = list(store.iter_resolved())
    except (CorpusError, OSError, UnicodeDecodeError) as e:
        # unreadable/undecodable records are exit 5, never a traceback
        typer.echo(f"analyze failed: records unreadable ({type(e).__name__})", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    try:
        filters_text = store.read_snapshot(SIDECAR_FILTERS)
        # Short-circuit preserved verbatim from the pre-extraction reader:
        # with no filters snapshot, labels.json is never read, so a corrupt
        # bridge file cannot turn the honest 'missing' posture into a failure
        # (and parse_attribution decides before ever looking at it).
        labels_text = (
            store.read_snapshot(SIDECAR_LABELS) if filters_text is not None else None
        )
        posture, filters, id_to_name = parse_attribution(
            filters_text, labels_text, assume_no_filters=assume_no_filters
        )
    except (ValueError, OSError, UnicodeDecodeError) as e:
        # JSONDecodeError is a ValueError; wrong-shape payloads raise here
        typer.echo(
            f"analyze failed: snapshot unreadable ({type(e).__name__}: {e})",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    if assume_no_filters and filters_text is not None:
        # an attestation contradicting on-disk evidence is ignored, loudly;
        # the None-vs-not read of the snapshot is the existence check
        typer.echo(
            "WARNING: --assume-no-filters ignored: a filters.json snapshot "
            "exists (evidence beats attestation)",
            err=True,
        )

    summary, metrics = analyze_corpus(
        records,
        filters=filters,
        id_to_name=id_to_name,
        now=datetime.now(UTC),  # the CLI may read the clock; the core may not
        recent_months=recent_months,
        min_total=min_total,
        min_recent=min_recent,
    )
    # The posture NOTEs below matter only when there is mail to report on.
    # For an empty corpus the human path already ends with the
    # "corpus is empty — run dataset import-gmail/import-gyb first" line, and the
    # --json path prints no human report on stdout — its payload carries the
    # posture in filters_posture; NOTEs here would just bury the advice.
    if summary.messages:
        if posture == "bridge-incomplete":
            typer.echo(
                "NOTE: filters.json label ids are not all present in "
                "labels.json (missing or stale bridge) \u2014 attribution "
                "impossible, candidates withheld (invariant #6). Re-run "
                "`dataset import-gmail` to refresh both.",
                err=True,
            )
        elif filters is None:
            typer.echo(
                "NOTE: no filters.json snapshot \u2014 filter attribution "
                "unknown, candidates withheld (invariant #6). Run `dataset "
                "import-gmail` to capture it, or `--assume-no-filters` to attest "
                "a filterless mailbox.",
                err=True,
            )

    if as_json:
        # NOTE(intentional): labels[].reasons stay the ENGINE's vocabulary
        # here ('no-filter-snapshot' = what the core was told), while
        # summary.filters_posture names the real cause ('bridge-incomplete').
        # The table remaps for human eyes; JSON consumers read the posture
        # field — pinned by test_analyze_json_keeps_engine_reasons.
        payload = {
            "summary": dataclasses.asdict(summary) | {"filters_posture": posture},
            "labels": [dataclasses.asdict(m) for m in metrics],
        }

        def _default(o: object) -> str:
            if isinstance(o, datetime):  # ISO-8601, not str()'s space form
                return o.isoformat()
            return str(o)

        typer.echo(json.dumps(payload, default=_default, sort_keys=True))
        return

    if summary.messages == 0:
        typer.echo(
            "analyze: corpus is empty \u2014 run dataset import-gmail/import-gyb first"
        )
        return
    typer.echo(
        f"corpus: {summary.messages} messages, {summary.distinct_labels} labels, "
        f"{summary.duplicate_bodies} duplicated bodies, span "
        f"{summary.first_date.date() if summary.first_date else '-'} .. "
        f"{summary.last_date.date() if summary.last_date else '-'}"
    )
    typer.echo(f"filters: {posture}")
    width = min(max((len(m.label) for m in metrics), default=5) + 1, 64)
    typer.echo(
        f"{'label':<{width}} {'total':>6} {'recent':>6} {'dup%':>5} {'filt%':>5}  flags"
    )
    for m in metrics:
        reasons = m.reasons
        if posture == "bridge-incomplete":
            # the engine only sees filters=None; the CAUSE (broken id->name
            # bridge, not a missing snapshot) is display-side truth
            reasons = tuple(
                "bridge-incomplete" if r == "no-filter-snapshot" else r for r in reasons
            )
        flags = "CANDIDATE" if m.candidate else ",".join(reasons)
        # NO truncation: distinct >32-char labels must not collide on screen
        typer.echo(
            f"{m.label:<{width}} {m.total:>6} {m.recent:>6} "
            f"{m.dup_rate * 100:>4.0f}% {m.attribution_ratio * 100:>4.0f}%  {flags}"
        )
    candidates = [m.label for m in metrics if m.candidate]
    if candidates:
        typer.echo(f"\nManaged-Label candidates: {', '.join(candidates)}")
        typer.echo(
            "(candidates are a SHORTLIST for eval/graduation, never a config "
            "change \u2014 ADR-0005 keeps retraining and applying human-gated)"
        )


if __name__ == "__main__":
    app()
