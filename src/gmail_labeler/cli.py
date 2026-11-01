"""gmail-labeler command-line interface.

Command surface is frozen by the Phase 0 design (AGENTS.md "Layout");
later phases add shadow/apply/daemon/status (absent until wired, which
surfaces as a usage error). Phase 0 subcommands and the Phase 1
``train``/``eval`` commands are wired as they are implemented. Exit 3
is reserved for wired-but-stubbed commands and currently UNREACHABLE —
no command is stubbed; scripts can branch on 0/2/4/5 today.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import typer

from .analyze import filter_label_ids_strict
from .dataset_build import TrainingExample
from .export import DEFAULT_QUERY, SIDECAR_FILTERS, SIDECAR_LABELS
from .model import Model

# summary.filters_posture values, pinned for --json consumers
AttributionPosture = Literal[
    "snapshot", "attested-none", "missing", "bridge-incomplete"
]

if TYPE_CHECKING:
    from .config import Config
    from .gmail_client import GmailClient
    from .thresholds import BandsArtifact

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


def _write_secret_tight(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` 0600 FROM CREATION (no 0644 window —
    a write-then-chmod leaves a moment where the artifact is
    world-readable) and fchmod'd if the file pre-existed loose. The
    corpus-sidecar discipline for identity/report artifacts (the
    manifest, the sidecar, the eval report)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as fh:  # owns fd
        fh.write(text)


class BandsTargetRefusedError(Exception):
    """The existing --bands target is a verifiably-different Bands
    Artifact: replacing it would destroy identity material (the
    serving-gate hardening — a clobber by a drifted run is a lie)."""


def _write_bands_artifact(path: Path, artifact: BandsArtifact, text: str) -> None:
    """Write the Bands Artifact (ticket #21): 0600 from creation,
    EXCLUSIVE staging + atomic replace — a failure leaves any PREVIOUS
    artifact intact, and no fd can leak. The staging file is
    ``tempfile.mkstemp``'d (a unique name per process, 0600 from
    creation, fsync'd before publish), so two concurrent ``eval
    --bands`` runs on the same path can never interleave on a shared
    staging name and rename each other's partial bytes into place; the
    final ``os.replace`` is atomic, so each run publishes its own
    complete bytes.

    Before replacing an EXISTING artifact, its identity is checked
    against the current run (the serving-gate hardening): a parsed
    artifact whose identity fields disagree is refused
    (``BandsTargetRefusedError`` — exit 5, BEFORE any staging); an
    unreadable/garbage target is NOT identity material — the replace
    NOTICE proceeds (the artifact is regenerable by design, the
    recompute posture). Replacing an existing artifact is announced on
    stderr (the #20 ``--out`` NOTICE — a silent clobber of a typo'd
    path would be a lie). After the replace, the on-disk bytes are read
    back through ``BandsArtifact.from_json`` (train's sidecar readback
    symmetry) and compared by CONTENT HASH, not dataclass equality —
    the canonical payload serializes reasons sorted, so equality must
    be order-insensitive too (a reasons-order difference is the same
    meaning, not a torn write). A corrupt/torn file is refused by
    ``from_json`` ITSELF (ValueError — mapped to the same named
    operation failure, never a traceback); the hash-mismatch branch
    is the concurrent-writer signature (a valid artifact whose
    content hash differs — it DID parse); the OSError branch guards
    I/O failures around the publish (a concurrent writer, a disk
    error) — it is NOT the torn-write guard. The ONLY writer of the
    artifact in the package
    (recompute posture, AC4 — pinned by
    ``test_only_eval_writes_the_bands_artifact``)."""
    from .thresholds import BandsArtifact

    if path.is_file():
        try:
            existing = BandsArtifact.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            # an unreadable/garbage target is not identity material —
            # the replace NOTICE proceeds (regenerable by design)
            typer.echo(f"eval: NOTICE: replacing existing {path}", err=True)
        else:
            disagreeing = [
                field
                for field in (
                    "vocabulary_hash",
                    "corpus_fingerprint",
                    "split_manifest_hash",
                    "encoder_id",
                    "split_date",
                    "test_examples",
                )
                if getattr(existing, field) != getattr(artifact, field)
            ]
            if disagreeing:
                raise BandsTargetRefusedError(
                    f"existing artifact at {path} is bound to "
                    f"{', '.join(disagreeing)} — not this checkpoint"
                )
            typer.echo(f"eval: NOTICE: replacing existing {path}", err=True)
    fd, staging_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    staging = Path(staging_name)
    try:
        fh = os.fdopen(fd, "w", encoding="utf-8")  # owns fd from here
    except OSError:
        os.close(fd)  # fdopen failed — the fd is ours to close
        staging.unlink(missing_ok=True)
        raise
    try:
        with fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(staging, str(path))  # atomic on POSIX
        # readback symmetry (train's sidecar pattern): the on-disk
        # bytes must parse back to the minted artifact, compared by
        # CONTENT HASH (order-insensitive — see the docstring).
        # from_json refuses a corrupt/torn file FIRST (ValueError —
        # self-inconsistent bytes never reach the hash comparison);
        # that refusal is the same named operation failure, not a
        # traceback. The OSError branch below is the I/O guard
        # (concurrent writer, disk error) — not the torn-write guard.
        try:
            readback = BandsArtifact.from_json(path.read_text(encoding="utf-8"))
        except ValueError as e:
            raise OSError(
                f"artifact readback fault: {path} does not parse back "
                f"to a valid artifact ({type(e).__name__})"
            ) from e
        if readback.content_hash != artifact.content_hash:
            # it DID parse — a valid artifact whose content hash
            # differs: a concurrent writer published a different
            # artifact (the fault branch above is the torn-write one)
            raise OSError(
                f"artifact readback mismatch: {path} is a valid "
                "artifact whose content hash differs from the minted "
                "one (a concurrent writer published a different "
                "artifact)"
            )
    except OSError:
        staging.unlink(missing_ok=True)  # best-effort staging cleanup
        raise


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
        DEFAULT_QUERY,
        "--query",
        help=(
            "Gmail search query (single-sourced); replaces the default; "
            "DRAFT/CHAT ingest rejection still applies"
        ),
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
        f"skipped_present={stats.skipped_present} "
        f"skipped_draft_chat={stats.skipped_draft_chat} "
        f"retried={stats.retried} failed={len(stats.failed)}"
    )
    if stats.failed:
        for message_id, reason in stats.failed[:20]:
            typer.echo(f"  failed {message_id}: {reason}", err=True)
        if len(stats.failed) > 20:
            typer.echo(f"  ... and {len(stats.failed) - 20} more", err=True)
        raise typer.Exit(code=_EXPORT_ERROR_EXIT)  # never silently partial


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
    never mutated) and bodies go through the same MIME parser as export.
    """
    from .corpus_store import CorpusError, CorpusStore
    from .gyb_import import DB_FILENAME, GybImportError, import_gyb

    cfg = _load_config(config)
    if path is not None and not path.strip():
        typer.echo("import-gyb: --path is empty (not falling back silently)", err=True)
        raise typer.Exit(code=_EXPORT_ERROR_EXIT) from None
    root = Path(path).expanduser() if path else cfg.gyb.path
    if root is None:
        typer.echo(
            "import-gyb: no backup path — set [gyb] path in config or pass --path",
            err=True,
        )
        raise typer.Exit(code=_EXPORT_ERROR_EXIT) from None
    if not (root / DB_FILENAME).is_file():
        typer.echo(f"import-gyb: {DB_FILENAME} not found under {root}", err=True)
        raise typer.Exit(code=_EXPORT_ERROR_EXIT) from None
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
        raise typer.Exit(code=_EXPORT_ERROR_EXIT) from None
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
        raise typer.Exit(code=_EXPORT_ERROR_EXIT)  # never silently partial


def _load_attribution(
    store_root: Path, *, assume_no_filters: bool
) -> tuple[AttributionPosture, list[dict[str, object]] | None, dict[str, str]]:
    """(posture, filters, id_to_name) for the analyze report.

    Sidecars are untrusted on-disk state: parseable-but-wrong-shape raises
    ValueError (the command maps it to exit 5) — never an AttributeError
    escaping into a traceback. A filters.json whose label ids are NOT all
    resolvable through labels.json cannot attribute anything, and the
    filter-authored labels would surface as CANDIDATE — invariant #6's
    unsafe direction, silent and wrong. Such a snapshot demotes to the
    withheld posture 'bridge-incomplete' (deepseek CRITICAL).
    """
    sidecar = store_root / SIDECAR_FILTERS
    if not sidecar.is_file():
        if assume_no_filters:
            return "attested-none", [], {}
        return "missing", None, {}
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("filters snapshot is not a list")
    filters: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict) or not all(isinstance(k, str) for k in item):
            raise ValueError("filter entry is not a string-keyed object")
        filters.append({str(k): v for k, v in item.items()})
    id_to_name: dict[str, str] = {}
    labels_path = store_root / SIDECAR_LABELS
    if labels_path.is_file():
        raw_map = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(raw_map, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in raw_map.items()
        ):
            raise ValueError("labels snapshot is not an id->name object")
        id_to_name = {str(k): str(v) for k, v in raw_map.items()}
    needed: set[str] = set()
    for f in filters:
        # the shape rule lives in analyze.filter_label_ids_strict (shared
        # with the dataset builder): unreadable label bindings must not
        # contribute zero `needed` ids and let the posture claim
        # 'snapshot' — the same failure direction the non-str/non-list
        # guards closed (deepseek r3 / qwen MINOR)
        needed.update(filter_label_ids_strict(f))
    if needed - set(id_to_name):
        return "bridge-incomplete", None, {}
    return "snapshot", filters, id_to_name


@dataset_app.command("analyze")
def dataset_analyze(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
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

    cfg = _load_config(config)
    store = CorpusStore(cfg.corpus.dir)
    if not store.root.is_dir():
        # a config typo is an operation failure, NOT an "empty mailbox"
        typer.echo(
            f"analyze failed: corpus dir {store.root} is missing or not a "
            "directory \u2014 run dataset export/import-gyb first or fix "
            "[corpus] dir",
            err=True,
        )
        raise typer.Exit(code=_EXPORT_ERROR_EXIT) from None
    for offender in store.verify_modes():
        typer.echo(f"WARNING: {offender} is group/world-readable", err=True)
    try:
        records = list(store.iter_records())
    except (CorpusError, OSError, UnicodeDecodeError) as e:
        # unreadable/undecodable records are exit 5, never a traceback
        typer.echo(f"analyze failed: records unreadable ({type(e).__name__})", err=True)
        raise typer.Exit(code=_EXPORT_ERROR_EXIT) from None
    try:
        posture, filters, id_to_name = _load_attribution(
            store.root, assume_no_filters=assume_no_filters
        )
    except (ValueError, OSError, UnicodeDecodeError) as e:
        # JSONDecodeError is a ValueError; wrong-shape payloads raise here
        typer.echo(
            f"analyze failed: snapshot unreadable ({type(e).__name__}: {e})",
            err=True,
        )
        raise typer.Exit(code=_EXPORT_ERROR_EXIT) from None
    if assume_no_filters and (store.root / SIDECAR_FILTERS).is_file():
        # an attestation contradicting on-disk evidence is ignored, loudly
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
    if summary.messages:  # empty corpus: the friendly line IS the advice
        if posture == "bridge-incomplete":
            typer.echo(
                "NOTE: filters.json label ids are not all present in "
                "labels.json (missing or stale bridge) \u2014 attribution "
                "impossible, candidates withheld (invariant #6). Re-run "
                "`dataset export` to refresh both.",
                err=True,
            )
        elif filters is None:
            typer.echo(
                "NOTE: no filters.json snapshot \u2014 filter attribution "
                "unknown, candidates withheld (invariant #6). Run `dataset "
                "export` to capture it, or `--assume-no-filters` to attest "
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
            "analyze: corpus is empty \u2014 run dataset export/import-gyb first"
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


# Operation failure (train + eval share the one home): build refusal,
# checkpoint refusal, unreadable corpus, recipe failure
_OPERATION_FAILED_EXIT = 5


def _model_factory(
    cfg: Config,
    examples: Sequence[TrainingExample],
    texts: Sequence[str],
    vocabulary: Sequence[str],
    checkpoint_dir: Path,
    device: str,
) -> Model:
    """Model factory seam: tests inject a FakeModel producer here (the
    offline spine, invariant #9). The real implementation lazy-imports
    the ``train/`` recipe (torch — the optional extra), so the CLI
    module itself stays importable without torch. The encoder-path
    guard runs BEFORE the import: a config typo must not pay a torch
    import (or crash on its absence) to learn it is missing.
    ``texts`` is ALIGNED to ``examples`` (index i ↔ examples[i])."""
    if cfg.model.encoder_path is None:
        raise ValueError(
            "model.encoder_path is not set — training downloads nothing; "
            "point [model] encoder_path at local encoder weights"
        )
    from train.recipe import train_encoder

    return train_encoder(
        examples,
        texts,
        vocabulary,
        encoder_id=cfg.model.encoder,
        encoder_path=cfg.model.encoder_path,
        output_dir=checkpoint_dir,
        device=device,
    )


@app.command("train")
def train(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    split_date: str = typer.Option(
        ...,
        "--split-date",
        help=(
            "ISO date (YYYY-MM-DD): train is strictly before it, test at/"
            "after (invariant #7) — the temporal split is pinned in the "
            "Split Manifest the run mints"
        ),
    ),
    assume_no_filters: bool = typer.Option(
        False,
        "--assume-no-filters",
        help="Attest the mailbox has no labeling filters (GYB-only users)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Replace an existing checkpoint. Retraining is human-gated "
            "(ADR-0005): a silent overwrite would destroy the artifact a "
            "graduated label serves from, so an existing checkpoint dir "
            "refuses the run until --force is explicit. The replacement "
            "is staging-based: the new checkpoint is built in "
            "<dir>.tmp, its sidecar verified by readback, then swapped "
            "in — a crash leaves the OLD checkpoint intact, an EMPTY "
            "dir, or the NEW one; each state is refusal-safe, and no "
            "stale encoder file can survive into the new dir"
        ),
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Training device: cpu or cuda (single-device contract — a "
        "multi-GPU host is refused; set CUDA_VISIBLE_DEVICES to one)",
    ),
) -> None:
    """Fine-tune the encoder on the corpus; write checkpoint + sidecar.

    Fully local (AC#6): the corpus comes from disk, the encoder weights
    from a local path — no network at all. The run mints the Split
    Manifest, rebuilds the examples THROUGH it (manifest mode — the
    manifest is authoritative, invariant #7), trains behind the Model
    seam, and writes the JSON Checkpoint Sidecar pinning the run's
    identity (vocabulary hash, thresholds source, corpus fingerprint,
    manifest hash, encoder id).
    """
    import shutil
    from datetime import UTC, date, datetime

    from .corpus_record import corpus_fingerprint, preprocess_text
    from .corpus_store import CorpusError, CorpusStore
    from .dataset_build import BuildAbortedError, build_dataset
    from .model import (
        MANIFEST_FILENAME,
        SIDECAR_FILENAME,
        SIDECAR_FORMAT_VERSION,
        THRESHOLDS_SOURCE_NONE,
        CheckpointSidecar,
        vocabulary_hash,
    )

    cfg = _load_config(config)
    # strict DATE: time components are a usage error, not a split
    try:
        d = date.fromisoformat(split_date)
    except ValueError:
        typer.echo(
            f"train failed: --split-date {split_date!r} is not an ISO date "
            "(YYYY-MM-DD — no time component)",
            err=True,
        )
        raise typer.Exit(code=2) from None
    split_dt = datetime(d.year, d.month, d.day, tzinfo=UTC)  # a date is a UTC instant
    if device not in ("cpu", "cuda"):
        typer.echo(
            f"train failed: --device {device!r} must be 'cpu' or 'cuda'",
            err=True,
        )
        raise typer.Exit(code=2) from None

    store = CorpusStore(cfg.corpus.dir)
    if not store.root.is_dir():
        typer.echo(
            f"train failed: corpus dir {store.root} is missing or not a "
            "directory \u2014 run dataset export/import-gyb first or fix "
            "[corpus] dir",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        records = list(store.iter_records())
    except (CorpusError, OSError, UnicodeDecodeError):
        typer.echo("train failed: records unreadable", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if not records:
        typer.echo(
            "train failed: corpus is empty \u2014 run dataset export/import-gyb first",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    for offender in store.verify_modes():
        typer.echo(f"WARNING: {offender} is group/world-readable", err=True)
    try:
        posture, filters, id_to_name = _load_attribution(
            store.root, assume_no_filters=assume_no_filters
        )
    except (ValueError, OSError, UnicodeDecodeError) as e:
        typer.echo(
            f"train failed: snapshot unreadable ({type(e).__name__}: {e})",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if posture in ("missing", "bridge-incomplete"):
        # the build refuses (invariant #6) — name the REAL cause
        detail = (
            "no filters.json snapshot \u2014 run `dataset export` to "
            "capture it, or --assume-no-filters to attest a filterless "
            "mailbox"
            if posture == "missing"
            else "filters.json label ids are not all present in labels.json "
            "(missing or stale bridge) \u2014 re-run `dataset export` to "
            "refresh both"
        )
        typer.echo(f"train failed: build refused (invariant #6): {detail}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the existing-checkpoint refusal is hoisted BEFORE the two
    # corpus-scale build passes: a --force-less run on a 394k corpus
    # must not pay the builds to learn it is refused
    checkpoint_dir = cfg.model.checkpoint_dir
    sidecar_path = checkpoint_dir / SIDECAR_FILENAME
    if sidecar_path.is_file() and not force:
        typer.echo(
            f"train failed: checkpoint {checkpoint_dir} already exists "
            "(sidecar present) \u2014 retraining is human-gated (ADR-0005); "
            "pass --force to replace it",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()) and not force:
        typer.echo(
            f"train failed: checkpoint dir {checkpoint_dir} is not empty "
            "\u2014 refusing to overwrite; pass --force to replace",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    now = datetime.now(UTC)  # the CLI may read the clock; the core may not
    try:
        minted = build_dataset(
            records,
            filters=filters,
            id_to_name=id_to_name,
            now=now,
            split_date=split_dt,
            server_authoritative_only=True,
        )
        result = build_dataset(
            records,
            filters=filters,
            id_to_name=id_to_name,
            now=now,
            split_date=split_dt,
            manifest=minted.manifest,
        )
    except BuildAbortedError as e:
        typer.echo(f"train failed: build refused (invariant #6): {e.reason}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except ValueError as e:
        typer.echo(f"train failed: build refused: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    # the minted example list is no longer needed (result carries the
    # same manifest object) — drop it before the corpus-scale work below
    del minted

    train_ids = set(result.manifest.train_ids)
    examples = [example for example in result.examples if example.id in train_ids]
    if not examples:
        typer.echo(
            "train failed: no training examples before the split date \u2014 "
            "pick a LATER --split-date (train is strictly before it)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # preprocessing integrity (invariant #4): train and inference share
    # corpus_record.preprocess_text — the subject + body the store keeps
    by_id = {record.id: record for record in records}
    missing_bodies = 0
    texts: list[str] = []  # aligned to `examples` (index i ↔ examples[i])
    try:
        for example in examples:
            body = store.get_body(example.id)
            if body is None:
                missing_bodies += 1  # an orphaned shard (crash residue)
                body = ""
            texts.append(preprocess_text(by_id[example.id].subject, body))
    except (OSError, UnicodeDecodeError) as e:
        # a corrupt/undecodable shard is an operation failure, not a bug
        typer.echo(f"train failed: a body shard is unreadable: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if missing_bodies:
        typer.echo(
            f"WARNING: {missing_bodies} training record(s) had no body shard "
            "— trained on subject + trailing newline (crash residue; "
            "re-export to repair)",
            err=True,
        )
    vocabulary = result.manifest.vocabulary

    # staging-based replacement: the new checkpoint is built in
    # <dir>.tmp, its identity verified by readback, then swapped in.
    # Crash outcomes are each refusal-safe: mid-training → the OLD dir
    # is intact (or nothing exists); between rmtree and rename → an
    # EMPTY/absent dir (no identity); after rename → the NEW complete
    # dir. No stale encoder file can survive into the new dir.
    staging_dir = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    try:
        if staging_dir.exists():
            # leftover from a crashed run: garbage, the real checkpoint
            # dir is untouched — remove it
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(staging_dir, 0o700)  # BEFORE training: mail-derived
        # weights must not sit world-readable in a crash window
    except OSError as e:
        typer.echo(f"train failed: checkpoint dir unusable: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    try:
        model = _model_factory(cfg, examples, texts, vocabulary, staging_dir, device)
    except ImportError as e:
        # DECISION (both review families): the recipe's failure
        # taxonomy is environment/operation, not code. ImportError
        # (the optional `train` extra missing, or the recipe's own
        # import broken by a version skew), ValueError/OSError (bad
        # local weights, an unreadable file), and RuntimeError (a
        # missing device) all map to exit 5 with the message echoed.
        # Anything ELSE escaping (TypeError, AttributeError, ...) is a
        # CODE BUG: it stays a traceback and exits 1, which the
        # contract reserves for "unhandled crash — report it"
        # (README).
        typer.echo(
            f"train failed: the model recipe is unavailable ({e}) \u2014 install "
            "the optional extra: `uv sync --extra train` (if it is already "
            "installed, the recipe import itself is broken — report it)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except (ValueError, OSError, RuntimeError) as e:
        # recipe-level failures (bad local weights, a corrupt checkpoint,
        # a missing device) are operation failures, not code bugs
        typer.echo(f"train failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    # the sidecar must describe the TRAINER'S product, not the CLI's
    # claim: a vocabulary mismatch is a refusal, not a silent overwrite
    if model.vocabulary != tuple(vocabulary):
        typer.echo(
            "train failed: the trainer's vocabulary does not match the "
            "manifest's — refusing to write a lying sidecar",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    sidecar = CheckpointSidecar(
        format_version=SIDECAR_FORMAT_VERSION,
        vocabulary_hash=vocabulary_hash(vocabulary),
        thresholds_source=THRESHOLDS_SOURCE_NONE,
        # TRAIN-SPLIT scoped: append-only corpus growth must not move
        # the serving identity (#25's lifeline)
        corpus_fingerprint=corpus_fingerprint(records, result.manifest.train_ids),
        split_manifest_hash=result.manifest.content_hash,
        encoder_id=cfg.model.encoder,
    )
    manifest_path = staging_dir / MANIFEST_FILENAME
    sidecar_path = staging_dir / SIDECAR_FILENAME
    try:
        # the Split Manifest is materialized with the checkpoint: train/
        # eval rebuild against it (invariant #7), and its content hash is
        # independently verifiable against the sidecar — 0600 from
        # creation (no 0644 window)
        _write_secret_tight(manifest_path, result.manifest.to_json())
        _write_secret_tight(sidecar_path, sidecar.to_json())
        # readback verification: the on-disk identity must parse and
        # equal what we intended — a torn write is a refusal, not a
        # served lie
        if (
            CheckpointSidecar.from_json(sidecar_path.read_text(encoding="utf-8"))
            != sidecar
        ):
            typer.echo(
                "train failed: sidecar readback mismatch — the on-disk "
                "identity does not parse back to what was written",
                err=True,
            )
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        if checkpoint_dir.exists():
            # --force: the old checkpoint is replaced WHOLESALE — stale
            # encoder-specific files (tokenizer, weight shards) cannot
            # survive into the new dir
            shutil.rmtree(checkpoint_dir)
        staging_dir.rename(checkpoint_dir)
        os.chmod(checkpoint_dir, 0o700)
    except OSError as e:
        typer.echo(f"train failed: sidecar write failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    typer.echo(
        f"train: examples={len(examples)} vocab={len(vocabulary)} "
        f"checkpoint={checkpoint_dir} sidecar=ok"
    )


def _eval_model_factory(checkpoint_dir: Path, vocabulary: tuple[str, ...]) -> Model:
    """Model factory seam for ``eval`` (the ``load_model`` factory
    shape): tests inject a FakeModel producer (the offline spine,
    invariant #9). The real implementation lazy-imports the ``train/``
    recipe (torch — the optional extra), so the CLI module itself stays
    importable without torch."""
    from train.recipe import load_hf_model

    return load_hf_model(checkpoint_dir, vocabulary)


@app.command("eval")
def eval(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    assume_no_filters: bool = typer.Option(
        False,
        "--assume-no-filters",
        help="Attest the mailbox has no labeling filters (GYB-only users)",
    ),
    out: str | None = typer.Option(
        None,
        "--out",
        help="Write the report to PATH (the same bytes as stdout)",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Machine-readable output (graduation workflows)"
    ),
    bands: str | None = typer.Option(
        None,
        "--bands",
        help="Write the calibrated Threshold Bands artifact to PATH "
        "(opt-in; replacing an existing file is announced)",
    ),
) -> None:
    """Score the temporal test split; report per-label P/R at candidate
    thresholds.

    The checkpoint is sidecar-validated (refusal contract — a mismatch
    refuses, no silent fallback); the test-side membership is rebuilt
    THROUGH the checkpoint's Split Manifest (invariant #7 — the split
    is pinned, never re-derived, so there is no --split-date);
    predictions run through the Model seam. Nothing goes near the
    mailbox: the report is labels, counts, and probabilities only.

    ``--bands`` is the deliberate act that calibrates the per-label
    Threshold Bands from the measured table and writes the versioned
    artifact (ticket #21) — bound to the checkpoint + eval identity,
    with the config's per-label band overrides applied. It is opt-in
    (human-gated, ADR-0005: writing/updating the serving bands is a
    decision, never a side effect of a report run); ``eval`` alone
    never writes it.
    """
    import contextlib
    from datetime import UTC, datetime

    from .corpus_record import corpus_fingerprint, preprocess_text
    from .corpus_store import CorpusError, CorpusStore
    from .dataset_build import BuildAbortedError, SplitManifest, build_dataset
    from .eval import eval_metrics, render_report
    from .model import (
        MANIFEST_FILENAME,
        THRESHOLDS_SOURCE_NONE,
        CheckpointRefusedError,
        ExpectedIdentity,
        load_model,
    )

    cfg = _load_config(config)
    # usage refusals up front (exit 2, before any corpus-scale work):
    # an empty --bands would degenerate to Path("") and fail awkwardly
    # mid-run; --bands and --out naming the same file would let the
    # artifact write silently destroy the report; --bands naming the
    # config file would destroy identity material this very run just
    # read; --bands INSIDE the checkpoint dir (manifest, sidecar,
    # weights) or the corpus store (records index, labels/filters
    # sidecars, bodies) would clobber identity files the garbage-
    # tolerance NOTICE branch cannot see — a destructive collision,
    # not a regenerable-report NOTICE
    if bands is not None:
        if not bands.strip():
            typer.echo("eval failed: --bands PATH must be non-empty", err=True)
            raise typer.Exit(code=2) from None
        if out is not None and Path(bands).resolve() == Path(out).resolve():
            typer.echo(
                "eval failed: --bands and --out must name different files",
                err=True,
            )
            raise typer.Exit(code=2) from None
        bands_path = Path(bands).resolve()
        if bands_path.is_dir():
            # an existing directory at the target would fail at
            # os.replace (EISDIR) AFTER the report — a usage error up
            # front, like the other usage refusals
            typer.echo(
                f"eval failed: --bands {bands!r} names an existing "
                f"directory ({bands_path}) \u2014 the artifact is a file",
                err=True,
            )
            raise typer.Exit(code=2) from None
        from .config import find_config

        # the artifact must not live beside identity files: ANY target
        # inside the checkpoint dir or the corpus store is refused
        # (both sides RESOLVED — a symlink whose name sits outside but
        # points inside is caught). The canonical artifact location
        # lands with the daemon phase; until then, any other path.
        for root, what in (
            (cfg.model.checkpoint_dir.resolve(), "the checkpoint directory"),
            (Path(cfg.corpus.dir).resolve(), "the corpus store"),
        ):
            if bands_path == root or root in bands_path.parents:
                typer.echo(
                    f"eval failed: --bands {bands!r} lives inside {what} "
                    f"({root}) \u2014 the artifact must not live beside "
                    "identity files; use any other path",
                    err=True,
                )
                raise typer.Exit(code=2) from None
        if bands_path == find_config(config).resolve():
            typer.echo(
                f"eval failed: --bands {bands!r} names the config file "
                f"({find_config(config).resolve()}) \u2014 refusing to "
                "destroy it",
                err=True,
            )
            raise typer.Exit(code=2) from None
    store = CorpusStore(cfg.corpus.dir)
    if not store.root.is_dir():
        typer.echo(
            f"eval failed: corpus dir {store.root} is missing or not a "
            "directory \u2014 run dataset export/import-gyb first or fix "
            "[corpus] dir",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        records = list(store.iter_records())
    except (CorpusError, OSError, UnicodeDecodeError):
        typer.echo("eval failed: records unreadable", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if not records:
        typer.echo(
            "eval failed: corpus is empty \u2014 run dataset export/import-gyb first",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    for offender in store.verify_modes():
        typer.echo(f"WARNING: {offender} is group/world-readable", err=True)
    try:
        posture, filters, id_to_name = _load_attribution(
            store.root, assume_no_filters=assume_no_filters
        )
    except (ValueError, OSError, UnicodeDecodeError) as e:
        typer.echo(
            f"eval failed: snapshot unreadable ({type(e).__name__}: {e})",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if posture in ("missing", "bridge-incomplete"):
        # the build refuses (invariant #6) — name the REAL cause
        detail = (
            "no filters.json snapshot \u2014 run `dataset export` to "
            "capture it, or --assume-no-filters to attest a filterless "
            "mailbox"
            if posture == "missing"
            else "filters.json label ids are not all present in labels.json "
            "(missing or stale bridge) \u2014 re-run `dataset export` to "
            "refresh both"
        )
        typer.echo(f"eval failed: build refused (invariant #6): {detail}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the checkpoint's materialized manifest is the split's authority
    # (invariant #7): read + verify it BEFORE loading the model
    checkpoint_dir = cfg.model.checkpoint_dir
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        typer.echo(
            f"eval failed: checkpoint {checkpoint_dir} has no "
            f"{MANIFEST_FILENAME} \u2014 no verifiable split identity "
            "(run train first)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        manifest = SplitManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        typer.echo(
            f"eval failed: checkpoint manifest unreadable or tampered: {e}",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    expected = ExpectedIdentity(
        vocabulary=manifest.vocabulary,
        corpus_fingerprint=corpus_fingerprint(records, manifest.train_ids),
        split_manifest_hash=manifest.content_hash,
        thresholds_source=THRESHOLDS_SOURCE_NONE,
        encoder_id=cfg.model.encoder,
    )
    try:
        model = load_model(checkpoint_dir, expected, _eval_model_factory)
    except CheckpointRefusedError as e:
        # the refusal contract (AC#4): named fields, exit 5, no silent
        # fallback — a mismatched checkpoint is never scored
        typer.echo(
            f"eval failed: checkpoint refused ({', '.join(e.fields)}) \u2014 "
            "no silent fallback; rebuild the checkpoint",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except ImportError as e:
        # DECISION (both review families, train's boundary): the
        # optional `train` extra is missing — the recipe's lazy import
        # fails. Operation failure, exit 5 with an install hint, never
        # a raw ModuleNotFoundError traceback
        typer.echo(
            f"eval failed: the model recipe is unavailable ({e}) \u2014 install "
            "the optional extra: `uv sync --extra train` (if it is already "
            "installed, the recipe import itself is broken \u2014 report it)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except (ValueError, OSError, RuntimeError) as e:
        # recipe-level failures (bad local weights, a corrupt
        # checkpoint, a missing device) are operation failures, not
        # code bugs (train's documented taxonomy)
        typer.echo(f"eval failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if model.vocabulary != manifest.vocabulary:
        # a lying column order would silently misalign every score —
        # refuse before scoring (train's guard, the serving side)
        typer.echo(
            "eval failed: the model's vocabulary does not match the "
            "manifest's \u2014 refusing to score against a lying column "
            "order",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    now = datetime.now(UTC)  # the CLI may read the clock; the core may not
    try:
        result = build_dataset(
            records,
            filters=filters,
            id_to_name=id_to_name,
            now=now,
            split_date=manifest.split_date,
            manifest=manifest,
        )
    except BuildAbortedError as e:
        typer.echo(f"eval failed: build refused (invariant #6): {e.reason}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except ValueError as e:
        typer.echo(f"eval failed: build refused: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the TEST side only (invariant #7): membership comes from the
    # manifest — the split-boundary trust filter is a split-identity
    # decision owned by it, inherited, never re-applied here
    test_ids = set(manifest.test_ids)
    test_examples = [example for example in result.examples if example.id in test_ids]
    if not test_examples:
        typer.echo(
            "eval failed: no test-side examples \u2014 the manifest's test "
            "split is empty; retrain with an EARLIER --split-date (test "
            "is at/after it)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # preprocessing integrity (invariant #4): the SAME preprocess_text
    # train used — subject + body, trailing newline on a missing shard
    by_id = {record.id: record for record in records}
    missing_bodies = 0
    texts: list[str] = []  # aligned to test_examples (index i ↔ i)
    try:
        for example in test_examples:
            body = store.get_body(example.id)
            if body is None:
                missing_bodies += 1  # an orphaned shard (crash residue)
                body = ""
            texts.append(preprocess_text(by_id[example.id].subject, body))
    except (OSError, UnicodeDecodeError) as e:
        typer.echo(f"eval failed: a body shard is unreadable: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if missing_bodies:
        typer.echo(
            f"WARNING: {missing_bodies} test record(s) had no body shard "
            "\u2014 scored on subject + trailing newline (crash residue; "
            "re-export to repair)",
            err=True,
        )

    try:
        rows = model.predict(texts)
        if len(rows) != len(test_examples):
            # a named operator message, not raw zip internals: a wrong
            # row count is a checkpoint/recipe defect
            raise ValueError(
                f"the model returned {len(rows)} score rows for "
                f"{len(test_examples)} texts \u2014 checkpoint/recipe defect"
            )
        # alignment by EXAMPLE ID (not position): the Protocol's row i
        # ↔ texts[i] is re-keyed onto the example ids before the core
        # sees it. A misaligned/NaN row (the core's refusals) is the
        # same "no silent fallback" family as a refused checkpoint:
        # exit 5, never a traceback
        scores = {
            example.id: row for example, row in zip(test_examples, rows, strict=True)
        }
        table = eval_metrics(test_examples, scores, vocabulary=manifest.vocabulary)
    except (ValueError, OSError, RuntimeError) as e:
        typer.echo(f"eval failed: prediction failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    trust = (
        "server-authoritative-only" if manifest.server_authoritative_only else "none"
    )
    if as_json:
        payload = {
            "trust": trust,
            # split identity: two reports from different splits (or a
            # retrain) must be tellable apart by the consumer (#22)
            "split_date": manifest.split_date.isoformat(),
            "split_manifest_hash": manifest.content_hash,
            "test_examples": table.test_examples,
            "thresholds": list(table.thresholds),
            "rows": [
                {
                    "label": row.label,
                    "support": row.support,
                    "cells": [
                        {
                            "threshold": t,
                            "tp": cell.tp,
                            "fp": cell.fp,
                            "fn": cell.fn,
                            "precision": cell.precision,
                            "recall": cell.recall,
                        }
                        for t, cell in zip(table.thresholds, row.cells, strict=True)
                    ],
                }
                for row in table.rows
            ],
        }
        output = json.dumps(payload, sort_keys=True)
        typer.echo(output)
        file_text = output + "\n"
    else:
        output = (
            f"eval: trust filter: {trust} (inherited from the split "
            f"manifest)\n"
            + render_report(
                table,
                split_date=manifest.split_date,
                split_manifest_hash=manifest.content_hash,
            )
        )
        typer.echo(output, nl=False)
        file_text = output
    if out is not None:
        # the report carries the user's label taxonomy — treat it like
        # every other artifact this repo writes: 0600 from creation,
        # never world-readable (the corpus-sidecar discipline). Staging
        # + atomic replace (train's staging pattern): a failure leaves
        # any PREVIOUS report at the path intact, and no fd can leak.
        # Replacing an EXISTING file is announced on stderr — a silent
        # clobber of a typo'd path would be a lie (train refuses
        # overwrites; --force is overkill for a regenerable report)
        target = Path(out)
        if target.exists():
            typer.echo(f"eval: NOTICE: replacing existing {out}", err=True)
        staging = Path(str(out) + ".tmp")
        try:
            _write_secret_tight(staging, file_text)
            os.replace(staging, out)  # atomic on POSIX
        except OSError as e:
            with contextlib.suppress(OSError):
                staging.unlink()  # best-effort staging cleanup
            typer.echo(f"eval failed: cannot write --out: {e}", err=True)
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    # the deliberate calibration act (ticket #21, opt-in): the measured
    # table → per-label Threshold Bands → the versioned artifact bound
    # to the checkpoint + eval identity. Rendered AFTER the report (the
    # report is never lost — a calibration refusal, e.g. an override
    # for a label the vocabulary does not carry, is exit 5 with the
    # report already on stdout); the write is the ONLY artifact writer
    # in the package (recompute posture, AC4).
    if bands is not None:
        # serving-gate hardening: the artifact feeds the daemon's
        # shadow path, so it must never be minted from degraded data —
        # a corpus with missing body shards was scored on subject +
        # trailing newline only (the WARNING above), and those scores
        # are not the scores the bands would serve with. Refuse with
        # exit 5 — the operation-failure convention every other
        # post-report eval refusal uses (the calibration refusal below,
        # the write failure); the report is already on stdout and is
        # preserved, mirroring that posture.
        if missing_bodies:
            typer.echo(
                f"eval failed: --bands refused: {missing_bodies} test "
                "record(s) had no body shard \u2014 the scores are degraded "
                "(subject + trailing newline only) and the Bands Artifact "
                "feeds the daemon's serving gate; re-export to repair, "
                "then re-run",
                err=True,
            )
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        from .thresholds import (
            BandTargets,
            build_artifact,
            calibrate_bands,
        )

        # the effective-targets merge (config defaults + per-label
        # overrides) lives here for now — it moves into thresholds
        # when the daemon consumer lands (one site, not two)
        overrides = {
            label: BandTargets(
                min_high_precision=(
                    override.min_high_precision
                    if override.min_high_precision is not None
                    else cfg.bands.min_high_precision
                ),
                min_med_precision=(
                    override.min_med_precision
                    if override.min_med_precision is not None
                    else cfg.bands.min_med_precision
                ),
            )
            for label, override in cfg.bands.per_label.items()
        }
        try:
            calibrated = calibrate_bands(
                table,
                min_high_precision=cfg.bands.min_high_precision,
                min_med_precision=cfg.bands.min_med_precision,
                overrides=overrides or None,
            )
            # the eval identity (test_examples) and the default
            # targets are DERIVED from the calibration's provenance —
            # build_artifact no longer takes caller-told values
            artifact = build_artifact(
                calibrated,
                vocabulary=manifest.vocabulary,
                corpus_fingerprint=expected.corpus_fingerprint,
                split_manifest_hash=manifest.content_hash,
                encoder_id=cfg.model.encoder,
                split_date=manifest.split_date,
            )
        except (ValueError, OSError) as e:
            typer.echo(f"eval failed: band calibration refused: {e}", err=True)
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        try:
            _write_bands_artifact(Path(bands), artifact, artifact.to_json())
        except BandsTargetRefusedError as e:
            # the serving-gate hardening: the existing target is a
            # verifiably-different artifact — exit 5, before any
            # staging, the previous bytes intact
            typer.echo(f"eval failed: --bands refused: {e}", err=True)
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        except OSError as e:
            # the helper already cleaned up its (mkstemp-named) staging
            typer.echo(f"eval failed: cannot write --bands: {e}", err=True)
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        # the human-readable summary (the text-report mode only — the
        # --json payload is the machine-readable output, and the
        # artifact file is the machine-readable bands). On STDERR, so
        # the --out file stays "the same bytes as stdout" (the
        # documented --out contract — the summary is status, not
        # report content)
        if not as_json:
            typer.echo(
                "bands: "
                + "; ".join(
                    f"{lb.label} high={lb.high if lb.high is not None else 'none'} "
                    f"med={lb.med if lb.med is not None else 'none'}"
                    + (f" ({', '.join(lb.reasons)})" if lb.reasons else "")
                    for lb in calibrated.labels
                ),
                err=True,
            )


if __name__ == "__main__":
    app()
