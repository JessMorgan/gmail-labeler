"""gmail-labeler command-line interface.

Command surface is frozen by the Phase 0 design (AGENTS.md "Layout");
later phases add apply/daemon/status (absent until wired, which
surfaces as a usage error). Phase 0 subcommands, the Phase 1
``train``/``eval`` commands, and the Phase 1 exit ``shadow`` commands
are wired as they are implemented. Exit 3 is reserved for
wired-but-stubbed commands and currently UNREACHABLE — no command is
stubbed; scripts can branch on 0/2/4/5 today.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from .attribution import SIDECAR_FILTERS, SIDECAR_LABELS, parse_attribution
from .dataset_build import TrainingExample
from .gmail_import import DEFAULT_QUERY
from .model import Model

if TYPE_CHECKING:
    from .config import Config
    from .corpus_store import CorpusStore
    from .gmail_client import FilterInfo, GmailClient
    from .gmail_import import (  # the progress pair is annotations only
        GmailImportStats,
        ImportOutcome,
        ProgressSnapshot,
    )
    from .graduation import GraduationTable
    from .label_sync import FullResyncStats, SyncStats
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
_USAGE_ERROR_EXIT = 2  # typer's own usage-error code, reused for a
# contradictory flag combination refused before any config read or quota
# spend (ticket #75: --dry-run + --also-import-gmail)


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


def _write_out_report(out: str, file_text: str) -> None:
    """The eval ``--out`` write: 0600 from creation, EXCLUSIVE staging
    + atomic replace — a failure leaves any PREVIOUS report at the
    path intact, and no fd can leak. The staging file is
    ``tempfile.mkstemp``'d (a unique name per process, 0600 from
    creation, fsync'd before publish), so two concurrent ``eval`` runs
    on the same path can never interleave on a shared staging name and
    rename each other's partial bytes into place (the inherited
    #20-era fixed ``<out>.tmp`` name had exactly that hazard — the
    artifact writers got the mkstemp treatment in #21; this is the
    class fix for the report write). The final ``os.replace`` is
    atomic, so each run publishes its own complete bytes. Replacing an
    EXISTING file is announced on stderr — a silent clobber of a
    typo'd path would be a lie (train refuses overwrites; --force is
    overkill for a regenerable report). The report file lands BEFORE
    the serving-gate acts (bands/graduation) in TEXT mode, so a run
    that later refuses (a degraded corpus, a calibration refusal, a
    write failure) still REPLACES a previous report file — the report
    is the run's output, and the refusal is about the artifacts, not
    the report (in --json mode the file lands at the payload emit,
    which is after the acts — the bytes are the same). Raises OSError
    on failure (the caller maps it to exit 5)."""
    target = Path(out)
    if target.exists():
        typer.echo(f"eval: NOTICE: replacing existing {out}", err=True)
    fd, staging_name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".")
    staging = Path(staging_name)
    try:
        fh = os.fdopen(fd, "w", encoding="utf-8")  # owns fd from here
    except OSError:
        # fdopen failed — the fd is ours to close; both cleanups are
        # best-effort (a close/unlink failure must not replace the
        # primary error), and any stray-staging litter is silent BY
        # DESIGN (the primary error outranks cleanup visibility)
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            staging.unlink(missing_ok=True)
        raise
    try:
        with fh:
            fh.write(file_text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(staging, str(target))  # atomic on POSIX
    except OSError:
        # the cleanup is best-effort: an unlink failure must not
        # REPLACE the primary error (it would surface as a new
        # "cannot write" cause that masks the real one)
        with contextlib.suppress(OSError):
            staging.unlink(missing_ok=True)
        raise


class BandsTargetRefusedError(Exception):
    """The existing --bands target is a verifiably-different Bands
    Artifact: replacing it would destroy identity material (the
    serving-gate hardening — a clobber by a drifted run is a lie)."""


class ArtifactReadbackError(OSError):
    """The post-replace readback failed — the replace ALREADY happened
    (the on-disk bytes do not parse back, or parse to a different
    content hash). Distinct from a staging failure, where the previous
    file is intact: the CLI's message must not say "cannot write" when
    the target was in fact replaced (the previous file may be gone).
    Subclasses ``OSError`` so the existing write-failure mapping still
    catches it (exit 5)."""


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
    (``BandsTargetRefusedError`` — exit 5, BEFORE any staging); a
    target that parses as the SIBLING artifact (a Graduation Table) is
    a KNOWN artifact, not garbage — deleting it behind the NOTICE
    would destroy identity material, so it is refused naming the
    confusion; an unreadable/garbage target (parses as NOTHING) is NOT
    identity material — the replace NOTICE proceeds (the artifact is
    regenerable by design, the recompute posture). Replacing an
    existing artifact is announced on
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
            # a target that parses as the SIBLING artifact (a
            # Graduation Table) is a KNOWN artifact, not garbage —
            # deleting it behind the NOTICE would destroy identity
            # material; refuse, naming the confusion
            from .graduation import GraduationTable

            try:
                GraduationTable.from_json(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                # an unreadable/garbage target (parses as NOTHING) is
                # not identity material — the replace NOTICE proceeds
                # (regenerable by design)
                typer.echo(f"eval: NOTICE: replacing existing {path}", err=True)
            else:
                raise BandsTargetRefusedError(
                    f"existing file at {path} is a Graduation Table, not a "
                    "Bands Artifact \u2014 refusing to delete it (use a "
                    "different path for --bands)"
                )
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
        # fdopen failed — the fd is ours to close; both cleanups are
        # best-effort (a close/unlink failure must not replace the
        # primary error), and any stray-staging litter is silent BY
        # DESIGN (the primary error outranks cleanup visibility)
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
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
        # traceback. The OUTER `except OSError` (staging cleanup +
        # re-raise) is the I/O guard (concurrent writer, disk error)
        # — not the torn-write guard; the readback's own
        # `except OSError` (inside the try) is the post-replace
        # read-failure guard, wrapped into ArtifactReadbackError.
        try:
            readback = BandsArtifact.from_json(path.read_text(encoding="utf-8"))
        except ValueError as e:
            # a UnicodeDecodeError is a ValueError subclass — it lands
            # here (the file exists, it just is not a valid artifact)
            raise ArtifactReadbackError(
                f"artifact readback fault: {path} does not parse back "
                f"to a valid artifact ({type(e).__name__})"
            ) from e
        except OSError as e:
            # the replace ALREADY happened — a readback read failure
            # is a post-replace failure, not a write failure ("cannot
            # write" would be a lie; the outer except OSError would
            # print exactly that)
            raise ArtifactReadbackError(
                f"artifact readback fault: {path} failed to read back "
                f"({type(e).__name__})"
            ) from e
        if readback.content_hash != artifact.content_hash:
            # it DID parse — a valid artifact whose content hash
            # differs: a concurrent writer published a different
            # artifact (the fault branch above is the torn-write one)
            raise ArtifactReadbackError(
                f"artifact readback mismatch: {path} is a valid "
                "artifact whose content hash differs from the minted "
                "one (a concurrent writer published a different "
                "artifact)"
            )
    except OSError:
        # the cleanup is best-effort: an unlink failure must not
        # REPLACE the primary error (it would surface as a new
        # "cannot write" cause that masks the real one)
        with contextlib.suppress(OSError):
            staging.unlink(missing_ok=True)
        raise


class GraduationTargetRefusedError(Exception):
    """The existing --graduation target is a verifiably-different
    Graduation Table: replacing it would destroy identity material (the
    serving-gate hardening — a clobber by a drifted run is a lie)."""


def _write_graduation_table(path: Path, table: GraduationTable, text: str) -> None:
    """Write the Graduation Table (ticket #22): 0600 from creation,
    EXCLUSIVE staging + atomic replace — a failure leaves any PREVIOUS
    table intact, and no fd can leak. The staging file is
    ``tempfile.mkstemp``'d (a unique name per process, 0600 from
    creation, fsync'd before publish), so two concurrent ``eval
    --graduation`` runs on the same path can never interleave on a
    shared staging name and rename each other's partial bytes into
    place; the final ``os.replace`` is atomic, so each run publishes its
    own complete bytes.

    Before replacing an EXISTING table, its identity is checked against
    the current run (the serving-gate hardening): a parsed table whose
    identity fields disagree is refused
    (``GraduationTargetRefusedError`` — exit 5, BEFORE any staging); a
    target that parses as the SIBLING artifact (a Bands Artifact) is a
    KNOWN artifact, not garbage — deleting it behind the NOTICE would
    destroy identity material, so it is refused naming the confusion;
    an unreadable/garbage target (parses as NOTHING) is NOT identity
    material — the replace NOTICE proceeds (the table is regenerable by
    design, the recompute posture). Replacing an existing table is
    announced on stderr (the
    #20 ``--out`` NOTICE — a silent clobber of a typo'd path would be a
    lie). After the replace, the on-disk bytes are read back through
    ``GraduationTable.from_json`` (train's sidecar readback symmetry)
    and compared by CONTENT HASH, not dataclass equality — the
    canonical payload serializes reasons sorted, so equality must be
    order-insensitive too (a reasons-order difference is the same
    meaning, not a torn write). A corrupt/torn file is refused by
    ``from_json`` ITSELF (ValueError — mapped to the same named
    operation failure, never a traceback); the hash-mismatch branch is
    the concurrent-writer signature (a valid table whose content hash
    differs — it DID parse); the OSError branch guards I/O failures
    around the publish (a concurrent writer, a disk error) — it is NOT
    the torn-write guard. The ONLY writer of the table in the package
    (recompute posture, AC4 — pinned by
    ``test_only_eval_confirm_writes_the_graduation_table``)."""
    from .graduation import GraduationTable, identity_disagreement

    if path.is_file():
        try:
            existing = GraduationTable.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            # a target that parses as the SIBLING artifact (a Bands
            # Artifact) is a KNOWN artifact, not garbage — deleting it
            # behind the NOTICE would destroy identity material;
            # refuse, naming the confusion
            from .thresholds import BandsArtifact

            try:
                BandsArtifact.from_json(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                # an unreadable/garbage target (parses as NOTHING) is
                # not identity material — the replace NOTICE proceeds
                # (regenerable by design)
                typer.echo(f"eval: NOTICE: replacing existing {path}", err=True)
            else:
                raise GraduationTargetRefusedError(
                    f"existing file at {path} is a Bands Artifact, not a "
                    "Graduation Table \u2014 refusing to delete it (use a "
                    "different path for --graduation)"
                )
        else:
            disagreeing = identity_disagreement(existing, table)
            if disagreeing:
                raise GraduationTargetRefusedError(
                    f"existing table at {path} is bound to "
                    f"{', '.join(disagreeing)} — not this checkpoint/eval"
                )
            typer.echo(f"eval: NOTICE: replacing existing {path}", err=True)
    fd, staging_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    staging = Path(staging_name)
    try:
        fh = os.fdopen(fd, "w", encoding="utf-8")  # owns fd from here
    except OSError:
        # fdopen failed — the fd is ours to close; both cleanups are
        # best-effort (a close/unlink failure must not replace the
        # primary error), and any stray-staging litter is silent BY
        # DESIGN (the primary error outranks cleanup visibility)
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            staging.unlink(missing_ok=True)
        raise
    try:
        with fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(staging, str(path))  # atomic on POSIX
        # readback symmetry (train's sidecar pattern): the on-disk
        # bytes must parse back to the minted table, compared by
        # CONTENT HASH (order-insensitive — see the docstring).
        # from_json refuses a corrupt/torn file FIRST (ValueError —
        # self-inconsistent bytes never reach the hash comparison);
        # that refusal is the same named operation failure, not a
        # traceback. The OUTER `except OSError` (staging cleanup +
        # re-raise) is the I/O guard (concurrent writer, disk error)
        # — not the torn-write guard; the readback's own
        # `except OSError` (inside the try) is the post-replace
        # read-failure guard, wrapped into ArtifactReadbackError.
        try:
            readback = GraduationTable.from_json(path.read_text(encoding="utf-8"))
        except ValueError as e:
            # a UnicodeDecodeError is a ValueError subclass — it lands
            # here (the file exists, it just is not a valid table)
            raise ArtifactReadbackError(
                f"table readback fault: {path} does not parse back "
                f"to a valid table ({type(e).__name__})"
            ) from e
        except OSError as e:
            # the replace ALREADY happened — a readback read failure
            # is a post-replace failure, not a write failure ("cannot
            # write" would be a lie; the outer except OSError would
            # print exactly that)
            raise ArtifactReadbackError(
                f"table readback fault: {path} failed to read back ({type(e).__name__})"
            ) from e
        if readback.content_hash != table.content_hash:
            # it DID parse — a valid table whose content hash
            # differs: a concurrent writer published a different
            # table (the fault branch above is the torn-write one)
            raise ArtifactReadbackError(
                f"table readback mismatch: {path} is a valid "
                "table whose content hash differs from the minted "
                "one (a concurrent writer published a different "
                "table)"
            )
    except OSError:
        # the cleanup is best-effort: an unlink failure must not
        # REPLACE the primary error (it would surface as a new
        # "cannot write" cause that masks the real one)
        with contextlib.suppress(OSError):
            staging.unlink(missing_ok=True)
        raise


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

# The closing pointer and the unseen NOTE are shared between the
# standalone sync report and the composite second pass (ticket #75): one
# tail, pinned once. Ticket #77 (fix-round deferral): the pointer
# claimed "the updated candidate gates" after a --dry-run plan-only run —
# the same honesty class as the WOULD-birth split (9a6da1c), mirrored the
# same way: a dry-run twin sharing one tail constant with the real pointer
# so the modes cannot drift, selected at the command's own echo site
# (--dry-run refuses composition, so the composite pointer at the
# import-gmail pass is always a real-behavior close).
_ANALYZE_POINTER_TAIL = (
    " for the updated candidate gates (ADR-0005 keeps retraining and "
    "applying human-gated)"
)
_ANALYZE_POINTER = "\nNext: run `dataset analyze`" + _ANALYZE_POINTER_TAIL
_ANALYZE_POINTER_DRY = (
    "\nNext: this dry run wrote nothing \u2014 re-run the same command "
    "without --dry-run to apply the plan, then `dataset analyze`"
    + _ANALYZE_POINTER_TAIL
)
_UNSEEN_NOTE = (
    "NOTE: {n} mailbox id(s) are not in the corpus \u2014 "
    "history events and label inventories cannot establish that "
    "these are new mail or that they match the import query; ingest "
    "stays the acquisition paths' alone (dataset "
    "import-gmail/import-gyb), never sync's (spec #64)"
)
# ADR-0008 consequence 1 provenance disclosure, shared by the standalone
# event walk and the composite sync pass (ticket #75): one rendering.
# The dry-run twin mirrors the --full-resync branch's WOULD-birth split
# (fix-round review): under --dry-run nothing is written, so the NOTE must
# say WOULD birth — stats.birthed keeps its core semantics (set from the
# ledger READ, not the write) and only the rendering branches per mode.
# One shared tail (deepseek review NIT): the two modes differ in the verb
# clause only, so the wording cannot drift apart.
_BIRTH_NOTE_TAIL = (
    " (no consumed baseline to diff against yet; the corpus's label state "
    "may predate the cursor). " + _RESYNC_GUIDANCE_NOTE
)
_BIRTH_NOTE = (
    "NOTE: this corpus had no Sync Cursor \u2014 it was birthed "
    "from the mailbox profile and applies NOTHING" + _BIRTH_NOTE_TAIL
)
_BIRTH_NOTE_DRY = (
    "NOTE: this corpus had no Sync Cursor \u2014 sync WOULD birth "
    "it from the mailbox profile and apply NOTHING" + _BIRTH_NOTE_TAIL
)


def _render_failures(failed: list[tuple[str, str]]) -> None:
    """Failure detail shared by every dataset report (20-line cap, stderr)."""
    for message_id, reason in failed[:20]:
        typer.echo(f"  failed {message_id}: {reason}", err=True)
    if len(failed) > 20:
        typer.echo(f"  ... and {len(failed) - 20} more", err=True)


def _import_gmail_client(config: Config) -> GmailClient:
    """Client factory seam: tests inject FakeGmailClient here."""
    from .auth import credentials as build_credentials
    from .gmail_client import GmailRestClient

    return GmailRestClient(build_credentials(config))


def _dataset_auth_bootstrap(
    cfg: Config, *, failure_prefix: str
) -> tuple[GmailClient, dict[str, str], list[FilterInfo], GmailImportStats]:
    """Shared dataset auth gate: build the client and run the settings
    bootstrap behind the pinned exit-code contract — auth failures print
    ``auth: …`` and exit 4; a transport failure prints
    ``{failure_prefix}: …`` and exits 5. The gate runs FIRST: a
    not-signed-in run must not touch the corpus tree (the dataset commands
    must not create or re-mode it — a typo'd corpus.dir especially).
    Raises typer.Exit; on success returns
    (client, id_to_name, filters, bootstrap-stats)."""
    from .auth import AuthError
    from .gmail_errors import GmailAuthError, GmailError
    from .gmail_import import fetch_settings
    from .tokens import TokenStoreError

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
        typer.echo(f"{failure_prefix}: {e}", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    return client, id_to_name, filters, bootstrap


def _dataset_store_modes(store: CorpusStore, *, enforce: bool = True) -> None:
    """Shared store mode handling for the dataset import and sync
    commands: tighten the tree, then WARNING-loop over anything still
    group/world-readable.
    Pass ``enforce=False`` for read-only runs (``--dry-run``): per
    corpus_store.enforce_modes' own contract ("Call from WRITE-side
    commands… not from read-only tools", corpus_store.py ~431), a run
    that writes nothing must not re-mode anything either. The
    non-destructive verify_modes WARNING runs either way — it names the
    structural paths the later write run would tighten (verify_modes is
    a structural scan; the full-tree tightening is enforce_modes' job)."""
    if enforce:
        store.enforce_modes()
    for offender in store.verify_modes():
        typer.echo(f"WARNING: {offender} is group/world-readable", err=True)


@dataset_app.command("import-gmail")
def dataset_import_gmail(
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
    also_sync_labels: bool = typer.Option(
        False,
        "--also-sync-labels",
        help=(
            "After the import completes durably, run the Label Sync "
            "delta from the Sync Cursor — one report carries both "
            "passes' stats; a sync-pass failure exits non-zero but "
            "never invalidates the import (spec #64). Opt-in by "
            "design: without it import-gmail keeps its crisp "
            "append-new-messages contract"
        ),
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        metavar="",  # count option: the inferred <int> metavar would
        # advertise a value the flag never takes (fix round NIT)
        help=(
            "Repeat for more progress on STDERR (no value): -v one page "
            "summary per completed page, plus one final line when the loop "
            "returns early (max-messages bound or page-fetch failure); -vv "
            "adds per-message outcome lines; stdout and exit codes are "
            "unchanged at every level (ticket #61)"
        ),
    ),
) -> None:
    """Import labeled mail from the Gmail API into the corpus.

    --also-sync-labels (ticket #75, opt-in by design): once the import has
    completed durably, run the Label Sync delta too — one report, both
    passes. A --max-messages-bounded run completes durably too, so the
    sync pass still walks from the ledger's seed (birthed or pre-existing).
    Without the flag the command keeps its crisp "append new messages,
    never touch existing records" meaning (spec #64).
    """
    from .corpus_store import CorpusError, CorpusStore
    from .gmail_errors import GmailAuthError, GmailError
    from .gmail_import import (
        import_gmail,
        write_filters_snapshot,
        write_labels_snapshot,
    )

    cfg = _load_config(config)
    # Auth gate FIRST via the shared bootstrap (_dataset_auth_bootstrap)
    client, id_to_name, filters, bootstrap = _dataset_auth_bootstrap(
        cfg, failure_prefix="import failed"
    )
    store = CorpusStore(cfg.corpus.dir)
    _dataset_store_modes(store)
    # Opt-in progress (#61): the engine stays I/O-free and hands frozen
    # ProgressSnapshots here; ALL printing lives in this adapter. STDERR on
    # purpose — the stdout summary is a machine-parseable contract at every
    # level. Level 0 wires nothing (byte-identical default run, pinned).
    # That byte-identity is exact for a HEALTHY stderr: the deliberate
    # dead-stderr divergence (typed exit 5 where the base run died at 1)
    # lives in this channel by choice (#61 fix round 2), not pin scope.
    #
    # Callback safety is the ADAPTER's duty (engine contract: callbacks must
    # not raise): ONE BrokenPipeError — e.g. `dataset import-gmail -v 2>&1 | head`
    # — permanently closes the progress channel (#52's guard-precedence
    # style: the first failure decides; no repeated attempts against a dead
    # pipe, no spam, no traceback), the run continues to its typed outcome,
    # and resume stays structural. Any OTHER callback error stays fail-loud.
    # The post-run failure LIST shares this channel (fix round 2), so a
    # dead pipe costs the listing, never the typed exit code.
    progress_closed = False

    def _emit(line: str) -> None:
        nonlocal progress_closed
        if progress_closed:
            return
        try:
            typer.echo(line, err=True)
        except BrokenPipeError:
            progress_closed = True

    def _page_line(snap: ProgressSnapshot) -> None:
        _emit(
            f"import-gmail: page scanned={snap.scanned} written={snap.written} "
            f"skipped_present={snap.skipped_present} "
            f"skipped_draft_chat={snap.skipped_draft_chat} "
            # bootstrap blips happened BEFORE the loop, so the engine's
            # retried= excludes them while the stdout summary merges them —
            # offset here so every -v line and the summary share one scale
            # (deepseek r1 MINOR; both sources pinned in test_cli)
            f"retried={snap.retried + bootstrap.retried} "
            f"failed={snap.failed_count}"
        )

    def _message_line(message_id: str, outcome: ImportOutcome) -> None:
        _emit(f"import-gmail: {message_id} {outcome}")

    on_page = _page_line if verbose >= 1 else None
    on_message = _message_line if verbose >= 2 else None
    try:
        write_filters_snapshot(store, filters)
        write_labels_snapshot(store, id_to_name)
        stats = import_gmail(
            client,
            store,
            id_to_name=id_to_name,
            query=query,
            max_messages=max_messages,
            on_page=on_page,
            on_message=on_message,
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
        f"skipped_present={stats.skipped_present} "
        f"skipped_draft_chat={stats.skipped_draft_chat} "
        f"retried={stats.retried} failed={len(stats.failed)}"
    )
    if stats.failed:
        # the failure list shares the guarded stderr channel (#61 fix
        # round 2): a dead pipe stops the LIST too, never the typed exit;
        # ids normalized like the store key and the -vv events so all
        # three surfaces agree (one .lower() at this site). Same 20-line
        # cap and stderr channel as _render_failures; this site adds the
        # dead-pipe guard and the normalization the progress surfaces need.
        for message_id, reason in stats.failed[:20]:
            _emit(f"  failed {message_id.lower()}: {reason}")
        if len(stats.failed) > 20:
            _emit(f"  ... and {len(stats.failed) - 20} more")
        raise typer.Exit(code=_OPERATION_ERROR_EXIT)  # never silently partial
    if also_sync_labels:
        _sync_labels_pass(client, store, id_to_name=id_to_name)
        # the pointer closes the WHOLE report and has ONE owner — the
        # command, after its pass chain — in both composite directions
        # (review round 1: pointer-inside-the-pass was a double-print trap)
        typer.echo(_ANALYZE_POINTER)


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
            "containing commas cannot be expressed via this flag. Extra "
            "tokens are stripped from label sets only — they do NOT "
            "trigger DRAFT/CHAT row rejection, which matches canonical "
            "tokens alone"
        ),
    ),
) -> None:
    """Import a Got Your Back backup directory into the corpus.

    Holds a shared read lock on msg-db.sqlite for the whole run — finish
    an in-flight GYB backup first, or its writes will block.

    Fully offline — msg-db.sqlite is opened read-only (your backup is
    never mutated) and bodies go through the same MIME parser as import-gmail.
    DRAFT/CHAT rows are rejected at ingest exactly like the API import's
    policy layer (ticket #32's completion of #17: one corpus-wide policy,
    two ingest layers).
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
    _dataset_store_modes(store)
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
            # the rejection count rides the aborted-run echo too (ticket
            # #32): rows rejected BEFORE the stop must not vanish from the
            # operator's accounting
            f" — {partial.written} record(s) durably imported so far are "
            f"kept (skipped_draft_chat={partial.skipped_draft_chat}); "
            "re-running resumes"
            if partial is not None
            else " — records durably imported so far are kept; re-running resumes"
        )
        typer.echo(f"import failed: {e}{state}", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    typer.echo(
        f"import-gyb: scanned={stats.scanned} written={stats.written} "
        f"skipped_present={stats.skipped_present} "
        f"skipped_draft_chat={stats.skipped_draft_chat} "
        f"failed={len(stats.failed)}"
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
        _render_failures(stats.failed)
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
    also_import_gmail: bool = typer.Option(
        False,
        "--also-import-gmail",
        help=(
            "After the sync delta is durable (Sync Cursor advanced), run "
            "the full Gmail Import pass (default query) — one report "
            "carries both passes' stats; an import-pass failure exits "
            "non-zero but never invalidates the sync (spec #64)"
        ),
    ),
) -> None:
    """Refresh the corpus's label state from mailbox changes (Label Sync).

    Default path: walks the Gmail history from the corpus's Sync Cursor and
    appends a label Revision for every touched message — metadata only,
    bodies never (invariant #5). Event ids the corpus has never held are
    counted as Unseen Ids and reported, never ingested here; deleted mail
    keeps its record.

    --full-resync (recovery path, ticket #74): reconciles the corpus's
    label state against the mailbox directly when the history window is
    gone or was never consumed — cost-estimated up front from one
    labels.list plus local resolved-view counts, cheaper of the
    label-inventory scan and the corpus-metadata walk picked with a printed
    rationale, the cursor reseeded to the seed captured BEFORE the scan
    (step-down-permitted: a rolled-back mailbox's higher stored cursor
    yields to the seed — replay from the lower seed is harmless under
    ADR-0007) only after the reconciliation's revisions are durable
    (ADR-0008).

    --also-import-gmail (ticket #75): once this command's own delta is
    durable, run the full Gmail Import pass too — content and label
    freshness refreshed in one sitting, each pass's durability its own.
    """
    if dry_run and also_import_gmail:
        # --dry-run promises ZERO writes; the import pass writes records —
        # the combination is contradictory, not silently degraded (exit 2
        # before any config read or quota spend)
        typer.echo(
            "sync-labels: --dry-run writes nothing, so the "
            "--also-import-gmail pass has no durability point to follow "
            "— refusing the combination",
            err=True,
        )
        raise typer.Exit(code=_USAGE_ERROR_EXIT)
    from .corpus_store import CorpusError, CorpusStore
    from .gmail_errors import GmailAuthError, GmailError, GmailResyncError
    from .label_sync import FullResyncStats, sync_labels
    from .label_sync import full_resync as run_full_resync

    cfg = _load_config(config)
    # Auth gate FIRST via the shared bootstrap (import-gmail on the same
    # spine). Deliberate reuse of the landed bootstrap (spec #64 seam 3):
    # the filters fetch is unused on the standalone path — a filters-only
    # outage therefore blocks sync too (loudly, exit 5). Accepted: one
    # cheap shared call keeps every dataset command on the SAME
    # auth/retry spine; a labels-only bootstrap is a refactor for the
    # daemon phase. --also-import-gmail puts the filters to their real
    # use: the import pass's attribution snapshot (ticket #75).
    client, id_to_name, filters, bootstrap = _dataset_auth_bootstrap(
        cfg, failure_prefix="sync-labels failed"
    )
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
    _dataset_store_modes(store, enforce=not dry_run)
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
    if isinstance(stats, FullResyncStats):
        # structural narrowing (PR #86 review): the concrete type IS the
        # discriminator — only the --full-resync call above can have
        # produced a FullResyncStats. Structural honesty, not a bug fix:
        # the replaced asserts never fired (output was already
        # byte-identical under -O), but a stripped assert is a narrowing
        # that silently disappears, while this type test stays visible
        # to every reader and to pyright
        _report_full_resync(stats, dry_run=dry_run)
        _render_gains_losses(stats.gains, stats.losses)
        if stats.unseen:
            typer.echo(_UNSEEN_NOTE.format(n=stats.unseen), err=True)
    else:
        _report_sync(stats, dry_run=dry_run)
    if stats.failed:
        # an incomplete run gets NO second pass: the composite's precondition
        # is the first pass's final durability point (cursor advanced), which
        # a failed sync never reached (ticket #75)
        _render_failures(stats.failed)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT)  # never silently partial
    if also_import_gmail:
        _import_gmail_pass(client, store, id_to_name=id_to_name, filters=filters)
    # reaching this line means every pass that ran was complete — an
    # incomplete run does not recommend analyzing a half-synced corpus.
    # Under --dry-run the gates did NOT move (ticket #77): the plan-only
    # twin closes instead; composition is refused under the flag, so no
    # real second pass can hide behind the dry wording here.
    typer.echo(_ANALYZE_POINTER_DRY if dry_run else _ANALYZE_POINTER)


def _import_gmail_pass(
    client: GmailClient,
    store: CorpusStore,
    *,
    id_to_name: dict[str, str],
    filters: list[FilterInfo],
) -> None:
    """The second pass of ``dataset sync-labels --also-import-gmail``
    (ticket #75): the FULL Gmail Import — the same snapshot writes +
    ``import_gmail`` call the standalone command makes, delegated, never
    re-implemented (spec #64: ingest stays single-sourced in the import
    path). Reached only after the sync pass cleared its own durability gate,
    so the Sync Cursor is already advanced and import's birth-when-absent
    leaves it byte-identical (ADR-0008). ``bootstrap.retried`` is NOT added
    again here — the sync report already carries it, and one composite
    report must not double-count the shared settings blips. One bootstrap
    serves both passes: the snapshots this pass writes reflect the settings
    read BEFORE the sync walk, not a refetch at import start — label or
    filter churn during the sync window surfaces in the next standalone
    import or sync, a deliberate one-call quota trade (spec #64 keeps the
    dataset commands on one shared bootstrap spine)."""
    from .corpus_store import CorpusError
    from .gmail_errors import GmailAuthError, GmailError
    from .gmail_import import (
        import_gmail,
        write_filters_snapshot,
        write_labels_snapshot,
    )

    try:
        write_filters_snapshot(store, filters)
        write_labels_snapshot(store, id_to_name)
        stats = import_gmail(client, store, id_to_name=id_to_name, query=DEFAULT_QUERY)
    except GmailAuthError as e:  # revoked token mid-pass: the fix is human
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except (GmailError, CorpusError) as e:
        # the standalone import's mapping, unchanged — snapshot writes
        # convert their OSError to CorpusError; the record-append block's
        # write/fsync is the residual pre-existing gap, identical here and
        # standalone. The sync pass's durable work is never rolled back by
        # a second-pass fault
        typer.echo(f"import failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    typer.echo(
        f"import-gmail (also): scanned={stats.scanned} written={stats.written} "
        f"skipped_present={stats.skipped_present} retried={stats.retried} "
        f"failed={len(stats.failed)}"
    )
    _render_failures(stats.failed)
    if stats.failed:
        raise typer.Exit(code=_OPERATION_ERROR_EXIT)  # never silently partial


def _sync_labels_pass(
    client: GmailClient,
    store: CorpusStore,
    *,
    id_to_name: dict[str, str],
) -> None:
    """The second pass of ``dataset import-gmail --also-sync-labels``
    (ticket #75): the Label Sync event walk — the same ``sync_labels`` call
    the standalone command makes, delegated, never re-implemented. Reached
    only after the import completed durably, so the walk starts from the
    ledger the import birthed (or found) — seeded BEFORE its paging
    (ADR-0008): relabels made during the import land in this window, and
    freshly imported ids touched by events are re-verified (self-healing
    appends the fold absorbs). ``bootstrap.retried`` already rode the
    import report — not added again here. The event WALK only:
    ``--full-resync`` is an explicit recovery decision, never an implicit
    second pass."""
    from .corpus_store import CorpusError
    from .gmail_errors import GmailAuthError, GmailError, GmailResyncError
    from .label_sync import sync_labels

    try:
        stats = sync_labels(client, store, id_to_name=id_to_name)
    except GmailResyncError as e:
        # the standalone sync's 404 contract, kept whole under composition:
        # exact recovery command printed, ledger untouched, never automatic
        typer.echo(f"sync-labels: {e}", err=True)
        typer.echo(_RESYNC_GUIDANCE, err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    except GmailAuthError as e:  # revoked token mid-pass: the fix is human
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except (GmailError, CorpusError, OSError, UnicodeDecodeError) as e:
        # the import pass's durable work is never rolled back by a
        # second-pass fault (spec #64)
        typer.echo(f"sync-labels failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_ERROR_EXIT) from None
    _render_sync_report(stats, "sync-labels (also):", dry_run=False)
    _render_failures(stats.failed)
    if stats.failed:
        raise typer.Exit(code=_OPERATION_ERROR_EXIT)  # never silently partial


def _report_full_resync(stats: FullResyncStats, *, dry_run: bool) -> None:
    """The --full-resync report: mode-named summary line + the birth NOTE
    + the step-down NOTE (#95).
    Takes the CONCRETE stats type (PR #86 review) — the caller dispatches
    on isinstance, so no runtime assert narrows here: the replaced assert
    never fired (it is structural honesty, not a bug fix), but a stripped
    assert disappears under ``python -O`` while this signature stays
    visible to every reader and to pyright."""
    prefix = (
        "sync-labels --full-resync (dry-run):"
        if dry_run
        else "sync-labels --full-resync:"
    )
    # the flag-rate is the scan's drift signal; on the walk every id
    # is fetched by construction, so quoting a 100% "flag-rate" there
    # would read as drift that never happened (deepseek review). Runs
    # that never finished ENUMERATING (sentinel failures: "<cursor>",
    # "<labels>", "<label:ID>") stay silent too — a 0% rate from a scan
    # that never ran reads as measured absence of drift (PR #88
    # review). A completed scan with per-message failures KEEPS its
    # rate: it was really measured, and it is the diagnostic a human
    # reads after a failed run.
    rate = (
        f" (flag-rate {stats.flag_rate:.0%})"
        if stats.strategy == "scan"
        and not any(key.startswith("<") for key, _reason in stats.failed)
        else ""
    )
    typer.echo(
        f"{prefix} strategy={stats.strategy} "
        f"corpus={stats.corpus_messages} "
        f"flagged={stats.flagged}{rate} "
        f"revised={stats.revised} unchanged={stats.unchanged} "
        f"deleted_kept={stats.deleted_kept} unseen={stats.unseen} "
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
    if stats.stepped_down_from is not None:
        # the step-down disclosure (#95): data rides the core's stats, the
        # NOTE renders here (the birth-provenance precedent) — one line,
        # honest about plan-vs-completed under --dry-run
        if dry_run:
            typer.echo(
                f"NOTE: plan includes stepping the Sync Cursor down from "
                f"{stats.stepped_down_from} to {stats.seed_history_id} "
                f"(ADR-0008 \u2014 the kept-high cursor would be "
                f"unreachable; ground truth written on completion)."
            )
        else:
            typer.echo(
                f"NOTE: Sync Cursor stepped down from "
                f"{stats.stepped_down_from} to {stats.seed_history_id} \u2014 "
                f"the mailbox reported below the stored ledger; the kept-high "
                f"Sync Cursor was unreachable and the reconciliation just "
                f"wrote ground truth (ADR-0008)."
            )


def _report_sync(stats: SyncStats, *, dry_run: bool) -> None:
    """The event-walk report: summary line + the birth NOTE (ADR-0008
    consequence 1's provenance disclosure) + the per-label counts and the
    unseen NOTE. The concrete-stats twin of ``_report_full_resync`` — same
    isinstance-dispatch rationale; the body itself renders through the
    one shared source the composite pass also uses (ticket #75)."""
    _render_sync_report(
        stats,
        "sync-labels (dry-run):" if dry_run else "sync-labels:",
        dry_run=dry_run,
    )


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


def _render_sync_report(stats: SyncStats, prefix: str, *, dry_run: bool) -> None:
    """The event-walk sync report body (summary, birth provenance NOTE,
    per-label counts, unseen NOTE) — shared by the standalone
    ``sync-labels`` command and the ``import-gmail --also-sync-labels``
    composite pass (ticket #75): one rendering, pinned once. ``dry_run`` is
    a required keyword (deepseek review): a future dry-run-capable call site
    cannot silently render the actual-birth NOTE by omission — the composite
    pass states ``dry_run=False`` explicitly (``import-gmail`` offers no
    ``--dry-run`` flag today, so the composite always writes; were one ever
    added, the mode must be threaded through rather than defaulting to an
    actual-birth render). The analyze pointer
    is the COMMAND's — it closes the whole report after any second pass; the
    failure detail is printed by each command's own gate."""
    typer.echo(
        f"{prefix} events={stats.events} stale={stats.stale} "
        f"revised={stats.revised} deleted_kept={stats.deleted_kept} "
        f"unseen={stats.unseen} retried={stats.retried} "
        f"failed={len(stats.failed)}"
    )
    if stats.birthed:
        typer.echo(_BIRTH_NOTE_DRY if dry_run else _BIRTH_NOTE)
    _render_gains_losses(stats.gains, stats.losses)
    if stats.unseen:
        typer.echo(_UNSEEN_NOTE.format(n=stats.unseen), err=True)


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
            "train failed: corpus is empty \u2014 run dataset import-gmail/import-gyb "
            "first",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    for offender in store.verify_modes():
        typer.echo(f"WARNING: {offender} is group/world-readable", err=True)
    try:
        filters_text = store.read_snapshot(SIDECAR_FILTERS)
        # conditional bridge read (analyze idiom): a corrupt labels.json
        # beside an absent snapshot cannot fail the run
        labels_text = (
            store.read_snapshot(SIDECAR_LABELS) if filters_text is not None else None
        )
        posture, filters, id_to_name = parse_attribution(
            filters_text, labels_text, assume_no_filters=assume_no_filters
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
    graduation: str | None = typer.Option(
        None,
        "--graduation",
        help="Recompute the Graduation Table to PATH (opt-in; the "
        "human-confirm diff old→new goes to stderr; nothing is written "
        "without --confirm-graduation)",
    ),
    confirm_graduation: bool = typer.Option(
        False,
        "--confirm-graduation",
        help="Confirm the recomputed Graduation Table and write it "
        "(requires --graduation)",
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

    ``--graduation PATH`` recomputes the Graduation Table (ticket
    #22) — the per-label gates (min examples + min precision at the
    calibrated high band) applied to the Managed Labels — and prints
    the human-confirm diff (old → new per label) to stderr; it writes
    NOTHING without ``--confirm-graduation`` (ADR-0005: no pipeline
    run ever mutates what is live by itself). The table is bound to
    the checkpoint + eval identity + the Bands Artifact's content
    hash; an existing table bound to a different identity refuses the
    confirm (exit 5, the previous bytes intact).
    """
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
    from .config import find_config

    # usage refusals up front (exit 2, before any corpus-scale work):
    # --confirm-graduation without --graduation has nothing to confirm;
    # an empty --bands/--graduation would degenerate to Path("") and
    # fail awkwardly mid-run; --bands/--graduation and --out naming the
    # same file would let the artifact write silently destroy the
    # report; --bands and --graduation naming the same file would let
    # the second write silently destroy the first; an artifact naming
    # the config file would destroy identity material this very run
    # just read (an artifact OR --out naming the config file); an
    # artifact INSIDE the checkpoint dir (manifest, sidecar, weights)
    # or the corpus store (records index, labels/filters sidecars,
    # bodies) would clobber identity files the garbage-tolerance NOTICE
    # branch cannot see — a destructive collision, not a
    # regenerable-report NOTICE
    if confirm_graduation and graduation is None:
        typer.echo(
            "eval failed: --confirm-graduation requires --graduation",
            err=True,
        )
        raise typer.Exit(code=2) from None
    # the empty checks run BEFORE the equal-path checks: an empty
    # target would resolve to the cwd (and --out would write a stray
    # staging file there) and report a spurious "different files"
    # collision instead of "non-empty"
    for flag, target in (
        ("--out", out),
        ("--bands", bands),
        ("--graduation", graduation),
    ):
        if target is not None and not target.strip():
            typer.echo(f"eval failed: {flag} PATH must be non-empty", err=True)
            raise typer.Exit(code=2) from None
    if (
        bands is not None
        and graduation is not None
        and Path(bands).resolve() == Path(graduation).resolve()
    ):
        typer.echo(
            "eval failed: --bands and --graduation must name different files",
            err=True,
        )
        raise typer.Exit(code=2) from None
    # the config path is bound ONCE for the refusal checks below
    # (find_config is called up to 4x per run otherwise)
    config_path = find_config(config).resolve()
    for flag, target in (
        ("--out", out),
        ("--bands", bands),
        ("--graduation", graduation),
    ):
        if target is None:
            continue
        if (
            flag != "--out"
            and out is not None
            and (Path(target).resolve() == Path(out).resolve())
        ):
            typer.echo(
                f"eval failed: {flag} and --out must name different files",
                err=True,
            )
            raise typer.Exit(code=2) from None
        target_path = Path(target).resolve()
        if target_path.is_dir():
            # an existing directory at the target would fail at
            # os.replace (EISDIR) AFTER the report — a usage error up
            # front, like the other usage refusals
            typer.echo(
                f"eval failed: {flag} {target!r} names an existing "
                f"directory ({target_path}) \u2014 the target is a file",
                err=True,
            )
            raise typer.Exit(code=2) from None
        # the target must not live beside identity files: ANY target
        # (report OR artifact) inside the checkpoint dir or the corpus
        # store is refused (both sides RESOLVED — a symlink whose name
        # sits outside but points inside is caught). The canonical
        # artifact location lands with the daemon phase; until then,
        # any other path.
        for root, what in (
            (cfg.model.checkpoint_dir.resolve(), "the checkpoint directory"),
            (Path(cfg.corpus.dir).resolve(), "the corpus store"),
        ):
            if target_path == root or root in target_path.parents:
                typer.echo(
                    f"eval failed: {flag} {target!r} lives inside {what} "
                    f"({root}) \u2014 the target must not live beside "
                    "identity files; use any other path",
                    err=True,
                )
                raise typer.Exit(code=2) from None
        if target_path == config_path:
            typer.echo(
                f"eval failed: {flag} {target!r} names the config file "
                f"({config_path}) \u2014 refusing to destroy it",
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
        filters_text = store.read_snapshot(SIDECAR_FILTERS)
        # conditional bridge read (analyze idiom)
        labels_text = (
            store.read_snapshot(SIDECAR_LABELS) if filters_text is not None else None
        )
        posture, filters, id_to_name = parse_attribution(
            filters_text, labels_text, assume_no_filters=assume_no_filters
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
    # text mode: the report renders NOW and the --out file lands NOW —
    # the report is never lost: a post-report refusal below exits 5
    # with the report on stdout AND the --out file in place (the
    # documented "same bytes as stdout" promise holds even when an
    # artifact act refuses)
    if not as_json:
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
            try:
                _write_out_report(out, file_text)
            except OSError as e:
                typer.echo(f"eval failed: cannot write --out: {e}", err=True)
                raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # --json: the payload dump (and the --out write) is deferred until
    # after the serving-gate computation below, so the graduation
    # proposal can ride in the payload; on a post-report refusal the
    # BASE payload (the report, without the proposal) is emitted first
    # — the report is never lost (stdout stays exactly one parseable
    # JSON object; the diff lines are on stderr, never stdout). The
    # payload is built LAZILY inside the emitter (called at most once
    # per run) — a text run never pays for the row/cell tree.
    def _emit_json_payload(
        graduation_payload: dict[str, object] | None = None,
    ) -> None:
        # the parameter is named graduation_payload (not "proposal") —
        # it must not shadow the block-level `proposal` variable the
        # success/write-failure emits pass in
        nonlocal file_text
        payload: dict[str, object] = {
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
        if graduation_payload is not None:
            # the proposal rides in the payload (the machine-readable
            # diff surface — the diff lines stay on stderr)
            payload["graduation"] = graduation_payload
        output = json.dumps(payload, sort_keys=True)
        typer.echo(output)
        file_text = output + "\n"
        if out is not None:
            try:
                _write_out_report(out, file_text)
            except OSError as e:
                typer.echo(f"eval failed: cannot write --out: {e}", err=True)
                raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the deliberate serving-gate acts (ticket #21 bands, ticket #22
    # graduation): the measured table → per-label Threshold Bands →
    # the versioned artifacts bound to the checkpoint + eval identity.
    # Rendered AFTER the report (the report is never lost — a
    # calibration refusal, e.g. an override for a label the vocabulary
    # does not carry, is exit 5 with the report already on stdout);
    # the writes are the ONLY artifact writers in the package
    # (recompute posture, AC4).
    if bands is not None or graduation is not None:
        # the graduation proposal (the machine-readable diff surface) —
        # pre-initialized so the success emit's `proposal if
        # graduation is not None else None` reads one variable, never
        # an unbound name
        proposal: dict[str, object] | None = None
        # serving-gate hardening: the artifacts feed the daemon's
        # serving gate, so they must never be minted from degraded data
        # — a corpus with missing body shards was scored on subject +
        # trailing newline only (the WARNING above), and those scores
        # are not the scores the bands/graduation would serve with.
        # Refuse with exit 5 — the operation-failure convention every
        # other post-report eval refusal uses (the calibration refusal
        # below, the write failure); the report is already on stdout
        # and is preserved, mirroring that posture.
        if missing_bodies:
            degraded_flags = " and ".join(
                flag
                for flag, given in (
                    ("--bands", bands is not None),
                    ("--graduation", graduation is not None),
                )
                if given
            )
            # the refusal reason prints BEFORE the json emit — an emit
            # failure (the --out write inside the emitter) must not
            # swallow the cause
            typer.echo(
                f"eval failed: {degraded_flags} refused: {missing_bodies} "
                "test record(s) had no body shard \u2014 the scores are "
                "degraded (subject + trailing newline only) and the "
                "serving-gate artifacts must not be minted from them; "
                "re-export to repair, then re-run",
                err=True,
            )
            if as_json:
                _emit_json_payload()  # the report is never lost
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
            # the refusal reason prints BEFORE the json emit — an emit
            # failure (the --out write inside the emitter) must not
            # swallow the cause
            typer.echo(f"eval failed: band calibration refused: {e}", err=True)
            if as_json:
                _emit_json_payload()  # the report is never lost
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        # the PURE gate computation (evaluate + build_table) runs
        # BEFORE the --bands write: a gate evaluation refusal must not
        # leave the bands artifact published (the partial-effect class
        # is closed at the source; a write refusal AFTER a prior write
        # remains possible and is named in the message)
        if graduation is not None:
            from .graduation import (
                GateTargets,
                GraduationTable,
                build_table,
                evaluate_graduation,
                identity_disagreement,
                render_graduation_diff,
            )

            # the effective-gates merge (config defaults + per-label
            # overrides) — the same shape as the bands merge
            gate_overrides = {
                label: GateTargets(
                    min_examples=(
                        override.min_examples
                        if override.min_examples is not None
                        else cfg.graduation.min_examples
                    ),
                    min_precision=(
                        override.min_precision
                        if override.min_precision is not None
                        else cfg.graduation.min_precision
                    ),
                )
                for label, override in cfg.graduation.per_label.items()
            }
            try:
                grad = evaluate_graduation(
                    table,
                    artifact,
                    min_examples=cfg.graduation.min_examples,
                    min_precision=cfg.graduation.min_precision,
                    overrides=gate_overrides or None,
                    managed_labels=cfg.labels.managed,
                )
                new_table = build_table(
                    grad,
                    vocabulary=manifest.vocabulary,
                    corpus_fingerprint=expected.corpus_fingerprint,
                    split_manifest_hash=manifest.content_hash,
                    encoder_id=cfg.model.encoder,
                    split_date=manifest.split_date,
                )
            except (ValueError, OSError) as e:
                # the refusal reason prints BEFORE the json emit — an
                # emit failure (the --out write inside the emitter)
                # must not swallow the cause
                typer.echo(
                    f"eval failed: graduation gate evaluation refused: {e}",
                    err=True,
                )
                if as_json:
                    _emit_json_payload()  # the report is never lost
                raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
            # the proposal (the machine-readable diff surface) — parsed
            # ONCE, shared by the success emit and the write-failure
            # emits (no re-parse of the just-serialized text)
            proposal = json.loads(new_table.to_json())
        if bands is not None:
            try:
                _write_bands_artifact(Path(bands), artifact, artifact.to_json())
            except BandsTargetRefusedError as e:
                # the serving-gate hardening: the existing target is a
                # verifiably-different artifact — exit 5, before any
                # staging, the previous bytes intact; the refusal
                # reason prints BEFORE the json emit (an emit failure
                # must not swallow the cause)
                typer.echo(f"eval failed: --bands refused: {e}", err=True)
                if as_json:
                    _emit_json_payload()  # the report is never lost
                raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
            except ArtifactReadbackError as e:
                # the replace ALREADY happened — "cannot write" would
                # be a lie; the previous file may be gone
                typer.echo(
                    f"eval failed: --bands write failed AFTER the target "
                    f"was replaced: {e}; the previous file may be gone; "
                    "re-run to repair",
                    err=True,
                )
                if as_json:
                    _emit_json_payload()  # the report is never lost
                raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
            except OSError as e:
                # the helper already cleaned up its (mkstemp-named)
                # staging; the refusal reason prints BEFORE the json
                # emit (an emit failure must not swallow the cause)
                typer.echo(f"eval failed: cannot write --bands: {e}", err=True)
                if as_json:
                    _emit_json_payload()  # the report is never lost
                raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
            # the human-readable summary (the text-report mode only —
            # the --json payload is the machine-readable output, and
            # the artifact file is the machine-readable bands). On
            # STDERR, so the --out file stays "the same bytes as
            # stdout" (the documented --out contract — the summary is
            # status, not report content)
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
        if graduation is not None:
            # the human-confirm diff (ADR-0005): old → new per label,
            # on STDERR (the report/--out byte-identity stays literally
            # true — the diff is status, not report content). The
            # "existing" side is read + parsed from the target file;
            # an absent/garbage target is the everything-vs-nothing
            # baseline.
            graduation_path = Path(graduation)
            existing: GraduationTable | None = None
            unparseable_existing = False
            unreadable_existing = False
            if graduation_path.is_file():
                # the file EXISTS: the diff says what is wrong with it
                # (honesty about what is on disk) — an UNREADABLE file
                # (an I/O failure) is named differently from an
                # UNPARSEABLE one (a parse failure); a readable file
                # that is not valid UTF-8 is undecodable, hence not a
                # Graduation Table — it is named as unparseable, not
                # "unreadable (I/O failure)" (a misnomer); "no
                # existing table" is reserved for the absent target
                try:
                    text = graduation_path.read_text(encoding="utf-8")
                except OSError:
                    existing = None
                    unreadable_existing = True
                except UnicodeDecodeError:
                    existing = None
                    unparseable_existing = True
                else:
                    try:
                        existing = GraduationTable.from_json(text)
                    except ValueError:
                        existing = None
                        unparseable_existing = True
            disagreeing: tuple[str, ...] = ()
            if existing is not None:
                # the identity fields are named in ONE place
                # (graduation.IDENTITY_FIELDS — the write-path replace
                # gate reads the same list)
                disagreeing = identity_disagreement(existing, new_table)
            typer.echo(
                render_graduation_diff(
                    existing,
                    new_table,
                    path=str(graduation),
                    disagreeing_fields=disagreeing,
                    unparseable_existing=unparseable_existing,
                    unreadable_existing=unreadable_existing,
                ),
                err=True,
                nl=False,
            )
            if confirm_graduation:
                # the write is the CLI-gated act: the helper re-checks
                # the existing target's identity (a foreign-bound table
                # refuses with GraduationTargetRefusedError — exit 5,
                # before any staging, the previous bytes intact)
                table_text = new_table.to_json()
                try:
                    _write_graduation_table(Path(graduation), new_table, table_text)
                except GraduationTargetRefusedError as e:
                    # the refusal reason prints BEFORE the json emit
                    # (an emit failure must not swallow the cause); the
                    # message names the partial effect when the --bands
                    # artifact was already written this run
                    typer.echo(
                        f"eval failed: --graduation refused: {e}"
                        + (
                            f" (the --bands artifact was already written at {bands})"
                            if bands is not None
                            else ""
                        ),
                        err=True,
                    )
                    if as_json:
                        # the report is never lost — and the proposal
                        # rides (the computation succeeded)
                        _emit_json_payload(proposal)
                    raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
                except ArtifactReadbackError as e:
                    # the replace ALREADY happened — "cannot write"
                    # would be a lie; the previous file may be gone
                    typer.echo(
                        f"eval failed: --graduation write failed AFTER "
                        f"the target was replaced: {e}; the previous "
                        "file may be gone; re-run to repair"
                        + (
                            f" (the --bands artifact was already written at {bands})"
                            if bands is not None
                            else ""
                        ),
                        err=True,
                    )
                    if as_json:
                        # the report is never lost — and the proposal
                        # rides (the computation succeeded)
                        _emit_json_payload(proposal)
                    raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
                except OSError as e:
                    # the helper already cleaned up its (mkstemp-named)
                    # staging; the refusal reason prints BEFORE the
                    # json emit (an emit failure must not swallow the
                    # cause), naming the partial effect when the
                    # --bands artifact was already written this run
                    typer.echo(
                        f"eval failed: cannot write --graduation: {e}"
                        + (
                            f" (the --bands artifact was already written at {bands})"
                            if bands is not None
                            else ""
                        ),
                        err=True,
                    )
                    if as_json:
                        # the report is never lost — and the proposal
                        # rides (the computation succeeded)
                        _emit_json_payload(proposal)
                    raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        if as_json:
            # the success path: the payload carries the proposal when
            # the graduation table was computed
            _emit_json_payload(proposal if graduation is not None else None)
    elif as_json:
        # no serving-gate flags: the base payload (the report)
        _emit_json_payload()


shadow_app = typer.Typer(
    help="Shadow Pass: score the recent Backlog, apply NOTHING (ADR-0005).",
    no_args_is_help=True,
)
app.add_typer(shadow_app, name="shadow")


def _shadow_client(config: Config) -> GmailClient:
    """Client factory seam: tests inject FakeGmailClient here (the
    offline spine, invariant #9). The read surface only — the Shadow
    Pass performs NO Gmail write (the write surface lands #24/#25; the
    zero-write pin in the tests depends on that)."""
    from .auth import credentials as build_credentials
    from .gmail_client import GmailRestClient

    return GmailRestClient(build_credentials(config))


def _shadow_model_factory(checkpoint_dir: Path, vocabulary: tuple[str, ...]) -> Model:
    """Model factory seam for ``shadow`` (the ``load_model`` factory
    shape): tests inject a FakeModel producer (the offline spine,
    invariant #9). The real implementation lazy-imports the ``train/``
    recipe (torch — the optional extra), so the CLI module itself stays
    importable without torch."""
    from train.recipe import load_hf_model

    return load_hf_model(checkpoint_dir, vocabulary)


def _backlog_query(since: date, query: str | None) -> str:
    """The Backlog query (the recent inbox, GLOSSARY: Backlog): the
    default is ``in:inbox newer_than:YYYY/MM/DD`` (the --since window);
    an explicit --query is the escape hatch (it replaces the default
    wholesale — the operator's query, the operator's risk)."""
    if query is not None:
        return query
    return f"in:inbox newer_than:{since:%Y/%m/%d}"


@shadow_app.command("run")
def shadow_run(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    bands: str = typer.Option(
        ...,
        "--bands",
        help="Bands Artifact path (required — the calibrated Threshold Bands)",
    ),
    graduation: str = typer.Option(
        ...,
        "--graduation",
        help="Graduation Table path (required — the per-label gate results)",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help=("ISO date (YYYY-MM-DD): score mail newer than it (default: 90 days ago)"),
    ),
    limit: int = typer.Option(
        100, "--limit", min=1, help="Max messages to score (default 100)"
    ),
    query: str | None = typer.Option(
        None,
        "--query",
        help="Gmail search query — replaces the default in:inbox newer_than:… query",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Machine-readable output (graduation workflows)"
    ),
) -> None:
    """Score the recent Backlog through the sidecar-validated
    checkpoint, the Graduation Table, and the Threshold Bands — and
    APPLY NOTHING (Shadow Pass, GLOSSARY: Shadow Pass).

    The verdicts are recorded in the state DB (``[store] db_path``) as
    one transaction per run (a crash mid-run leaves no half rows); the
    report (``shadow report``) diffs them against the mailbox's actual
    labels. Message bodies are fetched for scoring (in-memory only) and
    never enter the DB, the summary, or any log path (invariant #5).

    A message whose body cannot be fetched is SKIPPED (per-message
    fail-safe: warning + count, the run continues) — the documented
    posture choice vs ``eval``'s whole-run refusal: shadow is a
    read-only survey, not a serving-gate artifact mint.
    """
    from datetime import UTC, date, datetime, timedelta

    from .appdb import AppDB, AppDBError, ShadowRow
    from .auth import AuthError
    from .corpus_record import corpus_fingerprint, preprocess_text
    from .corpus_store import CorpusError, CorpusStore
    from .dataset_build import SplitManifest
    from .gmail_client import MessageStub
    from .gmail_errors import GmailAuthError, GmailError
    from .labels import normalize_label_names
    from .mime_extract import decode_gmail_raw, parse_mime
    from .model import (
        MANIFEST_FILENAME,
        THRESHOLDS_SOURCE_NONE,
        CheckpointRefusedError,
        ExpectedIdentity,
        load_model,
    )
    from .tiers import would_apply
    from .tokens import TokenStoreError

    # usage refusals up front (exit 2, before any corpus-scale work):
    # an empty --bands/--graduation would degenerate to Path("") and
    # fail awkwardly mid-run; a non-ISO --since is a usage error (the
    # train --split-date precedent); an empty --query would send an
    # unbounded query to Gmail (the escape hatch is explicit, not
    # accidental)
    for flag, target in (("--bands", bands), ("--graduation", graduation)):
        if not target.strip():
            typer.echo(f"shadow run failed: {flag} PATH must be non-empty", err=True)
            raise typer.Exit(code=2) from None
    if query is not None and not query.strip():
        typer.echo("shadow run failed: --query must be non-empty", err=True)
        raise typer.Exit(code=2) from None
    if since is not None:
        try:
            since_date = date.fromisoformat(since)
        except ValueError:
            typer.echo(
                f"shadow run failed: --since {since!r} is not an ISO date (YYYY-MM-DD)",
                err=True,
            )
            raise typer.Exit(code=2) from None
    else:
        since_date = datetime.now(UTC).date() - timedelta(days=90)

    cfg = _load_config(config)
    # auth gate FIRST (exit 4 precedent): a not-signed-in run must not
    # pay the corpus-scale work below
    try:
        client = _shadow_client(cfg)
    except (AuthError, TokenStoreError) as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None

    store = CorpusStore(cfg.corpus.dir)
    if not store.root.is_dir():
        typer.echo(
            f"shadow run failed: corpus dir {store.root} is missing or not a "
            "directory \u2014 run dataset export/import-gyb first or fix "
            "[corpus] dir",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        records = list(store.iter_records())
    except (CorpusError, OSError, UnicodeDecodeError):
        typer.echo("shadow run failed: records unreadable", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if not records:
        typer.echo(
            "shadow run failed: corpus is empty \u2014 run dataset "
            "export/import-gyb first",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the checkpoint's materialized manifest is the split's authority
    # (invariant #7) — read + verify it BEFORE loading the model (the
    # eval pattern)
    checkpoint_dir = cfg.model.checkpoint_dir
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        typer.echo(
            f"shadow run failed: checkpoint {checkpoint_dir} has no "
            f"{MANIFEST_FILENAME} \u2014 no verifiable split identity "
            "(run train first)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        manifest = SplitManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        typer.echo(
            f"shadow run failed: checkpoint manifest unreadable or tampered: {e}",
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
        model = load_model(checkpoint_dir, expected, _shadow_model_factory)
    except CheckpointRefusedError as e:
        # the refusal contract (the eval precedent): named fields, exit
        # 5, no silent fallback — a mismatched checkpoint is never scored
        typer.echo(
            f"shadow run failed: checkpoint refused ({', '.join(e.fields)}) "
            "\u2014 no silent fallback; rebuild the checkpoint",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except ImportError as e:
        typer.echo(
            f"shadow run failed: the model recipe is unavailable ({e}) "
            "\u2014 install the optional extra: `uv sync --extra train` "
            "(if it is already installed, the recipe import itself is "
            "broken \u2014 report it)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except (ValueError, OSError, RuntimeError) as e:
        typer.echo(f"shadow run failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if model.vocabulary != manifest.vocabulary:
        # a lying column order would silently misalign every score —
        # refuse before scoring (the eval guard, the serving side)
        typer.echo(
            "shadow run failed: the model's vocabulary does not match the "
            "manifest's \u2014 refusing to score against a lying column "
            "order",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the Bands Artifact: read + parse + verify against the run's
    # identity (the FIRST real consumer of verify_bands — the artifact
    # identity gate becomes real here). Cause naming per #22's
    # doctrine: absent vs unreadable (I/O failure) vs undecodable/
    # unparseable — each named by its cause, never misdiagnosed.
    from .thresholds import BandsArtifact, BandsRefusedError, verify_bands

    bands_path = Path(bands)
    if not bands_path.is_file():
        typer.echo(
            f"shadow run failed: no bands artifact at {bands_path} \u2014 "
            "run `eval --bands` first",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        bands_text = bands_path.read_text(encoding="utf-8")
    except OSError:
        typer.echo(
            f"shadow run failed: bands artifact at {bands_path} is "
            "unreadable (I/O failure)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except UnicodeDecodeError:
        typer.echo(
            f"shadow run failed: bands artifact at {bands_path} is not "
            "valid UTF-8 \u2014 it does not parse as a Bands Artifact",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        artifact = BandsArtifact.from_json(bands_text)
    except ValueError as e:
        typer.echo(
            f"shadow run failed: bands artifact at {bands_path} does not "
            f"parse as a Bands Artifact: {e}",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        verify_bands(
            artifact,
            vocabulary=manifest.vocabulary,
            corpus_fingerprint=expected.corpus_fingerprint,
            split_manifest_hash=manifest.content_hash,
            encoder_id=cfg.model.encoder,
            split_date=manifest.split_date,
            test_examples=len(manifest.test_ids),
        )
    except BandsRefusedError as e:
        # the identity refusal (the sidecar doctrine): named fields,
        # exit 5 — a foreign-bound artifact is never served
        typer.echo(
            f"shadow run failed: bands artifact refused ({', '.join(e.fields)}) "
            "\u2014 it disagrees with this checkpoint/eval; re-run "
            "`eval --bands` against this checkpoint",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except ValueError as e:
        # a naive-dated split_date is a caller bug, not identity drift
        # (verify_bands' documented refusal) — operation failure
        typer.echo(f"shadow run failed: bands verification failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the Graduation Table: read + parse + verify against the run's
    # identity + the bands binding (the FIRST real consumer of
    # verify_graduation — the table identity gate becomes real here)
    from .graduation import (
        GraduationRefusedError,
        GraduationTable,
        verify_graduation,
    )

    graduation_path = Path(graduation)
    if not graduation_path.is_file():
        typer.echo(
            f"shadow run failed: no graduation table at {graduation_path} "
            "\u2014 run `eval --graduation --confirm-graduation` first",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        graduation_text = graduation_path.read_text(encoding="utf-8")
    except OSError:
        typer.echo(
            f"shadow run failed: graduation table at {graduation_path} is "
            "unreadable (I/O failure)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except UnicodeDecodeError:
        typer.echo(
            f"shadow run failed: graduation table at {graduation_path} is "
            "not valid UTF-8 \u2014 it does not parse as a Graduation Table",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        table = GraduationTable.from_json(graduation_text)
    except ValueError as e:
        typer.echo(
            f"shadow run failed: graduation table at {graduation_path} "
            f"does not parse as a Graduation Table: {e}",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        verify_graduation(
            table,
            vocabulary=manifest.vocabulary,
            corpus_fingerprint=expected.corpus_fingerprint,
            split_manifest_hash=manifest.content_hash,
            encoder_id=cfg.model.encoder,
            split_date=manifest.split_date,
            test_examples=len(manifest.test_ids),
            bands_content_hash=artifact.content_hash,
        )
    except GraduationRefusedError as e:
        typer.echo(
            f"shadow run failed: graduation table refused "
            f"({', '.join(e.fields)}) \u2014 it disagrees with this "
            "checkpoint/eval/bands; re-run `eval --graduation` against "
            "this checkpoint",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except ValueError as e:
        typer.echo(
            f"shadow run failed: graduation verification failed: {e}",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the label bridge (presence + System Labels): the id→name map and
    # the system-name set (the read path — list_labels only)
    try:
        labels = client.list_labels()
    except GmailAuthError as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailError as e:
        typer.echo(f"shadow run failed: label list failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    id_to_name = {li.id: li.name for li in labels}
    system_names = frozenset(li.name for li in labels if li.type == "system")

    # the Backlog fetch (the read path — list_messages only): the
    # pageToken loop honors --limit across pages (#17's historyId
    # wire-fidelity precedent)
    q = _backlog_query(since_date, query)
    try:
        stubs: list[MessageStub] = []
        seen_ids: set[str] = set()
        token: str | None = None
        while len(stubs) < limit:
            page = client.list_messages(q, token)
            new = [s for s in page.stubs if s.id not in seen_ids]
            # a page that yields NO new ids (a repeated/empty-with-token
            # page) would loop forever — break on no progress
            if not new and token is not None:
                break
            for s in new:
                seen_ids.add(s.id)
                stubs.append(s)
            token = page.next_page_token
            if token is None:
                break
        stubs = stubs[:limit]
    except GmailAuthError as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailError as e:
        # transport failure after retries (the gmail_errors precedent)
        typer.echo(f"shadow run failed: backlog fetch failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # per message: get_message → decode → parse → preprocess_text
    # (invariant #4: the scoring text comes ONLY from
    # corpus_record.preprocess_text). A fetch failure is a per-message
    # SKIP (the documented posture choice vs eval's refusal — shadow is
    # a read-only survey): warning + count, the run continues.
    texts: list[str] = []
    fetched: list[tuple[str, frozenset[str]]] = []  # (id, present names)
    skipped = 0
    for stub in stubs:
        try:
            raw = client.get_message(stub.id)
        except GmailAuthError as e:
            typer.echo(f"auth: {e}", err=True)
            raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
        except GmailError as e:
            skipped += 1
            typer.echo(f"WARNING: skipped message {stub.id}: {e}", err=True)
            continue
        try:
            data = decode_gmail_raw(raw.raw_b64url)
        except (ValueError, UnicodeEncodeError) as e:
            skipped += 1
            typer.echo(
                f"WARNING: skipped message {stub.id}: raw undecodable "
                f"({type(e).__name__})",
                err=True,
            )
            continue
        parsed = parse_mime(data)  # never raises (the mime contract)
        present = frozenset(normalize_label_names(raw.label_ids, id_to_name))
        texts.append(preprocess_text(parsed.subject, parsed.body_text))
        fetched.append((stub.id, present))
    if not fetched:
        typer.echo(
            f"shadow run failed: no messages to score (backlog query "
            f"{q!r} returned {len(stubs)} message(s), {skipped} skipped) "
            "\u2014 nothing recorded",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the batched predict (the #20 batching seam: ONE call over all
    # texts — the model materializes the iterable once)
    try:
        rows = model.predict(texts)
        if len(rows) != len(fetched):
            raise ValueError(
                f"the model returned {len(rows)} score rows for "
                f"{len(fetched)} texts \u2014 checkpoint/recipe defect"
            )
    except (ValueError, OSError, RuntimeError) as e:
        typer.echo(f"shadow run failed: prediction failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the would-apply verdicts (the single decision core — tiers.py):
    # graduated labels only, skip-present, System Labels never targeted
    bands_map = {entry.label: (entry.high, entry.med) for entry in artifact.labels}
    graduated = frozenset(entry.label for entry in table.labels if entry.graduated)
    run_ts = datetime.now(UTC)
    run_id = uuid.uuid4().hex
    all_rows: list[ShadowRow] = []
    json_rows: list[dict[str, object]] = []
    proposed = 0
    already_present = 0
    not_applied = 0
    for (message_id, present), row in zip(fetched, rows, strict=True):
        # the zip/dict bookkeeping is OUTSIDE the try: a vocabulary/
        # score-width defect there is a genuine bug (exit 1), not a
        # would-apply refusal
        scores = dict(zip(model.vocabulary, row, strict=True))
        try:
            decisions = would_apply(
                scores=scores,
                bands=bands_map,
                graduated=graduated,
                present=present,
                system_labels=system_names,
            )
        except ValueError as e:
            # would_apply's documented refusals (a lying score/band, a
            # graduated System Label, a missing bands entry) are named
            # operation failures (exit 5) — not an unhandled traceback
            # (exit 1, which the README defines as "a bug")
            typer.echo(f"shadow run failed: would-apply refused: {e}", err=True)
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        for d in decisions:
            all_rows.append(
                ShadowRow(
                    message_id=message_id,
                    label=d.label,
                    probability=d.score,
                    band=d.band,
                    run_ts=run_ts,
                )
            )
            json_rows.append(
                {
                    "message_id": message_id,
                    "label": d.label,
                    "probability": d.score,
                    "band": d.band,
                    "present": d.present,
                    "applied": d.applied,
                }
            )
            if d.applied:
                proposed += 1
            elif d.present and d.band is not None:
                already_present += 1
            else:
                not_applied += 1

    if not all_rows:
        # a run that produces ZERO verdicts (e.g. the Graduation Table
        # graduated no labels) records nothing — refusing with exit 5
        # keeps the run representable (a `shadow report` on an empty DB
        # would say "no runs recorded", a silent loss of the run)
        typer.echo(
            "shadow run failed: no verdicts to record (the Graduation "
            "Table graduated no labels) \u2014 nothing recorded",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the store (transaction per run — a crash mid-run leaves no half
    # rows; the DB carries ids/probabilities/labels/timestamps ONLY —
    # invariant #5)
    db = AppDB(cfg.store.db_path)
    try:
        db.open()
        db.record_run(run_id, run_ts, all_rows)
    except (AppDBError, OSError) as e:
        typer.echo(f"shadow run failed: state DB write failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    finally:
        db.close()

    if as_json:
        payload = {
            "run_id": run_id,
            "run_ts": run_ts.isoformat(),
            "query": q,
            "since": since_date.isoformat(),
            "limit": limit,
            "scored": len(fetched),
            "skipped": skipped,
            "proposed": proposed,
            "already_present": already_present,
            "not_applied": not_applied,
            "rows": json_rows,
        }
        typer.echo(json.dumps(payload, sort_keys=True))
    else:
        # the summary is STATUS, not report content — on stderr (the
        # report is `shadow report`'s stdout)
        typer.echo(
            f"shadow run: run={run_id} scored={len(fetched)} "
            f"skipped={skipped} proposed={proposed} "
            f"already_present={already_present} not_applied={not_applied}",
            err=True,
        )


@dataclass(frozen=True, slots=True)
class ReportRow:
    """One would-apply verdict row in the shadow report diff (ids +
    labels + scores + tiers ONLY — no subjects, no bodies, invariant
    #5). ``outcome`` is the diff verdict (``added`` / ``already-present``
    / ``not-applied``); ``tier`` is the Tier Label name (``None`` for
    skip-present and no-band — the single home is ``tiers.tier_for_band``)."""

    label: str
    probability: float
    band: str | None
    outcome: str
    tier: str | None


@shadow_app.command("report")
def shadow_report(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    run: str | None = typer.Option(
        None,
        "--run",
        help="The run id to report (default: the latest run)",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Machine-readable output (graduation workflows)"
    ),
) -> None:
    """Diff a Shadow Pass run's would-apply verdicts against the
    mailbox's ACTUAL labels (the Phase 1 exit review, GLOSSARY: Shadow
    Pass).

    Reads the run's rows from the state DB (``[store] db_path``),
    re-fetches each message's ACTUAL labels via the read path
    (get_message — the labels may have changed since the run), and
    renders a deterministic per-message diff: message id + label +
    score + tier ONLY (subjects OUT, bodies out by construction —
    invariant #5). Outcomes: ADDED (would apply, not present),
    ALREADY-PRESENT (would apply, already present — skip-present
    visible), NOT-APPLIED (no band verdict). A per-label summary block
    carries the would-be tier mix (ONLY labels that would actually be
    added — skip-present labels are not applied, so they are not
    counted). A message whose labels cannot be re-fetched is SKIPPED
    (per-message fail-safe: warning + count, the report continues) —
    the same posture as ``shadow run``; if EVERY message is skipped
    (no evidence) the report refuses with exit 5 (mirroring shadow
    run's zero-verdict posture).

    The verdicts are bound to the bands/graduation artifacts AS OF THE
    RUN — the report does NOT re-verify them (a basis disclosure, in
    the header + the ``basis`` payload field). Hash persistence is
    deferred to #26-27 (the status/corrections surface); re-run
    ``shadow run`` after any retrain/recalibration to refresh the
    basis.
    """
    from .appdb import AppDB, AppDBError
    from .auth import AuthError
    from .gmail_errors import GmailAuthError, GmailError
    from .labels import normalize_label_names
    from .tiers import BAND_HIGH, BAND_MED, tier_for_band
    from .tokens import TokenStoreError

    # usage refusal up front (exit 2): an empty --run would degenerate
    # to a lookup on "" (the latest-run default is the empty-ABSENT
    # case, not an empty-string run id)
    if run is not None and not run.strip():
        typer.echo("shadow report failed: --run must be non-empty", err=True)
        raise typer.Exit(code=2) from None

    cfg = _load_config(config)
    # auth gate FIRST (exit 4 precedent): the report re-fetches the
    # actual labels via the read path
    try:
        client = _shadow_client(cfg)
    except (AuthError, TokenStoreError) as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None

    # the state DB (create-all-if-missing — a report on an empty DB is
    # a named "no runs" refusal, not a crash)
    db = AppDB(cfg.store.db_path)
    try:
        db.open()
        run_id = run if run is not None else db.latest_run()
        if run_id is None:
            typer.echo(
                "shadow report failed: no shadow runs recorded \u2014 run "
                "`shadow run` first",
                err=True,
            )
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        # an unknown --run is refused by load_run's NAMED refusal (a
        # separate run_exists pre-check would be a redundant second
        # lookup — the reader is the authority)
        run_ts, rows = db.load_run(run_id)
    except (AppDBError, OSError) as e:
        typer.echo(f"shadow report failed: state DB read failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    finally:
        db.close()

    # the label bridge (the read path — list_labels only)
    try:
        labels = client.list_labels()
    except GmailAuthError as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailError as e:
        typer.echo(f"shadow report failed: label list failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    id_to_name = {li.id: li.name for li in labels}

    # re-fetch the ACTUAL labels per message (the read path —
    # get_message only). A fetch failure is a per-message SKIP (the
    # documented posture choice — the report is a review, not a
    # serving-gate mint): warning + count, the report continues.
    message_ids = sorted({row.message_id for row in rows})
    actual: dict[str, frozenset[str]] = {}
    skipped = 0
    for message_id in message_ids:
        try:
            raw = client.get_message(message_id)
        except GmailAuthError as e:
            typer.echo(f"auth: {e}", err=True)
            raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
        except GmailError as e:
            skipped += 1
            typer.echo(f"WARNING: skipped message {message_id}: {e}", err=True)
            continue
        actual[message_id] = frozenset(normalize_label_names(raw.label_ids, id_to_name))

    # the deterministic per-message diff (ids + labels + scores + tiers
    # ONLY — subjects OUT, bodies out by construction, invariant #5).
    # Outcomes: ADDED (band + not present), ALREADY-PRESENT (band +
    # present — skip-present visible), NOT-APPLIED (no band).
    per_message: dict[str, list[ReportRow]] = {}
    per_label: dict[str, dict[str, int]] = {}
    outcomes = {"added": 0, "already_present": 0, "not_applied": 0}
    for row in rows:  # load_run returns (message_id, label) order
        present = actual.get(row.message_id)
        if present is None:
            # the message was skipped (fetch failure) — its rows are
            # omitted from the diff (the skip is counted + warned)
            continue
        is_present = row.label in present
        if row.band is None:
            outcome = "not-applied"
        elif is_present:
            outcome = "already-present"
        else:
            outcome = "added"
        # the band→tier mapping + skip-present live in tiers.tier_for_band
        # (the single decision core — no second implementation): an
        # already-present label earns NO Tier Label (present=True → None)
        tier = tier_for_band(row.band, present=is_present)
        outcomes[outcome.replace("-", "_")] += 1
        per_message.setdefault(row.message_id, []).append(
            ReportRow(
                label=row.label,
                probability=row.probability,
                band=row.band,
                outcome=outcome,
                tier=tier,
            )
        )
        label_summary = per_label.setdefault(
            row.label,
            {
                "added": 0,
                "already_present": 0,
                "not_applied": 0,
                "would_be_high": 0,
                "would_be_med": 0,
            },
        )
        label_summary[outcome.replace("-", "_")] += 1
        # the would-be tier mix reflects ONLY labels that would ACTUALLY
        # be added (skip-present: an already-present label is not
        # applied, so it is NOT counted in the would-be mix) — the band
        # constants come from tiers (the single decision core)
        if outcome == "added":
            if row.band == BAND_HIGH:
                label_summary["would_be_high"] += 1
            elif row.band == BAND_MED:
                label_summary["would_be_med"] += 1

    if not per_message:
        # ALL messages were skipped (fetch failure) — no evidence to
        # diff. Refuse with exit 5 (mirroring shadow run's zero-verdict
        # posture): a report with zero evidence is not a review.
        typer.echo(
            f"shadow report failed: no evidence to report (all {skipped} "
            "message(s) were skipped — labels could not be re-fetched) "
            "\u2014 re-run when the mailbox is reachable",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the basis disclosure (T3): the verdicts are bound to the
    # bands/graduation artifacts AS OF THE RUN — the report does NOT
    # re-verify them (hash persistence is deferred to #26-27, when the
    # status/corrections surface needs it). A retrain/recalibration
    # moves the basis; re-run shadow run to refresh.
    basis = {
        "note": (
            "verdicts bound to the bands/graduation artifacts as of the "
            "run — not re-verified; re-run shadow run after any "
            "retrain/recalibration"
        ),
    }

    if as_json:
        payload = {
            "run_id": run_id,
            "run_ts": run_ts.isoformat(),
            "message_count": len(per_message),
            "skipped": skipped,
            "outcomes": outcomes,
            "basis": basis,
            "messages": [
                {
                    "message_id": message_id,
                    "rows": [
                        {
                            "label": r.label,
                            "probability": r.probability,
                            "band": r.band,
                            "outcome": r.outcome,
                            "tier": r.tier,
                        }
                        for r in per_message[message_id]
                    ],
                }
                for message_id in sorted(per_message)
            ],
            "per_label": [
                {"label": label, **per_label[label]} for label in sorted(per_label)
            ],
        }
        typer.echo(json.dumps(payload, sort_keys=True))
        return

    # text mode: the report is the stdout content (the review surface)
    typer.echo(
        f"shadow report: run={run_id} ts={run_ts.isoformat()} "
        f"messages={len(per_message)} skipped={skipped}"
    )
    typer.echo(f"  basis: {basis['note']}")
    for message_id in sorted(per_message):
        typer.echo(f"  {message_id}:")
        for r in per_message[message_id]:
            band = r.band if r.band is not None else "-"
            tier = r.tier if r.tier is not None else "-"
            # the outcome renders UPPERCASE (the design's ADDED /
            # ALREADY-PRESENT / NOT-APPLIED); the JSON keeps lowercase
            # (machine-readable)
            typer.echo(
                f"    {r.outcome.upper():<16} {r.label:<24} "
                f"score={r.probability:.2f} band={band} tier={tier}"
            )
    typer.echo("  per-label:")
    for label in sorted(per_label):
        s = per_label[label]
        typer.echo(
            f"    {label}: added={s['added']} "
            f"already-present={s['already_present']} "
            f"not-applied={s['not_applied']} "
            f"(would-be: high={s['would_be_high']} med={s['would_be_med']})"
        )


if __name__ == "__main__":
    app()
