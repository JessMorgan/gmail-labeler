"""gmail-labeler command-line interface.

Command surface is frozen by the Phase 0 design (AGENTS.md "Layout");
the Phase 1 ``train``/``eval``/``shadow`` commands, the Phase 2 ``daemon``
(ticket #25), and the Phase 2 ``apply`` bulk path (ticket #26) are wired
as they are implemented; ``status`` remains absent until wired (which
surfaces as a usage error). Exit 3 is reserved for wired-but-stubbed
commands and currently UNREACHABLE — no command is stubbed; scripts can
branch on 0/2/4/5 today.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import signal
import tempfile
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

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
    from .config import Config, LabelsConfig
    from .corpus_record import CorpusRecord
    from .daemon import ServingContext
    from .gmail_client import GmailClient
    from .graduation import GraduationTable
    from .thresholds import BandsArtifact
    from .wizard import ClientFileState, Prompt

app = typer.Typer(
    help="Strictly-local Gmail auto-labeling, trained on your own inbox.",
    epilog=(
        "Exit codes: 0 ok · 2 usage error · 3 not implemented · "
        "4 auth/config unusable · 5 operation failed or incomplete · "
        "130 cancelled (Ctrl-C)"
    ),
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Corpus acquisition and analysis.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")

auth_app = typer.Typer(help="Gmail OAuth credential (ADR-0004).", no_args_is_help=False)
app.add_typer(auth_app, name="auth")

corrections_app = typer.Typer(
    help="Corrections: the human hand-fixes captured as training signal.",
    no_args_is_help=True,
)
app.add_typer(corrections_app, name="corrections")

_AUTH_ERROR_EXIT = 4  # 2 = usage, 3 = not implemented, 4 = auth/config unusable
_INCOMPLETE_EXIT = 5  # operation failed or incomplete (write/declined/export)
# Ctrl-C at a prompt exits 130 — the SAME code typer already uses for a raw
# KeyboardInterrupt (`except KeyboardInterrupt: raise Exit(130)`), so the whole
# CLI signals "cancelled by signal" identically. (typer's prompt/confirm would
# otherwise downgrade a prompt-cancel to Abort→exit 1; the wizard maps it back
# to 130 — see _typer_ask/_typer_confirm.)
_CANCEL_EXIT = 130
# (typer requires concrete annotations: str | None, not Any)


@auth_app.callback(invoke_without_command=True)
def auth_status(
    ctx: typer.Context,
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
) -> None:
    """Report credential status; `auth login` runs the consent flow, `auth
    setup` the guided ADR-0004 onboarding."""
    if ctx.invoked_subcommand is not None:
        return
    from .auth import status as _status
    from .config import ConfigError, load_config

    # A MISSING config (fresh profile) is "no credentials", not a hard error —
    # the wizard is the answer. An EXPLICIT --config that is broken stays a
    # config error (exit 4): the user pointed at a bad path, not a fresh box.
    if config is not None:
        try:
            cfg = load_config(config)
        except ConfigError as e:
            typer.echo(f"config error: {e}", err=True)
            raise typer.Exit(code=_AUTH_ERROR_EXIT) from e
    else:
        # No explicit --config: distinguish "no config file anywhere" (a fresh
        # profile → the wizard is the answer) from "a config exists but is
        # invalid" (a real config error → exit 4, the message preserved). A
        # bare `except ConfigError: cfg=None` would conflate the two (review
        # MAJOR 3): a typo'd ./config.toml must NOT be reported as "no config".
        from .config import find_config

        try:
            found = find_config()
        except ConfigError:
            cfg = None  # genuinely nothing found → fresh profile
        else:
            try:
                cfg = load_config(str(found))
            except ConfigError as e:
                typer.echo(f"config error: {e}", err=True)
                raise typer.Exit(code=_AUTH_ERROR_EXIT) from e
    if cfg is None:
        # B7: the informative hint (looked-in paths + copy-example.toml) that
        # the C2-era rewrite silently dropped — restored; exit stays 4 (F1).
        typer.echo(
            "auth: NOT SIGNED IN — no config.toml found (looked in ./ and "
            "~/.config/gmail-labeler/); copy example.toml and edit, or run "
            "`gmail-labeler auth setup` to create one."
        )
    else:
        st = _status(cfg)
        state = "SIGNED IN" if st.signed_in else "NOT SIGNED IN"
        typer.echo(f"auth: {state} ({st.backend}) — {st.detail}")
        if st.signed_in:
            return  # signed in: bare `auth` keeps its status meaning, exit 0
    # No credentials → the wizard ENTRY-POINT decision. Interactive only on a
    # real terminal; a script must never half-prompt.
    if _stdin_is_tty():
        typer.echo(
            "Not signed in. Starting the guided setup (ADR-0004) — about 15 "
            "minutes in the Google console, nothing shared with the maintainer."
        )
        _run_setup_wizard(config)
    else:
        # F1 (brief corrected at source): an unsigned non-TTY `auth` is an
        # AUTH-STATE, not an operation failure — it stays exit 4 (auth/config
        # unusable), the contract scripts have always branched on. The
        # wizard-entry NAMING stays (it names `auth setup`); only the code is
        # 4, not 5.
        typer.echo(
            "Not signed in and stdin is not a terminal — run "
            "`gmail-labeler auth setup` interactively.",
            err=True,
        )
        raise typer.Exit(code=_AUTH_ERROR_EXIT)


@auth_app.command("setup")
def auth_setup(
    config: str | None = typer.Option(
        None, "--config", help="Where to WRITE config.toml (default: ./config.toml)"
    ),
) -> None:
    """Guided ADR-0004 onboarding: create your own GCP project + desktop OAuth
    client and wire the token store, then sign in. Interactive (a TTY); it
    COMPOSES the existing `auth login` flow rather than forking it."""
    _run_setup_wizard(config)


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


# --- auth-setup wizard wiring (the CLI side of the pure machine) -------------
def _stdin_is_tty() -> bool:
    """Whether bare `auth` may auto-launch the wizard. Kept as a tiny seam so
    the tests can drive both the interactive and the script path without a
    controlling terminal."""
    import sys

    return sys.stdin.isatty()


def _project_id_shape_ok(pid: str) -> bool:
    """GCP project-id SHAPE only (a paste-error guard): 6-30 chars, lowercase
    letters/digits/hyphens, starting with a letter. The project's EXISTENCE and
    API enablement are not locally checkable — the wizard says so, never fakes
    a check."""
    import re

    return re.fullmatch(r"[a-z][a-z0-9-]{5,29}", pid) is not None


class _RealEnvironmentProbe:
    """The filesystem/keyring facts the wizard may check LOCALLY. Returns
    verdicts only — never file contents — so secret material cannot reach the
    pure machine (invariant #8)."""

    def client_file_state(self, path: str) -> ClientFileState:
        from .wizard import ClientFileState

        p = Path(path).expanduser()
        if not p.is_file():
            return ClientFileState.MISSING
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return ClientFileState.MALFORMED
        if not isinstance(data, dict) or not isinstance(data.get("installed"), dict):
            return ClientFileState.NOT_DESKTOP  # e.g. a "web" wrapper
        inst = data["installed"]
        if not inst.get("client_id") or not inst.get("client_secret"):
            return ClientFileState.MALFORMED
        return ClientFileState.DESKTOP_OK

    def project_id_shape_ok(self, pid: str) -> bool:
        return _project_id_shape_ok(pid)

    def keyring_available(self) -> bool:
        # reuse the existing keyring posture (tokens.py) — a working backend
        # answers, a headless box raises TokenStoreError.
        from .tokens import KeyringStore, TokenStoreError

        try:
            KeyringStore().probe()
        except TokenStoreError:
            return False
        return True

    def config_target_exists(self, path: Path) -> bool:
        # resolve()-compare posture, stated: existence is a resolved-path
        # compare; the samefile/fd caution family is issue #47's scope.
        try:
            return Path(path).expanduser().resolve().exists()
        except OSError:
            return False


def _typer_ask(prompt: Prompt) -> str:
    from typer.exceptions import Abort

    from .wizard import WizardCancelled

    try:
        return typer.prompt(prompt.text)
    except (Abort, KeyboardInterrupt, EOFError):
        raise WizardCancelled from None


def _typer_confirm(text: str) -> bool:
    from typer.exceptions import Abort

    from .wizard import WizardCancelled

    try:
        # default=False: an overwrite is never the accidental Enter key.
        return typer.confirm(text, default=False)
    except (Abort, KeyboardInterrupt, EOFError):
        raise WizardCancelled from None


def _write_config_toml(target: Path, cfg: Config) -> None:
    """Write a minimal config.toml carrying ONLY the gmail wiring (paths). No
    secret values, ever (invariant #8). Absolute paths so the record is
    unambiguous regardless of the later cwd."""
    cred = cfg.gmail.credentials_path
    cred_s = str(cred.expanduser().resolve()) if cred is not None else ""
    body = (
        "# gmail-labeler configuration — written by `auth setup` (ADR-0004).\n"
        "# Secrets are paths only — never tokens, never inline values.\n"
        "[gmail]\n"
        f"credentials_path = {json.dumps(cred_s)}\n"
        f"token_backend = {json.dumps(cfg.gmail.token_backend)}\n"
        f"token_file = {json.dumps(str(cfg.gmail.token_file.expanduser()))}\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    # atomic write: temp in the same dir, fsync, then replace — a torn config
    # would break every later command (the durability the comment promises,
    # matching FileTokenStore.save). The config is paths-only (invariant #8), so
    # it is mode 0644 like example.toml — NOT the 0600 a token file needs. An
    # OSError here (parent is a file, read-only dir, …) propagates and the
    # wizard's write_config catch turns it into a write-error outcome (exit 5) —
    # never a raw traceback (exit 1).
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".")
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _setup_config_target(config: str | None) -> Path:
    from .config import ConfigError, find_config

    if config is not None:
        return Path(config).expanduser()
    try:
        return find_config()
    except ConfigError:
        return Path.cwd() / "config.toml"


def _echo_out(text: str) -> None:
    """The wizard's output sink — a named seam so tests can simulate a closed
    stdout (BrokenPipeError) without patching typer globally."""
    typer.echo(text)


def _run_setup_wizard(config: str | None) -> None:
    """Wire the real I/O + the real auth.login/status/config-writer into the
    pure machine, then map its outcome onto the CLI exit contract."""
    from .auth import login as _login
    from .auth import status as _status
    from .wizard import WizardCancelled, WizardIO, run_wizard

    target = _setup_config_target(config)
    io = WizardIO(ask=_typer_ask, confirm=_typer_confirm, out=_echo_out)
    try:
        result = run_wizard(
            probe=_RealEnvironmentProbe(),
            io=io,
            login=_login,
            status=_status,
            write_config=_write_config_toml,
            config_target=target,
        )
    except WizardCancelled:
        typer.echo("setup cancelled.", err=True)
        raise typer.Exit(code=_CANCEL_EXIT) from None
    except BrokenPipeError:
        # stdout closed mid-run (`auth setup | head`): the pipe is gone, NOT a
        # config-write failure. The old broad `except OSError` mislabeled this
        # as "setup failed writing config.toml" while nothing was being
        # written — cancel-class (130) is the honest handling.
        raise typer.Exit(code=_CANCEL_EXIT) from None
    if result.outcome == "completed" and result.signed_in:
        return  # exit 0
    if result.outcome == "failed":
        raise typer.Exit(code=_AUTH_ERROR_EXIT)  # 4 — sign-in did not complete
    # write-error / status-error: reported on STDERR (consistent with config
    # error / setup cancelled / the non-TTY message), one surface, then exit 5.
    if result.outcome == "write-error":
        typer.echo(f"setup: {result.detail}", err=True)
        raise typer.Exit(code=_INCOMPLETE_EXIT)  # 5 — could not write the config
    if result.outcome == "status-error":
        typer.echo(f"setup: {result.detail}", err=True)
        raise typer.Exit(code=_INCOMPLETE_EXIT)  # 5 — the sign-in check crashed
    # completed-but-not-signed-in, or declined overwrite: operation incomplete.
    raise typer.Exit(code=_INCOMPLETE_EXIT)  # 5


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


def _write_out_report(out: str, file_text: str, *, label: str = "eval") -> None:
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
    which is after the acts — the bytes are the same). ``label`` names
    the announcing command (default "eval"; corrections export passes
    its own name so the NOTICE is not misattributed). Raises OSError
    on failure (the caller maps it to exit 5)."""
    target = Path(out)
    if target.exists():
        typer.echo(f"{label}: NOTICE: replacing existing {out}", err=True)
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


def _is_str_keyed_dict(obj: object) -> TypeGuard[dict[str, object]]:
    return isinstance(obj, dict) and all(isinstance(k, str) for k in obj)


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
    raw_filters = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(raw_filters, list):
        raise ValueError("filters snapshot is not a list")
    filters: list[dict[str, object]] = []
    for item in raw_filters:
        if not _is_str_keyed_dict(item):
            raise ValueError("filter entry is not a string-keyed object")
        filters.append({str(k): v for k, v in item.items()})
    id_to_name: dict[str, str] = {}
    labels_path = store_root / SIDECAR_LABELS
    if labels_path.is_file():
        raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if not _is_str_keyed_dict(raw_labels) or not all(
            isinstance(v, str) for v in raw_labels.values()
        ):
            raise ValueError("labels snapshot is not an id->name object")
        id_to_name = {str(k): str(v) for k, v in raw_labels.items()}
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
    # The posture NOTEs below matter only when there is mail to report on.
    # For an empty corpus the human path already ends with the
    # "corpus is empty — run dataset export/import-gyb first" line, and the
    # --json path prints no human report on stdout — its payload carries the
    # posture in filters_posture; NOTEs here would just bury the advice.
    if summary.messages:
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
    Pass performs NO Gmail write (the write surface LANDED in #24 — the
    zero-write pin catches a read-path write BY NAME)."""
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


def _resolve_db_path(cfg: Config, db: str | None, command: str) -> Path:
    """The effective state DB path (D3): the ``--db`` flag is a
    usage-level override of ``[store] db_path`` (consistent with the
    flag doctrine — an explicit flag beats the config). An empty --db is
    a usage refusal (exit 2). The value is STRIPPED (a trailing-space
    filename would be a bug) and ``~`` is expanded (matching the config
    path fields). A relative --db anchors to the CWD (like --bands /
    --graduation) — unlike the config's ``[store] db_path``, which
    anchors to the config dir."""
    if db is not None:
        if not db.strip():
            typer.echo(f"{command} failed: --db PATH must be non-empty", err=True)
            raise typer.Exit(code=2) from None
        return Path(db.strip()).expanduser()
    return cfg.store.db_path


def _check_store_db_path(db_path: Path, cfg: Config, command: str, origin: str) -> None:
    """S3: the state DB path must not live inside the checkpoint dir or
    the corpus store (both sides RESOLVED — a link named outside
    pointing inside is caught). A state DB in there is silently
    destroyable: ``train --force`` destroys the checkpoint dir (a state
    DB in it dies with it), and a corpus re-export rewrites the store.
    Usage refusal exit 2 (the eval containment loop shape). ``origin``
    names the SOURCE of the path (``--db`` vs ``[store] db_path``) so
    the refusal message points at the right knob (disposition 6-cli)."""
    db_path = db_path.resolve()
    for root, what in (
        (cfg.model.checkpoint_dir.resolve(), "the checkpoint directory"),
        (Path(cfg.corpus.dir).resolve(), "the corpus store"),
    ):
        if db_path == root or root in db_path.parents:
            typer.echo(
                f"{command} failed: {origin} {db_path} lives inside "
                f"{what} ({root}) \u2014 a state DB there is destroyable by "
                "train --force / a corpus re-export; use any other path",
                err=True,
            )
            raise typer.Exit(code=2) from None


def _validate_store_db(db_path: Path, command: str = "shadow run") -> None:
    """S5: validate the state DB FAIL-FAST (open + ownership + version)
    before any network work — a foreign/corrupt/foreign-version DB is
    refused (exit 5) before we pay for list_labels/list_messages. The
    connection is closed immediately (validation only); the write
    re-opens it at the store step. ``command`` names the calling
    command in the refusal (the daemon reuses the gate; the shadow
    default keeps its sealed message verbatim)."""
    from .appdb import AppDB, AppDBError

    db = AppDB(db_path)
    try:
        db.open()
    except (AppDBError, OSError) as e:
        typer.echo(f"{command} failed: state DB unusable: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    finally:
        db.close()


def _now() -> datetime:
    """The injectable clock seam (fold-4 item 2): the single
    ``datetime.now(UTC)`` read per ``shadow run`` command (its window math
    and run_ts both go through it — ONE read; the other commands
    (train/eval/analyze) still read the clock directly, which is #24's
    clock-seam reuse item, not this ticket's scope). A test can
    monkeypatch it to a fixed aware datetime and pin the EXACT shipped
    default (now - 90d, time kept) at the CLI level (the recorded
    ``list_messages`` query)."""
    return datetime.now(UTC)


def _backlog_cutoff(since_date: date | None, now: datetime) -> datetime:
    """The UTC-aware cutoff datetime for the Backlog query (S4): a
    --since date → UTC MIDNIGHT of that date — the named day is
    INCLUDED (``after:<epoch>`` is strictly-newer than that midnight, so
    mail from the named day onward is in the window); the default
    (``None``) → ``now - 90 days`` WITH the time kept (the boundary is
    "90 days ago", not "90 days ago at midnight" — the documented
    intent, disposition 1). Pure (takes ``now``) so the epoch math is
    testable with a fixed instant."""
    if since_date is not None:
        return datetime(since_date.year, since_date.month, since_date.day, tzinfo=UTC)
    return now - timedelta(days=90)


def _backlog_query(cutoff_dt: datetime, query: str | None) -> str:
    """The Backlog query (the recent inbox, GLOSSARY: Backlog): the
    default is ``in:inbox after:<epoch-seconds>`` — the EPOCH form is
    unambiguous. The Gmail API guide warns that calendar dates in API
    queries are interpreted as MIDNIGHT PST (so a calendar string like
    ``after:2026-10-01`` is NOT the UTC boundary we want), and
    documents epoch seconds for exact boundaries; ``newer_than:`` is
    documented as RELATIVE-DURATION only (d/m/y units), not an absolute
    date. An explicit --query is the escape hatch (it replaces the
    whole backlog window — the default query and any --since alike — the
    operator's query, the operator's risk)."""
    if query is not None:
        return query
    return f"in:inbox after:{int(cutoff_dt.timestamp())}"


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
        help=(
            "ISO date (YYYY-MM-DD): score mail from that day onward (the "
            "named day is INCLUDED — the cutoff is UTC midnight of it); "
            "default: 90 days ago (now - 90d, time kept)"
        ),
    ),
    limit: int = typer.Option(
        100, "--limit", min=1, help="Max messages to score (default 100)"
    ),
    query: str | None = typer.Option(
        None,
        "--query",
        help=(
            "Gmail search query — REPLACES the backlog window entirely "
            "(the default in:inbox after:<epoch> query and --since alike)"
        ),
    ),
    db: str | None = typer.Option(
        None,
        "--db",
        help=(
            "State DB path (overrides [store] db_path; ~ expanded, "
            "relative anchors to the CWD; the opt-in demo redirects it "
            "to a tmp path)"
        ),
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
    # ONE injectable clock read per run (fold-4 item 2): the default
    # window is now - 90 days WITH the time kept (the documented intent);
    # --since → UTC midnight of the named day (the day is INCLUDED —
    # `after:` is strictly-newer than that midnight)
    now = _now()
    if since is not None:
        # strict ISO (YYYY-MM-DD) — strptime, not date.fromisoformat
        # (3.11+ fromisoformat is lenient and would accept "20261001")
        try:
            since_date = datetime.strptime(since, "%Y-%m-%d").date()
        except ValueError:
            typer.echo(
                f"shadow run failed: --since {since!r} is not an ISO date (YYYY-MM-DD)",
                err=True,
            )
            raise typer.Exit(code=2) from None
        cutoff_dt = _backlog_cutoff(since_date, now)  # UTC midnight of that day
    else:
        cutoff_dt = _backlog_cutoff(None, now)  # now - 90d, time kept

    cfg = _load_config(config)
    # D3: the effective state DB path (--db overrides [store] db_path)
    db_path = _resolve_db_path(cfg, db, "shadow run")
    db_origin = "--db PATH" if db is not None else "[store] db_path"
    # S3: the state DB path must not live inside the checkpoint dir or
    # the corpus store (usage refusal exit 2, before any work)
    _check_store_db_path(db_path, cfg, "shadow run", db_origin)
    # auth gate FIRST (exit 4 precedent): a not-signed-in run must not
    # pay the corpus-scale work below
    try:
        client = _shadow_client(cfg)
    except (AuthError, TokenStoreError) as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None

    # S5: the state DB is validated FAIL-FAST (open + ownership +
    # version) BEFORE any network work — a foreign/corrupt/foreign-
    # version DB is refused before we pay for list_labels/list_messages.
    # The connection is closed immediately (validation only); the write
    # re-opens it at the store step (idempotent create-all-if-missing).
    _validate_store_db(db_path)

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
    q = _backlog_query(cutoff_dt, query)
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
    run_ts = now  # the SAME injectable clock read (fold-4 item 2 — ONE read)
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
    # invariant #5). The DB was validated fail-fast before the network
    # work (S5); it is opened here for the single write point.
    store_db = AppDB(db_path)
    try:
        store_db.open()
        store_db.record_run(run_id, run_ts, all_rows)
    except (AppDBError, OSError) as e:
        typer.echo(f"shadow run failed: state DB write failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    finally:
        store_db.close()

    if as_json:
        # fold-4 item 3 + fold-5 A1: the window fields are derived from
        # the SAME cutoff_dt that built the query — emitted whenever NO
        # --query override is in effect (--since windows included, since
        # the cutoff_dt built the query in every non-override case).
        # Under --query the cutoff_epoch/cutoff fields would describe a
        # window the query never used, so they are OMITTED (no phantom
        # window).
        payload = {
            "run_id": run_id,
            "run_ts": run_ts.isoformat(),
            "query": q,
            "limit": limit,
            "scored": len(fetched),
            "skipped": skipped,
            "proposed": proposed,
            "already_present": already_present,
            "not_applied": not_applied,
            "rows": json_rows,
        }
        if query is None:
            # no --query override (default OR --since window): the cutoff
            # fields describe the SAME cutoff_dt that built the query
            # (epoch + its ISO datetime — the payload's legacy date-only
            # `since` KEY is retired, it diverged from the shipped
            # default, which keeps the time; the --since FLAG is not)
            payload["cutoff_epoch"] = int(cutoff_dt.timestamp())
            payload["cutoff"] = cutoff_dt.isoformat()
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


def _tier_name_for_band(
    band: str | None, *, present: bool, labels: LabelsConfig
) -> str | None:
    """The CONFIGURED Tier Label NAME for a band verdict (D2): the
    band→tier DECISION lives in tiers.tier_for_band (the single home);
    this maps the DEFAULT tier constant to the CONFIGURED name
    (``[labels] high`` / ``[labels] med``) — the config drives the
    report's tier names, so a custom ``[labels]`` spelling is honored,
    not silently ignored (refuse-never-drop). Skip-present earns no
    tier (invariant #2). The name→id resolution SHIPPED in #24
    (gmail_client.ensure_label + applications.plan_message_write taking
    the resolved map); the daemon (#25) / apply (#26) call it at
    startup (the contract is recorded in the tiers module docstring)."""
    from .tiers import TIER_HIGH, TIER_MED, tier_for_band

    default_tier = tier_for_band(band, present=present)
    if default_tier == TIER_HIGH:
        return labels.high
    if default_tier == TIER_MED:
        return labels.med
    return None


@dataclass(frozen=True, slots=True)
class ReportRow:
    """One would-apply verdict row in the shadow report diff (ids +
    labels + scores + tiers ONLY — no subjects, no bodies, invariant
    #5). ``outcome`` is the diff verdict (``added`` / ``already-present``
    / ``not-applied``); ``tier`` is the CONFIGURED Tier Label name (the
    ``[labels]`` high/med — the single home is ``_tier_name_for_band``,
    which maps the ``tiers.tier_for_band`` decision to the configured
    name; ``None`` for skip-present and no-band)."""

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
    db: str | None = typer.Option(
        None,
        "--db",
        help=(
            "State DB path (overrides [store] db_path; ~ expanded, "
            "relative anchors to the CWD; the opt-in demo redirects it "
            "to a tmp path)"
        ),
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
    the header + the ``basis`` payload field). Per-run hash PERSISTENCE
    is deferred to post-merge calibration; the shipped interim mechanism
    is the `status` basis line (#27), which names the current
    bands/graduation content hashes and cross-checks the pair. Re-run
    ``shadow run`` after any retrain/recalibration to refresh the
    basis.
    """
    from .appdb import AppDB, AppDBError, RunNotFoundError
    from .auth import AuthError
    from .gmail_errors import GmailAuthError, GmailError
    from .labels import normalize_label_names
    from .tiers import BAND_HIGH, BAND_MED
    from .tokens import TokenStoreError

    # usage refusal up front (exit 2): an empty --run would degenerate
    # to a lookup on "" (the latest-run default is the empty-ABSENT
    # case, not an empty-string run id)
    if run is not None and not run.strip():
        typer.echo("shadow report failed: --run must be non-empty", err=True)
        raise typer.Exit(code=2) from None

    cfg = _load_config(config)
    # D3: the effective state DB path (--db overrides [store] db_path)
    db_path = _resolve_db_path(cfg, db, "shadow report")
    db_origin = "--db PATH" if db is not None else "[store] db_path"
    # S3: the state DB path must not live inside the checkpoint dir or
    # the corpus store (usage refusal exit 2, before any work)
    _check_store_db_path(db_path, cfg, "shadow report", db_origin)
    # auth gate FIRST (exit 4 precedent): the report re-fetches the
    # actual labels via the read path
    try:
        client = _shadow_client(cfg)
    except (AuthError, TokenStoreError) as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None

    # the state DB (create-all-if-missing — a report on an empty DB is
    # a named "no runs" refusal, not a crash). The open (ownership +
    # version) is BEFORE the network work (S5 fail-fast).
    store_db = AppDB(db_path)
    try:
        store_db.open()
        run_id = run if run is not None else store_db.latest_run()
        if run_id is None:
            typer.echo(
                "shadow report failed: no shadow runs recorded \u2014 run "
                "`shadow run` first",
                err=True,
            )
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        # an unknown --run is refused by load_run's NAMED not-found
        # refusal (a separate run_exists pre-check would be a redundant
        # second lookup — the reader is the authority)
        run_ts, rows = store_db.load_run(run_id)
    except RunNotFoundError as e:
        # a named NOT-FOUND refusal (cause-naming doctrine — not wrapped
        # in "state DB read failed", which would misname the cause)
        typer.echo(f"shadow report failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except (AppDBError, OSError) as e:
        typer.echo(f"shadow report failed: state DB read failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    finally:
        store_db.close()

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
        # the band→tier DECISION lives in tiers.tier_for_band (the single
        # home); the CONFIGURED name comes from [labels] (D2 — a custom
        # spelling is honored, not silently ignored): an already-present
        # label earns NO Tier Label (present=True → None)
        tier = _tier_name_for_band(row.band, present=is_present, labels=cfg.labels)
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
    # re-verify them (per-run hash persistence is deferred to post-merge
    # calibration; the shipped interim mechanism is the `status` basis line
    # (#27), which names the current hashes and cross-checks the pair). A
    # retrain/recalibration moves the basis; re-run shadow run to refresh.
    basis = {
        "note": (
            "verdicts bound to the bands/graduation artifacts as of the "
            "run — not re-verified; re-run shadow run after any "
            "retrain/recalibration"
        ),
    }

    if as_json:
        # disposition 8: the rows serialize via dataclasses.asdict — the
        # five-field contract (label/probability/band/outcome/tier) is
        # now in LOCKSTEP with the ReportRow dataclass (a field added to
        # the dataclass is automatically in the payload, not a silent
        # drift)
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
                    "rows": [dataclasses.asdict(r) for r in per_message[message_id]],
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


# -- daemon (ticket #25): poll, decide, write ---------------------------------


def _daemon_client(config: Config) -> GmailClient:
    """Client factory seam: tests inject FakeGmailClient here (the
    offline spine, invariant #9). The daemon is the WRITE path — the
    full Protocol surface (history polling, apply_labels, ensure_label)."""
    from .auth import credentials as build_credentials
    from .gmail_client import GmailRestClient

    return GmailRestClient(build_credentials(config))


def _daemon_model_factory(checkpoint_dir: Path, vocabulary: tuple[str, ...]) -> Model:
    """Model factory seam for ``daemon`` (the ``load_model`` factory
    shape): tests inject a FakeModel producer. The real implementation
    lazy-imports the ``train/`` recipe (torch — the optional extra), so
    the CLI module itself stays importable without torch."""
    from train.recipe import load_hf_model

    return load_hf_model(checkpoint_dir, vocabulary)


def _sleep_until_stopped(flag: threading.Event, seconds: float) -> None:
    """The interruptible poll sleep (AC: mid-sleep SIGTERM → immediate
    exit): ``Event.wait`` returns the MOMENT the signal handler sets the
    flag — no polling chunk, no real time.sleep in the loop's contract."""
    flag.wait(timeout=seconds)


_PreviousHandlers = dict[
    signal.Signals, Callable[[int, FrameType | None], Any] | int | None
]


def _install_shutdown_handlers(flag: threading.Event) -> _PreviousHandlers:
    """SIGTERM + SIGINT set the stop flag (the loop consults it between
    messages, between pages, and around the sleep; the current delta
    still runs to its commit point). Returns the previous dispositions
    for the finally-restore (a daemon run must not leak handlers)."""

    def handler(signum: int, frame: FrameType | None) -> None:
        flag.set()

    previous: _PreviousHandlers = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, handler)
    except BaseException:
        # a mid-install failure (a non-main thread, a platform
        # refusal) must not leave the earlier handler installed
        for sig, prior in previous.items():
            signal.signal(sig, prior)
        raise
    return previous


def _restore_shutdown_handlers(previous: _PreviousHandlers) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


@dataclass(frozen=True, slots=True)
class _ServeStartup:
    """The gates-first startup bundle BOTH the daemon (#25) and the
    ``apply`` command (#26) serve from — the ONE extraction of #25's
    startup chain (the design's "startup = ONE extraction": a second
    copy is the drift the shared helper exists to prevent). The config,
    the effective state DB path, the authenticated client, the
    sidecar-validated model, the ``ServingContext`` (bands / graduated /
    System Labels / id-bridge + the two name→id maps the plan consumes),
    and the corpus records (apply's AC5 provenance scan reads them; the
    daemon ignores them — the fingerprint was already verified at load).

    Every startup refusal raises ``typer.Exit`` with the PROJECT
    exit-code contract (2 usage · 4 auth/config incl. the tier-name
    collision · 5 artifact/store/wire failure); ``command`` names the
    calling command in every refusal message (the daemon's sealed
    "daemon failed:" strings ride ``command="daemon"``, byte-identical)."""

    cfg: Config
    db_path: Path
    client: GmailClient
    model: Model
    ctx: ServingContext
    # a TUPLE, not a list: a frozen dataclass carrying a mutable list
    # would let a caller mutate the "immutable" bundle's records in place
    # (the AC5 provenance scan reads them; the daemon ignores them)
    records: tuple[CorpusRecord, ...]


def _load_filters_snapshot(sidecar: Path) -> list[dict[str, object]] | None:
    """Read the offline filter-attribution snapshot for the daemon's
    correction arm (#27, invariant #6). A LOCAL file read — never a Gmail
    call, so the startup exact-call pins stand. THREE outcomes, mirroring
    analyze.py's ``filters=None`` posture (the #22 mechanism reused):
    - ``None`` — the sidecar is ABSENT or malformed: attribution is
      UNKNOWN, so an 'added' Correction must be withheld as 'unverifiable'
      rather than credited to the human (a missing snapshot is NOT an
      attestation that no filter exists — that fail-open is what F1
      removes);
    - ``[]`` — the sidecar is present and lists no filters: the operator's
      ATTESTATION that none exist (an 'added' is then 'human' signal);
    - ``[filter, ...]`` — the parsed snapshot.
    A malformed file is ``None`` (unknown), never ``[]`` (a corrupt file
    must not masquerade as an attested-empty one)."""
    if not sidecar.is_file():
        return None
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    if isinstance(raw, list) and all(isinstance(f, dict) for f in raw):
        return [dict(f) for f in raw]
    return None


def _serve_startup(
    *,
    command: str,
    config: str | None,
    bands: str,
    graduation: str,
    db: str | None,
    mint_labels: bool = True,
) -> _ServeStartup:
    """The daemon's gates-first startup, extracted VERBATIM (ticket #26):
    identity gates run FIRST (the shadow-run shape: checkpoint, Graduation
    Table, Bands Artifact), then the three configured Tier Label names
    resolve ONCE via ensure_label (a System-Label name collision refuses
    AT MINT at exit 4, naming the [labels] key). The client and model
    factories are the module-level seams (``_daemon_client`` /
    ``_daemon_model_factory``) so a test's monkeypatch reaches BOTH the
    daemon and the apply path — the seams are shared, not forked. The
    daemon command and the apply command are the only callers; the
    daemon's poll loop and the apply batch both consume the returned
    bundle unchanged.

    ``mint_labels`` is the preview's read-only discipline (ticket #26
    AC1): the daemon and the ``apply --apply`` backfill pass True (the
    get-or-create ensure_label, the daemon's sealed behavior — a
    System-Label name collision still refuses at exit 4). A dry PREVIEW
    passes False: the Tier Label names resolve by LOOKUP ONLY (the
    already-listed labels), and a name the mailbox does not carry hits
    the MINT GUARD — refused NAMED (exit 5), never silently created (a
    preview performs zero wire writes, so it cannot mint its own Tier
    Labels; the operator runs ``apply --apply``/``daemon`` or creates
    them first). The collision case is still a config error (exit 4, the
    same rename diagnosis ensure_label gives)."""
    from .auth import AuthError
    from .corpus_record import corpus_fingerprint
    from .corpus_store import CorpusError, CorpusStore
    from .daemon import ServingContext
    from .dataset_build import SplitManifest
    from .gmail_client import find_user_label, system_label_namesake
    from .gmail_errors import GmailAuthError, GmailError
    from .model import (
        MANIFEST_FILENAME,
        THRESHOLDS_SOURCE_NONE,
        CheckpointRefusedError,
        ExpectedIdentity,
        load_model,
    )
    from .tiers import TIER_APPLIED, TIER_HIGH, TIER_MED
    from .tokens import TokenStoreError

    # usage refusals up front (exit 2, before any work — the shadow
    # precedent): an empty --bands/--graduation would degenerate to
    # Path("") and fail awkwardly mid-run
    for flag, target in (("--bands", bands), ("--graduation", graduation)):
        if not target.strip():
            typer.echo(f"{command} failed: {flag} PATH must be non-empty", err=True)
            raise typer.Exit(code=2) from None

    # PROJECT exit-code contract (cli epilog + every command suite):
    # config load failure is 4 (auth/config unusable) — _load_config's
    # own mapping; the serve startup does not fork the contract
    cfg = _load_config(config)
    db_path = _resolve_db_path(cfg, db, command)
    db_origin = "--db PATH" if db is not None else "[store] db_path"
    _check_store_db_path(db_path, cfg, command, db_origin)
    # auth gate FIRST (exit 4 precedent): a not-signed-in run must not
    # pay the corpus-scale identity work below
    try:
        client = _daemon_client(cfg)
    except (AuthError, TokenStoreError) as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None

    # S5: the state DB validated FAIL-FAST before any network work
    _validate_store_db(db_path, command=command)

    store = CorpusStore(cfg.corpus.dir)
    if not store.root.is_dir():
        typer.echo(
            f"{command} failed: corpus dir {store.root} is missing or not a "
            "directory \u2014 run dataset export/import-gyb first or fix "
            "[corpus] dir",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        records = list(store.iter_records())
    except (CorpusError, OSError, UnicodeDecodeError):
        typer.echo(f"{command} failed: records unreadable", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if not records:
        typer.echo(
            f"{command} failed: corpus is empty \u2014 run dataset "
            "export/import-gyb first",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the checkpoint (identity gate, the shadow shape): the materialized
    # manifest is the split's authority (invariant #7)
    checkpoint_dir = cfg.model.checkpoint_dir
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        typer.echo(
            f"{command} failed: checkpoint {checkpoint_dir} has no "
            f"{MANIFEST_FILENAME} \u2014 no verifiable split identity "
            "(run train first)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        manifest = SplitManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        typer.echo(
            f"{command} failed: checkpoint manifest unreadable or tampered: {e}",
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
        model = load_model(checkpoint_dir, expected, _daemon_model_factory)
    except CheckpointRefusedError as e:
        typer.echo(
            f"{command} failed: checkpoint refused ({', '.join(e.fields)}) "
            "\u2014 no silent fallback; rebuild the checkpoint",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except ImportError as e:
        typer.echo(
            f"{command} failed: the model recipe is unavailable ({e}) "
            "\u2014 install the optional extra: `uv sync --extra train` "
            "(if it is already installed, the recipe import itself is "
            "broken \u2014 report it)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except (ValueError, OSError, RuntimeError) as e:
        typer.echo(f"{command} failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    if model.vocabulary != manifest.vocabulary:
        typer.echo(
            f"{command} failed: the model's vocabulary does not match the "
            "manifest's \u2014 refusing to score against a lying column "
            "order",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the Bands Artifact + the Graduation Table (identity gates, the
    # shadow shape — cause naming per #22's doctrine)
    from .graduation import GraduationRefusedError, GraduationTable, verify_graduation
    from .thresholds import BandsArtifact, BandsRefusedError, verify_bands

    bands_path = Path(bands)
    if not bands_path.is_file():
        typer.echo(
            f"{command} failed: no bands artifact at {bands_path} \u2014 "
            "run `eval --bands` first",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        bands_text = bands_path.read_text(encoding="utf-8")
    except OSError:
        typer.echo(
            f"{command} failed: bands artifact at {bands_path} is unreadable "
            "(I/O failure)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except UnicodeDecodeError:
        typer.echo(
            f"{command} failed: bands artifact at {bands_path} is not valid "
            "UTF-8 \u2014 it does not parse as a Bands Artifact",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        artifact = BandsArtifact.from_json(bands_text)
    except ValueError as e:
        typer.echo(
            f"{command} failed: bands artifact at {bands_path} does not parse "
            f"as a Bands Artifact: {e}",
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
        typer.echo(
            f"{command} failed: bands artifact refused ({', '.join(e.fields)}) "
            "\u2014 it disagrees with this checkpoint/eval; re-run "
            "`eval --bands` against this checkpoint",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except ValueError as e:
        typer.echo(f"{command} failed: bands verification failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    graduation_path = Path(graduation)
    if not graduation_path.is_file():
        typer.echo(
            f"{command} failed: no graduation table at {graduation_path} "
            "\u2014 run `eval --graduation --confirm-graduation` first",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        graduation_text = graduation_path.read_text(encoding="utf-8")
    except OSError:
        typer.echo(
            f"{command} failed: graduation table at {graduation_path} is "
            "unreadable (I/O failure)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except UnicodeDecodeError:
        typer.echo(
            f"{command} failed: graduation table at {graduation_path} is not "
            "valid UTF-8 \u2014 it does not parse as a Graduation Table",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        table = GraduationTable.from_json(graduation_text)
    except ValueError as e:
        typer.echo(
            f"{command} failed: graduation table at {graduation_path} does not "
            f"parse as a Graduation Table: {e}",
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
            f"{command} failed: graduation table refused ({', '.join(e.fields)}) "
            "\u2014 it disagrees with this checkpoint/eval/bands; re-run "
            "`eval --graduation` against this checkpoint",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except ValueError as e:
        typer.echo(f"{command} failed: graduation verification failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the label bridge (presence + System Labels — the read path)
    try:
        labels = client.list_labels()
    except GmailAuthError as e:
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailError as e:
        typer.echo(f"{command} failed: label list failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    id_to_name = {li.id: li.name for li in labels}
    system_names = frozenset(li.name for li in labels if li.type == "system")
    graduated = frozenset(entry.label for entry in table.labels if entry.graduated)
    bands_map = {entry.label: (entry.high, entry.med) for entry in artifact.labels}

    # STARTUP tier resolution: the three configured [labels] names →
    # Gmail label ids ONCE (ensure_label: get-or-create; the Q1 ruling
    # keeps the signature and pays one list round-trip per name). The
    # collision refusal is a BARE ValueError (the C0 Protocol contract —
    # NOT the gmail_errors taxonomy): it maps to exit 4 (config error)
    # and names the offending [labels] KEY (the #24 carryover — a bare
    # ValueError would otherwise land as exit-1 'a bug').
    tier_constants = {"applied": TIER_APPLIED, "high": TIER_HIGH, "med": TIER_MED}
    tier_label_ids: dict[str, str] = {}
    for key, name in (
        ("applied", cfg.labels.applied),
        ("high", cfg.labels.high),
        ("med", cfg.labels.med),
    ):
        if not mint_labels:
            # the preview's MINT GUARD (AC1): a dry run never creates a
            # label, so the Tier Label names resolve by LOOKUP ONLY — via
            # the SAME read-only resolvers ensure_label rides (gmail_client.
            # find_user_label / system_label_namesake), NOT a forked copy
            # of its two-pass logic (the drift this seam exists to stop).
            # An exact-case USER label of the name is used as-is; a
            # System-Label name-sake is the config error it always was
            # (exit 4); an ABSENT exact-case name is refused NAMED (exit 5)
            # — the preview cannot mint the label it needs to compose a
            # faithful plan, so it says so instead of writing
            hit = find_user_label(labels, name)
            if hit is not None:
                tier_label_ids[tier_constants[key]] = hit
                continue
            if system_label_namesake(labels, name) is not None:
                typer.echo(
                    f"{command} failed: [labels] {key} = {name!r} collides "
                    "with a System Label name in the mailbox (System Labels "
                    f"are never write targets) \u2014 rename [labels] {key} "
                    "and restart",
                    err=True,
                )
                raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
            # honest wording: an exact-case USER label is absent — a
            # case-DIFFERING user label may well exist (the match is
            # exact-case), so "is not present in the mailbox" would be a
            # lie; say what is actually missing
            typer.echo(
                f"{command} failed: [labels] {key} = {name!r} has no "
                "exact-case USER label in the mailbox \u2014 a preview does "
                "not create Tier Labels; run `apply --apply` (or `daemon`) "
                "to create it, or create it yourself",
                err=True,
            )
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
        try:
            tier_label_ids[tier_constants[key]] = client.ensure_label(name)
        except ValueError as e:
            typer.echo(
                f"{command} failed: [labels] {key} = {name!r} refused: {e} "
                f"\u2014 rename [labels] {key} and restart",
                err=True,
            )
            raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
        except GmailAuthError as e:
            # ensure_label's own round-trips (list + create) are WIRE
            # calls: they take the SAME arms as every other client call
            # site (the review round's blocker — they escaped the
            # ValueError-only arm to exit 1); auth is human-gated (4)
            typer.echo(f"auth: {e}", err=True)
            raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
        except GmailError as e:
            # transient/permanent wire failure at startup resolution:
            # operation failure (5), named, never a traceback
            typer.echo(f"{command} failed: tier label resolution failed: {e}", err=True)
            raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    # the Managed Label name→id map for the would-apply names (the
    # bridge, user labels only, exact-case first hit wins — the plan's
    # _resolve_id fail-safes any name absent here)
    managed_label_ids: dict[str, str] = {}
    for li in labels:
        if (
            li.type == "user"
            and li.name in graduated
            and li.name not in managed_label_ids
        ):
            managed_label_ids[li.name] = li.id

    # the offline filter-attribution snapshot for the daemon's correction
    # arm (#27, invariant #6): a LOCAL file read (never a Gmail call — the
    # startup exact-call pins stand). _load_filters_snapshot distinguishes
    # UNKNOWN (None: absent or malformed — an 'added' is withheld as
    # 'unverifiable', NOT credited to the human) from ATTESTED-NONE ([]: the
    # operator ran the export and has no filters — an 'added' is 'human').
    # The old code collapsed absent/malformed into [], failing open exactly
    # as the bool F1 removed did, one level up. apply never reads it.
    filters_snapshot = _load_filters_snapshot(store.root / SIDECAR_FILTERS)
    if filters_snapshot is None and command == "daemon":
        typer.echo(
            f"daemon: NOTE no usable filters snapshot at "
            f"{store.root / SIDECAR_FILTERS} — correction attribution is "
            "UNKNOWN, so every 'added' Correction is withheld as "
            "'unverifiable' (not credited as training signal). Run "
            "`dataset export` against this store and RESTART the daemon to "
            "enable filter attribution (the snapshot is read at startup "
            "only).",
            err=True,
        )
    elif filters_snapshot is not None and command == "daemon":
        # a snapshot with one or more UNREADABLE filter entries (a present-
        # but-wrong-shaped action/addLabelIds) makes _attribution withhold
        # EVERY 'added' (authorship can't be ruled out) — name the entries at
        # startup so the operator is not left with a silent rising
        # 'unverifiable' and no cause (the #22 bridge-malformed posture:
        # name it, don't guess). Entries are named by 1-based INDEX + which
        # FIELD is unreadable (the strict reader's ValueError names
        # action/addLabelIds, NEVER a value — a from:/to: criterion is a mail
        # address and must not reach the log, invariant #5 adjacent); a
        # legitimately action-less filter reads [] and is NOT counted here.
        unreadable_entries: list[str] = []
        for idx, f in enumerate(filters_snapshot, start=1):
            try:
                filter_label_ids_strict(f)
            except ValueError as e:
                unreadable_entries.append(f"#{idx}: {e}")
        if unreadable_entries:
            n = len(unreadable_entries)
            noun = "entry" if n == 1 else "entries"
            verb = "has" if n == 1 else "have"
            typer.echo(
                f"daemon: NOTE {n} filter {noun} in the snapshot {verb} an "
                f"unreadable action/addLabelIds (1-based entry numbers, "
                f"entry #1 is the first filter in the snapshot: "
                f"{'; '.join(unreadable_entries)}) — authorship cannot be "
                "determined, so every 'added' Correction, for every label, "
                "is withheld until the snapshot is clean (an add a readable "
                "filter provably covers is still attributed 'filter_covered' "
                "and remains withheld). If the sidecar was hand-edited into a "
                "bad shape, re-run `dataset export` and RESTART the daemon "
                "(the snapshot is read at startup only); if the malformation "
                "came off the wire, export reproduces it — fix the filter in "
                "Gmail instead.",
                err=True,
            )

    ctx = ServingContext(
        bands=bands_map,
        graduated=graduated,
        system_labels=system_names,
        id_to_name=id_to_name,
        managed_label_ids=managed_label_ids,
        tier_label_ids=tier_label_ids,
        filters=filters_snapshot,
    )
    return _ServeStartup(
        cfg=cfg,
        db_path=db_path,
        client=client,
        model=model,
        ctx=ctx,
        records=tuple(records),
    )


@app.command("daemon")
def daemon(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    bands: str = typer.Option(
        ...,
        "--bands",
        help="Bands Artifact path (required — the calibrated Threshold Bands)",
    ),
    graduation: str = typer.Option(
        ...,
        "--graduation",
        help="Graduation Table path (required — only GRADUATED labels apply)",
    ),
    db: str | None = typer.Option(
        None,
        "--db",
        help="State DB path (overrides [store] db_path; ~ expanded)",
    ),
) -> None:
    """Poll the Gmail History API (~[gmail] history_poll_seconds),
    score arriving mail, and APPLY only graduated labels — additive
    only, skip-present, Tier Labels on every Application (ticket #25;
    GLOSSARY: Durable Handling).

    The cursor is persisted ONLY after the delta's events are durably
    handled (invariant #3): the Gmail writes land first, then ONE
    transaction carries the delta's Application rows AND the cursor
    advance — COMMIT = handled; a kill before the commit replays the
    delta and skip-present makes the retry idempotent (daemon.py).
    Identity gates run FIRST (the shadow-run shape: checkpoint,
    Graduation Table, Bands Artifact), then the three configured Tier
    Label names resolve ONCE via ensure_label (a System-Label name
    collision refuses AT MINT at exit 4, naming the [labels] key: an
    existing exact-case USER label of that name is reused first). SIGTERM/
    SIGINT shut down gracefully: the current delta completes to its
    commit point, exit 0 (the handlers wrap the LOOP only: Ctrl-C
    during the startup gates/corpus read keeps the default
    disposition — nothing is open yet, so nothing needs unwinding).
    Bodies never enter the DB or any log (invariant #5). Exit
    codes: 0 clean stop · 2 usage · 4 auth/config (incl. the tier-name
    collision) · 5 artifact/store failure, startup tier-resolution wire
    failure, or a loop I/O failure."""
    import logging
    import sys

    from .appdb import AppDB, AppDBError
    from .daemon import daemon_loop
    from .gmail_errors import GmailAuthError

    # the gates-first startup, SHARED with apply (#26): the identity
    # gates, the client/model seams, the ensure_label tier resolution,
    # and the ServingContext all live in _serve_startup — the daemon's
    # suite rides it VERBATIM (the extraction is behavior-preserving; a
    # test edit would mean the extraction is wrong)
    startup = _serve_startup(
        command="daemon", config=config, bands=bands, graduation=graduation, db=db
    )
    cfg = startup.cfg
    client = startup.client
    model = startup.model
    ctx = startup.ctx
    db_path = startup.db_path
    graduated = ctx.graduated
    typer.echo(
        f"daemon: serving {len(graduated)} graduated label(s); poll "
        f"{cfg.gmail.history_poll_seconds}s; db {db_path}",
        err=True,
    )
    stop_flag = threading.Event()
    previous_handlers = _install_shutdown_handlers(stop_flag)
    # OPERATOR VISIBILITY (F2 MINOR 7): the per-delta counter line is
    # INFO, and an unconfigured root logger shows WARNING+ only
    # (logging.lastResort) — the daemon installs a stderr handler for
    # ITS logger so the counter line reaches the operator; the finally
    # removes it (a daemon run must not leak handlers, the signal
    # precedent above). caplog pins keep working: propagation to the
    # root is untouched.
    store_db = AppDB(db_path)
    daemon_log = logging.getLogger("gmail_labeler.daemon")
    log_handler = logging.StreamHandler(sys.stderr)
    log_handler.setLevel(logging.INFO)
    previous_log_level = daemon_log.level
    # propagation stays ON (the caplog pins ride it); today no root
    # handler exists, so nothing double-prints (verified in review)
    daemon_log.setLevel(logging.INFO)
    daemon_log.addHandler(log_handler)
    try:
        store_db.open()
        daemon_loop(
            client=client,
            db=store_db,
            model=model,
            ctx=ctx,
            poll_seconds=cfg.gmail.history_poll_seconds,
            clock=_now,
            sleep=lambda seconds: _sleep_until_stopped(stop_flag, seconds),
            should_stop=stop_flag.is_set,
        )
        typer.echo("daemon: stopped", err=True)
    except GmailAuthError as e:
        # the loop's propagating human-gated arm (exit 4, never a
        # traceback; the cursor stands where it was — the loop holds it)
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except AppDBError as e:
        # the store refusal family (a tampered cursor row, an I/O
        # failure at the serve-time open): operation failure, named
        typer.echo(f"daemon failed: state DB: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except OSError as e:
        # a bare I/O failure from anywhere in the loop (not necessarily
        # the state DB): the cause is named neutrally — a "state DB"
        # diagnosis over a network fault would be a lie (review finding)
        typer.echo(f"daemon failed: I/O failure: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    finally:
        daemon_log.removeHandler(log_handler)
        daemon_log.setLevel(previous_log_level)
        _restore_shutdown_handlers(previous_handlers)
        store_db.close()


@app.command("apply")
def apply_cmd(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    bands: str = typer.Option(
        ...,
        "--bands",
        help="Bands Artifact path (required — the calibrated Threshold Bands)",
    ),
    graduation: str = typer.Option(
        ...,
        "--graduation",
        help="Graduation Table path (required — only GRADUATED labels apply)",
    ),
    since: str = typer.Option(
        ...,
        "--since",
        help=(
            "ISO date (YYYY-MM-DD): the Backfill boundary — process mail "
            "from that day onward (the named day is INCLUDED — the cutoff "
            "is UTC midnight of it). REQUIRED: the only bulk path never "
            "guesses its own window"
        ),
    ),
    apply_flag: bool = typer.Option(
        False,
        "--apply",
        help=(
            "Perform the backfill (write graduated labels + record "
            "Applications). WITHOUT it: a dry preview that scores the "
            "window and writes NOTHING"
        ),
    ),
    db: str | None = typer.Option(
        None,
        "--db",
        help="State DB path (overrides [store] db_path; ~ expanded)",
    ),
) -> None:
    """Preview or backfill graduated labels over HISTORICAL mail since
    ``--since`` — the ONLY bulk path, always an explicit human decision
    (ticket #26; GLOSSARY: Application, Backfill).

    Without ``--apply`` this is a dry preview: it scores the window and
    reports the would-apply lines (id + label names + band/tier + scores)
    and performs ZERO Gmail writes (AC1). With ``--apply`` it backfills:
    the SAME decision core the daemon runs (``applications.
    plan_message_write`` — one core, no second implementation, AC2), the
    write-first ordering identical to the daemon, and Applications
    recorded identically to daemon-applied ones (AC3). skip-present makes
    a re-run idempotent (AC4): a label that already landed reads present,
    composes an empty plan, earns no second write and no duplicate row.

    The wire query is ``after:<epoch-seconds>`` ONLY — deliberately
    WITHOUT ``in:inbox`` (backfill targets historical mail, which is
    mostly archived); the epoch form is the unambiguous one (a calendar
    date in an API query is midnight PST, not the UTC boundary). The
    cutoff reuses the shadow-run rule (``_backlog_cutoff``: UTC midnight
    of the named day). apply NEVER reads or writes the daemon's history
    cursor (invariant #3 is the daemon's alone); a mid-run SIGTERM
    completes the current batch to its commit point (the shared shutdown
    handler), and a transient write failure holds NOTHING — the message
    stays un-labeled and a re-run picks it up.

    Records in the window lacking a Date Provenance stamp (pre-#17
    corpora) surface ONE counted warning at the boundary (AC5): the
    date-based window is only as trustworthy as the dates. Bodies never
    enter the DB or any log (invariant #5). Exit codes: 0 clean · 2
    usage · 4 auth/config (incl. the tier-name collision) · 5 artifact/
    store failure, a preview whose Tier Labels are absent, or a model /
    wire failure."""
    import logging
    import sys

    from .appdb import AppDB, AppDBError
    from .apply import run_apply
    from .corpus_record import SERVER_AUTHORITATIVE_PROVENANCES
    from .gmail_errors import GmailAuthError, GmailError

    # --since is REQUIRED (the only bulk path never guesses its window);
    # typer rejects an absent option itself. Strict ISO (strptime, not
    # the lenient 3.11+ fromisoformat — the shadow-run precedent): a
    # non-ISO value is a usage error (exit 2)
    try:
        since_date = datetime.strptime(since, "%Y-%m-%d").date()
    except ValueError:
        typer.echo(
            f"apply failed: --since {since!r} is not an ISO date (YYYY-MM-DD)",
            err=True,
        )
        raise typer.Exit(code=2) from None
    now = _now()
    cutoff_dt = _backlog_cutoff(since_date, now)  # UTC midnight of the named day
    # the backfill query: after:<epoch-seconds> ONLY — no in:inbox (the
    # window is historical mail, mostly archived; the epoch form is the
    # unambiguous one per #23's primary-source doctrine)
    query = f"after:{int(cutoff_dt.timestamp())}"

    # the gates-first startup, SHARED with the daemon (_serve_startup):
    # a dry preview FORBIDS minting Tier Labels (mint_labels=False — the
    # AC1 zero-write discipline: a preview that needs a label it lacks
    # fails named, never silently creates); the backfill resolves them via
    # ensure_label exactly as the daemon does
    startup = _serve_startup(
        command="apply",
        config=config,
        bands=bands,
        graduation=graduation,
        db=db,
        mint_labels=apply_flag,
    )
    client = startup.client
    model = startup.model
    ctx = startup.ctx
    db_path = startup.db_path

    # AC5: the --since boundary is a DATE; a corpus record whose date has
    # no server-authoritative provenance stamp (pre-#17 lines read as
    # "unknown") cannot be trusted to sit on the right side of it. ONE
    # counted warning at the boundary (never per-message spam)
    unprovenanced = sum(
        1
        for rec in startup.records
        if rec.date >= cutoff_dt
        and rec.date_provenance not in SERVER_AUTHORITATIVE_PROVENANCES
    )
    if unprovenanced:
        typer.echo(
            f"apply: WARNING {unprovenanced} corpus record(s) dated at or "
            f"after the --since boundary ({cutoff_dt.date()}) lack a "
            "server-authoritative Date Provenance stamp — the date-based "
            "window is only as trustworthy as those dates (GLOSSARY: Date "
            "Provenance)",
            err=True,
        )

    typer.echo(
        f"apply: {'backfill' if apply_flag else 'preview'} query={query!r} "
        f"serving {len(ctx.graduated)} graduated label(s); db {db_path}",
        err=True,
    )
    stop_flag = threading.Event()
    previous_handlers = _install_shutdown_handlers(stop_flag)
    # the preview lines and per-batch counters are INFO on the apply
    # logger; install a stderr handler so they reach the operator (the
    # daemon's F2 MINOR 7 precedent), removed in the finally
    store_db = AppDB(db_path)
    apply_log = logging.getLogger("gmail_labeler.apply")
    log_handler = logging.StreamHandler(sys.stderr)
    log_handler.setLevel(logging.INFO)
    previous_log_level = apply_log.level
    apply_log.setLevel(logging.INFO)
    apply_log.addHandler(log_handler)
    try:
        store_db.open()
        outcome = run_apply(
            client=client,
            db=store_db,
            model=model,
            ctx=ctx,
            query=query,
            clock=_now,
            should_stop=stop_flag.is_set,
            dry_run=not apply_flag,
        )
    except GmailAuthError as e:
        # the propagating human-gated arm (exit 4, never a traceback)
        typer.echo(f"auth: {e}", err=True)
        raise typer.Exit(code=_AUTH_ERROR_EXIT) from None
    except GmailError as e:
        # a page-read failure: apply has no cursor to resume from, so the
        # honest posture is a loud stop (a re-run re-pages, skip-present
        # idempotent)
        typer.echo(f"apply failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except AppDBError as e:
        typer.echo(f"apply failed: state DB: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    except (ValueError, OSError, RuntimeError) as e:
        # a model shape defect or a model-raised failure: the cause is
        # named by CLASS only (a model error may quote the scored TEXT,
        # which carries the body — invariant #5)
        typer.echo(
            f"apply failed: prediction/processing failed ({type(e).__name__})",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    finally:
        apply_log.removeHandler(log_handler)
        apply_log.setLevel(previous_log_level)
        _restore_shutdown_handlers(previous_handlers)
        store_db.close()
    typer.echo(
        f"apply: {'backfill' if apply_flag else 'preview'} done "
        f"batches={outcome.batches} messages={outcome.messages} "
        f"written={outcome.written} skipped={outcome.skipped} "
        f"warnings={outcome.warnings} rows={outcome.rows}",
        err=True,
    )


@corrections_app.command("export")
def corrections_export(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    db: str | None = typer.Option(
        None,
        "--db",
        help="State DB path (overrides [store] db_path; ~ expanded)",
    ),
    out: str = typer.Option("-", "--out", help="Output file path (default: stdout)"),
    fmt: str = typer.Option("json", "--format", help="Export format (json)"),
) -> None:
    """Export the recorded Corrections as future TRAINING SIGNAL: a JSON
    list of ``{message_id, label, kind, attribution}`` (ticket #27;
    GLOSSARY: Correction). This is a faithful RAW dump — every row is
    written with its attribution so the artifact is auditable (you can see
    WHICH adds were withheld and why); the actual signal-filtering is the
    builder's correction-folding seam's single job (it drops every
    non-'human' 'added' before the corpus ever sees it). Reads the LOCAL
    state DB only — no Gmail call — and NEVER retrains: this writes an
    artifact; folding it into the corpus is a composition point a human
    opts into, and running a retrain is a SEPARATE human decision (the
    AC's no-auto-retrain gate). NO
    bodies (invariant #5): the export carries ids/labels/kinds/attributions
    only. Exit contract 0/2/4/5."""
    from .appdb import AppDB, AppDBError

    if fmt != "json":
        typer.echo(
            f"corrections export failed: --format {fmt!r} unsupported (only json)",
            err=True,
        )
        raise typer.Exit(code=2) from None

    # NO auth gate, NO client: the export is a LOCAL DB read (like status)
    cfg = _load_config(config)
    db_path = _resolve_db_path(cfg, db, "corrections export")
    db_origin = "--db PATH" if db is not None else "[store] db_path"
    _check_store_db_path(db_path, cfg, "corrections export", db_origin)
    # the export READS the store — refuse a typo'd --db rather than
    # silently create an empty DB and export [] with exit 0
    if not db_path.exists():
        typer.echo(
            f"corrections export failed: no state DB at {db_path} — the "
            "export reads the store, it does not create one (run the daemon "
            "first, or check --db)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    _validate_store_db(db_path, command="corrections export")

    store_db = AppDB(db_path)
    try:
        store_db.open()
        rows = store_db.all_corrections()
    except AppDBError as e:
        typer.echo(f"corrections export failed: state DB: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    finally:
        store_db.close()

    payload = [
        {
            "message_id": r.message_id,
            "label": r.label,
            "kind": r.kind,
            "attribution": r.attribution,
        }
        for r in rows
    ]
    text = json.dumps(payload, indent=2, sort_keys=True)
    if out == "-":
        typer.echo(text)
        return

    # --out is a WRITE target — guard it like `eval` guards its outputs.
    # The headline danger: --out naming the state DB would overwrite the
    # SQLite file with JSON and destroy the corrections + cursor (only a
    # full resync recovers). Refuse that, plus the identity-file roots and
    # an existing directory, BEFORE writing. ~ is expanded here so a
    # `--out ~/c.json` names the same file the guards compare and the write
    # lands (F2 item 12: parity with --db, which expands via _resolve_db_path).
    out = str(Path(out).expanduser())
    out_path = Path(out).resolve()
    if out_path == db_path.resolve():
        typer.echo(
            f"corrections export failed: --out {out!r} is the state DB path "
            f"— exporting there would DESTROY the corrections + cursor "
            f"(choose a different file)",
            err=True,
        )
        raise typer.Exit(code=2) from None
    for root, what in (
        (cfg.model.checkpoint_dir.resolve(), "the checkpoint directory"),
        (Path(cfg.corpus.dir).resolve(), "the corpus store"),
    ):
        if out_path == root or root in out_path.parents:
            typer.echo(
                f"corrections export failed: --out {out!r} lives inside "
                f"{what} ({root}) — refusing to write there",
                err=True,
            )
            raise typer.Exit(code=2) from None
    if out_path.is_dir():
        typer.echo(
            f"corrections export failed: --out {out!r} names an existing "
            f"directory — the target is a file",
            err=True,
        )
        raise typer.Exit(code=2) from None
    # the SAME 0600 + mkstemp/fsync/os.replace atomic write + clobber NOTICE
    # `eval` uses (F2 item 6: the claim this guarded "like eval" was false
    # while it was a bare 0664 write_text beside a 0600 store)
    try:
        _write_out_report(out, text + "\n", label="corrections export")
    except OSError as e:
        # an unwritable target is an operation failure (5), NOT the exit-1
        # FileNotFoundError traceback the contract forbids (README: a `1`
        # is an unhandled crash)
        typer.echo(f"corrections export failed: cannot write {out!r}: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None


@app.command("status")
def status(
    config: str | None = typer.Option(None, "--config", help="Path to config.toml"),
    bands: str = typer.Option(
        ...,
        "--bands",
        help="Bands Artifact path (required — must be the pair this "
        "graduation table was measured against; a mismatch refuses exit 5)",
    ),
    graduation: str = typer.Option(
        ...,
        "--graduation",
        help="Graduation Table path (required — the graduated label set)",
    ),
    db: str | None = typer.Option(
        None,
        "--db",
        help="State DB path (overrides [store] db_path; ~ expanded)",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Answer "is it working?" with realized precision per graduated
    label — derived from the LOCAL Corrections + Applications store
    ALONE (ticket #27; GLOSSARY: Correction, Status).

    ``status`` performs NO Gmail call on ANY path: it reads the state DB
    and the two local artifacts (bands/graduation) and nothing else, so
    it answers on a sick or offline mailbox (an UNAUTHENTICATED config
    still works — there is no auth gate here, by design; the AC pins it
    with a no-client call). It NEVER CREATES a state DB (a missing --db is
    a refusal, not an empty store), but opening an EXISTING older-version
    one runs the standard atomic schema migration (a v1/v2/v3 file comes
    back at the current version — a write, the same transition every other
    command's open performs). Bodies never enter the output (invariant #5):
    labels, counts and rates only.

    Per graduated label: written (Application rows), removed (matching
    'removed' Corrections), precision = (written - removed)/written or
    'no data' when nothing is applied yet, and the positive-signal
    added Corrections split into the two withheld slices — 'filter-covered'
    (a Gmail filter verifiably would have applied it) and 'unverifiable'
    (a relevant filter makes authorship undecidable, so it is withheld not
    credited — invariant #6 / #22's exclusion-not-guessing) — BOTH
    EXCLUDED from the net human signal and disclosed. The basis line names
    the attribution vocabulary it counted under and the CURRENT
    bands/graduation artifact content hashes and states the honest v1
    caveats: rows can span artifact versions (per-run binding deferred
    to post-merge calibration), and precision assumes unseen-but-removed
    mail is rare (archive-equals-dismiss). Exit contract: 0 answered · 2 usage · 4
    config unusable · 5 store/artifact failure (3 is the project's
    reserved not-implemented code; status never emits it)."""
    from .appdb import CORRECTION_ATTRIBUTIONS, AppDB, AppDBError
    from .graduation import GraduationTable
    from .status import PrecisionInvariantError, compute_precision
    from .thresholds import BandsArtifact

    # The status caveat + attribution vocabulary, built ONCE and shared by
    # the text NOTE and the --json basis so the two surfaces can never
    # drift (F2 item 8). The vocabulary is RENDERED from CORRECTION_ATTRIBUTIONS
    # — not a literal — so a future vocabulary change shows up on the line
    # (the drift-pin asserts every constant value reaches the rendered text).
    _vocab = " / ".join(CORRECTION_ATTRIBUTIONS)
    _caveat = (
        "'written' counts Application ROWS (a re-earned label is a new row, "
        "so written can exceed the number of distinct messages) while "
        "'removed' is deduped per (message,label); a label removed then later "
        "re-added still counts as removed (the event log is historical truth, "
        "not reconciled state); a Gmail label RENAME desyncs the name-keyed "
        "join (orphaned corrections); rows may span artifact versions "
        "(per-run binding deferred to post-merge calibration); precision "
        "assumes unseen-but-removed mail is rare (archive-equals-dismiss)"
    )
    # the canonical THREE-cause sentence for 'unverifiable' (F2 item 14) —
    # mirrored verbatim in GLOSSARY / README / CHANGELOG
    _unverifiable_causes = (
        "no usable filters snapshot (absent or malformed); a relevant filter "
        "whose criteria cannot be simulated from the kept fields (outside "
        "from/to/subject, or malformed); or an add whose authorship is "
        "otherwise undecidable (its message fields were not fetched, or a "
        "filter's addLabelIds is unreadable — which is GLOBAL: every 'added' "
        "Correction, for every label, is withheld until the snapshot is clean)"
    )
    _attribution_note = (
        f"'added' counts are attributed under the corrections vocabulary "
        f"({_vocab} — a future vocabulary change would change these lines): "
        f"an 'added' is 'unverifiable' — withheld, not guessed — when "
        f"authorship cannot be determined: {_unverifiable_causes}. A "
        f"'filter-covered' add (a simulatable filter verifiably would have "
        f"applied it) is likewise withheld. Both slices are excluded from the "
        f"human signal, not guessed."
    )

    for flag, target in (("--bands", bands), ("--graduation", graduation)):
        if not target.strip():
            typer.echo(f"status failed: {flag} PATH must be non-empty", err=True)
            raise typer.Exit(code=2) from None
    # ~ is expanded for the artifact paths too (F2 item 12: parity with
    # --db, which expands via _resolve_db_path — a `--bands ~/b.json` must
    # name the file the operator meant, not a literal '~/' exit-5 miss)
    bands = str(Path(bands).expanduser())
    graduation = str(Path(graduation).expanduser())

    # NO auth gate, NO client, NO model, NO corpus: status is the DB-only
    # face (the AC — it must answer offline on a sick mailbox)
    cfg = _load_config(config)  # config load failure -> exit 4 (its own mapping)
    db_path = _resolve_db_path(cfg, db, "status")
    db_origin = "--db PATH" if db is not None else "[store] db_path"
    _check_store_db_path(db_path, cfg, "status", db_origin)
    # status READS the store — it must not silently CREATE one on a typo'd
    # --db (which would answer 'no data' with exit 0 and hide the mistake)
    if not db_path.exists():
        typer.echo(
            f"status failed: no state DB at {db_path} — status reads the "
            "store, it does not create one (run the daemon/apply first, or "
            "check --db)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    _validate_store_db(db_path, command="status")

    # the two artifacts (LOCAL reads; from_json self-validates the content
    # hash — a corrupt/foreign file refuses to exit 5, never a traceback)
    try:
        table = GraduationTable.from_json(Path(graduation).read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError) as e:
        typer.echo(
            f"status failed: graduation table {graduation!r} unreadable or "
            f"not a Graduation Table: {e}",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    try:
        artifact = BandsArtifact.from_json(Path(bands).read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError) as e:
        typer.echo(
            f"status failed: bands artifact {bands!r} unreadable or not a "
            f"Bands Artifact: {e}",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    # the bands/graduation PAIRING cross-check (F2 item 7): the Graduation
    # Table records the bands artifact it was MEASURED against
    # (bands_content_hash). A mismatched --bands/--graduation pair would
    # otherwise print a basis line claiming a pairing that was never
    # measured together — this finally makes --bands load-bearing (it was
    # only ever displayed). Refuse NAMED at exit 5.
    if artifact.content_hash != table.bands_content_hash:
        typer.echo(
            f"status failed: --bands {bands!r} (content_hash "
            f"{artifact.content_hash[:12]}) is NOT the bands artifact this "
            f"graduation table was measured against (bands_content_hash "
            f"{table.bands_content_hash[:12]}) — pass the matching pair "
            "(re-run `eval --bands`/`eval --graduation` together)",
            err=True,
        )
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    graduated = frozenset(entry.label for entry in table.labels if entry.graduated)

    store_db = AppDB(db_path)
    try:
        store_db.open()
        # READ ORDER MATTERS (F2 item 9, the torn-read): these are two
        # separate autocommit SELECTs, so a daemon delta can commit BETWEEN
        # them. Reading corrections FIRST is sound because `applications` is
        # INSERT-only (no UPDATE/DELETE/DROP anywhere in src/) and a 'removed'
        # Correction is only ever detected for an app row ALREADY in the join
        # (normally an EARLIER delta). So any removed visible in the
        # corrections read (T1) has an app row committed at <= T1, and the
        # later written read (T2 >= T1) necessarily includes it -> removed <=
        # written across the split. Reading written FIRST would let a delta
        # land a (removed, its app) pair between the reads and spuriously
        # trip the removed>written refuse. Do NOT "simplify" the order back.
        correction_stats = store_db.correction_label_stats()
        written_by_label = store_db.application_label_counts()
    except AppDBError as e:
        typer.echo(f"status failed: state DB: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None
    finally:
        store_db.close()

    try:
        rows = compute_precision(
            graduated, written_by_label, correction_stats, db_path=str(db_path)
        )
    except PrecisionInvariantError as e:
        # a tampered store (removed > written): refuse named at exit 5,
        # never print a negative precision (F2 item 9)
        typer.echo(f"status failed: {e}", err=True)
        raise typer.Exit(code=_OPERATION_FAILED_EXIT) from None

    if as_json:
        payload = {
            "basis": {
                "bands_content_hash": artifact.content_hash,
                "graduation_content_hash": table.content_hash,
                "caveat": _caveat,
                "attribution_vocabulary": list(CORRECTION_ATTRIBUTIONS),
                "unverifiable_causes": _unverifiable_causes,
            },
            "labels": [
                {
                    "label": r.label,
                    "written": r.written,
                    "removed": r.removed,
                    "precision": r.precision,
                    "added": r.added,
                    "added_filter_covered": r.added_filter_covered,
                    "added_unverifiable": r.added_unverifiable,
                    "human_added": r.human_added,
                }
                for r in rows
            ],
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    typer.echo(
        f"status: {len(rows)} graduated label(s) · basis bands="
        f"{artifact.content_hash[:12]} graduation={table.content_hash[:12]}"
    )
    typer.echo(f"status: NOTE v1 caveats — {_caveat}; {_attribution_note}")
    if not rows:
        typer.echo("status: no graduated labels (nothing to report)")
        return
    for r in rows:
        precision = "no data" if r.precision is None else f"{r.precision:.2f}"
        typer.echo(
            f"  {r.label}: written={r.written} removed={r.removed} "
            f"precision={precision} added={r.added} "
            f"(filter-covered={r.added_filter_covered} "
            f"unverifiable={r.added_unverifiable} "
            f"human={r.human_added})"
        )


if __name__ == "__main__":
    app()
