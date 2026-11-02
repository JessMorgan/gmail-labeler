"""The `auth` onboarding wizard (ADR-0004) — a PURE state machine.

Ticket #28. The wizard walks a new user from a fresh profile to a working
`auth status`: create your own GCP project + desktop OAuth client, point the
tool at the client JSON, wire the token store, sign in. This module is the
DECISION CORE only — every branch is a function of (scripted answers, probed
facts). It performs NO filesystem, network, or keyring I/O and never imports
typer or the OAuth flow: the CLI (``cli.py``) injects real prompts + a real
``EnvironmentProbe`` + the real ``auth.login``/``auth.status``/config writer,
and the tests drive the whole matrix offline through fakes (the AC's
"testable offline: pure over injected prompts/answers"). (The one environment
touch is `Path.expanduser()`, an `$HOME` lookup — not a file read; the AST
purity pin bans the real I/O surface, not that.)

Two design rules the pins enforce:

* **No verification theater.** Gmail API enablement and the consent-screen
  *In production* mode are NOT locally checkable; the wizard says so PLAINLY
  and names the first successful sign-in as the real proof. It never implies
  it checked what it could not.
* **Secret discipline (invariant #8).** Only file PATHS and env NAMES are ever
  emitted. The client JSON's id/secret and the project id the user typed never
  reach output or the result detail — the tests tripwire every one of them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from .auth import AuthError, AuthStatus
from .config import Config, GmailConfig, TokenBackend
from .tokens import TokenStoreError

ConsoleUrl = str
GCP_PROJECTS_URL: ConsoleUrl = "https://console.cloud.google.com/projectselector2/home"
GCP_OAUTH_CREDENTIALS_URL: ConsoleUrl = (
    "https://console.cloud.google.com/apis/credentials"
)
GCP_CONSENT_SCREEN_URL: ConsoleUrl = (
    "https://console.cloud.google.com/apis/credentials/consent"
)

#: The declared step ids, in the order the machine walks them. Tests assert
#: the executed sequence equals this (order + restart determinism).
STEP_IDS: tuple[str, ...] = (
    "intro",
    "project-id",
    "client-json",
    "token-backend",
    "config",
    "login",
    "status",
)

Outcome = Literal["completed", "failed", "aborted", "status-error", "write-error"]


class ClientFileState(Enum):
    """What the probe found at the user-supplied client-JSON path. The probe
    reads the file and returns ONLY this verdict — never the parsed contents,
    so secret material cannot reach the machine (invariant #8)."""

    MISSING = "missing"
    MALFORMED = "malformed"  # unreadable / not valid JSON
    NOT_DESKTOP = "not-desktop"  # parses, but no "installed" wrapper (Web app)
    DESKTOP_OK = "desktop-ok"


@runtime_checkable
class EnvironmentProbe(Protocol):
    """The facts the wizard can check LOCALLY, injected so the machine stays
    pure and every test is offline. Anything not on this interface is NOT
    locally verifiable — the wizard must say so, never fake a check."""

    def client_file_state(self, path: str) -> ClientFileState:
        """exists? parses? is a desktop-type client? (never the contents)."""
        ...

    def project_id_shape_ok(self, pid: str) -> bool:
        """Well-formed GCP project-id SHAPE only (a paste-error guard). The
        project's existence / API enablement are NOT locally checkable."""
        ...

    def keyring_available(self) -> bool:
        """Whether the OS keyring backend works on this box."""
        ...

    def config_target_exists(self, path: Path) -> bool:
        """Whether a config file already sits at the write target (the
        confirm-before-overwrite gate)."""
        ...


@dataclass(frozen=True, slots=True)
class Prompt:
    """One question the machine asks. ``id`` is the test's handle; ``text`` is
    the human-facing prompt (never contains secret material)."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class WizardIO:
    """Injected interactive I/O. ``ask``/``confirm`` raise ``WizardCancelled``
    on Ctrl-C (the CLI wires typer's ``Abort`` to it); ``out`` is the output
    sink (typer.echo in the CLI, a recorder in tests)."""

    ask: Callable[[Prompt], str]
    confirm: Callable[[str], bool]
    out: Callable[[str], None]


class WizardCancelled(Exception):
    """The user cancelled (Ctrl-C at a prompt). Propagated out of the machine
    — it never catches cancel, so no partial state is written. The CLI maps it
    to the repo's Ctrl-C exit convention (130)."""


@dataclass(frozen=True, slots=True)
class WizardResult:
    """The machine's outcome. ``steps`` is the ordered list of step ids it
    actually ran; ``detail`` is a value-free summary (never a secret)."""

    outcome: Outcome
    signed_in: bool
    steps: tuple[str, ...]
    detail: str


def run_wizard(
    *,
    probe: EnvironmentProbe,
    io: WizardIO,
    login: Callable[[Config], None],
    status: Callable[[Config], AuthStatus],
    write_config: Callable[[Path, Config], None],
    config_target: Path,
) -> WizardResult:
    """Drive the declared steps to completion, failure, or a clean decline.

    ``login``/``status``/``write_config`` are injected (the CLI wires the real
    ``auth.login``/``auth.status``/config writer) — the wizard COMPOSES the
    existing login flow rather than forking the exchange logic. Cancel is not
    caught here: a ``WizardCancelled`` from any prompt propagates, and every
    write is downstream of its own prompt, so a cancel AT A PROMPT leaves
    nothing on disk. (A cancel during the composed browser sign-in happens
    after config is written — it leaves a valid config but no token; re-running
    finishes it. That residual is benign and disclosed, not "nothing partial".)
    """
    executed: list[str] = []

    # --- intro -------------------------------------------------------------
    executed.append("intro")
    _emit_intro(io)

    # --- project-id (shape check only; NOT enablement) ---------------------
    executed.append("project-id")
    if not _run_project_id(io=io, probe=probe):
        # capped strikes + the user declined to skip: terminate cleanly (this
        # loop ends on every input — qwen's hang demonstration is closed).
        io.out(
            "  Stopping here — the project-id shape check did not pass and you "
            "chose not to skip it. Nothing was written. The README's manual "
            "setup covers the same steps if you'd rather drive them yourself."
        )
        return WizardResult(
            outcome="aborted",
            signed_in=False,
            steps=tuple(executed),
            detail="project-id shape check not passed (manual setup suggested)",
        )

    # --- client-json (exists / parses / desktop-type) ----------------------
    executed.append("client-json")
    client_path = _run_client_json(io=io, probe=probe)
    if client_path is None:
        # capped strikes: no valid desktop client. Terminate cleanly (this loop
        # ends on every input too — the F3-4b cap).
        return WizardResult(
            outcome="aborted",
            signed_in=False,
            steps=tuple(executed),
            detail="no valid desktop client after 3 tries (manual setup suggested)",
        )

    # --- token-backend (keyring posture, honest file fallback) -------------
    executed.append("token-backend")
    backend = _run_token_backend(io=io, probe=probe)

    # --- config (confirm-before-overwrite, then write) ---------------------
    executed.append("config")
    cfg = Config(
        gmail=GmailConfig(
            # expanduser so the value the probe verified (it expands too) is
            # the SAME value login() receives — otherwise a `~/…` path passes
            # the local check then fails inside login with a misleading
            # console-side diagnosis (review MAJOR 1).
            credentials_path=Path(client_path).expanduser(),
            token_backend=backend,
        )
    )
    if backend == "file":
        io.out(f"  token file: {cfg.gmail.token_file} (kept at mode 0600)")
    # resolve()-compare posture, stated: existence is decided by a
    # Path.resolve() compare in the probe — the samefile/fd-level caution
    # family is issue #47's scope, deliberately NOT extended here.
    if probe.config_target_exists(config_target) and not io.confirm(
        f"A config.toml already exists at {config_target}. Continuing REPLACES "
        "that ENTIRE file with just the gmail wiring this wizard collected — "
        "every other section and comment is lost. To keep your current config, "
        "answer no. Overwrite it?"
    ):
        io.out(
            "Not overwriting — your config.toml is untouched. If you only need "
            "to re-authenticate, run `gmail-labeler auth login`; if the client "
            "path changed, edit gmail.credentials_path by hand. The wizard "
            "wrote nothing."
        )
        return WizardResult(
            outcome="aborted",
            signed_in=False,
            steps=tuple(executed),
            detail="config not written (overwrite declined)",
        )
    try:
        write_config(config_target, cfg)
    except OSError as e:
        # Narrowed to THIS call (review A5): a config-write failure (parent is
        # a file, read-only dir, …) is an operation failure (exit 5). A
        # BrokenPipeError from the OUTPUT sink is NOT caught here — io.out is
        # outside this try — so the CLI can tell "stdout closed mid-run" apart
        # from "the write failed". The diagnosis is reported on STDERR by the CLI
        # (consistent with every other auth-path error), not echoed here.
        return WizardResult(
            outcome="write-error",
            signed_in=False,
            steps=tuple(executed),
            detail=f"could not write {config_target} ({type(e).__name__}) — "
            "nothing was signed in; fix the path/permissions and re-run",
        )
    io.out(f"  wrote {config_target}")

    # --- login (COMPOSES the existing auth.login flow) ---------------------
    executed.append("login")
    try:
        login(cfg)
    except WizardCancelled:
        raise  # a cancel is not a sign-in failure — let it reach the 130 mapping
    except BrokenPipeError:
        # auth.login prints the unverified-app notice / auth URL to stdout; a
        # closed pipe there must reach the CLI's cancel-class (130) handler, not
        # be swallowed below as a console-side sign-in failure (exit 4).
        raise
    except (AuthError, TokenStoreError) as e:
        # These are secret-free BY CONSTRUCTION (auth.py / tokens.py raise them
        # from type names + paths only). We still do not echo the raw text — a
        # caller can hand us a value-bearing error — but naming the type and
        # splitting local-vs-console causes is the actionable part (review
        # MINOR 5: the old bare `except Exception` turned a local "file not
        # found" into a wild-goose console diagnosis).
        _emit_login_failure(
            io, cause=type(e).__name__, store=isinstance(e, TokenStoreError)
        )
        return WizardResult(
            outcome="failed",
            signed_in=False,
            steps=tuple(executed),
            detail="sign-in did not complete (see the guidance above)",
        )
    except Exception as e:  # unexpected: type name only, never the text
        _emit_login_failure(io, cause=type(e).__name__, store=False)
        return WizardResult(
            outcome="failed",
            signed_in=False,
            steps=tuple(executed),
            detail="sign-in did not complete (see the guidance above)",
        )
    # NOT "signed in" yet — that verdict belongs to the status check below, not
    # here (F3 NIT-a: a green line immediately before a possible failure read
    # as a premature verdict).
    io.out("  refresh token stored — checking status…")

    # --- status ------------------------------------------------------------
    executed.append("status")
    try:
        st = status(cfg)
    except BrokenPipeError:
        raise  # stdout closed mid-check → the CLI's cancel-class (130), not a 5
    except Exception as e:
        # The injected status action crashed — a CHECK failure, not an auth
        # state (the sign-in itself may have stored a token). The CLI reports it
        # on STDERR (consistent with the auth-path errors) and exits 5; it never
        # escapes as a raw traceback (exit 1, off the 0/2/4/5/130 contract).
        return WizardResult(
            outcome="status-error",
            signed_in=False,
            steps=tuple(executed),
            detail=f"status check failed ({type(e).__name__}) — sign-in may have "
            "worked; run `gmail-labeler auth status` to check",
        )
    io.out(
        f"auth: {'signed in' if st.signed_in else 'NOT signed in'} "
        f"({st.backend}) — {st.detail}"
    )
    return WizardResult(
        outcome="completed",
        signed_in=st.signed_in,
        steps=tuple(executed),
        detail=st.detail,
    )


def _emit_intro(io: WizardIO) -> None:
    """The ADR-0004 explanation + the honesty clause. The wizard surfaces the
    console steps and says PLAINLY what it cannot verify locally; it does not
    duplicate the OAuth flow's own unverified-app notice (that rides the
    composed ``auth.login``)."""
    io.out(
        "gmail-labeler setup — bring your own Google Cloud project (ADR-0004).\n"
        "Nothing is shared with the maintainer: you create the project and the\n"
        "OAuth client, and the tool talks to Gmail only as you. Budget ~15\n"
        "minutes of console clicking.\n"
        f"\n"
        f"  1. Create/select a project: {GCP_PROJECTS_URL}\n"
        f"  2. Enable the Gmail API in it.\n"
        f"  3. Make an OAuth client ID of type DESKTOP APP:\n"
        f"     {GCP_OAUTH_CREDENTIALS_URL}\n"
        f"     (a Web-application client cannot complete the local loopback flow)\n"
        f"  4. Set the OAuth consent screen to IN PRODUCTION, not Testing: "
        f"{GCP_CONSENT_SCREEN_URL}\n"
        f"     (in Testing the refresh token expires every 7 days)\n"
        f"  5. Download the client JSON and keep note of where it landed.\n"
        "\n"
        "I can only check things on THIS machine: that the client file exists,\n"
        "parses, and is a desktop-type client, and that a token store is usable.\n"
        "I CANNOT check from here whether the Gmail API is enabled in your project\n"
        "or whether the consent screen is set to In production. Signing in proves\n"
        "the OAuth client and your consent; the FIRST GMAIL CALL (e.g. `dataset\n"
        "export`) is what proves the API is enabled; and a token still working\n"
        "after 7 days is what proves In-production mode. I will not pretend to\n"
        "verify any of that now."
    )


def _run_project_id(*, io: WizardIO, probe: EnvironmentProbe) -> bool:
    """Collect the project id and check its SHAPE only (a paste-error guard).
    The value is never echoed and never stored — the config holds no project
    id, so logging one would be a leak (design landmine).

    Returns True to continue, False to abort the wizard. After `max_strikes`
    malformed answers the skip decision is FINAL (accept → continue, decline →
    abort), so THIS loop terminates on every input — the old unbounded loop hung
    when a declined skip kept re-asking (qwen's demonstration). (The client-json
    loop is likewise bounded only by the user's willingness to supply a valid
    path; there is no refused decision to force there.)"""
    max_strikes = 3
    strikes = 0
    while True:
        pid = io.ask(
            Prompt(
                id="project-id",
                text="Your GCP project id (Console → top bar; e.g. 'my-project-123'):",
            )
        )
        if probe.project_id_shape_ok(pid):
            io.out("  project id looks well-formed (shape only — I can't reach it)")
            return True
        strikes += 1
        io.out(
            "  That doesn't look like a GCP project id (lowercase letters, digits,\n"
            "  and hyphens; 6-30 chars, starting with a letter). I won't echo it back."
        )
        if strikes >= max_strikes:
            # cap reached — the skip decision is FINAL (no more re-asks).
            if io.confirm(
                "Skip the project-id shape check and continue? (It's only a "
                "paste-error guard — I cannot verify API enablement or the "
                "project's existence from here either way.) Answer no to stop "
                "and use the manual setup instead."
            ):
                io.out(
                    "  Continuing without the shape check. Remember: I cannot "
                    "verify the Gmail API is enabled or the consent screen is In "
                    "production — the first Gmail call proves enablement, not "
                    "anything here."
                )
                return True
            return False
        # else: re-ask (bounded — at most max_strikes malformed before the
        # forced decision)


def _run_client_json(*, io: WizardIO, probe: EnvironmentProbe) -> str | None:
    """Ask for the downloaded client-JSON path until it verifies as a
    desktop-type client. Only the PATH is echoed (config legitimately holds a
    path); the file's contents never reach here (the probe returns a verdict).

    Returns the verified path, or None to abort the wizard after `max_strikes`
    rejected paths. A valid client is REQUIRED (login cannot proceed without
    one), so unlike the project-id shape guard there is nothing to skip — the cap
    simply terminates with a guide-to-manual message. F3-4b: the same
    terminate-on-every-input guarantee, so this loop cannot spin forever either
    (C1's headline claim, now true for both loops)."""
    max_strikes = 3
    strikes = 0
    while True:
        path = io.ask(
            Prompt(
                id="client-json",
                text="Path to the desktop OAuth client JSON you downloaded "
                "(~ or a full path):",
            )
        )
        state = probe.client_file_state(path)
        if state is ClientFileState.DESKTOP_OK:
            io.out(f"  found a desktop-type client at {path}")
            return path
        strikes += 1
        if state is ClientFileState.MISSING:
            io.out(
                f"  Client file not found at {path}. Download the Desktop-app "
                "client JSON from the console and give me where it landed."
            )
        elif state is ClientFileState.MALFORMED:
            io.out(
                f"  {path} is not a usable desktop client — it is unreadable "
                "(check its file permissions), not valid JSON, or missing "
                "client_id/client_secret; if the file itself is wrong, "
                "re-download it (Google generates it — don't hand-edit it)."
            )
        else:  # NOT_DESKTOP
            io.out(
                f"  {path} is not a DESKTOP-app client — it has no 'installed' "
                "wrapper, which means a Web-application client was chosen in the "
                "console. Create an OAuth client of type Desktop app instead and "
                "point me at its JSON."
            )
        if strikes >= max_strikes:
            io.out(
                "  Stopping here — I could not verify a desktop client after "
                f"{max_strikes} tries. Nothing was written. The README's manual "
                "setup covers the same steps if you'd rather drive them yourself."
            )
            return None
        # else: re-ask (bounded — at most max_strikes rejects before the abort)


def _run_token_backend(*, io: WizardIO, probe: EnvironmentProbe) -> TokenBackend:
    """Keyring first (the existing posture); an honest 0600-file fallback when
    the box has no working keyring backend — stated, never silent."""
    if probe.keyring_available():
        io.out("  tokens will be stored in the OS keyring (the default, ADR-0004)")
        return "keyring"
    io.out(
        "  No usable OS keyring on this box — falling back to a token FILE at "
        'mode 0600 (you can switch back with gmail.token_backend = "keyring" '
        "once a keyring is available)."
    )
    return "file"


def _emit_login_failure(io: WizardIO, *, cause: str, store: bool) -> None:
    """The composed login raised. Echo nothing of the exception TEXT (a caller
    can hand us a value-bearing error); name its TYPE (secret-free by
    construction) and split the likely cause — a local store/path problem is
    fixable here, everything else is console-side and was never checkable."""
    io.out(f"  Sign-in did not complete ({cause}).")
    if store:
        io.out(
            "  This is the TOKEN STORE, not the console: on a headless box set "
            'gmail.token_backend = "file" in config.toml (a 0600 file), or fix '
            "your OS keyring, then re-run `gmail-labeler auth setup`."
        )
        return
    io.out(
        "  A missing or unreadable client path is a LOCAL problem to fix "
        "directly (re-check the file you pointed me at). Otherwise the usual "
        "causes are console-side, and I could not check any of them:\n"
        "    * the Gmail API is not ENABLED in your project;\n"
        "    * the OAuth consent screen is in Testing, not In production;\n"
        "    * the 'Google hasn't verified this app' screen needs Advanced → Go to\n"
        "      project → Allow.\n"
        "  Fix the console side and re-run `gmail-labeler auth setup` (or the raw\n"
        "  `gmail-labeler auth login` once your config is in place)."
    )
