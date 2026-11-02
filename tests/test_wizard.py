"""Ticket #28 — the auth/onboarding wizard as a PURE state machine.

Every branch here is a function of (scripted answers, probed facts): the
machine never touches the filesystem, the keyring, the network, or typer. The
CLI (test_wizard_cli.py) wires the real I/O; THIS file drives the whole matrix
offline through fake seams — the AC's "testable offline: pure over injected
prompts/answers". The sentinels below are the secret-discipline tripwires
(invariant #8): if any of them reaches wizard output or the result detail, a
pin reddens.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path

import pytest

from gmail_labeler.auth import AuthError, AuthStatus
from gmail_labeler.config import Config
from gmail_labeler.wizard import (
    STEP_IDS,
    ClientFileState,
    EnvironmentProbe,
    Prompt,
    WizardCancelled,
    WizardIO,
    WizardResult,
    run_wizard,
)

# --- secret-discipline sentinels (invariant #8) ------------------------------
# These stand in for real secret material the wizard must NEVER echo/log: the
# client JSON's id+secret, and a project id the user typed. A leak of any of
# them into `out` or the result detail is a red pin.
CLIENT_ID = "SENTINEL-client-id.apps"
CLIENT_SECRET = "SENTINEL-client-secret-never-echoed"
PROJECT_ID = "sentinel-project-42"
CLIENT_PATH = "/home/user/secrets/desktop-client.json"
TOKEN_FILE = "/home/user/.config/gmail-labeler/token.json"
CONFIG_TARGET = Path("/home/user/work/config.toml")


# --- fake seams --------------------------------------------------------------
@dataclass
class FakeProbe:
    """Scripted environment facts. The client-file verdicts are consumed in
    order (a retry walks the list); the last one repeats."""

    client_states: list[ClientFileState] = field(
        default_factory=lambda: [ClientFileState.DESKTOP_OK]
    )
    project_ok: bool = True
    keyring: bool = True
    config_exists: bool = False

    def _next(self) -> ClientFileState:
        if len(self.client_states) == 1:
            return self.client_states[0]
        return self.client_states.pop(0)

    def client_file_state(self, path: str) -> ClientFileState:
        return self._next()

    def project_id_shape_ok(self, pid: str) -> bool:
        return self.project_ok

    def keyring_available(self) -> bool:
        return self.keyring

    def config_target_exists(self, path: Path) -> bool:
        return self.config_exists


@dataclass
class FakeStdin:
    """Scripted prompt answers keyed by prompt id; a list value is consumed
    per invocation (retry walks it). Raises WizardCancelled on a Cancel."""

    answers: dict[str, list[str]] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)

    def ask(self, prompt: Prompt) -> str:
        self.seen.append(prompt.id)
        seq = self.answers.get(prompt.id)
        if seq is None:
            raise AssertionError(f"unexpected prompt: {prompt.id}")
        if not seq:
            raise AssertionError(f"ran out of answers for {prompt.id}")
        value = seq.pop(0) if len(seq) > 1 else seq[0]
        if value == "__CANCEL__":
            raise WizardCancelled()
        return value


@dataclass
class FakeConfirms:
    """Scripted yes/no keyed by a substring of the confirm text."""

    answers: dict[str, bool] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)

    def confirm(self, text: str) -> bool:
        self.seen.append(text)
        for key, val in self.answers.items():
            if key.lower() in text.lower():
                return val
        return True  # default yes


class Recorder:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def out(self, text: str) -> None:
        self.lines.append(text)

    @property
    def all_text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class FakeActions:
    """Records write_config/login/status wiring; login/status may be told to fail."""

    login_error: Exception | None = None
    status_error: Exception | None = None
    login_called: list[Config] = field(default_factory=list)
    written: list[tuple[Path, Config]] = field(default_factory=list)
    signed_in: bool = True

    def login(self, cfg: Config) -> None:
        self.login_called.append(cfg)
        if self.login_error is not None:
            raise self.login_error

    def status(self, cfg: Config) -> AuthStatus:
        if self.status_error is not None:
            raise self.status_error
        return AuthStatus(
            signed_in=self.signed_in,
            backend="keyring",
            detail="usable refresh token present (gmail.modify)",
        )

    def write_config(self, target: Path, cfg: Config) -> None:
        self.written.append((target, cfg))


def _run(
    *,
    probe: FakeProbe | None = None,
    stdin: FakeStdin | None = None,
    confirms: FakeConfirms | None = None,
    actions: FakeActions | None = None,
) -> tuple[WizardResult, Recorder, FakeActions, FakeStdin]:
    probe = probe or FakeProbe()
    stdin = stdin or FakeStdin(
        answers={"project-id": [PROJECT_ID], "client-json": [CLIENT_PATH]}
    )
    confirms = confirms or FakeConfirms()
    actions = actions or FakeActions()
    rec = Recorder()
    io = WizardIO(ask=stdin.ask, confirm=confirms.confirm, out=rec.out)
    result = run_wizard(
        probe=probe,
        io=io,
        login=actions.login,
        status=actions.status,
        write_config=actions.write_config,
        config_target=CONFIG_TARGET,
    )
    return result, rec, actions, stdin


# --- happy path + step order -------------------------------------------------
def test_happy_path_completes_signed_in() -> None:
    result, _rec, actions, _ = _run()
    assert result.outcome == "completed"
    assert result.signed_in is True
    assert actions.login_called  # login COMPOSED once
    assert actions.written == [(CONFIG_TARGET, actions.login_called[0])]


def test_executed_steps_are_the_declared_order() -> None:
    # the machine reports the step ids it ran, in the declared order — the
    # test's handle on "order/restart" (design: declared list with ids)
    result, *_ = _run()
    assert result.steps == STEP_IDS


def test_restart_reproduces_the_same_step_order() -> None:
    # a fresh run from scratch walks the SAME ids (restart determinism)
    first, *_ = _run()
    second, *_ = _run()
    assert first.steps == second.steps == STEP_IDS


def test_config_written_with_credentials_path_and_backend() -> None:
    _result, _rec, actions, _ = _run()
    target, cfg = actions.written[0]
    assert target == CONFIG_TARGET
    assert str(cfg.gmail.credentials_path) == CLIENT_PATH
    assert cfg.gmail.token_backend == "keyring"


# --- local verification branches (each CAN-check pinned) ---------------------
def test_client_missing_then_not_desktop_then_ok_converges() -> None:
    probe = FakeProbe(
        client_states=[
            ClientFileState.MISSING,
            ClientFileState.NOT_DESKTOP,
            ClientFileState.DESKTOP_OK,
        ]
    )
    stdin = FakeStdin(
        answers={
            "project-id": [PROJECT_ID],
            "client-json": [CLIENT_PATH, CLIENT_PATH, CLIENT_PATH],
        }
    )
    result, rec, actions, _ = _run(probe=probe, stdin=stdin)
    assert result.outcome == "completed"
    assert len(actions.login_called) == 1  # converged, login ran once
    assert "not found" in rec.all_text.lower()


def test_not_desktop_is_a_named_refusal() -> None:
    # a Web-app client is the classic console mistake — the REFUSAL itself
    # must name it. "web" also appears in the intro's step list, so a bare
    # "web in output" assertion is satisfied by the wrong line (presence is
    # not pairing — the F4 lesson). Pin the refusal's OWN line: the one that
    # names the missing 'installed' wrapper.
    probe = FakeProbe(
        client_states=[ClientFileState.NOT_DESKTOP, ClientFileState.DESKTOP_OK]
    )
    stdin = FakeStdin(
        answers={
            "project-id": [PROJECT_ID],
            "client-json": [CLIENT_PATH, CLIENT_PATH],
        }
    )
    _result, rec, _actions, _ = _run(probe=probe, stdin=stdin)
    refusal = [ln for ln in rec.lines if "installed" in ln.lower()]
    assert refusal, "the desktop-type refusal (installed-wrapper) was never emitted"
    line = refusal[0].lower()
    assert "desktop" in line
    assert "web" in line  # names the wrong console choice IN THE REFUSAL


def test_malformed_client_json_is_named_and_retried() -> None:
    probe = FakeProbe(
        client_states=[ClientFileState.MALFORMED, ClientFileState.DESKTOP_OK]
    )
    stdin = FakeStdin(
        answers={
            "project-id": [PROJECT_ID],
            "client-json": [CLIENT_PATH, CLIENT_PATH],
        }
    )
    result, rec, _actions, _ = _run(probe=probe, stdin=stdin)
    assert result.outcome == "completed"
    assert (
        "malformed" in rec.all_text.lower() or "not valid json" in rec.all_text.lower()
    )


def test_malformed_names_both_remedies_permissions_and_redownload() -> None:
    # B8: EACCES is mapped to MALFORMED by the probe; the guidance must NOT say
    # only "re-download" (wrong remedy for a permissions problem). Name both.
    probe = FakeProbe(
        client_states=[ClientFileState.MALFORMED, ClientFileState.DESKTOP_OK]
    )
    stdin = FakeStdin(
        answers={
            "project-id": [PROJECT_ID],
            "client-json": [CLIENT_PATH, CLIENT_PATH],
        }
    )
    _result, rec, _actions, _ = _run(probe=probe, stdin=stdin)
    low = rec.all_text.lower()
    assert "permission" in low  # the EACCES remedy
    assert "re-download" in low  # the genuinely-corrupt-file remedy


def test_project_id_malformed_retries_without_echoing_the_value() -> None:
    probe = FakeProbe(project_ok=False)
    stdin = FakeStdin(
        answers={"project-id": [PROJECT_ID, PROJECT_ID], "client-json": [CLIENT_PATH]}
    )
    confirms = FakeConfirms({"Skip": True})  # explicit key matching the prompt
    _result, rec, _actions, _ = _run(probe=probe, stdin=stdin, confirms=confirms)
    # the malformed value is NEVER echoed back (it is secret-adjacent input)
    assert PROJECT_ID not in rec.all_text
    assert "project" in rec.all_text.lower()
    # A2: the skip was genuinely OFFERED and driven by the scripted key (not
    # carried by FakeConfirms' default) — flip-default mutant must now FAIL.
    assert any("Skip" in s for s in confirms.seen)


def test_project_id_skip_carries_a_plain_consequence() -> None:
    probe = FakeProbe(project_ok=False)
    stdin = FakeStdin(
        answers={"project-id": [PROJECT_ID, PROJECT_ID], "client-json": [CLIENT_PATH]}
    )
    confirms = FakeConfirms({"Skip": True})
    result, rec, _actions, _ = _run(probe=probe, stdin=stdin, confirms=confirms)
    assert result.outcome == "completed"  # skip is allowed, not fatal
    assert any("Skip" in s for s in confirms.seen)  # the scripted skip drove it
    low = rec.all_text.lower()
    # the consequence is stated PLAINLY: enablement is still not checkable
    assert "cannot" in low and "enable" in low


def test_project_id_cap_terminates_on_declined_skip() -> None:
    # A1 (qwen's hang demonstration closed): 4 malformed strikes with the skip
    # DECLINED must TERMINATE (aborted), not loop forever. The cap forces the
    # decision at strike 3, so a 4th answer is never consumed.
    probe = FakeProbe(project_ok=False)
    stdin = FakeStdin(
        answers={
            "project-id": [PROJECT_ID] * 4,  # more than the cap would consume
            "client-json": [CLIENT_PATH],
        }
    )
    confirms = FakeConfirms({"Skip": False})  # decline the forced skip
    result, _rec, actions, _ = _run(probe=probe, stdin=stdin, confirms=confirms)
    assert result.outcome == "aborted"  # terminated, not hung
    assert result.steps == ("intro", "project-id")  # never reached client-json
    assert actions.written == []  # nothing written
    # the loop asked exactly max_strikes (3) times, then forced the decision
    assert stdin.seen.count("project-id") == 3


# --- NO verification theater (the AC's honesty clause) -----------------------
def test_wizard_states_enablement_and_production_are_not_locally_checkable() -> None:
    # the honesty clause, pinned by its CONTIGUOUS signature — not by the bare
    # words "enable"/"production"/"cannot", which also appear in the console
    # STEP list (presence of those words proved nothing: the F4 lesson). This
    # exact phrase exists ONLY in the "what I cannot check" sentence.
    _result, rec, _actions, _ = _run()
    low = rec.all_text.lower()
    assert "cannot check from here whether the gmail api is enabled" in low
    assert "consent screen is set to in production" in low
    # and it never claims to have CHECKED them (no fake success theater)
    assert "verified the gmail api" not in low
    assert "enabled confirmed" not in low


def test_wizard_names_the_real_proofs_precisely() -> None:
    # the design's honesty, corrected (review MINOR 8): sign-in proves the
    # client+consent, the first GMAIL CALL proves enablement, and a token that
    # survives 7 days proves In-production mode — the wizard must not name
    # sign-in as the proof of API enablement (it isn't).
    _result, rec, _actions, _ = _run()
    low = rec.all_text.lower()
    assert "signing in proves" in low and "oauth client" in low
    assert "first gmail call" in low and "proves the api is enabled" in low
    assert "after 7 days" in low and "in-production mode" in low


# --- token backend degrade is honest (keyring posture) -----------------------
def test_keyring_available_uses_keyring_backend() -> None:
    probe = FakeProbe(keyring=True)
    _result, _rec, actions, _ = _run(probe=probe)
    assert actions.written[0][1].gmail.token_backend == "keyring"


def test_keyring_unavailable_falls_back_to_file_with_plain_notice() -> None:
    probe = FakeProbe(keyring=False)
    _result, rec, actions, _ = _run(probe=probe)
    assert actions.written[0][1].gmail.token_backend == "file"
    low = rec.all_text.lower()
    assert "keyring" in low
    assert "file" in low
    assert "0600" in rec.all_text  # the fallback's mode is named


# --- confirm-before-overwrite on the config path -----------------------------
def test_existing_config_asks_before_overwrite_and_writes_on_yes() -> None:
    probe = FakeProbe(config_exists=True)
    confirms = FakeConfirms({"overwrite": True})
    _result, _rec, actions, _ = _run(probe=probe, confirms=confirms)
    assert any("overwrite" in c.lower() for c in confirms.seen)
    # MAJOR-2 disclosure pinned: the confirm must plainly say it REPLACES the
    # whole file (a mild "overwrite it with the wiring" wording hid the blast
    # radius — the reviewer's m_confirmwording mutant)
    confirm_text = " ".join(confirms.seen).lower()
    assert "replaces" in confirm_text and "entire file" in confirm_text
    assert actions.written  # confirmed → it writes


def test_declined_overwrite_does_not_write_and_aborts_cleanly() -> None:
    probe = FakeProbe(config_exists=True)
    confirms = FakeConfirms({"overwrite": False})
    result, rec, actions, _ = _run(probe=probe, confirms=confirms)
    assert actions.written == []  # NOTHING overwritten
    assert result.outcome != "completed"  # aborted, not a silent continue
    # MAJOR-2: the decline names the lighter re-auth route (m_declinehint)
    assert "auth login" in rec.all_text
    low = rec.all_text.lower()
    assert "manual" in low or "config.toml" in low  # names the manual path


# --- login failure maps to the named console step (no theater) ---------------
def test_login_failure_maps_to_named_console_steps() -> None:
    actions = FakeActions(
        login_error=AuthError("consent flow did not complete (AttributeError)")
    )
    result, rec, _a, _ = _run(actions=actions)
    assert result.outcome == "failed"
    assert result.signed_in is False
    low = rec.all_text.lower()
    # the failure points at the human console steps, plainly framed as a hint
    assert "enable" in low and "production" in low
    assert "unverified" in low or "advanced" in low


def test_login_failure_never_leaks_secret_or_traceback() -> None:
    actions = FakeActions(
        login_error=AuthError(f"boom {CLIENT_SECRET} {CLIENT_ID}"),
    )
    result, rec, _a, _ = _run(actions=actions)
    assert result.outcome == "failed"
    assert CLIENT_SECRET not in rec.all_text
    assert CLIENT_ID not in rec.all_text
    assert CLIENT_SECRET not in result.detail
    assert CLIENT_ID not in result.detail


def test_login_token_store_error_gets_store_guidance_not_console_guess() -> None:
    # review MINOR 5: a headless-keyring failure is LOCAL, not a console
    # red-herring. TokenStoreError -> the store-specific remedy (and the type
    # is named, never the raw text).
    from gmail_labeler.tokens import TokenStoreError

    actions = FakeActions(
        login_error=TokenStoreError("OS keyring write failed (RuntimeError)")
    )
    result, rec, _a, _ = _run(actions=actions)
    assert result.outcome == "failed"
    # scope to the FAILURE message (the intro's honesty clause mentions
    # "enabled" too — a whole-output check would be presence-not-pairing)
    tail = rec.all_text.split("Sign-in did not complete", 1)[1].lower()
    assert "token store" in tail and "0600" in tail
    assert "enabled" not in tail  # NOT misdiagnosed as a console enablement issue


def test_login_unexpected_error_names_type_and_never_echoes_text() -> None:
    # the generic fallback: an unexpected error is reported by TYPE only
    actions = FakeActions(login_error=RuntimeError(f"unexpected {CLIENT_SECRET}"))
    result, rec, _a, _ = _run(actions=actions)
    assert result.outcome == "failed"
    assert "RuntimeError" in rec.all_text  # the type is named for diagnosis
    assert CLIENT_SECRET not in rec.all_text  # the raw text is never echoed


def test_cancel_during_login_propagates_not_reported_as_failure() -> None:
    # review MINOR 5: a WizardCancelled raised inside the composed login must
    # NOT be swallowed into a "failed" sign-in — it propagates to the 130 path
    actions = FakeActions(login_error=WizardCancelled())
    io = WizardIO(
        ask=FakeStdin(
            answers={"project-id": [PROJECT_ID], "client-json": [CLIENT_PATH]}
        ).ask,
        confirm=FakeConfirms().confirm,
        out=Recorder().out,
    )
    with pytest.raises(WizardCancelled):
        run_wizard(
            probe=FakeProbe(),
            io=io,
            login=actions.login,
            status=actions.status,
            write_config=actions.write_config,
            config_target=CONFIG_TARGET,
        )


def test_status_check_crash_becomes_named_status_error_not_escape() -> None:
    # A3: an exception escaping the injected status action must NOT break the
    # exit contract (raw traceback -> exit 1). It becomes a NAMED status-error.
    actions = FakeActions(status_error=RuntimeError(f"boom {CLIENT_SECRET}"))
    result, rec, _a, _ = _run(actions=actions)
    assert result.outcome == "status-error"
    assert result.signed_in is False
    assert "status check failed" in result.detail
    # the raw text is never echoed (type only), and the surface is named
    assert CLIENT_SECRET not in rec.all_text
    assert "status check failed" in rec.all_text.lower()


def test_completed_wizard_reports_not_signed_in_when_status_says_so() -> None:
    # A4: the honesty branch — login succeeded but status reports signed_in=False
    # through the WHOLE completed wizard. The machine must NOT hardcode green.
    actions = FakeActions(signed_in=False)
    result, rec, _a, _ = _run(actions=actions)
    assert result.outcome == "completed"  # steps ran to the end
    assert result.signed_in is False  # NOT hardcoded True
    tail = rec.all_text.split("auth:", 1)[1].lower()  # the status line
    assert "not signed in" in tail  # line-swap mutant (prints "signed in") reddens


# --- secret discipline: sentinels absent from ALL output + detail ------------
def test_no_secret_sentinels_in_any_wizard_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # the invariant-#8 tripwire: client id/secret + the typed project id must
    # be absent from every out line AND the result detail, on the HAPPY path
    result, rec, _actions, _ = _run()
    for sentinel in (CLIENT_ID, CLIENT_SECRET, PROJECT_ID):
        assert sentinel not in rec.all_text, f"{sentinel} leaked to output"
        assert sentinel not in result.detail, f"{sentinel} leaked to detail"
    # B9: the "and logs" half of the invariant — the wizard emits NO log records
    # today, so this is honest-but-guarding: if anyone adds logging on the auth
    # path, a sentinel reaching a LogRecord reddens this (the surface is covered,
    # not assumed clean).
    with caplog.at_level("DEBUG"):
        _run()
    for record in caplog.records:
        msg = record.getMessage()
        for sentinel in (CLIENT_ID, CLIENT_SECRET, PROJECT_ID):
            assert sentinel not in msg, f"{sentinel} leaked to a log record"


def test_no_secret_sentinels_on_the_retry_and_failure_paths() -> None:
    probe = FakeProbe(
        client_states=[
            ClientFileState.MISSING,
            ClientFileState.NOT_DESKTOP,
            ClientFileState.DESKTOP_OK,
        ],
        project_ok=False,
    )
    stdin = FakeStdin(
        answers={
            "project-id": [PROJECT_ID, PROJECT_ID],
            "client-json": [CLIENT_PATH, CLIENT_PATH, CLIENT_PATH],
        }
    )
    confirms = FakeConfirms({"skip": True})
    result, rec, _actions, _ = _run(probe=probe, stdin=stdin, confirms=confirms)
    for sentinel in (CLIENT_ID, CLIENT_SECRET, PROJECT_ID):
        assert sentinel not in rec.all_text
        assert sentinel not in result.detail


# --- cancel (Ctrl-C) propagates as a clean exception, writes nothing ---------
def test_cancel_at_a_prompt_raises_and_writes_nothing() -> None:
    stdin = FakeStdin(
        answers={"project-id": ["__CANCEL__"], "client-json": [CLIENT_PATH]}
    )
    actions = FakeActions()
    probe = FakeProbe()
    rec = Recorder()
    io = WizardIO(ask=stdin.ask, confirm=FakeConfirms().confirm, out=rec.out)
    with pytest.raises(WizardCancelled):
        run_wizard(
            probe=probe,
            io=io,
            login=actions.login,
            status=actions.status,
            write_config=actions.write_config,
            config_target=CONFIG_TARGET,
        )
    assert actions.written == []  # cancel before any write
    assert actions.login_called == []


def test_cancel_at_client_json_prompt_writes_nothing() -> None:
    # cancel AFTER project-id accepted but at the client-json prompt: the
    # config write is still downstream → nothing on disk, no login
    stdin = FakeStdin(
        answers={"project-id": [PROJECT_ID], "client-json": ["__CANCEL__"]}
    )
    actions = FakeActions()
    io = WizardIO(ask=stdin.ask, confirm=FakeConfirms().confirm, out=Recorder().out)
    with pytest.raises(WizardCancelled):
        run_wizard(
            probe=FakeProbe(),
            io=io,
            login=actions.login,
            status=actions.status,
            write_config=actions.write_config,
            config_target=CONFIG_TARGET,
        )
    assert actions.written == []
    assert actions.login_called == []


# --- the seam is a real Protocol (the CLI can satisfy it structurally) -------
def test_probe_is_a_runtime_checkable_protocol() -> None:
    # the fake probe structurally satisfies EnvironmentProbe (no ABC coupling)
    assert isinstance(FakeProbe(), EnvironmentProbe)


def test_result_is_a_frozen_value_object() -> None:
    result, *_ = _run()
    with pytest.raises(FrozenInstanceError):
        result.outcome = "mutated"  # type: ignore[misc]


# --- the machine is PURE: pinned at the SOURCE/AST level --------------------
def test_wizard_module_has_no_io_imports_or_calls() -> None:
    # Purity, pinned precisely: the machine performs NO filesystem, network, or
    # keyring I/O and never imports typer / the OAuth flow. (It DOES read $HOME
    # via Path.expanduser — an environment lookup, not a file read — so the
    # claim is scoped to real I/O, not "touches nothing".) The old "poison the
    # flow module" test could never fail; this AST pin reddens if I/O is added.
    # Docstring prose about typer creates no Import/Call nodes, so it stays
    # honest and fail-able.
    import ast

    import gmail_labeler.wizard as wmod

    tree = ast.parse(Path(wmod.__file__).read_text(encoding="utf-8"))
    banned = {"typer", "keyring", "google_auth_oauthlib", "subprocess", "socket"}
    # bare open() AND Path/FS attribute calls — the reviewer's m_pathio mutant
    # (Path(...).read_text() inside run_wizard) slipped past an open()-only pin.
    banned_calls = {
        "open",
        "read_text",
        "write_text",
        "read_bytes",
        "write_bytes",
        "iterdir",
        "glob",
        "stat",
        "exists",
        "is_file",
        "is_dir",
        "mkdir",
        "touch",
    }
    imported: set[str] = set()
    impure_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if name in banned_calls:
                impure_calls.add(name)
    assert not (imported & banned), (
        f"wizard.py imports I/O modules: {imported & banned}"
    )
    assert not impure_calls, f"wizard.py performs file I/O: {impure_calls}"
