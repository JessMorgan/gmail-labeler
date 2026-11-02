# Coding Standards

Python ≥ 3.12 (CI on 3.12 + 3.13). `uv` for environments, lockfiles, and
builds. Single package, `src/gmail_labeler/` layout. Format and lint with
`ruff` (`ruff format` + `ruff check`); both are gates, not suggestions.

## Architecture: pure core, bounded I/O

The classification-relevant logic is **pure**: dataset construction
(HTML/quote stripping, feature extraction, label mapping), threshold
application, graduation decisions, eval metrics, and the corpus → training-set
transform import nothing from the Gmail, filesystem, or config layers, do no
I/O, and are deterministic on any input. Everything that touches the world —
the Gmail API client, the GYB reader, the SQLite store, model load/infer — is
an adapter behind a `Protocol`, bounded (page sizes, body caps, timeouts), and
fakeable.

**Preprocessing integrity (the one non-negotiable):** training and inference
call the *same* `preprocess` function from the *same* package. Any feature of
the input format that exists in one and not the other is a bug of the highest
class — this is the drift auto-github needed a dedicated gotcha to police,
and living in one process removes the excuse.

## Vocabulary is normative

`GLOSSARY.md` terms are the code's terms: `ManagedLabel`, `TierLabel`,
`Application`, `Correction`, `Graduation`, `ShadowPass`, `FilterAuthoredLabel`.
The `_Avoid_` lists apply to identifiers as much as prose. If you need a
concept the glossary lacks, add it there in the same commit.

## The application path

- **Additive-only** (ADR-0005): the only Gmail write is `addLabelIds`.
- **Skip-present**: predicted labels already on the message are not written;
  if nothing new is added, there is no Application and no Tier Labels.
- **Fail-safe**: any error on the application path results in *no* labels —
  a never-labeling failure is always preferred to a wrong-label write.
- **Idempotent by construction**: the audit record is persisted after the
  Gmail write succeeds; a crash in between is recovered by the next poll, and
  skip-present makes the retry harmless. No application may depend on
  in-memory state.
- **Cursor discipline**: the `historyId` cursor persists only after all events
  of the poll's delta are durably handled; a replayed delta must be harmless.

## Secrets, privacy, logging

- OAuth refresh tokens live in the OS keyring; the documented file fallback is
  0600 with a startup warning. No secret or token value in any log line.
- **Message bodies never enter logs or the SQLite DB** — message IDs, label
  IDs, probabilities, and timestamps only. The DB must be safe to attach to a
  public bug report; a test asserts bodies cannot leak into log output.
- Config holds paths and env-var names, never secret values.

## Types

Full annotations everywhere; `pyright` in strict mode for the pure core and
basic for adapters. Value objects are frozen dataclasses; the config surface
is validated with `pydantic` (the one place a validation framework earns its
dependency); Gmail error responses map to a typed error taxonomy
(`transient` / `permanent` / `rate_limited`) so retry decisions are
deterministic, never string-matched at call sites.

## Tests

Test-first at the seams (see the project-local `tdd` skill). The Gmail API is
**always faked** — a `FakeGmailClient` behind the same `Protocol` powers unit
and integration tests alike; live-API tests carry a `--live` opt-in marker and
never run in CI. GYB layout parsing and the corpus transform are golden-tested
on fixture trees. Coverage: 80% overall gate, ~90% held on the pure core.
Model-load tests run on a tiny stub checkpoint, never a real download.

## Git and docs

Work happens in dedicated worktrees; per-commit review gates follow the
global `git-management` skill. Every commit keeps the docs honest:
README/docs/ surfaces, `GLOSSARY.md` (new terms land the moment they
crystallize), and an ADR whenever a change *overrides* one — supersede, never
silently contradict. License AGPL-3.0-or-later; a contribution is a license
acceptance, said in CONTRIBUTING.
