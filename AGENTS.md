# AGENTS.md — project context for fresh agent sessions

`gmail-labeler`: a strictly-local, public, general-purpose tool that
fine-tunes a multi-label classifier on the user's own Gmail history and applies
labels to arriving mail as it lands. Anyone can use it on their own inbox;
nothing leaves the user's machines; nothing is centralized on the maintainer.
Docs live where you expect: `GLOSSARY.md`, `CODING-STANDARDS.md`,
`docs/adr/`, and `docs/ROADMAP.md` (the phase plan of record).
Phase 0 (corpus) lands with the open PR stack; the layout and command
sections below mark what is wired vs. planned, and each gains detail as
its phase lands.

## Design spine (read the ADRs before questioning these)

- ADR-0001 multi-label encoder + sigmoid heads — **not** a Kev-style
  single-choice fine-tune, despite the sibling-project lineage.
- ADR-0002 pure Python, single process, model loaded in-process; no Go, no
  HTTP serving layer.
- ADR-0003 Gmail History-API polling (default 30s) is the sync spine; Pub/Sub
  push is a deferred trigger, not a redesign.
- ADR-0004 each user brings their own GCP project + desktop OAuth client
  (production mode, not testing); the README Setup section
  guides today's flow — the interactive `auth` wizard ADR-0004 sketches is not
  built yet (see its status note).
- ADR-0005 additive-only writes; Shadow Pass until per-label Graduation;
  human-gated retrain/threshold changes.

## Layout

- `src/gmail_labeler/` — landed: `config`, `labels`, `corpus_record`,
  `corpus_store`, `mime_extract`, `analyze`, `auth`, `tokens`,
  `gmail_client`, `gmail_errors`, `fake_gmail`, `export`, `gyb_import`,
  `cli`. Planned (later phases): model core (`dataset` build,
  `thresholds`, `graduation`, `eval`) + `model` and `daemon` adapters.
- `train/` — planned: fine-tune recipe (HF `Trainer`), checkpoint output
  convention.
- `docs/adr/`, `docs/ROADMAP.md`, `GLOSSARY.md`, `CODING-STANDARDS.md`,
  `example.toml`.
- CLI surface: `auth` (status) + `auth login` and
  `dataset {export,import-gyb,analyze}` are wired; `train`, `eval`,
  `shadow {run,report}`, `apply`, `daemon`, `status` are frozen by the
  design conversation, not yet wired — change names with cause.

## Commands

```sh
scripts/local-ci.sh    # pre-push gate: uv.lock freshness + all CI gates
uv run pytest          # -q --cov -m 'not live' come from pyproject addopts
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

The coverage gate (`fail_under = 80`, scope `src/gmail_labeler`) and the
live-test exclusion live in `pyproject.toml`
(`[tool.pytest.ini_options]`, `[tool.coverage.*]`), so a bare
`uv run pytest` cannot bypass either.

CI parity (`.github/workflows/ci.yml`: ubuntu + macOS, Python 3.12/3.13)
is the merge gate; `--live` Gmail tests never run there (invariant #9).

## Git management — project inputs

- Dedicated worktrees only (global `git-worktree` skill); review gates per
  global `git-management` skill.
- Keep in sync with every change: README quickstart, `docs/`, `example.toml`,
  `GLOSSARY.md` (new domain terms land there *in the same commit*), and ADRs —
  overriding a decided shape means superseding its ADR.
- License is AGPL-3.0-or-later; keep the header policy stated in
  `CODING-STANDARDS.md`.

### Agent GitHub identity (maintainer box)

All agent activity on GitHub happens as the `jm-coding-agent[bot]` GitHub
App installation — never via the owner's personal `gh` login. This
subsection documents maintainer-machine plumbing, not product behavior;
contributor machines are configured independently.

- **Commit identity**: commits must carry
  `coding-agent[bot] <350299513+jm-coding-agent[bot]@users.noreply.github.com>`.
  That pin is machine-local git config (not version-controlled); this
  repo's local config on the maintainer box is already set to it. The
  box's *global* identity is the owner's personal one — if
  `git config --local user.name` in a checkout is not
  `coding-agent[bot]`, run
  `git config --local user.name coding-agent[bot]` and
  `git config --local user.email 350299513+jm-coding-agent[bot]@users.noreply.github.com`
  before committing; never let commits fall back to the global
  identity.
- **Auth lifecycle (gh + git)**: one installation token at a time,
  kept in the `gh` login store — nothing sits in the remote URL.
  Before any GitHub call, check validity; re-mint only when needed:

  ```sh
  if ! gh api rate_limit >/dev/null 2>&1; then
    generate-github-token.sh | gh auth login --with-token
  fi
  ```

  Then plain `gh …` runs as the bot. The mint script (on `PATH`,
  `~/.local/bin/generate-github-token.sh`) prints only a fresh
  `ghs_…` token on stdout and on any failure exits non-zero with
  empty stdout, so the loop cannot store garbage (`gh auth login`
  also validates the token before saving it). Tokens live ~1 h; the
  check-first loop keeps mint requests to about one per hour. The
  token acts as the *integration*, not a user: `gh api user` answers
  403 "Resource not accessible by integration" by design; prefer
  plain `gh api …` calls over user-scoped `gh` subcommands. The
  script needs `python3` + `PyJWT`, `curl`, and `jq`; `APP_ID`,
  `INSTALLATION_ID`, `PRIVATE_KEY_PATH`, `API_BASE` are
  env-overridable — on key rotation, install the new key and update
  the default `PRIVATE_KEY_PATH`.
- **`git push`/`fetch`**: authenticate through the repo-local
  credential helper `!gh auth git-credential` — no token in the
  remote URL. A fresh checkout needs all of:

  ```sh
  git remote set-url origin https://github.com/JessMorgan/gmail-labeler.git
  git config --local --unset-all credential.helper
  git config --local --add credential.helper ''
  git config --local --add credential.helper '!gh auth git-credential'
  ```

  The empty-string entry matters: it clears the box's global
  `credential.helper=oauth`, which would otherwise try to open a
  browser for the owner's personal account mid-push. If git then
  reports an auth error, the stored token expired — rerun the
  validity loop above and retry.
- **Live-token redaction**: `gh auth token`, `gh auth status -t`, and
  `~/.config/gh/hosts.yml` expose live tokens — never paste their
  output into logs, commits, review reports, or PR text.

## Invariants (seed gotchas; expand the list as code lands)

1. **Additive-only**: the only Gmail write is `addLabelIds`; no code path may
   remove labels, ever (ADR-0005). Fail-safe: any application-path error means
   *no write at all*.
2. **Skip-present**: predicted label already present → not written; no new
   label added → no Application, no Tier Labels. This also makes crash-retry
   idempotent — preserve that property (CODING-STANDARDS).
3. **Cursor durability**: persist `historyId` only after the delta's events
   are durably handled; replay must be harmless.
4. **Preprocessing integrity**: train and infer share one `preprocess`
   function; any format divergence is the worst class of bug here.
5. **Bodies stay out**: message bodies never enter logs or the SQLite DB;
   IDs/probabilities/labels only. The DB must be publicly attachable.
6. **Filter-Authored Labels are not training signal**: `dataset analyze`
   attributes them via `users.settings.filters.list`; a Managed Label earns
   Graduation only on what filters don't cover.
7. **Temporal split only**: train/test by message date; a random split across
   near-duplicate template mail is fabricated accuracy.
8. **Secrets by path/env-name** in config; tokens in keyring (scoped
   exception: the ~1 h bot installation token in the machine-local gh
   login store, see "Agent GitHub identity (maintainer box)"); no
   token values in logs.
9. **No live API in CI**; the `FakeGmailClient` is the test spine.

## Domain docs & skills

- Single context: root `GLOSSARY.md` + `docs/adr/`. The `domain-modeling`
  skill discipline is active in design sessions — challenge terms that drift
  from the glossary.
- Project-local skills in `.agents/skills/` (`tdd`, `grilling`,
  `domain-modeling`, `code-review`, `diagnosing-bugs`, `research`, `wizard`,
  `pr`, ...). TypeScript-only skills from the auto-github set were deliberately
  not carried over.

## Known external facts

- GYB backup shape the importer codes against (verify any refinement against
  a real backup before tightening; the maintainer keeps a verified probe
  summary in `AGENTS.local.md` in the maintainer's **primary checkout**
  (gitignored; structurally absent from worktrees)):
  `<year>/<month>/<day>/<hexid>.eml` under the backup root where `hexid`
  **is the Gmail API message id**, plus `msg-db.sqlite` — read-only — with
  `messages(message_num, message_filename, message_internaldate)`,
  `labels(message_num, label)`, `uids(message_num, uid=hexid)`: the label
  map joins straight onto Gmail API ids, no .eml header parsing needed.
  System-label noise (`INBOX`, `UNREAD`, `SENT`, `CATEGORY_*`) must be
  filtered; slash-nested user labels are label *paths*. Decade-scale corpus
  spans make `dataset analyze`'s recency gates load-bearing, and temporal
  splits alone would over-weight the oldest mail.
- The importer must never depend on any user's backup existing, and backup
  freshness vs the API export must be reconciled at import time.
- The maintainer's runtime box is a homelab host (CPU inference; a GPU host
  appears only for `train`); topology details live in `AGENTS.local.md`.
