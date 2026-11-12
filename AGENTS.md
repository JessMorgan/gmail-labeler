# AGENTS.md — project context for fresh agent sessions

`gmail-labeler`: a strictly-local, public, general-purpose tool that
fine-tunes a multi-label classifier on the user's own Gmail history and applies
labels to arriving mail as it lands. Anyone can use it on their own inbox;
nothing leaves the user's machines; nothing is centralized on the maintainer.
Docs live where you expect: `GLOSSARY.md`, `CODING-STANDARDS.md`, `docs/adr/`.
Phase 0 (corpus) has landed; the layout and command sections below mark
what is wired vs. planned, and each gains detail as its phase lands.

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
- ADR-0006 acquisition commands speak from the Corpus's perspective —
  `import-gmail`/`import-gyb`; `export` is reserved for a future corpus-dump
  command, never re-proposed for corpus population.

## Layout

- `src/gmail_labeler/` — landed: `config`, `labels`, `corpus_record`,
  `corpus_store`, `mime_extract`, `analyze`, `attribution`, `auth`, `tokens`,
  `gmail_client`, `gmail_errors`, `fake_gmail`, `gmail_import`, `gyb_import`,
  `cli`. Planned (later phases): model core (`dataset` build,
  `thresholds`, `graduation`, `eval`) + `model` and `daemon` adapters.
- `train/` — planned: fine-tune recipe (HF `Trainer`), checkpoint output
  convention.
- `docs/adr/`, `GLOSSARY.md`, `CODING-STANDARDS.md`, `example.toml`.
- CLI surface: `auth` (status) + `auth login` and
  `dataset {import-gmail,import-gyb,analyze}` are wired (the acquisition pair
  is pinned by ADR-0006; the old `export` name is a usage error, no alias);
  `train`, `eval`, `shadow {run,report}`, `apply`, `daemon`, `status` are
  frozen by the design conversation, not yet wired — change names with cause.

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

## Invariants (seed gotchas; expand the list as code lands)

1. **Additive-only**: the only Gmail write is `addLabelIds`; no code path may
   remove labels, ever (ADR-0005). Fail-safe: any application-path error means
   *no write at all*.
2. **Skip-present**: predicted label already present → not written; no new
   label added → no Application, no Tier Labels. This also makes crash-retry
   idempotent — preserve that property (CODING-STANDARDS).
3. **Cursor durability**: persist `historyId` only after the page's events are
   durably handled; replay must be harmless.
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
   login store); no token values in logs.
9. **No live API in CI**; the `FakeGmailClient` is the test spine.

## Domain docs & skills

- Single context: root `GLOSSARY.md` + `docs/adr/`. The `domain-modeling`
  skill discipline is active in design sessions — challenge terms that drift
  from the glossary.
- Skills come from the maintainer box's global skill set
  (`~/.config/opencode/skills/`); the project carries no local skill tree —
  do not re-create `.agents/`.

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
  freshness vs the Gmail Import must be reconciled at import time.
- The maintainer's runtime box is a homelab host (CPU inference; a GPU host
  appears only for `train`); topology details live in `AGENTS.local.md`.
