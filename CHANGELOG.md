# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
does not ship releases yet.

## Unreleased

### Added

- Phase 0 (corpus) scaffold and CI: pyproject/uv packaging, `gmail-labeler`
  CLI entry point, GitHub Actions workflow (ubuntu + macOS, Python
  3.12/3.13), `scripts/local-ci.sh` pre-push gate.
- Config (`config.py`): TOML discovery, config-dir-relative path anchoring,
  typed all-problems validation; `example.toml`.
- Label vocabulary (`labels.py`): System Label filtering, label paths,
  id→name normalization.
- Corpus store (`corpus_store.py`): append-only records + sharded bodies,
  enforced 0600/0700 modes, crash-tail handling.
- Corpus revision log (`corpus_store.py`, ADR-0007, ticket #70):
  `append_revision()` appends a full revised record for a resident id
  (deliberately separate from the never-duplicate `put()`),
  `iter_resolved()` is the resolved view (last-wins fold per id, a no-op
  on duplicate-free logs); `dataset analyze` consumes the resolved view,
  eliminating duplicate-id inflation of the volume, recency, and
  duplicate metrics — and the attribution-ratio deflation that could
  slip the filter-covered gate (invariant #6).
- OAuth (`auth.py`, `tokens.py`): desktop-app consent flow, keyring/file
  token stores, `auth` / `auth login` commands.
- Shared MIME parsing (`mime_extract.py`) for both corpus sources.
- Gmail REST client (`gmail_client.py`) + `FakeGmailClient` test spine and
  typed error taxonomy (`gmail_errors.py`).
- `dataset import-gmail`: resumable Gmail Import, `--query`/`--max-messages`,
  `filters.json` + `labels.json` sidecars.
- `dataset sync-labels` (ticket #73, spec #64): the Label Sync event walk.
  `label_sync.sync_labels(client, store, id_to_name, sleeper, clock,
  dry_run)` — a pure-core module the daemon will reuse — pages Gmail
  history from the Sync Cursor, marks every event-named corpus-resident
  id stale (refresh-on-any-touch), metadata-fetches each one (never raw —
  a call-log test proves the sync path is body-free, invariant #5) and
  appends an ADR-0007 Revision via `dataclasses.replace()` on the resolved
  record (field parity caller-owned, `id:` convention for now-unresolvable
  labels). The cursor advances only after the delta's revisions are durable
  and only when nothing failed (invariant #3 — crash-retry tests prove the
  last-wins fold absorbs replays). `messagesDeleted`-touched ids keep their
  record and are counted; non-resident event ids are counted as Unseen Ids,
  never ingested; a cursorless corpus births from one profile call and
  applies nothing (ADR-0008 consequence 1). The report prints per-label
  gains/losses and closes with the `dataset analyze` pointer; `--dry-run`
  prints the identical plan writing no records and leaving the cursor
  byte-identical; an expired cursor (history
  404) exits `5` with interim recovery guidance that names no unbuilt
  flag (finalized by ticket #74). The gmail_import-private retry loop is
  generalized into `retry.py` (shared pacing, duck-typed `retried`) and
  now re-raises `GmailResyncError` instead of swallowing it to a warning —
  silent success over a blind window was the bug.
- `dataset sync-labels --full-resync` (ticket #74, spec #64): ground-truth
  label reconciliation needing no history — the expired-cursor remedy and
  the first reconciliation path for GYB-sourced Corpora.
  `label_sync.full_resync(client, store, id_to_name, sleeper, clock,
  dry_run, progress)` (pure core, daemon-reusable) captures the profile
  seed BEFORE the scan and persists it only after the reconciliation's
  revisions are durable — a reseed, or a birth-when-absent on a
  cursorless corpus with the same ordering; the reseed is
  step-down-permitted when a rolled-back mailbox reports below the
  stored cursor (a kept-high watermark would be unreachable; replay
  from the lower seed is harmless under ADR-0007) (ADR-0008). ONE `labels.list`
  (per-label `messagesTotal`) plus local resolved-view counts yield the
  cost estimate, printed with the strategy choice before any enumeration
  spend; the estimator runs the cheaper of the label-inventory scan
  (enumerate every mailbox user label's members by label id via the new
  `list_messages(label_ids=...)` protocol extension — real adapter, fake,
  and the surface-table tripwires; diff against the Corpus's sets;
  metadata-fetch only flagged ids; vanished labels' corpus members fetched
  directly — the `id:` convention kept where unresolvable, and an id that
  resolved since import converging via one direct fetch; system-label-only
  churn flags nothing) or the corpus-metadata walk (metadata-get every
  corpus id, diff, revise only on drift); ties and an unpriceable scan
  (`messagesTotal` absent) go to the walk, whose number is exact. On the
  scan the report prints the flag-rate before the metadata pass and
  quotes the worst-case cost vs a from-scratch reload when a corpus-wide
  drift is flagged (the walk prices its full fetch up front instead);
  call-log tests
  keep both strategies body-free (invariant #5), failed runs hold the
  ledger and replay harmlessly (invariant #3), and `--dry-run` prints the
  estimate, choice, and plan with zero writes. The 404 guidance is
  finalized to ADR-0008's wording printing the exact
  `dataset sync-labels --full-resync` command (#73's vaporware-absence pin
  flipped to a presence pin).
- `dataset sync-labels --full-resync` step-down NOTE (ticket #95, PR #86
  follow-up closing the round-2 review's observability nit): the report
  prints one NOTE line when the reseed steps a higher stored cursor down
  (`FullResyncStats.stepped_down_from` carries the stored value; the CLI
  renders — data in the pure core, disclosure in the report, the
  birth-provenance precedent), and `--dry-run` shows the planned step-down;
  silent on birth, equal, and advance.
- Sync Cursor sidecar (`sync_cursor.py`, `corpus_store.py`, ADR-0008,
  ticket #72): `dataset import-gmail` births `sync-cursor.json`
  (`{version, history_id, last_sync}`, tz-aware UTC `last_sync`) inside the
  corpus root when the corpus has none — seeded before any message is
  ingested from the first page's newest stub, with one profile call as the
  fallback (a permanently unusable seed warns and leaves the corpus
  birth-eligible rather than failing the import). Birth-once: a live cursor
  is never advanced or overwritten by import (byte-identical across a
  second import), GYB Import never births, and an unparseable or invalid
  cursor fails loud (`CorpusError`) rather than silently re-birthing. Mode
  coverage rides a named protected-file set (attribution snapshots +
  cursor, deliberately split from the attribution-named tuple) driving
  both `enforce_modes` and `verify_modes`.
- `dataset import-gyb`: offline backup import (read-only `msg-db.sqlite`),
  folder-date correction, `--system-labels`.
- `dataset analyze`: per-label counts, recency, duplicate rate, filter
  attribution, candidate shortlist, `--json`.

- Design finalized: ADRs 0001–0005 (initial set), GLOSSARY, CODING-STANDARDS, AGENTS.
- Project scaffolded; implementation begins at Phase 0 (import + analysis).
- Design: ADR-0007 (the Corpus is a revision log) and ADR-0008 (the Sync
  Cursor is sync's ledger) for spec #64 (Label Sync), with glossary terms
  Backfill, Label Sync, Sync Cursor, Revision, and Resolved view.

### Changed

- Renamed `dataset export` to `dataset import-gmail` (ticket #68, ADR-0006):
  the acquisition commands speak from the Corpus's side, symmetric with
  `import-gyb`. Hard cut — no compat alias (pre-1.0); the old name exits 2
  as a usage error. The module (`gmail_import.py`), its entry function,
  stats class, CLI seam, test file, and the GLOSSARY **Gmail Import** term
  follow; stored corpus data is untouched (record `source` values and the
  on-disk layout are unchanged). `export` is reserved for a future
  corpus-dump command.
- The Label Sync report stat for non-resident event ids is named `unseen`
  (GLOSSARY: Unseen Id; ticket #73 review, renamed before any release):
  history events cannot establish that an absent id is new mail or that it
  matches the import query — a `messagesDeleted`-touched id that was never
  resident lands in this bucket — so the review-stage name `new_match`
  overstated what the count means.

### Fixed

- `dataset sync-labels --dry-run` no longer tightens corpus file modes:
  `corpus_store.enforce_modes` is WRITE-side-only per its own contract
  ("Call from WRITE-side commands… not from read-only tools"), so the dry
  run — which writes nothing — now skips it; the non-destructive
  `verify_modes` WARNINGs still run and name the structural paths the
  later write run would tighten (#73 review).
- `dataset import-gmail` no longer dies listing labels: the Gmail v1 discovery
  tree has only `users` at top level — `labels` live under it — and the
  REST client called `svc.labels()`, an `AttributeError` on the first live
  import-gmail run once #56 made construction survive. The `build_service` fake now
  mirrors the real tree, and a structural test captures each surface
  method's `HttpRequest` against the bundled offline discovery doc,
  pinning method + URI path + the exact query-param set (pageToken
  forwarding included) + the parsed result each canned response must
  yield. The case table must also cover the whole client
  Protocol, so path and query-param drift on the read surfaces fails in
  CI instead of on the user's inbox (#59).
- `dataset import-gmail` no longer dies on its first live call: the REST client's
  default transport now composes `AuthorizedHttp` around a timeout-bounded
  httplib2 and hands `build()` only `http` — google-api-python-client 2.x
  rejects `http`+`credentials` as mutually exclusive, which surfaced as a
  raw `ValueError` bypassing the typed error taxonomy (#56).
- `auth login` on headless boxes: a missing browser no longer aborts the
  consent flow before the authorization URL is printed — the flow retries
  browser-less and renders the URL with an `ssh -L` port-forward recipe for
  completing consent from another machine, on line-buffered stdout so it
  appears immediately even through `ssh box '… auth login'` or `| tee` (#53).
