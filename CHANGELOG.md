# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
does not ship releases yet.

## Unreleased

### Added

- Date Provenance on the record contract (`CorpusRecord.date_provenance`):
  export stamps server `internalDate` vs MIME `Date`-header fallback
  distinctly, GYB import stamps its own source; pre-existing lines lacking
  the field read as explicit `unknown` (append-only corpus, never rewritten).
  Temporal-split consumers can now filter on the flag (invariant #7).

- Phase 0 (corpus) scaffold and CI: pyproject/uv packaging, `gmail-labeler`
  CLI entry point, GitHub Actions workflow (ubuntu + macOS, Python
  3.12/3.13), `scripts/local-ci.sh` pre-push gate.
- Config (`config.py`): TOML discovery, config-dir-relative path anchoring,
  typed all-problems validation; `example.toml`.
- Label vocabulary (`labels.py`): System Label filtering, label paths,
  id→name normalization.
- Corpus store (`corpus_store.py`): append-only records + sharded bodies,
  enforced 0600/0700 modes, crash-tail handling.
- OAuth (`auth.py`, `tokens.py`): desktop-app consent flow, keyring/file
  token stores, `auth` / `auth login` commands.
- Shared MIME parsing (`mime_extract.py`) for both corpus sources.
- Gmail REST client (`gmail_client.py`) + `FakeGmailClient` test spine and
  typed error taxonomy (`gmail_errors.py`).
- `dataset export`: resumable API export, `--query`/`--max-messages`,
  `filters.json` + `labels.json` sidecars.
- `dataset import-gyb`: offline backup import (read-only `msg-db.sqlite`),
  folder-date correction, `--system-labels`.
- `dataset analyze`: per-label counts, recency, duplicate rate, filter
  attribution, candidate shortlist, `--json`.
- Design finalized: ADRs 0001–0005, GLOSSARY, CODING-STANDARDS, AGENTS.
- Project scaffolded; implementation begins at Phase 0 (export + analysis).
- `docs/ROADMAP.md` — the phase plan of record (Phase 0 remnants →
  Phase 1 → Phase 2 → explicitly-parked Phase 3); the README's
  phase-plan pointer now links there.

### Changed

- GYB import rejects DRAFT/CHAT rows at ingest (maintainer decision
  2026-11-02, ticket #32, completing #17 corpus-wide): a backup row
  whose raw label tokens carry the canonical token DRAFT or CHAT is
  counted `skipped_draft_chat` and never stored (a row already in the
  corpus answers `skipped_present`; a not-yet-stored row whose
  backup-index path escapes the backup root aborts the run, exit 5,
  uncounted) — previously
  the token was stripped from the row's label set and the
  contaminated row imported silently. The exclusion pair now has one
  shared home (`labels.EXCLUDED_CORPUS_LABEL_IDS`) that both ingest
  layers import,
  so export and import cannot drift; `dataset import-gyb` reports the
  count in its summary line.
  Upgrade consequence: corpora assembled before the pair landed keep
  their DRAFT/CHAT-contaminated records — anything exported before #17
  (2026-11-01) may carry it (an explicit `--query` may already have
  dropped drafts, but chats had no reliable exclusion), anything
  GYB-imported before #32 carries it, and the store never deletes in
  place. Resume runs report those ids as
  `skipped_present` (skip-present wins), never `skipped_draft_chat`, so
  pre-existing contamination stays invisible to the counter; remediation
  is a fresh `corpus.dir` (the corpus store is rooted at exactly that
  path) plus a full re-ingest — a GYB re-import, or a re-export for
  API-sourced corpora.
- `dataset export` scope (now recorded as Corpus Exclusion) excludes
  DRAFT and CHAT (maintainer decision
  2026-11-01, ticket #17): the default query now carries the documented
  `-in:draft` token only, and the ingest layer hard-skips any message
  whose raw wire labelIds contain the system label id DRAFT or CHAT —
  no documented chat query token exists (is:chat retired; -in:chat
  undocumented, with a word-degradation hazard), so chat exclusion
  lives in code, where an explicit `--query` override cannot bypass
  it. With Additive-only + skip-present and the append-only corpus,
  the old implicit inclusion would be a permanent one-way door.

### Fixed

- `auth login` on headless boxes: a missing browser no longer aborts the
  consent flow before the authorization URL is printed — the flow retries
  browser-less and renders the URL with an `ssh -L` port-forward recipe for
  completing consent from another machine, on line-buffered stdout so it
  appears immediately even through `ssh box '… auth login'` or `| tee` (#53).
