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
- Model seam (`model.py`): the `Model` Protocol — batch predict →
  per-label sigmoid scores over the checkpoint's Label-Path vocabulary
  (rows aligned to inputs, columns to the vocabulary; `texts` is an
  Iterable that implementations materialize once) plus vocabulary
  exposure — the ONE new test seam (spec #13). Conformance is
  structural (a consumer annotated `Model` is the pin; no
  `isinstance` anywhere). `FakeModel` is the offline spine
  (hand-authored score vectors, call log, refuses misaligned/out-of-
  range programming, RAISES on an unprogrammed text unless the caller
  passed an explicit `default` — no silent 0.5) mirroring
  `FakeGmailClient`'s role; `vocabulary_hash` is the ORDER-SENSITIVE
  sha256 identity the checkpoint sidecar carries (a reordered
  vocabulary must refuse). The sorted-unique helper now lives in
  `dataset_build.sorted_unique` (one home, shared). No torch import
  anywhere in the importable core — the real encoder stays behind the
  `train/` recipe (pinned: importing `gmail_labeler.model` leaves
  `torch` out of `sys.modules`).
- Phase 1 dataset builder core (`dataset_build.py`): the shared pure
  Corpus → Training Example transform — Label Path vocabulary (a leaf
  path is a distinct class; a parent may scope its subtree), examples
  sorted by id with canonical sorted label tuples and Date Provenance
  stamped, required `now`, input-order invariant.
- Dataset builder bridge (invariant #6): Filter-Authored evidence is
  excluded from training examples (per-message for simulatable filters,
  whole-label for unverifiable/malformed ones); a missing, incomplete,
  or malformed filters/labels snapshot aborts the build loudly
  (`BuildAbortedError`, a `ValueError`). The effective bridge is pinned
  in the Split Manifest (`bridge_hash`, canonical — insensitive to filter
  order and unreferenced label entries) so a drifted snapshot refuses the
  rebuild. The filter-shape rule is shared with `dataset analyze`
  (`filter_label_ids` / `filter_label_ids_strict`).
- Checkpoint identity + loader refusal (`model.py`,
  `corpus_record.py`): a checkpoint is a plain HF-style directory
  (gitignored) plus a JSON Checkpoint Sidecar — label-vocabulary
  hash, thresholds source (`none` until calibration exists; #21's
  artifact slots in without a format change), Corpus fingerprint
  (new `corpus_fingerprint`: sha256 of the canonical record lines of
  the TRAIN-SPLIT slice only, order-invariant — append-only corpus
  growth cannot invalidate a serving checkpoint, #25's lifeline),
  train-split manifest hash, encoder identifier. Hash-shaped fields
  are 64-hex validated at parse (empty strings refused).
  `load_model` REFUSES any checkpoint whose sidecar disagrees with the
  Corpus/config it is asked to serve against — each field refused
  independently, all disagreements named in one
  `CheckpointRefusedError` (a `ValueError`); the format version is
  checked FIRST with its own `sidecar-version` code (not conflated
  with `sidecar-corrupt`); the materialized `manifest.json` is
  verified against the sidecar (`manifest-missing` /
  `manifest-corrupt` / `split-manifest-mismatch` — a tampered
  manifest refuses). `corpus_record.preprocess_text` states its
  contract literally (`subject + "\n" + body`; an empty body ends
  with a trailing newline). Pinned offline with `tmp_path` fixtures +
  `FakeModel` (AC#3), incl. the append-only-growth serving test.
- Dataset builder recency weighting: each Training Example carries a
  recency weight (1.0 at/after the cutoff, `old_weight` before) mirroring
  analyze's `months_ago` window shape; the cutoff (UTC-normalized) and
  both constants ride in the Split Manifest (FORMAT_VERSION=3) so a
  21-year corpus is not dominated by its oldest mail and a manifest-input
  rebuild is reproducible from the artifact alone.

- Design finalized: ADRs 0001–0005, GLOSSARY, CODING-STANDARDS, AGENTS.
- Project scaffolded; implementation begins at Phase 0 (export + analysis).
- `docs/ROADMAP.md` — the phase plan of record (Phase 0 remnants →
  Phase 1 → Phase 2 → explicitly-parked Phase 3); the README's
  phase-plan pointer now links there.

### Changed

- `dataset export` scope excludes DRAFT and CHAT (maintainer decision
  2026-11-01, ticket #17): the default query now carries the documented
  `-in:draft` token only, and the ingest layer hard-skips any message
  whose raw wire labelIds contain the system label id DRAFT or CHAT —
  no documented chat query token exists (is:chat retired; -in:chat
  undocumented, with a word-degradation hazard), so chat exclusion
  lives in code, where an explicit `--query` override cannot bypass
  it. With Additive-only + skip-present and the append-only corpus,
  the old implicit inclusion would be a permanent one-way door.
