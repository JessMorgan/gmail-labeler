# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
does not ship releases yet.

## Unreleased

### Added

- Per-label threshold calibration (ticket #21): the pure core
  (`thresholds.py`, pyright strict) turns the eval metric table into
  per-label Threshold Bands — "confident" becomes a measured per-label
  claim, never a global constant. The band-selection rule (pinned by
  golden tests): `high` = the HIGHEST candidate threshold whose
  measured precision is defined (tp+fp > 0) and meets the label's
  min-precision target (`>=` — a cell exactly at the target earns the
  band); `med` = the highest qualifier STRICTLY below `high` when high
  is earned (an empty/inverted med band is meaningless; unconstrained
  when high is not earned); a band that is not earned is an explicit
  `none` with a named reason, never a silent zero (the strict-below-
  high constraint gets its own reason name — a reason must not lie).
  The run's AND every per-label target get the same validation
  (out-of-range, bool, and non-tenth targets refuse). Calibration reads
  tp/fp/fn and derives the ratio itself — a zero-denominator cell
  (tp=0 AND fp=0, no predictions) has UNDEFINED precision and never
  meets any target, not even 0.0 (the eval's 0.0 rendering is not a
  measurement). Refusals: non-canonical vocabulary, a candidate grid
  that is empty/out-of-range/non-ascending/non-tenth (the eval stance
  inherited), misaligned rows/cells, a target outside [0, 1] (a bool
  is not a target), a non-tenth target (the tenth-grid stance, #20),
  and an override for a label the table does not carry (named).
  Deterministic total order (same table → same bands; input-order
  invariant).
- Bands Artifact (`thresholds.py`): the versioned JSON artifact
  (format version + content hash over every field — the
  `dataset_build.SplitManifest` discipline: canonical dump, UTC
  datetimes, `validate()` from BOTH entry paths) binding the
  calibrated bands to the checkpoint identity (vocabulary hash, corpus
  fingerprint, split manifest hash, encoder id — the Checkpoint
  Sidecar's fields, mirrored) and the eval identity (split date, test
  example count), carrying the candidate grid + default targets + the
  per-label effective targets (provenance — an override is visible).
  `verify_bands` REFUSES an artifact that disagrees with the
  checkpoint/eval it is served against — every field checked
  independently, all disagreements named in one `BandsRefusedError`
  (the sidecar doctrine, AC2). Same eval → byte-identical artifact
  (AC5, pinned across re-runs and a reordered record index).
- `eval --bands PATH` (ticket #21): the deliberate act that calibrates
  the per-label Threshold Bands from the measured table and writes the
  artifact — opt-in (human-gated, ADR-0005: writing/updating the
  serving bands is a decision, never a side effect of a report run; a
  bare `eval` writes nothing), 0600 from creation, staging + atomic
  replace, replacing an existing artifact announced on stderr (the
  #20 `--out` NOTICE), the report rendered before calibration and the
  write (never lost — even a calibration refusal leaves the report on
  stdout), and a one-line human summary on stderr in text mode (the
  `--out` file stays the same bytes as stdout). Usage refusals up
  front (exit 2): an empty path; `--bands` and `--out` naming the same
  file (the artifact write would destroy the report); `--bands`
  naming an existing directory; `--bands` naming the config file; and
  ANY `--bands` target whose resolved path lies inside the checkpoint
  dir (manifest, sidecar, weights) or the corpus store (records
  index, labels/filters sidecars, bodies) — the artifact must not
  live beside identity files; the canonical location lands with the
  daemon phase, any other path works. The recompute posture is
  pinned: eval remains the only command that writes the artifact — a
  structural source-scan tripwire against named mints (a tripwire,
  not a sandbox: getattr-construction and a hand-rolled dict evade
  any string scan) + the behavioral pins (AC4). Exit-code contract
  preserved (config error → 4; run-time refusals → 5, the report
  preserved on stdout: a degraded corpus — test records with missing
  body shards, scored subject-only — because the artifact feeds the
  daemon's serving gate; a calibration refusal — an unknown override
  label, a lying identity; replacing an existing artifact whose
  identity fields disagree with the current run — refused BEFORE any
  staging, the previous bytes intact; a write failure or a
  post-write readback mismatch, i.e. a torn write).
- Config `[bands]` (ticket #21): the run's default min-precision
  targets (`min_high_precision` 0.9 / `min_med_precision` 0.7) +
  per-label overrides as `[bands."<label>"]` subtables (label names
  may carry slashes — quoted TOML keys). The minimal posture: only
  min-precision TARGETS are overridable — an explicit band threshold
  would let the operator set a threshold no measured cell meets, and
  "confident" would stop being a measured claim; the operator changes
  what "confident" means, the measured table picks the threshold.
  Overrides REFUSE unknown keys, scalars under a label name (named as
  unknown/typo'd keys), out-of-range values, non-tenth values (the
  tenth-grid stance, #20), TOML bools AND strings (pydantic's lax
  float coercion would silently turn `true` into 1.0 / `"0.9"` into
  0.9 — refused on the raw value, named), empty tables (a no-op
  override is refused, not dropped), a raw `per_label` key (a
  reserved field name — the label-path spelling is the only
  per-label form; the declared field is the typed programmatic API
  and gets the same value checks), blank/over-length label keys
  (keys are stripped — a padded key must not survive to
  calibration), and the same label named more than once (refused,
  not silently shadowed); every problem is reported together (a
  single raw-seeing validator — a field-constraint failure can no
  longer skip the collection); an override for a label the
  checkpoint's vocabulary does not carry refuses the run, named
  (fail loud). The config carries no path (the artifact path is
  the CLI's `--bands` argument), so the config-file relative-path rule
  has nothing to anchor here. GLOSSARY gains the Bands Artifact term.
- `dataset export` progress: opt-in `-v` prints per-page summaries and
  `-vv` per-message outcome lines to STDERR, fed by engine callbacks
  (`on_page`/`on_message`) that hand the CLI frozen `ProgressSnapshot`s —
  the engine stays I/O-free, a broken progress pipe closes only the stderr
  channel (never the run), the default run is byte-identical, and #62's
  progress bar is specified to consume the same seam (ticket #61).
- Date Provenance on the record contract (`CorpusRecord.date_provenance`):
  export stamps server `internalDate` vs MIME `Date`-header fallback
  distinctly, GYB import stamps its own source; pre-existing lines lacking
  the field read as explicit `unknown` (append-only corpus, never rewritten).
  Temporal-split consumers can now filter on the flag (invariant #7).

- Phase 0 (corpus) scaffold and CI: pyproject/uv packaging, `gmail-labeler`
  CLI entry point, GitHub Actions workflow (ubuntu + macOS, Python
  3.12/3.13; matrix later re-shaped to floor + latest — see Changed),
  `scripts/local-ci.sh` pre-push gate.
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
  record; a keep is counted in `deleted_kept` only when the refresh get
  confirms the 404 (the get decides, never the event type — a failed get is
  failure accounting alone); non-resident event ids are counted as Unseen Ids,
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
  stored cursor (a kept-high Sync Cursor would be unreachable; replay
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
- Composition flags (ticket #75, spec #64): `dataset sync-labels
  --also-import-gmail` runs the full Gmail Import pass after the Label
  Sync delta is durable (Sync Cursor advanced); `dataset import-gmail
  --also-sync-labels` runs the Label Sync event walk after the import
  completes durably. Opt-in by design — without the flags both commands
  keep their standalone contracts byte-identically (pinned). One report
  carries both passes' stats; each second pass delegates to the landed
  core (no re-implementation — ingest stays single-sourced, and the
  composite sync pass is the event walk only, never an implicit
  `--full-resync`); a second-pass failure exits non-zero per the pinned
  taxonomy (4 auth, 5 failed — the expired-cursor guidance contract
  included) and never rolls back the first pass's durable records or
  cursor; a failed first pass never starts the second. `--dry-run`
  refuses `--also-import-gmail` as a usage error (`2`): dry-run writes
  nothing, the import pass writes records.
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
- `train` command (ticket #19): fine-tunes the encoder on the corpus,
  fully local (AC#6). The run mints the Split Manifest, rebuilds the
  examples THROUGH it (manifest mode — the manifest is authoritative,
  invariant #7; the split boundary uses only server-authoritative
  dates; the rebuild-via-manifest call is pinned by a `build_dataset`
  spy), trains behind the Model seam (the recipe is reached through a
  monkeypatchable factory — the offline tests drive `FakeModel`; the
  factory's return is consumed: a trainer vocabulary that does not
  match the manifest's refuses before any sidecar is written),
  materializes the Split Manifest (`manifest.json`) with the
  checkpoint, and writes the JSON Checkpoint Sidecar pinning the
  run's identity (train-split scoped corpus fingerprint).
  `--split-date` (required, strict ISO DATE — time components are a
  usage error → 2) pins the temporal split; `--device cpu|cuda`
  (default cpu) is plumbed to the recipe; `--assume-no-filters`
  attests a filterless mailbox (the builder refuses a
  missing/incomplete filters/labels bridge — invariant #6); the
  existing-checkpoint refusal is hoisted BEFORE the two corpus-scale
  build passes and REFUSES the run until `--force` is explicit
  (retraining is human-gated, ADR-0005). `--force` replaces
  staging-based: build in `<dir>.tmp` (0700 before training), verify
  the sidecar by readback, rmtree the old dir, rename in — a crash
  leaves the OLD checkpoint intact, an EMPTY dir, or the NEW one,
  each refusal-safe, and no stale encoder file survives into the new
  dir. The train text is `corpus_record.preprocess_text` (invariant
  #4: train and inference share one preprocess function); missing
  body shards warn with a count, an undecodable shard is exit 5. A
  missing `train` extra (torch) is exit 5 with an install hint, not a
  traceback (the recipe exception boundary is a documented decision:
  ImportError/ValueError/OSError/RuntimeError → 5; anything else is a
  code bug → traceback/exit 1). The dead `_stub`/`_NOT_IMPLEMENTED`
  surface is removed (exit 3 stays documented as currently
  unreachable); `store.verify_modes()` runs on the read path. Config
  gains `[model]` (`encoder`, `encoder_path` — local weights only,
  `checkpoint_dir`); exit-code contract 0/2/3/4/5 preserved (bad
  `--split-date`/`--device` are usage errors → 2; build refusal,
  unreadable corpus, recipe failure, and the checkpoint refusal are
  5).
- `train/` recipe (ticket #19): HF `Trainer` fine-tune over the
  builder's Training Examples — one sigmoid head per Label Path
  (multi-label BCE), the recency weight applied as a WEIGHTED MEAN of
  per-example BCE losses from `outputs.logits` (`reduction='none'`,
  `.mean(dim=-1)`, `(per*w).sum()/w.sum()` — the model's scalar
  `outputs.loss` would make the weight mathematically inert),
  LAZY tokenization (the dataset holds raw texts; the collator
  tokenizes per batch — peak memory O(batch), not O(corpus)),
  single-device contract (`use_cpu=True` on CPU; a multi-GPU host is
  refused on the GPU path), `load_hf_model` refuses a checkpoint whose
  `num_labels` does not match the served vocabulary, encoder weights
  from a LOCAL path only (`local_files_only` everywhere — pinned by
  spy, AC#6), checkpoint saved as a plain HF-style directory with the
  `encoder_id` recorded on the HF config. The ONLY code that imports
  torch/transformers, reached exclusively through lazy imports;
  `torch` + `transformers` (v5 line) + `accelerate` (Trainer
  hard-requires it) are the optional `train` extra. CPU-smoke tests
  (incl. non-uniform-weight order-sensitivity, collator row
  alignment, and the `local_files_only` pin) and one GPU test sit
  behind the opt-in `smoke`/`gpu` markers, excluded from every
  CI/local-ci run (AC#1/AC#2: the matrix stays torch-free and green);
  the README carries the CPU-only torch install hint
  (`uv sync --extra train --torch-backend=cpu`).
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
- `eval` command (ticket #20): scores the temporal test split and
  reports per-label precision/recall at candidate thresholds — the
  measured table Threshold Bands and Graduation both read (GLOSSARY:
  Eval Report). The checkpoint is sidecar-validated through
  `load_model` (a mismatched checkpoint refuses with exit 5, naming
  every disagreeing field — no silent fallback); the test-side
  membership is rebuilt THROUGH the checkpoint's materialized Split
  Manifest (invariant #7 — the split is pinned, never re-derived, so
  there is no `--split-date`); predictions run through the Model seam
  (offline tests drive `FakeModel` behind a monkeypatchable factory);
  the text is `corpus_record.preprocess_text` (invariant #4, the same
  function train used). The split-boundary trust filter is a
  split-identity decision owned by the manifest — eval inherits it,
  never re-applies it (the report states it). The pure core
  (`eval.py`) scores in-memory examples with scores aligned by
  EXAMPLE ID (a missing score, a lying vector, a misaligned row, a
  non-canonical vocabulary, a label outside the vocabulary, or a bad
  threshold grid refuses with `ValueError`); the candidate-threshold
  grid is a fixed named constant (tenths, `DEFAULT_THRESHOLDS` — #21's
  calibration supersedes it); a non-tenth grid refuses (the text
  report labels columns to one decimal — `0.11` would render `0.1`);
  a score exactly at the threshold is a positive; zero denominators
  are 0.0, never a fabricated 1.0. The report is deterministic and
  diff-friendly (per-label rows x threshold columns with P/R +
  support; same inputs → byte-identical), opened by a split-
  provenance header (split date + manifest hash short form — a saved
  `.txt` is never indistinguishable from a stale one) and closed by a
  convention footer naming the zero-denominator rendering, to stdout
  by default, `--out PATH` for the same bytes in a file (0600 from
  creation; replacing an existing file is announced on stderr — no
  silent clobber), and `--json` for the machine-readable payload
  (graduation workflows). Message bodies never enter the report, the
  `--out` file, the `--json` payload, or any log path (invariant #5,
  pinned by content + caplog tests). Exit-code contract 0/2/3/4/5
  preserved (config error → 4; corpus/build/checkpoint refusal, empty
  test split, unreadable shard, missing `train` extra, and factory
  failure → 5).
- `eval` metric-table core (`eval.py`): the pure-core half of ticket
  #20 — `eval_metrics` (in-memory examples + hand-authored score
  vectors → the deterministic metric table; refuses non-tenth
  threshold grids and labels outside the vocabulary) and
  `render_report` (byte-stable text with the split-provenance header
  and the zero-denominator convention footer; validates its argument).
  Golden metric-table tests with hand-authored score vectors (the
  `test_analyze` pattern), incl. the `>=`-threshold boundary,
  zero-support labels, input-order invariance, the alignment-by-id
  pin, and every refusal. GLOSSARY gains the Eval Report term. Joins
  the pyright strict list.
- Design finalized: ADRs 0001–0005, GLOSSARY, CODING-STANDARDS, AGENTS.
- Project scaffolded; implementation begins at Phase 0 (export + analysis).
- Design finalized: ADRs 0001–0005 (initial set), GLOSSARY,
  CODING-STANDARDS, AGENTS.
- Project scaffolded; implementation begins at Phase 0 (import + analysis).
- Design: ADR-0007 (the Corpus is a revision log) and ADR-0008 (the Sync
  Cursor is sync's ledger) for spec #64 (Label Sync), with glossary terms
  Backfill, Label Sync, Sync Cursor, Revision, Resolved view, and Compaction.
- `docs/ROADMAP.md` — the phase plan of record (Phase 0 remnants →
  Phase 1 → Phase 2 → explicitly-parked Phase 3); the README's
  phase-plan pointer now links there.

### Changed

- CI matrix: floor + latest. Minimum supported (3.12) runs on ubuntu only;
  latest (3.15) runs on ubuntu and macOS. Drops the macOS 3.12 cell (runner
  queue pressure) and replaces 3.13 with 3.15 as the tested latest; 3.13 and
  3.14 remain allowed by `requires-python >=3.12`, just untested.
- `HfModel.predict` is batched (ticket #20, #19's deferral): eval is
  the first bulk caller, and a single pass over a large test split
  would pad every text to the longest one — token memory
  O(corpus x max length), the exact hazard the lazy per-batch
  collator exists to avoid for training. predict now tokenizes in
  `_PREDICT_BATCH_SIZE` (32 — the token budget, ~32 x 512 tokens per
  batch — named in one place) chunks, transfers each batch to the
  host once (`cpu().tolist()`), and concatenates the sigmoid rows;
  row i still corresponds to text i (pinned by smoke tests that force
  a small batch size AND cross the real default 32 over 33 texts).
- Identity/report artifacts are written 0600 FROM CREATION
  (`cli._write_secret_tight`: `os.open(..., 0o600)` + `fchmod`), not
  write-then-`chmod` — the microsecond 0644 window is closed. Covers
  the train manifest/sidecar writes and the eval `--out` report.
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
- Spec #64 pre-merge fold-in (PR #88 round 2): `--full-resync` now prints
  the cost estimate and strategy BEFORE reading the profile seed (US8's
  "before spending any quota beyond one `labels.list` call" order; the
  seed still precedes all enumeration, and the seed-failed report now
  names the chosen strategy where it once rendered `strategy=None`, with
  the flag-rate suppressed only for sentinel-failed runs whose scan
  never enumerated — a completed scan keeps its measured rate); the
  Sync Cursor `_Avoid_` list is honored in the step-down NOTEs and the
  ADR/CHANGELOG prose that said "kept-high watermark"; GLOSSARY gains the
  report-surfaced **Stale Id** term; and ADR-0008 records that the list
  response carries no page-level `historyId` in the current wire
  contract, superseding spec #64's seed-source sentence.
- `dataset import-gyb` rejects DRAFT/CHAT rows at ingest (maintainer decision
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
  so the two layers cannot drift; `dataset import-gyb` reports the
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
  path) plus a full re-ingest — a GYB re-import, or a fresh
  Gmail Import for API-sourced corpora.
- `dataset import-gmail` scope (now recorded as Corpus Exclusion) excludes
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

- `ParsedMail` no longer carries the subject or the canonicalized body in
  its `repr()`: repr-into-a-log is a log channel, and both fields are
  message content — the fields stay on the value (the importers read them
  by name; equality and construction are unaffected), only the rendering
  drops them, mirroring the `RawMessage` pin (#78, latent invariant-#5
  channel).
- `dataset sync-labels --dry-run` (event walk and `--full-resync`) no
  longer closes with "run `dataset analyze` for the updated candidate
  gates" — a plan-only run moves no gates, so dry-run now closes with a
  plan-only twin ("this dry run wrote nothing — re-run the same command
  without --dry-run to apply the plan, then `dataset analyze` …"),
  mirroring the WOULD-birth split; both modes are pinned by CLI tests
  (#77, #64 fix-round deferral).
- `dataset sync-labels --dry-run` on a cursorless corpus no longer prints
  "it was birthed from the mailbox profile" — dry-run writes nothing, so the
  event-walk report now says WOULD birth, mirroring the
  `--full-resync` branch's existing WOULD/actual split; both render modes
  are pinned by CLI tests (#64 fix round).
- `RawMessage` no longer carries the base64 message body in its `repr()`:
  repr-into-a-log is a log channel, and invariant #5 keeps bodies out of
  logs — the field stays on the value (the import parses it), only the
  rendering drops it (#71 deferral, closed in the #64 fix round).
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
