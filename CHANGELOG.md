# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
does not ship releases yet.

## Unreleased

### Added

- Shadow Pass (ticket #23): the Phase 1 exit — `shadow run` scores the
  recent Backlog through the sidecar-validated checkpoint, the
  Graduation Table, and the Threshold Bands and APPLIES NOTHING;
  `shadow report` diffs the stored would-apply verdicts against the
  mailbox's ACTUAL labels (the human go/no-go review before anything
  goes live, ADR-0005).
  - `tiers.py` (pure core, pyright strict): the SINGLE would-apply
    decision core. Tier Label constants pinned to the GLOSSARY
    spellings (`ai-applied` umbrella, `ai-high`, `ai-med`); band
    verdicts `score >= high -> high`, `>= med -> med` (inclusive `>=`
    per #21's threshold doctrine); graduated labels only (a label
    without a graduated entry is never proposed); skip-present (an
    already-present label is not proposed and earns NO Tier Label —
    invariant #2's posture, defined here so #24/#26 reuse it); System
    Labels never targeted (refused, named); deterministic total order.
    `best_tier` derives the per-message Tier Label from the applied
    decisions.
  - `[store] db_path` config: a PLAIN path field (no label-key surface
    — the heavy [bands]/[graduation] per-label doctrine does not
    apply), relative anchored to the config dir (the corpus/model dir
    rule), default `<config dir>/gmail-labeler.db`.
  - `appdb.py`: the SQLite state DB adapter — create-all-if-missing,
    `schema_version` row, the `shadow_results` table (run id, message
    id, label, probability, band, run_ts — NO bodies, invariant #5),
    transaction-per-run (a crash mid-run leaves no half rows), the #24
    applications-table seam (same file, same versioning story).
  - `shadow run` CLI: auth gate (exit 4) -> sidecar-validated
    checkpoint load (the eval `ExpectedIdentity` pattern) ->
    `verify_bands` AND `verify_graduation` against the run's identity
    (the FIRST real consumers of both verify_* helpers — the artifact
    identity gates become real; refusal exit 5 naming fields;
    unreadable-vs-unparseable-vs-absent cause naming per #22's
    doctrine) -> Backlog fetch (`list_messages` pageToken loop honoring
    `--limit` across pages; `q = in:inbox newer_than:YYYY/MM/DD` from
    `--since` default 90d, `--query` escape hatch) -> `get_message` per
    id -> `preprocess_text` (invariant #4) -> `model.predict` (batched
    seam) -> `tiers.would_apply` -> store. Degraded (body fetch
    failure) = per-message SKIP + stderr warning + count (the
    documented posture choice vs eval's refusal — shadow is a read-only
    survey). Summary to stderr, not stdout; `--json` one parseable
    object.
  - `shadow report` CLI: reads the run's rows from the state DB (latest
    run by default, `--run` to pick), re-fetches each message's ACTUAL
    labels via the read path, and renders a deterministic per-message
    diff (message id + label + score + tier ONLY — subjects OUT, bodies
    out by construction, invariant #5): ADDED / ALREADY-PRESENT
    (skip-present visible) / NOT-APPLIED outcomes + a per-label summary
    with the would-be tier mix. `--json` one parseable object.
  - Zero-Gmail-writes AC: `FakeGmailClient` call-recording pin asserts
    the recorded calls are a subset of the read surface
    (list_messages, get_message, list_labels, list_filters) — no write
    method of any shape (the Protocol has none in this ticket; the
    write surface lands #24/#25).
  - The `[labels]` applied/high/med config drives the report's tier
    names (a custom spelling is honored, not silently ignored —
    refuse-never-drop); `tiers.py` keeps the DEFAULT spellings as the
    pure-core constants, and the daemon (#24) resolves the configured
    names to Gmail label ids (the contract is recorded in the tiers
    module docstring).
  - A `--db PATH` flag is a usage-level override of `[store] db_path`
    (the opt-in live demo redirects the state DB to a tmp path).
  - The Backlog query is `in:inbox after:<epoch-seconds>` (the EPOCH
    form — unambiguous; calendar dates in API queries are interpreted
    as MIDNIGHT PST, and `newer_than:` is relative-duration only);
    `--since` → UTC midnight of that date → epoch; default → UTC now
    minus 90 days.
  - Exit-code contract at tip: 2 usage (empty/whitespace `--bands`/
    `--graduation`/`--db`, empty `--query`, bad `--since`, `--limit
    <= 0`, empty `--run`, `[store]` db_path inside the checkpoint dir
    or corpus store — containment); 4 auth; 5 operation
    (checkpoint/bands/graduation identity refusal, would-apply
    refusal, no-verdicts run, no-evidence report refusal, state DB
    ownership/corrupt/foreign-version refusal, DB failure, transport
    failure). Bodies-absent pinned across DB rows + report
    stdout/stderr + caplog.
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
  live beside identity files (by resolved path — a link named inside
  pointing outside is allowed and replaces the link); the canonical
  location lands with the daemon phase, any other path works. The
  recompute posture is pinned: eval remains the only command that
  writes the artifact — a structural source-scan tripwire against
  named mints (a tripwire, not a sandbox: getattr-construction and
  a hand-rolled dict evade any string scan) + the behavioral pins
  (AC4). Exit-code contract preserved (config error → 4; run-time
  refusals → 5, the report preserved on stdout: a degraded corpus —
  test records with missing body shards, scored subject-only —
  because the artifact feeds the daemon's serving gate; a
  calibration refusal — an unknown override label, a lying identity;
  replacing an existing artifact whose identity fields disagree with
  the current run — refused BEFORE any staging, the previous bytes
  intact; a write failure; a post-write readback fault — the
  on-disk bytes do not parse back to a valid artifact (a
  torn/corrupt write); or a readback mismatch — a valid artifact
  whose content hash differs from the minted one, a concurrent
  writer's signature).
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
  per-label form; an EMPTY raw table is the typed default and is
  accepted — configs WITHOUT per-label overrides round-trip
  through model_dump() -> model_validate, while a non-empty raw
  table stays refused and a populated dump is refused by design
  (the typed construction — BandOverride values — is the
  programmatic API); the declared field gets the same value
  checks), blank/over-length label keys
  (keys are stripped — a padded key must not survive to
  calibration), and the same label named more than once (refused,
  not silently shadowed); every problem is reported together (a
  single raw-seeing validator — a field-constraint failure can no
  longer skip the collection); an override for a label the
  checkpoint's vocabulary does not carry refuses the run, named
  (fail loud). The config carries no path (the artifact path is
  the CLI's `--bands` argument), so the config-file relative-path rule
  has nothing to anchor here. GLOSSARY gains the Bands Artifact term.
- Per-label Graduation gates (ticket #22): the pure core
  (`graduation.py`, pyright strict) applies the gates to the measured
  Eval Table against the calibrated Bands Artifact: a Managed Label
  graduates when `examples >= min_examples` AND the measured precision
  AT ITS CALIBRATED HIGH BAND THRESHOLD `>= min_precision`. The gate
  operators MIRROR `analyze.py` exactly (the strict-gt precedent:
  `total < min_total` / `dup_rate > max_dup_rate` — a value EXACTLY at
  a gate passes, pinned by golden boundary tests). Unearned labels
  carry named reasons (`below-min-examples`, `precision-below-gate`,
  `no-high-band`); a label whose band has no high threshold cannot
  graduate, and the med band never graduates anything by itself. The
  bands artifact is a separate input: if it was calibrated on a
  DIFFERENT eval table (same vocabulary + grid), the cell at the high
  band may have no measured predictions on the current table
  (tp=fp=0, undefined precision) — refused with `ValueError` (the
  refusal family), never a `ZeroDivisionError`.
  Gate targets are NOT tenth-grid (documented asymmetry with `[bands]`):
  a band target must match the candidate grid (looked up in it), a gate
  precision target is compared against measured values (any rational
  tp/(tp+fp)) — range + bool refusals do apply; `min_examples` must be
  an exact `int` (a bool is an int subclass, a float is not a count —
  `type(...) is int`, the from_json two-entry-path symmetry).
  Deterministic total order (label-sorted entries — same eval →
  byte-identical table, AC5).
- Graduation Table (`graduation.py`): the versioned JSON artifact
  (format version + content hash over every field — the
  `dataset_build.SplitManifest` discipline: canonical dump, UTC
  datetimes, `validate()` from BOTH entry paths) binding the gate
  results to the checkpoint identity (vocabulary hash, corpus
  fingerprint, split manifest hash, encoder id — the Checkpoint
  Sidecar's fields, mirrored), the eval identity (split date, test
  example count), and the `bands_content_hash` of the Bands Artifact
  it was computed from (AC3). `verify_graduation` REFUSES a table that
  disagrees with the checkpoint/eval/bands it is served against — every
  field checked independently, all disagreements named in one
  `GraduationRefusedError` (the sidecar doctrine). Same eval →
  byte-identical table (AC5, pinned across re-runs).
- `eval --graduation PATH` + `--confirm-graduation` (ticket #22): the
  deliberate act that recomputes the Graduation Table and prints the
  human-confirm diff (old → new per label, on stderr — the report/
  `--out` byte-identity stays literally true; with `--json` the
  proposal rides in the payload under the `graduation` key and stdout
  stays exactly one parseable JSON object — on a post-report refusal
  the BASE payload (the report, without the proposal) is emitted
  first, so the report is never lost on stdout; on a graduation write
  failure AND a foreign-bound confirm refusal the proposal rides,
  since the computation succeeded; the stderr refusal reason prints
  BEFORE the json emit, so an emit failure cannot swallow the cause).
  The `--out` file lands BEFORE the serving-gate acts in TEXT mode
  and at the payload emit (after the acts) in `--json` mode — in
  both, a run that later refuses still replaces a previous report
  file (the refusal is about the artifacts, not the report). Nothing
  applies itself:
  without `--confirm-graduation` the run writes NOTHING (ADR-0005 —
  no pipeline run ever mutates what is live by itself); the write is
  the CLI-gated confirm act (0600 from creation, staging + atomic
  replace, replacing an existing file announced on stderr, post-write
  readback compared by content hash). The "existing" side of the diff
  is read + parsed from the target file (absent target = the
  everything-vs-nothing baseline; an existing file that does not
  parse as a Graduation Table is named as such, and an UNREADABLE
  existing file (I/O failure) is named as unreadable — not
  "no existing table", and not misdiagnosed as a parse failure); an
  existing table bound to a DIFFERENT identity is named
  in the diff (the disagreeing fields) and refuses the confirm. The
  diff lines carry the EFFECTIVE gate targets + the measured values
  (examples, precision@high) — a gate change (min_examples 10 →
  5) is VISIBLE old → new, not a blind "graduated -> graduated". The run must have bands available (the
  precision-at-high comes from the run's calibration — computed
  in-memory, written only with `--bands`), and a degraded corpus
  (test records with missing body shards) refuses `--graduation` too —
  the same serving-gate posture as `--bands`, sharing the guard. The
  PURE gate computation (evaluate + build_table) runs BEFORE the
  `--bands` write, so a gate evaluation refusal can no longer leave
  the bands artifact published (the partial-effect class is closed
  at the source; a write refusal after a prior write remains
  possible and is named in the message).
  Usage refusals up front (exit 2): `--confirm-graduation` without
  `--graduation`; an empty `--out`/`--bands`/`--graduation` path;
  `--graduation` and `--out` naming the
  same file; `--bands` and `--graduation` naming the same file;
  `--graduation` naming an existing directory; `--graduation` naming
  the config file (for `--bands`, `--graduation`, or `--out`); and
  ANY target (`--out`, `--bands`, or `--graduation`) whose resolved
  path lies inside the checkpoint dir (manifest, sidecar, weights) or
  the
  corpus store (records index, labels/filters sidecars, bodies) — the
  table must not live beside identity files (by resolved path — a link
  named inside pointing outside is allowed and replaces the link).
  Run-time refusals (exit 5, the report preserved on stdout): a
  degraded corpus; a band calibration refusal (an unknown band
  override label, a lying identity); a gate evaluation refusal (an
  empty/duplicate Managed Label list, a Managed Label the vocabulary
  does not carry — named, an unknown gate override label, a lying
  bands/table identity, a bands artifact whose high band has no
  measured predictions on the table); an existing target that parses
  as the SIBLING artifact (a Bands Artifact for `--graduation`, a
  Graduation Table for `--bands` — refused naming the confusion,
  before any staging; garbage that parses as NOTHING keeps the
  NOTICE-replace posture); confirming over an existing table whose
  identity fields disagree with the current run (refused BEFORE any
  staging, the previous bytes intact); a write failure (naming the
  partial effect when the `--bands` artifact was already written this
  run); a post-write readback fault — the on-disk bytes do not parse
  back to a valid table (a torn/corrupt write); or a readback
  mismatch — a valid table whose content hash differs from the
  minted one, a concurrent writer's signature. The readback
  fault/mismatch messages say the write failed AFTER the target was
  replaced (the replace already happened — the previous file may be
  gone; re-run to repair), not "cannot write". Config error → exit
  4 (an out-of-range gate value, a raw bool/string). The recompute
  posture is pinned:
  `train`/`eval` are the only recomputers, and ONLY the confirm path
  writes the table — a structural source-scan tripwire against named
  mints (a tripwire, not a sandbox: getattr-construction and a
  hand-rolled dict evade any string scan) + the behavioral pins (AC4).
- Config `[graduation]` (ticket #22): the run's default gate values
  (`min_examples` 10 / `min_precision` 0.9) + per-label overrides as
  `[graduation."<label>"]` subtables (label names may carry slashes —
  quoted TOML keys). The hardened config doctrine (the `[bands]`
  precedent, reused): a single raw-seeing wrap validator collects
  EVERY problem, refuses TOML bools AND strings on the raw value
  (pydantic's lax coercion would silently turn `true` into 1 / 1.0 and
  `"10"` into 10 — named, never coerced), out-of-range values
  (`min_examples` >= 0, `min_precision` in [0.0, 1.0]), scalars under
  a label name (named as unknown/typo'd keys), empty tables (a no-op
  override is refused, not dropped), a raw `per_label` key (a reserved
  field name — the label-path spelling is the only per-label form; an
  EMPTY raw table is the typed default and is accepted — configs
  WITHOUT per-label overrides round-trip through model_dump() ->
  model_validate, while a non-empty raw table stays refused and a
  populated dump is refused by design (the typed construction —
  GateOverride values — is the programmatic API); the declared field
  gets the same value checks), blank/over-length label keys (keys are stripped — a padded
  key must not survive to the gate evaluation), and the same label
  named more than once (refused, not silently shadowed). `min_precision`
  is NOT tenth-grid (the documented asymmetry with `[bands]` — a gate
  target is compared against measured values, never looked up in the
  candidate grid). The config carries no path (the table path is the
  CLI's `--graduation` argument). GLOSSARY gains the Graduation Table
  term and the Graduation entry carries the earned-by-gates +
  identity-binding wording.

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
- `docs/ROADMAP.md` — the phase plan of record (Phase 0 remnants →
  Phase 1 → Phase 2 → explicitly-parked Phase 3); the README's
  phase-plan pointer now links there.

### Changed

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
- `dataset export` scope excludes DRAFT and CHAT (maintainer decision
  2026-11-01, ticket #17): the default query now carries the documented
  `-in:draft` token only, and the ingest layer hard-skips any message
  whose raw wire labelIds contain the system label id DRAFT or CHAT —
  no documented chat query token exists (is:chat retired; -in:chat
  undocumented, with a word-degradation hazard), so chat exclusion
  lives in code, where an explicit `--query` override cannot bypass
  it. With Additive-only + skip-present and the append-only corpus,
  the old implicit inclusion would be a permanent one-way door.
