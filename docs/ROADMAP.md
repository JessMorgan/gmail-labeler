# ROADMAP — the phase plan of record

What exists today, what comes next, and what is explicitly parked. Phase 0
(corpus acquisition) code lands with the open PR stack (#8/#9), then Phase 1
starts. This plan is extracted from the spec (#13), itself recovered
verbatim from the 2026-10-29 design session; the spec's *Further Notes*
records the deviations from that session.

## Phase 0 remnants — close before Phase 1 trains

- Finish the open PRs (analyze CLI, docs-sync) and merge them into `main`
  (#14, maintainer action).
- Write `docs/ROADMAP.md` from the spec's phase plan and fix the README's
  dangling "phase plan in AGENTS.md" pointer (done — this document; the
  README now links here, #15).
- Forward-fix the Phase 0 review non-blockers (#10 items 1–3 plus the
  post-push additions) — comment/wording/test only, zero behavior changes
  (#16).
- Export scope (maintainer decision recorded in #17, 2026-11-01):
  DRAFT and CHAT are excluded from the API export — the query carries
  the documented `-in:draft` token, and the ingest-layer labelIds
  rejection (raw wire DRAFT/CHAT) is the load-bearing layer, covering
  chat where no documented query token exists. The rationale: the
  previous default `in:anywhere -in:spam -in:trash` included both, and
  under append-only corpus storage + skip-present that was a one-way
  door. Implementation lands in #17 — before Phase 1 trains.
- Date-provenance field on the record contract (#17): a record's date may
  come from the server's internalDate, a fallback to the MIME `Date`
  header, or GYB's internal date — the record stamps which. A forged or
  future `Date` header can shift a fallback-sourced record across
  the temporal split (invariant #7) undetectably. A schema change —
  must land before Phase 1 trains.

**Exit:** all remnants closed — the open PRs merged and #16/#17 landed —
before Phase 1 trains.

## Phase 1 — train, eval, and the gates (nothing live)

Landed so far: the temporal-split builder core with Filter-Authored
exclusion and recency weighting (#18); the Model seam + `FakeModel`,
checkpoint sidecar identity + loader refusal, the `train/` recipe
(torch optional extra, smoke/gpu markers), and the `train` command
(#19); the `eval` command — per-label precision/recall at candidate
thresholds on the temporal test split, sidecar-validated checkpoint,
test-side membership rebuilt through the Split Manifest (#20);
per-label threshold calibration — the pure core (highest qualifying
candidate threshold per band, explicit no-band state with named
reasons, undefined-precision cells never meet a target), the versioned
Bands Artifact bound to checkpoint + eval identity (refused on any
disagreement), the config's `[bands]` per-label overrides, and
`eval --bands` as the only writer (recompute posture, #21); per-label
Graduation gates — the pure core (minimum examples + minimum
precision at the calibrated high band, analyze's strict-gt operator
precedent, named reasons, the med band never graduates), the versioned
Graduation Table bound to checkpoint + eval identity + the Bands
Artifact's content hash (refused on any disagreement), the config's
`[graduation]` per-label overrides, and `eval --graduation` with the
human-confirm diff + CLI-gated `--confirm-graduation` write (ADR-0005;
recompute posture, #22); the Shadow Pass — the pure `tiers.py`
would-apply decision core (the single home #24/#26 reuse), the
`[store]` state DB (`appdb.py`, transaction-per-run, no bodies),
`shadow run` (scores the recent Backlog, applies nothing; the FIRST
real consumers of `verify_bands`/`verify_graduation`), and `shadow
report` (diffs the verdicts against the mailbox's actual labels — the
Phase 1 exit review; #23).

Scope:

- path-aware Label Path vocabulary (maintainer decision): classes are
  leaf Label Paths; a parent Managed Label's gate may scope its subtree —
  path-aware from day one;
- temporal train/test split builder (invariant #7; materialized as a
  versioned split manifest, recency weighting informed by `dataset
  analyze`'s windows so the oldest mail doesn't dominate);
- Filter-Authored exclusion (invariant #6): the builder consumes the
  filters/labels sidecar bridge and aborts loudly on a missing or
  bridge-incomplete snapshot, never silently training on filter-authored
  signal;
- fine-tune the encoder on the user's machine — CPU smoke, GPU for real —
  each checkpoint carrying a JSON sidecar the loader validates
  (label vocabulary hash, thresholds source, corpus fingerprint, split
  manifest hash, encoder identifier);
- per-label threshold calibration into Threshold Bands (high/medium; a
  score below the threshold earns no band — the glossary's "none");
- per-label Graduation gates (minimum examples + minimum precision on the
  held-out future window);
- the `eval` report — per-label precision/recall at candidate thresholds
  on the temporal split;
- `shadow run` over the recent Backlog — predictions recorded locally, no
  Gmail write.

**Exit:** a human reviews the shadow report — diffed against the
mailbox's actual labels — and chooses the graduates.

## Phase 2 — live application

Scope:

- `daemon`: ~30 s History-API polling, cursor durability (invariant #3),
  skip-present idempotency (invariant #2), Tier Labels on every
  Application;
- Correction capture (manual fixes on applied mail become the next
  retrain's signal) and `status` (realized precision per graduated label);
- the deliberate bulk path: `apply --since … --apply` — always
  human-triggered, never automatic;
- the `auth` wizard (ADR-0004) — ships in Phase 0's spirit but is
  scheduled in Phase 2 (it is onboarding polish, not a gate to training).

**Exit:** high-band labels apply silently; the user works the `ai-med`
queue.

## Phase 3 — deferred (explicitly parked)

Written down, not implied. Each is a future decision, not debt — and a
parked phase has no exit criteria until a decision unparks it.

- Pub/Sub push — Gmail history-change notification as a trigger. ADR-0003
  keeps polling as the spine; Pub/Sub is a deferred trigger, not a
  redesign.
- Retrain assistance — tooling around the human-gated retraining loop
  (ADR-0005) as the Corpus grows and Corrections accumulate.
- PyPI publication — an installable release; today the tool is a source
  checkout.
- The community-checkpoint experiment — shared encoder checkpoints as a
  seed; an experiment, never a substitute for a user's own fine-tune.

## Excluded from all exit criteria

Bulk-fsync group commit (issue #12) is explicitly deprioritized by the
maintainer: future quality-of-life, not a gate in any phase.

## Never (out of scope)

- Any code path that removes labels (invariant #1; ADR-0005) — including
  "auto-clean" features.
- IMAP/app-password fallback (ADR-0004, decided against for v1).
- Server/multi-user deployments, central accounts, telemetry — one user,
  one machine, one process (ADR-0002).
- Retroactive labeling of non-incoming mail except via the deliberate
  `apply` command.
- Fine-tuning anything bigger than the encoder class in ADR-0001; generative
  labeling; per-device/per-account personalization.
- Live-API tests in CI (invariant #9).

## Cross-references

- [Spec #13](https://github.com/JessMorgan/gmail-labeler/issues/13)
  (source of record);
  [#10](https://github.com/JessMorgan/gmail-labeler/issues/10)
  (Phase 0 follow-ups);
  [#12](https://github.com/JessMorgan/gmail-labeler/issues/12)
  (the parked bulk-fsync item);
  [#14](https://github.com/JessMorgan/gmail-labeler/issues/14)
  (merge the open PR stack);
  [#15](https://github.com/JessMorgan/gmail-labeler/issues/15)
  (this document);
  [#16](https://github.com/JessMorgan/gmail-labeler/issues/16)
  (the Phase 0 hygiene forward-fixes);
  [#17](https://github.com/JessMorgan/gmail-labeler/issues/17)
  (the record contract: export scope + date provenance).
- [ADRs 0001–0005](adr/), [GLOSSARY.md](../GLOSSARY.md) (normative
  vocabulary), [CODING-STANDARDS.md](../CODING-STANDARDS.md).
