# Gmail Labeler

Autonomous multi-label classification of incoming Gmail messages from a
strictly-local model, trained on the owner's own labeling history.

## Language

**Managed Label**:
One of the small set of content labels the owner permits the system to apply.
All other labels are untouchable.
_Avoid_: target label, predicted label

**Tier Label**:
A label attached alongside the content labels marking the confidence band under
which they were applied, so every AI application is visible and reviewable.
Three spellings (the `tiers.py` defaults, mirrored by the `[labels]` config
keys `applied` / `high` / `med`): the `ai-applied` umbrella ("the AI
contributed something"), `ai-high` (the high-confidence band), and `ai-med`
(the review-me band). The configured `high` / `med` names drive the shadow
report's tier names NOW (the report renders the band tier name for would-be
rows); the `applied` umbrella and the name→Gmail-label-id resolution are a
WRITE-time concern the daemon WILL implement in the Application phase (#24/#25)
— unshipped here (the report attaches no labels).
_Avoid_: confidence label, meta label

**Application**:
A single act of applying one or more Managed Labels to one message, performed
only when at least one predicted Managed Label is genuinely absent from the
message (present labels are never re-applied). The unit that corrections and
audits are counted against. The REAL contract: every Application carries the
umbrella Tier Label (`[labels] applied`) AND exactly one band Tier Label
(`[labels] high` or `med` — the BEST band among the applied labels, high beats
med; a skip-present label earns no band tier, it is not applied). That
composition (umbrella + one band tier) is done at WRITE time by the (future)
daemon/apply phase (#24/#25), which also resolves the configured names to
Gmail label ids. The shadow report is a PREVIEW, not an Application: it renders
the band tier NAME for would-be rows and attaches no labels (no umbrella, no
band tier on the wire).
_Avoid_: auto-label (the event, not the label)

**Additive-only**:
The write policy under which the system only ever adds labels; Corrections are
captured as training signal but never acted on retroactively.

**Graduation**:
A Managed Label's promotion from Shadow Pass to live application, earned by
meeting its per-label gates on the temporal test split (the future
window): minimum examples AND minimum measured precision at the label's
calibrated high band (a value EXACTLY at a gate passes — analyze's
strict-gt precedent; a label whose band has no high threshold cannot
graduate, and the med band never graduates anything by itself; unearned
labels carry named reasons — `below-min-examples`, `precision-below-gate`,
`no-high-band`). The gates are recomputed by `eval --graduation` and land
as a human-confirm diff (old → new per label) before anything live
changes (ADR-0005); the confirmed result is the Graduation Table.
_Avoid_: go-live, enablement

**Threshold Band**:
The per-(message, Managed Label) verdict — high, medium, or none — derived
from the label's calibrated probability and its threshold. The persisted
verdict values are `high` / `med` / `none` (the `tiers.py` `BAND_HIGH` /
`BAND_MED` constants; "none" is the explicit no-band state).
_Avoid_: medium (the persisted value is `med`)

**System Label**:
Gmail's own mailbox-state labels (`INBOX`, `UNREAD`, `SENT`, the `CATEGORY_*`
family, ...) — never content labels, never trainable, never touchable.
_Avoid_: built-in label

**Label Path**:
A user label's slash-separated hierarchy (`Money/Bills`); graduation is
planned to reason over paths, where a parent may scope its children. The
landed `dataset analyze` treats each full name as its own string —
`Money/Bills` and `Bills` are separate rows there.
_Avoid_: nested label, label tree

**Filter-Authored Label**:
A label that an existing Gmail filter applied on arrival rather than the
owner. Excluded from training signal; the model must earn its keep on labels
that filters do not cover.
_Avoid_: auto-label (reserved for AI applications)

**filter-covered / filter-unverifiable (analyze statuses)**:
`dataset analyze` report statuses for a label's filter attribution:
*filter-covered* — a simulatable (from/to/subject) Gmail filter already
applies the label on most of its mail; *filter-unverifiable* — a filter
touching the label carries criteria that cannot be simulated from the kept
record fields, so candidacy is excluded, never guessed. The report's
`filters_posture` field (`snapshot`, `attested-none`, `missing`,
`bridge-incomplete`) records why attribution was or was not performed.
The remaining reason codes name themselves: `no-filter-snapshot`,
`low-volume`, `low-recent`, `template-heavy`.
_Avoid_: auto-excluded, shadow-flagged

**Correction**:
A later manual label edit by the owner on a message the system applied to
(removing an applied label or adding one it missed), captured as new training
signal.

**Shadow Pass**:
Running predictions on messages and recording them locally without any Gmail
write, used for backlog evaluation before any label goes live.
_Avoid_: dry run (reserved for the CLI flag)

**Shadow Run**:
One `shadow run` invocation: it scores the recent Backlog through the
sidecar-validated checkpoint, the Graduation Table, and the Threshold Bands
and records the would-apply verdicts in the state DB (one transaction per run)
— applying NOTHING. Each run is identified by a run id; the verdicts are bound
to the bands/graduation artifacts as of the run (the report discloses this
basis and does not re-verify it).
_Avoid_: shadow evaluation, dry run

**Shadow Report**:
The `shadow report` output: the diff of a Shadow Run's stored would-apply
verdicts against the mailbox's ACTUAL labels (re-fetched via the read path) —
the Phase 1 exit go/no-go artifact a human reviews before choosing the
graduates. It shows per-message ADDED / ALREADY-PRESENT / NOT-APPLIED outcomes
(ids + labels + scores + tiers only — no subjects, no bodies) and a per-label
summary with the would-be tier mix.
_Avoid_: shadow summary, evaluation report

**state DB**:
The SQLite file at `[store] db_path` (default `<config dir>/gmail-labeler.db`)
holding the `shadow_results` table (run id, message id, label, probability,
band, run timestamp) and, with the daemon phase, the `applications` table. It
carries ids, probabilities, labels, and timestamps ONLY — never bodies
(invariant #5) — so it is publicly attachable. One transaction per run (a crash
mid-run leaves no half rows); a foreign schema_version is refused.
_Avoid_: shadow DB, label DB

**Backlog**:
The pre-existing mailbox history, as opposed to newly-arriving mail. The
default `shadow run` window (90 days) includes recent arrivals as well as the
older history — it is "recent mailbox", not strictly "pre-existing".

**Corpus**:
A user's labeled mail, assembled for training. Sourced either from a live
Gmail API export or an existing backup import; never shipped with or derived
from another user's data.

**GYB Import**:
Building a Corpus from an existing Got Your Back (GYB) backup directory rather
than pulling through the Gmail API. Optional fast path; the API export is the
canonical source.
_Avoid_: backup restore (this reads, never restores)

**Training Example**:
One corpus record turned for training: its id, date, label names, a
recency weight, and the record's Date Provenance stamped for
split-boundary filtering. The unit the model trains on; the corpus →
examples transform is one shared pure function.
_Avoid_: sample, data point

**Split Manifest**:
The versioned, hashable artifact materializing the temporal train/test
split: format version (v2 adds the bridge pin; v3 adds the recency-
weighting fields), split date, train/test id sets, the label vocabulary,
the split-boundary trust filter, a canonical hash of the effective
filters/labels bridge the build consumed (insensitive to filter order
and unreferenced label entries), the recency-weighting parameters
(window, old weight, and the frozen UTC cutoff), and a content hash over
all of it. `train`/`eval` rebuild against it — same records + manifest →
same examples, no re-split (invariant #7); a drifted bridge, a shrunken
corpus, or a violated format invariant refuses the rebuild.
_Avoid_: split config, train/test file

**Recency Weight**:
The per-Training-Example weight from the recency window: 1.0 at or after
the cutoff (`months_ago(now, recent_months)`, analyze's shape),
`old_weight` before it. The cutoff and both constants ride in the Split
Manifest — no ad-hoc decay curve — so a 21-year corpus cannot be
dominated by its oldest mail and a manifest-input rebuild weights from
the frozen cutoff, not a fresh `now`. Scoped to fixed-offset/UTC-
normalized inputs (callers should pass `now` in UTC for cross-host
stability); whether an eval applies the weight is the consumer's choice.
_Avoid_: decay factor, age weight

**Checkpoint**:
The fine-tuned model artifact: a plain HF-style directory (gitignored
content) plus its JSON Checkpoint Sidecar. The Model loader refuses to
serve any checkpoint whose sidecar disagrees with the Corpus/config it
is asked to serve against — a stale or mismatched model can never be
served silently.
_Avoid_: model snapshot, weights bundle

**Checkpoint Sidecar**:
The checkpoint's JSON identity file: format version, label-vocabulary
hash (kills vocabulary drift), thresholds source, Corpus fingerprint,
train-split manifest hash (the Split Manifest's content hash — the
split is pinned, not re-derived), and encoder identifier. Each field
is verified INDEPENDENTLY by the loader; any disagreement refuses the
load, naming every disagreeing field. The loader ALSO verifies the
materialized `manifest.json` against the sidecar (a tampered manifest
refuses with `split-manifest-mismatch`) and checks the format version
first (an unknown version refuses with `sidecar-version`, not
`sidecar-corrupt`).
_Avoid_: model metadata, signature file

**Corpus Fingerprint**:
The Corpus's machine-checkable serving identity: the sha256 of the
canonical record lines of the TRAIN-SPLIT slice only (the Split
Manifest's train ids — each record's byte-stable JSON line, sorted,
newline-joined) — input-order invariant, content-sensitive. Scoped to
the train split ON PURPOSE: the corpus is append-only, so whole-corpus
digesting would move the fingerprint on the first new email after
training and the Phase 2 daemon would refuse its own checkpoint
(#25's lifeline); append-only growth leaves it unchanged, a changed
train slice moves it. Pinned in the Checkpoint Sidecar so a
checkpoint trained on one train split can never be served against
another.
_Avoid_: corpus hash (unqualified), dataset checksum

**Thresholds Source**:
The Checkpoint Sidecar field naming where the serving thresholds come
from. While no calibrated Threshold Bands exist it is `none` (train
writes it, the loader expects it); the calibration artifact (#21)
slots into this field (e.g. `calibrated:<hash>`) without a sidecar
format change.
_Avoid_: threshold config, band source

**Eval Report**:
The per-label precision/recall table at candidate thresholds on the
temporal test split — the measured table Threshold Bands and
Graduation both read. Rows are Label Paths, columns the fixed
candidate-threshold grid (tenths, named in the eval core, never
scattered); each cell carries precision/recall, and each row its
support count. The test side is EXACTLY the Split Manifest's test
ids: the split-boundary trust filter
(`server_authoritative_only`) is a split-identity decision owned by
the manifest — eval inherits it, never re-applies it (no second split
implementation, invariant #7). Deterministic and diff-friendly (same
inputs → byte-identical); message bodies never enter it.
_Avoid_: eval table, metric dump

**Bands Artifact**:
The versioned JSON artifact `eval --bands` writes (opt-in — a bare
`eval` never writes it; the recompute posture: `train`/`eval` are the
only recomputers, and only `eval` writes the artifact). It carries the
calibrated per-label Threshold Bands — the high/med thresholds chosen
from the candidate grid, or the explicit no-band state with its named
reason — bound to the checkpoint identity (vocabulary hash, corpus
fingerprint, split manifest hash, encoder id: the Checkpoint Sidecar's
fields, mirrored) and the eval identity (split date, test example
count); its content hash covers every field (the SplitManifest
discipline), and a consumer verifying it against a disagreeing
checkpoint/eval REFUSES it (every field checked independently, all
disagreements named). The config's per-label band targets
(`[bands]`) ride in its per-label effective targets (provenance). The
shadow pass and the daemon (later phases) read it.
_Avoid_: bands file, threshold config

**Graduation Table**:
The versioned JSON artifact `eval --graduation --confirm-graduation`
writes (opt-in — a bare `eval`, and `eval --graduation` without the
confirm flag, never write it; the recompute posture: `train`/`eval` are
the only recomputers, and only the confirm path writes the table). It
carries the per-label gate results — examples, precision at the
calibrated high band, the high/med thresholds, the effective gate
targets, the graduated verdict, and the named reasons for unearned
labels — bound to the checkpoint identity (vocabulary hash, corpus
fingerprint, split manifest hash, encoder id: the Checkpoint Sidecar's
fields, mirrored), the eval identity (split date, test example count),
and the content hash of the Bands Artifact it was computed from; its
content hash covers every field (the SplitManifest discipline), and a
consumer verifying it against a disagreeing checkpoint/eval/bands
REFUSES it (every field checked independently, all disagreements named).
_Avoid_: graduation file, gate report

**Model (seam)**:
The `Model` Protocol — batch predict → per-label sigmoid scores over the
checkpoint's Label-Path vocabulary (row i ↔ text i, column j ↔
`vocabulary[j]`), plus vocabulary exposure. The ONE new test seam
(spec #13): `FakeModel` (hand-authored score vectors) is the offline
spine mirroring `FakeGmailClient`'s role; the real encoder (torch,
ADR-0001) sits behind a lazy import in the `train/` recipe code only,
so CI stays torch-free.
_Avoid_: classifier interface, predictor API

**Filter-Authored exclusion (builder doctrine)**:
The dataset builder excludes Filter-Authored evidence from training
examples — per-message for a simulatable filter, the whole label for an
unverifiable/malformed one (exclusion, not guessing, analyze's shared
doctrine). A missing, incomplete, or malformed filters/labels snapshot
aborts the build loudly (`BuildAbortedError`) rather than silently
training on filter-covered signal (invariant #6).
_Avoid_: filter filtering, auto-excluded

**Date Provenance**:
Where a Corpus record's date came from: the server `internalDate`
(`internal_date`), the MIME `Date` header fallback (`mime_date`), the GYB
backup's internal-date stamp (`gyb_internaldate`), or `unknown` when not
stamped by a writer (pre-existing lines, explicit nulls). Part of the
record contract, never derived — temporal-split consumers filter on
it, so a forged or future `Date` header can no longer silently move a
record across the split. Trust grouping: `internal_date` and
`gyb_internaldate` are the SAME trust class — both
server-authoritative (the GYB stamp is Gmail's own timestamp as
written by the backup) — while `mime_date` is sender-claimed and
`unknown` is legacy; the two server forms must not be
treated as different trust classes by a split-builder.
_Avoid_: date source, timestamp origin
