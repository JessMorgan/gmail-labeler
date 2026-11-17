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
_Avoid_: confidence label, meta label

**Application**:
A single act of applying one or more Managed Labels plus one Tier Label to one
message, performed only when at least one predicted Managed Label is genuinely
absent from the message (present labels are never re-applied). The unit that
corrections and audits are counted against.
_Avoid_: auto-label (the event, not the label)

**Additive-only**:
The write policy under which the system only ever adds labels; Corrections are
captured as training signal but never acted on retroactively. Governs writes
to Gmail only — it does not constrain the Corpus's label freshness (see
Label Sync).
_Avoid_: add-only corpus (conflates the Gmail write policy with Corpus storage)

**Graduation**:
A Managed Label's promotion from Shadow Pass to live application, earned by
meeting its per-label evaluation gate on the temporal test split.
_Avoid_: go-live, enablement

**Threshold Band**:
The per-(message, Managed Label) verdict — high, medium, or none — derived
from the label's calibrated probability and its threshold.

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

**Backlog**:
The pre-existing mailbox history, as opposed to newly-arriving mail.

**Backfill**:
The owner manually labeling Backlog mail to bring a label over the analysis
candidate gates. Distinct from Correction: Backfill edits mail the system
never applied to.
_Avoid_: relabeling campaign, data repair

**Corpus**:
A user's labeled mail, assembled for training. Sourced either from a Gmail
Import or a GYB Import; never shipped with or derived from another user's
data. Label-state freshness reaches only as far as its last Label Sync.

**Gmail Import**:
Building a Corpus by fetching labeled mail through the Gmail API — the
canonical sourcing path (GYB Import is the optional fast path). Resumable
via the corpus id set — a structural resume point, distinct from the Sync
Cursor.
_Avoid_: export (the command fills the Corpus, so it is named from the
Corpus's side; "export" is reserved for a future Corpus-dump command),
pull (implies fetch-and-merge; at ingest the store only appends)

**GYB Import**:
Building a Corpus from an existing Got Your Back (GYB) backup directory rather
than fetching through the Gmail API. Optional fast path; the Gmail Import is
the canonical source.
_Avoid_: backup restore (this reads, never restores)

**Full Resync**:
The ground-truth Label Reconciliation that needs no history: one
`labels.list` plus local resolved-view counts price both strategies up
front, and the cheaper runs (ties and an unpriceable scan go to the walk) —
the Label Inventory Scan (enumerate every
mailbox user label's current members by label id and diff against the
Corpus's sets, metadata-fetching only flagged ids; vanished labels'
corpus members fetched directly) or the Corpus Metadata Walk (fetch every
corpus id's current labels, diff, revise). The remedy for an expired Sync
Cursor and the first reconciliation for GYB-sourced Corpora; its reseed
writes the profile seed, stepping a higher stored Cursor down when the
mailbox has rolled back — replay from the lower seed is harmless
(ADR-0008).
_Avoid_: re-import (nothing is re-assembled), re-sync (Gmail's name for
the prescribed full re-pull)

**Label Sync**:
Bringing a Corpus's label state up to the mailbox's current label state,
without re-assembling the Corpus. An operation on freshness, not on ingest:
newly-arrived mail stays owned by the assembly paths (Gmail Import, GYB
Import).
_Avoid_: corpus update, re-sync ("re-sync" is Gmail's name for the prescribed
full re-pull; Label Sync re-assembles nothing)

**Unseen Id**:
An id the Corpus has never held, surfaced by the Label Sync history walk or by
Full Resync's Label Inventory Scan. Both sync paths count Unseen Ids and never
ingest them — ingest is the acquisition paths' alone. Neither history events nor
label inventories can establish that an Unseen Id is new mail or that it matches
the import query.
_Avoid_: new match, New Match stat

**Stale Id**:
A Corpus-resident id a Label Sync walk touched as changed — owed a
refresh, after which the Corpus's resolved label state matches the
mailbox again. A removal event marks a Stale Id too; keep-versus-revise
is the refresh's outcome, never a guess from the event type.
_Avoid_: dirty id, pending id

**Sync Cursor**:
A Corpus's record of how far it has consumed the mailbox's event stream;
owned by Label Sync — two doors may birth it when absent: Gmail Import,
and the first Label Sync on a cursorless Corpus, which applies nothing
(ADR-0008 consequence 1); only a Label Sync moves it: advancing, or
reseeding (step-down permitted after a rollback, ADR-0008). Distinct from the Gmail Import resume point (the corpus
id set).
_Avoid_: watermark, checkpoint

**Revision**:
A record line appended later for a message id already in the Corpus's log,
superseding the earlier line's label set. A Revision carries the record's
`body_hash`, date, source, and envelope fields unchanged — only labels
differ, and no body is written.
_Avoid_: update, correction (a Correction is realized-precision signal, not
stored label state)

**Resolved view**:
The Corpus's logical state after the store folds its record log by message
id, last occurrence wins. It is the read interface for label state; the log
itself stays append-only at ingest.
_Avoid_: current state, live view

**Compaction**:
Rewriting the Corpus's record log so each message id keeps only its newest
line (a Revision where superseded) — safe when atomic and exclusive, never
load-bearing for correctness (ADR-0007): the last-wins fold is the truth,
so compaction changes only the fold's cost, never the resolved state.
_Avoid_: garbage collection, log rotation (both imply the history itself is
what's managed; the fold already resolves it)

**Corpus Exclusion**:
DRAFT and CHAT (the exclusion pair) never enter the Corpus, from either
ingest layer, from this policy onward: Gmail Import rejects messages
whose raw wire labelIds carry the canonical id (skip-present wins first
on both layers: an already-stored message is answered `skipped_present`
from the resume id set before the fetch, and the policy never sees it),
and GYB import rejects rows whose raw label tokens
carry the canonical token (exact-case match on the canonical tokens —
localized `system_labels_extra` equivalents do not trigger rejection;
a matching GYB row that is not already in the Corpus is counted
`skipped_draft_chat` — unless its backup-index path escapes the backup
root, whereupon the fail-closed guard aborts the whole run (exit 5)
before the rejection is counted — a row already stored is reported
`skipped_present`, because the store's id set is consulted first).
One policy, one home — `labels.EXCLUDED_CORPUS_LABEL_IDS`; both layers
import the same constant so they cannot drift. Rationale: under
Additive-only and the corpus store's append-only write surface
(`CorpusStore` has no delete path), a wrong inclusion is permanent
(a one-way door), and
exclusion is the safe direction. The policy is prospective only:
nothing ever leaves the store, so records imported before it stay — it
guards entry, not what is already in.
_Avoid_: Export Scope (the decision's original name; the policy is
corpus-wide, not export-only)
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
