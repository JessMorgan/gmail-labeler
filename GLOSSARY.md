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
Building a Corpus by pulling labeled mail through the Gmail API — the
canonical sourcing path (GYB Import is the optional fast path). Resumable
via the corpus id set — a structural resume point, distinct from the Sync
Cursor.
_Avoid_: export (the command fills the Corpus, so it is named from the
Corpus's side; "export" is reserved for a future Corpus-dump command),
pull (implies fetch-and-merge; at ingest the store only appends)

**GYB Import**:
Building a Corpus from an existing Got Your Back (GYB) backup directory rather
than pulling through the Gmail API. Optional fast path; the Gmail Import is
the canonical source.
_Avoid_: backup restore (this reads, never restores)

**Label Sync**:
Bringing a Corpus's label state up to the mailbox's current label state,
without re-assembling the Corpus. An operation on freshness, not on ingest:
newly-arrived mail stays owned by the assembly paths (Gmail Import, GYB
Import).
_Avoid_: corpus update, re-sync ("re-sync" is Gmail's name for the prescribed
full re-pull; Label Sync re-assembles nothing)

**Unseen Id**:
An event id the Corpus has never held, named by the history walk. Label Sync
counts Unseen Ids and never ingests them — ingest is the acquisition paths'
alone. History events cannot establish that an Unseen Id is new mail or that
it matches the import query.
_Avoid_: new match, New Match stat

**Sync Cursor**:
A Corpus's record of how far it has consumed the mailbox's event stream;
owned by Label Sync — Gmail Import may birth it when absent, and only a
sync advances it. Distinct from the Gmail Import resume point (the corpus
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
itself stays append-only.
_Avoid_: current state, live view
