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
captured as training signal but never acted on retroactively.

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

**Corpus**:
A user's labeled mail, assembled for training. Sourced either from a live
Gmail API export or an existing backup import; never shipped with or derived
from another user's data.

**GYB Import**:
Building a Corpus from an existing Got Your Back (GYB) backup directory rather
than pulling through the Gmail API. Optional fast path; the API export is the
canonical source.
_Avoid_: backup restore (this reads, never restores)

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
