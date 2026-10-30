# Gmail Labeler

Autonomous multi-label classification of incoming Gmail messages from a
strictly-local model, trained on the owner's own labeling history.

## Language

**Managed Label**:
One of the small set of content labels (a subset of the owner's ~40) that the
system is permitted to apply. All other labels are untouchable.
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

**Filter-Authored Label**:
A label that an existing Gmail filter applied on arrival rather than the
owner. Excluded from training signal; the model must earn its keep on labels
that filters do not cover.
_Avoid_: auto-label (reserved for AI applications)

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
