# Additive-only writes; shadow-first graduation

The system never removes a label — not on disagreement, not after retraining,
not from Corrections — because removal is the one failure mode that turns a
labeling assistant hostile against the user's own mailbox. Content labels go
live per-label through Graduation: a label is only applied silently after
meeting its per-label precision/example-count gate on a temporal test split,
and only new mail is ever live-labeled (the Backlog is Shadow Pass only; bulk
`apply` is deliberate and human-triggered). Thresholds and the graduation set
are recomputed only by `train`/`eval`, always behind a human-confirm diff.
Filter-Authored Labels are excluded from training signal so a Managed Label
can only graduate on what filters do not already cover.
