# The Corpus is a revision log, not a snapshot

A Corpus's label state goes stale the moment it is assembled: owners
reorganize labels and run Backfills, and a one-record-per-message snapshot
can only be corrected by hand-deleting records or re-importing everything.
We decided label updates are recorded by appending a full revised record for
the same message id, with readers resolving last-occurrence-wins. Physical
writes stay append-only — no durable record line is ever rewritten in place
(crash recovery may seal or truncate an unterminated tail, never rewrite
committed bytes) — and existing corpora need no migration because a log
with no duplicate ids folds to itself.

Revisions travel through a dedicated append path, deliberately separate
from `put`, which keeps its never-duplicate contract (re-putting a present
id stays a no-op): the acquisition commands stay structurally incapable of
writing revisions. That path shares the store's durability machinery —
tail repair, append-only fd, flush/fsync ordering — and writes no body at
all, so the body-before-record ordering simply does not apply to it. There
is no new per-line marker field — a record's log position is its version —
so the record line format is unchanged for writers and readers alike, and
the fold is a no-op on duplicate-free logs.

Label revisions carry the record's `body_hash`, `date`, `source`, and
envelope fields (`subject`, `sender`, `recipients`, `thread_id`)
unchanged; only the label set differs. Sync therefore never fetches
message bodies: per Gmail's sync guidance a cached message's content is
immutable — only its `labelIds` can change — so the retained body stays
authoritative. Integrity checking is unmoved by this decision too: the
landed tripwire is `put()`'s write-time hash-vs-record check, and because
a revision writes no body, the sync path cannot create body/hash drift;
deeper read-path verification (`--verify-bodies`) is a named future
capability, not a promise this design leans on.

## Considered Options

- Overlay file of label deltas joined at read time — rejected: the join
  would be carried by every consumer (analyze, train, eval, daemon) instead
  of once in the store.
- Mutable corpus file edited in place — rejected: a crash mid-rewrite
  corrupts the whole Corpus instead of one torn tail line, and in-place
  edits destroy the diffable, body-hash-tripwired log the record format is
  built for.

## Consequences

- Consumers depend on the store's resolved view, not on id uniqueness.
  `dataset analyze` assumed "ids are unique by importer contract" because
  duplicate ids skew its report in both directions: volume, recency, and
  duplicate metrics *inflate* on superseded lines, while the attribution
  ratio — a deduped id set over a raw record count — *deflates* on
  revisions, which could pull a filter-covered label under its ceiling
  and into the candidate gate (invariant #6's unsafe direction).
  Switching analyze to the resolved view eliminates both skews rather
  than guarding them.
- Physical append-only stays literally true, and the corpus-side
  vocabulary ADR's "append-only and never merges" survives: both describe
  ingest. The fold resolves label state at read time and never mutates
  stored bytes, so "fetch-and-merge" still names something this store
  does not do at ingest.
- Re-applying a revision is harmless by construction — the last-wins fold
  absorbs replays, which is what makes crash-retry of a Label Sync safe
  alongside the cursor-durability rule.
- Messages deleted upstream keep their records (v1: reported, not
  tombstoned); tombstones are a future format change, not a fold change.
- Compaction of superseded revisions is always safe but never load-bearing
  for correctness.
