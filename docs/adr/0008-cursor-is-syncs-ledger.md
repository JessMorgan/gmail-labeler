# The Sync Cursor is sync's ledger

The Corpus must remember how far the mailbox's event stream has been
consumed, and more than one command touches that memory. We decided the
Sync Cursor is owned by the Label Sync *operation* — `dataset
sync-labels` is its first door, and the daemon (ADR-0003), when it
lands, advances the same ledger through the shared sync code. Gmail Import
(`dataset import-gmail`) may *birth* a cursor but never *advance* or
overwrite an existing one, and GYB Import (`dataset import-gyb`) never
births at all: a backup snapshots messages, not the event stream, and
yields no `historyId` comparable to one. Birth is always
birth-when-absent: no acquisition command ever overwrites an existing
cursor (only Label Sync moves one — by advance or by reseed, below).

Birth seeds the cursor from the history id the sync guide prescribes —
that of the most recent message, the first in the list response —
wherever the response actually carries it. Gmail's documented list stub is
`id` plus `threadId` and does not contractually promise more, so one
profile call (1 quota unit) is the fallback and in practice the reliable
path. Either seed is captured before any message is ingested — the page
seed from the first page's newest message, or a profile call taken up
front: events after capture carry higher ids, so the seed is a lower
bound and relabels made during the import land in the next sync's window
instead of being silently skipped. The page-derived seed carries a known
trade-off: it is the history id of the newest message *matching the import
query*, so a corpus of old or narrowly filtered mail can birth a cursor
already outside the retention window, where a profile seed can never be
born expired. Birth keeps the guide-ordered page seed with a profile
fallback to save the call whenever the wire carries the stub field; the
failure mode it accepts is a first sync that fails loudly as an expired
cursor — the same never-automatic recovery the last consequence
prescribes.

The cursor is a versioned JSON sidecar in the corpus root
(`sync-cursor.json`: `{version, history_id, last_sync}`, `last_sync`
tz-aware UTC ISO-8601), written temp+fsync+rename on the established
sidecar discipline and covered by the store's 0700/0600 mode enforcement
and verification through a protected-file set — the attribution snapshots
plus the cursor, explicitly split from the attribution-named tuple so the
cursor is covered as protected state and never conflated with attribution
data; the read-path mode verification covers it too. It is corpus-local
state, protected by file modes — deliberately not a keyring entry; keyring
secrecy is for tokens. A cursor that exists but does not parse *or*
validate — wrong version, or a missing, null, negative, or non-integer
`history_id` — fails loud (CorpusError): never a silent re-birth, because
guessing here is the silent-gap class this ledger exists to prevent.
Advancement happens only after the delta's revision lines are appended and fsynced
(invariant #3: durable handling first, cursor second). The Full Resync reseed may step
a higher stored cursor DOWN to the profile seed: a rolled-back mailbox can report a
history id below the ledger, and there the kept-high watermark is unreachable — it
404s every future walk and defeats the very command that exists to recover. The step-
down is harmless: the reconciliation just wrote ground truth, and ADR-0007's fold
absorbs any replay the lower seed re-opens — on a genuinely rolled-back mailbox no
retained history sits between the seed and the old watermark, so the step-down itself
re-appends nothing; where a bogus seed re-opens a window that does hold history, the
walk re-appends duplicate revision lines (it appends even when nothing moved) and the
fold absorbs them — the same harmless-replay consequence a second Gmail Import already
accepts.

## Considered Options

- Gmail Import refreshes the cursor on every run — rejected: import,
  Backfill, import (with no sync between) would silently skip the
  Backfills forever; the naive version looks correct, which is what makes
  the trap worth recording.

## Consequences

- The first Label Sync on a cursorless corpus births the cursor and
  reports — it applies nothing, because there is no consumed baseline to
  diff against yet, and the report discloses that provenance: the cursor
  is fresh while the corpus's label state may be months behind (a
  GYB-sourced corpus always is), and it recommends
  `dataset sync-labels --full-resync` for the earlier drift.
- A second Gmail Import on a live cursor leaves the window unconsumed
  for the next sync — replaying already-ingested ids is harmless (the
  revision log's last-wins fold absorbs it).
- An idle mailbox's cursor ages toward history retention — records are
  typically available for at least one week and often longer, the window
  might be significantly shorter, and records might be unavailable in
  rare cases — and dies as an explicit 404: sync fails loudly, printing
  the exact recovery command `dataset sync-labels --full-resync`.
  Recovery is never automatic.
- The daemon (ADR-0003, not yet built) is the eventual freshness owner;
  when it lands it consumes and advances this same ledger through the
  shared sync code — Label Sync ownership, not a second cursor. The
  store's single-process contract (ADR-0002: one writer per corpus, no
  cross-process locking) then means the daemon owns the corpus while it
  runs: its door and the CLI's are never concurrent.

## Implementation note (ticket #72, PR #84)

The birth wiring landed with one policy the original text left open: an
unusable seed degrades loudly, not fatally. Both candidate seeds are
held to positivity (non-positive is a wire lie — history ids count from
1): a non-positive page seed takes the single profile call, and if the
profile seed is unusable too, import warns, ingests normally, and writes
no cursor — the corpus stays birth-eligible, because seed unavailability
must never jeopardize acquisition. A cursor *write* fault is the
opposite case and stays fatal (typed exit 5): half-written ledger state
is worse than none. An existing-but-invalid cursor is the third case: it
raises CorpusError before ingesting anything, never a silent re-birth. A
bounded --max-messages smoke import births as well — the seed is still a
lower bound. One format asymmetry: the parse census tolerates a stored 0
(format rule is non-negative), but birth refuses non-positive and no
advancer may persist < 1 — the API answers such a cursor with a permanent
400, which the fake pins.
