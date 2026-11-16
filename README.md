# gmail-labeler

Strictly-local auto-labeling for Gmail: fine-tune a multi-label classifier on
**your own** mailbox, then let it label arriving mail — quietly when it's
confident, reviewably when it isn't.

> **Status: pre-alpha — Phase 0 (corpus acquisition) and Phase 1's
> `train` have landed.** The architecture decisions, domain language,
> and development standards are committed; the CLI installs
> (`uv sync`, then `gmail-labeler --help`) with a working OAuth flow,
> `dataset import-gmail` / `import-gyb` / `sync-labels` / `analyze`, and
> `train` (fine-tune on your own machine, checkpoint + identity sidecar).
> Graduation, `eval`, the shadow pass, and the daemon arrive in later
> phases — follow the phase plan in [docs/ROADMAP.md](docs/ROADMAP.md).

## Why

Gmail filters are dumb (sender/keyword rules) and hosted AI labelers mean your
mail — or its contents' fingerprints — leaves your machine. You already
labeled thousands of emails by hand. That history is a training set. This
project turns it into a small local model that labels new arrivals the way
you would.

## What this guarantees

- **Your mail never leaves your machines.** Training, inference, and storage
  are 100% local. No telemetry, no hosted training, no API keys to us.
- **It only ever adds labels.** `gmail-labeler` will not remove anything, and
  every application carries an `ai-applied` umbrella plus a confidence tier
  (`ai-high` / `ai-med`) so you can audit every decision it ever made.
- **Nothing goes live unproven.** Each label is trained, scored on a recent
  held-out slice of *your* mail, and shadow-scored before it is ever applied
  silently. Labels that don't earn it stay in review.
- **You own the whole stack.** Bring your own Google OAuth client (their
  console takes ~15 min; the [setup](#setup) section walks you through it).

## How it works

1. **Import** — fetch your labeled mail via the Gmail API, or import an
   existing [Got Your Back](https://github.com/GAM-team/got-your-back)
   backup. Keep it fresh as you relabel or Backfill by hand:
   `dataset sync-labels` brings the label changes in without
   re-assembling the Corpus. A corpus with no Sync Cursor yet — a
   GYB-only corpus is always one — should start with `--full-resync`;
   the Backfill→sync→analyze loop, see [CLI](#cli).
2. **Analyze & train** — per-label report (counts, filter-authored labels,
   duplicates), then fine-tune a small encoder with one sigmoid head per
   label. CPU works; a GPU is faster.
3. **Graduate** — labels pass per-label precision gates on a temporal test
   split before they're allowed to apply silently.
4. **Run** — a daemon polls the Gmail History API (~30s), classifies new
   messages, and applies graduated labels with confidence tiers. Your manual
   corrections are captured as training signal for the next round.

## Setup

Every user brings their own GCP project and desktop-app OAuth client
(ADR-0004) — roughly 15 minutes in Google's console, nothing shared with
us.

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
   and enable the **Gmail API** in it.
2. Create an **OAuth client ID** of type **Desktop app** and download its
   JSON. (A *Web application* client cannot complete the local loopback
   flow — `auth login` rejects it.)
3. Set the project's OAuth consent screen to **In production**, not
   *Testing* — in Testing mode the refresh token expires every 7 days
   (ADR-0004).
4. Copy [example.toml](example.toml) to `config.toml` (your working
   directory, or `~/.config/gmail-labeler/`) and set
   `gmail.credentials_path` to the client JSON. Paths in the config are
   relative to the config file's directory; `~` is expanded.
5. Run `gmail-labeler auth login`. A local browser window opens; Google
   shows a "Google hasn't verified this app" screen — expected for a
   personal client: **Advanced → Go to your project → Allow**. The refresh
   token lands in your OS keyring (set `token_backend = "file"` on headless
   boxes — see [example.toml](example.toml)).
6. `gmail-labeler auth` should report `SIGNED IN`.

The token grants one scope (`gmail.modify`); token values never appear in
logs, config, or the corpus.

### Headless / remote boxes

When no browser can be launched at all, `auth login` notices the failure and
retries browser-less: it prints the authorization URL plus the remote-consent
recipe — port-forward the loopback port shown in the URL's `redirect_uri`
parameter from your workstation (`ssh -L <port>:localhost:<port> <this-box>`),
then open the URL there; the consent redirect returns to the server box
through the tunnel. The URL appears immediately even when stdout is a pipe
(`ssh box 'gmail-labeler auth login'`, `| tee`): `auth login` forces
line-buffered stdout (`ssh -t` does the same for any prompt). If a browser
launcher exists but silently fails to open a window, that is not treated as a
launch failure: the library's plain "Please visit this URL" prompt prints
instead — no recipe — and the `ssh -L` tunnel above is still required, since
a browser on your workstation opening the `localhost` URL reaches your
workstation's own loopback. Pair this with `token_backend = "file"` (step 5
above / [example.toml](example.toml)) so the refresh token has somewhere to
land without a keyring.

## CLI

```sh
gmail-labeler auth                    # credential status
gmail-labeler auth login              # one-time desktop OAuth consent flow
gmail-labeler dataset import-gmail    # fetch labeled mail via the Gmail API
gmail-labeler dataset import-gyb      # import a Got Your Back backup (offline)
gmail-labeler dataset sync-labels     # refresh corpus labels from mailbox changes (--full-resync: no history needed)
gmail-labeler dataset analyze         # per-label report: counts, recency, dup rate, filter attribution

# The Backfill loop: label Backlog mail by hand in Gmail, then —
gmail-labeler dataset sync-labels && gmail-labeler dataset analyze
# Either side refreshes the other in one sitting:
gmail-labeler dataset sync-labels --also-import-gmail  # sync labels, then the full Gmail Import (import-gmail's default query)
gmail-labeler dataset import-gmail --also-sync-labels  # import, then the Label Sync delta
gmail-labeler train --split-date …    # fine-tune the encoder on your corpus (fully local)
```

(These assume the project venv is active; from a fresh checkout, prefix any
line with `uv run ` — e.g. `uv run gmail-labeler --help` always works.)

Every command takes `--config PATH`. `dataset import-gmail` takes `--query`
(default `in:anywhere -in:spam -in:trash -in:draft`), `--max-messages` (smoke
runs), and `-v`/`-vv` (opt-in progress on STDERR: `-v` one page summary per
completed page, `-vv` adds per-message outcome lines; a final line is also
printed when the loop returns early — max-messages bound or page-fetch
failure — while an exception abort prints nothing extra). stdout and exit
codes are unchanged at every level. DRAFT and CHAT are deliberately out of the import — with Additive-only
+ skip-present and the append-only corpus, including them would be a
permanent one-way door. Exclusion is two-layer: the query carries the
documented `-in:draft` token, and the ingest layer hard-skips any message
whose raw wire labelIds carry the system label id DRAFT or CHAT (no
documented chat query token exists, and this layer is load-bearing: an
explicit `--query` replaces the default wholesale but cannot bypass it).
Records already in the corpus are unaffected (append-only). GYB import
applies the same DRAFT/CHAT exclusion at ingest (#32): a backup row whose
raw label tokens carry the canonical token is never stored. The count is new-row-only: a row
already in the corpus is answered skip-present first and reports
`skipped_present` without aborting the run; only a not-yet-stored row
counts `skipped_draft_chat`. Rejection is canonical-exact-token only —
`--system-labels` (`system_labels_extra`) equivalents are stripped from
label sets but do not trigger row rejection. A backup storing DRAFT/CHAT
with case drift imports, and the drifted token survives label filtering
as a content label (training signal). The containment guard outranks
the policy for rows not already stored: a DRAFT/CHAT row whose
backup-index path escapes the backup root aborts the whole run
(exit 5) instead of being counted; a row already in the corpus
never reaches the path check (skip-present wins — no abort).
It writes `filters.json` and `labels.json` sidecars
next to the corpus — the offline snapshot `dataset analyze` uses for filter
attribution — and births the `sync-cursor.json` Sync Cursor (ADR-0008)
when the corpus has none, seeded from the first list page before any
message is ingested (or warns and leaves the corpus birth-eligible when no
seed is available); import never advances an existing cursor, and
`dataset import-gyb` never writes one. `dataset import-gyb` takes
`--path`, `--max-messages`, and
`--system-labels` (comma-separated extras — the delimiter is a comma, so
label names containing commas cannot be expressed via the flag); it does
not write sidecars, so for GYB-only corpora `dataset analyze` withholds
candidates unless you pass `--assume-no-filters` (an attestation that the
mailbox has no labeling filters). `dataset sync-labels` walks the Gmail
history from the corpus's Sync Cursor (ADR-0008) and appends a label
Revision for every corpus message an event touched — metadata only, bodies
never (invariant #5) — so relabels and Backfills reach the corpus with no
re-assembly. The cursor advances only after the delta's revisions are
durable, a killed run is safe to re-run, `--dry-run` prints the identical
plan writing no records and leaving the cursor byte-identical, ids the
mailbox names that the corpus has never held are counted as Unseen Ids and
reported but never ingested here (history events and label inventories
cannot establish that they are new mail or that they match the import
query), and upstream-deleted mail keeps its record (counted). An expired
cursor (history 404) exits `5` printing the exact recovery command —
recovery is never automatic (ADR-0008).
`dataset sync-labels --full-resync` is that recovery: a ground-truth
reconciliation that needs no history at all — the remedy for a cursor aged
past Gmail's retention and the first reconciliation path for GYB-sourced
Corpora (which never had a cursor). One `labels.list` call (per-label
`messagesTotal`) plus local resolved-view counts yield a cost estimate, and
the strategy choice prints with its numbers before any enumeration quota is
spent, running the cheaper of the label-inventory scan (enumerate every
mailbox user label's current members by label id, diff against the
Corpus's sets, metadata-fetch only flagged ids — vanished labels' corpus
members are fetched directly) or the corpus-metadata walk (fetch every
corpus id's current labels, diff, revise). On the scan the report prints
the flag-rate before the metadata pass, so a corpus-wide drift is visible
early with the honest worst-case cost story; the walk prices its full
fetch up front instead — every id is read by construction there, so drift
is measured by the diff, not a flag-rate. The profile seed is captured
before the scan and written only after the reconciliation's revisions
are durable — stepping a higher stored cursor down when the mailbox has
rolled back (the report prints a NOTE naming both values — on
`--dry-run`, as a plan), since replay from the lower seed is harmless (ADR-0008's reseed contract; ADR-0007's fold absorbs it) —
so a relabel made mid-run lands in the next sync's window; `--dry-run`
prints the estimate, the choice, and the plan, writing nothing. The Gmail Import
and Label Sync commands compose, opt-in by design:
`sync-labels --also-import-gmail` runs the full Gmail Import pass after
the label delta is durable (Sync Cursor advanced), and
`import-gmail --also-sync-labels` runs the Label Sync walk after the
import completes durably — one report carries both passes' stats, a
second-pass failure exits non-zero without ever rolling back the first
pass's durable records or cursor, and a failed first pass never starts
the second. Default-off keeps `import-gmail`'s crisp "append new
messages, never touch existing records" meaning unambiguous (the daemon
is the eventual freshness owner, ADR-0003/0008); `--dry-run` refuses
`--also-import-gmail` as a usage error (`2`) — dry-run writes nothing,
the import pass writes records. Both imports are resumable:
the corpus id set is the resume point, so re-running after a failure
skips what is already present. Transient failures resume cleanly; a
deterministic abort (a corrupt backup row, an escaping filename) repeats
until the backup itself is fixed — resumable is not the same as
completable. `dataset analyze`
gates candidates on volume
(`--min-total`, `--min-recent`), recency window (`--recent-months`,
default 24), duplicate-body rate, and filter coverage; attribution needs
the sidecar pair — with a missing or incomplete `labels.json` bridge it
withholds candidates instead of guessing (invariant #6), and
`--assume-no-filters` is ignored when a `filters.json` snapshot exists.
`--json` emits machine-readable output whose attribution posture is at
`summary.filters_posture` (label-level `reasons` stay the engine's
vocabulary; the posture names the real cause); timestamps are ISO-8601.
Candidates are a shortlist for later graduation,
never a config change (ADR-0005 keeps retraining and applying
human-gated). `train --split-date YYYY-MM-DD` fine-tunes the encoder
on your corpus — fully local: the corpus comes from disk, the encoder
weights from a LOCAL path (`[model] encoder_path` in the config;
nothing is ever downloaded, and the `train` extra installs torch +
transformers + accelerate — on a CPU box install the smaller
CPU-only torch with `uv sync --extra train --torch-backend=cpu`).
It mints the temporal Split Manifest, rebuilds the examples through
it, trains only the train split (recency-weighted), and writes a
checkpoint dir with a JSON sidecar pinning its identity
(label-vocabulary hash, thresholds source, train-split corpus
fingerprint, split-manifest hash, encoder id) — the loader refuses
any checkpoint whose sidecar disagrees with the Corpus/config it is
asked to serve against, and refuses a tampered `manifest.json`. A
missing or incomplete filters/labels bridge refuses the build
(invariant #6; `--assume-no-filters` attests a filterless mailbox),
and an existing checkpoint dir refuses the run until `--force` is
explicit — retraining is human-gated (ADR-0005). `--force` replaces
staging-based (build in `<dir>.tmp`, verify the sidecar by readback,
swap in): a crash leaves the old checkpoint intact, an empty dir, or
the new one — each refusal-safe. `--device cpu|cuda` selects the
device (single-device contract: a multi-GPU host is refused — set
`CUDA_VISIBLE_DEVICES` to one).

Exit codes: `0` ok · `2` usage error · `3` not implemented · `4`
auth/config unusable · `5` operation failed or incomplete.
`eval`, `shadow`, `apply`, `daemon`, and `status` are frozen command
names that arrive in later phases (not wired yet) — until they are,
they exit `2` as unknown commands; `3` is reserved for wired-but-stubbed
commands and currently unreachable. Scripts can branch on 0/2/4/5
today; a `1` means an unhandled crash — that's a bug, please report it.

## Docs

- [AGENTS.md](AGENTS.md) — architecture spine
- [docs/ROADMAP.md](docs/ROADMAP.md) — phase plan of record
- [GLOSSARY.md](GLOSSARY.md) — the domain language the code uses verbatim
- [CODING-STANDARDS.md](CODING-STANDARDS.md) — how this code is written
- [docs/adr/](docs/adr/) — the eight shape-defining decisions

## License

AGPL-3.0-or-later. A strictly-local personal-data tool stays that way by
design — see [CONTRIBUTING.md](CONTRIBUTING.md) for what contributing means
licensing-wise.
