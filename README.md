# gmail-labeler

Strictly-local auto-labeling for Gmail: fine-tune a multi-label classifier on
**your own** mailbox, then let it label arriving mail — quietly when it's
confident, reviewably when it isn't.

> **Status: pre-alpha — Phase 0 (corpus acquisition) and Phase 1's
> `train`, `eval`, and the Shadow Pass (`shadow run` / `shadow report`)
> have landed.** The architecture decisions, domain language, and
> development standards are committed; the CLI installs (`uv sync`,
> then `gmail-labeler --help`) with a working OAuth flow,
> `dataset export` / `import-gyb` / `analyze`, `train` (fine-tune on
> your own machine, checkpoint + identity sidecar), `eval` (per-label
> precision/recall on the temporal test split), and the Shadow Pass —
> `shadow run` scores the recent Backlog and applies nothing, and
> `shadow report` diffs the verdicts against your actual labels (the
> Phase 1 exit review). The daemon arrives in a later phase — follow
> the phase plan in [docs/ROADMAP.md](docs/ROADMAP.md).

## Why

Gmail filters are dumb (sender/keyword rules) and hosted AI labelers mean your
mail — or its contents' fingerprints — leaves your machine. You already
labeled thousands of emails by hand. That history is a training set. This
project turns it into a small local model that labels new arrivals the way
you would.

## What this guarantees

- **Your mail never leaves your machines.** Training, inference, and storage
  are 100% local. No telemetry, no hosted training, no API keys to us.
- **It only ever adds labels.** `gmail-labeler` will not remove anything,
  and every application (once the write phase lands) carries an `ai-applied`
  umbrella plus one band Tier Label (`ai-high` / `ai-med`) so you can audit
  every decision it ever made.
- **Nothing goes live unproven.** Each label is trained, scored on a recent
  held-out slice of *your* mail, and shadow-scored before it is ever applied
  silently. Labels that don't earn it stay in review.
- **You own the whole stack.** Bring your own Google OAuth client (their
  console takes ~15 min; the [setup](#setup) section walks you through it).

## How it works

1. **Export** — pull your labeled mail via the Gmail API, or import an
   existing [Got Your Back](https://github.com/GAM-team/got-your-back) backup.
2. **Analyze & train** — per-label report (counts, filter-authored labels,
   duplicates), then fine-tune a small encoder with one sigmoid head per
   label. CPU works; a GPU is faster.
3. **Graduate** — labels pass per-label precision gates on a temporal test
   split before they're allowed to apply silently.
4. **Run** — a daemon polls the Gmail History API (~30s), classifies new
   messages, and applies graduated labels with Tier Labels (once the write
   phase lands). Your manual corrections are captured as training signal
   for the next round.

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

## CLI

```sh
gmail-labeler auth                    # credential status
gmail-labeler auth login              # one-time desktop OAuth consent flow
gmail-labeler dataset export          # pull labeled mail via the Gmail API
gmail-labeler dataset import-gyb      # import a Got Your Back backup (offline)
gmail-labeler dataset analyze         # per-label report: counts, recency, dup rate, filter attribution
gmail-labeler train --split-date …    # fine-tune the encoder on your corpus (fully local)
gmail-labeler eval                    # per-label P/R at candidate thresholds on the test split
gmail-labeler eval --bands bands.json --graduation graduation.json --confirm-graduation
                                      # ^ mints the artifacts the Shadow Pass reads (ADR-0005 gate)
gmail-labeler shadow run --bands bands.json --graduation graduation.json
                                      # score the recent Backlog, apply NOTHING (Shadow Pass)
gmail-labeler shadow report           # diff the verdicts against your actual labels (Phase 1 exit)
```

(These assume the project venv is active; from a fresh checkout, prefix any
line with `uv run ` — e.g. `uv run gmail-labeler --help` always works.)

Every command takes `--config PATH`. `dataset export` takes `--query`
(default `in:anywhere -in:spam -in:trash -in:draft`) and
`--max-messages` (smoke runs). DRAFT and CHAT are deliberately out of
the API export — with Additive-only + skip-present and the append-only
corpus, including them would be a permanent one-way door. Exclusion is
two-layer: the query carries the documented `-in:draft` token, and the
ingest layer hard-skips any message whose raw wire labelIds carry the
system label id DRAFT or CHAT (no documented chat query token exists,
and this layer is load-bearing: an explicit `--query` replaces the
default wholesale but cannot bypass it). Records already exported are
unaffected (append-only corpus). GYB import follows whatever the backup
holds (tracked by #32). It writes `filters.json` and `labels.json`
sidecars next to the corpus — the offline snapshot `dataset analyze`
uses for filter attribution. `dataset import-gyb` takes `--path`,
`--max-messages`, and `--system-labels` (comma-separated extras — the
delimiter is a comma, so label names containing commas cannot be
expressed via the flag); it does not write sidecars, so for GYB-only
corpora `dataset analyze` withholds candidates unless you pass
`--assume-no-filters` (an attestation that the mailbox has no labeling
filters). Both export and import are resumable: the corpus id set is
the cursor, so re-running after a failure skips what is already
present. Transient failures resume cleanly; a deterministic abort
(a corrupt backup row, an escaping filename) repeats until the backup itself
is fixed — resumable is not the same as completable. `dataset analyze`
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
`CUDA_VISIBLE_DEVICES` to one). `eval` scores the temporal test split
and reports per-label precision/recall at candidate thresholds — the
measured table Threshold Bands and Graduation both read. It loads the
checkpoint through the sidecar-validated loader (a mismatched
checkpoint refuses with exit 5, naming every disagreeing field — no
silent fallback), rebuilds the test-side membership THROUGH the
checkpoint's materialized Split Manifest (the split is pinned, never
re-derived — there is no `--split-date`), and predicts through the
Model seam on the same `preprocess_text` train used. The split-boundary
trust filter is inherited from the manifest, never re-applied (the
report states it). The report is deterministic and diff-friendly
(per-label rows x threshold columns with P/R + support), opened by a
split-provenance header (split date + manifest hash — a saved report
is never indistinguishable from a stale one), to stdout by default,
`--out PATH` for the same bytes in a file (replacing an existing file
is announced on stderr), and `--json` for the machine-readable
payload; message bodies never enter it. `eval --bands PATH` is the
deliberate act that calibrates the per-label Threshold Bands from the
measured table (high/med per label, per-label min-precision targets
from the config's `[bands]` section) and writes the versioned Bands
Artifact — bound to the checkpoint + eval identity (verifiable
against the checkpoint sidecar's identity fields), byte-identical
for the same eval; its consumers (the daemon's shadow path, later
phases) will refuse it on any identity disagreement once they land.
It is opt-in: a bare `eval` never writes it
(human-gated, ADR-0005), and it is the only writer — `train`/`eval`
remain the only recomputers. Refusals: up front (exit 2) an empty
path, a path equal to `--out`, an existing directory, the config
file, and any target inside the checkpoint dir or the corpus store
(by resolved path — a link named inside pointing outside is allowed
and replaces the link; the artifact must not live beside identity
files; the canonical location lands with the daemon phase); at run
time (exit 5, the report preserved) a degraded corpus (test records
missing body shards — the artifact feeds the daemon's serving gate),
a calibration refusal (an unknown override label, a lying identity),
an existing artifact bound to a different identity (refused before
any staging — the previous bytes stay intact), a write failure; a
post-write readback fault — the on-disk bytes do not parse back to
a valid artifact (a torn/corrupt write); and a readback mismatch —
a valid artifact whose content hash differs (a concurrent writer
published a different artifact). `eval --graduation PATH` recomputes
the Graduation Table — the per-label gates (minimum examples +
minimum precision at the calibrated high band, from the config's
`[graduation]` section) applied to the Managed Labels — and prints
the human-confirm diff (old → new per label, with the effective gate
targets + measured values, so a gate change is visible) to stderr; it
writes NOTHING without `--confirm-graduation` (ADR-0005: no pipeline
run ever mutates what is live by itself). The pure gate computation
runs before the `--bands` write, so a gate refusal can no longer leave
the bands artifact published. The table is bound to the
checkpoint + eval identity + the Bands Artifact's content hash
(refused on any disagreement), byte-identical for the same eval; the
write is the CLI-gated confirm act (0600 from creation, staging +
atomic replace, post-write readback compared by content hash).
Refusals: up front (exit 2) `--confirm-graduation` without
`--graduation`, an empty `--out`/`--bands`/`--graduation` path, a path
equal to `--out` or `--bands`, an existing directory, the config file
(for `--bands`, `--graduation`, or `--out`), and ANY target
(`--out`, `--bands`, or `--graduation`) inside the checkpoint dir or
the corpus store (by resolved path — the target must not live beside
identity files); at run
time (exit 5, the report preserved) a degraded corpus (test records
missing body shards — the table feeds the serving gate, same posture
as `--bands`), a band calibration refusal, a gate evaluation refusal
(an empty/duplicate Managed Label list, a Managed Label the vocabulary
does not carry, an unknown gate override label, a lying identity), an
existing target that parses as the SIBLING artifact (a Bands Artifact
for `--graduation`, a Graduation Table for `--bands` — refused naming
the confusion, before any staging; garbage that parses as NOTHING
keeps the NOTICE-replace posture), confirming over an existing table
bound to a different identity (refused before any staging — the
previous bytes stay intact), a write failure (naming the partial
effect when the `--bands` artifact was already written this run); a
post-write readback fault — the on-disk bytes do not parse back to a
valid table (a torn/corrupt write); and a readback mismatch — a valid
table whose content hash differs (a concurrent writer published a
different table). The readback fault/mismatch messages say the write
failed AFTER the target was replaced (the previous file may be gone;
re-run to repair) — not "cannot write". Config error → exit 4 (an
out-of-range gate value, a raw bool/string). With `--json` the proposal rides in the payload
under the `graduation` key (stdout stays exactly one parseable JSON
object); on a post-report refusal the base payload (the report,
without the proposal) is emitted first — the report is never lost on
stdout — and on a graduation write failure AND a foreign-bound confirm
refusal the proposal rides (the computation succeeded). The `--out`
file lands before the serving-gate acts in TEXT mode and at the
payload emit (after the acts) in `--json` mode — in both, a run that
later refuses still replaces a previous report file.
`train`/`eval` remain the only recomputers, and
only the confirm path writes the table. The Shadow Pass (GLOSSARY:
Shadow Pass) is the Phase 1 exit path: `shadow run` scores the recent
Backlog through the sidecar-validated checkpoint, the Graduation
Table, and the Threshold Bands and APPLIES NOTHING — the verdicts are
recorded in the state DB (`[store] db_path`) as one transaction per
run (a crash mid-run leaves no half rows) — and the HUMAN review of
`shadow report` IS the Phase 1 exit (choosing the graduates,
ADR-0005). `shadow run` requires `--bands` and
`--graduation` (the calibrated Bands Artifact and the Graduation
Table — both are verified against the run's identity, the FIRST real
consumers of the `verify_bands`/`verify_graduation` gates; a
foreign-bound artifact refuses with exit 5, naming every disagreeing
field). The Backlog query defaults to `in:inbox after:<epoch-seconds>`
(the EPOCH form — calendar dates in API queries are interpreted as
MIDNIGHT PST; `--since` ISO date → UTC midnight of that date → epoch —
the named day is INCLUDED (mail from that day onward); default 90 days
ago with the time kept (now − 90d, not midnight); the default window
includes recent arrivals as
well as the older history — GLOSSARY: Backlog; `--limit` max messages,
default 100; `--query` replaces the whole backlog window — the default
and `--since` alike; `--db PATH`
overrides `[store] db_path` — `~` is expanded, and a relative `--db`
anchors to the CWD (like `--bands` / `--graduation`), unlike the
config's `[store] db_path`, which anchors to the config dir). A message
whose body cannot be fetched is skipped (per-message fail-safe:
warning + count, the run continues) — the documented posture choice
vs `eval`'s whole-run refusal: shadow is a read-only survey. A run
that produces no verdicts (the Graduation Table graduated no labels)
refuses with exit 5 ("no verdicts to record" — nothing is recorded, so
a `shadow report` would say "no runs recorded"), and a would-apply
refusal (a lying score/band, a graduated System Label, a missing bands
entry) refuses with exit 5 ("would-apply refused"). Refusals up front
(exit 2, before any work): an empty `--bands` / `--graduation` /
`--query` / `--db`, a non-ISO `--since`, `--limit <= 0`, and the state
DB path — `--db PATH` or `[store] db_path` — inside the checkpoint dir
or the corpus store (by resolved path, named by its source — the state
DB must not live beside identity files). The summary goes to stderr;
`--json` emits one parseable object (its `cutoff_epoch`/`cutoff` window
fields are present except under the `--query` override, which replaces the
window entirely). `shadow
report` diffs a run's stored verdicts against the mailbox's ACTUAL
labels (re-fetched via the read path — the labels may have changed
since the run): a deterministic per-message diff of message id + label
+ score + tier ONLY (subjects out, bodies out by construction —
invariant #5), with ADDED / ALREADY-PRESENT (skip-present visible) /
NOT-APPLIED outcomes and a per-label summary carrying the would-be
tier mix. `--run` picks a specific run (default: the latest); `--db
PATH` overrides `[store] db_path` the same way as for `shadow run`
(`~` expanded, a relative `--db` anchors to the CWD). Refusals up front
(exit 2, before any work): an empty `--run` / `--db`, and the state DB
path — `--db PATH` or `[store] db_path` — inside the checkpoint dir or
the corpus store (by resolved path, named by its source).
Config error -> exit 4; a no-DB / unknown-run / transport failure ->
exit 5.
Message bodies never enter the DB, the report, or any log path
(invariant #5) — the DB is publicly attachable. The report is a
PREVIEW, not an Application: it renders the band tier NAME (the
configured `[labels] high`/`med`) for would-be rows and attaches no
labels — the `applied` umbrella is validated at config NOW (stripped,
distinct from `high`/`med`) and consumed by the write phase later: the
umbrella + one band-tier composition (and the name→label-id resolution)
is a WRITE-time concern the daemon will implement in the Application
phase (#24/#25); GLOSSARY: Application, Tier Label.

### Reviewing the shadow report (Phase 1 exit demo)

The Phase 1 exit is a human reviewing the shadow report and choosing
the graduates (ADR-0005). The offline spine is `FakeModel` +
`FakeGmailClient` (CI, invariant #9); the opt-in REAL-checkpoint run
is the demo, behind the `live` + `smoke` markers (never run in CI —
it needs the `train` extra for the real encoder, a signed-in Gmail
account, a corpus, and the human-step artifacts below). The test is
READ-ONLY against Gmail (no write), costs API quota, writes ONLY a
tmp-redirected state DB (`--db`), and SKIPS (with a reason) when the
config / corpus / checkpoint / `bands.json` / `graduation.json` are
absent — it never trains and never confirms a Graduation Table
(those stay the human actions below):

```sh
uv run pytest -m "live and smoke" tests/test_shadow_cli.py::test_live_shadow_run_real_checkpoint
```

By hand, on a signed-in install (the human steps — a test never runs
these):

```sh
gmail-labeler train --split-date YYYY-MM-DD --force
gmail-labeler eval --bands bands.json
gmail-labeler eval --graduation graduation.json --confirm-graduation
gmail-labeler shadow run --bands bands.json --graduation graduation.json --limit 10
gmail-labeler shadow report
```

Exit codes: `0` ok · `2` usage error · `3` not implemented · `4`
auth/config unusable · `5` operation failed or incomplete.
`apply`, `daemon`, and `status` are frozen command names that arrive
in later phases (not wired yet) — until they are, they exit `2` as
unknown commands; `3` is reserved for wired-but-stubbed commands and
currently unreachable. Scripts can branch on 0/2/4/5 today; a `1`
means an unhandled crash — that's a bug, please report it.

## Docs

- [AGENTS.md](AGENTS.md) — architecture spine
- [docs/ROADMAP.md](docs/ROADMAP.md) — phase plan of record
- [GLOSSARY.md](GLOSSARY.md) — the domain language the code uses verbatim
- [CODING-STANDARDS.md](CODING-STANDARDS.md) — how this code is written
- [docs/adr/](docs/adr/) — the five shape-defining decisions

## License

AGPL-3.0-or-later. A strictly-local personal-data tool stays that way by
design — see [CONTRIBUTING.md](CONTRIBUTING.md) for what contributing means
licensing-wise.
