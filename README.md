# gmail-labeler

Strictly-local auto-labeling for Gmail: fine-tune a multi-label classifier on
**your own** mailbox, then let it label arriving mail — quietly when it's
confident, reviewably when it isn't.

> **Status: design complete, pre-alpha.** The architecture decisions, domain
> language, and development standards are committed; implementation begins
> with Phase 0 (export → dataset analysis). No installable tooling yet —
> watch the repo or follow the phase plan in [AGENTS.md](AGENTS.md) to see it
> land.

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
  console takes ~15 min; a setup wizard walks you through it).

## How it works

1. **Export** — pull your labeled mail via the Gmail API, or import an
   existing [Got Your Back](https://github.com/GAM-team/got-your-back) backup.
2. **Analyze & train** — per-label report (counts, filter-authored labels,
   duplicates), then fine-tune a small encoder with one sigmoid head per
   label. CPU works; a GPU is faster.
3. **Graduate** — labels pass per-label precision gates on a temporal test
   split before they're allowed to apply silently.
4. **Run** — a daemon polls the Gmail History API (~30s), classifies new
   messages, and applies graduated labels with confidence tiers. Your manual
   corrections are captured as training signal for the next round.

## Docs

- [AGENTS.md](AGENTS.md) — architecture spine and phase plan
- [GLOSSARY.md](GLOSSARY.md) — the domain language the code uses verbatim
- [CODING-STANDARDS.md](CODING-STANDARDS.md) — how this code is written
- [docs/adr/](docs/adr/) — the five shape-defining decisions

## License

AGPL-3.0-or-later. A strictly-local personal-data tool stays that way by
design — see [CONTRIBUTING.md](CONTRIBUTING.md) for what contributing means
licensing-wise.
