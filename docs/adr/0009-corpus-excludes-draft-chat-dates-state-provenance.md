# The Corpus excludes DRAFT and CHAT, and every date states its provenance

Ticket #17 settled two one-way doors before any training data existed.
Under Additive-only writes + skip-present (ADR-0005) and the append-only
Corpus (ADR-0007), whatever Gmail Import ingests is effectively
permanent; and under invariant #7 a temporal split is only as honest as
the dates it can tell apart. The maintainer decision (recorded on #17,
2026-11-01): DRAFT and CHAT are excluded from the Corpus — enforced now
on the Gmail Import path, with the GYB row rejection completing the same
promise as #32 — and every record stamps where its date came from.

The exclusion is enforced in two layers: only one class has a
documented query token, and even that token is no guarantee. The Gmail
Import query carries the Google-documented `-in:draft` (full default:
`in:anywhere -in:spam -in:trash -in:draft`); chat has no documented
exclusion at all (`is:chat` was retired ~2023, `-in:chat` is
undocumented) and an undocumented negative term can silently
word-degrade and drop legitimate mail mentioning "chat" — a corruption
hazard on a one-way door. The load-bearing layer is therefore the ingest
boundary: Gmail Import hard-skips any message whose raw-wire `labelIds`
contain `DRAFT` or `CHAT` — exact id match before `labels.py`
normalization, so a user label named "chat" is untouched — counted in
`skipped_draft_chat`, never stored, never a failure. The ingest layer is
load-bearing for DRAFT too: an explicit `--query` replaces the default
query wholesale, so the query is a courtesy, not a guarantee, and REST
composition of the token is the medium-confidence evidence grade. Chat
messages are still fetched once before rejection; that fetch cost is
accepted. The exclusion is import-time only: records already in a
Corpus are unaffected (append-only — nothing is purged).

Date provenance: `CorpusRecord.date_provenance` is required at
construction — writers stamp `internal_date`, `mime_date`, or
`gyb_internaldate`, and no writer path stamps `unknown`; the reader
maps pre-existing lines — and explicit nulls, fail-safe — to an
explicit `unknown` (never rewritten — ADR-0007's append-only contract).
Split-boundary consumers filter on
`SERVER_AUTHORITATIVE_PROVENANCES`, so a forged or future MIME `Date`
header can no longer move a record across the split unnoticed.
