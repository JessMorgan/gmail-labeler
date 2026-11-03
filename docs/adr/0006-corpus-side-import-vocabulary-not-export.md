# Corpus-side acquisition commands: import-gmail, not export

Both corpus-acquisition commands fill the Corpus, so both are named from the
Corpus's side: `dataset import-gmail` (pull labeled mail through the Gmail
API) sits symmetrically next to the established `dataset import-gyb` (load an
existing GYB backup). The old name, `dataset export`, spoke from the Gmail
source side and read to a mailbox owner as the exact opposite operation —
"export the Corpus" — while occupying the word `export`, which is now
deliberately reserved for a future corpus-dump command (Corpus → external
format). `pull` was weighed and rejected: it implies fetch-and-merge
semantics against a store that is append-only and never merges, and it would
also have forced renaming the settled GYB-Import term. The cut is hard —
pre-1.0, zero external users, no compat alias or deprecation shim — because
the glossary is normative: identifiers, tests, and docs swept with the
command in the same commit (ticket #68).
