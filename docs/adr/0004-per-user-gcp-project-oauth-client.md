# Each user brings their own GCP project and OAuth client

Every Gmail API consumer needs an OAuth client, and for a public tool whose
project is it is a real fork. We decided each user creates their own GCP
project, enables the Gmail API, and makes a desktop-app OAuth client
(credential file path from config; client in *production* mode, not
*testing*, to dodge the 7-day refresh-token expiry) — the `gcalcli`/`lieer`
pattern, guided by an interactive `auth` wizard. A shared project maintained
by us was rejected: it centralizes revocation risk and quota on one account
and its traffic is everyone's. An IMAP + app-password fallback was rejected
for v1: Gmail's announced app-password deprecation plus a second label-writing
data path for no capability the API lacks. Users pay ~15 minutes of console
clicking and one "unverified app" warning screen — documented with
screenshots — and in return nothing is centralized on the maintainer.

Status note (Phase 0 landed, 2026-10): the decision above stands unchanged. The
interactive `auth` wizard mentioned in this ADR is NOT implemented — the
one-shot `gmail-labeler auth login` loopback flow plus the README Setup
section is today's guidance. The body's "documented with screenshots" promise is likewise
outstanding — the Setup section is text-only until the wizard work
lands. Building the wizard (or amending this ADR's
wording) remains open; until then no document should claim it exists.
