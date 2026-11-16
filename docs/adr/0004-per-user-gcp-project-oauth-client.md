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

Status note (updated with the wizard, 2026-11 — ticket #28): the decision above
stands unchanged — every user still brings their OWN GCP project and desktop
OAuth client; the wizard automates the wiring, not the project ownership. The
interactive `auth` wizard this ADR promised is now implemented:
`gmail-labeler auth setup` walks the console steps, verifies locally what it can
(the client file exists, parses, and is a desktop-type client), wires the config
and token store, and *composes* the existing `gmail-labeler auth login` loopback
flow to sign in. It is explicit that Gmail API enablement and the consent
screen's *In production* mode are NOT locally checkable — signing in proves the
OAuth client and consent, the first Gmail call proves API enablement, and a
token surviving 7 days proves *In production* mode; none of that is verified up
front. The
body's "documented with screenshots" promise is still outstanding: the wizard
and the README Setup section are text plus console URLs, no screenshots.
