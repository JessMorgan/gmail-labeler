# Security Policy

## Scope

What matters most in this project, in order:

1. **OAuth tokens** (Gmail `gmail.modify` is a powerful grant) — keyring
   storage, never logged, never sent anywhere except Google's API.
2. **Mail content** — bodies must never enter logs, the audit DB, crash
   reports, or any network path other than the user's own Gmail API calls.
3. **Model inputs/outputs** — the classifier's training corpus is the user's
   entire mailbox; treat every file under the corpus directory as secret.

## Reporting

Report vulnerabilities privately via
[GitHub security advisories](https://github.com/JessMorgan/gmail-labeler/security/advisories/new).
No public disclosure before a fix window, please.
