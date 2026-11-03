# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
does not ship releases yet.

## Unreleased

### Added

- Phase 0 (corpus) scaffold and CI: pyproject/uv packaging, `gmail-labeler`
  CLI entry point, GitHub Actions workflow (ubuntu + macOS, Python
  3.12/3.13), `scripts/local-ci.sh` pre-push gate.
- Config (`config.py`): TOML discovery, config-dir-relative path anchoring,
  typed all-problems validation; `example.toml`.
- Label vocabulary (`labels.py`): System Label filtering, label paths,
  id→name normalization.
- Corpus store (`corpus_store.py`): append-only records + sharded bodies,
  enforced 0600/0700 modes, crash-tail handling.
- OAuth (`auth.py`, `tokens.py`): desktop-app consent flow, keyring/file
  token stores, `auth` / `auth login` commands.
- Shared MIME parsing (`mime_extract.py`) for both corpus sources.
- Gmail REST client (`gmail_client.py`) + `FakeGmailClient` test spine and
  typed error taxonomy (`gmail_errors.py`).
- `dataset export`: resumable API export, `--query`/`--max-messages`,
  `filters.json` + `labels.json` sidecars.
- `dataset import-gyb`: offline backup import (read-only `msg-db.sqlite`),
  folder-date correction, `--system-labels`.
- `dataset analyze`: per-label counts, recency, duplicate rate, filter
  attribution, candidate shortlist, `--json`.

- Design finalized: ADRs 0001–0005, GLOSSARY, CODING-STANDARDS, AGENTS.
- Project scaffolded; implementation begins at Phase 0 (export + analysis).

### Fixed

- `dataset export` no longer dies listing labels: the Gmail v1 discovery
  tree has only `users` at top level — `labels` live under it — and the
  REST client called `svc.labels()`, an `AttributeError` on the first live
  export once #56 made construction survive. The `build_service` fake now
  mirrors the real tree, and a structural test captures each surface
  method's `HttpRequest` against the bundled offline discovery doc,
  pinning method + URI path + the exact query-param set (pageToken
  forwarding included) + the parsed result each canned response must
  yield. The case table must also cover the whole client
  Protocol, so path and query-param drift on the read surfaces fails in
  CI instead of on the user's inbox (#59).
- `dataset export` no longer dies on its first live call: the REST client's
  default transport now composes `AuthorizedHttp` around a timeout-bounded
  httplib2 and hands `build()` only `http` — google-api-python-client 2.x
  rejects `http`+`credentials` as mutually exclusive, which surfaced as a
  raw `ValueError` bypassing the typed error taxonomy (#56).
- `auth login` on headless boxes: a missing browser no longer aborts the
  consent flow before the authorization URL is printed — the flow retries
  browser-less and renders the URL with an `ssh -L` port-forward recipe for
  completing consent from another machine, on line-buffered stdout so it
  appears immediately even through `ssh box '… auth login'` or `| tee` (#53).
