# Contributing

Thanks for interest in a tool that keeps mail local. Start with an issue to
describe the change you want; small, self-contained PRs are much more likely
to land.

## Ground rules

- **The privacy guarantees are the product.** Any change that sends message
  content off the user's machine, phones home, or enables label *removal*
  without an ADR superseding [ADR-0005](docs/adr/0005-additive-only-shadow-first-labeling.md)
  will be closed.
- `CODING-STANDARDS.md` is enforced by CI (ruff, pyright, pytest with the
  coverage gate). `GLOSSARY.md` terms are the code's terms — new domain
  concepts land in the glossary in the same PR.
- Tests fake the Gmail API; do not add live-API tests to CI.

## License

This project is AGPL-3.0-or-later. Submitting a contribution constitutes
agreeing to license it under the same terms.

## Dev environment

`uv sync`, then run the gate commands listed in `AGENTS.md`. Once code lands,
this section gains the full local-CI script.
