#!/usr/bin/env bash
# Local CI parity: the gates .github/workflows/ci.yml runs, plus uv.lock
# freshness checks around them. Required pre-push gate (AGENTS.md).
# Coverage and the live-test exclusion come from pyproject addopts; a
# targeted local run may pass --no-cov.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset UV_FROZEN  # hermeticity: ambient value would break --locked and downgrade uv lock --check

echo "== uv.lock freshness"
uv lock --check

uv sync --group dev --locked -q

echo "== interpreter (must be a CI Python version)"
uv run --frozen python - <<'EOF'
import sys
assert sys.version_info[:2] in {(3, 12), (3, 15)}, (
    f"local CI runs on a CI Python version (3.12/3.15), got {sys.version_info[:2]}"
    " — rerun with UV_PYTHON=3.12 or UV_PYTHON=3.15"
)
print("interpreter %d.%d OK" % sys.version_info[:2])
EOF

echo "== ruff check"
uv run --frozen ruff check .

echo "== ruff format"
uv run --frozen ruff format --check .

echo "== pyright"
uv run --frozen pyright

echo "== pytest + coverage (gate + live exclusion from pyproject addopts)"
uv run --frozen pytest -m "not live"

{ git diff --exit-code uv.lock; git diff --cached --exit-code uv.lock; }

echo "local-ci: all gates passed"
