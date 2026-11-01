#!/usr/bin/env bash
# Local CI parity: the gates .github/workflows/ci.yml runs, plus uv.lock
# freshness checks around them. Required pre-push gate (AGENTS.md).
# Coverage comes from pyproject addopts; the live/smoke/gpu exclusions
# are stated in the pytest -m flag below (which overrides addopts); a
# targeted local run may pass --no-cov.
# NOTE: cannot run from a git-archive export tree — the uv.lock git checks
# at the bottom need a .git directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset UV_FROZEN  # hermeticity: ambient value would break --locked and downgrade uv lock --check
export PYTHONTZPATH=/nonexistent/tzdb  # mirror CI's tz-db-less env (ci.yml); pinned by tests/test_tzenv.py

echo "== uv.lock freshness"
uv lock --check

uv sync --group dev --locked -q

echo "== interpreter (must be a CI matrix version)"
uv run --frozen python - <<'EOF'
import sys
assert sys.version_info[:2] in {(3, 12), (3, 13)}, (
    f"local CI runs on a matrix interpreter (3.12/3.13), got {sys.version_info[:2]}"
)
print("interpreter %d.%d OK" % sys.version_info[:2])
EOF

echo "== ruff check"
uv run --frozen ruff check .

echo "== ruff format"
uv run --frozen ruff format --check .

echo "== pyright"
uv run --frozen pyright

echo "== pytest + coverage (gate + opt-in exclusions from pyproject addopts)"
# -m overrides addopts' marker expression, so the live/smoke/gpu
# exclusions are stated here too (CI parity; the venv is torch-free).
uv run --frozen pytest -m "not live and not smoke and not gpu"

{ git diff --exit-code uv.lock; git diff --cached --exit-code uv.lock; }

echo "local-ci: all gates passed"
