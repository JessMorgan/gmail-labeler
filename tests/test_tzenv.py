"""Pin the tz-db-less environment declaration (ci.yml / local-ci.sh).

Contract: under CI (``CI``/``GITHUB_ACTIONS`` truthy — falsy-ish
strings like ``false``/``0``/``no`` do not count) this test NEVER
skips — a deleted or edited ci.yml declaration must fail loudly, not
vanish. It asserts the declared state (``PYTHONTZPATH`` exactly the
nonexistent path, and no existing directory left in ``zoneinfo.TZPATH``)
and that ``ZoneInfo`` cannot load any zone while it is in effect.
Outside CI, the same assertions run when the declared value is present
(local-ci.sh); a plain dev run skips with a reason. If a ``tzdata``
package ever shadows TZPATH and makes the declaration silently
ineffective, the assertions fail and the decision gets made deliberately
again.
"""

from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import TZPATH, ZoneInfo, ZoneInfoNotFoundError

import pytest

TZDB_LESS = "/nonexistent/tzdb"


def _declared() -> bool:
    return os.environ.get("PYTHONTZPATH") == TZDB_LESS


def _in_ci() -> bool:
    # Falsy-ish strings ("0"/"false"/"no") do not turn a dev shell into CI;
    # both flaggers in review agreed the fail-loud gate must key on
    # truthy-ish values only.
    return any(
        (os.environ.get(name) or "").strip().lower() not in ("", "0", "false", "no")
        for name in ("CI", "GITHUB_ACTIONS")
    )


def test_tz_db_less_declaration_in_effect() -> None:
    if not _in_ci() and not _declared():
        pytest.skip("tz-db-less declaration not in effect (plain dev run)")
    # Under CI the declaration is mandatory: a missing/edited ci.yml
    # entry must fail here, never skip.
    assert _declared(), "PYTHONTZPATH declaration missing or edited"
    assert not any(Path(p).exists() for p in TZPATH), (
        f"tz database reachable via TZPATH={TZPATH}"
    )
    with pytest.raises(ZoneInfoNotFoundError):
        ZoneInfo("America/New_York")
