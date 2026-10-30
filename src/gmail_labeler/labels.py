"""The pure label vocabulary: system-label filtering and label paths.

Pure core (CODING-STANDARDS): no I/O, no config imports, deterministic and
panic-free on any input. Both the Gmail adapter (label *ids*) and the GYB
reader (label *names*) normalize through here so the two corpus sources speak
one language.

Space contract: matching is EXACT-CASE (Gmail's system namespace is; a
lowercase look-alike is user content, and case-normalizing here would
manufacture false negatives). The API path uses label IDS; the GYB path was
verified over a 21-year corpus to store system labels as exactly the uppercase
tokens below. Do NOT assume display NAMES equal these tokens (UI strings like
"Category Promotions", and GYB additionally ships a 478-entry (73-locale)
labellang map of LOCALIZED system names — INKASSIE, GELÄNLÄR, ...): the GYB
reader MUST canonicalize localized system names via that map (or an
equivalent) before this filter runs, or non-English mailboxes leak system
labels into training signal.

Known limitation: a user label may literally be named ``id:Something`` or
``CATEGORY_x`` and would then be mistaken for the canonical forms above
(collision probability near-zero; surfaced rather than silently dropped).
``id:``-namespace entries must never enter ``labels.managed``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

#: Gmail system label ids, best-effort — Google states its list "isn't
#: exhaustive" (https://developers.google.com/workspace/gmail/api/guides/labels).
#: Provenance by group: the first block is Google's guide table; CHAT/SNOOZE/
#: UNDO per community enumerations (GAM/lieer — not in the guide table);
#: the starred family is complete per the ^ss_* superstar mapping, YELLOW_STAR
#: additionally observed in the owner's GYB corpus. PINK_STAR deliberately
#: absent: no Gmail star format or ^ss_* mapping enumerates a pink star
#: (six colors only) — re-add ONLY with a source, not by symmetry-hunch.
#: Over-inclusion trade-off: in the ID space user labels are `Label_*`-prefixed
#: so nothing is suppressed; in the NAME space (GYB) Gmail provably reserves
#: only its documented system names, so an ALL-CAPS user label literally named
#: like a community-enumerated token (e.g. "SNOOZE") would be dropped. Accepted
#: and near-zero-probability, but not zero — this is NOT an authoritative or
#: exhaustive list: re-dump a real mailbox and re-audit when touching it.
SYSTEM_LABEL_IDS: frozenset[str] = frozenset(
    {
        "INBOX",
        "SENT",
        "DRAFT",
        "SPAM",
        "TRASH",
        "UNREAD",
        "STARRED",
        "IMPORTANT",
        "UNDO",
        "CHAT",
        "SNOOZE",
        "CATEGORY_PERSONAL",
        "CATEGORY_SOCIAL",
        "CATEGORY_PROMOTIONS",
        "CATEGORY_UPDATES",
        "CATEGORY_FORUMS",
        "YELLOW_STAR",
        "ORANGE_STAR",
        "RED_STAR",
        "BLUE_STAR",
        "GREEN_STAR",
        "PURPLE_STAR",
        # Superstar format icons (non-star variants): API label ids per the
        # community ^ss_* mapping table (npm gmail-cli, lieer, GAM); both
        # RED_CIRCLE (API id) and RED_BANG (UI name) cover ^ss_cr, as do
        # YELLOW_GUILLEMET/YELLOW_BANG for ^ss_cy.
        "RED_CIRCLE",
        "RED_BANG",
        "ORANGE_GUILLEMET",
        "YELLOW_GUILLEMET",
        "YELLOW_BANG",
        "GREEN_CHECK",
        "BLUE_INFO",
        "PURPLE_QUESTION",
    }
)

_SEPARATOR = "/"
_DELETED_PREFIX = "id:"


def is_system_label_id(label_id: str) -> bool:
    """True if the Gmail label id is system state rather than user content."""
    return label_id in SYSTEM_LABEL_IDS or label_id.startswith("CATEGORY_")


def label_path(label_name: str) -> tuple[str, ...]:
    """Split a Gmail label name into its path segments.

    Empty segments are dropped, so "Money/Bills" -> ("Money", "Bills") and
    degenerate inputs collapse: "" and "/" -> (). Distinct names MAY share a
    path ("Money/" and "Money" both -> ("Money",) — Gmail itself disallows
    that collision, so it indicates corpus tampering, not normal data).
    """
    return tuple(seg for seg in label_name.split(_SEPARATOR) if seg)


def path_is_at_or_under(candidate: tuple[str, ...], ancestor: tuple[str, ...]) -> bool:
    """True if `candidate` is `ancestor` or a descendant of it.

    ("Money", "Bills") is at or under ("Money",); ("Money",) is at or under
    itself; the empty ancestor matches nothing (callers pass a real path).
    Comparison is per-segment — ("Money2",) is NOT under ("Money",).
    """
    if not ancestor:
        return False
    return candidate[: len(ancestor)] == ancestor


def normalize_label_names(
    label_ids: Sequence[str], id_to_name: Mapping[str, str]
) -> list[str]:
    """Map label ids to names, dropping System Labels.

    Order is preserved and duplicates are NOT removed (upstream sources are
    duplicate-free: Gmail labelIds and GYB's UNIQUE index). Unknown ids (no
    entry in the user's label list — e.g. a label deleted after messages were
    tagged) — as do ids whose mapping is present but blank/whitespace-only —
    pass through under the raw id namespaced with ``id:`` so they stay visible
    to `dataset analyze` rather than vanishing into negatives; they must never
    enter ``labels.managed``.
    """
    names: list[str] = []
    for label_id in label_ids:
        if is_system_label_id(label_id):
            continue
        name = id_to_name.get(label_id, "")
        # A blank mapped name is as unusable as an unresolved id: "" would be
        # silently dropped downstream (the vanishing this namespace prevents).
        names.append(name if name.strip() else f"{_DELETED_PREFIX}{label_id}")
    return names
