"""Attribution posture for `dataset analyze` (invariant #6) — pure core.

Decides whether filter attribution can be trusted from the attribution
snapshot sidecars, and parses/validates them into the shapes
``analyze_corpus`` consumes. Pure core (CODING-STANDARDS): no I/O, no config
or Gmail-layer imports — the caller reads the sidecar text
(``CorpusStore.read_snapshot``) and hands it over; ``None`` means absent.
Any future consumer (daemon, graduation) imports this instead of dragging
typer in through the CLI (spec #65).

Owns the corpus-format contract pieces: the sidecar filenames and the
``AttributionPosture`` literal pinned for ``--json`` consumers. Serialization
stays writer-side (gmail_import) because the filter-info type belongs to the Gmail
layer and the pure core must never import it (ADR-0002).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal, TypeGuard, cast

# summary.filters_posture values, pinned for --json consumers
AttributionPosture = Literal[
    "snapshot", "attested-none", "missing", "bridge-incomplete"
]

SIDECAR_FILTERS = "filters.json"
SIDECAR_LABELS = "labels.json"  # id->name snapshot: filters speak ids, the
# corpus speaks names — attribution (invariant #6) needs the bridge offline


def _is_str_keyed_dict(obj: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(obj, dict):
        return False
    # iterating a dict yields its keys; the cast types them as object (the
    # isinstance above is the runtime proof) — a bare dict binds Unknown
    # keys, which pyright strict rejects even in a comprehension
    return all(isinstance(key, str) for key in cast("Iterable[object]", obj))


def parse_attribution(
    filters_text: str | None,
    labels_text: str | None,
    *,
    assume_no_filters: bool,
) -> tuple[AttributionPosture, list[dict[str, object]] | None, dict[str, str]]:
    """(posture, filters, id_to_name) for the analyze report.

    ``None`` means the snapshot is absent on disk (→ ``missing`` /
    ``attested-none``); existing-but-empty text is present-but-unparseable and
    flows into the wrong-shape ``ValueError`` path below. When the filters
    text is ``None`` the labels text is never consulted — callers perform the
    bridge read only when a filters snapshot exists (the CLI's conditional
    ``read_snapshot``), so a corrupt bridge file beside an absent snapshot
    cannot fail the run.

    Sidecars are untrusted on-disk state: parseable-but-wrong-shape raises
    ValueError (the command maps it to exit 5) — never an AttributeError
    escaping into a traceback. A filters snapshot whose label ids are NOT all
    resolvable through the labels bridge cannot attribute anything, and the
    filter-authored labels would surface as CANDIDATE — invariant #6's unsafe
    direction, silent and wrong. Such a snapshot demotes to the withheld
    posture 'bridge-incomplete' (deepseek CRITICAL).
    """
    if filters_text is None:
        if assume_no_filters:
            return "attested-none", [], {}
        return "missing", None, {}
    raw_filters: object = json.loads(filters_text)
    if not isinstance(raw_filters, list):
        raise ValueError("filters snapshot is not a list")
    filters: list[dict[str, object]] = []
    entries = cast("list[object]", raw_filters)  # narrowed by isinstance above
    for item in entries:
        if not _is_str_keyed_dict(item):
            raise ValueError("filter entry is not a string-keyed object")
        filters.append({str(k): v for k, v in item.items()})
    id_to_name: dict[str, str] = {}
    if labels_text is not None:
        raw_labels: object = json.loads(labels_text)
        if not _is_str_keyed_dict(raw_labels):
            raise ValueError("labels snapshot is not an id->name object")
        if not all(isinstance(v, str) for v in raw_labels.values()):
            raise ValueError("labels snapshot is not an id->name object")
        id_to_name = {str(k): str(v) for k, v in raw_labels.items()}
    needed: set[str] = set()
    for f in filters:
        action = f.get("action")
        if action is not None and not isinstance(action, dict):
            # unreadable label bindings must not contribute zero `needed`
            # ids and let the posture claim 'snapshot' — the same failure
            # direction the non-str/non-list guards closed (deepseek r3)
            raise ValueError("action must be an object")
        # JSON objects are str-keyed by construction; the isinstance above
        # is the runtime proof the cast rests on (corpus_record.py pattern)
        ids: object = (
            cast("dict[str, object]", action).get("addLabelIds")
            if isinstance(action, dict)
            else None
        )
        if ids is not None and not isinstance(ids, list):
            # present-but-wrong-shaped is as untrusted as a non-str entry:
            # silently skipping would let a filter's labels surface
            # CANDIDATE un-attributed (qwen MINOR)
            raise ValueError("addLabelIds must be a list of strings")
        if isinstance(ids, list):
            id_entries = cast("list[object]", ids)  # narrowed by isinstance above
            if any(not isinstance(entry, str) for entry in id_entries):
                # a type-corrupt id list would bypass the bridge check and
                # silently un-attribute its label — the CRITICAL's failure
                # mode, so it raises like every other wrong-shape field
                raise ValueError("addLabelIds entries must be strings")
            # every entry proven str by the guard above; narrowing cannot
            # upgrade the Unknown element type, so the cast rests on the proof
            needed.update(cast("list[str]", id_entries))
    if needed - set(id_to_name):
        return "bridge-incomplete", None, {}
    return "snapshot", filters, id_to_name
