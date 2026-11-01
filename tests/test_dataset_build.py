"""dataset build: vocabulary, temporal split, split manifest, determinism."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from gmail_labeler.corpus_record import CorpusRecord, DateProvenance, body_hash
from gmail_labeler.dataset_build import BuildResult, SplitManifest, build_dataset

NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)
SPLIT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _rec(
    i: int,
    labels: tuple[str, ...],
    *,
    days_ago: int = 1,
    date: datetime | None = None,
    provenance: DateProvenance = "internal_date",
) -> CorpusRecord:
    return CorpusRecord(
        id=f"{i:016x}",
        date=date if date is not None else NOW - timedelta(days=days_ago),
        subject="hello",
        sender="x@y.z",
        recipients=("me@x.y",),
        labels=labels,
        body_hash=body_hash(f"unique body {i}"),
        source="api",
        date_provenance=provenance,
    )


def _build(
    recs: list[CorpusRecord],
    *,
    split_date: datetime = SPLIT,
    manifest: SplitManifest | None = None,
    server_authoritative_only: bool = False,
) -> BuildResult:
    return build_dataset(
        recs,
        now=NOW,
        split_date=split_date,
        manifest=manifest,
        server_authoritative_only=server_authoritative_only,
    )


# ---- vocabulary: Label Paths ----


def test_vocabulary_is_label_paths() -> None:
    # vocabulary = the label paths as seen (full names), incl. a parent
    # used as a label alongside its children; GLOSSARY "Label Path" is
    # authoritative — Money/Bills is a distinct path from Money
    recs = [
        _rec(1, ("Money",)),
        _rec(2, ("Money/Bills",)),
        _rec(3, ("Money", "Money/Bills")),
    ]
    result = _build(recs)
    assert result.manifest.vocabulary == ("Money", "Money/Bills")


def test_vocabulary_sorted_order_pinned() -> None:
    # ≥6 labels where sorted order != insertion order: a set() (unordered)
    # or a mis-sorted vocabulary fails this — the 2-name fixture was
    # seed-luck (deepseek). Reversed input must give the identical result.
    labels = ["Zeta", "Alpha", "Mike", "Bob", "Yan", "Ann"]
    recs = [_rec(i, (lab,)) for i, lab in enumerate(labels, start=1)]
    result = _build(recs)
    assert result.manifest.vocabulary == tuple(sorted(labels))
    assert result.manifest.vocabulary != tuple(labels)  # sorted != insertion
    assert (
        _build(list(reversed(recs))).manifest.vocabulary == result.manifest.vocabulary
    )  # input order irrelevant


def test_members_of_is_exact_path() -> None:
    result = _build([_rec(1, ("Money",)), _rec(2, ("Money/Bills",))])
    assert result.members_of("Money") == ("0000000000000001",)
    assert result.members_of("Money/Bills") == ("0000000000000002",)
    assert result.members_of("Bills") == ()  # no class named "Bills"


def test_subtree_members_scopes_descendants() -> None:
    recs = [
        _rec(1, ("Money",)),
        _rec(2, ("Money/Bills",)),
        _rec(3, ("Money/Bills/2025",)),
        _rec(4, ("Money2",)),
    ]
    result = _build(recs)
    # a parent Managed Label's gate may scope its subtree
    assert result.subtree_members("Money") == (
        "0000000000000001",
        "0000000000000002",
        "0000000000000003",
    )
    assert result.subtree_members("Money2") == ("0000000000000004",)
    # per-segment: ("Money","Bills") is NOT under ("Bills",)
    assert result.subtree_members("Bills") == ()


# ---- examples: shape + determinism ----


def test_examples_sorted_by_id_input_order_invariant() -> None:
    recs = [_rec(3, ("A",)), _rec(1, ("B",)), _rec(2, ("A",))]
    r1 = _build(recs)
    r2 = _build(list(reversed(recs)))
    assert r1 == r2  # input order irrelevant
    assert [e.id for e in r1.examples] == [
        "0000000000000001",
        "0000000000000002",
        "0000000000000003",
    ]


def test_example_carries_labels_and_provenance_stamp() -> None:
    (example,) = _build([_rec(1, ("A", "B"), provenance="mime_date")]).examples
    assert example.labels == ("A", "B")
    assert example.date_provenance == "mime_date"


def test_example_labels_are_sorted() -> None:
    # neither Gmail labelIds nor the GYB label map guarantees stable
    # order — the example's label tuple is canonical (sorted)
    (example,) = _build([_rec(1, ("B", "A"))]).examples
    assert example.labels == ("A", "B")


def test_subtree_members_empty_class_name() -> None:
    result = _build([_rec(1, ("Money",))])
    assert result.subtree_members("") == ()


# ---- temporal split (invariant #7) ----


def test_temporal_split_membership() -> None:
    recs = [
        _rec(1, ("A",), days_ago=320),  # 2025-12-14 → train
        _rec(2, ("A",), days_ago=10),  # 2026-10-20 → test
    ]
    result = _build(recs)
    assert result.manifest.train_ids == ("0000000000000001",)
    assert result.manifest.test_ids == ("0000000000000002",)


def test_split_boundary_train_strictly_before() -> None:
    # date == split_date → test (train is strictly before the split)
    result = _build([_rec(1, ("A",), date=SPLIT)])
    assert result.manifest.train_ids == ()
    assert result.manifest.test_ids == ("0000000000000001",)


def test_naive_split_date_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="split_date"):
        build_dataset([_rec(1, ("A",))], now=NOW, split_date=datetime(2026, 1, 1))


def test_naive_now_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="timezone-aware"):
        build_dataset([_rec(1, ("A",))], now=datetime(2026, 1, 1), split_date=SPLIT)


def test_duplicate_record_id_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        _build([_rec(1, ("A",)), _rec(1, ("B",))])


# ---- date provenance at the split boundary ----


def test_server_authoritative_only_excludes_fallback_sourced() -> None:
    recs = [
        _rec(1, ("A",), provenance="internal_date"),
        _rec(2, ("A",), provenance="mime_date"),
        _rec(3, ("A",), provenance="unknown"),
        _rec(4, ("A",), provenance="gyb_internaldate"),
    ]
    result = _build(recs, server_authoritative_only=True)
    ids = set(result.manifest.train_ids) | set(result.manifest.test_ids)
    # the SERVER_AUTHORITATIVE_PROVENANCES trust class, both server forms
    assert ids == {"0000000000000001", "0000000000000004"}
    assert result.manifest.server_authoritative_only is True


def test_forged_mime_date_crossing_split_is_distinguishable() -> None:
    # a forged MIME Date just before the split: by default it lands in
    # train but is stamped mime_date (eval can filter it); under
    # server_authoritative_only it is out of the split entirely
    forged = _rec(1, ("A",), date=SPLIT - timedelta(minutes=1), provenance="mime_date")
    honest = _rec(2, ("A",), date=SPLIT - timedelta(minutes=1))
    default = _build([forged, honest])
    assert default.manifest.train_ids == ("0000000000000001", "0000000000000002")
    stamped = {e.id: e.date_provenance for e in default.examples}
    assert stamped["0000000000000001"] == "mime_date"
    strict = _build([forged, honest], server_authoritative_only=True)
    assert strict.manifest.train_ids == ("0000000000000002",)


# ---- split manifest: versioned, hashable, byte-deterministic ----


def test_manifest_determinism_byte_identical() -> None:
    recs = [_rec(i, ("A",), days_ago=i) for i in range(20)]
    r1 = _build(recs)
    r2 = _build(list(reversed(recs)))
    assert r1.manifest.content_hash == r2.manifest.content_hash
    assert r1.manifest.to_json() == r2.manifest.to_json()  # byte-identical
    assert r1.examples == r2.examples
    # all 20 records fall inside the recency window (cutoff 2024-10-30),
    # so every weight is the in-window value 1.0 — this guards the
    # in-window weight, not a "flat until weighting lands" default
    assert all(e.weight == 1.0 for e in r1.examples)


def test_manifest_roundtrip() -> None:
    manifest = _build([_rec(1, ("A",))]).manifest
    assert SplitManifest.from_json(manifest.to_json()) == manifest


def test_manifest_hash_mismatch_rejected() -> None:
    manifest = _build([_rec(1, ("A",))]).manifest
    tampered = json.loads(manifest.to_json())
    tampered["train_ids"] = [*list(tampered["train_ids"]), "0000000000000009"]
    import pytest

    with pytest.raises(ValueError, match="content_hash"):
        SplitManifest.from_json(json.dumps(tampered))


def test_manifest_unknown_format_version_rejected() -> None:
    manifest = _build([_rec(1, ("A",))]).manifest
    import pytest

    with pytest.raises(ValueError, match="format_version"):
        SplitManifest.from_json(
            json.dumps(json.loads(manifest.to_json()) | {"format_version": 99})
        )


def test_manifest_not_an_object_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="not an object"):
        SplitManifest.from_json("[1, 2, 3]")


# ---- manifest-input mode: same records + manifest → same examples ----


def test_manifest_mode_no_resplit() -> None:
    recs = [_rec(1, ("A",), days_ago=320), _rec(2, ("A",), days_ago=10)]
    first = _build(recs)
    # a new record dated before the split: the manifest is authoritative —
    # no re-split, the unknown record is excluded, the manifest unchanged
    recs2 = [*recs, _rec(3, ("A",), days_ago=320)]
    result = _build(recs2, manifest=first.manifest)
    assert result.manifest == first.manifest
    assert "0000000000000003" not in result.manifest.train_ids
    assert all(e.id != "0000000000000003" for e in result.examples)
    assert [e.id for e in result.examples] == [e.id for e in first.examples]


def test_manifest_mode_ignores_caller_split_date() -> None:
    recs = [_rec(1, ("A",), days_ago=320), _rec(2, ("A",), days_ago=10)]
    manifest = _build(recs).manifest
    result = _build(
        recs, split_date=datetime(2020, 1, 1, tzinfo=UTC), manifest=manifest
    )
    assert result.manifest == manifest
    assert result.manifest.train_ids == ("0000000000000001",)


def test_manifest_mode_rejects_unknown_format_version() -> None:
    manifest = _build([_rec(1, ("A",))]).manifest
    import pytest

    with pytest.raises(ValueError, match="format_version"):
        _build([_rec(1, ("A",))], manifest=replace(manifest, format_version=99))


def test_manifest_mode_trust_filter_is_authoritative() -> None:
    recs = [
        _rec(1, ("A",), provenance="internal_date"),
        _rec(2, ("A",), provenance="mime_date"),
    ]
    lenient = _build(recs).manifest  # server_authoritative_only=False
    # caller asks for strict; the manifest (lenient) wins
    result = _build(recs, manifest=lenient, server_authoritative_only=True)
    assert {e.id for e in result.examples} == {
        "0000000000000001",
        "0000000000000002",
    }
    # and the other direction
    strict = _build(recs, server_authoritative_only=True).manifest
    result2 = _build(recs, manifest=strict, server_authoritative_only=False)
    assert {e.id for e in result2.examples} == {"0000000000000001"}


def test_manifest_mode_refuses_vocabulary_drift() -> None:
    # the corpus drifted since the manifest was built (a label renamed):
    # an example label the manifest's vocabulary does not know refuses
    # the rebuild loudly
    manifest = _build([_rec(1, ("A",))]).manifest
    import pytest

    with pytest.raises(ValueError, match="vocabulary"):
        _build([_rec(1, ("Renamed",))], manifest=manifest)


def test_manifest_mode_refuses_declared_member_absent() -> None:
    # two-way drift (a): a declared train member that is absent from the
    # (post-trust-filter) records refuses the rebuild — the manifest is
    # authoritative, so a shrunken corpus is drift, not a valid variant
    recs = [_rec(1, ("A",), days_ago=320), _rec(2, ("A",), days_ago=10)]
    manifest = _build(recs).manifest
    import pytest

    # record 1 (a declared train member) is absent from the corpus
    with pytest.raises(ValueError, match="declared split members"):
        _build([_rec(2, ("A",), days_ago=10)], manifest=manifest)


def test_manifest_mode_refuses_vocab_shrink_drift() -> None:
    # two-way drift (b, shrink direction): a label the manifest declared
    # has no member evidence left (all declared members still present) →
    # refuse (not silently dropped)
    recs = [
        _rec(1, ("A", "B"), days_ago=320),
        _rec(2, ("A",), days_ago=10),
    ]
    manifest = _build(recs).manifest  # vocabulary ("A", "B")
    import pytest

    # record 1 keeps its id (declared member present) but loses "B" →
    # "B" has no member evidence left
    with pytest.raises(ValueError, match="vocabulary"):
        _build(
            [_rec(1, ("A",), days_ago=320), _rec(2, ("A",), days_ago=10)],
            manifest=manifest,
        )


def test_manifest_mode_date_moved_is_accepted_by_design() -> None:
    # no-resplit: a record whose date has moved across the split stays in
    # its declared split — membership comes from the manifest, dates are
    # NOT re-evaluated (accepted by design; do NOT refuse dates)
    recs = [_rec(1, ("A",), days_ago=320), _rec(2, ("A",), days_ago=10)]
    manifest = _build(recs).manifest  # record 1 declared train
    # record 1's date now lands after the split, but the manifest says train
    recs2 = [_rec(1, ("A",), days_ago=10), _rec(2, ("A",), days_ago=10)]
    result = _build(recs2, manifest=manifest)
    assert "0000000000000001" in result.manifest.train_ids  # stays train
    assert [e.id for e in result.examples] == [
        "0000000000000001",
        "0000000000000002",
    ]


def test_split_date_normalized_to_utc_in_manifest() -> None:
    # the same instant in another zone gives the same manifest hash
    from datetime import timezone

    recs = [_rec(1, ("A",))]
    m1 = _build(recs).manifest
    m2 = _build(
        recs, split_date=SPLIT.astimezone(timezone(timedelta(hours=2)))
    ).manifest
    assert m1.content_hash == m2.content_hash


def test_validate_requires_utc_split_date() -> None:
    # a non-UTC split_date (utcoffset != 0) would make the same instant
    # hash differently on a tz-offset host — validate() refuses it
    # (both entry paths: in-memory and from_json)
    from datetime import timezone

    import pytest

    manifest = _build([_rec(1, ("A",))]).manifest
    non_utc = replace(
        manifest, split_date=SPLIT.astimezone(timezone(timedelta(hours=2)))
    )
    with pytest.raises(ValueError, match="UTC"):
        non_utc.validate()
    # and the from_json path (hash-consistent, so the tz check is what fires)
    non_utc = replace(non_utc, content_hash=non_utc.compute_hash())
    with pytest.raises(ValueError, match="UTC"):
        SplitManifest.from_json(non_utc.to_json())


def test_manifest_mode_validates_in_memory_manifest() -> None:
    # an in-memory manifest is refused exactly like a parsed one — the
    # full format validation, not just the version
    manifest = _build([_rec(1, ("A",))]).manifest
    import pytest

    # stale hash: a field mutated after hashing
    with pytest.raises(ValueError, match="content_hash"):
        _build(
            [_rec(1, ("A",))],
            manifest=replace(manifest, server_authoritative_only=True),
        )
    # naive split_date: a typed ValueError
    with pytest.raises(ValueError, match="UTC"):
        _build(
            [_rec(1, ("A",))],
            manifest=replace(manifest, split_date=datetime(2026, 1, 1)),
        )
    # unsorted ids
    with pytest.raises(ValueError, match="sorted and unique"):
        _build([_rec(1, ("A",))], manifest=replace(manifest, train_ids=("b", "a")))


def _hash_consistent_manifest(**overrides: object) -> str:
    base = _build([_rec(1, ("A",))]).manifest
    manifest = replace(base, **overrides)
    return replace(manifest, content_hash=manifest.compute_hash()).to_json()


def test_from_json_rejects_format_invariants() -> None:
    import pytest

    cases = {
        "train_ids": ("0000000000000002", "0000000000000001"),  # unsorted
        "vocabulary": ("B", "A"),  # unsorted
    }
    for field, value in cases.items():
        with pytest.raises(ValueError):
            SplitManifest.from_json(_hash_consistent_manifest(**{field: value}))


def test_from_json_rejects_duplicate_and_overlapping_ids() -> None:
    import pytest

    with pytest.raises(ValueError, match="sorted and unique"):
        SplitManifest.from_json(_hash_consistent_manifest(train_ids=("a", "a")))
    with pytest.raises(ValueError, match="sorted and unique"):
        SplitManifest.from_json(_hash_consistent_manifest(test_ids=("b", "a")))
    with pytest.raises(ValueError, match="disjoint"):
        SplitManifest.from_json(
            _hash_consistent_manifest(train_ids=("a",), test_ids=("a",))
        )


def test_from_json_rejects_naive_dates() -> None:
    import pytest

    with pytest.raises(ValueError, match="UTC"):
        SplitManifest.from_json(
            _hash_consistent_manifest(split_date=datetime(2026, 1, 1))
        )


def test_from_json_rejects_wrong_field_types() -> None:
    import pytest

    text = json.loads(_build([_rec(1, ("A",))]).manifest.to_json())
    for field, value in (
        ("format_version", True),  # bool is not an integer
        ("server_authoritative_only", 1),  # int is not a bool
        ("vocabulary", [1]),  # non-str entry
        ("vocabulary", "A"),  # a string, not a list
    ):
        payload = dict(text)
        payload[field] = value
        with pytest.raises(ValueError, match="field invalid"):
            SplitManifest.from_json(json.dumps(payload))


def test_from_json_rejects_missing_field() -> None:
    import pytest

    payload = json.loads(_build([_rec(1, ("A",))]).manifest.to_json())
    del payload["test_ids"]
    with pytest.raises(ValueError, match="field invalid"):
        SplitManifest.from_json(json.dumps(payload))


# ---- no random-split path: structural (source-scan) + behavioral pins ----


def test_no_rng_import_anywhere_in_module_source() -> None:
    # STRUCTURAL no-rng pin (BOTH ways): scan the module SOURCE for any
    # import of random/secrets/numpy — module-level OR function-local
    # (a vars() check evades a function-local import; deepseek + qwen
    # both proved that evasion). Any rng import, anywhere, fails this.
    import inspect
    import re

    import gmail_labeler.dataset_build as dataset_build

    source = inspect.getsource(dataset_build)
    for module in ("random", "secrets", "numpy"):
        assert not re.search(
            rf"\bimport\s+{re.escape(module)}\b|\bfrom\s+{re.escape(module)}\b",
            source,
        ), f"module imports {module!r}"


def test_split_is_purely_temporal_not_order_or_rng_dependent() -> None:
    # BEHAVIORAL no-rng pin: shuffle ~200 records with several seeds and
    # assert the split is EXACTLY the temporal predicate — any order-
    # dependent or rng-dependent split (seeded shuffle, per-id rng,
    # function-local random) fails this. The fixture spans BOTH sides of
    # the split (~100 train / ~100 test) so a drop-a-side mutant (e.g.
    # train_ids=()) also fails — an all-one-side fixture would let it
    # through. (This is the real-permutation version of the old
    # reversed-input determinism test.)
    import random

    # ~100 records before the split (train) + ~100 after (test)
    recs = [_rec(i, ("A",), days_ago=300 + i) for i in range(100)] + [
        _rec(1000 + i, ("A",), days_ago=i) for i in range(100)
    ]
    expected_train = {r.id for r in recs if r.date < SPLIT}
    expected_test = {r.id for r in recs} - expected_train
    assert expected_train and expected_test  # both sides non-empty
    for seed in range(5):
        rng = random.Random(seed)
        shuffled = rng.sample(recs, len(recs))
        result = _build(shuffled)
        assert set(result.manifest.train_ids) == expected_train
        assert set(result.manifest.test_ids) == expected_test


# ---- empty corpus ----


def test_empty_corpus() -> None:
    result = _build([])
    assert result.examples == ()
    assert result.manifest.vocabulary == ()
    assert result.manifest.train_ids == () and result.manifest.test_ids == ()
