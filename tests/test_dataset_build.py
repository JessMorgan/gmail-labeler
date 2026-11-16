"""dataset build: vocabulary, temporal split, split manifest, determinism."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

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
    sender: str = "x@y.z",
) -> CorpusRecord:
    return CorpusRecord(
        id=f"{i:016x}",
        date=date if date is not None else NOW - timedelta(days=days_ago),
        subject="hello",
        sender=sender,
        recipients=("me@x.y",),
        labels=labels,
        body_hash=body_hash(f"unique body {i}"),
        source="api",
        date_provenance=provenance,
    )


def _build(
    recs: list[CorpusRecord],
    *,
    filters: Sequence[Mapping[str, object]] = (),
    id_to_name: Mapping[str, str] | None = None,
    now: datetime = NOW,
    split_date: datetime = SPLIT,
    manifest: SplitManifest | None = None,
    server_authoritative_only: bool = False,
    recent_months: int = 24,
    old_weight: float = 0.1,
) -> BuildResult:
    return build_dataset(
        recs,
        filters=filters,
        id_to_name=id_to_name if id_to_name is not None else {},
        now=now,
        split_date=split_date,
        manifest=manifest,
        server_authoritative_only=server_authoritative_only,
        recent_months=recent_months,
        old_weight=old_weight,
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
        build_dataset(
            [_rec(1, ("A",))],
            filters=[],
            id_to_name={},
            now=NOW,
            split_date=datetime(2026, 1, 1),
        )


def test_naive_now_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="timezone-aware"):
        build_dataset(
            [_rec(1, ("A",))],
            filters=[],
            id_to_name={},
            now=datetime(2026, 1, 1),
            split_date=SPLIT,
        )


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
    # the fixture's 20 records all fall inside the recency window, so
    # weights pin to its in-window value 1.0
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


def test_validate_requires_utc_weight_cutoff() -> None:
    # mirror the split_date pin: a non-UTC weight_cutoff (utcoffset != 0)
    # would make the same instant hash differently on a tz-offset host —
    # validate() refuses it (both entry paths: in-memory and from_json);
    # a naive weight_cutoff is refused too
    from datetime import timezone

    import pytest

    manifest = _build([_rec(1, ("A",))]).manifest
    non_utc = replace(
        manifest,
        weight_cutoff=manifest.weight_cutoff.astimezone(timezone(timedelta(hours=2))),
    )
    with pytest.raises(ValueError, match="UTC"):
        non_utc.validate()
    # and the from_json path (hash-consistent, so the tz check is what fires)
    non_utc = replace(non_utc, content_hash=non_utc.compute_hash())
    with pytest.raises(ValueError, match="UTC"):
        SplitManifest.from_json(non_utc.to_json())
    # naive weight_cutoff is refused too
    naive = replace(manifest, weight_cutoff=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="UTC"):
        naive.validate()


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
    with pytest.raises(ValueError, match="UTC"):
        SplitManifest.from_json(
            _hash_consistent_manifest(weight_cutoff=datetime(2026, 1, 1))
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


# ---- Filter-Authored exclusion (invariant #6) ----


def _filt(criteria: object, label_id: str = "L1") -> dict[str, object]:
    f: dict[str, object] = {"action": {"addLabelIds": [label_id]}}
    if criteria is not None:
        f["criteria"] = criteria
    return f


def test_missing_snapshot_aborts_build() -> None:
    # mirrors test_no_snapshot_withholds_everything: analyze withholds,
    # the builder ABORTS — training on filter-covered signal is louder
    # than a report row
    import pytest

    from gmail_labeler.dataset_build import BuildAbortedError

    with pytest.raises(BuildAbortedError) as exc:
        build_dataset(
            [_rec(1, ("A",))], filters=None, id_to_name={}, now=NOW, split_date=SPLIT
        )
    assert exc.value.reason == "no-filter-snapshot"


def test_bridge_incomplete_aborts_build() -> None:
    # a filter whose label id is not resolvable through the id→name map:
    # its filter-authored label would surface as trainable signal
    import pytest

    from gmail_labeler.dataset_build import BuildAbortedError

    with pytest.raises(BuildAbortedError) as exc:
        build_dataset(
            [_rec(1, ("A",))],
            filters=[_filt({"from": "x@y.z"}, "ZZ")],
            id_to_name={},  # ZZ unresolvable
            now=NOW,
            split_date=SPLIT,
        )
    assert exc.value.reason == "bridge-incomplete"


def test_filter_covered_evidence_excluded_per_message() -> None:
    # half the records are from cov@x.z → their "Cov" evidence is
    # excluded; the rest keep it
    recs = [
        _rec(i, ("Cov",), sender="cov@x.z" if i % 2 else "oth@x.z") for i in range(1, 9)
    ]
    result = _build(
        recs, filters=[_filt({"from": "cov@x.z"})], id_to_name={"L1": "Cov"}
    )
    labeled = {e.id: e.labels for e in result.examples}
    for i in range(1, 9):
        expected = () if i % 2 else ("Cov",)
        assert labeled[f"{i:016x}"] == expected
    # Cov still has evidence → stays in the vocabulary
    assert result.manifest.vocabulary == ("Cov",)


def test_unverifiable_filter_excludes_whole_label() -> None:
    # query criteria cannot be simulated → ALL of the label's evidence
    # is excluded (exclusion, not guessing — analyze's doctrine)
    recs = [_rec(i, ("Q",)) for i in range(1, 5)]
    result = _build(recs, filters=[_filt({"query": "ql"})], id_to_name={"L1": "Q"})
    assert all(e.labels == () for e in result.examples)
    assert result.manifest.vocabulary == ()  # no evidence → no class


def test_malformed_criteria_excludes_whole_label() -> None:
    result = _build([_rec(1, ("M",))], filters=[_filt("junk")], id_to_name={"L1": "M"})
    assert result.examples[0].labels == ()


def test_match_all_filter_covers_every_message() -> None:
    recs = [_rec(i, ("C",)) for i in range(1, 4)]
    result = _build(recs, filters=[_filt({})], id_to_name={"L1": "C"})
    assert all(e.labels == () for e in result.examples)


def test_exclusion_leaves_other_labels() -> None:
    result = _build(
        [_rec(1, ("Cov", "Other"), sender="cov@x.z")],
        filters=[_filt({"from": "cov@x.z"})],
        id_to_name={"L1": "Cov"},
    )
    assert result.examples[0].labels == ("Other",)


def test_filter_touching_label_absent_from_corpus_is_inert() -> None:
    # the id IS resolvable (bridge complete) but no record carries the
    # label: no abort, no effect
    result = _build(
        [_rec(1, ("A",))],
        filters=[_filt({"from": "x@y.z"})],
        id_to_name={"L1": "Ghost"},
    )
    assert result.examples[0].labels == ("A",)
    assert result.manifest.vocabulary == ("A",)


def test_filter_without_add_label_ids_is_inert() -> None:
    result = _build(
        [_rec(1, ("A",))],
        filters=[{"criteria": {"from": "x@y.z"}, "action": {}}],
        id_to_name={},
    )
    assert result.examples[0].labels == ("A",)


def test_malformed_action_shape_aborts() -> None:
    # unreadable bindings must not contribute zero ids and let the
    # filter's label surface as trainable signal (the cli.py guard,
    # deepseek CRITICAL precedent)
    import pytest

    from gmail_labeler.dataset_build import BuildAbortedError

    for bad in (
        {"criteria": {"from": "x@y.z"}, "action": "junk"},
        {"criteria": {"from": "x@y.z"}, "action": {"addLabelIds": "L1"}},
        {"criteria": {"from": "x@y.z"}, "action": {"addLabelIds": ["L1", 42]}},
    ):
        with pytest.raises(BuildAbortedError) as exc:
            build_dataset(
                [_rec(1, ("A",))],
                filters=[bad],
                id_to_name={"L1": "A"},
                now=NOW,
                split_date=SPLIT,
            )
        assert exc.value.reason == "bridge-malformed"


def test_blank_mapped_name_is_bridge_incomplete() -> None:
    # labels.py: a blank mapped name is namespaced id: — it must never
    # enter the vocabulary, so the filter's binding is unresolvable
    import pytest

    from gmail_labeler.dataset_build import BuildAbortedError

    with pytest.raises(BuildAbortedError) as exc:
        build_dataset(
            [_rec(1, ("A",))],
            filters=[_filt({"from": "x@y.z"})],
            id_to_name={"L1": "  "},
            now=NOW,
            split_date=SPLIT,
        )
    assert exc.value.reason == "bridge-incomplete"


def test_non_mapping_filter_element_aborts_malformed() -> None:
    # a filter element that is not a Mapping is bridge-malformed (the
    # abort classes are totalized — no raw AttributeError escapes)
    import pytest

    from gmail_labeler.dataset_build import BuildAbortedError

    with pytest.raises(BuildAbortedError) as exc:
        build_dataset(
            [_rec(1, ("A",))],
            filters=cast(Sequence[Mapping[str, object]], ["not-a-mapping"]),
            id_to_name={},
            now=NOW,
            split_date=SPLIT,
        )
    assert exc.value.reason == "bridge-malformed"


def test_build_aborted_error_is_a_value_error() -> None:
    # BuildAbortedError subclasses ValueError so a CLI mapping (#19/#25)
    # catches the whole build-refusal family with one `except ValueError`
    from gmail_labeler.dataset_build import BuildAbortedError

    assert issubclass(BuildAbortedError, ValueError)


def test_exclusion_observability_fields() -> None:
    # wholly-excluded classes leave a trace (label, reason); records that
    # had labels but came out all-negative are counted (item 16)
    recs = [
        _rec(1, ("Q", "Other")),  # Q unverifiable → excluded; Other kept
        _rec(2, ("Q",)),  # Q unverifiable → emptied (all-negative)
    ]
    result = _build(recs, filters=[_filt({"query": "ql"})], id_to_name={"L1": "Q"})
    assert result.excluded_classes == (("Q", "unverifiable"),)
    assert result.emptied_examples == 1  # only record 2 came out all-negative
    # malformed criteria → reason "malformed"
    result2 = _build([_rec(1, ("M",))], filters=[_filt("junk")], id_to_name={"L1": "M"})
    assert result2.excluded_classes == (("M", "malformed"),)
    assert result2.emptied_examples == 1


def test_manifest_mode_exclusion_reproduces_identically() -> None:
    # positive reproduction (item 17): identical bridge + manifest →
    # identical excluded examples (not just the drift-refusal pins)
    recs = [
        _rec(i, ("Cov",), sender="cov@x.z" if i % 2 else "oth@x.z") for i in range(1, 9)
    ]
    first = _build(recs, filters=[_filt({"from": "cov@x.z"})], id_to_name={"L1": "Cov"})
    result = _build(
        recs,
        manifest=first.manifest,
        filters=[_filt({"from": "cov@x.z"})],
        id_to_name={"L1": "Cov"},
    )
    assert result.examples == first.examples
    assert result.excluded_classes == first.excluded_classes
    assert result.emptied_examples == first.emptied_examples
    assert result.manifest == first.manifest


def test_manifest_mode_refuses_referenced_bridge_change() -> None:
    # a change to a REFERENCED filter (the "stricter snapshot silently
    # empties a declared class" direction) is drift → refuse the rebuild
    recs = [_rec(1, ("Cov",), sender="cov@x.z")]
    manifest = _build(
        recs, filters=[_filt({"from": "cov@x.z"})], id_to_name={"L1": "Cov"}
    ).manifest
    import pytest

    with pytest.raises(ValueError, match="bridge_hash"):
        _build(
            recs,
            manifest=manifest,
            filters=[_filt({"query": "q"})],
            id_to_name={"L1": "Cov"},
        )


def test_bridge_hash_insensitive_to_nonsemantic_variation() -> None:
    # canonical _bridge_hash (item 12): non-semantic variation must NOT
    # invalidate the manifest — filter reorder, an unreferenced-label
    # rename, and an unreferenced-id addition all give the same hash
    recs = [_rec(1, ("Cov",), sender="cov@x.z")]
    f1 = _filt({"from": "cov@x.z"})
    f2 = _filt({"to": "me@x.y"})
    base = _build(recs, filters=[f1, f2], id_to_name={"L1": "Cov", "L2": "Ghost"})
    # filter reorder: same effective bridge
    assert (
        _build(
            recs, filters=[f2, f1], id_to_name={"L1": "Cov", "L2": "Ghost"}
        ).manifest.bridge_hash
        == base.manifest.bridge_hash
    )
    # unreferenced-label rename (L2 "Ghost" → "Phantom", not referenced)
    assert (
        _build(
            recs, filters=[f1, f2], id_to_name={"L1": "Cov", "L2": "Phantom"}
        ).manifest.bridge_hash
        == base.manifest.bridge_hash
    )
    # unreferenced-id addition (L3, not referenced by any filter)
    assert (
        _build(
            recs,
            filters=[f1, f2],
            id_to_name={"L1": "Cov", "L2": "Ghost", "L3": "New"},
        ).manifest.bridge_hash
        == base.manifest.bridge_hash
    )
    # and the rebuild against the manifest is accepted for all of them
    for id_to_name in (
        {"L1": "Cov", "L2": "Ghost"},
        {"L1": "Cov", "L2": "Phantom"},
        {"L1": "Cov", "L2": "Ghost", "L3": "New"},
    ):
        result = _build(
            recs,
            manifest=base.manifest,
            filters=[f1, f2],
            id_to_name=id_to_name,
        )
        assert result.manifest == base.manifest


def test_bridge_hash_refuses_referenced_label_rename() -> None:
    # a REFERENCED label rename (id_to_name[L1]: "Cov" → "Cov2") IS drift —
    # the id→name value is part of the effective bridge, so the hash owns
    # this class (a mutant that hashes ids-only would let it through; the
    # vocabulary-drift backstop is separate and must not be the only guard)
    recs = [_rec(1, ("Cov",), sender="cov@x.z")]
    f1 = _filt({"from": "cov@x.z"})
    base = _build(recs, filters=[f1], id_to_name={"L1": "Cov"})
    renamed = _build(recs, filters=[f1], id_to_name={"L1": "Cov2"})
    assert renamed.manifest.bridge_hash != base.manifest.bridge_hash
    import pytest

    with pytest.raises(ValueError, match="bridge_hash"):
        _build(recs, manifest=base.manifest, filters=[f1], id_to_name={"L1": "Cov2"})


# ---- recency weighting (analyze's window shapes) ----


def test_recent_weight_one_old_weight_constant() -> None:
    recs = [
        _rec(1, ("A",), days_ago=1),  # inside the 24-month window
        _rec(2, ("A",), date=datetime(2005, 6, 1, tzinfo=UTC)),  # outside
    ]
    result = _build(recs)
    weights = {e.id: e.weight for e in result.examples}
    assert weights["0000000000000001"] == 1.0
    assert weights["0000000000000002"] == 0.1


def test_2005_tail_break_even_at_realistic_ratio() -> None:
    # at a realistic >=10:1 old:recent ratio, old_weight=0.1 is the
    # break-even point (1000*0.1 == 100*1.0) — the tail does not dominate,
    # and the break-even math is asserted explicitly (deepseek minor)
    old = [
        _rec(1000 + i, ("Old",), date=datetime(2005, 1, 1 + (i % 27), tzinfo=UTC))
        for i in range(1000)
    ]
    recent = [_rec(i, ("New",), days_ago=i) for i in range(100)]
    result = _build(old + recent)
    aggregate = {"Old": 0.0, "New": 0.0}
    for example in result.examples:
        for label in example.labels:
            aggregate[label] += example.weight
    # break-even: the tail's aggregate weight equals the recent mail's
    # (approx — 0.1 is not exact in binary, so 1000*0.1 accumulates a hair
    # over 100.0)
    import pytest

    assert aggregate["Old"] == pytest.approx(aggregate["New"])
    assert pytest.approx(100 * 1.0) == 1000 * 0.1  # the break-even math
    assert all(e.weight == 0.1 for e in result.examples if "Old" in e.labels)
    assert all(e.weight == 1.0 for e in result.examples if "New" in e.labels)


def test_weighting_window_is_analyze_shape() -> None:
    # the cutoff is months_ago(now, recent_months) — a record exactly at
    # the cutoff is inside (date >= cutoff), mirroring analyze's gate
    from gmail_labeler.analyze import months_ago

    cutoff = months_ago(NOW, 24)
    result = _build(
        [
            _rec(1, ("A",), date=cutoff),
            _rec(2, ("A",), date=cutoff - timedelta(seconds=1)),
        ]
    )
    weights = {e.id: e.weight for e in result.examples}
    assert weights["0000000000000001"] == 1.0
    assert weights["0000000000000002"] == 0.1


def test_weighting_params_ride_in_manifest() -> None:
    # no ad-hoc decay constants: the parameters are named in the
    # versioned format, so changing them changes the manifest
    recs = [_rec(1, ("A",))]
    m1 = _build(recs).manifest
    m2 = _build(recs, recent_months=12, old_weight=0.25).manifest
    assert (m1.recent_months, m1.old_weight) == (24, 0.1)
    assert (m2.recent_months, m2.old_weight) == (12, 0.25)
    assert m1.content_hash != m2.content_hash


def test_weighting_coerced_int_vs_float_identical_hash() -> None:
    # old_weight=1 (int) vs 1.0 (float) mint the same manifest (coerced at
    # mint), and old_weight=1 round-trips through from_json (which requires
    # a float) — the builder can parse its own artifact (deepseek minor)
    recs = [_rec(1, ("A",))]
    m_int = _build(recs, old_weight=1).manifest
    m_float = _build(recs, old_weight=1.0).manifest
    assert m_int.content_hash == m_float.content_hash
    assert m_int.old_weight == 1.0  # coerced to float at mint
    assert SplitManifest.from_json(m_int.to_json()) == m_int


def test_old_weight_bounds() -> None:
    import pytest

    with pytest.raises(ValueError, match="old_weight"):
        _build([_rec(1, ("A",))], old_weight=1.5)
    with pytest.raises(ValueError, match="old_weight"):
        _build([_rec(1, ("A",))], old_weight=-0.1)
    # a bool is not a weight: True would pass 0.0 <= x <= 1.0 (True == 1.0)
    # and mint weight 1.0 — reject it explicitly (pinned)
    with pytest.raises(ValueError, match="old_weight"):
        _build([_rec(1, ("A",))], old_weight=True)
    # 0.0 is a legitimate hard window
    result = _build(
        [_rec(1, ("A",), date=datetime(2005, 1, 1, tzinfo=UTC))], old_weight=0.0
    )
    assert result.examples[0].weight == 0.0


def test_manifest_mode_uses_manifest_weighting() -> None:
    # the manifest's parameters are authoritative — the caller's ignored
    recs = [_rec(1, ("A",), date=datetime(2005, 1, 1, tzinfo=UTC))]
    manifest = _build(recs, recent_months=24, old_weight=0.1).manifest
    result = _build(recs, manifest=manifest, recent_months=12, old_weight=0.5)
    assert result.examples[0].weight == 0.1
    assert result.manifest == manifest


def test_manifest_mode_reproducible_across_now() -> None:
    # the weighting anchor (weight_cutoff) rides in the manifest — a
    # rebuild with a different `now` gives identical examples. The record
    # is dated INSIDE the frozen window (2025-06-01 >= 2024-10-30 cutoff)
    # but OUTSIDE the now+400d window (2025-06-01 < 2025-11-04 cutoff), so
    # it discriminates: a recomputed (non-frozen) cutoff would flip it
    # 1.0 → 0.1 (deepseek major — the 2005 record couldn't discriminate).
    recs = [_rec(1, ("A",), date=datetime(2025, 6, 1, tzinfo=UTC))]
    manifest = _build(recs).manifest
    r1 = _build(recs, manifest=manifest)
    r2 = _build(recs, manifest=manifest, now=NOW + timedelta(days=400))
    assert r1.examples == r2.examples
    assert r1.examples[0].weight == 1.0  # frozen cutoff keeps it in-window


def test_weight_cutoff_normalized_to_utc() -> None:
    # the same `now` instant in another (fixed) zone gives the same
    # manifest hash — scoped to fixed-offset/UTC-normalized inputs (a DST
    # zone's offset change is out of scope; callers should pass `now` in
    # UTC for cross-host stability)
    from datetime import timezone

    recs = [_rec(1, ("A",))]
    m1 = _build(recs).manifest
    m2 = _build(recs, now=NOW.astimezone(timezone(timedelta(hours=2)))).manifest
    assert m1.content_hash == m2.content_hash


# ---- empty corpus ----


def test_empty_corpus() -> None:
    result = _build([])
    assert result.examples == ()
    assert result.manifest.vocabulary == ()
    assert result.manifest.train_ids == () and result.manifest.test_ids == ()
