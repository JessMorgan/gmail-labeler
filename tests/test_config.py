from __future__ import annotations

from pathlib import Path

import pytest

from gmail_labeler.config import Config, ConfigError, find_config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, body: str, name: str = "config.toml") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_empty_config_is_all_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "")
    monkeypatch.chdir(tmp_path)
    loaded = load_config()
    # Everything default EXCEPT the relative default paths (corpus.dir,
    # model.checkpoint_dir), which load anchors to the config file's
    # directory (never the invocation cwd).
    assert loaded.corpus.dir == tmp_path / "corpus"
    assert loaded.model.checkpoint_dir == tmp_path / "checkpoints/latest"
    bare = Config()
    assert loaded.model_dump() == bare.model_dump() | {
        "corpus": bare.corpus.model_dump() | {"dir": loaded.corpus.dir},
        "model": bare.model.model_dump()
        | {"checkpoint_dir": loaded.model.checkpoint_dir},
    }


def test_example_defaults_are_literal_pinned() -> None:
    # The golden comparison must not be self-referential (Config vs Config);
    # pin the actual expected values so a changed default goes red HERE.
    expected = {
        "credentials_path": None,
        "token_file": str(Path.home() / ".config/gmail-labeler/token.json"),
        "token_backend": "keyring",
        "history_poll_seconds": 30,
    }
    dumped = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in Config().gmail.model_dump().items()
    }
    assert dumped == expected


def test_example_toml_matches_code_defaults(tmp_path: Path) -> None:
    # Every value example.toml STATES must equal the code default. Both sides
    # go through the identical load pipeline (expansion + anchoring), so drift
    # in either direction — a changed default or an edited example value —
    # turns this red.
    example = tmp_path / "config.toml"
    example.write_bytes((REPO_ROOT / "example.toml").read_bytes())
    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    assert load_config(str(example)) == load_config(str(empty))


def test_example_toml_bands_example_uncomments_to_a_valid_override(
    tmp_path: Path,
) -> None:
    # the commented per-label example in example.toml must be
    # UNCOMMENTABLE: uncommenting the documented subtable lines yields
    # valid TOML that loads with the documented override (the example
    # sits BELOW the two defaults — a layout above them would
    # duplicate the keys and parse-fail)
    from gmail_labeler.config import BandOverride

    text = (REPO_ROOT / "example.toml").read_text(encoding="utf-8")
    out: list[str] = []
    in_bands = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            in_bands = stripped == "[bands]"
        if in_bands and (
            stripped.startswith("# [bands.")
            or stripped.startswith("# min_high_precision = ")
            or stripped.startswith("# min_med_precision = ")
        ):
            out.append(line.removeprefix("# "))
        else:
            out.append(line)
    p = tmp_path / "config.toml"
    p.write_text("".join(out), encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.bands.min_high_precision == 0.9
    assert cfg.bands.min_med_precision == 0.7
    assert cfg.bands.per_label == {
        "Money/Bills": BandOverride(min_high_precision=0.9, min_med_precision=0.8)
    }


def test_example_toml_graduation_example_uncomments_to_a_valid_override(
    tmp_path: Path,
) -> None:
    # the commented per-label example in example.toml must be
    # UNCOMMENTABLE: uncommenting the documented subtable lines yields
    # valid TOML that loads with the documented override (the example
    # sits BELOW the two defaults — a layout above them would
    # duplicate the keys and parse-fail)
    from gmail_labeler.config import GateOverride

    text = (REPO_ROOT / "example.toml").read_text(encoding="utf-8")
    out: list[str] = []
    in_graduation = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            in_graduation = stripped == "[graduation]"
        if in_graduation and (
            stripped.startswith("# [graduation.")
            or stripped.startswith("# min_examples = ")
            or stripped.startswith("# min_precision = ")
        ):
            out.append(line.removeprefix("# "))
        else:
            out.append(line)
    p = tmp_path / "config.toml"
    p.write_text("".join(out), encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.graduation.min_examples == 10
    assert cfg.graduation.min_precision == 0.9
    assert cfg.graduation.per_label == {
        "Money/Bills": GateOverride(min_examples=20, min_precision=0.95)
    }


def test_full_config_parses(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
[gmail]
credentials_path = "client.json"
token_backend = "file"
token_file = "tok.json"
history_poll_seconds = 120
[labels]
applied = "robot"
high = "robot-high"
med = "robot-med"
managed = ["Shipping", "Money/Bills"]
[corpus]
dir = "~/mail-corpus"
[gyb]
path = "backup/gmail"
""",
    )
    cfg = load_config(str(p))
    assert cfg.gmail.history_poll_seconds == 120
    assert cfg.labels.managed == ["Shipping", "Money/Bills"]
    # Relative paths resolve against the CONFIG FILE dir, not cwd.
    assert cfg.gmail.credentials_path == tmp_path / "client.json"
    assert cfg.gyb.path == tmp_path / "backup/gmail"
    assert str(cfg.corpus.dir).startswith("/") and "~" not in str(cfg.corpus.dir)


def test_values_are_stored_normalized(tmp_path: Path) -> None:
    # A padded Managed Label that never matches Gmail's real name would
    # silently never apply (skip-present) — normalization is the contract.
    p = _write(
        tmp_path,
        '[labels]\napplied = " ai-applied "\nmanaged = [" Shipping ", "Money/Bills"]\n',
    )
    cfg = load_config(str(p))
    assert cfg.labels.applied == "ai-applied"
    assert cfg.labels.managed == ["Shipping", "Money/Bills"]


def test_empty_path_value_is_typed_error(tmp_path: Path) -> None:
    p = _write(tmp_path, '[corpus]\ndir = ""\n')
    with pytest.raises(ConfigError, match="must be non-empty"):
        load_config(str(p))


def test_all_section_defaults_are_literal_pinned() -> None:
    # Golden comparison can't see drift on None-defaults that example.toml
    # doesn't state; pin every section's actual default values.
    assert Config().corpus.dir == Path("./corpus")
    assert Config().gyb.path is None
    assert Config().model.model_dump() == {
        "encoder": "deberta-v3-base",
        "encoder_path": None,
        "checkpoint_dir": Path("./checkpoints/latest"),
    }
    assert Config().labels.model_dump() == {
        "applied": "ai-applied",
        "high": "ai-high",
        "med": "ai-med",
        "managed": [],
    }
    assert Config().bands.model_dump() == {
        "min_high_precision": 0.9,
        "min_med_precision": 0.7,
        "per_label": {},
    }
    assert Config().graduation.model_dump() == {
        "min_examples": 10,
        "min_precision": 0.9,
        "per_label": {},
    }


def test_instances_do_not_share_submodels() -> None:
    a, b = Config(), Config()
    assert a.gmail is not b.gmail
    assert a.corpus is not b.corpus
    assert a.gyb is not b.gyb


def test_expansion_not_persistent_across_homes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "")
    monkeypatch.chdir(tmp_path)
    for home in ("h1", "h2"):
        monkeypatch.setenv("HOME", str(tmp_path / home))
        cfg = load_config()
        assert str(cfg.gmail.token_file).startswith(str(tmp_path / home))


def test_unknown_user_in_path_is_typed_error(tmp_path: Path) -> None:
    p = _write(tmp_path, '[corpus]\ndir = "~nosuchuser9999/x"\n')
    with pytest.raises(ConfigError, match="cannot expand home"):
        load_config(str(p))


def test_utf16_config_is_typed_error(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_bytes("[gmail]\nhistory_poll_seconds = 60\n".encode("utf-16"))
    with pytest.raises(ConfigError, match="not valid UTF-8"):
        load_config(str(p))


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, "[gmail]\nbatch_size = 5\n")
    with pytest.raises(ConfigError, match=r"batch_size|Extra"):
        load_config(str(p))


# ---- [bands] (ticket #21: per-label band overrides) ----


def test_bands_defaults_are_literal_pinned() -> None:
    # the defaults ride the named constants — pin the literal values so
    # a changed default goes red HERE (and in example.toml's match test)
    from gmail_labeler.thresholds import (
        DEFAULT_MIN_HIGH_PRECISION,
        DEFAULT_MIN_MED_PRECISION,
    )

    bands = Config().bands
    assert bands.min_high_precision == DEFAULT_MIN_HIGH_PRECISION == 0.9
    assert bands.min_med_precision == DEFAULT_MIN_MED_PRECISION == 0.7
    assert bands.per_label == {}


def test_bands_per_label_override_parses(tmp_path: Path) -> None:
    # a per-label subtable (label names may carry slashes — quoted TOML
    # keys); only the stated target is overridden, the other stays None
    p = _write(
        tmp_path,
        "[bands]\nmin_high_precision = 0.8\n\n"
        '[bands."Money/Bills"]\nmin_high_precision = 0.9\n',
    )
    cfg = load_config(str(p))
    assert cfg.bands.min_high_precision == 0.8
    assert cfg.bands.min_med_precision == 0.7
    assert set(cfg.bands.per_label) == {"Money/Bills"}
    override = cfg.bands.per_label["Money/Bills"]
    assert override.min_high_precision == 0.9
    assert override.min_med_precision is None


def test_bands_override_unknown_key_rejected(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        '[bands."Money/Bills"]\nmin_high = 0.9\n',
    )
    with pytest.raises(ConfigError, match="min_high"):
        load_config(str(p))


def test_bands_non_tenth_target_rejected(tmp_path: Path) -> None:
    # the tenth-grid stance (#20): a target between grid points refuses
    for section in (
        "[bands]\nmin_high_precision = 0.93\n",
        "[bands]\nmin_med_precision = 0.65\n",
        '[bands."Bills"]\nmin_high_precision = 0.93\n',
    ):
        p = _write(tmp_path, section)
        with pytest.raises(ConfigError, match="tenth"):
            load_config(str(p))


def test_bands_out_of_range_rejected(tmp_path: Path) -> None:
    for section in (
        "[bands]\nmin_high_precision = 1.5\n",
        "[bands]\nmin_med_precision = -0.1\n",
        '[bands."Bills"]\nmin_med_precision = 1.1\n',
    ):
        p = _write(tmp_path, section)
        with pytest.raises(ConfigError, match="must be in"):
            load_config(str(p))


def test_bands_string_target_rejected(tmp_path: Path) -> None:
    # a string is not a target: pydantic's lax float coercion would
    # turn "0.9" into 0.9 — refused on the RAW value, named (numeric
    # and non-numeric strings alike), never silently coerced
    for section in (
        '[bands]\nmin_high_precision = "0.9"\n',
        '[bands]\nmin_med_precision = "abc"\n',
        '[bands."Bills"]\nmin_high_precision = "0.9"\n',
    ):
        p = _write(tmp_path, section)
        with pytest.raises(ConfigError, match="not a string"):
            load_config(str(p))


def test_bands_raw_per_label_key_is_reserved(tmp_path: Path) -> None:
    # the per_label back door: a raw per_label table in the [bands]
    # table bypassed every custom refusal (bool→1.0, non-tenth, empty
    # table, silent discard against the label-path form). It is a
    # reserved field name — refused with the label-path spelling
    # shown, so an operator who wrote it gets guidance
    for section in (
        '[bands]\nper_label = { "Bills" = { min_high_precision = true } }\n',
        '[bands]\nper_label = { "Bills" = { min_high_precision = 0.93 } }\n',
        '[bands]\nper_label = { "Bills" = {} }\n',
    ):
        p = _write(tmp_path, section)
        with pytest.raises(ConfigError, match="reserved field name"):
            load_config(str(p))
        p = _write(
            tmp_path,
            '[bands]\nper_label = { "Bills" = { min_high_precision = 0.9 } }\n'
            '[bands."Bills"]\nmin_med_precision = 0.8\n',
        )
        with pytest.raises(ConfigError, match="reserved field name"):
            load_config(str(p))
        # the guidance shows the label-path spelling
        p = _write(
            tmp_path,
            '[bands]\nper_label = { "Bills" = { min_high_precision = 0.9 } }\n',
        )
        with pytest.raises(ConfigError, match=r'bands\."Some/Label"'):
            load_config(str(p))


def test_bands_empty_raw_per_label_table_refused(tmp_path: Path) -> None:
    # per_label = {} (an EMPTY dict) is refused like every other raw
    # form: all() over an empty mapping is vacuously true, so the typed
    # check alone would accept it (the inherited #21-surface bug)
    p = _write(tmp_path, "[bands]\nper_label = {}\n")
    with pytest.raises(ConfigError, match="reserved field name"):
        load_config(str(p))


def test_bands_handler_failure_still_collects_all_problems(
    tmp_path: Path,
) -> None:
    # aggregation completeness: a handler/coercion failure must not skip
    # the collection — the range problem on the OTHER default is
    # collected in the same error (the raw scan runs before the handler)
    p = _write(
        tmp_path,
        "[bands]\nmin_high_precision = 1.5\nmin_med_precision = [0.7]\n",
    )
    with pytest.raises(ConfigError) as ei:
        load_config(str(p))
    msg = str(ei.value)
    assert "must be in [0.0, 1.0]" in msg  # the range problem (raw scan)
    assert "got list" in msg  # the raw-type problem


def test_bands_programmatic_per_label_gets_the_same_checks() -> None:
    # defense in depth: the declared per_label field (the programmatic
    # API — typed BandOverride values) gets the SAME tenth-grid, range,
    # and empty-override checks as the label-path form
    from gmail_labeler.config import BandOverride, BandsConfig

    good = BandsConfig(per_label={"Bills": BandOverride(min_high_precision=0.5)})
    assert good.per_label["Bills"].min_high_precision == 0.5
    for bad in (
        BandOverride(min_high_precision=0.93),  # non-tenth
        BandOverride(min_high_precision=1.5),  # out of range
        BandOverride(),  # empty override
    ):
        with pytest.raises(ValueError, match=r"bands\.Bills"):
            BandsConfig(per_label={"Bills": bad})


def test_bands_same_label_named_twice_refused(tmp_path: Path) -> None:
    # the same label named more than once — a typed per_label entry
    # AND a [bands."<key>"] subtable, or padded keys that strip to
    # one — is refused, not silently shadowed (the LabelsConfig
    # uniqueness precedent, post-clean)
    from gmail_labeler.config import BandOverride, BandsConfig

    with pytest.raises(ValueError, match="duplicate"):
        BandsConfig.model_validate(
            {
                "per_label": {"Bills": BandOverride(min_high_precision=0.5)},
                "Bills": {"min_high_precision": 0.9},
            }
        )
    p = _write(
        tmp_path,
        '[bands."Foo"]\nmin_high_precision = 0.9\n'
        '[bands."Foo "]\nmin_med_precision = 0.8\n',
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(str(p))


def test_bands_label_key_stripped_and_bounded(tmp_path: Path) -> None:
    # a whitespace-padded key is stored stripped (the LabelsConfig
    # stripping precedent — a padded key must not survive to
    # calibration); blank-after-strip and over-length keys refuse
    p = _write(tmp_path, '[bands."  Bills  "]\nmin_high_precision = 0.9\n')
    cfg = load_config(str(p))
    assert set(cfg.bands.per_label) == {"Bills"}
    p = _write(tmp_path, '[bands."   "]\nmin_high_precision = 0.9\n')
    with pytest.raises(ConfigError, match="non-empty after stripping"):
        load_config(str(p))
    long_key = "B" * 257
    p = _write(tmp_path, f'[bands."{long_key}"]\nmin_high_precision = 0.9\n')
    with pytest.raises(ConfigError, match="longer than"):
        load_config(str(p))


def test_bands_long_slash_nested_label_path_is_accepted(tmp_path: Path) -> None:
    # the label ACCEPTANCE bound (256) is separate from the error-ECHO
    # bound (40): GYB corpora carry deeply slash-nested Label Paths,
    # and a valid 60-char path must be ACCEPTED and gated, not refused
    long_path = "Money/Purchases/Online/Subscriptions/Streaming/Services"
    assert len(long_path) == 55
    p = _write(tmp_path, f'[bands."{long_path}"]\nmin_high_precision = 0.9\n')
    cfg = load_config(str(p))
    assert set(cfg.bands.per_label) == {long_path}
    assert cfg.bands.per_label[long_path].min_high_precision == 0.9


def test_bands_bool_target_rejected(tmp_path: Path) -> None:
    # a TOML bool is not a target: pydantic's lax float coercion would
    # turn true into 1.0 (false into 0.0) — refused on the RAW value,
    # never silently coerced (the core's "a bool is not a target")
    for section in (
        "[bands]\nmin_high_precision = true\n",
        "[bands]\nmin_med_precision = false\n",
        '[bands."Bills"]\nmin_high_precision = true\n',
    ):
        p = _write(tmp_path, section)
        with pytest.raises(ConfigError, match="not a bool"):
            load_config(str(p))


def test_bands_empty_override_table_rejected(tmp_path: Path) -> None:
    # an empty [bands."<label>"] table sets nothing — a no-op that
    # would silently calibrate with the defaults; refused, not dropped
    p = _write(tmp_path, '[bands."Bills"]\n')
    with pytest.raises(ConfigError, match="at least one target"):
        load_config(str(p))


def test_all_band_problems_reported_together(tmp_path: Path) -> None:
    # aggregation, not first-fail (the LabelsConfig doctrine): a bad
    # default + two bad per-label tables are ALL named in one error
    p = _write(
        tmp_path,
        "[bands]\nmin_high_precision = 0.93\n"
        '[bands."Bills"]\nmin_med_precision = 1.5\n'
        '[bands."Shipping"]\nmin_high = 0.9\n',
    )
    with pytest.raises(ConfigError) as ei:
        load_config(str(p))
    msg = str(ei.value)
    assert "tenth" in msg  # the bad default
    # the out-of-range override (the collecting validator's range
    # message — the ge/le field constraints are gone, so this problem
    # can no longer skip the collection)
    assert "Bills" in msg and "must be in [0.0, 1.0]" in msg
    assert "Shipping" in msg and "min_high" in msg  # the unknown key


def test_all_band_problems_reported_together_including_bool(tmp_path: Path) -> None:
    # the aggregation mutation pin: a RAW bool used to fire the
    # separate pre-scan (which raised before the collection ran) —
    # now bool + non-tenth + out-of-range + unknown key are ALL named
    # in one error by the single raw-seeing validator
    p = _write(
        tmp_path,
        "[bands]\nmin_high_precision = true\nmin_med_precision = 0.93\n"
        '[bands."Bills"]\nmin_med_precision = 1.5\n'
        '[bands."Shipping"]\nmin_high = 0.9\n',
    )
    with pytest.raises(ConfigError) as ei:
        load_config(str(p))
    msg = str(ei.value)
    assert "not a bool" in msg  # the raw bool
    assert "tenth" in msg  # the non-tenth default
    assert "must be in [0.0, 1.0]" in msg  # the out-of-range override
    assert "Shipping" in msg and "min_high" in msg  # the unknown key


def test_bands_scalar_override_rejected(tmp_path: Path) -> None:
    # a per-label override must be a TABLE of targets — a scalar under a
    # label name is a typo, refused (not silently dropped)
    p = _write(tmp_path, '[bands]\n"Bills" = 0.9\n')
    with pytest.raises(ConfigError, match="Bills"):
        load_config(str(p))


# ---- [graduation] (ticket #22: per-label graduation gates) ----


def test_graduation_defaults_are_literal_pinned() -> None:
    # the defaults ride the named constants — pin the literal values so
    # a changed default goes red HERE (and in example.toml's match test)
    from gmail_labeler.graduation import (
        DEFAULT_MIN_EXAMPLES,
        DEFAULT_MIN_PRECISION,
    )

    grad = Config().graduation
    assert grad.min_examples == DEFAULT_MIN_EXAMPLES == 10
    assert grad.min_precision == DEFAULT_MIN_PRECISION == 0.9
    assert grad.per_label == {}


def test_graduation_per_label_override_parses(tmp_path: Path) -> None:
    # a per-label subtable (label names may carry slashes — quoted TOML
    # keys); only the stated gate is overridden, the other stays None
    p = _write(
        tmp_path,
        "[graduation]\nmin_examples = 5\n\n"
        '[graduation."Money/Bills"]\nmin_precision = 0.95\n',
    )
    cfg = load_config(str(p))
    assert cfg.graduation.min_examples == 5
    assert cfg.graduation.min_precision == 0.9
    assert set(cfg.graduation.per_label) == {"Money/Bills"}
    override = cfg.graduation.per_label["Money/Bills"]
    assert override.min_examples is None
    assert override.min_precision == 0.95


def test_graduation_non_tenth_precision_is_allowed(tmp_path: Path) -> None:
    # the documented asymmetry with [bands]: a gate precision target is
    # COMPARED against measured values (any rational tp/(tp+fp)), never
    # looked up in the candidate grid — a non-tenth like 0.93 is a
    # legitimate gate (the tenth-grid stance does NOT apply here)
    p = _write(
        tmp_path,
        "[graduation]\nmin_precision = 0.93\n"
        '[graduation."Bills"]\nmin_precision = 0.875\n',
    )
    cfg = load_config(str(p))
    assert cfg.graduation.min_precision == 0.93
    assert cfg.graduation.per_label["Bills"].min_precision == 0.875


def test_graduation_override_unknown_key_rejected(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        '[graduation."Money/Bills"]\nmin_example = 10\n',
    )
    with pytest.raises(ConfigError, match="min_example"):
        load_config(str(p))


def test_graduation_out_of_range_rejected(tmp_path: Path) -> None:
    for section in (
        "[graduation]\nmin_precision = 1.5\n",
        "[graduation]\nmin_precision = -0.1\n",
        "[graduation]\nmin_examples = -1\n",
        '[graduation."Bills"]\nmin_precision = 1.1\n',
        '[graduation."Bills"]\nmin_examples = -3\n',
    ):
        p = _write(tmp_path, section)
        with pytest.raises(ConfigError, match=r"must be in|must be >="):
            load_config(str(p))


def test_graduation_string_value_rejected(tmp_path: Path) -> None:
    # a string is not a gate value: pydantic's lax int/float coercion
    # would turn "10" into 10 / "0.9" into 0.9 — refused on the RAW
    # value, named, never silently coerced
    for section in (
        '[graduation]\nmin_examples = "10"\n',
        '[graduation]\nmin_precision = "0.9"\n',
        '[graduation]\nmin_precision = "abc"\n',
        '[graduation."Bills"]\nmin_examples = "10"\n',
    ):
        p = _write(tmp_path, section)
        with pytest.raises(ConfigError, match="not a string"):
            load_config(str(p))


def test_graduation_bool_value_rejected(tmp_path: Path) -> None:
    # a TOML bool is not a gate value: pydantic's lax coercion would
    # turn true into 1 / 1.0 — refused on the RAW value, never silently
    # coerced (the core's "a bool is not a count/target")
    for section in (
        "[graduation]\nmin_examples = true\n",
        "[graduation]\nmin_precision = false\n",
        '[graduation."Bills"]\nmin_precision = true\n',
    ):
        p = _write(tmp_path, section)
        with pytest.raises(ConfigError, match="not a bool"):
            load_config(str(p))


def test_graduation_raw_per_label_key_is_reserved(tmp_path: Path) -> None:
    # the per_label back door: a raw per_label table in the [graduation]
    # table bypassed every custom refusal (bool→1, negative, empty
    # table, silent discard against the label-path form). It is a
    # reserved field name — refused with the label-path spelling shown
    for section in (
        '[graduation]\nper_label = { "Bills" = { min_examples = true } }\n',
        '[graduation]\nper_label = { "Bills" = { min_precision = 1.5 } }\n',
        '[graduation]\nper_label = { "Bills" = {} }\n',
    ):
        p = _write(tmp_path, section)
        with pytest.raises(ConfigError, match="reserved field name"):
            load_config(str(p))
        p = _write(
            tmp_path,
            '[graduation]\nper_label = { "Bills" = { min_examples = 5 } }\n'
            '[graduation."Bills"]\nmin_precision = 0.95\n',
        )
        with pytest.raises(ConfigError, match="reserved field name"):
            load_config(str(p))
        # the guidance shows the label-path spelling
        p = _write(
            tmp_path,
            '[graduation]\nper_label = { "Bills" = { min_examples = 5 } }\n',
        )
        with pytest.raises(ConfigError, match=r'graduation\."Some/Label"'):
            load_config(str(p))


def test_graduation_empty_raw_per_label_table_refused(tmp_path: Path) -> None:
    # per_label = {} (an EMPTY dict) is refused like every other raw
    # form: all() over an empty mapping is vacuously true, so the typed
    # check alone would accept it (the inherited #21-surface bug)
    p = _write(tmp_path, "[graduation]\nper_label = {}\n")
    with pytest.raises(ConfigError, match="reserved field name"):
        load_config(str(p))


def test_graduation_handler_failure_still_collects_all_problems(
    tmp_path: Path,
) -> None:
    # aggregation completeness: a handler/coercion failure must not skip
    # the collection — the range problem on the OTHER default is
    # collected in the same error (the raw scan runs before the handler)
    p = _write(
        tmp_path,
        "[graduation]\nmin_examples = -1\nmin_precision = [0.9]\n",
    )
    with pytest.raises(ConfigError) as ei:
        load_config(str(p))
    msg = str(ei.value)
    assert "must be >=" in msg  # the range problem (raw scan)
    assert "got list" in msg  # the raw-type problem


def test_graduation_programmatic_per_label_gets_the_same_checks() -> None:
    # defense in depth: the declared per_label field (the programmatic
    # API — typed GateOverride values) gets the SAME range and
    # empty-override checks as the label-path form
    from gmail_labeler.config import GateOverride, GraduationConfig

    good = GraduationConfig(per_label={"Bills": GateOverride(min_examples=5)})
    assert good.per_label["Bills"].min_examples == 5
    for bad in (
        GateOverride(min_precision=1.5),  # out of range
        GateOverride(min_examples=-1),  # negative
        GateOverride(),  # empty override
    ):
        with pytest.raises(ValueError, match=r"graduation\.Bills"):
            GraduationConfig(per_label={"Bills": bad})


def test_graduation_same_label_named_twice_refused(tmp_path: Path) -> None:
    # the same label named more than once — a typed per_label entry AND
    # a [graduation."<key>"] subtable, or padded keys that strip to one
    # — is refused, not silently shadowed (the LabelsConfig uniqueness
    # precedent, post-clean)
    from gmail_labeler.config import GateOverride, GraduationConfig

    with pytest.raises(ValueError, match="duplicate"):
        GraduationConfig.model_validate(
            {
                "per_label": {"Bills": GateOverride(min_examples=5)},
                "Bills": {"min_examples": 10},
            }
        )
    p = _write(
        tmp_path,
        '[graduation."Foo"]\nmin_examples = 5\n'
        '[graduation."Foo "]\nmin_precision = 0.95\n',
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(str(p))


def test_graduation_label_key_stripped_and_bounded(tmp_path: Path) -> None:
    # a whitespace-padded key is stored stripped (the LabelsConfig
    # stripping precedent — a padded key must not survive to the gate
    # evaluation); blank-after-strip and over-length keys refuse
    p = _write(tmp_path, '[graduation."  Bills  "]\nmin_examples = 5\n')
    cfg = load_config(str(p))
    assert set(cfg.graduation.per_label) == {"Bills"}
    p = _write(tmp_path, '[graduation."   "]\nmin_examples = 5\n')
    with pytest.raises(ConfigError, match="non-empty after stripping"):
        load_config(str(p))
    long_key = "B" * 257
    p = _write(tmp_path, f'[graduation."{long_key}"]\nmin_examples = 5\n')
    with pytest.raises(ConfigError, match="longer than"):
        load_config(str(p))


def test_graduation_long_slash_nested_label_path_is_accepted(tmp_path: Path) -> None:
    # the label ACCEPTANCE bound (256) is separate from the error-ECHO
    # bound (40): GYB corpora carry deeply slash-nested Label Paths,
    # and a valid 60-char path must be ACCEPTED and gated, not refused
    long_path = "Money/Purchases/Online/Subscriptions/Streaming/Services"
    assert len(long_path) == 55
    p = _write(tmp_path, f'[graduation."{long_path}"]\nmin_examples = 5\n')
    cfg = load_config(str(p))
    assert set(cfg.graduation.per_label) == {long_path}
    assert cfg.graduation.per_label[long_path].min_examples == 5


def test_graduation_scalar_override_rejected(tmp_path: Path) -> None:
    # a per-label override must be a TABLE of gates — a scalar under a
    # label name is a typo, refused (not silently dropped)
    p = _write(tmp_path, '[graduation]\n"Bills" = 10\n')
    with pytest.raises(ConfigError, match="Bills"):
        load_config(str(p))


def test_graduation_empty_override_table_rejected(tmp_path: Path) -> None:
    # an empty [graduation."<label>"] table sets nothing — a no-op that
    # would silently gate with the defaults; refused, not dropped
    p = _write(tmp_path, '[graduation."Bills"]\n')
    with pytest.raises(ConfigError, match="at least one gate"):
        load_config(str(p))


def test_all_graduation_problems_reported_together(tmp_path: Path) -> None:
    # aggregation, not first-fail (the LabelsConfig doctrine): a bad
    # default + two bad per-label tables are ALL named in one error
    p = _write(
        tmp_path,
        "[graduation]\nmin_precision = 1.5\n"
        '[graduation."Bills"]\nmin_examples = -1\n'
        '[graduation."Shipping"]\nmin_example = 10\n',
    )
    with pytest.raises(ConfigError) as ei:
        load_config(str(p))
    msg = str(ei.value)
    assert "must be in [0.0, 1.0]" in msg  # the bad default
    assert "Bills" in msg and "must be >=" in msg  # the negative override
    assert "Shipping" in msg and "min_example" in msg  # the unknown key


def test_all_graduation_problems_reported_together_including_bool(
    tmp_path: Path,
) -> None:
    # the aggregation mutation pin: a RAW bool used to fire a separate
    # pre-scan (which raised before the collection ran) — now bool +
    # out-of-range + unknown key are ALL named in one error by the
    # single raw-seeing validator
    p = _write(
        tmp_path,
        "[graduation]\nmin_examples = true\nmin_precision = 1.5\n"
        '[graduation."Bills"]\nmin_examples = -1\n'
        '[graduation."Shipping"]\nmin_example = 10\n',
    )
    with pytest.raises(ConfigError) as ei:
        load_config(str(p))
    msg = str(ei.value)
    assert "not a bool" in msg  # the raw bool
    assert "must be in [0.0, 1.0]" in msg  # the out-of-range default
    assert "Bills" in msg and "must be >=" in msg  # the negative override
    assert "Shipping" in msg and "min_example" in msg  # the unknown key


def test_duplicate_tier_names_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, '[labels]\napplied = "x"\nhigh = "x"\n')
    with pytest.raises(ConfigError, match="distinct"):
        load_config(str(p))


def test_blank_tier_name_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, '[labels]\napplied = "  "\n')
    with pytest.raises(ConfigError, match="non-empty"):
        load_config(str(p))


def test_all_label_problems_reported_together(tmp_path: Path) -> None:
    p = _write(
        tmp_path, '[labels]\napplied = "x"\nhigh = "x"\nmanaged = ["A", "A", ""]\n'
    )
    with pytest.raises(ConfigError) as ei:
        load_config(str(p))
    msg = str(ei.value)
    assert "distinct" in msg and "unique" in msg  # aggregation, not first-fail


def test_managed_blank_and_dupes_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, '[labels]\nmanaged = ["Shipping", "  "]\n')
    with pytest.raises(ConfigError, match="non-empty"):
        load_config(str(p))
    p = _write(tmp_path, '[labels]\nmanaged = ["A", "A"]\n')
    with pytest.raises(ConfigError, match="unique"):
        load_config(str(p))


def test_poll_bounds_and_backend_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, "[gmail]\nhistory_poll_seconds = 1\n")
    with pytest.raises(ConfigError, match="history_poll_seconds"):
        load_config(str(p))
    p = _write(tmp_path, '[gmail]\ntoken_backend = "vault"\n')
    with pytest.raises(ConfigError, match="token_backend"):
        load_config(str(p))


def test_bad_toml(tmp_path: Path) -> None:
    p = _write(tmp_path, "this is not toml = = =")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(str(p))


def test_find_config_explicit_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    here = _write(tmp_path, "[gmail]\nhistory_poll_seconds = 42\n")
    elsewhere = tmp_path / "other.toml"
    elsewhere.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert find_config(str(elsewhere)) == elsewhere
    assert find_config(None) == here


def test_find_config_explicit_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        find_config(str(tmp_path / "nope.toml"))


def test_find_config_home_fallback_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".config/gmail-labeler").mkdir(parents=True)
    home_cfg = home / ".config/gmail-labeler/config.toml"
    home_cfg.write_text("", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert find_config(None) == home_cfg
    # cwd wins when both exist
    cwd_cfg = _write(tmp_path, "")
    assert find_config(None) == cwd_cfg


def test_find_config_missing_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    with pytest.raises(ConfigError, match=r"no config\.toml found"):
        find_config(None)
