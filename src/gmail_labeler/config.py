"""Config loading and validation (TOML, pydantic).

Discovery order: --config path > ./config.toml > ~/.config/gmail-labeler/config.toml.
Relative paths resolve against the CONFIG FILE's directory (not the invocation
cwd — the corpus directory holds secret mail material and must not silently
materialize wherever the CLI happened to run). Secrets are never inline:
OAuth client and token locations are *paths* (ADR-0004, CODING-STANDARDS
"no inline secrets"). An invalid config is a typed ConfigError listing every
problem, never a partial config object.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, Field, ValidationError, model_validator

# Deliberate edge: config→dataset_build (measured ~4ms import, no cycle —
# the CLI already imports both); is_tenth is the single tenth-grid source.
from .dataset_build import is_tenth

# Deliberate edge: config→graduation (the gate defaults ride the named
# constants — the equality is pinned by test_config; graduation imports
# no config, so the direction is cycle-free — the dataset_build
# precedent).
from .graduation import DEFAULT_MIN_EXAMPLES, DEFAULT_MIN_PRECISION

_CONFIG_FILENAME = "config.toml"
# Bounds (does NOT redact — a short secret pasted as a key or value still
# echoes, truncated) the echo of config keys AND raw values in error
# messages: label keys and echoed raw values are cut to this bound.
_MAX_LOC_SEGMENT = 40
# The label ACCEPTANCE bound (separate from the echo bound above): a
# Label Path may be long — GYB corpora carry deeply slash-nested paths
# (Money/Purchases/Online/Subscriptions/Streaming, ...), and a valid
# path must be ACCEPTED and gated, not refused. The echo bound
# (_MAX_LOC_SEGMENT) only truncates what an error message prints.
_MAX_LABEL_LENGTH = 256

TokenBackend = Literal["keyring", "file"]


class ConfigError(Exception):
    """Config missing, unreadable, or invalid; message lists all problems."""


def _default_token_file() -> Path:
    # default_factory (not a class attribute): pydantic reuses bare mutable/
    # model defaults by reference; a factory also re-reads HOME per instance.
    return Path.home() / ".config/gmail-labeler" / "token.json"


class GmailConfig(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    #: OAuth client JSON from the user's own GCP project (ADR-0004).
    credentials_path: Path | None = None
    token_backend: TokenBackend = "keyring"
    #: Only consulted when token_backend = "file"; 0600 required at use time.
    token_file: Path = Field(default_factory=_default_token_file)
    history_poll_seconds: int = Field(default=30, ge=5, le=3600)


class LabelsConfig(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    #: Umbrella label marking every Application (GLOSSARY: Tier Label).
    applied: str = "ai-applied"
    high: str = "ai-high"
    med: str = "ai-med"
    #: Managed Label names/paths this install may apply (empty = apply nothing;
    #: populated from dataset analyze's graduation candidates).
    managed: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _all_label_problems_together(self) -> Self:
        """One validator collecting EVERY label problem, not first-fail.

        Values are STORED normalized (stripped): a padded Managed Label would
        never match Gmail's real label name, and under additive-only /
        skip-present it would silently never apply. Safe to setattr on self —
        default_factory guarantees per-instance models.
        """
        problems: list[str] = []
        tiers = {
            "applied": self.applied.strip(),
            "high": self.high.strip(),
            "med": self.med.strip(),
        }
        for name, value in tiers.items():
            if not value:
                problems.append(f"labels.{name} must be non-empty")
        if len({tiers["applied"], tiers["high"], tiers["med"]}) != 3:
            problems.append("labels.applied/high/med must be three distinct names")
        cleaned = [s.strip() for s in self.managed]
        if any(not s for s in cleaned):
            problems.append("labels.managed entries must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            problems.append("labels.managed entries must be unique")
        if problems:
            raise ValueError("; ".join(problems))
        object.__setattr__(self, "applied", tiers["applied"])
        object.__setattr__(self, "high", tiers["high"])
        object.__setattr__(self, "med", tiers["med"])
        object.__setattr__(self, "managed", cleaned)
        return self


class CorpusConfig(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    #: Corpus directory (GLOSSARY: Corpus). Bodies and metadata live here;
    #: treat the whole directory as secret material (SECURITY.md).
    dir: Path = Path("./corpus")


class GybConfig(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    #: Root of a Got Your Back backup (contains msg-db.sqlite); optional.
    path: Path | None = None


_TARGET_NAMES = ("min_high_precision", "min_med_precision")


def _shorten_echo(value: object) -> str:
    """The bounded echo of a raw value in an error message (the module
    bound — hygiene, not redaction: a short secret pasted as a value
    still echoes, truncated)."""
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= _MAX_LOC_SEGMENT else text[:_MAX_LOC_SEGMENT] + "…"


def _target_raw_problems(loc: str, value: object) -> list[str]:
    """The raw-type refusal for one target value — the raw-seeing
    validator's job: pydantic's lax float coercion would launder a
    bool into 1.0/0.0 and a string into its number. Accept int/float;
    refuse bool, refuse str (named), refuse everything else."""
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"{loc}: a target is a number, not a bool"]
    if isinstance(value, str):
        return [
            f"{loc}: a target is a number, not a string "
            f"({_shorten_echo(value)!r} — no silent coercion)"
        ]
    if not isinstance(value, (int, float)):
        return [f"{loc}: a target is a number, got {type(value).__name__}"]
    return []


def _band_raw_value_problems(
    loc_prefix: str, sub: dict[str, object], *, check_empty: bool
) -> list[str]:
    """The RAW-value problems for one band target value or subtable:
    the [0.0, 1.0] range, the tenth-grid stance (#20), and — for a
    per-label subtable (``check_empty``) — the empty override. Checked
    on the RAW values (before the handler) so they are collected in
    BOTH paths — the normal path and the handler-raise path (a
    coercion failure must not skip the collection). A value with a RAW
    type problem (bool/str — reported by _target_raw_problems) is
    skipped here (no double-report). The empty check does NOT apply to
    the defaults section: absent keys there mean the code defaults,
    not an empty override."""
    problems: list[str] = []
    for name in _TARGET_NAMES:
        target = sub.get(name)
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            continue  # raw-type problem already reported (or absent)
        if not 0.0 <= target <= 1.0:
            problems.append(f"{loc_prefix}.{name} {target} must be in [0.0, 1.0]")
        elif not is_tenth(target):
            problems.append(
                f"{loc_prefix}.{name} {target} is not a tenth — the bands "
                "ride the candidate grid (the tenth-grid stance, #20)"
            )
    # the empty-override consequence is reported only when the
    # subtable carries NO other keys: a typo'd-key-only subtable
    # (e.g. min_high = 0.9) is empty BECAUSE of the typo — the
    # unknown-key cause (reported by the model_validate step) must
    # come first, not the consequence
    if (
        check_empty
        and not (set(sub) - set(_TARGET_NAMES))
        and all(sub.get(name) is None for name in _TARGET_NAMES)
    ):
        problems.append(
            f"{loc_prefix}: an override must set at least one target "
            "(min_high_precision / min_med_precision)"
        )
    return problems


def _override_value_problems(key: str, override: BandOverride) -> list[str]:
    """The value problems for one collected per-label override from the
    DECLARED field (typed programmatic construction — the raw scan
    cannot see typed values): the [0.0, 1.0] range, the tenth-grid
    stance (#20), and the empty override (a no-op that would silently
    calibrate with the defaults)."""
    problems: list[str] = []
    for name in _TARGET_NAMES:
        target = getattr(override, name)
        if target is None:
            continue
        if not 0.0 <= target <= 1.0:
            problems.append(f"bands.{key}.{name} {target} must be in [0.0, 1.0]")
        elif not is_tenth(target):
            problems.append(
                f"bands.{key}.{name} {target} is not a tenth — the bands "
                "ride the candidate grid (the tenth-grid stance, #20)"
            )
    if override.min_high_precision is None and override.min_med_precision is None:
        problems.append(
            f"bands.{key}: an override must set at least one target "
            "(min_high_precision / min_med_precision)"
        )
    return problems


class BandOverride(BaseModel):
    """One per-label min-precision target override (ticket #21): the
    floor a candidate column's measured precision must meet to earn the
    band. ``None`` = keep the run's default for that band. The raw
    type (bool/str refused), the [0.0, 1.0] range, and the tenth-grid
    stance are checked by the parent's single collecting validator —
    the field constraints are deliberately ABSENT: a field-constraint
    failure would skip the model-level collection (pydantic's rule),
    and the "all problems together" doctrine dies with it."""

    model_config = {"extra": "forbid", "frozen": True}

    min_high_precision: float | None = None
    min_med_precision: float | None = None


class BandsConfig(BaseModel):
    """Per-label Threshold Band calibration (ticket #21): the run's
    default min-precision targets + per-label overrides. The per-label
    entries are the [bands."<label>"] subtables (label names may carry
    slashes — quoted TOML keys); they are collected from the extra
    fields into ``per_label`` by the validator, which refuses an
    override that is not a table of targets (fail loud — a typo'd
    label must not silently calibrate nothing). A raw ``per_label``
    key in the [bands] table is a RESERVED field name — refused with
    the label-path spelling shown: the declared field is the
    validator's collection target and the CLI's read surface, not an
    input surface (typed programmatic construction with BandOverride
    values is the API, and gets the same VALUE checks — range,
    tenth-grid, empty-override; not the raw-type checks, which the
    wrap validator applies to raw input: a bool passed into
    BandOverride is coerced by pydantic before the validator sees it).
    The config
    carries NO path: the artifact path is the CLI's ``eval --bands``
    argument, so the config-file relative-path rule has nothing to
    anchor here."""

    model_config = {"extra": "allow", "frozen": True}

    min_high_precision: float = 0.9
    min_med_precision: float = 0.7
    #: The collected per-label overrides (the CLI reads this). NOT an
    # input surface: a raw ``per_label`` key in the [bands] table is
    # refused (reserved name — see the docstring). A leading underscore
    # is not an option: pydantic would treat it as a private attribute
    # and break the TOML→CLI contract this field exists to serve, so
    # the reserved-name refusal is the guard.
    per_label: dict[str, BandOverride] = Field(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def _band_target_problems_together(cls, data: object, handler, info) -> Self:
        """One validator collecting EVERY band-target problem (the
        LabelsConfig doctrine) — and the ONLY one, on purpose: it sees
        the RAW input (mode="wrap"), so bools and strings are refused
        on the raw value before pydantic's lax float coercion can
        launder them (true → 1.0, "0.9" → 0.9). EVERY problem class —
        raw types, ranges, tenth-grid, empty-override, key hygiene —
        is collected on the RAW values BEFORE the handler, so no
        failure class skips the collection: a field-constraint failure
        (pydantic skips model after-validators when a field constraint
        fires — the ge/le constraints and the separate bool pre-scan
        are gone for exactly this reason) and a handler/coercion
        failure both raise with the full problem list. The tenth-grid
        stance (#20) applies to the defaults AND every per-label
        override; a raw ``per_label`` key is a reserved field name —
        an EMPTY raw table is accepted (it equals the typed default
        {}, the model_dump round-trip form), while ANY non-empty raw
        table WITH UNTYPED VALUES is refused (the typed
        BandOverride-valued programmatic path accepts); a label key
        is stripped, refused blank-after-strip, and
        acceptance-bounded at _MAX_LABEL_LENGTH (the echo bound
        _MAX_LOC_SEGMENT only truncates error messages — a long
        slash-nested Label Path is accepted, not refused)."""
        problems: list[str] = []
        raw = cast("dict[str, object]", data) if isinstance(data, dict) else {}
        raw_pl = raw.get("per_label")
        # an EMPTY raw dict is the typed DEFAULT ({}), accepted — the
        # Config.model_validate(cfg.model_dump()) round-trip produces
        # exactly this form; a NON-EMPTY raw dict with untyped values
        # is the back door and is refused (all() over the typed values
        # is the check)
        pl_is_typed = isinstance(raw_pl, dict) and all(
            isinstance(v, BandOverride) for v in raw_pl.values()
        )
        if "per_label" in raw and not pl_is_typed:
            # the back door: a raw per_label table bypassed every
            # custom refusal (the old pre-scan skipped it, the
            # collection walked only the extras). Reserved name — the
            # label-path spelling is the only per-label form
            problems.append(
                "bands.per_label is a reserved field name — per-label "
                "overrides use the label-path spelling "
                '([bands."Some/Label"]), not a raw per_label table'
            )
        for name in _TARGET_NAMES:
            problems.extend(_target_raw_problems(f"bands.{name}", raw.get(name)))
        problems.extend(_band_raw_value_problems("bands", raw, check_empty=False))
        for label, sub in raw.items():
            if label in (*_TARGET_NAMES, "per_label"):
                continue
            key = label.strip()
            if not key:
                problems.append(
                    f"bands.{_shorten_echo(label)}: a label key must be "
                    "non-empty after stripping"
                )
                continue
            if len(key) > _MAX_LABEL_LENGTH:
                problems.append(
                    f"bands.{_shorten_echo(label)}: a label key is longer "
                    f"than {_MAX_LABEL_LENGTH} chars"
                )
                continue
            if not isinstance(sub, dict):
                # a scalar under a label name is a typo'd/unknown key,
                # not a half-written table — name it as such
                problems.append(
                    f"bands.{key}: unknown/typo'd key \u2014 a per-label "
                    f'override is a subtable of targets ([bands."{key}"])'
                )
                continue
            nested = cast("dict[str, object]", sub)
            for name in _TARGET_NAMES:
                problems.extend(
                    _target_raw_problems(f"bands.{key}.{name}", nested.get(name))
                )
            problems.extend(
                _band_raw_value_problems(f"bands.{key}", nested, check_empty=True)
            )
        try:
            self = handler(data)
        except ValidationError as e:
            # the raw scan already collected EVERY raw-value problem —
            # the handler failure adds the coercion errors, and the
            # combined list is raised (no failure class skips the
            # collection)
            problems.extend(
                f"bands.{_shorten_loc(err['loc'])}: {err['msg']}" for err in e.errors()
            )
            raise ValueError("; ".join(problems)) from None
        per_label: dict[str, BandOverride] = {}
        # the declared field (defense in depth for programmatic
        # construction — the raw scan cannot see typed values): the
        # SAME key hygiene + value checks as the extra subtables
        for label, override in self.per_label.items():
            key = label.strip()
            if not key:
                problems.append(
                    "bands.<blank>: a label key must be non-empty after stripping"
                )
                continue
            if len(key) > _MAX_LABEL_LENGTH:
                problems.append(
                    f"bands.{_shorten_echo(label)}: a label key is longer "
                    f"than {_MAX_LABEL_LENGTH} chars"
                )
                continue
            if key in (*_TARGET_NAMES, "per_label"):
                problems.append(
                    f"bands.{key}: a label key may not be a reserved field name"
                )
                continue
            if key in per_label:
                # a padded key stripping onto an existing one (the
                # LabelsConfig uniqueness precedent, post-clean)
                problems.append(
                    f"bands.{key}: duplicate per-label override — the "
                    "same label is named more than once; refusing to "
                    "silently shadow"
                )
                continue
            problems.extend(_override_value_problems(key, override))
            per_label[key] = override
        for label, sub in (self.__pydantic_extra__ or {}).items():
            if not isinstance(sub, dict):
                continue  # already reported at the raw level
            key = label.strip()
            if not key or len(key) > _MAX_LABEL_LENGTH:
                continue  # already reported at the raw level
            if key in (*_TARGET_NAMES, "per_label"):
                problems.append(
                    f"bands.{key}: a label key may not be a reserved field name"
                )
                continue
            try:
                override = BandOverride.model_validate(sub)
            except ValidationError as e:
                problems.extend(
                    f"bands.{key}.{_shorten_loc(err['loc'])}: {err['msg']}"
                    for err in e.errors()
                )
                continue
            if key in per_label:
                # the same label named more than once: a typed
                # per_label entry AND a [bands."<key>"] subtable, or
                # padded keys that strip to one — silent shadowing
                # violates refuse-never-drop
                problems.append(
                    f"bands.{key}: duplicate per-label override — the "
                    "same label is named more than once; refusing to "
                    "silently shadow"
                )
                continue
            # range/tenth/empty are collected at the RAW level (both
            # paths); the declared-field-only checks are the typed
            # values, which the raw scan cannot see
            per_label[key] = override
        if problems:
            raise ValueError("; ".join(problems))
        if per_label:
            object.__setattr__(self, "per_label", per_label)
        return self


_GATE_NAMES = ("min_examples", "min_precision")


def _min_examples_raw_problems(loc: str, value: object) -> list[str]:
    """The raw-type refusal for one min_examples value — the raw-seeing
    validator's job: pydantic's lax int coercion would launder a bool
    into 1/0 and a string into its number. Accept int; refuse bool,
    refuse str (named), refuse everything else."""
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"{loc}: min_examples is an integer, not a bool"]
    if isinstance(value, str):
        return [
            f"{loc}: min_examples is an integer, not a string "
            f"({_shorten_echo(value)!r} — no silent coercion)"
        ]
    if not isinstance(value, int):
        return [f"{loc}: min_examples is an integer, got {type(value).__name__}"]
    return []


def _min_precision_raw_problems(loc: str, value: object) -> list[str]:
    """The raw-type refusal for one min_precision value — the raw-seeing
    validator's job: pydantic's lax float coercion would launder a bool
    into 1.0/0.0 and a string into its number. Accept int/float; refuse
    bool, refuse str (named), refuse everything else."""
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"{loc}: min_precision is a number, not a bool"]
    if isinstance(value, str):
        return [
            f"{loc}: min_precision is a number, not a string "
            f"({_shorten_echo(value)!r} — no silent coercion)"
        ]
    if not isinstance(value, (int, float)):
        return [f"{loc}: min_precision is a number, got {type(value).__name__}"]
    return []


def _gate_raw_value_problems(
    loc_prefix: str, sub: dict[str, object], *, check_empty: bool
) -> list[str]:
    """The RAW-value problems for one gate value or subtable: the
    min_examples >= 0 range, the min_precision [0.0, 1.0] range, and —
    for a per-label subtable (``check_empty``) — the empty override.
    Checked on the RAW values (before the handler) so they are
    collected in BOTH paths — the normal path and the handler-raise
    path (a coercion failure must not skip the collection). A value
    with a RAW type problem (bool/str — reported by the raw-type
    helpers) is skipped here (no double-report). The empty check does
    NOT apply to the defaults section: absent keys there mean the code
    defaults, not an empty override. NOTE: min_precision is NOT
    tenth-grid (the documented asymmetry with [bands] — a gate target
    is compared against measured values, never looked up in the
    candidate grid)."""
    problems: list[str] = []
    me = sub.get("min_examples")
    if isinstance(me, int) and not isinstance(me, bool) and me < 0:
        problems.append(f"{loc_prefix}.min_examples {me} must be >= 0")
    mp = sub.get("min_precision")
    if (
        isinstance(mp, (int, float))
        and not isinstance(mp, bool)
        and not 0.0 <= mp <= 1.0
    ):
        problems.append(f"{loc_prefix}.min_precision {mp} must be in [0.0, 1.0]")
    # the empty-override consequence is reported only when the
    # subtable carries NO other keys: a typo'd-key-only subtable
    # (e.g. min_example = 10) is empty BECAUSE of the typo — the
    # unknown-key cause (reported by the model_validate step) must
    # come first, not the consequence
    if (
        check_empty
        and not (set(sub) - set(_GATE_NAMES))
        and all(sub.get(name) is None for name in _GATE_NAMES)
    ):
        problems.append(
            f"{loc_prefix}: an override must set at least one gate "
            "(min_examples / min_precision)"
        )
    return problems


def _gate_value_problems(key: str, override: GateOverride) -> list[str]:
    """The value problems for one collected per-label override from the
    DECLARED field (typed programmatic construction — the raw scan
    cannot see typed values): the min_examples >= 0 range, the
    min_precision [0.0, 1.0] range, and the empty override (a no-op
    that would silently gate with the defaults)."""
    problems: list[str] = []
    if override.min_examples is not None and override.min_examples < 0:
        problems.append(
            f"graduation.{key}.min_examples {override.min_examples} must be >= 0"
        )
    if override.min_precision is not None and not 0.0 <= override.min_precision <= 1.0:
        problems.append(
            f"graduation.{key}.min_precision {override.min_precision} "
            "must be in [0.0, 1.0]"
        )
    if override.min_examples is None and override.min_precision is None:
        problems.append(
            f"graduation.{key}: an override must set at least one gate "
            "(min_examples / min_precision)"
        )
    return problems


class GateOverride(BaseModel):
    """One per-label gate override (ticket #22): the gate values the
    label must meet to graduate. ``None`` = keep the run's default for
    that gate. The raw type (bool/str refused), the ranges, and the
    empty-override refusal are checked by the parent's single
    collecting validator — the field constraints are deliberately
    ABSENT: a field-constraint failure would skip the model-level
    collection (pydantic's rule), and the "all problems together"
    doctrine dies with it."""

    model_config = {"extra": "forbid", "frozen": True}

    min_examples: int | None = None
    min_precision: float | None = None


class GraduationConfig(BaseModel):
    """Per-label Graduation gates (ticket #22): the run's default gate
    values + per-label overrides. The per-label entries are the
    [graduation."<label>"] subtables (label names may carry slashes —
    quoted TOML keys); they are collected from the extra fields into
    ``per_label`` by the validator, which refuses an override that is
    not a table of gates (fail loud — a typo'd label must not silently
    gate nothing). A raw ``per_label`` key in the [graduation] table is
    a RESERVED field name — refused with the label-path spelling shown:
    the declared field is the validator's collection target and the
    CLI's read surface, not an input surface (typed programmatic
    construction with GateOverride values is the API, and gets the same
    VALUE checks — ranges, empty-override; not the raw-type checks,
    which the wrap validator applies to raw input: a bool passed into
    GateOverride is coerced by pydantic before the validator sees it).

    The gate values are NOT tenth-grid (the documented asymmetry with
    [bands]): a band target must match the eval's candidate grid
    (thresholds are looked up in it), while a gate precision target is
    compared against MEASURED values — measured precision is any
    rational (tp/(tp+fp)), so a non-tenth target like 0.91 is a
    legitimate gate. Range + bool + str refusals DO apply (the config
    doctrine). The config carries NO path: the table path is the CLI's
    ``eval --graduation`` argument, so the config-file relative-path
    rule has nothing to anchor here."""

    model_config = {"extra": "allow", "frozen": True}

    # the defaults ride the named constants in graduation.py (the
    # sanctioned config→graduation edge — cycle-free; the equality is
    # pinned by test_config)
    min_examples: int = DEFAULT_MIN_EXAMPLES
    min_precision: float = DEFAULT_MIN_PRECISION
    #: The collected per-label overrides (the CLI reads this). NOT an
    # input surface: a raw ``per_label`` key in the [graduation] table
    # is refused (reserved name — see the docstring). A leading
    # underscore is not an option: pydantic would treat it as a private
    # attribute and break the TOML→CLI contract this field exists to
    # serve, so the reserved-name refusal is the guard.
    per_label: dict[str, GateOverride] = Field(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def _gate_problems_together(cls, data: object, handler, info) -> Self:
        """One validator collecting EVERY gate problem (the
        LabelsConfig doctrine) — and the ONLY one, on purpose: it sees
        the RAW input (mode="wrap"), so bools and strings are refused
        on the raw value before pydantic's lax int/float coercion can
        launder them (true → 1 / 1.0, "10" → 10, "0.9" → 0.9). EVERY
        problem class — raw types, ranges, empty-override, key
        hygiene — is collected on the RAW values BEFORE the handler,
        so no failure class skips the collection: a field-constraint
        failure (pydantic skips model after-validators when a field
        constraint fires — the ge/le constraints and the separate bool
        pre-scan are gone for exactly this reason) and a handler/
        coercion failure both raise with the full problem list. The
        ranges apply to the defaults AND every per-label override (no
        tenth-grid — see the class docstring); a raw ``per_label`` key
        is a reserved field name — an EMPTY raw table is accepted (it
        equals the typed default {}, the model_dump round-trip form),
        while ANY non-empty raw table WITH UNTYPED VALUES is refused
        (the typed GateOverride-valued programmatic path accepts); a
        label key is stripped, refused blank-after-strip, and
        acceptance-bounded at
        _MAX_LABEL_LENGTH (the echo bound _MAX_LOC_SEGMENT only
        truncates error messages — a long slash-nested Label Path is
        accepted, not refused)."""
        problems: list[str] = []
        raw = cast("dict[str, object]", data) if isinstance(data, dict) else {}
        raw_pl = raw.get("per_label")
        # an EMPTY raw dict is the typed DEFAULT ({}), accepted — the
        # Config.model_validate(cfg.model_dump()) round-trip produces
        # exactly this form; a NON-EMPTY raw dict with untyped values
        # is the back door and is refused (all() over the typed values
        # is the check)
        pl_is_typed = isinstance(raw_pl, dict) and all(
            isinstance(v, GateOverride) for v in raw_pl.values()
        )
        if "per_label" in raw and not pl_is_typed:
            # the back door: a raw per_label table bypassed every
            # custom refusal (the old pre-scan skipped it, the
            # collection walked only the extras). Reserved name — the
            # label-path spelling is the only per-label form
            problems.append(
                "graduation.per_label is a reserved field name — "
                "per-label overrides use the label-path spelling "
                '([graduation."Some/Label"]), not a raw per_label table'
            )
        problems.extend(
            _min_examples_raw_problems(
                "graduation.min_examples", raw.get("min_examples")
            )
        )
        problems.extend(
            _min_precision_raw_problems(
                "graduation.min_precision", raw.get("min_precision")
            )
        )
        problems.extend(_gate_raw_value_problems("graduation", raw, check_empty=False))
        for label, sub in raw.items():
            if label in (*_GATE_NAMES, "per_label"):
                continue
            key = label.strip()
            if not key:
                problems.append(
                    f"graduation.{_shorten_echo(label)}: a label key must be "
                    "non-empty after stripping"
                )
                continue
            if len(key) > _MAX_LABEL_LENGTH:
                problems.append(
                    f"graduation.{_shorten_echo(label)}: a label key is longer "
                    f"than {_MAX_LABEL_LENGTH} chars"
                )
                continue
            if not isinstance(sub, dict):
                # a scalar under a label name is a typo'd/unknown key,
                # not a half-written table — name it as such
                problems.append(
                    f"graduation.{key}: unknown/typo'd key \u2014 a per-label "
                    f'override is a subtable of gates ([graduation."{key}"])'
                )
                continue
            nested = cast("dict[str, object]", sub)
            for name in _GATE_NAMES:
                problems.extend(
                    (
                        _min_examples_raw_problems
                        if name == "min_examples"
                        else _min_precision_raw_problems
                    )(f"graduation.{key}.{name}", nested.get(name))
                )
            problems.extend(
                _gate_raw_value_problems(f"graduation.{key}", nested, check_empty=True)
            )
        try:
            self = handler(data)
        except ValidationError as e:
            # the raw scan already collected EVERY raw-value problem —
            # the handler failure adds the coercion errors, and the
            # combined list is raised (no failure class skips the
            # collection)
            problems.extend(
                f"graduation.{_shorten_loc(err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            raise ValueError("; ".join(problems)) from None
        per_label: dict[str, GateOverride] = {}
        # the declared field (defense in depth for programmatic
        # construction — the raw scan cannot see typed values): the
        # SAME key hygiene + value checks as the extra subtables
        for label, override in self.per_label.items():
            key = label.strip()
            if not key:
                problems.append(
                    "graduation.<blank>: a label key must be non-empty after stripping"
                )
                continue
            if len(key) > _MAX_LABEL_LENGTH:
                problems.append(
                    f"graduation.{_shorten_echo(label)}: a label key is longer "
                    f"than {_MAX_LABEL_LENGTH} chars"
                )
                continue
            if key in (*_GATE_NAMES, "per_label"):
                problems.append(
                    f"graduation.{key}: a label key may not be a reserved field name"
                )
                continue
            if key in per_label:
                # a padded key stripping onto an existing one (the
                # LabelsConfig uniqueness precedent, post-clean)
                problems.append(
                    f"graduation.{key}: duplicate per-label override — the "
                    "same label is named more than once; refusing to "
                    "silently shadow"
                )
                continue
            problems.extend(_gate_value_problems(key, override))
            per_label[key] = override
        for label, sub in (self.__pydantic_extra__ or {}).items():
            if not isinstance(sub, dict):
                continue  # already reported at the raw level
            key = label.strip()
            if not key or len(key) > _MAX_LABEL_LENGTH:
                continue  # already reported at the raw level
            if key in (*_GATE_NAMES, "per_label"):
                problems.append(
                    f"graduation.{key}: a label key may not be a reserved field name"
                )
                continue
            try:
                override = GateOverride.model_validate(sub)
            except ValidationError as e:
                problems.extend(
                    f"graduation.{key}.{_shorten_loc(err['loc'])}: {err['msg']}"
                    for err in e.errors()
                )
                continue
            if key in per_label:
                # the same label named more than once: a typed
                # per_label entry AND a [graduation."<key>"] subtable,
                # or padded keys that strip to one — silent shadowing
                # violates refuse-never-drop
                problems.append(
                    f"graduation.{key}: duplicate per-label override — the "
                    "same label is named more than once; refusing to "
                    "silently shadow"
                )
                continue
            # range/empty are collected at the RAW level (both paths);
            # the declared-field-only checks are the typed values,
            # which the raw scan cannot see
            per_label[key] = override
        if problems:
            raise ValueError("; ".join(problems))
        if per_label:
            object.__setattr__(self, "per_label", per_label)
        return self


class StoreConfig(BaseModel):
    """State DB location (ticket #23): the SQLite file holding the
    shadow_results table (and, with the daemon phase, the applications
    table — #24). A PLAIN path field on purpose: the [store] section
    carries no label-key surface, so the heavy [bands]/[graduation]
    per-label doctrine does not apply here. Relative paths anchor to
    the config file's directory (the corpus/model dir rule); the
    default is ``<config dir>/gmail-labeler.db``. The DB is
    attachable-public material by construction (invariant #5): ids,
    probabilities, labels, and run timestamps only — never bodies."""

    model_config = {"extra": "forbid", "frozen": True}

    db_path: Path = Path("./gmail-labeler.db")


class ModelConfig(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    #: Encoder identifier (the Checkpoint Sidecar's ``encoder_id`` field).
    encoder: str = "deberta-v3-base"
    #: LOCAL path to the encoder weights — training downloads nothing
    #: (AC#6); required for `train`, optional until it is used.
    encoder_path: Path | None = None
    #: Where `train` writes the checkpoint directory (gitignored).
    checkpoint_dir: Path = Path("./checkpoints/latest")

    @model_validator(mode="after")
    def _encoder_identifier_nonempty(self) -> Self:
        if not self.encoder.strip():
            raise ValueError("model.encoder must be non-empty")
        return self


#: (section, field) pairs holding filesystem paths needing expansion.
_PATH_FIELDS: tuple[tuple[str, str], ...] = (
    ("gmail", "credentials_path"),
    ("gmail", "token_file"),
    ("corpus", "dir"),
    ("gyb", "path"),
    ("model", "encoder_path"),
    ("model", "checkpoint_dir"),
    ("store", "db_path"),
)


class Config(BaseModel):
    # default_factory everywhere: pydantic reuses bare model defaults BY
    # REFERENCE, so without factories all Config() instances would share ONE
    # GmailConfig (verified cross-HOME drift). Factories make every instance
    # independent; no after-validator mutation exists anywhere.
    model_config = {"extra": "forbid", "frozen": True}

    gmail: GmailConfig = Field(default_factory=GmailConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    gyb: GybConfig = Field(default_factory=GybConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    bands: BandsConfig = Field(default_factory=BandsConfig)
    graduation: GraduationConfig = Field(default_factory=GraduationConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)


def _anchor_relative(cfg: Config, base_dir: Path) -> Config:
    """Resolve DEFAULT paths that are relative (file-supplied ones were
    already anchored by _prepare_raw). model_copy builds a new instance."""
    updates: dict[str, object] = {}
    if not cfg.corpus.dir.is_absolute():
        updates["corpus"] = cfg.corpus.model_copy(
            update={"dir": base_dir / cfg.corpus.dir}
        )
    if not cfg.model.checkpoint_dir.is_absolute():
        updates["model"] = cfg.model.model_copy(
            update={"checkpoint_dir": base_dir / cfg.model.checkpoint_dir}
        )
    if not cfg.store.db_path.is_absolute():
        updates["store"] = cfg.store.model_copy(
            update={"db_path": base_dir / cfg.store.db_path}
        )
    if not updates:
        return cfg
    return cfg.model_copy(update=updates)


def _prepare_raw(raw: dict[str, object], base_dir: Path, problems: list[str]) -> None:
    """Expand '~' and resolve relative paths against the config file's dir.

    Mutates the raw dict BEFORE validation (in: from TOML, ours to shape).
    A failing expansion is appended to problems, never raised raw.
    """
    for section, field in _PATH_FIELDS:
        block = raw.get(section)
        if not isinstance(block, dict) or field not in block:
            continue
        value = block[field]
        if not isinstance(value, str):
            continue  # type errors belong to pydantic, not here
        if not value.strip():
            problems.append(f"{section}.{field} must be non-empty")
            continue
        try:
            p = Path(value).expanduser()
        except RuntimeError as e:
            problems.append(f"{section}.{field}: cannot expand home: {e}")
            continue
        if not p.is_absolute():
            p = base_dir / p
        block[field] = str(p)


def find_config(explicit: str | None = None) -> Path:
    """Resolve the config file path or raise ConfigError."""
    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        return p
    local = Path.cwd() / _CONFIG_FILENAME
    if local.is_file():
        return local
    home = Path.home() / ".config/gmail-labeler" / _CONFIG_FILENAME
    if home.is_file():
        return home
    raise ConfigError(
        f"no {_CONFIG_FILENAME} found (looked in . and ~/.config/gmail-labeler/); "
        "copy example.toml and edit, or pass --config"
    )


def _shorten_loc(loc: tuple[object, ...]) -> str:
    return ".".join(
        (seg[:_MAX_LOC_SEGMENT] + "…") if len(seg := str(s)) > _MAX_LOC_SEGMENT else seg
        for s in loc
    )


def load_config(explicit: str | None = None) -> Config:
    path = find_config(explicit)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as e:
        raise ConfigError(f"{path}: not valid UTF-8 (save as UTF-8): {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e
    except OSError as e:
        raise ConfigError(f"{path}: unreadable: {e}") from e
    problems: list[str] = []
    _prepare_raw(raw, path.resolve().parent, problems)
    try:
        cfg = Config.model_validate(raw)
    except ValidationError as e:
        problems.extend(
            f"{_shorten_loc(err['loc']) or '<root>'}: {err['msg']}"
            for err in e.errors()
        )
    if problems:
        raise ConfigError(f"{path}: invalid config: " + "; ".join(problems))
    return _anchor_relative(cfg, path.resolve().parent)
