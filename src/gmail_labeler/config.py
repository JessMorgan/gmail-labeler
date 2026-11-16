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

_CONFIG_FILENAME = "config.toml"
# Bounds (does NOT redact — a short secret pasted as a key or value still
# echoes, truncated) the echo of config keys AND raw values in error
# messages: label keys and echoed raw values are cut to this bound.
_MAX_LOC_SEGMENT = 40

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


def _override_value_problems(
    key: str, override: BandOverride, skip: frozenset[tuple[str, str]] = frozenset()
) -> list[str]:
    """The value problems for one collected per-label override: the
    [0.0, 1.0] range, the tenth-grid stance (#20), and the empty
    override (a no-op that would silently calibrate with the
    defaults). ``skip`` names the (label, target) pairs whose RAW type
    problem was already reported (no double-report)."""
    problems: list[str] = []
    for name in _TARGET_NAMES:
        if (key, name) in skip:
            continue
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
        launder them (true → 1.0, "0.9" → 0.9), and no field-constraint
        failure can skip the collection (pydantic skips model
        after-validators when a field constraint fires — the ge/le
        constraints and the separate bool pre-scan are gone for
        exactly this reason). The tenth-grid stance (#20) applies to
        the defaults AND every per-label override; a raw ``per_label``
        key is a reserved field name; a label key is stripped, refused
        blank-after-strip, and length-bounded (the LabelsConfig
        stripping precedent — a padded key must not survive to
        calibration)."""
        problems: list[str] = []
        raw = cast("dict[str, object]", data) if isinstance(data, dict) else {}
        raw_pl = raw.get("per_label")
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
        flagged: set[tuple[str, str]] = set()
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
            if len(key) > _MAX_LOC_SEGMENT:
                problems.append(
                    f"bands.{_shorten_echo(label)}: a label key is longer "
                    f"than {_MAX_LOC_SEGMENT} chars"
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
                nested_problems = _target_raw_problems(
                    f"bands.{key}.{name}", nested.get(name)
                )
                if nested_problems:
                    flagged.add((key, name))
                    problems.extend(nested_problems)
        try:
            self = handler(data)
        except ValidationError as e:
            problems.extend(
                f"bands.{_shorten_loc(err['loc'])}: {err['msg']}" for err in e.errors()
            )
            raise ValueError("; ".join(problems)) from None
        for name, value in (
            ("min_high_precision", self.min_high_precision),
            ("min_med_precision", self.min_med_precision),
        ):
            if not 0.0 <= value <= 1.0:
                problems.append(f"bands.{name} {value} must be in [0.0, 1.0]")
            elif not is_tenth(value):
                problems.append(
                    f"bands.{name} {value} is not a tenth — the bands ride "
                    "the candidate grid (the tenth-grid stance, #20)"
                )
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
            if len(key) > _MAX_LOC_SEGMENT:
                problems.append(
                    f"bands.{_shorten_echo(label)}: a label key is longer "
                    f"than {_MAX_LOC_SEGMENT} chars"
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
            if not key or len(key) > _MAX_LOC_SEGMENT:
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
            skip = frozenset(
                (key, name) for name in _TARGET_NAMES if (key, name) in flagged
            )
            problems.extend(_override_value_problems(key, override, skip=skip))
            per_label[key] = override
        if problems:
            raise ValueError("; ".join(problems))
        if per_label:
            object.__setattr__(self, "per_label", per_label)
        return self


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
