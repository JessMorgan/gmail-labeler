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

from .dataset_build import is_tenth

_CONFIG_FILENAME = "config.toml"
# Bounds (does NOT redact — a short secret pasted as a KEY still echoes) the
# echo of config keys in error messages; values are never echoed at all.
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


class BandOverride(BaseModel):
    """One per-label min-precision target override (ticket #21): the
    floor a candidate column's measured precision must meet to earn the
    band. ``None`` = keep the run's default for that band. Values are
    tenths (the tenth-grid stance, #20) — checked by the parent
    validator, which collects every problem."""

    model_config = {"extra": "forbid", "frozen": True}

    min_high_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    min_med_precision: float | None = Field(default=None, ge=0.0, le=1.0)


class BandsConfig(BaseModel):
    """Per-label Threshold Band calibration (ticket #21): the run's
    default min-precision targets + per-label overrides. The per-label
    entries are the [bands."<label>"] subtables (label names may carry
    slashes — quoted TOML keys); they are collected from the extra
    fields into ``per_label`` by the validator, which refuses an
    override that is not a table of targets (fail loud — a typo'd
    label must not silently calibrate nothing). The config carries NO
    path: the artifact path is the CLI's ``eval --bands`` argument, so
    the config-file relative-path rule has nothing to anchor here."""

    model_config = {"extra": "allow", "frozen": True}

    min_high_precision: float = Field(default=0.9, ge=0.0, le=1.0)
    min_med_precision: float = Field(default=0.7, ge=0.0, le=1.0)
    per_label: dict[str, BandOverride] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _band_target_bools_refused(cls, data: object) -> object:
        """A TOML bool is not a target: pydantic's lax float coercion
        would turn ``true`` into 1.0 (and ``false`` into 0.0) — the
        core's "a bool is not a target" refusal must not be silently
        bypassed at the config layer. Checked on the RAW values (after
        coercion the bool is indistinguishable from a literal 1.0)."""
        if not isinstance(data, dict):
            return data
        payload = cast("dict[str, object]", data)
        problems: list[str] = []
        for name in ("min_high_precision", "min_med_precision"):
            if isinstance(payload.get(name), bool):
                problems.append(f"bands.{name}: a target is a number, not a bool")
        for label, raw in payload.items():
            if label in ("min_high_precision", "min_med_precision", "per_label"):
                continue
            if isinstance(raw, dict):
                nested = cast("dict[str, object]", raw)
                for name in ("min_high_precision", "min_med_precision"):
                    if isinstance(nested.get(name), bool):
                        problems.append(
                            f"bands.{label}.{name}: a target is a number, not a bool"
                        )
        if problems:
            raise ValueError("; ".join(problems))
        return data

    @model_validator(mode="after")
    def _band_target_problems_together(self) -> Self:
        """One validator collecting EVERY band-target problem (the
        LabelsConfig doctrine): the tenth-grid stance (#20 — a target
        between grid points would render mislabeled in any report of
        the bands) applies to the defaults AND every per-label
        override; an override that is not a table of targets is a
        typo, refused, not dropped."""
        problems: list[str] = []
        for name, value in (
            ("min_high_precision", self.min_high_precision),
            ("min_med_precision", self.min_med_precision),
        ):
            if not is_tenth(value):
                problems.append(
                    f"bands.{name} {value} is not a tenth — the bands ride "
                    "the candidate grid (the tenth-grid stance, #20)"
                )
        per_label: dict[str, BandOverride] = {}
        for label, raw in (self.__pydantic_extra__ or {}).items():
            if not isinstance(raw, dict):
                problems.append(
                    f"bands.{label}: a per-label override must be a table "
                    'of targets ([bands."<label>"])'
                )
                continue
            try:
                override = BandOverride.model_validate(raw)
            except ValidationError as e:
                problems.extend(
                    f"bands.{label}.{_shorten_loc(err['loc'])}: {err['msg']}"
                    for err in e.errors()
                )
                continue
            for name, target in (
                ("min_high_precision", override.min_high_precision),
                ("min_med_precision", override.min_med_precision),
            ):
                if target is not None and not is_tenth(target):
                    problems.append(
                        f"bands.{label}.{name} {target} is not a tenth — "
                        "the bands ride the candidate grid (the "
                        "tenth-grid stance, #20)"
                    )
            if (
                override.min_high_precision is None
                and override.min_med_precision is None
            ):
                # an empty table sets nothing — a no-op that would
                # silently calibrate with the defaults (the docstring's
                # "a typo'd label must not silently calibrate nothing")
                problems.append(
                    f"bands.{label}: an override must set at least one "
                    "target (min_high_precision / min_med_precision)"
                )
            per_label[label] = override
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
