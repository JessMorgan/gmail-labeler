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
from typing import Literal, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

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


#: (section, field) pairs holding filesystem paths needing expansion.
_PATH_FIELDS: tuple[tuple[str, str], ...] = (
    ("gmail", "credentials_path"),
    ("gmail", "token_file"),
    ("corpus", "dir"),
    ("gyb", "path"),
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


def _anchor_relative(cfg: Config, base_dir: Path) -> Config:
    """Resolve DEFAULT paths that are relative (file-supplied ones were
    already anchored by _prepare_raw). model_copy builds a new instance."""
    if cfg.corpus.dir.is_absolute():
        return cfg
    return cfg.model_copy(
        update={
            "corpus": cfg.corpus.model_copy(update={"dir": base_dir / cfg.corpus.dir})
        }
    )


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
