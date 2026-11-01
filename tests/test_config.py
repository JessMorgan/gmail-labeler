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
