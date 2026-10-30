from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import pytest

from gmail_labeler.tokens import (
    FileTokenStore,
    KeyringStore,
    TokenStoreError,
    make_token_json,
    parse_token_json,
    store_for_backend,
)

SECRET = "1//0refreshtokensecretvalue"


class FakeKeyring:
    def __init__(self, fail: bool = False) -> None:
        self.kv: dict[tuple[str, str], str] = {}
        self.fail = fail
        self.fail_msg = "no Secret Service"

    def get_password(self, service: str, user: str) -> str | None:
        if self.fail:
            raise RuntimeError(self.fail_msg)
        return self.kv.get((service, user))

    def set_password(self, service: str, user: str, value: str) -> None:
        if self.fail:
            raise RuntimeError(self.fail_msg)
        self.kv[(service, user)] = value


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    mod: Any = FakeKeyring()
    monkeypatch.setitem(__import__("sys").modules, "keyring", mod)
    return mod


def test_file_store_round_trip_and_mode(tmp_path: Path) -> None:
    store = FileTokenStore(tmp_path / "tok.json")
    assert store.load() is None
    store.save(SECRET)
    assert store.load() == SECRET
    assert store.path.stat().st_mode & 0o077 == 0  # 0600


def test_file_store_warns_on_loose_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FileTokenStore(tmp_path / "sub" / "tok.json")
    store.save(SECRET)
    store.path.chmod(0o644)
    assert store.load() == SECRET
    err = capsys.readouterr().err
    assert "group/world-readable" in err
    assert SECRET not in err  # the warning never echoes the token


def test_keyring_store_uses_injected_backend(fake_keyring: FakeKeyring) -> None:
    store = KeyringStore()
    assert store.load() is None
    store.save(SECRET)
    assert store.load() == SECRET
    assert list(fake_keyring.kv.values()) == [SECRET]


def test_keyring_unavailable_names_the_fix(fake_keyring: FakeKeyring) -> None:
    fake_keyring.fail = True
    store = KeyringStore()
    with pytest.raises(TokenStoreError, match='token_backend = "file"'):
        store.load()


def test_store_selection_and_unknown_backend(tmp_path: Path) -> None:
    assert isinstance(store_for_backend("keyring", tmp_path / "t"), KeyringStore)
    assert isinstance(store_for_backend("file", tmp_path / "t"), FileTokenStore)
    with pytest.raises(TokenStoreError, match="unknown token_backend"):
        store_for_backend("vault", tmp_path / "t")  # type: ignore[arg-type]


def test_token_json_round_trip() -> None:
    raw = make_token_json(SECRET, "cid.apps", "csec")
    data = parse_token_json(raw)
    assert data["refresh_token"] == SECRET
    assert data["scopes"] == ["https://www.googleapis.com/auth/gmail.modify"]


def test_parse_token_json_rejects_garbage() -> None:
    from gmail_labeler.tokens import is_signed_in_data

    with pytest.raises(TokenStoreError, match="malformed"):
        parse_token_json("[1,2,3]")
    with pytest.raises(TokenStoreError, match="malformed"):
        parse_token_json("not json")
    # parse = shape; usability = is_signed_in_data (single contract shared by
    # status() and credentials())
    data = parse_token_json('{"client_id": "x"}')
    assert not is_signed_in_data(data)
    assert is_signed_in_data(json.loads(make_token_json(SECRET, "cid", "csec")))


def test_keyring_backend_echoing_value_cannot_leak(fake_keyring: FakeKeyring) -> None:
    # The backend exception TEXT embeds the secret (worst realistic case).
    fake_keyring.fail = True
    fake_keyring.fail_msg = f"backend exploded on value {SECRET}"
    store = KeyringStore()
    try:
        store.load()
    except TokenStoreError as e:
        assert SECRET not in str(e)
        assert SECRET not in traceback.format_exc()
    else:  # pragma: no cover
        raise AssertionError("expected TokenStoreError")


def test_keyring_write_leak_guard(fake_keyring: FakeKeyring) -> None:
    fake_keyring.fail = True
    fake_keyring.fail_msg = f"set failed for {SECRET}"
    try:
        KeyringStore().save(SECRET)
    except TokenStoreError as e:
        assert SECRET not in str(e)
        assert SECRET not in traceback.format_exc()
    else:  # pragma: no cover
        raise AssertionError("expected TokenStoreError")


def test_save_tightens_pre_existing_loose_file(tmp_path: Path) -> None:
    # The FILE becomes 0600 via atomic replace; the PRE-EXISTING parent
    # dir is NOT re-permissioned (not ours to clamp — review MAJOR).
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    store = FileTokenStore(parent / "tok.json")
    store.path.write_text("stale", encoding="utf-8")
    store.path.chmod(0o644)
    store.save(SECRET)
    assert store.path.stat().st_mode & 0o077 == 0
    assert parent.stat().st_mode & 0o777 == 0o755  # untouched
    assert store.load() == SECRET


def test_save_creates_and_tightens_own_parent(tmp_path: Path) -> None:
    store = FileTokenStore(tmp_path / "fresh" / "tok.json")
    store.save(SECRET)
    assert store.path.parent.stat().st_mode & 0o777 == 0o700


def test_save_is_atomic_no_residue(tmp_path: Path) -> None:
    store = FileTokenStore(tmp_path / "sub" / "tok.json")
    store.save("first")
    store.save(SECRET)
    assert list((tmp_path / "sub").iterdir()) == [store.path]  # no .tmp leftovers
    assert store.load() == SECRET


def test_load_non_utf8_is_typed_error(tmp_path: Path) -> None:
    store = FileTokenStore(tmp_path / "tok.json")
    store.path.write_bytes(b"\xff\xfe broken")
    with pytest.raises(TokenStoreError, match="malformed"):
        store.load()


def test_save_no_temp_residue_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # deepseek MINOR: ENOSPC-style OSError mid-write must not leave a
    # token-bearing .tok-*.tmp (the OSError branch previously returned
    # before the cleanup branch).
    store = FileTokenStore(tmp_path / "tok.json")
    import os as _os

    real_fsync = _os.fsync

    def boom(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_os, "fsync", boom)
    with pytest.raises(TokenStoreError, match="unwritable"):
        store.save(SECRET)
    monkeypatch.setattr(_os, "fsync", real_fsync)
    assert [p.name for p in tmp_path.iterdir()] == []  # no residue, no file


def test_probe_exercises_writability(tmp_path: Path) -> None:
    store = FileTokenStore(tmp_path / "sub" / "tok.json")
    store.probe()  # creates parent + round-trips a probe file
    assert (tmp_path / "sub").is_dir()
    assert list((tmp_path / "sub").iterdir()) == []
    ro = tmp_path / "ro"
    ro.mkdir(mode=0o500)
    with pytest.raises(TokenStoreError):
        FileTokenStore(ro / "tok.json").probe()
    ro.chmod(0o700)
