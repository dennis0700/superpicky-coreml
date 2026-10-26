import hashlib
from pathlib import Path

import pytest

from modelconv.download import ChecksumError, fetch, file_digest, verify


def _write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_file_digest_matches_hashlib(tmp_path: Path) -> None:
    digest = _write(tmp_path / "a.bin", b"regulus")
    assert file_digest(tmp_path / "a.bin") == digest


def test_verify_rejects_mismatch(tmp_path: Path) -> None:
    _write(tmp_path / "a.bin", b"regulus")
    with pytest.raises(ChecksumError):
        verify(tmp_path / "a.bin", {"sha256": "0" * 64})


def test_verify_skips_empty_values(tmp_path: Path) -> None:
    _write(tmp_path / "a.bin", b"regulus")
    verify(tmp_path / "a.bin", {"sha256": "", "sha1": ""})


def test_fetch_downloads_and_verifies(tmp_path: Path) -> None:
    digest = _write(tmp_path / "src.bin", b"weights")
    dest = tmp_path / "out" / "w.bin"
    fetch((tmp_path / "src.bin").as_uri(), dest, {"sha256": digest})
    assert dest.read_bytes() == b"weights"
    assert not dest.with_name("w.bin.part").exists()


def test_fetch_skips_a_verified_existing_file(tmp_path: Path) -> None:
    digest = _write(tmp_path / "w.bin", b"weights")
    fetch("https://invalid.example/w.bin", tmp_path / "w.bin", {"sha256": digest})
    assert (tmp_path / "w.bin").read_bytes() == b"weights"


def test_fetch_removes_part_file_on_mismatch(tmp_path: Path) -> None:
    _write(tmp_path / "src.bin", b"weights")
    dest = tmp_path / "w.bin"
    with pytest.raises(ChecksumError):
        fetch((tmp_path / "src.bin").as_uri(), dest, {"sha256": "0" * 64})
    assert not dest.exists()
    assert not dest.with_name("w.bin.part").exists()
