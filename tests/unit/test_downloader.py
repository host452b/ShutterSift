# tests/unit/test_downloader.py
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
from shuttersift.engine.downloader import verify_sha256, _download_file


def test_sha256_correct(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert verify_sha256(f, expected) is True


def test_sha256_wrong(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello world")
    assert verify_sha256(f, "deadbeef" * 8) is False


def test_sha256_missing_file(tmp_path):
    assert verify_sha256(tmp_path / "nonexistent.bin", "abc") is False
