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


def test_download_retries_on_failure(tmp_path):
    """_download_file retries up to max_retries times on network failure."""
    dest = tmp_path / "model.bin"
    call_count = 0

    def fake_retrieve(url, path):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise OSError("network error")
        # Write fake content to the tmp file
        Path(path).write_bytes(b"model data")

    with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
        with patch("time.sleep"):  # avoid actual sleep
            result = _download_file("http://example.com/model.bin", dest, max_retries=3)

    assert result is True
    assert dest.exists()
    assert call_count == 2
