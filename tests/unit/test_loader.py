# tests/unit/test_loader.py
import sys
import numpy as np
import cv2
import pytest
from pathlib import Path
from shuttersift.engine.loader import load_image, SUPPORTED_FORMATS


def test_load_jpeg(tmp_path, sharp_image):
    path = tmp_path / "test.jpg"
    cv2.imwrite(str(path), sharp_image)
    img = load_image(path)
    assert img is not None
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_load_png(tmp_path, sharp_image):
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), sharp_image)
    img = load_image(path)
    assert img is not None


def test_load_missing_file_returns_none():
    img = load_image(Path("/nonexistent/file.jpg"))
    assert img is None


def test_supported_formats_includes_raw():
    for ext in [".cr2", ".nef", ".arw", ".dng"]:
        assert ext in SUPPORTED_FORMATS


def test_load_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"not an image at all")
    img = load_image(path)
    assert img is None


def test_load_raw_without_rawpy_returns_none(tmp_path, monkeypatch):
    """load_image returns None for RAW files when rawpy is not available."""
    raw_file = tmp_path / "test.cr2"
    raw_file.write_bytes(b"not a real raw file")
    monkeypatch.setitem(sys.modules, "rawpy", None)
    result = load_image(raw_file)
    assert result is None
