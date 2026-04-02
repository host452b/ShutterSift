# tests/fixtures/conftest.py
import numpy as np
import cv2
import pytest
from pathlib import Path


@pytest.fixture
def sharp_image() -> np.ndarray:
    """Checkerboard — high Laplacian variance (~3000+)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for x in range(0, 640, 8):
        for y in range(0, 480, 8):
            if (x // 8 + y // 8) % 2 == 0:
                img[y:y+8, x:x+8] = 200
    return img


@pytest.fixture
def blurry_image(sharp_image) -> np.ndarray:
    """Heavily blurred — low Laplacian variance (<5)."""
    return cv2.GaussianBlur(sharp_image, (51, 51), 20)


@pytest.fixture
def dark_image() -> np.ndarray:
    """Underexposed: all pixels ~ 8."""
    return np.full((480, 640, 3), 8, dtype=np.uint8)


@pytest.fixture
def bright_image() -> np.ndarray:
    """Overexposed: all pixels ~ 248."""
    return np.full((480, 640, 3), 248, dtype=np.uint8)


@pytest.fixture
def normal_image() -> np.ndarray:
    """Normal exposure: gradient 60–190."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for x in range(640):
        val = int(60 + (x / 639) * 130)
        img[:, x] = val
    return img


@pytest.fixture
def tmp_jpeg_dir(tmp_path, sharp_image) -> Path:
    """3 JPEG files in a temp directory."""
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"DSC_{i:04d}.jpg"), sharp_image)
    return tmp_path
