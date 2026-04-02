import cv2
import numpy as np
from pathlib import Path
from shuttersift.engine.pipeline import Engine
from shuttersift.config import Config


def test_engine_analyze_single_jpeg(tmp_path, sharp_image):
    path = tmp_path / "DSC001.jpg"
    cv2.imwrite(str(path), sharp_image)
    out = tmp_path / "out"

    engine = Engine(Config())
    result = engine.analyze(tmp_path, out, resume=False)

    assert len(result.photos) == 1
    assert result.photos[0].path == path
    assert result.photos[0].score > 0
    assert result.photos[0].decision in ("keep", "review", "reject")


def test_engine_capabilities_returns_dict():
    engine = Engine(Config())
    caps = engine.capabilities()
    assert "gpu" in caps
    assert "rawpy" in caps


def test_blurry_image_rejected(tmp_path, blurry_image):
    path = tmp_path / "blurry.jpg"
    cv2.imwrite(str(path), blurry_image)
    out = tmp_path / "out"

    engine = Engine(Config())
    result = engine.analyze(tmp_path, out, resume=False)

    assert result.photos[0].decision == "reject"
    assert result.photos[0].hard_rejected is True


def test_progress_callback_called(tmp_path, sharp_image):
    path = tmp_path / "DSC001.jpg"
    cv2.imwrite(str(path), sharp_image)
    out = tmp_path / "out"

    calls = []
    engine = Engine(Config())
    engine.analyze(tmp_path, out, resume=False, on_progress=lambda i, t, r: calls.append(i))

    assert len(calls) == 1
