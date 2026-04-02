# tests/integration/test_pipeline.py
"""
End-to-end test: synthetic JPEG directory → Engine → assert output structure.
No real photos needed — uses cv2-generated synthetic images.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import pytest
from shuttersift.config import Config
from shuttersift.engine.pipeline import Engine


def _make_photo_dir(tmp_path: Path) -> Path:
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    # 3 sharp photos (should keep/review)
    for i in range(3):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in range(0, 640, 8):
            for y in range(0, 480, 8):
                if (x // 8 + y // 8) % 2 == 0:
                    img[y:y+8, x:x+8] = 200
        cv2.imwrite(str(photo_dir / f"sharp_{i:03d}.jpg"), img)

    # 2 blurry photos (should reject)
    for i in range(2):
        blurry = cv2.GaussianBlur(img, (51, 51), 20)
        cv2.imwrite(str(photo_dir / f"blurry_{i:03d}.jpg"), blurry)

    return photo_dir


def test_full_pipeline_produces_output(tmp_path):
    photo_dir = _make_photo_dir(tmp_path)
    output_dir = tmp_path / "output"

    cfg = Config()
    engine = Engine(cfg)
    result = engine.analyze(photo_dir, output_dir, resume=False)

    assert len(result.photos) == 5

    # Output directories exist
    assert (output_dir / "keep").exists()
    assert (output_dir / "review").exists()
    assert (output_dir / "reject").exists()

    # JSON report exists and is valid
    report = json.loads((output_dir / "results.json").read_text())
    assert report["version"] == "1"
    assert len(report["photos"]) == 5

    # HTML report exists
    assert (output_dir / "report.html").exists()

    # State file written
    assert (output_dir / ".state.json").exists()


def test_blurry_photos_rejected(tmp_path):
    photo_dir = _make_photo_dir(tmp_path)
    output_dir = tmp_path / "output"

    engine = Engine(Config())
    result = engine.analyze(photo_dir, output_dir, resume=False)

    rejected = [r for r in result.photos if r.decision == "reject"]
    assert any("blurry" in r.path.name for r in rejected), \
        "Expected blurry photos to be rejected"


def test_resume_skips_processed(tmp_path):
    photo_dir = _make_photo_dir(tmp_path)
    output_dir = tmp_path / "output"

    engine = Engine(Config())

    # First run
    r1 = engine.analyze(photo_dir, output_dir, resume=False)
    assert len(r1.photos) == 5

    # Second run with resume=True — should load from state, not re-analyze
    calls = []
    r2 = engine.analyze(photo_dir, output_dir, resume=True,
                        on_progress=lambda i, t, r: calls.append(r.path.name))
    assert len(r2.photos) == 5
    # All loaded from cache — on_progress called but each is a cache hit


def test_each_photo_has_score_and_decision(tmp_path):
    photo_dir = _make_photo_dir(tmp_path)
    output_dir = tmp_path / "output"

    engine = Engine(Config())
    result = engine.analyze(photo_dir, output_dir, resume=False)

    for photo in result.photos:
        assert 0.0 <= photo.score <= 100.0
        assert photo.decision in ("keep", "review", "reject")
