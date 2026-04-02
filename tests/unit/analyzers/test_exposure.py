# tests/unit/analyzers/test_exposure.py
from shuttersift.engine.analyzers.exposure import exposure_score


def test_normal_exposure_scores_high(normal_image):
    score = exposure_score(normal_image)
    assert score > 65, f"Normal image scored {score}"


def test_dark_image_scores_low(dark_image):
    score = exposure_score(dark_image)
    assert score < 35, f"Dark image scored {score}"


def test_bright_image_scores_low(bright_image):
    score = exposure_score(bright_image)
    assert score < 35, f"Overexposed image scored {score}"


def test_score_in_range(normal_image):
    score = exposure_score(normal_image)
    assert 0.0 <= score <= 100.0
