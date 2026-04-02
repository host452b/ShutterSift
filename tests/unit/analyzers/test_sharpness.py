# tests/unit/analyzers/test_sharpness.py
from shuttersift.engine.analyzers.sharpness import sharpness_score


def test_sharp_image_scores_high(sharp_image):
    score = sharpness_score(sharp_image)
    assert score > 60, f"Sharp image scored {score}, expected > 60"


def test_blurry_image_scores_low(blurry_image):
    score = sharpness_score(blurry_image)
    assert score < 20, f"Blurry image scored {score}, expected < 20"


def test_score_in_range(sharp_image):
    score = sharpness_score(sharp_image)
    assert 0.0 <= score <= 100.0


def test_blur_scores_less_than_sharp(sharp_image, blurry_image):
    assert sharpness_score(blurry_image) < sharpness_score(sharp_image)
