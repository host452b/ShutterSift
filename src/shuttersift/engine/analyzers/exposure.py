from __future__ import annotations
import numpy as np
import cv2


def exposure_score(img: np.ndarray) -> float:
    """
    Returns exposure quality score [0–100].
    Penalizes over-exposed (>240) and under-exposed (<15) pixel ratios.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    overexp_ratio = float(hist[241:].sum() / total_pixels)
    underexp_ratio = float(hist[:15].sum() / total_pixels)

    clip_penalty = min(1.0, (overexp_ratio + underexp_ratio) * 4.0)

    mean_brightness = float(gray.mean())
    if 60 <= mean_brightness <= 200:
        brightness_bonus = 1.0
    else:
        dist = min(abs(mean_brightness - 60), abs(mean_brightness - 200))
        brightness_bonus = max(0.0, 1.0 - dist / 60.0)

    base_score = brightness_bonus * (1.0 - clip_penalty)
    return round(max(0.0, min(100.0, base_score * 100.0)), 2)
