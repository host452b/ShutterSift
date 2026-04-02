from __future__ import annotations
import math
import numpy as np
import cv2

_MAX_SHARP_VAR = 1500.0
_BLUR_FLOOR = 2.0


def sharpness_score(img: np.ndarray) -> float:
    """Returns sharpness score [0–100] using Laplacian variance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if laplacian_var < _BLUR_FLOOR:
        return 0.0

    log_val = math.log1p(laplacian_var)
    log_max = math.log1p(_MAX_SHARP_VAR)
    score = min(100.0, (log_val / log_max) * 100.0)
    return round(score, 2)


def laplacian_variance(img: np.ndarray) -> float:
    """Raw Laplacian variance — used for calibration and hard-reject thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
