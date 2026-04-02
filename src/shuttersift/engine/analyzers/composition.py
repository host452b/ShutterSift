from __future__ import annotations
import numpy as np

# Rule-of-thirds nodes (relative coords)
_THIRDS_NODES = [
    (1/3, 1/3), (2/3, 1/3),
    (1/3, 2/3), (2/3, 2/3),
]
# Max distance from any thirds node to a frame corner is ~0.47; 0.4 captures
# the "close enough" zone without penalising the entire frame.
_THIRDS_FALLOFF = 0.4


def composition_score(
    img: np.ndarray,
    face_bboxes: list[tuple[float, float, float, float]],
) -> float:
    """
    Rule-based composition score [0–100].
    img is accepted for API consistency but currently unused.
    No faces → neutral 50.
    Faces present → score by:
      - Proximity to rule-of-thirds nodes (+40)
      - Not clipped at frame edge (+30)
      - Not dead-center (slight bonus) (+15)
      - Reasonable face size (+15)
    """
    if not face_bboxes:
        return 50.0

    scores = [_score_single_face(bbox) for bbox in face_bboxes]
    # Use the best-positioned face (primary subject)
    return round(max(scores), 2)


def _score_single_face(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    # Normalize: ensure x1 <= x2 and y1 <= y2
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    face_w = x2 - x1
    face_h = y2 - y1

    score = 0.0

    # 1. Thirds proximity (max 40 pts)
    min_thirds_dist = min(
        ((cx - nx)**2 + (cy - ny)**2) ** 0.5
        for nx, ny in _THIRDS_NODES
    )
    thirds_score = max(0.0, 40.0 * (1.0 - min_thirds_dist / _THIRDS_FALLOFF))
    score += thirds_score

    # 2. Not clipped at edges (max 30 pts)
    margin = 0.02  # 2% tolerance
    clipped = x1 < -margin or y1 < -margin or x2 > 1 + margin or y2 > 1 + margin
    if not clipped:
        score += 30.0
    else:
        clip_amount = max(
            max(0.0, -x1), max(0.0, x2 - 1.0),
            max(0.0, -y1), max(0.0, y2 - 1.0),
        )
        score += max(0.0, 30.0 - clip_amount * 400.0)

    # 3. Not dead-center (max 15 pts — slight penalty for perfectly centered)
    center_dist = ((cx - 0.5)**2 + (cy - 0.5)**2) ** 0.5
    if center_dist > 0.05:
        score += 15.0
    else:
        score += center_dist * 15.0 / 0.05

    # 4. Reasonable face size: 5–40% of frame width is ideal (max 15 pts)
    if 0.05 <= face_w <= 0.40:
        score += 15.0
    elif face_w < 0.05:
        score += face_w / 0.05 * 15.0
    else:
        score += max(0.0, 15.0 - (face_w - 0.40) / 0.20 * 15.0)

    return max(0.0, min(100.0, score))
