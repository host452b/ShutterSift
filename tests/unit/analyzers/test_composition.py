# tests/unit/analyzers/test_composition.py
from shuttersift.engine.analyzers.composition import composition_score


def test_no_faces_returns_neutral(normal_image):
    score = composition_score(normal_image, face_bboxes=[])
    assert score == 50.0  # neutral when no detectable subjects


def test_face_at_thirds_node_scores_high(normal_image):
    # Face centered at top-left rule-of-thirds node (1/3, 1/3)
    # bbox in relative coords: (x1, y1, x2, y2)
    face_center_x = 1 / 3
    face_center_y = 1 / 3
    w, h = 0.15, 0.20
    bbox = (face_center_x - w/2, face_center_y - h/2,
            face_center_x + w/2, face_center_y + h/2)
    score = composition_score(normal_image, face_bboxes=[bbox])
    assert score >= 65, f"Expected ≥65 for thirds-node face, got {score}"


def test_face_at_edge_penalized(normal_image):
    # Face clipped at left edge
    bbox = (-0.05, 0.3, 0.1, 0.7)
    score = composition_score(normal_image, face_bboxes=[bbox])
    assert score < 50


def test_score_in_range(normal_image):
    score = composition_score(normal_image, face_bboxes=[])
    assert 0.0 <= score <= 100.0
