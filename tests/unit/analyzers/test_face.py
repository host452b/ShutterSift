# tests/unit/analyzers/test_face.py
from unittest.mock import MagicMock
import numpy as np
from shuttersift.engine.analyzers.face import FaceAnalyzer, FaceResult


def test_face_result_defaults():
    r = FaceResult()
    assert r.count == 0
    assert r.eye_open_score == 1.0   # neutral when no faces
    assert r.smile_score == 0.0
    assert r.all_eyes_closed is False


def test_no_face_returns_neutral(normal_image):
    """A plain gradient image should yield zero faces."""
    analyzer = FaceAnalyzer()
    result = analyzer.analyze(normal_image)
    assert result.count == 0
    assert result.all_eyes_closed is False


def test_face_quality_score_with_no_faces(normal_image):
    analyzer = FaceAnalyzer()
    result = analyzer.analyze(normal_image)
    assert result.face_quality_score == 75.0  # neutral


def test_all_eyes_closed_detection():
    """Mock blendshapes to simulate closed eyes."""
    analyzer = FaceAnalyzer()

    mock_blendshapes = {
        "eyeBlinkLeft": 0.95,
        "eyeBlinkRight": 0.92,
        "mouthSmileLeft": 0.1,
        "mouthSmileRight": 0.1,
        "cheekSquintLeft": 0.0,
        "cheekSquintRight": 0.0,
    }
    result = analyzer._compute_face_scores([mock_blendshapes])
    assert result.all_eyes_closed is True
    assert result.eye_open_score < 0.25


def test_detection_without_mesh_returns_partial(monkeypatch):
    """When detection finds faces but mesh fails, return partial result with face_quality_score=60."""
    from shuttersift.engine import analyzers  # noqa: F401
    import shuttersift.engine.analyzers.face as face_mod

    mock_det = MagicMock()
    mock_bbox = MagicMock()
    mock_bbox.xmin = 0.1
    mock_bbox.ymin = 0.1
    mock_bbox.width = 0.3
    mock_bbox.height = 0.4
    mock_detection = MagicMock()
    mock_detection.location_data.relative_bounding_box = mock_bbox
    mock_det.detections = [mock_detection]

    mock_mesh = MagicMock()
    mock_mesh.multi_face_landmarks = None

    mock_detection_obj = MagicMock()
    mock_detection_obj.process.return_value = mock_det

    mock_mesh_obj = MagicMock()
    mock_mesh_obj.process.return_value = mock_mesh

    monkeypatch.setattr(face_mod, "_MP_LOADED", True)
    monkeypatch.setattr(face_mod, "_face_detection", mock_detection_obj)
    monkeypatch.setattr(face_mod, "_face_mesh", mock_mesh_obj)

    analyzer = face_mod.FaceAnalyzer()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = analyzer.analyze(img)

    assert result.count == 1
    assert result.face_quality_score == 60.0
