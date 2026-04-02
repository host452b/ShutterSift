# tests/unit/analyzers/test_face.py
from unittest.mock import MagicMock, patch
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
