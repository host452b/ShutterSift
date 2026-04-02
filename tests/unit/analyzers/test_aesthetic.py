# tests/unit/analyzers/test_aesthetic.py
from unittest.mock import patch, MagicMock
from shuttersift.engine.analyzers.aesthetic import AestheticAnalyzer


def test_score_in_range(normal_image):
    analyzer = AestheticAnalyzer(use_gpu=False)
    score = analyzer.score(normal_image)
    assert 0.0 <= score <= 100.0


def test_brisque_fallback_when_pyiqa_unavailable(normal_image, monkeypatch):
    """Should fall back to BRISQUE (scikit-image) without pyiqa."""
    monkeypatch.setattr(
        "shuttersift.engine.analyzers.aesthetic._PYIQA_AVAILABLE", False
    )
    analyzer = AestheticAnalyzer(use_gpu=False)
    score = analyzer.score(normal_image)
    assert 0.0 <= score <= 100.0


def test_returns_float(normal_image):
    analyzer = AestheticAnalyzer(use_gpu=False)
    score = analyzer.score(normal_image)
    assert isinstance(score, float)
