# tests/unit/analyzers/test_aesthetic.py
from unittest.mock import MagicMock
from shuttersift.engine.analyzers.aesthetic import AestheticAnalyzer


def test_score_in_range(normal_image):
    analyzer = AestheticAnalyzer(use_gpu=False)
    score = analyzer.score(normal_image)
    assert 0.0 <= score <= 100.0


def test_laplacian_fallback_when_brisque_unavailable(normal_image, monkeypatch):
    """Falls back to Laplacian estimate when both pyiqa and cv2.quality are unavailable."""
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


def test_musiq_backend_selected_when_pyiqa_available(monkeypatch):
    """When pyiqa is available, _load() sets backend to 'musiq'."""
    import types
    fake_pyiqa = types.ModuleType("pyiqa")
    fake_pyiqa.create_metric = MagicMock(return_value=MagicMock())

    monkeypatch.setattr("shuttersift.engine.analyzers.aesthetic._PYIQA_AVAILABLE", True)
    monkeypatch.setattr("shuttersift.engine.analyzers.aesthetic.pyiqa", fake_pyiqa, raising=False)

    analyzer = AestheticAnalyzer(use_gpu=False)
    analyzer._load()
    assert analyzer._backend == "musiq"
    assert analyzer._loaded is True


def test_brisque_path_when_cv2_quality_available(normal_image, monkeypatch):
    """BRISQUE path: mocked cv2.quality returns a tuple, score is inverted correctly."""
    import cv2
    mock_compute = MagicMock(return_value=(30.0,))  # raw BRISQUE=30 → score=70
    monkeypatch.setattr(cv2, "quality", MagicMock(), raising=False)
    monkeypatch.setattr("cv2.quality.QualityBRISQUE_compute", mock_compute)

    monkeypatch.setattr(
        "shuttersift.engine.analyzers.aesthetic._PYIQA_AVAILABLE", False
    )
    analyzer = AestheticAnalyzer(use_gpu=False)
    score = analyzer.score(normal_image)
    assert score == 70.0
