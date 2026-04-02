# tests/unit/test_explainer.py
from unittest.mock import patch, MagicMock
from pathlib import Path
from shuttersift.engine.explainer import Explainer
from shuttersift.engine import PhotoResult
from shuttersift.config import Config


def test_skip_when_no_vlm(tmp_path):
    cfg = Config()
    explainer = Explainer(cfg, gguf_path=None, api_key_anthropic=None, api_key_openai=None)
    result = PhotoResult(path=tmp_path / "test.jpg", decision="review")
    explanation = explainer.explain(result.path, result)
    assert explanation == ""


def test_only_explains_review(tmp_path):
    cfg = Config()
    explainer = Explainer(cfg, gguf_path=None, api_key_anthropic=None, api_key_openai=None)
    keep_result = PhotoResult(path=tmp_path / "test.jpg", decision="keep")
    assert explainer.explain(keep_result.path, keep_result) == ""


def test_anthropic_api_called_for_review(tmp_path, monkeypatch):
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="Great portrait, sharp eyes.")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    monkeypatch.setattr("shuttersift.engine.explainer.anthropic", MagicMock(Anthropic=MagicMock(return_value=mock_client)))

    cfg = Config()
    explainer = Explainer(cfg, gguf_path=None, api_key_anthropic="fake-key", api_key_openai=None)

    img_path = tmp_path / "photo.jpg"
    import cv2, numpy as np
    cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

    result = PhotoResult(path=img_path, decision="review", score=55.0)
    text = explainer.explain(img_path, result)
    assert text == "Great portrait, sharp eyes."
