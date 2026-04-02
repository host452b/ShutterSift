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


def test_prompt_template_is_english():
    """_PROMPT_TEMPLATE must be in English, not Chinese."""
    from shuttersift.engine.explainer import _PROMPT_TEMPLATE
    chinese_chars = any('\u4e00' <= c <= '\u9fff' for c in _PROMPT_TEMPLATE)
    assert not chinese_chars, f"Prompt template contains Chinese characters: {_PROMPT_TEMPLATE[:80]}"

def test_prompt_template_contains_score_placeholder():
    from shuttersift.engine.explainer import _PROMPT_TEMPLATE
    assert "{score" in _PROMPT_TEMPLATE

def test_moondream_explain_skips_when_no_package(tmp_path, monkeypatch):
    """_explain_moondream returns '' when moondream is not installed."""
    import sys
    monkeypatch.setitem(sys.modules, "moondream", None)
    from importlib import reload
    import shuttersift.engine.explainer as exp_mod
    from shuttersift.config import Config
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"fake")
    exp = exp_mod.Explainer(
        config=Config(),
        gguf_path=tmp_path / "model.mf",
        api_key_anthropic=None,
        api_key_openai=None,
    )
    result = exp._explain_moondream(photo, "test prompt")
    assert result == ""
