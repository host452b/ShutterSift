# tests/unit/test_capabilities.py
from shuttersift.engine.capabilities import Capabilities


def test_capabilities_returns_dict():
    caps = Capabilities.detect()
    assert isinstance(caps.gpu, bool)
    assert isinstance(caps.rawpy, bool)
    assert isinstance(caps.api_vlm, bool)
    assert isinstance(caps.gguf_vlm, bool)


def test_capabilities_summary_string():
    caps = Capabilities.detect()
    summary = caps.summary()
    assert "GPU" in summary
    assert "RAW" in summary


def test_gguf_false_when_no_models(tmp_path, monkeypatch):
    import shuttersift.engine.capabilities as caps_module
    monkeypatch.setattr(caps_module, "MODELS_DIR", tmp_path)
    caps = Capabilities.detect()
    assert caps.gguf_vlm is False
