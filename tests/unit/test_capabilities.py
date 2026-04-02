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


def test_capabilities_detects_mf_model(tmp_path, monkeypatch):
    """Capabilities should detect moondream .mf files, not .gguf."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    mf_file = models_dir / "moondream2-int8.mf"
    mf_file.touch()
    import shuttersift.engine.capabilities as cap_mod
    monkeypatch.setattr(cap_mod, "MODELS_DIR", models_dir)
    caps = cap_mod.Capabilities.detect()
    assert caps.gguf_model_path == mf_file


def test_capabilities_does_not_detect_gguf(tmp_path, monkeypatch):
    """A .gguf file alone should NOT trigger local VLM (wrong format)."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "some_model.gguf").touch()
    import shuttersift.engine.capabilities as cap_mod
    monkeypatch.setattr(cap_mod, "MODELS_DIR", models_dir)
    caps = cap_mod.Capabilities.detect()
    assert caps.gguf_model_path is None
