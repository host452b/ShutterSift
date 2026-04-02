from shuttersift.config import Config, ScoringWeights, Thresholds

def test_default_weights_sum_to_one():
    w = ScoringWeights()
    total = w.sharpness + w.exposure + w.aesthetic + w.face_quality + w.composition
    assert abs(total - 1.0) < 1e-9

def test_default_thresholds():
    t = Thresholds()
    assert t.keep == 70
    assert t.reject == 40
    assert t.hard_reject_sharpness == 30.0
    assert t.eye_open_min == 0.25
    assert t.burst_gap_seconds == 2.0

def test_config_loads_from_yaml(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("scoring:\n  thresholds:\n    keep: 80\n")
    cfg = Config.from_yaml(yaml_file)
    assert cfg.thresholds.keep == 80
    assert cfg.thresholds.reject == 40  # default preserved

def test_config_defaults_without_file():
    cfg = Config()
    assert cfg.weights.sharpness == 0.30
    assert cfg.workers == 4

def test_invalid_weights_raise():
    from pydantic import ValidationError
    import pytest
    with pytest.raises(ValidationError):
        ScoringWeights(sharpness=0.50, exposure=0.15, aesthetic=0.25,
                       face_quality=0.20, composition=0.10)  # sums to 1.20

def test_config_load_from_cwd_shuttersift_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "shuttersift.yaml"
    cfg_file.write_text("scoring:\n  thresholds:\n    keep: 85\n")
    cfg = Config.load()
    assert cfg.thresholds.keep == 85

def test_config_load_from_cwd_config_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("scoring:\n  thresholds:\n    keep: 77\n")
    cfg = Config.load()
    assert cfg.thresholds.keep == 77

def test_config_load_explicit_path_takes_priority(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("scoring:\n  thresholds:\n    keep: 77\n")
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("scoring:\n  thresholds:\n    keep: 99\n")
    cfg = Config.load(explicit)
    assert cfg.thresholds.keep == 99

def test_config_calibrated_defaults_false():
    cfg = Config()
    assert cfg.calibrated is False

def test_config_calibrated_round_trips_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("calibrated: true\nscoring:\n  thresholds:\n    hard_reject_sharpness: 42.3\n")
    cfg = Config.from_yaml(cfg_file)
    assert cfg.calibrated is True
    assert cfg.thresholds.hard_reject_sharpness == 42.3
