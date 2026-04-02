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
