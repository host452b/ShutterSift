from shuttersift.engine import SubScores
from shuttersift.engine.scorer import Scorer
from shuttersift.config import Config


def _scorer() -> Scorer:
    return Scorer(Config())


def test_perfect_scores_give_keep():
    scorer = _scorer()
    sub = SubScores(sharpness=100, exposure=100, aesthetic=100,
                    face_quality=100, composition=100)
    total = scorer.compute(sub)
    assert total == 100.0
    assert scorer.decide(total) == "keep"


def test_all_zero_gives_reject():
    scorer = _scorer()
    sub = SubScores(sharpness=0, exposure=0, aesthetic=0,
                    face_quality=0, composition=0)
    total = scorer.compute(sub)
    assert total == 0.0
    assert scorer.decide(total) == "reject"


def test_weights_applied_correctly():
    scorer = _scorer()
    # Only sharpness is 100 (weight 0.30)
    sub = SubScores(sharpness=100, exposure=0, aesthetic=0,
                    face_quality=0, composition=0)
    total = scorer.compute(sub)
    assert abs(total - 30.0) < 0.01


def test_decision_thresholds():
    scorer = _scorer()
    assert scorer.decide(70.0) == "keep"
    assert scorer.decide(69.9) == "review"
    assert scorer.decide(40.0) == "review"
    assert scorer.decide(39.9) == "reject"


def test_custom_thresholds():
    from shuttersift.config import Config, Thresholds
    cfg = Config(thresholds=Thresholds(keep=80, reject=50))
    scorer = Scorer(cfg)
    assert scorer.decide(80.0) == "keep"
    assert scorer.decide(79.9) == "review"
    assert scorer.decide(50.0) == "review"
    assert scorer.decide(49.9) == "reject"
