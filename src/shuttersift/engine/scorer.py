from __future__ import annotations
from shuttersift.config import Config
from shuttersift.engine import SubScores


class Scorer:
    def __init__(self, config: Config):
        self._cfg = config

    def compute(self, sub: SubScores) -> float:
        """Weighted average of sub-scores → overall score [0–100]."""
        w = self._cfg.weights
        total = (
            sub.sharpness   * w.sharpness   +
            sub.exposure    * w.exposure    +
            sub.aesthetic   * w.aesthetic   +
            sub.face_quality * w.face_quality +
            sub.composition * w.composition
        )
        return round(max(0.0, min(100.0, total)), 2)

    def decide(self, score: float) -> str:
        """Map overall score to keep / review / reject."""
        t = self._cfg.thresholds
        if score >= t.keep:
            return "keep"
        if score < t.reject:
            return "reject"
        return "review"
