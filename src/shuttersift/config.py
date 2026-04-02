from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, model_validator


class ScoringWeights(BaseModel):
    sharpness: float = 0.30
    exposure: float = 0.15
    aesthetic: float = 0.25
    face_quality: float = 0.20
    composition: float = 0.10

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ScoringWeights":
        total = self.sharpness + self.exposure + self.aesthetic + self.face_quality + self.composition
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.4f}")
        return self


class Thresholds(BaseModel):
    keep: int = 70
    reject: int = 40
    hard_reject_sharpness: float = 30.0
    eye_open_min: float = 0.25
    burst_gap_seconds: float = 2.0


class Config(BaseModel):
    weights: ScoringWeights = ScoringWeights()
    thresholds: Thresholds = Thresholds()
    workers: int = 4
    log_retention_runs: int = 30
    api_model_anthropic: str = "claude-haiku-4-5-20251001"
    api_model_openai: str = "gpt-4o-mini"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        # Flatten nested 'scoring' key
        scoring = data.pop("scoring", {})
        if "weights" in scoring:
            data["weights"] = scoring["weights"]
        if "thresholds" in scoring:
            data["thresholds"] = scoring["thresholds"]
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Load from path, fallback to ~/.shuttersift/config.yaml, then defaults."""
        candidates = [path] if path else []
        candidates.append(Path.home() / ".shuttersift" / "config.yaml")
        for p in candidates:
            if p and p.exists():
                return cls.from_yaml(p)
        return cls()
