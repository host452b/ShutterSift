from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, model_validator, ConfigDict


class ScoringWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    model_config = ConfigDict(extra="forbid")
    keep: int = 70
    reject: int = 40
    hard_reject_sharpness: float = 30.0
    eye_open_min: float = 0.25
    burst_gap_seconds: float = 2.0


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weights: ScoringWeights = ScoringWeights()
    thresholds: Thresholds = Thresholds()
    workers: int = 4
    log_retention_runs: int = 30
    api_model_anthropic: str = "claude-haiku-4-5-20251001"
    api_model_openai: str = "gpt-4o-mini"
    calibrated: bool = False  # set True after auto-calibration runs

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        scoring = data.pop("scoring", {})
        if "weights" in scoring:
            data["weights"] = scoring["weights"]
        if "thresholds" in scoring:
            data["thresholds"] = scoring["thresholds"]
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Load config from first found location:
        1. explicit --config path
        2. ./shuttersift.yaml
        3. ./config.yaml
        4. ~/.shuttersift/config.yaml
        5. built-in defaults
        """
        candidates: list[Path] = []
        if path:
            candidates.append(path)
        candidates += [
            Path.cwd() / "shuttersift.yaml",
            Path.cwd() / "config.yaml",
            Path.home() / ".shuttersift" / "config.yaml",
        ]
        for p in candidates:
            if p.exists():
                return cls.from_yaml(p)
        return cls()

    def save_to_user_config(self) -> Path:
        """Persist this config to ~/.shuttersift/config.yaml and return the path."""
        import yaml as _yaml
        dest = Path.home() / ".shuttersift" / "config.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "calibrated": self.calibrated,
            "scoring": {
                "thresholds": {
                    "keep": self.thresholds.keep,
                    "reject": self.thresholds.reject,
                    "hard_reject_sharpness": self.thresholds.hard_reject_sharpness,
                    "eye_open_min": self.thresholds.eye_open_min,
                    "burst_gap_seconds": self.thresholds.burst_gap_seconds,
                }
            },
            "workers": self.workers,
            "log_retention_runs": self.log_retention_runs,
        }
        dest.write_text(_yaml.dump(data, default_flow_style=False))
        return dest
