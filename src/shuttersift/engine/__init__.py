from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import datetime


@dataclass
class SubScores:
    sharpness: float = 0.0
    exposure: float = 0.0
    aesthetic: float = 0.0
    face_quality: float = 75.0   # neutral: no faces detected
    composition: float = 50.0    # neutral: rule-based N/A


@dataclass
class PhotoResult:
    path: Path
    score: float = 0.0
    sub_scores: SubScores = field(default_factory=SubScores)
    decision: Literal["keep", "review", "reject"] = "review"
    reasons: list[str] = field(default_factory=list)
    explanation: str = ""
    face_count: int = 0
    is_duplicate: bool = False
    hard_rejected: bool = False
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class AnalysisResult:
    version: str = "1"
    shuttersift_version: str = "0.1.0"
    run_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    photos: list[PhotoResult] = field(default_factory=list)

    @property
    def keep(self) -> list[PhotoResult]:
        return [p for p in self.photos if p.decision == "keep"]

    @property
    def review(self) -> list[PhotoResult]:
        return [p for p in self.photos if p.decision == "review"]

    @property
    def reject(self) -> list[PhotoResult]:
        return [p for p in self.photos if p.decision == "reject"]


from .pipeline import Engine  # noqa: F401
