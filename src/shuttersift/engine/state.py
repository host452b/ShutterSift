from __future__ import annotations
import json
import logging
from dataclasses import asdict
from pathlib import Path

from shuttersift.engine import PhotoResult, SubScores

logger = logging.getLogger(__name__)

_STATE_FILE = ".state.json"


class StateManager:
    def __init__(self, output_dir: Path, fresh: bool = False):
        self._state_path = output_dir / _STATE_FILE
        self._records: dict[str, dict] = {}
        if not fresh and self._state_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        try:
            data = json.loads(self._state_path.read_text())
            self._records = {r["path"]: r for r in data.get("records", [])}
            logger.info("Resumed %d processed photos from state", len(self._records))
        except Exception as e:
            logger.warning("Could not read state file: %s", e)

    def is_processed(self, path: Path) -> bool:
        return str(path) in self._records

    def load(self, path: Path) -> PhotoResult | None:
        rec = self._records.get(str(path))
        if rec is None:
            return None
        sub = SubScores(**rec.get("sub_scores", {}))
        return PhotoResult(
            path=path,
            score=rec["score"],
            sub_scores=sub,
            decision=rec["decision"],
            reasons=rec.get("reasons", []),
            explanation=rec.get("explanation", ""),
            face_count=rec.get("face_count", 0),
            is_duplicate=rec.get("is_duplicate", False),
        )

    def save(self, result: PhotoResult) -> None:
        self._records[str(result.path)] = {
            "path": str(result.path),
            "score": result.score,
            "sub_scores": {
                "sharpness": result.sub_scores.sharpness,
                "exposure": result.sub_scores.exposure,
                "aesthetic": result.sub_scores.aesthetic,
                "face_quality": result.sub_scores.face_quality,
                "composition": result.sub_scores.composition,
            },
            "decision": result.decision,
            "reasons": result.reasons,
            "explanation": result.explanation,
            "face_count": result.face_count,
            "is_duplicate": result.is_duplicate,
        }
        self._flush()

    def _flush(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"records": list(self._records.values())}, indent=2))
        tmp.replace(self._state_path)  # atomic write
