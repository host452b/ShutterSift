from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Callable

from shuttersift.config import Config
from shuttersift.engine import AnalysisResult, PhotoResult, SubScores
from shuttersift.engine.capabilities import Capabilities
from shuttersift.engine.loader import load_image, SUPPORTED_FORMATS
from shuttersift.engine.analyzers.sharpness import sharpness_score, laplacian_variance
from shuttersift.engine.analyzers.exposure import exposure_score
from shuttersift.engine.analyzers.face import FaceAnalyzer
from shuttersift.engine.analyzers.aesthetic import AestheticAnalyzer
from shuttersift.engine.analyzers.composition import composition_score
from shuttersift.engine.analyzers.duplicates import (
    group_bursts, best_in_burst, read_exif_timestamps,
)
from shuttersift.engine.scorer import Scorer
from shuttersift.engine.state import StateManager
from shuttersift.engine.explainer import Explainer
from shuttersift.engine.organizer import organize
from shuttersift.engine.reporter import generate_report

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, config: Config):
        self._cfg = config
        self._caps = Capabilities.detect()
        self._face = FaceAnalyzer()
        self._aesthetic = AestheticAnalyzer(use_gpu=self._caps.gpu)
        self._scorer = Scorer(config)

    def capabilities(self) -> dict:
        return {
            "gpu": self._caps.gpu,
            "rawpy": self._caps.rawpy,
            "musiq": self._caps.musiq,
            "gguf_vlm": self._caps.gguf_vlm,
            "api_vlm": self._caps.api_vlm,
        }

    def analyze(
        self,
        input_dir: Path,
        output_dir: Path,
        on_progress: Callable[[int, int, PhotoResult], None] | None = None,
        resume: bool = True,
        dry_run: bool = False,
        explain: bool = False,
    ) -> AnalysisResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        state = StateManager(output_dir, fresh=not resume)

        # Stage 0: scan + burst grouping
        paths = self._scan(input_dir)
        if not paths:
            logger.warning("No supported photos found in %s", input_dir)
            return AnalysisResult()

        exif_ts = read_exif_timestamps(paths)
        burst_groups = group_bursts(
            paths, exif_ts, self._cfg.thresholds.burst_gap_seconds
        )

        # Auto-calibrate sharpness threshold from this session's distribution
        hard_reject_threshold = self._calibrate_sharpness(paths, state)

        # Main analysis loop
        results: list[PhotoResult] = []
        total = len(paths)

        for i, path in enumerate(paths):
            if state.is_processed(path):
                cached = state.load(path)
                if cached:
                    results.append(cached)
                    if on_progress:
                        on_progress(i + 1, total, cached)
                    continue

            t0 = time.perf_counter()
            result = self._analyze_one(path, hard_reject_threshold)
            result.duration_ms = (time.perf_counter() - t0) * 1000

            state.save(result)
            results.append(result)
            if on_progress:
                on_progress(i + 1, total, result)

        # Stage 5: burst dedup — mark duplicates
        all_scores = {r.path: r.score for r in results}
        for group in burst_groups:
            if len(group) <= 1:
                continue
            best = best_in_burst(group, all_scores)
            for r in results:
                if r.path in group and r.path != best:
                    r.is_duplicate = True
                    r.decision = "reject"
                    r.reasons.append("连拍重复（非最优帧）")

        # Stage 6: VLM explanation (optional)
        if explain and (self._caps.gguf_vlm or self._caps.api_vlm):
            explainer = Explainer(
                self._cfg,
                gguf_path=self._caps.gguf_model_path,
                api_key_anthropic=os.getenv("ANTHROPIC_API_KEY"),
                api_key_openai=os.getenv("OPENAI_API_KEY"),
            )
            review_results = [r for r in results if r.decision == "review"]
            for j, r in enumerate(review_results):
                r.explanation = explainer.explain(r.path, r)

        analysis = AnalysisResult(photos=results)
        organize(results, output_dir, dry_run=dry_run)
        generate_report(analysis, output_dir)
        return analysis

    def _analyze_one(self, path: Path, hard_reject_threshold: float) -> PhotoResult:
        result = PhotoResult(path=path)

        img = load_image(path)
        if img is None:
            result.decision = "reject"
            result.error = "Could not load image"
            result.reasons.append("文件损坏或格式不支持")
            return result

        # Stage 2: technical quality
        sharp = sharpness_score(img)
        expo = exposure_score(img)
        result.sub_scores.sharpness = sharp
        result.sub_scores.exposure = expo

        raw_var = laplacian_variance(img)
        if raw_var < hard_reject_threshold:
            result.decision = "reject"
            result.hard_rejected = True
            result.reasons.append(f"严重模糊 (Laplacian={raw_var:.1f})")
            result.score = self._scorer.compute(result.sub_scores)
            return result  # early stop

        if expo < 20:
            result.reasons.append(f"曝光问题 (score={expo:.0f})")

        # Stage 3: face analysis
        face_result = self._face.analyze(img)
        result.face_count = face_result.count
        result.sub_scores.face_quality = face_result.face_quality_score

        if face_result.count > 0 and face_result.all_eyes_closed:
            result.decision = "reject"
            result.hard_rejected = True
            result.reasons.append("所有人物闭眼")
            result.score = self._scorer.compute(result.sub_scores)
            return result  # early stop

        if face_result.count > 0 and face_result.eye_open_score < 0.5:
            result.reasons.append(f"眼睛半闭 (score={face_result.eye_open_score:.2f})")

        # Stage 4: aesthetic + composition
        aesthetic = self._aesthetic.score(img)
        comp = composition_score(img, face_result.face_bboxes)
        result.sub_scores.aesthetic = aesthetic
        result.sub_scores.composition = comp

        if aesthetic < 30:
            result.reasons.append(f"美学分低 ({aesthetic:.0f})")

        # Final score + decision
        result.score = self._scorer.compute(result.sub_scores)
        result.decision = self._scorer.decide(result.score)
        return result

    def _scan(self, input_dir: Path) -> list[Path]:
        paths = []
        for p in sorted(input_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS:
                paths.append(p)
        return paths

    def _calibrate_sharpness(
        self, paths: list[Path], state: StateManager
    ) -> float:
        """
        Compute p10 of Laplacian variance distribution to use as hard-reject threshold.
        Only fast-scans images not already in state.
        Falls back to config default if fewer than 10 new images.
        """
        default = self._cfg.thresholds.hard_reject_sharpness
        new_paths = [p for p in paths if not state.is_processed(p)]
        if len(new_paths) < 10:
            return default

        variances = []
        for p in new_paths[:200]:  # cap at 200 for speed
            img = load_image(p)
            if img is not None:
                variances.append(laplacian_variance(img))

        if len(variances) < 5:
            return default

        variances.sort()
        p10_idx = max(0, int(len(variances) * 0.10) - 1)
        calibrated = variances[p10_idx]
        logger.info(
            "Sharpness calibration: p10=%.1f (config default was %.1f)",
            calibrated, default
        )
        return calibrated
