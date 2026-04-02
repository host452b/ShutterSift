"""
Benchmark test suite: validate ShutterSift on the committed 100-image test set.

All tests skip automatically when tests/benchmark/images/ has not been populated.
To build the benchmark set, run:

    python scripts/build_benchmark.py --help

To run just these tests:

    pytest tests/benchmark/ -v
    pytest tests/benchmark/ -v --tb=short

Coverage
--------
CEW           closed-eye recall and open-eye false-alarm rate
KonIQ-10k     per-quintile score bounds (soft) + Spearman ρ (if scipy installed)
HDR+ burst    best-frame score rank and hard-reject rate
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import pytest

from shuttersift.config import Config
from shuttersift.engine import SubScores
from shuttersift.engine.analyzers.sharpness import sharpness_score, laplacian_variance
from shuttersift.engine.analyzers.exposure import exposure_score
from shuttersift.engine.analyzers.face import FaceAnalyzer
from shuttersift.engine.analyzers.aesthetic import AestheticAnalyzer
from shuttersift.engine.analyzers.composition import composition_score as comp_score
from shuttersift.engine.scorer import Scorer

_BENCHMARK_DIR = Path(__file__).parent
_IMAGES_DIR = _BENCHMARK_DIR / "images"
_GT_FILE = _BENCHMARK_DIR / "ground_truth.json"

_HAS_IMAGES = _IMAGES_DIR.exists() and any(_IMAGES_DIR.glob("*.jpg"))

pytestmark = pytest.mark.skipif(
    not _HAS_IMAGES,
    reason=(
        "Benchmark images not present. "
        "Run: python scripts/build_benchmark.py --help"
    ),
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ground_truth() -> list[dict[str, Any]]:
    data = json.loads(_GT_FILE.read_text())
    images = data.get("images", [])
    if not images:
        pytest.skip(
            "ground_truth.json is empty. "
            "Run: python scripts/build_benchmark.py --help"
        )
    return images


@pytest.fixture(scope="module")
def benchmark_scores() -> dict[str, dict[str, Any]]:
    """
    Score every image in tests/benchmark/images/ once per test session.

    Returns a mapping {filename: result_dict} where result_dict contains:
      composite     float  overall score 0–100
      decision      str    keep / review / reject
      hard_rejected bool   True when sharpness hard-reject or all eyes closed
      all_eyes_closed bool
      lap_var       float  raw Laplacian variance
      sub           dict   per-dimension sub-scores
    """
    config = Config()
    face_az = FaceAnalyzer()
    aes_az = AestheticAnalyzer(use_gpu=False)
    scorer = Scorer(config)

    results: dict[str, dict[str, Any]] = {}
    for img_path in sorted(_IMAGES_DIR.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        lap_var = laplacian_variance(img)
        sharp = sharpness_score(img)
        exposure = exposure_score(img)
        face = face_az.analyze(img)
        aesthetic = aes_az.analyze(img)
        composition = comp_score(img, face.face_bboxes)

        sub = SubScores(
            sharpness=sharp,
            exposure=exposure,
            aesthetic=aesthetic,
            face_quality=face.face_quality_score,
            composition=composition,
        )
        composite = scorer.compute(sub)
        decision = scorer.decide(composite)
        hard_rejected = (
            lap_var < config.thresholds.hard_reject_sharpness
            or face.all_eyes_closed
        )

        results[img_path.name] = {
            "composite": composite,
            "decision": decision,
            "hard_rejected": hard_rejected,
            "all_eyes_closed": face.all_eyes_closed,
            "lap_var": lap_var,
            "sub": {
                "sharpness": sharp,
                "exposure": exposure,
                "aesthetic": aesthetic,
                "face_quality": face.face_quality_score,
                "composition": composition,
            },
        }

    return results


# ── CEW: closed-eye detection ─────────────────────────────────────────────────

class TestCEWClosedEyes:
    """
    Closed-eye images (from CEW closedFace/) must be detected and hard_rejected.

    Passing threshold: recall >= 80%  (allows for very small / occluded faces
    where MediaPipe cannot resolve eye state).
    """

    def test_closed_eye_recall(self, ground_truth, benchmark_scores):
        closed = [e for e in ground_truth if e["scenario"] == "closed_eyes"]
        if not closed:
            pytest.skip("No closed_eyes entries in ground_truth.json")

        tp = sum(
            1 for e in closed
            if benchmark_scores.get(e["filename"], {}).get("hard_rejected", False)
        )
        recall = tp / len(closed)

        # Report per-image detail on failure
        failures = [
            f"{e['filename']}: hard_rejected=False  "
            f"(all_eyes_closed={benchmark_scores.get(e['filename'], {}).get('all_eyes_closed')}, "
            f"score={benchmark_scores.get(e['filename'], {}).get('composite', '?')})"
            for e in closed
            if not benchmark_scores.get(e["filename"], {}).get("hard_rejected", False)
        ]

        assert recall >= 0.80, (
            f"Closed-eye recall {recall*100:.1f}% ({tp}/{len(closed)}) < 80%.\n"
            + "\n".join(f"  {f}" for f in failures)
        )


class TestCEWOpenEyes:
    """
    Open-eye images must NOT be incorrectly flagged as all_eyes_closed.

    Passing threshold: false-alarm rate <= 15%.
    """

    def test_open_eye_false_alarm_rate(self, ground_truth, benchmark_scores):
        open_entries = [e for e in ground_truth if e["scenario"] == "open_eyes"]
        if not open_entries:
            pytest.skip("No open_eyes entries in ground_truth.json")

        false_alarms = [
            e["filename"]
            for e in open_entries
            if benchmark_scores.get(e["filename"], {}).get("all_eyes_closed", False)
        ]
        rate = len(false_alarms) / len(open_entries)

        assert rate <= 0.15, (
            f"Open-eye false-alarm rate {rate*100:.1f}% "
            f"({len(false_alarms)}/{len(open_entries)}) > 15%.\n"
            + "\n".join(f"  {f}" for f in false_alarms)
        )


# ── KonIQ-10k: quality score correlation ─────────────────────────────────────

class TestKonIQScoring:
    """
    ShutterSift scores should correlate positively with KonIQ-10k MOS.

    Because ShutterSift uses sharpness/exposure/face dimensions rather than
    a global aesthetic model, exact MOS matching is not expected.  We use:
      - per-quintile soft bounds (≤ 30% miss rate allowed)
      - Spearman ρ > 0.20 across all KonIQ images (if scipy is installed)
    """

    def test_high_quality_score_above_minimum(self, ground_truth, benchmark_scores):
        entries = [
            e for e in ground_truth
            if e.get("expected_score_min") is not None
            and e["source"] == "KonIQ-10k"
        ]
        if not entries:
            pytest.skip("No KonIQ entries with expected_score_min in ground_truth.json")

        misses = [
            f"{e['filename']}: score={benchmark_scores.get(e['filename'], {}).get('composite', '?')} "
            f"< min={e['expected_score_min']}  MOS={e.get('original_mos')}"
            for e in entries
            if benchmark_scores.get(e["filename"], {}).get("composite", 0) < e["expected_score_min"]
        ]
        miss_rate = len(misses) / len(entries)

        assert miss_rate <= 0.30, (
            f"{len(misses)}/{len(entries)} high-quality images scored below minimum "
            f"(miss rate {miss_rate*100:.1f}% > 30%):\n"
            + "\n".join(f"  {m}" for m in misses[:10])
        )

    def test_low_quality_score_below_maximum(self, ground_truth, benchmark_scores):
        entries = [
            e for e in ground_truth
            if e.get("expected_score_max") is not None
            and e["source"] == "KonIQ-10k"
        ]
        if not entries:
            pytest.skip("No KonIQ entries with expected_score_max in ground_truth.json")

        misses = [
            f"{e['filename']}: score={benchmark_scores.get(e['filename'], {}).get('composite', '?')} "
            f"> max={e['expected_score_max']}  MOS={e.get('original_mos')}"
            for e in entries
            if benchmark_scores.get(e["filename"], {}).get("composite", 100) > e["expected_score_max"]
        ]
        miss_rate = len(misses) / len(entries)

        assert miss_rate <= 0.30, (
            f"{len(misses)}/{len(entries)} low-quality images scored above maximum "
            f"(miss rate {miss_rate*100:.1f}% > 30%):\n"
            + "\n".join(f"  {m}" for m in misses[:10])
        )

    def test_spearman_rank_correlation(self, ground_truth, benchmark_scores):
        """Composite score must have Spearman ρ > 0.20 with MOS (requires scipy)."""
        try:
            from scipy.stats import spearmanr
        except ImportError:
            pytest.skip("scipy not installed — run: pip install scipy")

        pairs = [
            (
                benchmark_scores[e["filename"]]["composite"],
                e["original_mos"],
            )
            for e in ground_truth
            if e["source"] == "KonIQ-10k"
            and e.get("original_mos") is not None
            and e["filename"] in benchmark_scores
        ]
        if len(pairs) < 10:
            pytest.skip(f"Not enough KonIQ pairs ({len(pairs)} < 10) for correlation")

        scores, mos_vals = zip(*pairs)
        rho, pval = spearmanr(scores, mos_vals)
        rho = float(rho)

        assert rho > 0.20, (
            f"Spearman ρ = {rho:.3f} (n={len(pairs)}, p={float(pval):.2e}): "
            f"composite score has weak/negative correlation with MOS. "
            f"Expected ρ > 0.20."
        )


# ── HDR+ burst: multi-frame ranking ──────────────────────────────────────────

class TestHDRPlusBursts:
    """
    In each burst group, the annotated 'best' frame should:
      1. Score highest (or within 5 points of the highest frame)
      2. Not be hard_rejected

    Passing threshold: ≤ 30% of bursts may fail (accounts for cases where
    burst frames are nearly identical in quality).
    """

    def test_best_frame_has_highest_score(self, ground_truth, benchmark_scores):
        burst_ids = {
            e["burst_id"]
            for e in ground_truth
            if e.get("burst_id") is not None
        }
        if not burst_ids:
            pytest.skip("No burst entries in ground_truth.json")

        failures = []
        for bid in sorted(burst_ids):
            burst = [e for e in ground_truth if e.get("burst_id") == bid]
            scored = {
                e["filename"]: benchmark_scores[e["filename"]]
                for e in burst
                if e["filename"] in benchmark_scores
            }
            if len(scored) < 2:
                continue

            best_entries = [e for e in burst if e["scenario"] == "burst_best"]
            if not best_entries:
                continue

            best_fname = best_entries[0]["filename"]
            best_score = scored.get(best_fname, {}).get("composite", 0)
            other_scores = [
                scored[e["filename"]]["composite"]
                for e in burst
                if e["scenario"] != "burst_best"
                and e["filename"] in scored
            ]

            if other_scores and best_score < max(other_scores) - 5.0:
                failures.append(
                    f"{bid}: best={best_score:.1f} < "
                    f"other_max={max(other_scores):.1f}  "
                    f"(diff={max(other_scores) - best_score:.1f})"
                )

        fail_rate = len(failures) / len(burst_ids) if burst_ids else 0.0
        assert fail_rate <= 0.30, (
            f"{len(failures)}/{len(burst_ids)} bursts: 'best' frame not ranked "
            f"highest (fail rate {fail_rate*100:.1f}% > 30%):\n"
            + "\n".join(f"  {f}" for f in failures)
        )

    def test_best_frame_not_hard_rejected(self, ground_truth, benchmark_scores):
        best_entries = [e for e in ground_truth if e["scenario"] == "burst_best"]
        if not best_entries:
            pytest.skip("No burst_best entries in ground_truth.json")

        rejected = [
            e["filename"]
            for e in best_entries
            if benchmark_scores.get(e["filename"], {}).get("hard_rejected", False)
        ]
        rate = len(rejected) / len(best_entries)

        assert rate <= 0.10, (
            f"{len(rejected)}/{len(best_entries)} burst 'best' frames are "
            f"hard_rejected (rate {rate*100:.1f}% > 10%)"
        )
