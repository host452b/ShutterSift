#!/usr/bin/env python3
"""
Validate ShutterSift scoring against three public benchmark datasets.

Datasets
--------
CEW      Closed Eyes in the Wild — binary classification (closed / open face)
           https://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html
           Expected layout:  cew_dir/closedFace/   cew_dir/openFace/

KonIQ-10k  MOS-annotated quality images
           http://database.mmsp-kn.de/koniq-10k-database.html
           Expected layout:  koniq_dir/*.jpg
                             koniq_csv  (columns: image_name, MOS_mean)

HDR+ burst  Google HDR+ Burst dataset (JPEG sub-set or DNG)
           gs://hdrplusdata/20171106_subset/
           Expected layout:  hdrplus_dir/<burst_name>/*.jpg  (one dir per burst)

Outputs
-------
  Console table with per-dataset metrics
  JSON report written to --output (default: validation_report.json)

Usage examples
--------------
  python scripts/validate_datasets.py --cew-dir ~/datasets/CEW
  python scripts/validate_datasets.py \\
      --koniq-dir ~/datasets/KonIQ-10k \\
      --koniq-csv ~/datasets/KonIQ-10k/koniq10k_mos_with_names.csv
  python scripts/validate_datasets.py --hdrplus-dir ~/datasets/hdrplus
  python scripts/validate_datasets.py \\
      --cew-dir ~/datasets/CEW \\
      --koniq-dir ~/datasets/KonIQ-10k \\
      --koniq-csv ~/datasets/KonIQ-10k/koniq10k_mos_with_names.csv \\
      --hdrplus-dir ~/datasets/hdrplus
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from shuttersift.config import Config
from shuttersift.engine import SubScores
from shuttersift.engine.analyzers.sharpness import sharpness_score, laplacian_variance
from shuttersift.engine.analyzers.exposure import exposure_score
from shuttersift.engine.analyzers.face import FaceAnalyzer
from shuttersift.engine.analyzers.aesthetic import AestheticAnalyzer
from shuttersift.engine.analyzers.composition import composition_score as comp_score
from shuttersift.engine.scorer import Scorer

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Shared scorer ─────────────────────────────────────────────────────────────

def _score_image(
    path: Path,
    config: Config,
    face_az: FaceAnalyzer,
    aes_az: AestheticAnalyzer,
) -> dict | None:
    """Score one image. Returns a result dict or None on load / scoring failure."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    try:
        lap_var = laplacian_variance(img)
        sharp = sharpness_score(img)
        exposure = exposure_score(img)
        face = face_az.analyze(img)
        aesthetic = aes_az.score(img)
        composition = comp_score(img, face.face_bboxes)

        sub = SubScores(
            sharpness=sharp,
            exposure=exposure,
            aesthetic=aesthetic,
            face_quality=face.face_quality_score,
            composition=composition,
        )
        scorer = Scorer(config)
        composite = scorer.compute(sub)
        decision = scorer.decide(composite)
        hard_rejected = (
            lap_var < config.thresholds.hard_reject_sharpness
            or face.all_eyes_closed
        )

        return {
            "path": str(path),
            "filename": path.name,
            "composite": round(composite, 2),
            "decision": decision,
            "hard_rejected": hard_rejected,
            "all_eyes_closed": face.all_eyes_closed,
            "lap_var": round(lap_var, 2),
            "sub": {
                "sharpness": round(sharp, 2),
                "exposure": round(exposure, 2),
                "aesthetic": round(aesthetic, 2),
                "face_quality": round(face.face_quality_score, 2),
                "composition": round(composition, 2),
            },
        }
    except Exception as exc:
        print(f"  [WARN] {path.name}: {exc}")
        return None


def _collect_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in _IMG_EXTS)


# ── CEW ───────────────────────────────────────────────────────────────────────

def validate_cew(cew_dir: Path, config: Config) -> dict:
    """
    Metric: precision / recall / F1 for closed-eye hard-reject detection.

    True positive  = closed-eye image correctly hard_rejected
    False negative = closed-eye image NOT hard_rejected
    False positive = open-eye image incorrectly hard_rejected
    """
    closed_dir = cew_dir / "closedFace"
    open_dir = cew_dir / "openFace"
    missing = [d for d in (closed_dir, open_dir) if not d.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Expected closedFace/ and openFace/ inside --cew-dir. "
            f"Missing: {missing}"
        )

    face_az = FaceAnalyzer()
    aes_az = AestheticAnalyzer(use_gpu=False)

    print("[CEW] Scoring closed-eye images …")
    closed_images = _collect_images(closed_dir)
    closed_results = [
        r for p in closed_images
        if (r := _score_image(p, config, face_az, aes_az)) is not None
    ]

    print(f"[CEW] Scoring open-eye images …")
    open_images = _collect_images(open_dir)
    open_results = [
        r for p in open_images
        if (r := _score_image(p, config, face_az, aes_az)) is not None
    ]

    tp = sum(1 for r in closed_results if r["hard_rejected"])
    fn = len(closed_results) - tp
    fp = sum(1 for r in open_results if r["hard_rejected"])
    tn = len(open_results) - fp

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(
        f"  Closed ({len(closed_results)}): {tp} hard_rejected  "
        f"recall={recall*100:.1f}%"
    )
    print(
        f"  Open   ({len(open_results)}): {fp} false positives  "
        f"specificity={specificity*100:.1f}%"
    )
    print(f"  Precision={precision*100:.1f}%  Recall={recall*100:.1f}%  F1={f1*100:.1f}%")

    return {
        "n_closed": len(closed_results),
        "n_open": len(open_results),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
    }


# ── KonIQ-10k ─────────────────────────────────────────────────────────────────

def validate_koniq(
    koniq_dir: Path,
    koniq_csv: Path,
    config: Config,
    max_images: int = 1000,
) -> dict:
    """
    Metric: Spearman ρ between ShutterSift composite score and human MOS.

    A ρ > 0.30 indicates meaningful positive correlation.
    """
    if not koniq_dir.is_dir():
        raise FileNotFoundError(f"KonIQ-10k image directory not found: {koniq_dir}")
    if not koniq_csv.exists():
        raise FileNotFoundError(f"KonIQ-10k MOS CSV not found: {koniq_csv}")

    print(f"[KonIQ] Reading MOS scores from {koniq_csv.name} …")
    mos_map: dict[str, float] = {}
    with koniq_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("image_name") or row.get("image_id") or ""
            mos = float(row.get("MOS_mean") or row.get("mos") or 0)
            if name:
                mos_map[name] = mos

    print(f"[KonIQ] {len(mos_map)} MOS annotations loaded.")

    all_images = sorted(koniq_dir.glob("*.jpg"))
    annotated = [(p, mos_map[p.name]) for p in all_images if p.name in mos_map]

    # Uniform subsample if too many
    if len(annotated) > max_images:
        step = len(annotated) // max_images
        annotated = annotated[::step][:max_images]

    face_az = FaceAnalyzer()
    aes_az = AestheticAnalyzer(use_gpu=False)

    print(f"[KonIQ] Scoring {len(annotated)} images …")
    scores: list[float] = []
    mos_values: list[float] = []
    for path, mos in annotated:
        r = _score_image(path, config, face_az, aes_az)
        if r:
            scores.append(r["composite"])
            mos_values.append(mos)

    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(scores, mos_values)
        rho_val = float(rho)
        pval_val = float(pval)
        print(f"  Spearman ρ = {rho_val:.3f}  (p = {pval_val:.2e},  n = {len(scores)})")
    except ImportError:
        rho_val, pval_val = float("nan"), float("nan")
        print("  [WARN] scipy not installed — Spearman ρ not computed.")
        print("         Install with: pip install scipy")

    # Score distribution by MOS quintile
    quintile_stats = _koniq_quintile_stats(list(zip(mos_values, scores)))

    return {
        "n_images": len(scores),
        "spearman_rho": round(rho_val, 4) if not (rho_val != rho_val) else None,
        "p_value": pval_val,
        "score_mean": round(float(np.mean(scores)), 2),
        "score_std": round(float(np.std(scores)), 2),
        "mos_mean": round(float(np.mean(mos_values)), 2),
        "quintile_stats": quintile_stats,
    }


def _koniq_quintile_stats(pairs: list[tuple[float, float]]) -> list[dict]:
    """Return mean composite score per MOS quintile (low → high)."""
    if not pairs:
        return []
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    q_size = n // 5
    results = []
    for i in range(5):
        bucket = pairs[i * q_size: (i + 1) * q_size] if i < 4 else pairs[i * q_size:]
        mos_vals = [p[0] for p in bucket]
        score_vals = [p[1] for p in bucket]
        results.append({
            "quintile": i + 1,
            "mos_range": [round(min(mos_vals), 2), round(max(mos_vals), 2)],
            "mean_mos": round(float(np.mean(mos_vals)), 2),
            "mean_score": round(float(np.mean(score_vals)), 2),
            "n": len(bucket),
        })
    return results


# ── HDR+ burst ────────────────────────────────────────────────────────────────

def validate_hdrplus(
    hdrplus_dir: Path,
    config: Config,
    max_bursts: int = 50,
) -> dict:
    """
    Metric: within-burst score spread and best-frame quality rate.

    A large spread indicates the scorer differentiates frames within a burst.
    A high best-frame quality rate (score >= reject threshold) means
    the best frame in each burst is usable.
    """
    if not hdrplus_dir.is_dir():
        raise FileNotFoundError(f"HDR+ directory not found: {hdrplus_dir}")

    burst_dirs = sorted(d for d in hdrplus_dir.iterdir() if d.is_dir())[:max_bursts]
    face_az = FaceAnalyzer()
    aes_az = AestheticAnalyzer(use_gpu=False)

    print(f"[HDR+] Processing up to {max_bursts} burst directories …")
    burst_results = []

    for bdir in burst_dirs:
        images = _collect_images(bdir)
        if len(images) < 2:
            continue

        scored = [
            r for p in images
            if (r := _score_image(p, config, face_az, aes_az)) is not None
        ]
        if not scored:
            continue

        burst_scores = [r["composite"] for r in scored]
        burst_results.append({
            "burst_dir": bdir.name,
            "n_frames": len(scored),
            "best_score": round(max(burst_scores), 2),
            "worst_score": round(min(burst_scores), 2),
            "spread": round(max(burst_scores) - min(burst_scores), 2),
            "mean_score": round(float(np.mean(burst_scores)), 2),
        })

    if not burst_results:
        print("[HDR+] No valid burst directories found.")
        return {"n_bursts": 0}

    best_frame_quality_rate = sum(
        1 for b in burst_results
        if b["best_score"] >= config.thresholds.reject
    ) / len(burst_results)

    spreads = [b["spread"] for b in burst_results]
    avg_spread = float(np.mean(spreads))
    median_spread = float(np.median(spreads))

    print(f"  Bursts scored: {len(burst_results)}")
    print(
        f"  Best-frame quality rate (>= reject threshold): "
        f"{best_frame_quality_rate*100:.1f}%"
    )
    print(
        f"  Within-burst score spread: "
        f"avg={avg_spread:.1f}  median={median_spread:.1f}"
    )

    return {
        "n_bursts": len(burst_results),
        "best_frame_quality_rate": round(best_frame_quality_rate, 4),
        "avg_within_burst_spread": round(avg_spread, 2),
        "median_within_burst_spread": round(median_spread, 2),
        "burst_details": burst_results[:20],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cew-dir", type=Path, metavar="PATH",
                        help="CEW dataset root (contains closedFace/ and openFace/)")
    parser.add_argument("--koniq-dir", type=Path, metavar="PATH",
                        help="KonIQ-10k image directory (*.jpg files)")
    parser.add_argument("--koniq-csv", type=Path, metavar="PATH",
                        help="KonIQ-10k MOS CSV (image_name, MOS_mean columns)")
    parser.add_argument("--hdrplus-dir", type=Path, metavar="PATH",
                        help="HDR+ burst root (one sub-directory per burst)")
    parser.add_argument("--output", type=Path, default=Path("validation_report.json"),
                        help="Output JSON report (default: validation_report.json)")
    parser.add_argument("--max-koniq-images", type=int, default=1000,
                        help="Max KonIQ images to score (default: 1000)")
    parser.add_argument("--max-hdrplus-bursts", type=int, default=50,
                        help="Max HDR+ burst dirs to process (default: 50)")
    args = parser.parse_args()

    if not any([args.cew_dir, args.koniq_dir, args.hdrplus_dir]):
        parser.print_help()
        sys.exit(1)

    config = Config()
    report: dict = {"datasets": {}, "config": {
        "keep_threshold": config.thresholds.keep,
        "reject_threshold": config.thresholds.reject,
        "hard_reject_sharpness": config.thresholds.hard_reject_sharpness,
    }}
    t0 = time.time()

    if args.cew_dir:
        print("\n" + "=" * 50)
        try:
            report["datasets"]["cew"] = validate_cew(args.cew_dir, config)
        except Exception as e:
            print(f"[CEW] ERROR: {e}")
            report["datasets"]["cew"] = {"error": str(e)}

    if args.koniq_dir:
        print("\n" + "=" * 50)
        if not args.koniq_csv:
            print("[KonIQ] ERROR: --koniq-csv is required when --koniq-dir is specified")
            report["datasets"]["koniq"] = {"error": "--koniq-csv not provided"}
        else:
            try:
                report["datasets"]["koniq"] = validate_koniq(
                    args.koniq_dir, args.koniq_csv, config, args.max_koniq_images
                )
            except Exception as e:
                print(f"[KonIQ] ERROR: {e}")
                report["datasets"]["koniq"] = {"error": str(e)}

    if args.hdrplus_dir:
        print("\n" + "=" * 50)
        try:
            report["datasets"]["hdrplus"] = validate_hdrplus(
                args.hdrplus_dir, config, args.max_hdrplus_bursts
            )
        except Exception as e:
            print(f"[HDR+] ERROR: {e}")
            report["datasets"]["hdrplus"] = {"error": str(e)}

    report["elapsed_seconds"] = round(time.time() - t0, 1)

    args.output.write_text(json.dumps(report, indent=2))
    print(f"\n{'=' * 50}")
    print(f"Report saved → {args.output}  ({report['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
