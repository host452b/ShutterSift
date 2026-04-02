#!/usr/bin/env python3
"""
Build the committed 100-image ShutterSift benchmark set.

Samples representative images from CEW, KonIQ-10k, and Google HDR+ burst,
resizes them to ≤640px wide, and writes ground_truth.json with expected
decisions.  Run once by a maintainer, then commit the output to the repo.

Dataset sources
---------------
CEW          https://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html
             Layout: cew_dir/closedFace/  cew_dir/openFace/

KonIQ-10k    http://database.mmsp-kn.de/koniq-10k-database.html
             Layout: koniq_dir/*.jpg  + koniq_csv (image_name, MOS_mean)

HDR+ burst   gs://hdrplusdata/20171106_subset/
             Layout: hdrplus_dir/<burst_name>/*.jpg  (one dir per burst)

Default output
--------------
  tests/benchmark/images/     ← ~100 resized JPEG images
  tests/benchmark/ground_truth.json

Usage
-----
  python scripts/build_benchmark.py \\
      --cew-dir ~/datasets/CEW \\
      --koniq-dir ~/datasets/KonIQ-10k \\
      --koniq-csv ~/datasets/KonIQ-10k/koniq10k_mos_with_names.csv \\
      --hdrplus-dir ~/datasets/hdrplus

After running, review the images, then commit:
  git add tests/benchmark/
  git commit -m "feat: add 100-image benchmark test set"
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_BENCHMARK_DIR = _REPO / "tests" / "benchmark"
sys.path.insert(0, str(_REPO / "src"))

from shuttersift.config import Config
from shuttersift.engine import SubScores
from shuttersift.engine.analyzers.sharpness import sharpness_score, laplacian_variance
from shuttersift.engine.analyzers.exposure import exposure_score
from shuttersift.engine.analyzers.face import FaceAnalyzer
from shuttersift.engine.analyzers.aesthetic import AestheticAnalyzer
from shuttersift.engine.analyzers.composition import composition_score as comp_score
from shuttersift.engine.scorer import Scorer

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_MAX_WIDTH = 640
_JPEG_QUALITY = 85


# ── Image utilities ───────────────────────────────────────────────────────────

def _collect_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in _IMG_EXTS)


def _score_image(
    path: Path,
    config: Config,
    face_az: FaceAnalyzer,
    aes_az: AestheticAnalyzer,
) -> dict | None:
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
            sharpness=sharp, exposure=exposure, aesthetic=aesthetic,
            face_quality=face.face_quality_score, composition=composition,
        )
        scorer = Scorer(config)
        composite = scorer.compute(sub)
        decision = scorer.decide(composite)
        hard_rejected = (
            lap_var < config.thresholds.hard_reject_sharpness
            or face.all_eyes_closed
        )
        return {
            "composite": round(composite, 2),
            "decision": decision,
            "hard_rejected": hard_rejected,
            "all_eyes_closed": face.all_eyes_closed,
            "lap_var": round(lap_var, 2),
        }
    except Exception as exc:
        print(f"  [WARN] {path.name}: {exc}")
        return None


def _resize_save(src: Path, dest: Path, max_width: int = _MAX_WIDTH) -> None:
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"Cannot load {src}")
    h, w = img.shape[:2]
    if w > max_width:
        new_w = max_width
        new_h = int(h * max_width / w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])


# ── CEW sampling ──────────────────────────────────────────────────────────────

def sample_cew(
    cew_dir: Path,
    output_dir: Path,
    config: Config,
    n_closed: int = 10,
    n_open: int = 10,
) -> list[dict[str, Any]]:
    """
    Sample n_closed from closedFace/ and n_open from openFace/.

    Ground truth is derived from the directory label (not from runtime detection),
    so the benchmark test genuinely measures detection recall rather than
    confirming already-detected images.
    """
    closed_dir = cew_dir / "closedFace"
    open_dir = cew_dir / "openFace"
    if not closed_dir.is_dir() or not open_dir.is_dir():
        print(f"[CEW] Skipping: closedFace/ or openFace/ not found in {cew_dir}")
        return []

    entries: list[dict[str, Any]] = []

    for label, directory, n, scenario, exp_hard_rej, exp_decision, score_max in [
        ("closed", closed_dir, n_closed, "closed_eyes", True,  "reject", 40),
        ("open",   open_dir,   n_open,   "open_eyes",   False, None,     None),
    ]:
        images = _collect_images(directory)
        # Prefer loadable images; random order for diversity
        random.shuffle(images)
        selected: list[Path] = []
        for p in images:
            if len(selected) >= n:
                break
            if cv2.imread(str(p)) is not None:
                selected.append(p)

        print(f"  [CEW-{label}] Sampled {len(selected)}/{n}")

        for i, path in enumerate(selected):
            dest_name = f"cew_{label}_{i+1:03d}.jpg"
            dest = output_dir / dest_name
            _resize_save(path, dest)
            entries.append({
                "filename": dest_name,
                "source": "CEW",
                "scenario": scenario,
                "burst_id": None,
                "original_mos": None,
                "expected_hard_rejected": exp_hard_rej,
                "expected_decision": exp_decision,
                "expected_score_min": None,
                "expected_score_max": score_max,
                "notes": f"CEW {label}-eye sample (original: {path.name})",
            })

    return entries


# ── KonIQ-10k sampling ────────────────────────────────────────────────────────

def sample_koniq(
    koniq_dir: Path,
    koniq_csv: Path,
    output_dir: Path,
    config: Config,
    n_per_quintile: int = 10,
) -> list[dict[str, Any]]:
    """
    Sample n_per_quintile images from each of 5 MOS quintiles (~50 total).

    Score bounds are intentionally soft (±30% miss rate allowed) because
    ShutterSift uses different quality dimensions than human MOS.
    """
    if not koniq_dir.is_dir():
        print(f"[KonIQ] Skipping: not found: {koniq_dir}")
        return []
    if not koniq_csv.exists():
        print(f"[KonIQ] Skipping: CSV not found: {koniq_csv}")
        return []

    mos_data: list[tuple[str, float]] = []
    with koniq_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("image_name") or row.get("image_id") or ""
            mos = float(row.get("MOS_mean") or row.get("mos") or 0)
            if name:
                mos_data.append((name, mos))

    mos_data.sort(key=lambda x: x[1])
    n = len(mos_data)
    quintiles = [
        # (label,              slice,                scenario,           mos_min, mos_max)
        ("q1_low",         mos_data[:n//5],          "low_quality"),
        ("q2_below_avg",   mos_data[n//5:2*n//5],   "below_avg_quality"),
        ("q3_avg",         mos_data[2*n//5:3*n//5], "avg_quality"),
        ("q4_above_avg",   mos_data[3*n//5:4*n//5], "above_avg_quality"),
        ("q5_high",        mos_data[4*n//5:],        "high_quality"),
    ]

    entries: list[dict[str, Any]] = []
    global_idx = 1

    for quintile_label, bucket, scenario in quintiles:
        random.shuffle(bucket)
        selected: list[tuple[Path, float]] = []
        for name, mos in bucket:
            if len(selected) >= n_per_quintile:
                break
            p = koniq_dir / name
            if p.exists() and cv2.imread(str(p)) is not None:
                selected.append((p, mos))

        print(f"  [KonIQ-{quintile_label}] Sampled {len(selected)}/{n_per_quintile}")

        for path, mos in selected:
            mos_f = float(mos)
            dest_name = f"koniq_{quintile_label}_{global_idx:03d}.jpg"
            dest = output_dir / dest_name
            _resize_save(path, dest)
            entries.append({
                "filename": dest_name,
                "source": "KonIQ-10k",
                "scenario": scenario,
                "burst_id": None,
                "original_mos": round(mos_f, 3),
                "expected_hard_rejected": None,
                "expected_decision": None,
                # Soft bounds: high MOS → expect score ≥ 45, low MOS → expect score ≤ 65
                "expected_score_min": 45 if mos_f >= 3.5 else None,
                "expected_score_max": 65 if mos_f < 2.5 else None,
                "notes": f"MOS={mos_f:.2f} quintile={quintile_label}",
            })
            global_idx += 1

    return entries


# ── HDR+ burst sampling ───────────────────────────────────────────────────────

def sample_hdrplus(
    hdrplus_dir: Path,
    output_dir: Path,
    config: Config,
    n_bursts: int = 10,
    frames_per_burst: int = 3,
) -> list[dict[str, Any]]:
    """
    Sample n_bursts × frames_per_burst from HDR+ burst directories.

    Images are scored individually; the highest-scoring frame gets
    scenario="burst_best", the rest get scenario="burst_other".

    The benchmark test verifies that the best frame scores highest
    (within a 5-point tolerance) and is not hard_rejected.
    """
    if not hdrplus_dir.is_dir():
        print(f"[HDR+] Skipping: not found: {hdrplus_dir}")
        return []

    burst_dirs = sorted(d for d in hdrplus_dir.iterdir() if d.is_dir())
    random.shuffle(burst_dirs)

    face_az = FaceAnalyzer()
    aes_az = AestheticAnalyzer(use_gpu=False)

    entries: list[dict[str, Any]] = []
    burst_count = 0

    for bdir in burst_dirs:
        if burst_count >= n_bursts:
            break
        images = _collect_images(bdir)
        if len(images) < 2:
            continue

        scored: list[tuple[Path, dict]] = []
        for p in images:
            r = _score_image(p, config, face_az, aes_az)
            if r:
                scored.append((p, r))
        if len(scored) < 2:
            continue

        # Sort by composite score; take top frames_per_burst
        scored.sort(key=lambda x: x[1]["composite"], reverse=True)
        selected = scored[:frames_per_burst]
        best_score = selected[0][1]["composite"]
        burst_id = f"burst_{burst_count+1:03d}"

        print(
            f"  [HDR+] {burst_id}  {bdir.name[:30]}  "
            f"{len(selected)} frames, best={best_score}"
        )

        for rank, (path, result) in enumerate(selected):
            is_best = rank == 0
            dest_name = f"hdrplus_{burst_id}_f{rank+1:02d}.jpg"
            dest = output_dir / dest_name
            _resize_save(path, dest)
            entries.append({
                "filename": dest_name,
                "source": "HDR+",
                "scenario": "burst_best" if is_best else "burst_other",
                "burst_id": burst_id,
                "original_mos": None,
                "expected_hard_rejected": False if is_best else None,
                "expected_decision": None,
                "expected_score_min": 30 if is_best else None,
                "expected_score_max": None,
                "notes": (
                    f"{burst_id} rank={rank+1}/{len(selected)}, "
                    f"score={result['composite']}"
                ),
            })
        burst_count += 1

    return entries


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cew-dir", type=Path, metavar="PATH")
    parser.add_argument("--koniq-dir", type=Path, metavar="PATH")
    parser.add_argument("--koniq-csv", type=Path, metavar="PATH")
    parser.add_argument("--hdrplus-dir", type=Path, metavar="PATH")
    parser.add_argument(
        "--output-dir", type=Path, default=_BENCHMARK_DIR / "images",
        help="Destination for benchmark images (default: tests/benchmark/images/)",
    )
    parser.add_argument(
        "--gt-file", type=Path, default=_BENCHMARK_DIR / "ground_truth.json",
        help="Ground truth output (default: tests/benchmark/ground_truth.json)",
    )
    parser.add_argument("--n-cew-closed", type=int, default=10,
                        help="Closed-eye images from CEW (default: 10)")
    parser.add_argument("--n-cew-open", type=int, default=10,
                        help="Open-eye images from CEW (default: 10)")
    parser.add_argument("--n-koniq-per-quintile", type=int, default=10,
                        help="KonIQ-10k images per quality quintile (default: 10, total ~50)")
    parser.add_argument("--n-hdrplus-bursts", type=int, default=10,
                        help="Number of HDR+ bursts (default: 10)")
    parser.add_argument("--frames-per-burst", type=int, default=3,
                        help="Frames per burst (default: 3, total ~30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    args = parser.parse_args()

    if not any([args.cew_dir, args.koniq_dir, args.hdrplus_dir]):
        parser.print_help()
        sys.exit(1)

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = Config()
    all_entries: list[dict[str, Any]] = []

    if args.cew_dir:
        print("\n=== Sampling CEW ===")
        entries = sample_cew(
            args.cew_dir, args.output_dir, config,
            args.n_cew_closed, args.n_cew_open,
        )
        all_entries.extend(entries)
        print(f"  → {len(entries)} images")

    if args.koniq_dir:
        print("\n=== Sampling KonIQ-10k ===")
        if not args.koniq_csv:
            print("  [SKIP] --koniq-csv required for KonIQ-10k")
        else:
            entries = sample_koniq(
                args.koniq_dir, args.koniq_csv, args.output_dir, config,
                args.n_koniq_per_quintile,
            )
            all_entries.extend(entries)
            print(f"  → {len(entries)} images")

    if args.hdrplus_dir:
        print("\n=== Sampling HDR+ ===")
        entries = sample_hdrplus(
            args.hdrplus_dir, args.output_dir, config,
            args.n_hdrplus_bursts, args.frames_per_burst,
        )
        all_entries.extend(entries)
        print(f"  → {len(entries)} images")

    gt = {
        "version": "1",
        "generated_by": "scripts/build_benchmark.py",
        "description": (
            "100-image ShutterSift benchmark set. "
            "Sources: CEW (closed/open eyes), KonIQ-10k (quality tiers), "
            "Google HDR+ burst (multi-frame selection)."
        ),
        "n_images": len(all_entries),
        "images": all_entries,
    }
    args.gt_file.parent.mkdir(parents=True, exist_ok=True)
    args.gt_file.write_text(json.dumps(gt, indent=2))

    # Summary
    sources = {}
    for e in all_entries:
        sources[e["source"]] = sources.get(e["source"], 0) + 1

    print("\n" + "=" * 50)
    print(f"Total images : {len(all_entries)}")
    for src, cnt in sources.items():
        print(f"  {src:<12}: {cnt}")
    print(f"\nImages  → {args.output_dir}")
    print(f"GT file → {args.gt_file}")
    print(
        "\nNext steps:\n"
        "  1. Review images in output_dir\n"
        "  2. pytest tests/benchmark/ -v\n"
        "  3. git add tests/benchmark/ && git commit -m 'feat: add benchmark test set'"
    )


if __name__ == "__main__":
    main()
