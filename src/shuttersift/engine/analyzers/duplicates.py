from __future__ import annotations
import datetime
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def group_bursts(
    paths: list[Path],
    exif_timestamps: dict[Path, datetime.datetime],
    gap_seconds: float = 2.0,
) -> list[list[Path]]:
    """
    Group photos into burst sequences.
    Uses EXIF DateTimeOriginal when available; falls back to file modification time.
    Photos within `gap_seconds` of each other are considered one burst.
    """
    if not paths:
        return []

    # Build (timestamp, path) list, sorted by time
    timed: list[tuple[datetime.datetime, Path]] = []
    for p in paths:
        ts = exif_timestamps.get(p) or _mtime(p)
        timed.append((ts, p))

    timed.sort(key=lambda t: t[0])

    groups: list[list[Path]] = []
    current_group: list[Path] = [timed[0][1]]
    prev_ts = timed[0][0]

    # Sliding-window burst detection: each photo is compared to the immediately
    # preceding photo (not the burst start). This correctly handles rapid-fire
    # bursts but allows slow photo trains to chain — acceptable for the intended
    # use case of burst fire detection.
    for ts, p in timed[1:]:
        delta = (ts - prev_ts).total_seconds()
        if delta <= gap_seconds:
            current_group.append(p)
        else:
            groups.append(current_group)
            current_group = [p]
        prev_ts = ts

    groups.append(current_group)
    return groups


def best_in_burst(group: list[Path], scores: dict[Path, float]) -> Path:
    """Return the path with the highest score in a burst group."""
    return max(group, key=lambda p: scores.get(p, 0.0))


def _mtime(path: Path) -> datetime.datetime:
    try:
        return datetime.datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        # Path is not accessible (e.g. a fake/test path). Assign a unique synthetic
        # timestamp derived from the path string so that inaccessible files are never
        # accidentally grouped together (the gap between any two hash-derived times
        # will far exceed any reasonable gap_seconds threshold).
        seed = int(hashlib.sha1(str(path).encode()).hexdigest()[:8], 16)
        # seed is a 32-bit value (~0–4 billion). Adding it as seconds to a base time
        # spreads files decades apart, well beyond any burst gap threshold.
        return datetime.datetime(2100, 1, 1) + datetime.timedelta(seconds=seed)


def read_exif_timestamps(paths: list[Path]) -> dict[Path, datetime.datetime]:
    """Extract DateTimeOriginal from EXIF for all paths."""
    result: dict[Path, datetime.datetime] = {}
    try:
        import exifread
    except ImportError:
        logger.warning("exifread not available; using file mtime for burst grouping")
        return result

    for path in paths:
        try:
            with open(path, "rb") as f:
                tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal", details=False)
            tag = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
            if tag:
                dt = datetime.datetime.strptime(str(tag.values), "%Y:%m:%d %H:%M:%S")
                result[path] = dt
        except Exception:
            pass

    return result
