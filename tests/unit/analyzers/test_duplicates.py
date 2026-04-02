import datetime
from pathlib import Path
from shuttersift.engine.analyzers.duplicates import group_bursts, best_in_burst


def _make_paths(names):
    return [Path(f"/photos/{n}") for n in names]


def test_no_burst_single_files():
    paths = _make_paths(["a.jpg", "b.jpg", "c.jpg"])
    # No EXIF timestamps, files far apart in name
    groups = group_bursts(paths, exif_timestamps={})
    assert len(groups) == 3  # each file in own group


def test_burst_group_by_timestamp():
    paths = _make_paths(["DSC001.jpg", "DSC002.jpg", "DSC003.jpg", "DSC010.jpg"])
    base = datetime.datetime(2026, 4, 1, 12, 0, 0)
    timestamps = {
        paths[0]: base,
        paths[1]: base + datetime.timedelta(seconds=0.5),
        paths[2]: base + datetime.timedelta(seconds=0.9),
        paths[3]: base + datetime.timedelta(seconds=30),  # separate
    }
    groups = group_bursts(paths, exif_timestamps=timestamps, gap_seconds=2.0)
    assert len(groups) == 2
    assert len(groups[0]) == 3
    assert len(groups[1]) == 1


def test_best_in_burst_returns_highest_score():
    paths = _make_paths(["a.jpg", "b.jpg", "c.jpg"])
    scores = {paths[0]: 45.0, paths[1]: 78.0, paths[2]: 62.0}
    best = best_in_burst(paths, scores)
    assert best == paths[1]
