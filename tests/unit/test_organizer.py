from pathlib import Path
from shuttersift.engine import PhotoResult
from shuttersift.engine.organizer import organize


def _result(name: str, decision: str, tmp_path: Path) -> PhotoResult:
    src = tmp_path / "src" / f"{name}.jpg"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"fake")
    return PhotoResult(path=src, decision=decision, score=75.0)


def test_creates_output_dirs(tmp_path):
    results = [_result("a", "keep", tmp_path)]
    out = tmp_path / "out"
    organize(results, out, dry_run=False)
    assert (out / "keep").exists()
    assert (out / "review").exists()
    assert (out / "reject").exists()


def test_symlink_created(tmp_path):
    results = [_result("DSC001", "keep", tmp_path)]
    out = tmp_path / "out"
    organize(results, out, dry_run=False)
    link = out / "keep" / "DSC001.jpg"
    assert link.exists() or link.is_symlink()


def test_xmp_sidecar_created(tmp_path):
    results = [_result("DSC002", "keep", tmp_path)]
    out = tmp_path / "out"
    organize(results, out, dry_run=False)
    xmp = out / "keep" / "DSC002.xmp"
    assert xmp.exists()
    content = xmp.read_text()
    assert "Rating" in content
    assert "Green" in content


def test_dry_run_no_files_created(tmp_path):
    results = [_result("DSC003", "keep", tmp_path)]
    out = tmp_path / "out"
    organize(results, out, dry_run=True)
    assert not (out / "keep" / "DSC003.jpg").exists()
