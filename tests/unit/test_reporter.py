import json
from pathlib import Path
from shuttersift.engine import PhotoResult, AnalysisResult, SubScores
from shuttersift.engine.reporter import generate_report


def _make_results(tmp_path: Path) -> AnalysisResult:
    src = tmp_path / "DSC001.jpg"
    src.write_bytes(b"fake")
    r = PhotoResult(
        path=src, score=82.5, decision="keep",
        sub_scores=SubScores(sharpness=90, exposure=80, aesthetic=75,
                              face_quality=85, composition=70),
        reasons=[], explanation="Sharp eyes, natural smile.",
    )
    ar = AnalysisResult(photos=[r])
    return ar


def test_json_report_written(tmp_path):
    ar = _make_results(tmp_path)
    generate_report(ar, tmp_path)
    report_path = tmp_path / "results.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["version"] == "1"
    assert len(data["photos"]) == 1
    assert data["photos"][0]["score"] == 82.5


def test_html_report_written(tmp_path):
    ar = _make_results(tmp_path)
    generate_report(ar, tmp_path)
    html_path = tmp_path / "report.html"
    assert html_path.exists()
    content = html_path.read_text()
    assert "ShutterSift" in content
    assert "82.5" in content


def test_json_has_version_field(tmp_path):
    ar = _make_results(tmp_path)
    generate_report(ar, tmp_path)
    data = json.loads((tmp_path / "results.json").read_text())
    assert "version" in data
    assert "shuttersift_version" in data
    assert "run_at" in data
