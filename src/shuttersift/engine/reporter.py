from __future__ import annotations
import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from shuttersift.engine import AnalysisResult

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "templates"


def generate_report(result: AnalysisResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result, output_dir / "results.json")
    _write_html(result, output_dir / "report.html")


def _write_json(result: AnalysisResult, path: Path) -> None:
    photos = []
    for p in result.photos:
        photos.append({
            "path": str(p.path),
            "score": p.score,
            "decision": p.decision,
            "sub_scores": {
                "sharpness": p.sub_scores.sharpness,
                "exposure": p.sub_scores.exposure,
                "aesthetic": p.sub_scores.aesthetic,
                "face_quality": p.sub_scores.face_quality,
                "composition": p.sub_scores.composition,
            },
            "reasons": p.reasons,
            "explanation": p.explanation,
            "face_count": p.face_count,
            "is_duplicate": p.is_duplicate,
        })
    data = {
        "version": result.version,
        "shuttersift_version": result.shuttersift_version,
        "run_at": result.run_at,
        "photos": photos,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("JSON report → %s", path)


def _write_html(result: AnalysisResult, path: Path) -> None:
    try:
        env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)))
        tmpl = env.get_template("report.html.j2")
        html = tmpl.render(result=result)
        path.write_text(html, encoding="utf-8")
        logger.info("HTML report → %s", path)
    except Exception as e:
        logger.warning("Could not write HTML report: %s", e)
