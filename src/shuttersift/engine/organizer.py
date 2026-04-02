from __future__ import annotations
import logging
from pathlib import Path

from shuttersift.engine import PhotoResult

logger = logging.getLogger(__name__)

_XMP_TEMPLATE = """\
<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description rdf:about=''
      xmlns:xmp='http://ns.adobe.com/xap/1.0/'
      xmlns:lr='http://ns.adobe.com/lightroom/1.0/'>
      <xmp:Rating>{rating}</xmp:Rating>
      <xmp:Label>{label}</xmp:Label>
      <lr:hierarchicalSubject>
        <rdf:Seq><rdf:li>ShutterSift/{decision}</rdf:li></rdf:Seq>
      </lr:hierarchicalSubject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""

_DECISION_META = {
    "keep":   {"rating": 5, "label": "Green"},
    "review": {"rating": 3, "label": "Yellow"},
    "reject": {"rating": 1, "label": "Red"},
}


def organize(
    results: list[PhotoResult],
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    """Create output directory structure, symlinks, and XMP sidecars."""
    for decision in ("keep", "review", "reject"):
        (output_dir / decision).mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("Dry-run: skipping file operations")
        return

    for result in results:
        target_dir = output_dir / result.decision
        dest = target_dir / result.path.name
        _create_link(result.path, dest)
        _write_xmp(result, target_dir)


def _create_link(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        # Fallback: copy if symlinks not supported (unusual on Linux/Mac)
        import shutil
        shutil.copy2(src, dest)


def _write_xmp(result: PhotoResult, target_dir: Path) -> None:
    meta = _DECISION_META[result.decision]
    xmp_content = _XMP_TEMPLATE.format(
        rating=meta["rating"],
        label=meta["label"],
        decision=result.decision,
    )
    xmp_path = target_dir / result.path.with_suffix(".xmp").name
    xmp_path.write_text(xmp_content, encoding="utf-8")
