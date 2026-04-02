from __future__ import annotations
import hashlib
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".shuttersift" / "models"

# Model registry: name → (url, sha256, size_hint)
# sha256 values must be updated when model versions change.
MODEL_REGISTRY: dict[str, dict] = {
    "mediapipe_face_landmarker": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "dest": MODELS_DIR / "face_landmarker.task",
        "sha256": None,  # Google rotates this; skip checksum for MediaPipe task files
        "size_hint": "3 MB",
    },
    "moondream2_gguf": {
        "url": "https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-int8.mf",
        "dest": MODELS_DIR / "moondream2-int8.mf",
        "sha256": None,  # set on release
        "size_hint": "~1.7 GB",
    },
}


def verify_sha256(path: Path, expected: str) -> bool:
    if not path.exists():
        return False
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest() == expected
    except Exception:
        return False


def _download_file(url: str, dest: Path, max_retries: int = 3) -> bool:
    """Download url to dest with progress. Returns True on success."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)...", dest.name, attempt, max_retries)
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return True
        except Exception as e:
            logger.warning("Download failed: %s", e)
            if tmp.exists():
                tmp.unlink()
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return False


def download_mediapipe_models() -> bool:
    entry = MODEL_REGISTRY["mediapipe_face_landmarker"]
    dest: Path = entry["dest"]
    if dest.exists():
        logger.info("MediaPipe face_landmarker.task already present")
        return True
    return _download_file(entry["url"], dest)


def download_gguf_vlm(model_key: str = "moondream2_gguf") -> bool:
    entry = MODEL_REGISTRY[model_key]
    dest: Path = entry["dest"]
    if dest.exists():
        logger.info("%s already present", dest.name)
        return True
    logger.info("Downloading %s (%s)...", dest.name, entry["size_hint"])
    ok = _download_file(entry["url"], dest)
    if ok and entry["sha256"]:
        if not verify_sha256(dest, entry["sha256"]):
            logger.error("SHA256 mismatch for %s — file may be corrupt", dest.name)
            dest.unlink()
            return False
    return ok
