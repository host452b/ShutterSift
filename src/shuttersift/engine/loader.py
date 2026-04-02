from __future__ import annotations
import logging
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg",
    ".png",
    ".cr2", ".nef", ".arw", ".dng", ".rw2", ".orf", ".raf", ".pef",
}

RAW_FORMATS = {".cr2", ".nef", ".arw", ".dng", ".rw2", ".orf", ".raf", ".pef"}


def load_image(path: Path) -> np.ndarray | None:
    """Load any supported image to BGR numpy array. Returns None on failure."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None

    suffix = path.suffix.lower()

    if suffix in RAW_FORMATS:
        return _load_raw(path)
    else:
        return _load_standard(path)


def _load_raw(path: Path) -> np.ndarray | None:
    try:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    import io
                    arr = np.frombuffer(thumb.data, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
            except Exception:
                pass
            rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        logger.warning("rawpy not available, skipping RAW: %s", path.name)
        return None
    except Exception as e:
        logger.warning("Failed to load RAW %s: %s", path.name, e)
        return None


def _load_standard(path: Path) -> np.ndarray | None:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logger.warning("Failed to load %s: %s", path.name, e)
        return None
