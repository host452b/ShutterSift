from __future__ import annotations
import logging
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import pyiqa
    _PYIQA_AVAILABLE = True
except ImportError:
    _PYIQA_AVAILABLE = False


class AestheticAnalyzer:
    def __init__(self, use_gpu: bool = False):
        self._use_gpu = use_gpu
        self._model = None
        self._backend = "none"
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if _PYIQA_AVAILABLE:
            try:
                device = "cuda" if self._use_gpu else "cpu"
                # Try MPS on Apple Silicon
                try:
                    import torch
                    if torch.backends.mps.is_available() and self._use_gpu:
                        device = "mps"
                except Exception:
                    pass
                self._model = pyiqa.create_metric("musiq", device=device)
                self._backend = "musiq"
                logger.info("Aesthetic backend: MUSIQ (%s)", device)
                return
            except Exception as e:
                logger.warning("MUSIQ failed to load (%s), falling back to BRISQUE", e)
        self._backend = "brisque"
        logger.info("Aesthetic backend: BRISQUE (CPU)")

    def score(self, img: np.ndarray) -> float:
        """Return aesthetic quality score [0–100]."""
        self._load()
        if self._backend == "musiq":
            return self._score_musiq(img)
        return self._score_brisque(img)

    def _score_musiq(self, img: np.ndarray) -> float:
        import torch
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            raw = self._model(pil)
            # pyiqa MUSIQ returns a tensor; scores are ~0–100
            val = float(raw.item()) if hasattr(raw, "item") else float(raw)
        return round(max(0.0, min(100.0, val)), 2)

    def _score_brisque(self, img: np.ndarray) -> float:
        """
        BRISQUE via OpenCV. Raw score: lower = better quality (0–100 range, inverted).
        """
        try:
            brisque = cv2.quality.QualityBRISQUE_compute(
                img, "brisque_model_live.yml", "brisque_range_live.yml"
            )
            raw = float(brisque[0])
            # BRISQUE: 0 = perfect, 100 = worst. Invert to match our convention.
            return round(max(0.0, min(100.0, 100.0 - raw)), 2)
        except Exception:
            # Fallback: use Laplacian-based estimate as last resort
            from .sharpness import sharpness_score
            return sharpness_score(img) * 0.6 + 40.0  # bias toward mid-range
