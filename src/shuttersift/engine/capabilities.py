from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path.home() / ".shuttersift" / "models"


def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def _detect_gpu_device() -> str:
    """Returns 'cuda', 'mps', or 'cpu' — in priority order."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class Capabilities:
    gpu: bool
    gpu_device: str           # 'cuda', 'mps', or 'cpu'
    rawpy: bool
    musiq: bool
    gguf_vlm: bool            # True when a local Moondream .mf model + moondream package present
    gguf_model_path: Path | None
    api_vlm: bool

    @classmethod
    def detect(cls) -> "Capabilities":
        gpu_device = _detect_gpu_device()
        # Moondream models use the .mf (Moondream Format) extension
        mf_models = list(MODELS_DIR.glob("*.mf")) if MODELS_DIR.exists() else []
        return cls(
            gpu=gpu_device != "cpu",
            gpu_device=gpu_device,
            rawpy=_try_import("rawpy"),
            musiq=_try_import("pyiqa"),
            gguf_vlm=bool(mf_models) and _try_import("moondream"),
            gguf_model_path=mf_models[0] if mf_models else None,
            api_vlm=bool(
                os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
            ),
        )

    def summary(self) -> str:
        def flag(val: bool, label: str) -> str:
            return f"{label} {'✓' if val else '✗'}"

        gpu_label = f"GPU ({self.gpu_device.upper()})" if self.gpu else "GPU"
        parts = [
            flag(self.gpu, gpu_label),
            flag(self.rawpy, "RAW"),
            flag(self.musiq, "MUSIQ"),
            flag(self.gguf_vlm, "Local VLM"),
            flag(self.api_vlm, "API VLM"),
        ]
        return "  ".join(parts)
