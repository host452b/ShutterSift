from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path.home() / ".shuttersift" / "models"


def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        return False


@dataclass
class Capabilities:
    gpu: bool
    rawpy: bool
    musiq: bool
    gguf_vlm: bool
    gguf_model_path: Path | None
    api_vlm: bool

    @classmethod
    def detect(cls) -> "Capabilities":
        gguf_models = list(MODELS_DIR.glob("*.gguf")) if MODELS_DIR.exists() else []
        return cls(
            gpu=_has_gpu(),
            rawpy=_try_import("rawpy"),
            musiq=_try_import("pyiqa"),
            gguf_vlm=bool(gguf_models),
            gguf_model_path=gguf_models[0] if gguf_models else None,
            api_vlm=bool(
                os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
            ),
        )

    def summary(self) -> str:
        def flag(val: bool, label: str) -> str:
            return f"{label} {'✓' if val else '✗'}"

        parts = [
            flag(self.gpu, "GPU"),
            flag(self.rawpy, "RAW"),
            flag(self.musiq, "MUSIQ"),
            flag(self.gguf_vlm, "GGUF VLM"),
            flag(self.api_vlm, "API VLM"),
        ]
        return "  ".join(parts)
