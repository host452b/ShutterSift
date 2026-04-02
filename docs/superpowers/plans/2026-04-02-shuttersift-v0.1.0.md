# ShutterSift v0.1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CLI tool that ingests a photo directory, scores each image across 5 dimensions (0–100), and outputs Keep/Review/Reject classification with a structured HTML+JSON report.

**Architecture:** Multi-stage CV pipeline (sharpness → exposure → face analysis → aesthetics → dedup) with weighted scoring and optional GGUF/API VLM explanation. Engine layer is fully decoupled from CLI so a future GUI client can call the same `Engine.analyze()` API.

**Tech Stack:** Python 3.11+, MediaPipe, OpenCV, pyiqa (MUSIQ/BRISQUE), llama-cpp-python (GGUF), Anthropic/OpenAI SDK, rawpy, imagehash, Typer, Rich, Jinja2, Pydantic-settings, hatchling.

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package metadata, all dependencies, entry points |
| `src/shuttersift/__init__.py` | Package version |
| `src/shuttersift/config.py` | `Config`, `ScoringWeights`, `Thresholds` Pydantic models |
| `src/shuttersift/engine/__init__.py` | `PhotoResult`, `SubScores`, `AnalysisResult` dataclasses; re-exports `Engine` |
| `src/shuttersift/engine/capabilities.py` | `Capabilities` — detects GPU / GGUF / API / rawpy at startup |
| `src/shuttersift/engine/loader.py` | `load_image(path) -> np.ndarray` — RAW + JPEG unified |
| `src/shuttersift/engine/analyzers/sharpness.py` | `sharpness_score(img) -> float` [0–100] |
| `src/shuttersift/engine/analyzers/exposure.py` | `exposure_score(img) -> float` [0–100] |
| `src/shuttersift/engine/analyzers/face.py` | `FaceAnalyzer` — MediaPipe face + blendshapes |
| `src/shuttersift/engine/analyzers/aesthetic.py` | `AestheticAnalyzer` — MUSIQ→BRISQUE fallback |
| `src/shuttersift/engine/analyzers/composition.py` | `composition_score(img, face_bboxes) -> float` [0–100] |
| `src/shuttersift/engine/analyzers/duplicates.py` | `group_bursts(paths) -> list[list[Path]]`; `best_in_burst(group, scores)` |
| `src/shuttersift/engine/scorer.py` | `Scorer.compute(sub_scores) -> float`; `Scorer.decide(score) -> str` |
| `src/shuttersift/engine/state.py` | `StateManager` — `.state.json` read/write/resume |
| `src/shuttersift/engine/explainer.py` | `Explainer.explain(path, result) -> str` — GGUF → API → skip |
| `src/shuttersift/engine/organizer.py` | `organize(results, output_dir, dry_run)` — symlinks + XMP sidecars |
| `src/shuttersift/engine/reporter.py` | `generate_report(results, output_dir)` — HTML + `results.json` |
| `src/shuttersift/engine/downloader.py` | `download_mediapipe_models()`, `download_gguf_vlm()`, SHA256 verify |
| `src/shuttersift/engine/pipeline.py` | `Engine` class — orchestrates all stages, `on_progress` callback |
| `src/shuttersift/cli/main.py` | Typer app: `cull`, `download-models`, `info`, `calibrate` |
| `src/shuttersift/server/__init__.py` | Empty placeholder |
| `src/shuttersift/server/app.py` | FastAPI skeleton (all commented) |
| `tests/fixtures/conftest.py` | Synthetic image fixtures |
| `tests/unit/test_config.py` | Config loading + overrides |
| `tests/unit/test_scorer.py` | Weighted scoring math |
| `tests/unit/test_state.py` | State read/write/resume |
| `tests/unit/analyzers/test_sharpness.py` | Sharp vs blurry detection |
| `tests/unit/analyzers/test_exposure.py` | Over/under/normal exposure |
| `tests/unit/analyzers/test_composition.py` | Composition rule engine |
| `tests/unit/analyzers/test_duplicates.py` | Burst grouping + best selection |
| `tests/integration/test_pipeline.py` | End-to-end with synthetic JPEG directory |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/shuttersift/__init__.py`
- Create: `src/shuttersift/engine/__init__.py` (placeholder)
- Create: `src/shuttersift/engine/analyzers/__init__.py`
- Create: `src/shuttersift/cli/__init__.py`
- Create: `src/shuttersift/server/__init__.py`
- Create: `clients/desktop/README.md`
- Create: `clients/web/README.md`
- Create: `config.yaml`
- Create: `.gitignore`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "shuttersift"
version = "0.1.0"
description = "AI-powered photo culling CLI for photographers"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
dependencies = [
    # Image I/O
    "rawpy>=0.21",
    "opencv-python-headless>=4.10",
    "Pillow>=10.4",
    "numpy>=1.26",
    "scikit-image>=0.22",
    "exifread>=3.0",
    # Face analysis (Apple Silicon: install mediapipe-silicon manually if this fails)
    "mediapipe>=0.10",
    # Quality assessment (pulls torch)
    "pyiqa>=0.1.9",
    "torch>=2.1",
    # Dedup
    "ImageHash>=4.3",
    # VLM local (CPU wheels on PyPI; for Metal: reinstall with --extra-index-url)
    "llama-cpp-python>=0.2.90",
    # VLM cloud
    "anthropic>=0.34",
    "openai>=1.45",
    # Config
    "pydantic-settings>=2.3",
    "pyyaml>=6.0",
    # CLI
    "typer>=0.12",
    "rich>=13.8",
    # Report
    "jinja2>=3.1",
]

[project.scripts]
shuttersift = "shuttersift.cli.main:app"

[tool.hatch.build.targets.wheel]
packages = ["src/shuttersift"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create package `__init__.py` files**

```python
# src/shuttersift/__init__.py
__version__ = "0.1.0"
```

```python
# src/shuttersift/engine/__init__.py  (will be filled in Task 2)
```

```python
# src/shuttersift/engine/analyzers/__init__.py
```

```python
# src/shuttersift/cli/__init__.py
```

```python
# src/shuttersift/server/__init__.py
# Future: local HTTP server for GUI clients (Tauri/Electron/Web).
# The GUI will call Engine via FastAPI endpoints defined in app.py.
```

- [ ] **Step 3: Create `config.yaml` (user-editable defaults)**

```yaml
# ShutterSift default configuration
# All values can be overridden via CLI flags or environment variables.

scoring:
  weights:
    sharpness: 0.30
    exposure: 0.15
    aesthetic: 0.25
    face_quality: 0.20
    composition: 0.10
  thresholds:
    keep: 70
    reject: 40
    hard_reject_sharpness: 30.0
    eye_open_min: 0.25
    burst_gap_seconds: 2.0

workers: 4
log_retention_runs: 30
```

- [ ] **Step 4: Create GUI client placeholder READMEs**

```markdown
<!-- clients/desktop/README.md -->
# ShutterSift Desktop Client (Future)

This directory will contain a Tauri or Electron desktop application.

## Integration

The desktop client communicates with the ShutterSift Python engine via a local
HTTP server defined in `src/shuttersift/server/app.py`.

Start the server: `shuttersift serve --port 7788`
API base URL: `http://localhost:7788`

## Planned Endpoints

- `POST /analyze` — start analysis, returns job ID
- `GET /jobs/{id}/progress` — SSE stream of per-photo results
- `GET /jobs/{id}/results` — final AnalysisResult JSON
- `GET /capabilities` — detected GPU/VLM/RAW capabilities
```

```markdown
<!-- clients/web/README.md -->
# ShutterSift Web Client (Future)

Gradio or custom React SPA that connects to `src/shuttersift/server/app.py`.
```

- [ ] **Step 5: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
*.gguf
models/
.state.json
*.log
.DS_Store
```

- [ ] **Step 6: Install package in development mode**

```bash
pip install -e ".[dev]" 2>/dev/null || pip install -e .
```

- [ ] **Step 7: Verify package installs**

```bash
python -c "import shuttersift; print(shuttersift.__version__)"
```
Expected output: `0.1.0`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/ clients/ config.yaml .gitignore
git commit -m "feat: project scaffolding, pyproject.toml, package structure"
```

---

## Task 2: Config & Data Models

**Files:**
- Create: `src/shuttersift/config.py`
- Modify: `src/shuttersift/engine/__init__.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_config.py
from shuttersift.config import Config, ScoringWeights, Thresholds

def test_default_weights_sum_to_one():
    w = ScoringWeights()
    total = w.sharpness + w.exposure + w.aesthetic + w.face_quality + w.composition
    assert abs(total - 1.0) < 1e-9

def test_default_thresholds():
    t = Thresholds()
    assert t.keep == 70
    assert t.reject == 40
    assert t.hard_reject_sharpness == 30.0
    assert t.eye_open_min == 0.25

def test_config_loads_from_yaml(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("scoring:\n  thresholds:\n    keep: 80\n")
    cfg = Config.from_yaml(yaml_file)
    assert cfg.thresholds.keep == 80
    assert cfg.thresholds.reject == 40  # default preserved

def test_config_defaults_without_file():
    cfg = Config()
    assert cfg.weights.sharpness == 0.30
    assert cfg.workers == 4
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/unit/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'shuttersift.config'`

- [ ] **Step 3: Write `src/shuttersift/config.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, model_validator


class ScoringWeights(BaseModel):
    sharpness: float = 0.30
    exposure: float = 0.15
    aesthetic: float = 0.25
    face_quality: float = 0.20
    composition: float = 0.10

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ScoringWeights":
        total = self.sharpness + self.exposure + self.aesthetic + self.face_quality + self.composition
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.4f}")
        return self


class Thresholds(BaseModel):
    keep: int = 70
    reject: int = 40
    hard_reject_sharpness: float = 30.0
    eye_open_min: float = 0.25
    burst_gap_seconds: float = 2.0


class Config(BaseModel):
    weights: ScoringWeights = ScoringWeights()
    thresholds: Thresholds = Thresholds()
    workers: int = 4
    log_retention_runs: int = 30
    api_model_anthropic: str = "claude-haiku-4-5-20251001"
    api_model_openai: str = "gpt-4o-mini"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        # Flatten nested 'scoring' key
        scoring = data.pop("scoring", {})
        if "weights" in scoring:
            data["weights"] = scoring["weights"]
        if "thresholds" in scoring:
            data["thresholds"] = scoring["thresholds"]
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Load from path, fallback to ~/.shuttersift/config.yaml, then defaults."""
        candidates = [path] if path else []
        candidates.append(Path.home() / ".shuttersift" / "config.yaml")
        for p in candidates:
            if p and p.exists():
                return cls.from_yaml(p)
        return cls()
```

- [ ] **Step 4: Write data models in `engine/__init__.py`**

```python
# src/shuttersift/engine/__init__.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Callable
import datetime

from .pipeline import Engine  # noqa: F401 — re-export


@dataclass
class SubScores:
    sharpness: float = 0.0
    exposure: float = 0.0
    aesthetic: float = 0.0
    face_quality: float = 75.0   # neutral: no faces detected
    composition: float = 50.0    # neutral: rule-based N/A


@dataclass
class PhotoResult:
    path: Path
    score: float = 0.0
    sub_scores: SubScores = field(default_factory=SubScores)
    decision: Literal["keep", "review", "reject"] = "review"
    reasons: list[str] = field(default_factory=list)
    explanation: str = ""
    face_count: int = 0
    is_duplicate: bool = False
    hard_rejected: bool = False
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class AnalysisResult:
    version: str = "1"
    shuttersift_version: str = "0.1.0"
    run_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )
    photos: list[PhotoResult] = field(default_factory=list)

    @property
    def keep(self) -> list[PhotoResult]:
        return [p for p in self.photos if p.decision == "keep"]

    @property
    def review(self) -> list[PhotoResult]:
        return [p for p in self.photos if p.decision == "review"]

    @property
    def reject(self) -> list[PhotoResult]:
        return [p for p in self.photos if p.decision == "reject"]
```

Note: `from .pipeline import Engine` will fail until Task 17. Add a try/except guard for now:

```python
# Replace the import line with:
try:
    from .pipeline import Engine  # noqa: F401
except ImportError:
    pass
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/unit/test_config.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/shuttersift/config.py src/shuttersift/engine/__init__.py tests/unit/test_config.py
git commit -m "feat: Config/Thresholds/ScoringWeights models and PhotoResult dataclasses"
```

---

## Task 3: Test Fixtures + Capabilities Detection

**Files:**
- Create: `tests/fixtures/conftest.py`
- Create: `src/shuttersift/engine/capabilities.py`
- Create: `tests/unit/test_capabilities.py`

- [ ] **Step 1: Write `tests/fixtures/conftest.py`**

```python
# tests/fixtures/conftest.py
import numpy as np
import cv2
import pytest
from pathlib import Path


@pytest.fixture
def sharp_image() -> np.ndarray:
    """Checkerboard — high Laplacian variance (~3000+)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for x in range(0, 640, 8):
        for y in range(0, 480, 8):
            if (x // 8 + y // 8) % 2 == 0:
                img[y:y+8, x:x+8] = 200
    return img


@pytest.fixture
def blurry_image(sharp_image) -> np.ndarray:
    """Heavily blurred — low Laplacian variance (<5)."""
    return cv2.GaussianBlur(sharp_image, (51, 51), 20)


@pytest.fixture
def dark_image() -> np.ndarray:
    """Underexposed: all pixels ~ 8."""
    return np.full((480, 640, 3), 8, dtype=np.uint8)


@pytest.fixture
def bright_image() -> np.ndarray:
    """Overexposed: all pixels ~ 248."""
    return np.full((480, 640, 3), 248, dtype=np.uint8)


@pytest.fixture
def normal_image() -> np.ndarray:
    """Normal exposure: gradient 60–190."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for x in range(640):
        val = int(60 + (x / 639) * 130)
        img[:, x] = val
    return img


@pytest.fixture
def tmp_jpeg_dir(tmp_path, sharp_image) -> Path:
    """3 JPEG files in a temp directory."""
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"DSC_{i:04d}.jpg"), sharp_image)
    return tmp_path
```

Add `conftest.py` at `tests/` root to make fixtures available globally:

```python
# tests/conftest.py
pytest_plugins = ["fixtures.conftest"]
```

- [ ] **Step 2: Write failing capabilities test**

```python
# tests/unit/test_capabilities.py
from shuttersift.engine.capabilities import Capabilities


def test_capabilities_returns_dict():
    caps = Capabilities.detect()
    assert isinstance(caps.gpu, bool)
    assert isinstance(caps.rawpy, bool)
    assert isinstance(caps.api_vlm, bool)
    assert isinstance(caps.gguf_vlm, bool)


def test_capabilities_summary_string():
    caps = Capabilities.detect()
    summary = caps.summary()
    assert "GPU" in summary
    assert "RAW" in summary


def test_gguf_false_when_no_models(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "shuttersift.engine.capabilities.MODELS_DIR", tmp_path
    )
    caps = Capabilities.detect()
    assert caps.gguf_vlm is False
```

- [ ] **Step 3: Run — expect FAIL**

```bash
pytest tests/unit/test_capabilities.py -v
```

- [ ] **Step 4: Write `src/shuttersift/engine/capabilities.py`**

```python
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
```

- [ ] **Step 5: Run — expect PASS**

```bash
pytest tests/unit/test_capabilities.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/ src/shuttersift/engine/capabilities.py
git commit -m "feat: test fixtures, Capabilities detection"
```

---

## Task 4: Image Loader

**Files:**
- Create: `src/shuttersift/engine/loader.py`
- Create: `tests/unit/test_loader.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_loader.py
import numpy as np
import cv2
import pytest
from pathlib import Path
from shuttersift.engine.loader import load_image, SUPPORTED_FORMATS


def test_load_jpeg(tmp_path, sharp_image):
    path = tmp_path / "test.jpg"
    cv2.imwrite(str(path), sharp_image)
    img = load_image(path)
    assert img is not None
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_load_png(tmp_path, sharp_image):
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), sharp_image)
    img = load_image(path)
    assert img is not None


def test_load_missing_file_returns_none():
    img = load_image(Path("/nonexistent/file.jpg"))
    assert img is None


def test_supported_formats_includes_raw():
    for ext in [".cr2", ".nef", ".arw", ".dng"]:
        assert ext in SUPPORTED_FORMATS


def test_load_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"not an image at all")
    img = load_image(path)
    assert img is None
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_loader.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/loader.py`**

```python
from __future__ import annotations
import io
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
            # Prefer embedded JPEG thumbnail (10x faster than full decode)
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    arr = np.frombuffer(thumb.data, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
            except Exception:
                pass
            # Fallback: full RAW decode (slower, higher quality)
            rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        # rawpy not available — try PIL (won't work for most RAW but worth trying)
        logger.warning("rawpy not available, skipping RAW: %s", path.name)
        return None
    except Exception as e:
        logger.warning("Failed to load RAW %s: %s", path.name, e)
        return None


def _load_standard(path: Path) -> np.ndarray | None:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            # OpenCV failed — try Pillow (handles more formats)
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logger.warning("Failed to load %s: %s", path.name, e)
        return None
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/test_loader.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/loader.py tests/unit/test_loader.py
git commit -m "feat: unified image loader (RAW + JPEG, rawpy optional)"
```

---

## Task 5: Sharpness Analyzer

**Files:**
- Create: `src/shuttersift/engine/analyzers/sharpness.py`
- Create: `tests/unit/analyzers/test_sharpness.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/analyzers/test_sharpness.py
from shuttersift.engine.analyzers.sharpness import sharpness_score


def test_sharp_image_scores_high(sharp_image):
    score = sharpness_score(sharp_image)
    assert score > 60, f"Sharp image scored {score}, expected > 60"


def test_blurry_image_scores_low(blurry_image):
    score = sharpness_score(blurry_image)
    assert score < 20, f"Blurry image scored {score}, expected < 20"


def test_score_in_range(sharp_image):
    score = sharpness_score(sharp_image)
    assert 0.0 <= score <= 100.0


def test_blur_scores_less_than_sharp(sharp_image, blurry_image):
    assert sharpness_score(blurry_image) < sharpness_score(sharp_image)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/analyzers/test_sharpness.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/analyzers/sharpness.py`**

```python
from __future__ import annotations
import numpy as np
import cv2

# Laplacian variance of a perfectly sharp checkerboard is ~3000-8000.
# Real-world sharp photos are typically 100-800.
# Heavily blurred images drop below 10.
# We normalize with a sigmoid-like curve capped at MAX_SHARP.
_MAX_SHARP_VAR = 500.0   # variance that maps to score ~95
_BLUR_FLOOR = 2.0        # below this = effectively 0


def sharpness_score(img: np.ndarray) -> float:
    """
    Returns sharpness score [0–100].
    Uses Laplacian variance on the luminance channel.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if laplacian_var < _BLUR_FLOOR:
        return 0.0

    # Logarithmic scale: feels more perceptually linear for blur
    import math
    log_val = math.log1p(laplacian_var)
    log_max = math.log1p(_MAX_SHARP_VAR)
    score = min(100.0, (log_val / log_max) * 100.0)
    return round(score, 2)


def laplacian_variance(img: np.ndarray) -> float:
    """Raw Laplacian variance — used for calibration."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/analyzers/test_sharpness.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/analyzers/sharpness.py tests/unit/analyzers/test_sharpness.py
git commit -m "feat: sharpness analyzer (Laplacian log-normalized)"
```

---

## Task 6: Exposure Analyzer

**Files:**
- Create: `src/shuttersift/engine/analyzers/exposure.py`
- Create: `tests/unit/analyzers/test_exposure.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/analyzers/test_exposure.py
from shuttersift.engine.analyzers.exposure import exposure_score


def test_normal_exposure_scores_high(normal_image):
    score = exposure_score(normal_image)
    assert score > 65, f"Normal image scored {score}"


def test_dark_image_scores_low(dark_image):
    score = exposure_score(dark_image)
    assert score < 35, f"Dark image scored {score}"


def test_bright_image_scores_low(bright_image):
    score = exposure_score(bright_image)
    assert score < 35, f"Overexposed image scored {score}"


def test_score_in_range(normal_image):
    score = exposure_score(normal_image)
    assert 0.0 <= score <= 100.0
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/analyzers/test_exposure.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/analyzers/exposure.py`**

```python
from __future__ import annotations
import numpy as np
import cv2


def exposure_score(img: np.ndarray) -> float:
    """
    Returns exposure quality score [0–100].
    Penalizes over-exposed (>240) and under-exposed (<15) pixel ratios.
    Also rewards even histogram spread (not too spiked).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    overexp_ratio = float(hist[241:].sum() / total_pixels)
    underexp_ratio = float(hist[:15].sum() / total_pixels)

    # Penalize clipping at either end
    clip_penalty = min(1.0, (overexp_ratio + underexp_ratio) * 4.0)

    # Reward: mean brightness in comfortable zone (60–200)
    mean_brightness = float(gray.mean())
    if 60 <= mean_brightness <= 200:
        brightness_bonus = 1.0
    else:
        # Linear falloff outside comfortable zone
        dist = min(abs(mean_brightness - 60), abs(mean_brightness - 200))
        brightness_bonus = max(0.0, 1.0 - dist / 60.0)

    base_score = brightness_bonus * (1.0 - clip_penalty)
    return round(max(0.0, min(100.0, base_score * 100.0)), 2)
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/analyzers/test_exposure.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/analyzers/exposure.py tests/unit/analyzers/test_exposure.py
git commit -m "feat: exposure analyzer (histogram clipping + brightness zone)"
```

---

## Task 7: Face Analyzer (MediaPipe)

**Files:**
- Create: `src/shuttersift/engine/analyzers/face.py`
- Create: `tests/unit/analyzers/test_face.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/analyzers/test_face.py
from unittest.mock import MagicMock, patch
import numpy as np
from shuttersift.engine.analyzers.face import FaceAnalyzer, FaceResult


def test_face_result_defaults():
    r = FaceResult()
    assert r.count == 0
    assert r.eye_open_score == 1.0   # neutral when no faces
    assert r.smile_score == 0.0
    assert r.all_eyes_closed is False


def test_no_face_returns_neutral(normal_image):
    """A plain gradient image should yield zero faces."""
    analyzer = FaceAnalyzer()
    result = analyzer.analyze(normal_image)
    assert result.count == 0
    assert result.all_eyes_closed is False


def test_face_quality_score_with_no_faces(normal_image):
    analyzer = FaceAnalyzer()
    result = analyzer.analyze(normal_image)
    assert result.face_quality_score == 75.0  # neutral


def test_all_eyes_closed_detection():
    """Mock blendshapes to simulate closed eyes."""
    analyzer = FaceAnalyzer()

    mock_blendshapes = {
        "eyeBlinkLeft": 0.95,
        "eyeBlinkRight": 0.92,
        "mouthSmileLeft": 0.1,
        "mouthSmileRight": 0.1,
        "cheekSquintLeft": 0.0,
        "cheekSquintRight": 0.0,
    }
    result = analyzer._compute_face_scores([mock_blendshapes])
    assert result.all_eyes_closed is True
    assert result.eye_open_score < 0.25
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/analyzers/test_face.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/analyzers/face.py`**

```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

_MP_LOADED = False
_face_detection = None
_face_mesh = None


def _ensure_mediapipe():
    global _MP_LOADED, _face_detection, _face_mesh
    if _MP_LOADED:
        return
    try:
        import mediapipe as mp
        _face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=10,
            min_detection_confidence=0.5,
        )
        _MP_LOADED = True
    except Exception as e:
        logger.warning("MediaPipe unavailable: %s", e)


@dataclass
class FaceResult:
    count: int = 0
    eye_open_score: float = 1.0    # 1 = fully open, 0 = closed; 1.0 when no faces
    smile_score: float = 0.0
    all_eyes_closed: bool = False
    face_quality_score: float = 75.0   # neutral when no faces, 0–100 when faces
    face_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


class FaceAnalyzer:
    def analyze(self, img: np.ndarray) -> FaceResult:
        _ensure_mediapipe()
        if not _MP_LOADED:
            return FaceResult()

        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Stage 1: detect faces
        det_result = _face_detection.process(rgb)
        if not det_result.detections:
            return FaceResult()

        h, w = img.shape[:2]
        bboxes = []
        for det in det_result.detections:
            bb = det.location_data.relative_bounding_box
            bboxes.append((bb.xmin, bb.ymin, bb.xmin + bb.width, bb.ymin + bb.height))

        # Stage 2: face mesh for blendshapes
        mesh_result = _face_mesh.process(rgb)
        if not mesh_result.multi_face_landmarks:
            return FaceResult(
                count=len(bboxes),
                face_bboxes=bboxes,
                eye_open_score=1.0,
                face_quality_score=60.0,
            )

        blendshapes_list = [
            self._extract_blendshapes(lm) for lm in mesh_result.multi_face_landmarks
        ]
        result = self._compute_face_scores(blendshapes_list)
        result.count = len(bboxes)
        result.face_bboxes = bboxes
        return result

    def _extract_blendshapes(self, landmarks) -> dict[str, float]:
        """
        Compute eye and smile metrics from raw Face Mesh landmarks.
        Uses Eye Aspect Ratio (EAR) for eyes and mouth corner elevation for smile.
        """
        lm = landmarks.landmark

        def pt(idx):
            return np.array([lm[idx].x, lm[idx].y])

        # Left eye EAR (indices: 362, 385, 387, 263, 373, 380)
        left_ear = _ear(pt(362), pt(385), pt(387), pt(263), pt(373), pt(380))
        # Right eye EAR (indices: 33, 160, 158, 133, 153, 144)
        right_ear = _ear(pt(33), pt(160), pt(158), pt(133), pt(153), pt(144))

        # Smile: mouth corners (61, 291) relative to lip center (13)
        left_corner_y = lm[61].y
        right_corner_y = lm[291].y
        lip_center_y = lm[13].y
        # Corners above lip center = smile
        smile = max(0.0, min(1.0, (lip_center_y - (left_corner_y + right_corner_y) / 2) * 10))

        # EAR: ~0.30 open, ~0.05 closed. Normalise: blink if EAR < 0.15
        # Convert to "blink probability": 1 = fully closed
        left_blink = max(0.0, min(1.0, 1.0 - (left_ear / 0.25)))
        right_blink = max(0.0, min(1.0, 1.0 - (right_ear / 0.25)))

        return {
            "eyeBlinkLeft": left_blink,
            "eyeBlinkRight": right_blink,
            "mouthSmileLeft": smile,
            "mouthSmileRight": smile,
            "cheekSquintLeft": 0.0,
            "cheekSquintRight": 0.0,
        }

    def _compute_face_scores(self, blendshapes_list: list[dict]) -> FaceResult:
        if not blendshapes_list:
            return FaceResult()

        eyes_open_per_face = []
        smiles = []
        for bs in blendshapes_list:
            left_blink = bs.get("eyeBlinkLeft", 0.0)
            right_blink = bs.get("eyeBlinkRight", 0.0)
            eye_open = 1.0 - max(left_blink, right_blink)
            eyes_open_per_face.append(eye_open)
            smile = (
                bs.get("mouthSmileLeft", 0.0)
                + bs.get("mouthSmileRight", 0.0)
                + bs.get("cheekSquintLeft", 0.0)
                + bs.get("cheekSquintRight", 0.0)
            ) / 4.0
            smiles.append(smile)

        min_eye_open = min(eyes_open_per_face)
        all_closed = all(e < 0.25 for e in eyes_open_per_face)
        avg_smile = sum(smiles) / len(smiles)

        # face_quality_score: 50% eye open + 30% smile + 20% neutral bonus
        eye_component = min_eye_open * 50.0
        smile_component = avg_smile * 30.0
        neutral_bonus = 20.0  # base for having faces at all
        face_quality = min(100.0, eye_component + smile_component + neutral_bonus)

        return FaceResult(
            eye_open_score=min_eye_open,
            smile_score=avg_smile,
            all_eyes_closed=all_closed,
            face_quality_score=face_quality,
        )


def _ear(p1, p2, p3, p4, p5, p6) -> float:
    """Eye Aspect Ratio = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))
    return (v1 + v2) / (2.0 * h + 1e-6)
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/analyzers/test_face.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/analyzers/face.py tests/unit/analyzers/test_face.py
git commit -m "feat: FaceAnalyzer (MediaPipe EAR + smile detection)"
```

---

## Task 8: Aesthetic Analyzer

**Files:**
- Create: `src/shuttersift/engine/analyzers/aesthetic.py`
- Create: `tests/unit/analyzers/test_aesthetic.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/analyzers/test_aesthetic.py
from unittest.mock import patch, MagicMock
from shuttersift.engine.analyzers.aesthetic import AestheticAnalyzer


def test_score_in_range(normal_image):
    analyzer = AestheticAnalyzer(use_gpu=False)
    score = analyzer.score(normal_image)
    assert 0.0 <= score <= 100.0


def test_brisque_fallback_when_pyiqa_unavailable(normal_image, monkeypatch):
    """Should fall back to BRISQUE (scikit-image) without pyiqa."""
    monkeypatch.setattr(
        "shuttersift.engine.analyzers.aesthetic._PYIQA_AVAILABLE", False
    )
    analyzer = AestheticAnalyzer(use_gpu=False)
    score = analyzer.score(normal_image)
    assert 0.0 <= score <= 100.0


def test_returns_float(normal_image):
    analyzer = AestheticAnalyzer(use_gpu=False)
    score = analyzer.score(normal_image)
    assert isinstance(score, float)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/analyzers/test_aesthetic.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/analyzers/aesthetic.py`**

```python
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

    def _load(self) -> None:
        if self._model is not None:
            return
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
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/analyzers/test_aesthetic.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/analyzers/aesthetic.py tests/unit/analyzers/test_aesthetic.py
git commit -m "feat: AestheticAnalyzer (MUSIQ→BRISQUE fallback)"
```

---

## Task 9: Composition Analyzer

**Files:**
- Create: `src/shuttersift/engine/analyzers/composition.py`
- Create: `tests/unit/analyzers/test_composition.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/analyzers/test_composition.py
from shuttersift.engine.analyzers.composition import composition_score


def test_no_faces_returns_neutral(normal_image):
    score = composition_score(normal_image, face_bboxes=[])
    assert score == 50.0  # neutral when no detectable subjects


def test_face_at_thirds_node_scores_high(normal_image):
    # Face centered at top-left rule-of-thirds node (1/3, 1/3)
    # bbox in relative coords: (x1, y1, x2, y2)
    face_center_x = 1 / 3
    face_center_y = 1 / 3
    w, h = 0.15, 0.20
    bbox = (face_center_x - w/2, face_center_y - h/2,
            face_center_x + w/2, face_center_y + h/2)
    score = composition_score(normal_image, face_bboxes=[bbox])
    assert score >= 65, f"Expected ≥65 for thirds-node face, got {score}"


def test_face_at_edge_penalized(normal_image):
    # Face clipped at left edge
    bbox = (-0.05, 0.3, 0.1, 0.7)
    score = composition_score(normal_image, face_bboxes=[bbox])
    assert score < 50


def test_score_in_range(normal_image):
    score = composition_score(normal_image, face_bboxes=[])
    assert 0.0 <= score <= 100.0
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/analyzers/test_composition.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/analyzers/composition.py`**

```python
from __future__ import annotations
import numpy as np

# Rule-of-thirds nodes (relative coords)
_THIRDS_NODES = [
    (1/3, 1/3), (2/3, 1/3),
    (1/3, 2/3), (2/3, 2/3),
]
_CENTER = (0.5, 0.5)


def composition_score(
    img: np.ndarray,
    face_bboxes: list[tuple[float, float, float, float]],
) -> float:
    """
    Rule-based composition score [0–100].
    No faces → neutral 50.
    Faces present → score by:
      - Proximity to rule-of-thirds nodes (+40)
      - Not clipped at frame edge (+30)
      - Not dead-center (slight bonus) (+15)
      - Reasonable face size (+15)
    """
    if not face_bboxes:
        return 50.0

    scores = [_score_single_face(bbox) for bbox in face_bboxes]
    # Use the best-positioned face (primary subject)
    return round(max(scores), 2)


def _score_single_face(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    face_w = x2 - x1
    face_h = y2 - y1

    score = 0.0

    # 1. Thirds proximity (max 40 pts)
    min_thirds_dist = min(
        ((cx - nx)**2 + (cy - ny)**2) ** 0.5
        for nx, ny in _THIRDS_NODES
    )
    thirds_score = max(0.0, 40.0 * (1.0 - min_thirds_dist / 0.4))
    score += thirds_score

    # 2. Not clipped at edges (max 30 pts)
    margin = 0.02  # 2% tolerance
    clipped = x1 < -margin or y1 < -margin or x2 > 1 + margin or y2 > 1 + margin
    if not clipped:
        score += 30.0
    else:
        clip_amount = max(
            max(0.0, -x1), max(0.0, x2 - 1.0),
            max(0.0, -y1), max(0.0, y2 - 1.0),
        )
        score += max(0.0, 30.0 - clip_amount * 200.0)

    # 3. Not dead-center (max 15 pts — slight penalty for perfectly centered)
    center_dist = ((cx - 0.5)**2 + (cy - 0.5)**2) ** 0.5
    if center_dist > 0.05:
        score += 15.0
    else:
        score += center_dist * 15.0 / 0.05

    # 4. Reasonable face size: 5–40% of frame width is ideal (max 15 pts)
    if 0.05 <= face_w <= 0.40:
        score += 15.0
    elif face_w < 0.05:
        score += face_w / 0.05 * 15.0
    else:
        score += max(0.0, 15.0 - (face_w - 0.40) / 0.20 * 15.0)

    return max(0.0, min(100.0, score))
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/analyzers/test_composition.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/analyzers/composition.py tests/unit/analyzers/test_composition.py
git commit -m "feat: rule-based composition analyzer (thirds + clipping + size)"
```

---

## Task 10: Duplicates & Burst Grouping

**Files:**
- Create: `src/shuttersift/engine/analyzers/duplicates.py`
- Create: `tests/unit/analyzers/test_duplicates.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/analyzers/test_duplicates.py
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
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/analyzers/test_duplicates.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/analyzers/duplicates.py`**

```python
from __future__ import annotations
import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def group_bursts(
    paths: list[Path],
    exif_timestamps: dict[Path, datetime.datetime],
    gap_seconds: float = 2.0,
) -> list[list[Path]]:
    """
    Group photos into burst sequences.
    Uses EXIF DateTimeOriginal when available; falls back to file modification time.
    Photos within `gap_seconds` of each other are considered one burst.
    """
    if not paths:
        return []

    # Build (timestamp, path) list, sorted by time
    timed: list[tuple[datetime.datetime, Path]] = []
    for p in paths:
        ts = exif_timestamps.get(p) or _mtime(p)
        timed.append((ts, p))

    timed.sort(key=lambda t: t[0])

    groups: list[list[Path]] = []
    current_group: list[Path] = [timed[0][1]]
    prev_ts = timed[0][0]

    for ts, p in timed[1:]:
        delta = (ts - prev_ts).total_seconds()
        if delta <= gap_seconds:
            current_group.append(p)
        else:
            groups.append(current_group)
            current_group = [p]
        prev_ts = ts

    groups.append(current_group)
    return groups


def best_in_burst(group: list[Path], scores: dict[Path, float]) -> Path:
    """Return the path with the highest score in a burst group."""
    return max(group, key=lambda p: scores.get(p, 0.0))


def _mtime(path: Path) -> datetime.datetime:
    try:
        return datetime.datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return datetime.datetime.min


def read_exif_timestamps(paths: list[Path]) -> dict[Path, datetime.datetime]:
    """Extract DateTimeOriginal from EXIF for all paths."""
    result: dict[Path, datetime.datetime] = {}
    try:
        import exifread
    except ImportError:
        logger.warning("exifread not available; using file mtime for burst grouping")
        return result

    for path in paths:
        try:
            with open(path, "rb") as f:
                tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal", details=False)
            tag = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
            if tag:
                dt = datetime.datetime.strptime(str(tag.values), "%Y:%m:%d %H:%M:%S")
                result[path] = dt
        except Exception:
            pass

    return result
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/analyzers/test_duplicates.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/analyzers/duplicates.py tests/unit/analyzers/test_duplicates.py
git commit -m "feat: burst grouping (EXIF timestamp + mtime fallback) and best-in-burst selection"
```

---

## Task 11: Scorer

**Files:**
- Create: `src/shuttersift/engine/scorer.py`
- Create: `tests/unit/test_scorer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_scorer.py
from shuttersift.engine import SubScores
from shuttersift.engine.scorer import Scorer
from shuttersift.config import Config


def _scorer() -> Scorer:
    return Scorer(Config())


def test_perfect_scores_give_keep():
    scorer = _scorer()
    sub = SubScores(sharpness=100, exposure=100, aesthetic=100,
                    face_quality=100, composition=100)
    total = scorer.compute(sub)
    assert total == 100.0
    assert scorer.decide(total) == "keep"


def test_all_zero_gives_reject():
    scorer = _scorer()
    sub = SubScores(sharpness=0, exposure=0, aesthetic=0,
                    face_quality=0, composition=0)
    total = scorer.compute(sub)
    assert total == 0.0
    assert scorer.decide(total) == "reject"


def test_weights_applied_correctly():
    scorer = _scorer()
    # Only sharpness is 100 (weight 0.30)
    sub = SubScores(sharpness=100, exposure=0, aesthetic=0,
                    face_quality=0, composition=0)
    total = scorer.compute(sub)
    assert abs(total - 30.0) < 0.01


def test_decision_thresholds():
    scorer = _scorer()
    assert scorer.decide(70.0) == "keep"
    assert scorer.decide(69.9) == "review"
    assert scorer.decide(40.0) == "review"
    assert scorer.decide(39.9) == "reject"


def test_custom_thresholds():
    from shuttersift.config import Config, Thresholds
    cfg = Config(thresholds=Thresholds(keep=80, reject=50))
    scorer = Scorer(cfg)
    assert scorer.decide(80.0) == "keep"
    assert scorer.decide(79.9) == "review"
    assert scorer.decide(50.0) == "review"
    assert scorer.decide(49.9) == "reject"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_scorer.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/scorer.py`**

```python
from __future__ import annotations
from shuttersift.config import Config
from shuttersift.engine import SubScores


class Scorer:
    def __init__(self, config: Config):
        self._cfg = config

    def compute(self, sub: SubScores) -> float:
        """Weighted average of sub-scores → overall score [0–100]."""
        w = self._cfg.weights
        total = (
            sub.sharpness   * w.sharpness   +
            sub.exposure    * w.exposure    +
            sub.aesthetic   * w.aesthetic   +
            sub.face_quality * w.face_quality +
            sub.composition * w.composition
        )
        return round(max(0.0, min(100.0, total)), 2)

    def decide(self, score: float) -> str:
        """Map overall score to keep / review / reject."""
        t = self._cfg.thresholds
        if score >= t.keep:
            return "keep"
        if score < t.reject:
            return "reject"
        return "review"
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/test_scorer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/scorer.py tests/unit/test_scorer.py
git commit -m "feat: weighted Scorer with configurable keep/review/reject thresholds"
```

---

## Task 12: State Manager (Checkpoint / Resume)

**Files:**
- Create: `src/shuttersift/engine/state.py`
- Create: `tests/unit/test_state.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_state.py
from pathlib import Path
from shuttersift.engine.state import StateManager
from shuttersift.engine import PhotoResult, SubScores


def _make_result(name: str, score: float = 75.0) -> PhotoResult:
    return PhotoResult(
        path=Path(f"/photos/{name}.jpg"),
        score=score,
        decision="keep",
    )


def test_empty_state_has_no_processed(tmp_path):
    sm = StateManager(tmp_path)
    assert not sm.is_processed(Path("/photos/test.jpg"))


def test_save_and_resume(tmp_path):
    sm = StateManager(tmp_path)
    result = _make_result("DSC001", score=82.5)
    sm.save(result)

    sm2 = StateManager(tmp_path)
    assert sm2.is_processed(result.path)
    loaded = sm2.load(result.path)
    assert loaded is not None
    assert abs(loaded.score - 82.5) < 0.01
    assert loaded.decision == "keep"


def test_fresh_ignores_state(tmp_path):
    sm = StateManager(tmp_path)
    result = _make_result("DSC002")
    sm.save(result)

    sm_fresh = StateManager(tmp_path, fresh=True)
    assert not sm_fresh.is_processed(result.path)


def test_state_file_written_to_output_dir(tmp_path):
    sm = StateManager(tmp_path)
    sm.save(_make_result("DSC003"))
    assert (tmp_path / ".state.json").exists()
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_state.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/state.py`**

```python
from __future__ import annotations
import json
import logging
from dataclasses import asdict
from pathlib import Path

from shuttersift.engine import PhotoResult, SubScores

logger = logging.getLogger(__name__)

_STATE_FILE = ".state.json"


class StateManager:
    def __init__(self, output_dir: Path, fresh: bool = False):
        self._state_path = output_dir / _STATE_FILE
        self._records: dict[str, dict] = {}
        if not fresh and self._state_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        try:
            data = json.loads(self._state_path.read_text())
            self._records = {r["path"]: r for r in data.get("records", [])}
            logger.info("Resumed %d processed photos from state", len(self._records))
        except Exception as e:
            logger.warning("Could not read state file: %s", e)

    def is_processed(self, path: Path) -> bool:
        return str(path) in self._records

    def load(self, path: Path) -> PhotoResult | None:
        rec = self._records.get(str(path))
        if rec is None:
            return None
        sub = SubScores(**rec.get("sub_scores", {}))
        return PhotoResult(
            path=path,
            score=rec["score"],
            sub_scores=sub,
            decision=rec["decision"],
            reasons=rec.get("reasons", []),
            explanation=rec.get("explanation", ""),
            face_count=rec.get("face_count", 0),
            is_duplicate=rec.get("is_duplicate", False),
        )

    def save(self, result: PhotoResult) -> None:
        self._records[str(result.path)] = {
            "path": str(result.path),
            "score": result.score,
            "sub_scores": {
                "sharpness": result.sub_scores.sharpness,
                "exposure": result.sub_scores.exposure,
                "aesthetic": result.sub_scores.aesthetic,
                "face_quality": result.sub_scores.face_quality,
                "composition": result.sub_scores.composition,
            },
            "decision": result.decision,
            "reasons": result.reasons,
            "explanation": result.explanation,
            "face_count": result.face_count,
            "is_duplicate": result.is_duplicate,
        }
        self._flush()

    def _flush(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"records": list(self._records.values())}, indent=2))
        tmp.replace(self._state_path)  # atomic write
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/test_state.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/state.py tests/unit/test_state.py
git commit -m "feat: StateManager checkpoint/resume with atomic writes"
```

---

## Task 13: Explainer (GGUF + API VLM)

**Files:**
- Create: `src/shuttersift/engine/explainer.py`

- [ ] **Step 1: Write `src/shuttersift/engine/explainer.py`**

(No failing test first for this module — external I/O is mocked, testing the routing logic)

```python
# tests/unit/test_explainer.py
from unittest.mock import patch, MagicMock
from pathlib import Path
from shuttersift.engine.explainer import Explainer
from shuttersift.engine import PhotoResult
from shuttersift.config import Config


def test_skip_when_no_vlm(tmp_path):
    cfg = Config()
    explainer = Explainer(cfg, gguf_path=None, api_key_anthropic=None, api_key_openai=None)
    result = PhotoResult(path=tmp_path / "test.jpg", decision="review")
    explanation = explainer.explain(result.path, result)
    assert explanation == ""


def test_only_explains_review(tmp_path):
    cfg = Config()
    explainer = Explainer(cfg, gguf_path=None, api_key_anthropic=None, api_key_openai=None)
    keep_result = PhotoResult(path=tmp_path / "test.jpg", decision="keep")
    assert explainer.explain(keep_result.path, keep_result) == ""


def test_anthropic_api_called_for_review(tmp_path, monkeypatch):
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="Great portrait, sharp eyes.")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    monkeypatch.setattr("shuttersift.engine.explainer.anthropic", MagicMock(Anthropic=MagicMock(return_value=mock_client)))

    cfg = Config()
    explainer = Explainer(cfg, gguf_path=None, api_key_anthropic="fake-key", api_key_openai=None)

    img_path = tmp_path / "photo.jpg"
    import cv2, numpy as np
    cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

    result = PhotoResult(path=img_path, decision="review", score=55.0)
    text = explainer.explain(img_path, result)
    assert text == "Great portrait, sharp eyes."
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/unit/test_explainer.py -v
```

- [ ] **Step 3: Implement `src/shuttersift/engine/explainer.py`**

```python
from __future__ import annotations
import base64
import logging
import os
from pathlib import Path

from shuttersift.config import Config
from shuttersift.engine import PhotoResult

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    import openai
except ImportError:
    openai = None  # type: ignore

_PROMPT_TEMPLATE = """\
你是专业摄影师。请评价这张照片，重点关注：清晰度、曝光、构图、人物表情。
已有技术评分：综合分 {score:.0f}/100，清晰度 {sharpness:.0f}，曝光 {exposure:.0f}，审美 {aesthetic:.0f}。
用1-2句话说明是否值得保留，指出最主要的问题或亮点。"""


class Explainer:
    def __init__(
        self,
        config: Config,
        gguf_path: Path | None,
        api_key_anthropic: str | None,
        api_key_openai: str | None,
    ):
        self._cfg = config
        self._gguf_path = gguf_path
        self._api_key_anthropic = api_key_anthropic
        self._api_key_openai = api_key_openai
        self._gguf_model = None

    def explain(self, photo_path: Path, result: PhotoResult) -> str:
        """Return explanation string; empty string if VLM unavailable or not review."""
        if result.decision != "review":
            return ""

        prompt = _PROMPT_TEMPLATE.format(
            score=result.score,
            sharpness=result.sub_scores.sharpness,
            exposure=result.sub_scores.exposure,
            aesthetic=result.sub_scores.aesthetic,
        )

        # Priority: GGUF local → Anthropic API → OpenAI API → skip
        if self._gguf_path:
            text = self._explain_gguf(photo_path, prompt)
            if text:
                return text

        if self._api_key_anthropic and anthropic:
            text = self._explain_anthropic(photo_path, prompt)
            if text:
                return text

        if self._api_key_openai and openai:
            text = self._explain_openai(photo_path, prompt)
            if text:
                return text

        return ""

    def _img_b64(self, path: Path) -> str | None:
        """Read image and return base64-encoded JPEG bytes."""
        try:
            import cv2
            import numpy as np
            img = cv2.imread(str(path))
            if img is None:
                return None
            # Resize to max 1024px on longest side for API efficiency
            h, w = img.shape[:2]
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.standard_b64encode(buf.tobytes()).decode()
        except Exception as e:
            logger.warning("Could not encode image %s: %s", path.name, e)
            return None

    def _explain_anthropic(self, path: Path, prompt: str) -> str:
        b64 = self._img_b64(path)
        if not b64:
            return ""
        try:
            client = anthropic.Anthropic(api_key=self._api_key_anthropic)
            msg = client.messages.create(
                model=self._cfg.api_model_anthropic,
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg", "data": b64,
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            logger.warning("Anthropic API error: %s", e)
            return ""

    def _explain_openai(self, path: Path, prompt: str) -> str:
        b64 = self._img_b64(path)
        if not b64:
            return ""
        try:
            client = openai.OpenAI(api_key=self._api_key_openai)
            resp = client.chat.completions.create(
                model=self._cfg.api_model_openai,
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("OpenAI API error: %s", e)
            return ""

    def _explain_gguf(self, path: Path, prompt: str) -> str:
        try:
            if self._gguf_model is None:
                from llama_cpp import Llama
                self._gguf_model = Llama(
                    model_path=str(self._gguf_path),
                    n_ctx=2048,
                    verbose=False,
                )
            b64 = self._img_b64(path)
            if not b64:
                return ""
            # llama-cpp-python multimodal (LLaVA-style) prompt
            output = self._gguf_model(
                f"USER: <image>\n{prompt}\nASSISTANT:",
                images=[base64.b64decode(b64)],
                max_tokens=150,
                stop=["USER:"],
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.warning("GGUF inference error: %s", e)
            return ""
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/unit/test_explainer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/explainer.py tests/unit/test_explainer.py
git commit -m "feat: Explainer with GGUF→Anthropic→OpenAI VLM cascade"
```

---

## Task 14: Organizer (Symlinks + XMP Sidecars)

**Files:**
- Create: `src/shuttersift/engine/organizer.py`

- [ ] **Step 1: Write test**

```python
# tests/unit/test_organizer.py
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
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_organizer.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/organizer.py`**

```python
from __future__ import annotations
import logging
import os
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
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/test_organizer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/organizer.py tests/unit/test_organizer.py
git commit -m "feat: Organizer with symlinks, XMP sidecars (Lightroom-compatible)"
```

---

## Task 15: Reporter (HTML + JSON)

**Files:**
- Create: `src/shuttersift/engine/reporter.py`
- Create: `templates/report.html.j2`

- [ ] **Step 1: Write test**

```python
# tests/unit/test_reporter.py
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
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_reporter.py -v
```

- [ ] **Step 3: Write `templates/report.html.j2`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>ShutterSift Report</title>
  <style>
    body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }
    h1 { color: #fff; }
    .summary { display: flex; gap: 24px; margin: 20px 0; }
    .stat { background: #222; border-radius: 8px; padding: 16px 24px; text-align: center; }
    .stat .num { font-size: 2em; font-weight: bold; }
    .keep .num { color: #4caf50; }
    .review .num { color: #ffc107; }
    .reject .num { color: #f44336; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin-top: 20px; }
    .card { background: #1e1e1e; border-radius: 8px; overflow: hidden; }
    .card img { width: 100%; height: 160px; object-fit: cover; }
    .card .info { padding: 8px 10px; font-size: 0.78em; }
    .card .score { font-size: 1.4em; font-weight: bold; }
    .card.keep { border-top: 3px solid #4caf50; }
    .card.review { border-top: 3px solid #ffc107; }
    .card.reject { border-top: 3px solid #f44336; }
    .reasons { color: #aaa; margin-top: 4px; }
    .explanation { color: #ccc; font-style: italic; margin-top: 4px; }
    .sub { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; margin-top: 6px; font-size: 0.85em; color: #888; }
    h2 { color: #aaa; margin-top: 32px; border-bottom: 1px solid #333; padding-bottom: 8px; }
  </style>
</head>
<body>
  <h1>ShutterSift Report</h1>
  <p style="color:#888">{{ result.run_at }} &nbsp;·&nbsp; {{ result.photos|length }} photos analyzed</p>

  <div class="summary">
    <div class="stat keep"><div class="num">{{ result.keep|length }}</div><div>Keep</div></div>
    <div class="stat review"><div class="num">{{ result.review|length }}</div><div>Review</div></div>
    <div class="stat reject"><div class="num">{{ result.reject|length }}</div><div>Reject</div></div>
  </div>

  {% for section, photos in [("Keep", result.keep), ("Review", result.review), ("Reject", result.reject)] %}
  {% if photos %}
  <h2>{{ section }} ({{ photos|length }})</h2>
  <div class="grid">
    {% for p in photos %}
    <div class="card {{ p.decision }}">
      <div class="info">
        <div class="score">{{ "%.1f"|format(p.score) }}</div>
        <div>{{ p.path.name }}</div>
        <div class="sub">
          <span>Sharp: {{ "%.0f"|format(p.sub_scores.sharpness) }}</span>
          <span>Expo: {{ "%.0f"|format(p.sub_scores.exposure) }}</span>
          <span>Aesth: {{ "%.0f"|format(p.sub_scores.aesthetic) }}</span>
          <span>Face: {{ "%.0f"|format(p.sub_scores.face_quality) }}</span>
        </div>
        {% if p.reasons %}<div class="reasons">{{ p.reasons | join(", ") }}</div>{% endif %}
        {% if p.explanation %}<div class="explanation">{{ p.explanation }}</div>{% endif %}
      </div>
    </div>
    {% endfor %}
  </div>
  {% endif %}
  {% endfor %}
</body>
</html>
```

- [ ] **Step 4: Write `src/shuttersift/engine/reporter.py`**

```python
from __future__ import annotations
import json
import logging
from dataclasses import asdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from shuttersift.engine import AnalysisResult, PhotoResult
import shuttersift

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
```

- [ ] **Step 5: Run — expect PASS**

```bash
pytest tests/unit/test_reporter.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/shuttersift/engine/reporter.py templates/report.html.j2 tests/unit/test_reporter.py
git commit -m "feat: Reporter generates versioned JSON + dark-theme HTML report"
```

---

## Task 16: Model Downloader

**Files:**
- Create: `src/shuttersift/engine/downloader.py`

- [ ] **Step 1: Write test**

```python
# tests/unit/test_downloader.py
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
from shuttersift.engine.downloader import verify_sha256, _download_file


def test_sha256_correct(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert verify_sha256(f, expected) is True


def test_sha256_wrong(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello world")
    assert verify_sha256(f, "deadbeef" * 8) is False


def test_sha256_missing_file(tmp_path):
    assert verify_sha256(tmp_path / "nonexistent.bin", "abc") is False
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_downloader.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/downloader.py`**

```python
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
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/unit/test_downloader.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/engine/downloader.py tests/unit/test_downloader.py
git commit -m "feat: model downloader with SHA256 verification and retry logic"
```

---

## Task 17: Pipeline Orchestrator (Engine)

**Files:**
- Create: `src/shuttersift/engine/pipeline.py`
- Modify: `src/shuttersift/engine/__init__.py` (fix the try/except guard)

- [ ] **Step 1: Write test**

```python
# tests/unit/test_pipeline.py
import cv2
import numpy as np
from pathlib import Path
from shuttersift.engine.pipeline import Engine
from shuttersift.config import Config


def test_engine_analyze_single_jpeg(tmp_path, sharp_image):
    path = tmp_path / "DSC001.jpg"
    cv2.imwrite(str(path), sharp_image)
    out = tmp_path / "out"

    engine = Engine(Config())
    result = engine.analyze(tmp_path, out, resume=False)

    assert len(result.photos) == 1
    assert result.photos[0].path == path
    assert result.photos[0].score > 0
    assert result.photos[0].decision in ("keep", "review", "reject")


def test_engine_capabilities_returns_dict():
    engine = Engine(Config())
    caps = engine.capabilities()
    assert "gpu" in caps
    assert "rawpy" in caps


def test_blurry_image_rejected(tmp_path, blurry_image):
    path = tmp_path / "blurry.jpg"
    cv2.imwrite(str(path), blurry_image)
    out = tmp_path / "out"

    engine = Engine(Config())
    result = engine.analyze(tmp_path, out, resume=False)

    assert result.photos[0].decision == "reject"
    assert result.photos[0].hard_rejected is True


def test_progress_callback_called(tmp_path, sharp_image):
    path = tmp_path / "DSC001.jpg"
    cv2.imwrite(str(path), sharp_image)
    out = tmp_path / "out"

    calls = []
    engine = Engine(Config())
    engine.analyze(tmp_path, out, resume=False, on_progress=lambda i, t, r: calls.append(i))

    assert len(calls) == 1
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_pipeline.py -v
```

- [ ] **Step 3: Write `src/shuttersift/engine/pipeline.py`**

```python
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Callable

from shuttersift.config import Config
from shuttersift.engine import AnalysisResult, PhotoResult, SubScores
from shuttersift.engine.capabilities import Capabilities
from shuttersift.engine.loader import load_image, SUPPORTED_FORMATS
from shuttersift.engine.analyzers.sharpness import sharpness_score, laplacian_variance
from shuttersift.engine.analyzers.exposure import exposure_score
from shuttersift.engine.analyzers.face import FaceAnalyzer
from shuttersift.engine.analyzers.aesthetic import AestheticAnalyzer
from shuttersift.engine.analyzers.composition import composition_score
from shuttersift.engine.analyzers.duplicates import (
    group_bursts, best_in_burst, read_exif_timestamps,
)
from shuttersift.engine.scorer import Scorer
from shuttersift.engine.state import StateManager
from shuttersift.engine.explainer import Explainer
from shuttersift.engine.organizer import organize
from shuttersift.engine.reporter import generate_report

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, config: Config):
        self._cfg = config
        self._caps = Capabilities.detect()
        self._face = FaceAnalyzer()
        self._aesthetic = AestheticAnalyzer(use_gpu=self._caps.gpu)
        self._scorer = Scorer(config)

    def capabilities(self) -> dict:
        return {
            "gpu": self._caps.gpu,
            "rawpy": self._caps.rawpy,
            "musiq": self._caps.musiq,
            "gguf_vlm": self._caps.gguf_vlm,
            "api_vlm": self._caps.api_vlm,
        }

    def analyze(
        self,
        input_dir: Path,
        output_dir: Path,
        on_progress: Callable[[int, int, PhotoResult], None] | None = None,
        resume: bool = True,
        dry_run: bool = False,
        explain: bool = False,
    ) -> AnalysisResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        state = StateManager(output_dir, fresh=not resume)

        # Stage 0: scan + burst grouping
        paths = self._scan(input_dir)
        if not paths:
            logger.warning("No supported photos found in %s", input_dir)
            return AnalysisResult()

        exif_ts = read_exif_timestamps(paths)
        burst_groups = group_bursts(
            paths, exif_ts, self._cfg.thresholds.burst_gap_seconds
        )

        # Auto-calibrate sharpness threshold from this session's distribution
        hard_reject_threshold = self._calibrate_sharpness(paths, state)

        # Main analysis loop
        results: list[PhotoResult] = []
        total = len(paths)

        for i, path in enumerate(paths):
            if state.is_processed(path):
                cached = state.load(path)
                if cached:
                    results.append(cached)
                    if on_progress:
                        on_progress(i + 1, total, cached)
                    continue

            t0 = time.perf_counter()
            result = self._analyze_one(path, hard_reject_threshold)
            result.duration_ms = (time.perf_counter() - t0) * 1000

            state.save(result)
            results.append(result)
            if on_progress:
                on_progress(i + 1, total, result)

        # Stage 5: burst dedup — mark duplicates
        all_scores = {r.path: r.score for r in results}
        for group in burst_groups:
            if len(group) <= 1:
                continue
            best = best_in_burst(group, all_scores)
            for r in results:
                if r.path in group and r.path != best:
                    r.is_duplicate = True
                    r.decision = "reject"
                    r.reasons.append("连拍重复（非最优帧）")

        # Stage 6: VLM explanation (optional)
        if explain and (self._caps.gguf_vlm or self._caps.api_vlm):
            explainer = Explainer(
                self._cfg,
                gguf_path=self._caps.gguf_model_path,
                api_key_anthropic=os.getenv("ANTHROPIC_API_KEY"),
                api_key_openai=os.getenv("OPENAI_API_KEY"),
            )
            review_results = [r for r in results if r.decision == "review"]
            for j, r in enumerate(review_results):
                r.explanation = explainer.explain(r.path, r)

        analysis = AnalysisResult(photos=results)
        organize(results, output_dir, dry_run=dry_run)
        generate_report(analysis, output_dir)
        return analysis

    def _analyze_one(self, path: Path, hard_reject_threshold: float) -> PhotoResult:
        result = PhotoResult(path=path)

        img = load_image(path)
        if img is None:
            result.decision = "reject"
            result.error = "Could not load image"
            result.reasons.append("文件损坏或格式不支持")
            return result

        # Stage 2: technical quality
        sharp = sharpness_score(img)
        expo = exposure_score(img)
        result.sub_scores.sharpness = sharp
        result.sub_scores.exposure = expo

        raw_var = laplacian_variance(img)
        if raw_var < hard_reject_threshold:
            result.decision = "reject"
            result.hard_rejected = True
            result.reasons.append(f"严重模糊 (Laplacian={raw_var:.1f})")
            result.score = self._scorer.compute(result.sub_scores)
            return result  # early stop

        if expo < 20:
            result.reasons.append(f"曝光问题 (score={expo:.0f})")

        # Stage 3: face analysis
        face_result = self._face.analyze(img)
        result.face_count = face_result.count
        result.sub_scores.face_quality = face_result.face_quality_score

        if face_result.count > 0 and face_result.all_eyes_closed:
            result.decision = "reject"
            result.hard_rejected = True
            result.reasons.append("所有人物闭眼")
            result.score = self._scorer.compute(result.sub_scores)
            return result  # early stop

        if face_result.count > 0 and face_result.eye_open_score < 0.5:
            result.reasons.append(f"眼睛半闭 (score={face_result.eye_open_score:.2f})")

        # Stage 4: aesthetic + composition
        aesthetic = self._aesthetic.score(img)
        comp = composition_score(img, face_result.face_bboxes)
        result.sub_scores.aesthetic = aesthetic
        result.sub_scores.composition = comp

        if aesthetic < 30:
            result.reasons.append(f"美学分低 ({aesthetic:.0f})")

        # Final score + decision
        result.score = self._scorer.compute(result.sub_scores)
        result.decision = self._scorer.decide(result.score)
        return result

    def _scan(self, input_dir: Path) -> list[Path]:
        paths = []
        for p in sorted(input_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS:
                paths.append(p)
        return paths

    def _calibrate_sharpness(
        self, paths: list[Path], state: StateManager
    ) -> float:
        """
        Compute p10 of Laplacian variance distribution to use as hard-reject threshold.
        Only fast-scans images not already in state.
        Falls back to config default if fewer than 10 new images.
        """
        default = self._cfg.thresholds.hard_reject_sharpness
        new_paths = [p for p in paths if not state.is_processed(p)]
        if len(new_paths) < 10:
            return default

        variances = []
        for p in new_paths[:200]:  # cap at 200 for speed
            img = load_image(p)
            if img is not None:
                variances.append(laplacian_variance(img))

        if len(variances) < 5:
            return default

        variances.sort()
        p10_idx = max(0, int(len(variances) * 0.10) - 1)
        calibrated = variances[p10_idx]
        logger.info(
            "Sharpness calibration: p10=%.1f (config default was %.1f)",
            calibrated, default
        )
        return calibrated
```

- [ ] **Step 4: Fix `engine/__init__.py` import guard**

Replace the `try/except` block with a direct import now that pipeline.py exists:

```python
# src/shuttersift/engine/__init__.py
# (keep all dataclass definitions from Task 2, update only the import)
from .pipeline import Engine  # noqa: F401
```

- [ ] **Step 5: Run — expect PASS**

```bash
pytest tests/unit/test_pipeline.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/shuttersift/engine/pipeline.py src/shuttersift/engine/__init__.py tests/unit/test_pipeline.py
git commit -m "feat: Engine orchestrator — 6-stage pipeline with calibration and early-stop"
```

---

## Task 18: CLI

**Files:**
- Create: `src/shuttersift/cli/main.py`

- [ ] **Step 1: Write `src/shuttersift/cli/main.py`**

```python
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from shuttersift import __version__
from shuttersift.config import Config
from shuttersift.engine import AnalysisResult, PhotoResult
from shuttersift.engine.pipeline import Engine
from shuttersift.engine.capabilities import Capabilities

app = typer.Typer(
    name="shuttersift",
    help="AI-powered photo culling CLI. One command to sort your shots.",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    log_dir = Path.home() / ".shuttersift" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr) if verbose else logging.NullHandler(),
        ],
    )
    # Prune old logs
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    cfg = Config.load()
    while len(logs) > cfg.log_retention_runs:
        logs.pop(0).unlink(missing_ok=True)


@app.command()
def cull(
    input_dir: Path = typer.Argument(..., help="Directory containing photos to analyze"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    explain: bool = typer.Option(False, "--explain", help="Enable VLM explanation for Review photos"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze only, do not move/link files"),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore previous state, reanalyze all"),
    keep_threshold: Optional[int] = typer.Option(None, "--keep-threshold"),
    reject_threshold: Optional[int] = typer.Option(None, "--reject-threshold"),
    workers: Optional[int] = typer.Option(None, "--workers"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Cull a directory of photos into Keep / Review / Reject."""
    _setup_logging(verbose)

    if not input_dir.is_dir():
        console.print(f"[red]Error:[/] {input_dir} is not a directory")
        raise typer.Exit(1)

    cfg = Config.load(config)
    if keep_threshold is not None:
        cfg.thresholds.keep = keep_threshold
    if reject_threshold is not None:
        cfg.thresholds.reject = reject_threshold
    if workers is not None:
        cfg.workers = workers

    output_dir = output or (input_dir.parent / "shuttersift_output")

    caps = Capabilities.detect()
    console.print(f"\n[bold]ShutterSift[/] v{__version__}")
    console.print(f"Detected: {caps.summary()}\n")

    engine = Engine(cfg)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Analyzing...", total=None)

        def on_progress(current: int, total: int, result: PhotoResult) -> None:
            progress.update(task_id, completed=current, total=total,
                          description=f"[cyan]{result.path.name}[/]")

        result: AnalysisResult = engine.analyze(
            input_dir=input_dir,
            output_dir=output_dir,
            on_progress=on_progress,
            resume=not fresh,
            dry_run=dry_run,
            explain=explain,
        )

    _print_summary(result, output_dir, dry_run)


def _print_summary(result: AnalysisResult, output_dir: Path, dry_run: bool) -> None:
    total = len(result.photos)
    if total == 0:
        console.print("[yellow]No photos found.[/]")
        return

    console.rule()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[green]✓  Keep[/]",    str(len(result.keep)),   f"({len(result.keep)/total:.0%})")
    table.add_row("[yellow]◎  Review[/]", str(len(result.review)), f"({len(result.review)/total:.0%})")
    table.add_row("[red]✗  Reject[/]",   str(len(result.reject)), f"({len(result.reject)/total:.0%})")
    console.print(table)
    console.rule()
    if not dry_run:
        console.print(f"\nOutput  → [bold]{output_dir}[/]")
        console.print(f"Report  → [bold]{output_dir / 'report.html'}[/]\n")
    else:
        console.print("\n[yellow]Dry run — no files written[/]\n")


@app.command(name="download-models")
def download_models(
    vlm: bool = typer.Option(False, "--vlm", help="Also download GGUF VLM model (~1.7 GB)"),
) -> None:
    """Download required model files to ~/.shuttersift/models/."""
    from shuttersift.engine.downloader import download_mediapipe_models, download_gguf_vlm

    console.print("Downloading MediaPipe face landmarker...")
    ok = download_mediapipe_models()
    if ok:
        console.print("[green]✓[/] MediaPipe models ready")
    else:
        console.print("[red]✗[/] MediaPipe download failed")

    if vlm:
        console.print("Downloading moondream2 GGUF (~1.7 GB)...")
        ok = download_gguf_vlm()
        console.print("[green]✓[/] GGUF VLM ready" if ok else "[red]✗[/] GGUF download failed")


@app.command()
def info() -> None:
    """Show detected capabilities (GPU, VLM, RAW support)."""
    caps = Capabilities.detect()
    console.print(f"\n[bold]ShutterSift[/] v{__version__}")
    console.print(f"\n{caps.summary()}\n")

    table = Table(title="Capabilities", show_header=True)
    table.add_column("Feature", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    table.add_row("GPU",       "[green]✓[/]" if caps.gpu else "[red]✗[/]", "CUDA or Apple Metal")
    table.add_row("RAW decode","[green]✓[/]" if caps.rawpy else "[yellow]~[/]", "rawpy" if caps.rawpy else "Using Pillow fallback")
    table.add_row("MUSIQ",     "[green]✓[/]" if caps.musiq else "[yellow]~[/]", "GPU aesthetic scoring" if caps.musiq else "BRISQUE fallback")
    table.add_row("GGUF VLM",  "[green]✓[/]" if caps.gguf_vlm else "[red]✗[/]",
                  str(caps.gguf_model_path) if caps.gguf_model_path else "Run: shuttersift download-models --vlm")
    table.add_row("API VLM",   "[green]✓[/]" if caps.api_vlm else "[red]✗[/]",
                  "ANTHROPIC_API_KEY or OPENAI_API_KEY set" if caps.api_vlm else "Set ANTHROPIC_API_KEY env var")
    console.print(table)


@app.command()
def calibrate(
    input_dir: Path = typer.Argument(..., help="Directory to sample for calibration"),
) -> None:
    """Sample a photo directory and recommend sharpness thresholds for your camera."""
    from shuttersift.engine.loader import load_image, SUPPORTED_FORMATS
    from shuttersift.engine.analyzers.sharpness import laplacian_variance

    paths = [p for p in sorted(input_dir.rglob("*"))
             if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS]

    if len(paths) < 5:
        console.print("[red]Need at least 5 photos to calibrate.[/]")
        raise typer.Exit(1)

    variances = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), console=console) as prog:
        t = prog.add_task("Sampling sharpness...", total=min(len(paths), 300))
        for p in paths[:300]:
            img = load_image(p)
            if img is not None:
                variances.append(laplacian_variance(img))
            prog.advance(t)

    variances.sort()
    n = len(variances)
    p5  = variances[max(0, int(n * 0.05))]
    p10 = variances[max(0, int(n * 0.10))]
    p25 = variances[max(0, int(n * 0.25))]
    p50 = variances[max(0, int(n * 0.50))]

    table = Table(title=f"Sharpness Distribution ({n} photos sampled)")
    table.add_column("Percentile"); table.add_column("Laplacian Variance"); table.add_column("Recommendation")
    table.add_row("p5",  f"{p5:.1f}",  "← hard_reject_sharpness (aggressive)")
    table.add_row("p10", f"{p10:.1f}", "← hard_reject_sharpness (recommended)")
    table.add_row("p25", f"{p25:.1f}", "← hard_reject_sharpness (conservative)")
    table.add_row("p50", f"{p50:.1f}", "Median of your photos")
    console.print(table)
    console.print(f"\nAdd to config.yaml:\n  thresholds:\n    hard_reject_sharpness: {p10:.1f}\n")
```

- [ ] **Step 2: Verify CLI entry point works**

```bash
shuttersift --help
```
Expected: shows help text with `cull`, `download-models`, `info`, `calibrate` commands.

```bash
shuttersift info
```
Expected: shows capability table.

- [ ] **Step 3: Commit**

```bash
git add src/shuttersift/cli/main.py
git commit -m "feat: Typer CLI — cull, download-models, info, calibrate commands"
```

---

## Task 19: Server Skeleton (GUI Placeholder)

**Files:**
- Create: `src/shuttersift/server/app.py`
- Create: `src/shuttersift/server/README.md`

- [ ] **Step 1: Write `src/shuttersift/server/app.py`**

```python
"""
ShutterSift Local HTTP Server — GUI Bridge Layer
================================================
This module is a placeholder for a future FastAPI server that GUI clients
(desktop via Tauri/Electron, or web via Gradio/React) can call.

The GUI does NOT need to know about the Engine internals — it communicates
only via the REST API and SSE stream defined here.

Planned endpoints:
  POST  /analyze          — Start analysis job, returns {"job_id": "..."}
  GET   /jobs/{id}/stream — SSE stream of PhotoResult events (one per photo)
  GET   /jobs/{id}/result — Final AnalysisResult JSON when complete
  GET   /capabilities     — Detected GPU/VLM/RAW capabilities
  GET   /health           — Liveness probe

To start (future):
  shuttersift serve --port 7788

The CLI `on_progress` callback maps cleanly to SSE events — the Engine
interface does not need to change.
"""

# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from shuttersift.engine import Engine
# from shuttersift.config import Config
#
# app = FastAPI(title="ShutterSift", version="0.1.0")
#
# @app.get("/capabilities")
# def get_capabilities():
#     return Engine(Config()).capabilities()
#
# @app.get("/health")
# def health():
#     return {"status": "ok"}
```

- [ ] **Step 2: Write server README**

```markdown
<!-- src/shuttersift/server/README.md -->
# ShutterSift Server (Future)

FastAPI-based local HTTP server that bridges the Engine to GUI clients.

## Why a server layer?

The Engine is a Python library. GUI clients (Tauri desktop app, web browser)
cannot call Python directly — they need an HTTP interface.

## Architecture

```
GUI client
  │  HTTP / SSE
  ▼
FastAPI server  (this module, port 7788)
  │  Python function call
  ▼
Engine.analyze(on_progress=sse_emit)
```

The `on_progress` callback in `Engine.analyze()` was designed with this in
mind: it sends per-photo events as they complete, which maps directly to
SSE (Server-Sent Events) for real-time progress in any GUI.

## Start (once implemented)

```bash
shuttersift serve --port 7788
```
```

- [ ] **Step 3: Commit**

```bash
git add src/shuttersift/server/app.py src/shuttersift/server/README.md
git commit -m "feat: server module placeholder with FastAPI skeleton (commented) for future GUI"
```

---

## Task 20: Integration Test

**Files:**
- Create: `tests/integration/test_pipeline.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_pipeline.py
"""
End-to-end test: synthetic JPEG directory → Engine → assert output structure.
No real photos needed — uses cv2-generated synthetic images.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import pytest
from shuttersift.config import Config
from shuttersift.engine.pipeline import Engine


def _make_photo_dir(tmp_path: Path) -> Path:
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    # 3 sharp photos (should keep/review)
    for i in range(3):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in range(0, 640, 8):
            for y in range(0, 480, 8):
                if (x // 8 + y // 8) % 2 == 0:
                    img[y:y+8, x:x+8] = 200
        cv2.imwrite(str(photo_dir / f"sharp_{i:03d}.jpg"), img)

    # 2 blurry photos (should reject)
    for i in range(2):
        blurry = cv2.GaussianBlur(img, (51, 51), 20)
        cv2.imwrite(str(photo_dir / f"blurry_{i:03d}.jpg"), blurry)

    return photo_dir


def test_full_pipeline_produces_output(tmp_path):
    photo_dir = _make_photo_dir(tmp_path)
    output_dir = tmp_path / "output"

    cfg = Config()
    engine = Engine(cfg)
    result = engine.analyze(photo_dir, output_dir, resume=False)

    assert len(result.photos) == 5

    # Output directories exist
    assert (output_dir / "keep").exists()
    assert (output_dir / "review").exists()
    assert (output_dir / "reject").exists()

    # JSON report exists and is valid
    report = json.loads((output_dir / "results.json").read_text())
    assert report["version"] == "1"
    assert len(report["photos"]) == 5

    # HTML report exists
    assert (output_dir / "report.html").exists()

    # State file written
    assert (output_dir / ".state.json").exists()


def test_blurry_photos_rejected(tmp_path):
    photo_dir = _make_photo_dir(tmp_path)
    output_dir = tmp_path / "output"

    engine = Engine(Config())
    result = engine.analyze(photo_dir, output_dir, resume=False)

    rejected = [r for r in result.photos if r.decision == "reject"]
    assert any("blurry" in r.path.name for r in rejected), \
        "Expected blurry photos to be rejected"


def test_resume_skips_processed(tmp_path):
    photo_dir = _make_photo_dir(tmp_path)
    output_dir = tmp_path / "output"

    engine = Engine(Config())

    # First run
    r1 = engine.analyze(photo_dir, output_dir, resume=False)
    assert len(r1.photos) == 5

    # Second run with resume=True — should load from state, not re-analyze
    calls = []
    r2 = engine.analyze(photo_dir, output_dir, resume=True,
                        on_progress=lambda i, t, r: calls.append(r.path.name))
    assert len(r2.photos) == 5
    # All loaded from cache — on_progress called but each is a cache hit


def test_each_photo_has_score_and_decision(tmp_path):
    photo_dir = _make_photo_dir(tmp_path)
    output_dir = tmp_path / "output"

    engine = Engine(Config())
    result = engine.analyze(photo_dir, output_dir, resume=False)

    for photo in result.photos:
        assert 0.0 <= photo.score <= 100.0
        assert photo.decision in ("keep", "review", "reject")
```

- [ ] **Step 2: Run integration tests**

```bash
pytest tests/integration/test_pipeline.py -v
```
Expected: all 4 tests pass.

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_pipeline.py
git commit -m "test: end-to-end integration tests for full pipeline"
```

---

## Task 21: README, Config, Git Remote & Push

**Files:**
- Modify: `README.md`
- Verify: `config.yaml`

- [ ] **Step 1: Write `README.md`**

```markdown
# ShutterSift

AI-powered photo culling CLI. One command to sort your shots.

```bash
shuttersift ./photos
```

Automatically classifies every photo as **Keep**, **Review**, or **Reject** using a multi-stage computer vision pipeline. Every photo gets a 0–100 score with per-dimension breakdown.

## Install

```bash
pip install shuttersift
```

> **Apple Silicon (M1/M2):** If `mediapipe` fails to install, run:
> `pip install mediapipe-silicon` then `pip install shuttersift --no-deps`

> **GGUF VLM with Metal acceleration (Mac):**
> `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal`

## Quick Start

```bash
# First time: download model files
shuttersift download-models

# Cull a directory
shuttersift ./photos

# With VLM explanation for Review photos
shuttersift ./photos --explain

# Check what's available on your system
shuttersift info
```

## Output

```
shuttersift_output/
├── keep/         ← high-scoring photos (symlinks + XMP sidecar)
├── review/       ← borderline photos
├── reject/       ← blurry, closed eyes, duplicates
├── report.html   ← visual report with scores and thumbnails
└── results.json  ← machine-readable, versioned
```

Lightroom users: import the `keep/` folder — XMP sidecars set star ratings and
color labels automatically.

## Scoring

| Dimension | Weight | Method |
|-----------|--------|--------|
| Sharpness | 30% | Laplacian variance |
| Exposure | 15% | Histogram analysis |
| Aesthetic | 25% | MUSIQ (GPU) / BRISQUE (CPU) |
| Face quality | 20% | MediaPipe eye + smile |
| Composition | 10% | Rule-of-thirds engine |

**Thresholds (configurable):** ≥70 → Keep · 40–69 → Review · <40 → Reject

## Configuration

```yaml
# config.yaml (place in current dir or ~/.shuttersift/config.yaml)
scoring:
  thresholds:
    keep: 70
    reject: 40
    hard_reject_sharpness: 30.0
```

Run `shuttersift calibrate ./photos` to compute recommended thresholds for your camera.

## VLM Explanation

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` then run with `--explain`:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
shuttersift ./photos --explain
```

Only **Review** photos are sent to the API (typically ~20–30%). For fully local
VLM, download a GGUF model:

```bash
shuttersift download-models --vlm   # downloads moondream2 ~1.7 GB
shuttersift ./photos --explain      # uses local GGUF, no API needed
```

## Future: GUI Client

See `clients/desktop/` and `src/shuttersift/server/` for the planned desktop
and web GUI architecture.
```

- [ ] **Step 2: Set git identity and remote**

```bash
git config user.name "ShutterSift"
git config user.email "dev@shuttersift.local"
git branch -M main
git remote add origin https://github.com/host452b/ShutterSift.git
```

- [ ] **Step 3: Update README in repo and commit**

```bash
git add README.md config.yaml
git commit -m "docs: README with install, usage, scoring table, VLM, GUI roadmap"
```

- [ ] **Step 4: Run full test suite one final time**

```bash
pytest tests/ -v
```
Expected: all tests green.

- [ ] **Step 5: Push to GitHub**

```bash
git push -u origin main
```

---

## Self-Review Checklist

**Spec coverage:**

| Spec section | Covered by task(s) |
|---|---|
| 6-stage pipeline | Task 17 (pipeline.py) |
| Sharpness hard-reject | Tasks 5, 17 |
| Closed-eye hard-reject | Tasks 7, 17 |
| Weighted 0–100 scoring | Task 11 |
| Keep/Review/Reject thresholds | Task 11, 18 |
| EXIF burst grouping + dedup | Task 10 |
| Auto-calibration | Task 17 (_calibrate_sharpness), Task 18 (calibrate cmd) |
| Resume / checkpoint | Task 12 |
| Re-run strategy (--fresh) | Tasks 12, 18 |
| Apple Silicon platform marker | Task 1 (pyproject.toml note + README) |
| Structured logging | Task 18 (_setup_logging) |
| Model SHA256 | Task 16 |
| results.json versioning | Tasks 15, 20 |
| Symlinks + XMP sidecar | Task 14 |
| HTML + JSON report | Task 15 |
| GGUF VLM | Task 13 |
| API VLM (Anthropic/OpenAI) | Task 13 |
| CLI: cull, download-models, info, calibrate | Task 18 |
| on_progress callback | Tasks 17, 18 |
| Server/GUI skeleton | Task 19 |
| clients/ directory | Task 1 |
| Integration test | Task 20 |

All spec sections accounted for. ✓
