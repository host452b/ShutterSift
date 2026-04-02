# VLM Blocker Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three BLOCKER issues found in the pre-release review: broken local VLM (model format mismatch + wrong API), Chinese-language prompt template, and `llama-cpp-python` as a hard dependency that requires C++ compilation.

**Architecture:** Replace `llama-cpp-python` with the `moondream` PyPI package (pure Python, no compilation, natively loads `.mf` files). Update `capabilities.py` to glob `*.mf` instead of `*.gguf`. Rewrite `_explain_gguf` → `_explain_moondream` using Moondream's Python API. Change the prompt template to English. Move `llama-cpp-python` out of hard deps and add `moondream` as an optional `[vlm]` dependency. Also fix the `pydantic-settings` phantom dependency, add PyPI metadata, make `rawpy` optional, validate `--keep`/`--reject` inversion, and fix the hardcoded `12.4 MB` size string.

**Tech Stack:** Python 3.11+, `moondream` PyPI package (v0.0.5+), Pydantic v2, Typer, pytest

---

## File Map

| File | Action | Change |
|------|--------|--------|
| `src/shuttersift/engine/capabilities.py` | Modify | Glob `*.mf`, check `moondream` import |
| `src/shuttersift/engine/explainer.py` | Modify | Replace `_explain_gguf` with `_explain_moondream`, English prompt |
| `src/shuttersift/cli/main.py` | Modify | Fix `info` table label, fix size_hint from registry, validate --keep/--reject |
| `pyproject.toml` | Modify | Remove `llama-cpp-python` + `pydantic-settings` from hard deps; add `[vlm]` optional group; add `rawpy` to optional; add PyPI metadata |
| `tests/unit/test_explainer.py` | Modify | Add test for English prompt; update mock for moondream |
| `tests/unit/test_capabilities.py` | Modify | Update test to check `*.mf` glob |

---

## Task 1: Fix `capabilities.py` — glob `*.mf` and check `moondream`

**Files:**
- Modify: `src/shuttersift/engine/capabilities.py`
- Modify: `tests/unit/test_capabilities.py`

- [ ] **Step 1: Read current test file**

```bash
cat /localhome/swqa/workspace/photo_steth/tests/unit/test_capabilities.py
```

- [ ] **Step 2: Add a failing test for `.mf` detection**

Open `tests/unit/test_capabilities.py` and add at the end:

```python
def test_capabilities_detects_mf_model(tmp_path, monkeypatch):
    """Capabilities should detect moondream .mf files, not .gguf."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    mf_file = models_dir / "moondream2-int8.mf"
    mf_file.touch()
    import shuttersift.engine.capabilities as cap_mod
    monkeypatch.setattr(cap_mod, "MODELS_DIR", models_dir)
    caps = cap_mod.Capabilities.detect()
    assert caps.gguf_vlm is True
    assert caps.gguf_model_path == mf_file

def test_capabilities_does_not_detect_gguf_as_mf(tmp_path, monkeypatch):
    """A .gguf file should NOT trigger gguf_vlm detection (wrong format)."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "some_model.gguf").touch()
    import shuttersift.engine.capabilities as cap_mod
    monkeypatch.setattr(cap_mod, "MODELS_DIR", models_dir)
    caps = cap_mod.Capabilities.detect()
    assert caps.gguf_vlm is False
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/test_capabilities.py -v --tb=short -k "mf" 2>&1 | tail -15
```

Expected: FAIL — `.mf` not found because code globs `*.gguf`.

- [ ] **Step 4: Fix `capabilities.py`**

Replace the `detect` classmethod in `src/shuttersift/engine/capabilities.py`. The only change is the glob pattern and the label. Replace the full file with:

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
    except Exception:
        return False


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except Exception:
        return False


@dataclass
class Capabilities:
    gpu: bool
    rawpy: bool
    musiq: bool
    gguf_vlm: bool           # True when a local Moondream .mf model is present
    gguf_model_path: Path | None
    api_vlm: bool

    @classmethod
    def detect(cls) -> "Capabilities":
        # Moondream models use the .mf (Moondream Format) extension
        mf_models = list(MODELS_DIR.glob("*.mf")) if MODELS_DIR.exists() else []
        return cls(
            gpu=_has_gpu(),
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

        parts = [
            flag(self.gpu, "GPU"),
            flag(self.rawpy, "RAW"),
            flag(self.musiq, "MUSIQ"),
            flag(self.gguf_vlm, "Local VLM"),
            flag(self.api_vlm, "API VLM"),
        ]
        return "  ".join(parts)
```

- [ ] **Step 5: Run tests**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/test_capabilities.py -v --tb=short 2>&1 | tail -15
```

Expected: all pass (including the 2 new tests).

- [ ] **Step 6: Commit**

```bash
cd /localhome/swqa/workspace/photo_steth
git add src/shuttersift/engine/capabilities.py tests/unit/test_capabilities.py
git commit -m "fix: detect moondream .mf models, require moondream package for local VLM"
```

---

## Task 2: Fix `explainer.py` — Moondream API + English prompt

**Files:**
- Modify: `src/shuttersift/engine/explainer.py`
- Modify: `tests/unit/test_explainer.py`

- [ ] **Step 1: Read current test file**

```bash
cat /localhome/swqa/workspace/photo_steth/tests/unit/test_explainer.py
```

- [ ] **Step 2: Add failing tests**

Add to `tests/unit/test_explainer.py`:

```python
def test_prompt_template_is_english():
    """_PROMPT_TEMPLATE must be in English, not Chinese."""
    from shuttersift.engine.explainer import _PROMPT_TEMPLATE
    # Check for absence of common Chinese characters
    chinese_chars = any('\u4e00' <= c <= '\u9fff' for c in _PROMPT_TEMPLATE)
    assert not chinese_chars, "Prompt template contains Chinese characters"

def test_prompt_template_contains_score_placeholder():
    from shuttersift.engine.explainer import _PROMPT_TEMPLATE
    assert "{score" in _PROMPT_TEMPLATE

def test_moondream_explain_skips_when_no_package(tmp_path, monkeypatch):
    """_explain_moondream returns '' when moondream is not installed."""
    import sys
    monkeypatch.setitem(sys.modules, "moondream", None)
    from shuttersift.engine.explainer import Explainer
    from shuttersift.config import Config
    from unittest.mock import MagicMock
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"")
    exp = Explainer(
        config=Config(),
        gguf_path=tmp_path / "model.mf",
        api_key_anthropic=None,
        api_key_openai=None,
    )
    result = exp._explain_moondream(photo, "test prompt")
    assert result == ""
```

- [ ] **Step 3: Run to confirm tests fail**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/test_explainer.py -v --tb=short -k "english or moondream" 2>&1 | tail -15
```

Expected: FAIL — prompt is Chinese, method is named `_explain_gguf`.

- [ ] **Step 4: Rewrite `explainer.py`**

Replace `src/shuttersift/engine/explainer.py` with:

```python
from __future__ import annotations
import base64
import logging
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

_PROMPT_TEMPLATE = (
    "You are a professional photographer reviewing an image. "
    "Technical scores: overall {score:.0f}/100, sharpness {sharpness:.0f}, "
    "exposure {exposure:.0f}, aesthetic {aesthetic:.0f}. "
    "In 1-2 sentences, explain whether this photo is worth keeping "
    "and identify its main strength or weakness."
)


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
        self._moondream_model = None

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

        # Priority: local Moondream → Anthropic API → OpenAI API → skip
        if self._gguf_path and self._gguf_path.exists():
            text = self._explain_moondream(photo_path, prompt)
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
            img = cv2.imread(str(path))
            if img is None:
                return None
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

    def _explain_moondream(self, path: Path, prompt: str) -> str:
        """Run local Moondream2 inference on the image."""
        try:
            import moondream as md
        except ImportError:
            logger.warning("moondream package not installed; skipping local VLM")
            return ""
        try:
            if self._moondream_model is None:
                self._moondream_model = md.vl(model=str(self._gguf_path))
            image = md.Image.from_path(str(path))
            result = self._moondream_model.query(image, prompt)
            # moondream.query returns {"answer": "..."} or just a string depending on version
            if isinstance(result, dict):
                return result.get("answer", "").strip()
            return str(result).strip()
        except Exception as e:
            logger.warning("Moondream inference error: %s", e)
            return ""
```

- [ ] **Step 5: Run explainer tests**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/test_explainer.py -v --tb=short 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /localhome/swqa/workspace/photo_steth
git add src/shuttersift/engine/explainer.py tests/unit/test_explainer.py
git commit -m "fix: replace llama-cpp VLM with moondream API, English prompt template"
```

---

## Task 3: Fix `pyproject.toml` — deps + metadata

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read the current file**

```bash
cat /localhome/swqa/workspace/photo_steth/pyproject.toml
```

- [ ] **Step 2: Write the new `pyproject.toml`**

Replace the entire file with:

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
authors = [
    {name = "ShutterSift", email = "shuttersift@users.noreply.github.com"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Graphics",
    "Topic :: Utilities",
]
dependencies = [
    # Image I/O
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
    # VLM cloud
    "anthropic>=0.34",
    "openai>=1.45",
    # Config
    "pyyaml>=6.0",
    # CLI
    "typer>=0.12",
    "rich>=13.8",
    # Report
    "jinja2>=3.1",
]

[project.optional-dependencies]
raw = [
    # RAW file support (requires libraw headers; Pillow fallback used if absent)
    "rawpy>=0.21",
]
vlm = [
    # Local Moondream2 VLM inference (downloads moondream2-int8.mf via ss setup --vlm)
    "moondream>=0.0.5",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "scipy>=1.12",
]

[project.urls]
Homepage = "https://github.com/host452b/ShutterSift"
Repository = "https://github.com/host452b/ShutterSift"
"Bug Tracker" = "https://github.com/host452b/ShutterSift/issues"

[project.scripts]
shuttersift = "shuttersift.cli.main:app"
ss = "shuttersift.cli.main:app"

[tool.hatch.build.targets.wheel]
packages = ["src/shuttersift"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "tests"]
```

- [ ] **Step 3: Verify the existing unit tests still pass**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/ -q --tb=short 2>&1 | tail -10
```

Expected: all pass (removing deps from `pyproject.toml` doesn't affect the already-installed dev environment).

- [ ] **Step 4: Commit**

```bash
cd /localhome/swqa/workspace/photo_steth
git add pyproject.toml
git commit -m "fix: remove phantom pydantic-settings dep, make rawpy+llama-cpp optional, add PyPI metadata"
```

---

## Task 4: Fix `cli/main.py` — `info` table label, size_hint from registry, --keep/--reject validation

**Files:**
- Modify: `src/shuttersift/cli/main.py`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Add a failing test for --keep/--reject inversion**

Add to `tests/unit/test_cli.py`:

```python
def test_keep_reject_inversion_rejected(tmp_path):
    """--keep lower than --reject should fail with a clear error."""
    photos = tmp_path / "photos"
    photos.mkdir()
    result = runner.invoke(app, [str(photos), "--keep", "30", "--reject", "80", "-n"])
    assert result.exit_code != 0
    assert "keep" in result.output.lower() or "reject" in result.output.lower()
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/test_cli.py::test_keep_reject_inversion_rejected -v --tb=short
```

Expected: FAIL (no validation currently).

- [ ] **Step 3: Apply three fixes to `cli/main.py`**

**Fix A — Validate --keep/--reject in `_do_scan`**

Find this block in `_do_scan` (after the `if jobs is not None:` line):

```python
    if jobs is not None:
        cfg.workers = jobs

    output_dir = output or (input_dir.parent / "shuttersift_output")
```

Replace with:

```python
    if jobs is not None:
        cfg.workers = jobs

    # Validate threshold ordering
    effective_keep = keep if keep is not None else cfg.thresholds.keep
    effective_reject = reject if reject is not None else cfg.thresholds.reject
    if effective_keep <= effective_reject:
        console.print(
            f"[red]Error:[/] --keep ({effective_keep}) must be greater than "
            f"--reject ({effective_reject}). "
            f"Example: --keep 70 --reject 40"
        )
        raise typer.Exit(1)

    output_dir = output or (input_dir.parent / "shuttersift_output")
```

**Fix B — Use size_hint from registry in `setup` command**

Find this line in the `setup` command:

```python
    console.print("      Downloading...  [dim](12.4 MB)[/]")
```

Replace with:

```python
    from shuttersift.engine.downloader import MODEL_REGISTRY
    size = MODEL_REGISTRY["mediapipe_face_landmarker"]["size_hint"]
    console.print(f"      Downloading...  [dim]({size})[/]")
```

**Fix C — Update `info` table label from "GGUF VLM" to "Local VLM"**

Find both occurrences of `"GGUF VLM"` in the `info` command and change to `"Local VLM"`:

```python
    table.add_row("Local VLM", "[green]✓[/]" if caps.gguf_vlm else "[red]✗[/]",
                  str(caps.gguf_model_path) if caps.gguf_model_path else "Run: ss setup --vlm")
```

- [ ] **Step 4: Run all CLI tests**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/test_cli.py -v --tb=short 2>&1 | tail -20
```

Expected: 17/17 pass (16 existing + 1 new).

- [ ] **Step 5: Run full unit suite**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/ -q --tb=short 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /localhome/swqa/workspace/photo_steth
git add src/shuttersift/cli/main.py tests/unit/test_cli.py
git commit -m "fix: validate --keep>--reject, use size_hint from registry, rename Local VLM label"
```

---

## Task 5: Final smoke test

- [ ] **Step 1: Full test suite**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/ -v --tb=short 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 2: Verify ss info output**

```bash
ss info 2>&1
```

Expected: Shows "Local VLM" row (not "GGUF VLM").

- [ ] **Step 3: Verify --keep/--reject validation**

```bash
mkdir -p /tmp/ss_vlm_fix_test
ss /tmp/ss_vlm_fix_test --keep 30 --reject 80 -n 2>&1
echo "Exit: $?"
```

Expected: exit non-zero, error mentions keep/reject.

- [ ] **Step 4: Verify VLM capabilities detection**

```bash
python3 -c "
from shuttersift.engine.capabilities import Capabilities, MODELS_DIR
print('MODELS_DIR:', MODELS_DIR)
caps = Capabilities.detect()
print('gguf_vlm (expect False, no .mf installed):', caps.gguf_vlm)
print('gguf_model_path:', caps.gguf_model_path)
"
```

Expected: `gguf_vlm: False` (no `.mf` file installed in CI), `gguf_model_path: None`.

- [ ] **Step 5: Git log**

```bash
cd /localhome/swqa/workspace/photo_steth
git log --oneline -8
```

Report all results.
