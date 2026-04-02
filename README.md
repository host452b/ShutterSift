# ShutterSift

AI-powered photo culling CLI. One command to sort your shots.

```bash
shuttersift ./photos
```

Automatically classifies every photo as **Keep**, **Review**, or **Reject** using a multi-stage computer vision pipeline. Every photo gets a 0–100 score with per-dimension breakdown.

## Install

**From PyPI** (once published):
```bash
pip install shuttersift
```

**From the latest GitHub release** (available now):
```bash
pip install https://github.com/host452b/ShutterSift/releases/download/v0.1.0/shuttersift-0.1.0-py3-none-any.whl
```

**Directly from source**:
```bash
pip install git+https://github.com/host452b/ShutterSift.git
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
