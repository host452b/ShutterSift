# ShutterSift

AI-powered photo culling CLI. Drop a folder, get Keep / Review / Reject.

```bash
ss ./photos
```

Automatically classifies every photo with a 0–100 score across five quality
dimensions. Results land in organized folders with an interactive HTML report.

---

## Platform Support

| Feature | macOS | Linux | Windows |
|---------|-------|-------|---------|
| Core scanning | ✅ | ✅ | ✅ |
| RAW file decode | ✅ | ✅ | ✅ (needs Visual C++ Redistributable) |
| GPU acceleration | ✅ Metal | ✅ CUDA | ✅ CUDA |
| Symlinks in output | ✅ | ✅ | ⚠️ Requires Developer Mode or admin |
| Cloud VLM (`-e`) | ✅ | ✅ | ✅ |
| Local VLM (`setup --vlm`) | ✅ | ✅ | ✅ |
| Lightroom XMP import | ✅ | ✅ | ✅ |
| Python version | 3.11 / 3.12 | 3.11 / 3.12 | 3.11 recommended |

> **mediapipe on Windows:** Python 3.12 may have compatibility issues. Use Python 3.11 if installation fails.

---

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

> **Apple Silicon (M1/M2/M3):** If `mediapipe` fails to install:
> ```bash
> pip install mediapipe-silicon
> pip install shuttersift --no-deps
> ```

> **GPU acceleration on Mac (Metal):**
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
> ```

> **Windows — Python 3.11 recommended:** If `mediapipe` fails on Python 3.12, switch to 3.11.
>
> **Windows — `ss` / `shuttersift` command not found:** Python's `Scripts` folder
> is not always on PATH. Fix with one of:
>
> **Option A — Virtual environment (recommended):**
> ```cmd
> python -m venv ss-env
> ss-env\Scripts\activate
> pip install git+https://github.com/host452b/ShutterSift.git
> ss --help
> ```
>
> **Option B — Add Scripts to PATH permanently:**
> 1. Search for **"Edit the system environment variables"**
> 2. Click **Environment Variables → Path → Edit**
> 3. Add: `C:\Users\<you>\AppData\Local\Programs\Python\Python311\Scripts`
> 4. Restart your terminal
>
> **Windows — symlinks:** Output folders use symlinks by default. Without the steps
> below, files are copied instead (a yellow warning will appear):
> 1. Open **Settings → Privacy & Security → For Developers**
> 2. Enable **Developer Mode**
>
> Or simply run your terminal as Administrator.

---

## Quick Start

```bash
# Step 1 — download required model files (one time only)
ss setup

# Step 2 — scan your photos
ss ./photos

# Step 3 — open the report
open shuttersift_output/report.html              # macOS
xdg-open shuttersift_output/report.html          # Linux
start shuttersift_output\report.html             # Windows
```

Sharpness thresholds are calibrated automatically on first run based on your
photo library. No configuration needed.

Both `ss` and `shuttersift` are valid command names — use whichever you prefer.
Both `-h` and `--help` are supported on every command.

---

## Output

```
shuttersift_output/
├── keep/         ← high-scoring photos (symlinks + XMP sidecar)
├── review/       ← borderline photos worth a second look
├── reject/       ← blurry, closed eyes, duplicates
├── report.html   ← interactive visual report with scores and thumbnails
└── results.json  ← machine-readable, versioned
```

**Lightroom users:** import the `keep/` folder — XMP sidecars set star ratings
and color labels automatically.

---

## Common Options

```bash
ss ./photos -e               # enable VLM explanation for Review photos
ss ./photos -n               # dry run — analyze only, write nothing
ss ./photos -f               # force reanalysis (ignore cached results)
ss ./photos --recalibrate    # redo sharpness calibration for this library
ss ./photos -o ./sorted      # custom output directory
ss ./photos --keep 75 --reject 35   # custom score thresholds
ss ./photos -j 8             # use 8 parallel workers
ss ./photos -v               # verbose logging
```

| Flag | Long form | Default | Description |
|------|-----------|---------|-------------|
| `-e` | `--explain` | off | VLM explanation for borderline photos |
| `-n` | `--dry-run` | off | Analyze only, do not move or link files |
| `-f` | `--force` | off | Ignore cached results, reanalyze all |
| `-o` | `--output <dir>` | `../shuttersift_output` | Output directory |
| `-j` | `--jobs <n>` | 4 | Parallel worker count |
| `-v` | `--verbose` | off | Print debug logs to stderr |
| `-c` | `--config <file>` | auto | Path to config file |
| | `--keep <n>` | 70 | Minimum score to Keep |
| | `--reject <n>` | 40 | Maximum score before Reject |
| | `--recalibrate` | off | Force redo sharpness calibration |

---

## VLM Explanation

When `-e` / `--explain` is passed, borderline **Review** photos get an
AI-generated text description explaining what's wrong or interesting. Typically
20–30% of photos land in Review.

**Cloud API (Anthropic or OpenAI):**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
ss ./photos -e
```
```bash
export OPENAI_API_KEY=sk-...
ss ./photos -e
```

**Fully local (no internet required):**
```bash
ss setup --vlm        # downloads moondream2 GGUF, ~1.7 GB, one-time
ss ./photos -e        # uses local model, no API key needed
```

---

## Scoring

Every photo receives a 0–100 composite score:

| Dimension | Weight | Method |
|-----------|--------|--------|
| Sharpness | 30% | Laplacian variance |
| Exposure | 15% | Histogram analysis |
| Aesthetic | 25% | MUSIQ (GPU) / BRISQUE (CPU) |
| Face quality | 20% | MediaPipe eye-open + smile |
| Composition | 10% | Rule-of-thirds engine |

**Default thresholds:** ≥ 70 → Keep · 40–69 → Review · < 40 → Reject

Override with `--keep` and `--reject`.

---

## Advanced

### Configuration file

ShutterSift looks for a config file in this order:
1. `--config <path>` (explicit)
2. `./shuttersift.yaml` (current directory)
3. `./config.yaml` (current directory)
4. `~/.shuttersift/config.yaml` (user global)

Full reference (`~/.shuttersift/config.yaml`):

```yaml
calibrated: true               # set automatically on first run

scoring:
  thresholds:
    keep: 70                   # minimum score for Keep
    reject: 40                 # maximum score for Reject
    hard_reject_sharpness: 42.3  # written by auto-calibration
    eye_open_min: 0.25         # minimum eye-openness ratio (0–1)
    burst_gap_seconds: 2.0     # photos within this window = burst

workers: 4                     # parallel workers
log_retention_runs: 30         # how many log files to keep
```

### Manual calibration

```bash
ss calibrate ./photos
```

Samples up to 300 photos and prints the full Laplacian variance distribution,
then saves the recommended `hard_reject_sharpness` to `~/.shuttersift/config.yaml`.
Useful when switching cameras or shooting styles.

### Other commands

```bash
ss info          # show GPU, VLM, and RAW support status
ss setup --vlm   # download local moondream2 VLM model
```

### GUI client (planned)

A desktop and web GUI client is planned. See `clients/desktop/` and
`clients/web/` for the early scaffolding.

---

## Chinese README

[中文文档 →](README.zh.md)
