# ShutterSift CLI UX Redesign — Design Spec

Date: 2026-04-02  
Status: Approved

## Background

The current README documents `shuttersift ./photos` but the actual CLI requires
`shuttersift cull ./photos` (typer multi-command mode). Additional gaps:

- `cull` is opaque to non-English speakers
- `--keep-threshold` / `--reject-threshold` are verbose with no short forms
- `--fresh` is semantically unclear
- `--workers` uses no standard unix convention
- `-h` is not supported (only `--help`)
- `calibrate` outputs YAML snippets and asks the user to edit config manually
- `download-models` setup output gives no guidance on what to do next
- Config auto-discovery does not check the current directory (code bug vs README claim)
- README mixes beginner and advanced content with no clear progression

## Goals

1. `ss ./photos` works with zero prior knowledge
2. Flag names follow unix conventions and are unambiguous in any language
3. First-run experience is fully automatic (calibration, model check)
4. Config complexity is hidden from normal users
5. Two README files: `README.md` (English) and `README.zh.md` (Chinese), both detailed

## Non-goals

- Interactive wizard / guided mode (out of scope)
- Changing the scoring algorithm or output format
- Removing the `calibrate` subcommand (kept as hidden advanced command)

---

## 1. Entry Points

Two entry points in `pyproject.toml`:

```toml
[project.scripts]
shuttersift = "shuttersift.cli.main:app"
ss = "shuttersift.cli.main:app"
```

Both resolve to the same Typer app.

## 2. Command Structure

| Command | Description | Visibility |
|---------|-------------|------------|
| `ss [scan] <dir>` | Analyze and sort photos | default (no subcommand needed) |
| `ss setup` | Download required model files | shown in --help |
| `ss info` | Show system capabilities | shown in --help |
| `ss calibrate <dir>` | Sample sharpness distribution | hidden (advanced) |

`scan` is registered as the default invoke target so `ss ./photos` is equivalent
to `ss scan ./photos`.

## 3. `scan` Flag Redesign

| Old flag | New flag | Short | Notes |
|----------|----------|-------|-------|
| `--explain` | `--explain` | `-e` | Enable VLM explanation for Review photos |
| `--dry-run` | `--dry-run` | `-n` | Analyze only, do not write files (unix: make -n) |
| `--fresh` | `--force` | `-f` | Ignore cache, reanalyze all photos |
| `--keep-threshold N` | `--keep N` | — | Score threshold for Keep bucket |
| `--reject-threshold N` | `--reject N` | — | Score threshold for Reject bucket |
| `--workers N` | `--jobs N` | `-j N` | Parallel workers (unix: make -j) |
| `--verbose` | `--verbose` | `-v` | Verbose logging to stderr |
| `--output` | `--output` | `-o` | Output directory |
| `--config` | `--config` | `-c` | Path to config.yaml |
| *(new)* | `--recalibrate` | — | Force redo sharpness calibration |

`-h` / `--help` both supported (add `context_settings={"help_option_names": ["-h", "--help"]}`
to the Typer app).

## 4. Auto-calibration Behavior

On first `ss scan` invocation (no calibration record in `~/.shuttersift/config.yaml`):

1. Sample up to 300 photos from the input directory
2. Compute Laplacian variance distribution
3. Use p10 as `hard_reject_sharpness`
4. Write result to `~/.shuttersift/config.yaml`
5. Continue with analysis

On subsequent runs: skip calibration silently.

`--recalibrate` flag forces steps 1–4 regardless of existing config.

Output during calibration:
```
[2/3] Calibrating sharpness thresholds...
      Sampling 150 photos  ████████████████████  done
      p10 = 42.3  →  saved to ~/.shuttersift/config.yaml  ✓
```

The `calibrate` subcommand remains available for advanced users who want to
inspect the full percentile distribution. It is hidden from `--help` output.

## 5. `setup` Command Output

Structured, readable, with explicit next-step guidance:

```
ShutterSift Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/2] MediaPipe face landmarker
      Downloading...  (12.4 MB)
      ✓ Saved to ~/.shuttersift/models/face_landmarker.task

[2/2] Done.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next steps:
  ss ./photos          — start culling
  ss setup --vlm       — also download local VLM model (~1.7 GB)
  ss info              — check system capabilities
```

With `--vlm`:
```
[2/2] moondream2 GGUF model
      Downloading...  (1.7 GB)  ████████████░░░░  73%
      ✓ Saved to ~/.shuttersift/models/moondream2.gguf

      Tip: run  ss ./photos -e  to enable VLM explanations
```

## 6. Config File Discovery (Bug Fix)

Current `Config.load()` checks only `~/.shuttersift/config.yaml` as fallback.
README claims current directory is also checked. Fix: add `./shuttersift.yaml`
and `./config.yaml` to the discovery chain, in order:

1. Explicit `--config` path
2. `./shuttersift.yaml`
3. `./config.yaml`
4. `~/.shuttersift/config.yaml`
5. Built-in defaults

## 7. README Structure

Both `README.md` (English) and `README.zh.md` (Chinese) follow this structure:

```
# ShutterSift
Tagline + one-liner command

## Install
  pip / wheel / git+https
  Apple Silicon note
  Metal GPU note

## Quick Start
  Step 1: ss setup
  Step 2: ss ./photos
  Step 3: view report.html

## Output
  Directory tree
  Lightroom import note

## Common Options
  Flag quick-reference table

## VLM Explanation
  API key (Anthropic / OpenAI)
  Local GGUF (offline)

## Scoring
  Weight table + threshold explanation

## Advanced
  config.yaml full reference
  calibrate command
  GUI roadmap
```

Rules:
- No YAML or config content before the Advanced section
- `calibrate` not mentioned before Advanced section
- Quick Start must work copy-paste with zero prior knowledge
- Both files kept in sync

---

## Implementation Scope

### Code changes (`src/shuttersift/`)

1. `cli/main.py`
   - Rename `cull` → `scan`, register as default invoke target
   - Add `-h` support via `context_settings`
   - Rename flags per section 3
   - Inject auto-calibration step into `scan` flow
   - Rename `download-models` → `setup` with new output format
   - Hide `calibrate` from help

2. `config.py`
   - Extend `Config.load()` discovery chain (section 6)
   - Add `calibrated: bool` field to track whether calibration has run

3. `pyproject.toml`
   - Add `ss` entry point

### Documentation

4. `README.md` — full English rewrite
5. `README.zh.md` — full Chinese version
