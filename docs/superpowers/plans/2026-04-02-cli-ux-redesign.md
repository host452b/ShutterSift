# CLI UX Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign ShutterSift's CLI so `ss ./photos` works out-of-the-box with intuitive unix-style flags, auto-calibration, and an improved setup experience; then rewrite README in English and Chinese.

**Architecture:** A shared `_do_scan()` implementation function is called by both the default Typer callback (handles `ss ./photos`) and the explicit `scan` subcommand (handles `ss scan ./photos`). Config discovery is extended to check the current directory. Auto-calibration runs transparently on first use.

**Tech Stack:** Python 3.11+, Typer ≥0.12, Rich ≥13.8, Pydantic ≥2, PyYAML, pytest + typer.testing.CliRunner

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | Modify | Add `ss` entry point |
| `src/shuttersift/config.py` | Modify | Extended discovery chain; calibration state tracking |
| `src/shuttersift/cli/main.py` | Rewrite | New flag names, default command, auto-calibration, setup output |
| `tests/unit/test_config.py` | Modify | Add discovery chain tests |
| `tests/unit/test_cli.py` | Create | CLI flag + default command tests |
| `README.md` | Rewrite | English README |
| `README.zh.md` | Create | Chinese README |

---

## Task 1: Add `ss` entry point

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `ss` to `[project.scripts]`**

Open `pyproject.toml` and change the `[project.scripts]` block to:

```toml
[project.scripts]
shuttersift = "shuttersift.cli.main:app"
ss = "shuttersift.cli.main:app"
```

- [ ] **Step 2: Reinstall in dev mode and verify**

```bash
cd /localhome/swqa/workspace/photo_steth
pip install -e . -q
ss --help
```

Expected: help output printed with `ss` as the program name (typer uses argv[0], so it may still show `shuttersift` — that is fine).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add ss entry point alias"
```

---

## Task 2: Extend `Config` discovery chain and add calibration tracking

**Files:**
- Modify: `src/shuttersift/config.py`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing tests for new discovery chain**

Add to `tests/unit/test_config.py`:

```python
from pathlib import Path
import os

def test_config_load_from_cwd_shuttersift_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "shuttersift.yaml"
    cfg_file.write_text("scoring:\n  thresholds:\n    keep: 85\n")
    cfg = Config.load()
    assert cfg.thresholds.keep == 85

def test_config_load_from_cwd_config_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("scoring:\n  thresholds:\n    keep: 77\n")
    cfg = Config.load()
    assert cfg.thresholds.keep == 77

def test_config_load_explicit_path_takes_priority(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("scoring:\n  thresholds:\n    keep: 77\n")
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("scoring:\n  thresholds:\n    keep: 99\n")
    cfg = Config.load(explicit)
    assert cfg.thresholds.keep == 99

def test_config_calibrated_defaults_false():
    cfg = Config()
    assert cfg.calibrated is False

def test_config_calibrated_round_trips_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("calibrated: true\nscoring:\n  thresholds:\n    hard_reject_sharpness: 42.3\n")
    cfg = Config.from_yaml(cfg_file)
    assert cfg.calibrated is True
    assert cfg.thresholds.hard_reject_sharpness == 42.3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/test_config.py -v -k "discovery or calibrated"
```

Expected: FAIL — `Config` has no `calibrated` field; `Config.load()` does not check cwd.

- [ ] **Step 3: Implement extended discovery chain and `calibrated` field**

Replace `src/shuttersift/config.py` with:

```python
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, model_validator, ConfigDict


class ScoringWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    model_config = ConfigDict(extra="forbid")
    keep: int = 70
    reject: int = 40
    hard_reject_sharpness: float = 30.0
    eye_open_min: float = 0.25
    burst_gap_seconds: float = 2.0


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weights: ScoringWeights = ScoringWeights()
    thresholds: Thresholds = Thresholds()
    workers: int = 4
    log_retention_runs: int = 30
    api_model_anthropic: str = "claude-haiku-4-5-20251001"
    api_model_openai: str = "gpt-4o-mini"
    calibrated: bool = False  # set True after auto-calibration runs

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        scoring = data.pop("scoring", {})
        if "weights" in scoring:
            data["weights"] = scoring["weights"]
        if "thresholds" in scoring:
            data["thresholds"] = scoring["thresholds"]
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Load config from first found location:
        1. explicit --config path
        2. ./shuttersift.yaml
        3. ./config.yaml
        4. ~/.shuttersift/config.yaml
        5. built-in defaults
        """
        candidates: list[Path] = []
        if path:
            candidates.append(path)
        candidates += [
            Path.cwd() / "shuttersift.yaml",
            Path.cwd() / "config.yaml",
            Path.home() / ".shuttersift" / "config.yaml",
        ]
        for p in candidates:
            if p.exists():
                return cls.from_yaml(p)
        return cls()

    def save_to_user_config(self) -> Path:
        """Persist this config to ~/.shuttersift/config.yaml and return the path."""
        import yaml as _yaml
        dest = Path.home() / ".shuttersift" / "config.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "calibrated": self.calibrated,
            "scoring": {
                "thresholds": {
                    "keep": self.thresholds.keep,
                    "reject": self.thresholds.reject,
                    "hard_reject_sharpness": self.thresholds.hard_reject_sharpness,
                    "eye_open_min": self.thresholds.eye_open_min,
                    "burst_gap_seconds": self.thresholds.burst_gap_seconds,
                }
            },
            "workers": self.workers,
            "log_retention_runs": self.log_retention_runs,
        }
        dest.write_text(_yaml.dump(data, default_flow_style=False))
        return dest
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_config.py -v
```

Expected: all tests PASS (including the 5 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/shuttersift/config.py tests/unit/test_config.py
git commit -m "feat: extend config discovery chain and add calibration tracking"
```

---

## Task 3: Create CLI test scaffolding

**Files:**
- Create: `tests/unit/test_cli.py`

- [ ] **Step 1: Write tests for `-h`, default command, and renamed flags**

Create `tests/unit/test_cli.py`:

```python
"""Tests for CLI surface: flags, -h support, default command routing."""
from pathlib import Path
import pytest
from typer.testing import CliRunner
from shuttersift.cli.main import app

runner = CliRunner()


def test_h_flag_shows_help():
    result = runner.invoke(app, ["-h"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_help_flag_shows_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_scan_h_flag():
    result = runner.invoke(app, ["scan", "-h"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_default_command_requires_directory(tmp_path):
    """ss with no args shows help, does not crash."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_default_command_rejects_missing_dir(tmp_path):
    missing = tmp_path / "nonexistent"
    result = runner.invoke(app, [str(missing)])
    assert result.exit_code != 0
    assert "not a directory" in result.output.lower() or result.exit_code == 1


def test_scan_dry_run_short_flag(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    result = runner.invoke(app, [str(photos), "-n"])
    # dry-run should complete without error (0 photos found is ok)
    assert result.exit_code == 0


def test_scan_n_flag_equivalent_to_dry_run(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    r1 = runner.invoke(app, [str(photos), "-n"])
    r2 = runner.invoke(app, [str(photos), "--dry-run"])
    assert r1.exit_code == r2.exit_code


def test_scan_keep_flag(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    result = runner.invoke(app, [str(photos), "--keep", "80", "-n"])
    assert result.exit_code == 0


def test_scan_reject_flag(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    result = runner.invoke(app, [str(photos), "--reject", "30", "-n"])
    assert result.exit_code == 0


def test_scan_jobs_short_flag(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    result = runner.invoke(app, [str(photos), "-j", "2", "-n"])
    assert result.exit_code == 0


def test_setup_h_shows_help():
    result = runner.invoke(app, ["setup", "-h"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_info_command():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0


def test_calibrate_hidden_from_top_level_help():
    result = runner.invoke(app, ["--help"])
    assert "calibrate" not in result.output


def test_force_flag_accepted(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    result = runner.invoke(app, [str(photos), "-f", "-n"])
    assert result.exit_code == 0


def test_old_fresh_flag_rejected(tmp_path):
    """--fresh must no longer be a valid flag."""
    photos = tmp_path / "photos"
    photos.mkdir()
    result = runner.invoke(app, [str(photos), "--fresh"])
    assert result.exit_code != 0


def test_old_keep_threshold_rejected(tmp_path):
    """--keep-threshold must no longer be a valid flag."""
    photos = tmp_path / "photos"
    photos.mkdir()
    result = runner.invoke(app, [str(photos), "--keep-threshold", "70"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail (or partially fail)**

```bash
pytest tests/unit/test_cli.py -v
```

Expected: multiple FAILs — old flags still present, `-h` not supported, `scan` subcommand not yet renamed.

- [ ] **Step 3: Commit test file (red state)**

```bash
git add tests/unit/test_cli.py
git commit -m "test: add CLI surface tests (red)"
```

---

## Task 4: Rewrite `cli/main.py` — flags, default command, `scan`

**Files:**
- Rewrite: `src/shuttersift/cli/main.py`

This is the largest task. Replace the entire file.

- [ ] **Step 1: Write the new `cli/main.py`**

```python
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.rule import Rule

from shuttersift import __version__
from shuttersift.config import Config
from shuttersift.engine import AnalysisResult, PhotoResult
from shuttersift.engine.pipeline import Engine
from shuttersift.engine.capabilities import Capabilities

# -h and --help both work
app = typer.Typer(
    name="shuttersift",
    help="AI-powered photo culling CLI. One command to sort your shots.",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
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
    cfg = Config.load()
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    while len(logs) > cfg.log_retention_runs:
        logs.pop(0).unlink(missing_ok=True)


def _run_auto_calibration(input_dir: Path, cfg: Config) -> Config:
    """Sample input_dir, compute p10 sharpness, persist to user config. Returns updated cfg."""
    from shuttersift.engine.loader import load_image, SUPPORTED_FORMATS
    from shuttersift.engine.analyzers.sharpness import laplacian_variance

    paths = [p for p in sorted(input_dir.rglob("*"))
             if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS]
    sample = paths[:300]

    variances = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  console=console, transient=True) as prog:
        t = prog.add_task("      Sampling photos...", total=len(sample))
        for p in sample:
            img = load_image(p)
            if img is not None:
                variances.append(laplacian_variance(img))
            prog.advance(t)

    if not variances:
        return cfg

    variances.sort()
    p10 = variances[max(0, int(len(variances) * 0.10))]
    cfg.thresholds.hard_reject_sharpness = p10
    cfg.calibrated = True
    dest = cfg.save_to_user_config()
    console.print(f"      p10 = [cyan]{p10:.1f}[/]  →  saved to [dim]{dest}[/]  [green]✓[/]")
    return cfg


def _do_scan(
    input_dir: Path,
    output: Optional[Path],
    config: Optional[Path],
    explain: bool,
    dry_run: bool,
    force: bool,
    recalibrate: bool,
    keep: Optional[int],
    reject: Optional[int],
    jobs: Optional[int],
    verbose: bool,
) -> None:
    """Shared implementation for both the default callback and the `scan` subcommand."""
    _setup_logging(verbose)

    if not input_dir.is_dir():
        console.print(f"[red]Error:[/] {input_dir} is not a directory")
        raise typer.Exit(1)

    cfg = Config.load(config)
    if keep is not None:
        cfg.thresholds.keep = keep
    if reject is not None:
        cfg.thresholds.reject = reject
    if jobs is not None:
        cfg.workers = jobs

    output_dir = output or (input_dir.parent / "shuttersift_output")

    caps = Capabilities.detect()
    console.print(f"\n[bold]ShutterSift[/] v{__version__}")
    console.print(f"Detected: {caps.summary()}\n")

    # Auto-calibration: run on first use or when --recalibrate is passed
    if not cfg.calibrated or recalibrate:
        label = "Recalibrating" if recalibrate else "Calibrating"
        console.print(f"[2/3] {label} sharpness thresholds...")
        cfg = _run_auto_calibration(input_dir, cfg)
        step_prefix = "[3/3]"
    else:
        step_prefix = "[1/1]"

    console.print(f"{step_prefix} Analyzing photos...\n")

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

        try:
            result: AnalysisResult = engine.analyze(
                input_dir=input_dir,
                output_dir=output_dir,
                on_progress=on_progress,
                resume=not force,
                dry_run=dry_run,
                explain=explain,
            )
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(1)

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


# ── Default command (ss ./photos) ────────────────────────────────────────────

@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    input_dir: Optional[Path] = typer.Argument(None, help="Directory of photos to scan"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory"),
    config: Optional[Path] = typer.Option(None, "-c", "--config", help="Path to config file"),
    explain: bool = typer.Option(False, "-e", "--explain", help="Enable VLM explanation for Review photos"),
    dry_run: bool = typer.Option(False, "-n", "--dry-run", help="Analyze only, do not write files"),
    force: bool = typer.Option(False, "-f", "--force", help="Ignore cache, reanalyze all photos"),
    recalibrate: bool = typer.Option(False, "--recalibrate", help="Force redo sharpness calibration"),
    keep: Optional[int] = typer.Option(None, "--keep", help="Score threshold for Keep bucket (default 70)"),
    reject: Optional[int] = typer.Option(None, "--reject", help="Score threshold for Reject bucket (default 40)"),
    jobs: Optional[int] = typer.Option(None, "-j", "--jobs", help="Parallel workers"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Scan a directory of photos and sort into Keep / Review / Reject.

    When called without a subcommand, runs the scan directly:

        ss ./photos
    """
    if ctx.invoked_subcommand is not None:
        return
    if input_dir is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)
    _do_scan(input_dir, output, config, explain, dry_run, force,
             recalibrate, keep, reject, jobs, verbose)


# ── scan subcommand (explicit: ss scan ./photos) ──────────────────────────────

@app.command(name="scan")
def scan(
    input_dir: Path = typer.Argument(..., help="Directory of photos to scan"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory"),
    config: Optional[Path] = typer.Option(None, "-c", "--config", help="Path to config file"),
    explain: bool = typer.Option(False, "-e", "--explain", help="Enable VLM explanation for Review photos"),
    dry_run: bool = typer.Option(False, "-n", "--dry-run", help="Analyze only, do not write files"),
    force: bool = typer.Option(False, "-f", "--force", help="Ignore cache, reanalyze all photos"),
    recalibrate: bool = typer.Option(False, "--recalibrate", help="Force redo sharpness calibration"),
    keep: Optional[int] = typer.Option(None, "--keep", help="Score threshold for Keep bucket (default 70)"),
    reject: Optional[int] = typer.Option(None, "--reject", help="Score threshold for Reject bucket (default 40)"),
    jobs: Optional[int] = typer.Option(None, "-j", "--jobs", help="Parallel workers"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Scan a directory of photos and sort into Keep / Review / Reject."""
    _do_scan(input_dir, output, config, explain, dry_run, force,
             recalibrate, keep, reject, jobs, verbose)


# ── setup ─────────────────────────────────────────────────────────────────────

@app.command()
def setup(
    vlm: bool = typer.Option(False, "--vlm", help="Also download local VLM model (~1.7 GB)"),
) -> None:
    """Download required model files to ~/.shuttersift/models/."""
    from shuttersift.engine.downloader import download_mediapipe_models, download_gguf_vlm

    console.print(f"\n[bold]ShutterSift Setup[/]")
    console.print(Rule())

    total_steps = 2 if vlm else 1
    step = 1

    console.print(f"\n[{step}/{total_steps}] MediaPipe face landmarker")
    console.print("      Downloading...  [dim](12.4 MB)[/]")
    ok = download_mediapipe_models()
    if ok:
        dest = Path.home() / ".shuttersift" / "models" / "face_landmarker.task"
        console.print(f"      [green]✓[/] Saved to [dim]{dest}[/]")
    else:
        console.print("      [red]✗[/] MediaPipe download failed")
        raise typer.Exit(1)

    if vlm:
        step += 1
        console.print(f"\n[{step}/{total_steps}] moondream2 GGUF model")
        console.print("      Downloading...  [dim](~1.7 GB)[/]")
        ok = download_gguf_vlm()
        if ok:
            dest = Path.home() / ".shuttersift" / "models" / "moondream2.gguf"
            console.print(f"      [green]✓[/] Saved to [dim]{dest}[/]")
            console.print(f"\n      [dim]Tip: run[/]  [bold]ss ./photos -e[/]  [dim]to enable VLM explanations[/]")
        else:
            console.print("      [red]✗[/] GGUF download failed")
            raise typer.Exit(1)

    console.print(f"\n[{total_steps}/{total_steps}] Done.\n")
    console.print(Rule())
    console.print("\nNext steps:")
    console.print("  [bold]ss ./photos[/]          — start scanning")
    if not vlm:
        console.print("  [bold]ss setup --vlm[/]       — also download local VLM model (~1.7 GB)")
    console.print("  [bold]ss info[/]              — check system capabilities\n")


# ── info ──────────────────────────────────────────────────────────────────────

@app.command()
def info() -> None:
    """Show detected capabilities (GPU, VLM, RAW support)."""
    caps = Capabilities.detect()
    console.print(f"\n[bold]ShutterSift[/] v{__version__}\n")

    table = Table(title="Capabilities", show_header=True)
    table.add_column("Feature", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    table.add_row("GPU",        "[green]✓[/]" if caps.gpu    else "[red]✗[/]", "CUDA or Apple Metal")
    table.add_row("RAW decode", "[green]✓[/]" if caps.rawpy  else "[yellow]~[/]", "rawpy" if caps.rawpy else "Using Pillow fallback")
    table.add_row("MUSIQ",      "[green]✓[/]" if caps.musiq  else "[yellow]~[/]", "GPU aesthetic scoring" if caps.musiq else "BRISQUE fallback")
    table.add_row("GGUF VLM",  "[green]✓[/]" if caps.gguf_vlm else "[red]✗[/]",
                  str(caps.gguf_model_path) if caps.gguf_model_path else "Run: ss setup --vlm")
    table.add_row("API VLM",   "[green]✓[/]" if caps.api_vlm else "[red]✗[/]",
                  "ANTHROPIC_API_KEY or OPENAI_API_KEY set" if caps.api_vlm else "Set ANTHROPIC_API_KEY env var")
    console.print(table)
    console.print()


# ── calibrate (hidden — advanced users only) ──────────────────────────────────

@app.command(hidden=True)
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
    table.add_column("Percentile")
    table.add_column("Laplacian Variance")
    table.add_column("Recommendation")
    table.add_row("p5",  f"{p5:.1f}",  "← hard_reject_sharpness (aggressive)")
    table.add_row("p10", f"{p10:.1f}", "← hard_reject_sharpness (recommended)")
    table.add_row("p25", f"{p25:.1f}", "← hard_reject_sharpness (conservative)")
    table.add_row("p50", f"{p50:.1f}", "Median of your photos")
    console.print(table)

    cfg = Config.load()
    cfg.thresholds.hard_reject_sharpness = p10
    cfg.calibrated = True
    dest = cfg.save_to_user_config()
    console.print(f"\n[green]✓[/] Saved hard_reject_sharpness = {p10:.1f} to [dim]{dest}[/]\n")
```

- [ ] **Step 2: Run the CLI tests**

```bash
pytest tests/unit/test_cli.py -v
```

Expected: most tests PASS. If any FAIL, read the error and fix before continuing.

- [ ] **Step 3: Run full unit test suite to check for regressions**

```bash
pytest tests/unit/ -v --tb=short
```

Expected: all existing tests PASS (only CLI surface changed).

- [ ] **Step 4: Commit**

```bash
git add src/shuttersift/cli/main.py
git commit -m "feat: rename cull→scan, add ss default command, redesign flags"
```

---

## Task 5: Write `README.md` (English)

**Files:**
- Rewrite: `README.md`

- [ ] **Step 1: Write the English README**

Replace `README.md` with:

````markdown
# ShutterSift

AI-powered photo culling CLI. Drop a folder, get Keep / Review / Reject.

```bash
ss ./photos
```

Automatically classifies every photo with a 0–100 score across five quality
dimensions. Results land in organized folders with an interactive HTML report.

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

---

## Quick Start

```bash
# Step 1 — download required model files (one time only)
ss setup

# Step 2 — scan your photos
ss ./photos

# Step 3 — open the report
open shuttersift_output/report.html     # macOS
xdg-open shuttersift_output/report.html # Linux
```

Sharpness thresholds are calibrated automatically on first run based on your
photo library. No configuration needed.

Both `ss` and `shuttersift` are valid command names — use whichever you prefer.

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

When `--explain` / `-e` is passed, borderline **Review** photos get an
AI-generated text description explaining what's wrong or interesting. Typically
20–30% of photos land in Review.

**Cloud API (Anthropic or OpenAI):**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
ss ./photos -e
```
or
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
````

- [ ] **Step 2: Verify the README renders correctly**

```bash
# Quick sanity check — no broken markdown fences
python3 -c "
import re, pathlib
content = pathlib.Path('README.md').read_text()
fences = re.findall(r'^\`\`\`', content, re.MULTILINE)
print(f'Code fences: {len(fences)} (should be even)')
assert len(fences) % 2 == 0, 'Unmatched code fence!'
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README.md in English"
```

---

## Task 6: Write `README.zh.md` (Chinese)

**Files:**
- Create: `README.zh.md`

- [ ] **Step 1: Write the Chinese README**

Create `README.zh.md`:

````markdown
# ShutterSift

AI 驱动的照片筛选工具。丢一个文件夹进去，自动分出「保留 / 待定 / 淘汰」。

```bash
ss ./photos
```

每张照片都会获得一个 0–100 的综合评分，覆盖五个质量维度。结果自动整理进
子目录，并生成可交互的 HTML 报告。

---

## 安装

**从 PyPI 安装**（发布后可用）：
```bash
pip install shuttersift
```

**从 GitHub Release 安装**（现在可用）：
```bash
pip install https://github.com/host452b/ShutterSift/releases/download/v0.1.0/shuttersift-0.1.0-py3-none-any.whl
```

**从源码安装**：
```bash
pip install git+https://github.com/host452b/ShutterSift.git
```

> **Apple Silicon（M1/M2/M3）**：如果 `mediapipe` 安装失败，请改用：
> ```bash
> pip install mediapipe-silicon
> pip install shuttersift --no-deps
> ```

> **Mac GPU 加速（Metal）**：
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
> ```

---

## 快速开始

```bash
# 第一步 — 下载所需模型文件（只需一次）
ss setup

# 第二步 — 扫描你的照片
ss ./photos

# 第三步 — 查看报告
open shuttersift_output/report.html      # macOS
xdg-open shuttersift_output/report.html  # Linux
```

首次运行时会自动根据你的照片库校准锐度阈值，**无需任何手动配置**。

`ss` 和 `shuttersift` 都是合法的命令名，两者等价，用哪个都行。

---

## 输出结构

```
shuttersift_output/
├── keep/         ← 高分照片（符号链接 + XMP 附属文件）
├── review/       ← 边界照片，值得二次确认
├── reject/       ← 模糊、闭眼、重复照片
├── report.html   ← 带评分和缩略图的可交互报告
└── results.json  ← 机器可读，带版本号
```

**Lightroom 用户**：直接导入 `keep/` 文件夹，XMP 附属文件会自动设置星级和色标。

---

## 常用参数

```bash
ss ./photos -e               # 对「待定」照片启用 VLM 解释
ss ./photos -n               # 演习模式——只分析，不写入文件
ss ./photos -f               # 强制重新分析（忽略缓存）
ss ./photos --recalibrate    # 对当前图库重新校准锐度
ss ./photos -o ./sorted      # 指定输出目录
ss ./photos --keep 75 --reject 35   # 自定义评分阈值
ss ./photos -j 8             # 使用 8 个并行 worker
ss ./photos -v               # 详细日志输出
```

| 短参数 | 长参数 | 默认值 | 说明 |
|--------|--------|--------|------|
| `-e` | `--explain` | 关 | 对边界照片启用 VLM 文字解释 |
| `-n` | `--dry-run` | 关 | 只分析，不移动或链接文件 |
| `-f` | `--force` | 关 | 忽略缓存，重新分析所有照片 |
| `-o` | `--output <目录>` | `../shuttersift_output` | 输出目录 |
| `-j` | `--jobs <n>` | 4 | 并行 worker 数量 |
| `-v` | `--verbose` | 关 | 向 stderr 打印调试日志 |
| `-c` | `--config <文件>` | 自动 | 指定配置文件路径 |
| | `--keep <n>` | 70 | 保留所需最低分数 |
| | `--reject <n>` | 40 | 低于此分数直接淘汰 |
| | `--recalibrate` | 关 | 强制重新运行锐度校准 |

---

## VLM 解释功能

加上 `--explain` / `-e` 参数后，落在「待定」区间的边界照片会获得 AI 生成的
文字描述，解释问题所在或值得关注的点。通常有 20–30% 的照片落在待定区间。

**云端 API（Anthropic 或 OpenAI）**：
```bash
export ANTHROPIC_API_KEY=sk-ant-...
ss ./photos -e
```
或
```bash
export OPENAI_API_KEY=sk-...
ss ./photos -e
```

**完全本地运行（不联网）**：
```bash
ss setup --vlm        # 下载 moondream2 GGUF 模型，约 1.7 GB，仅需一次
ss ./photos -e        # 使用本地模型，无需 API Key
```

---

## 评分体系

每张照片获得一个 0–100 的综合评分：

| 维度 | 权重 | 方法 |
|------|------|------|
| 锐度 | 30% | Laplacian 方差 |
| 曝光 | 15% | 直方图分析 |
| 美学 | 25% | MUSIQ（GPU）/ BRISQUE（CPU）|
| 人脸质量 | 20% | MediaPipe 睁眼度 + 笑容检测 |
| 构图 | 10% | 三分法引擎 |

**默认阈值**：≥ 70 → 保留 · 40–69 → 待定 · < 40 → 淘汰

通过 `--keep` 和 `--reject` 自定义阈值。

---

## 高级用法

### 配置文件

ShutterSift 按以下顺序查找配置文件：
1. `--config <路径>`（显式指定）
2. `./shuttersift.yaml`（当前目录）
3. `./config.yaml`（当前目录）
4. `~/.shuttersift/config.yaml`（用户全局）

完整配置参考（`~/.shuttersift/config.yaml`）：

```yaml
calibrated: true               # 首次运行后自动设置

scoring:
  thresholds:
    keep: 70                   # 保留所需最低分数
    reject: 40                 # 低于此分数直接淘汰
    hard_reject_sharpness: 42.3  # 由自动校准写入
    eye_open_min: 0.25         # 最低睁眼比例（0–1）
    burst_gap_seconds: 2.0     # 此时间窗口内的连拍视为同一组

workers: 4                     # 并行 worker 数量
log_retention_runs: 30         # 保留的日志文件数量
```

### 手动校准

```bash
ss calibrate ./photos
```

采样最多 300 张照片，打印完整的 Laplacian 方差分布，
并将推荐的 `hard_reject_sharpness` 值保存到 `~/.shuttersift/config.yaml`。
切换相机或拍摄风格时使用。

### 其他命令

```bash
ss info          # 查看 GPU、VLM 和 RAW 支持状态
ss setup --vlm   # 下载本地 moondream2 VLM 模型
```

### GUI 客户端（规划中）

桌面端和 Web 端 GUI 客户端正在规划中。
参见 `clients/desktop/` 和 `clients/web/` 目录中的早期骨架。

---

## English README

[English Documentation →](README.md)
````

- [ ] **Step 2: Verify the Chinese README renders correctly**

```bash
python3 -c "
import re, pathlib
content = pathlib.Path('README.zh.md').read_text()
fences = re.findall(r'^\`\`\`', content, re.MULTILINE)
print(f'Code fences: {len(fences)} (should be even)')
assert len(fences) % 2 == 0, 'Unmatched code fence!'
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add README.zh.md
git commit -m "docs: add Chinese README (README.zh.md)"
```

---

## Task 7: Final verification

- [ ] **Step 1: Run full test suite**

```bash
cd /localhome/swqa/workspace/photo_steth
pytest tests/unit/ -v --tb=short
```

Expected: all tests PASS, no regressions.

- [ ] **Step 2: Smoke-test the CLI**

```bash
ss -h
ss --help
ss scan -h
ss setup -h
ss info
ss --help | grep -v calibrate   # calibrate must NOT appear
```

Expected: `-h` and `--help` both show help; `calibrate` is absent from top-level help.

- [ ] **Step 3: Verify default command**

```bash
mkdir /tmp/ss_test_empty
ss /tmp/ss_test_empty -n
```

Expected: exit 0, "No photos found." (dry-run, empty dir).

- [ ] **Step 4: Verify old flags are rejected**

```bash
ss /tmp/ss_test_empty --fresh 2>&1 | head -3
ss /tmp/ss_test_empty --keep-threshold 70 2>&1 | head -3
```

Expected: both exit non-zero with an "No such option" error.

- [ ] **Step 5: Commit final state**

```bash
git add -A
git status
git commit -m "feat: complete CLI UX redesign and README rewrite" --allow-empty
```
