from __future__ import annotations
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.table import Table
from rich.rule import Rule

from shuttersift import __version__
from shuttersift.config import Config
from shuttersift.engine import AnalysisResult, PhotoResult
from shuttersift.engine.pipeline import Engine
from shuttersift.engine.capabilities import Capabilities


# Custom group that routes bare paths/flags to the `scan` subcommand so that
# `ss ./photos -n` works just like `ss scan ./photos -n`.
class _DefaultToScan(typer.core.TyperGroup):
    _SUBCOMMANDS = {"scan", "setup", "info", "calibrate", "--help", "-h", "--version"}

    # Requires typer >= 0.9 / click >= 8.0 for TyperGroup.parse_args(ctx, args) signature.
    def parse_args(self, ctx, args):  # type: ignore[override]
        non_opt = [a for a in args if not a.startswith("-")]
        if non_opt and non_opt[0] not in self._SUBCOMMANDS:
            args = ["scan"] + list(args)
        try:
            return super().parse_args(ctx, args)
        except TypeError:
            # Fallback for Click versions with a different parse_args signature
            return super().parse_args(ctx)


# Both -h and --help work
app = typer.Typer(
    name="shuttersift",
    help="AI-powered photo culling CLI. One command to sort your shots.",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    cls=_DefaultToScan,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    log_dir = Path.home() / ".shuttersift" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
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

    if len(variances) < 5:
        console.print("[yellow]Too few readable photos for calibration; using default thresholds.[/]")
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

    caps = Capabilities.detect()
    console.print(f"\n[bold]ShutterSift[/] v{__version__}")
    console.print(f"Detected: {caps.summary()}\n")

    if not caps.gpu:
        console.print(
            "[yellow]⚠  No GPU detected — running on CPU.[/]  "
            "MUSIQ aesthetic scoring will use BRISQUE fallback and analysis will be slower.\n"
            "[yellow]   To enable GPU:[/] install a CUDA-enabled torch (Windows/Linux) "
            "or ensure Metal is available (macOS).\n"
        )

    # Auto-calibration: run on first use or when --recalibrate is passed
    if not cfg.calibrated or recalibrate:
        console.print("[1/3] Detecting capabilities...  ✓")
        label = "Recalibrating" if recalibrate else "Calibrating"
        console.print(f"[2/3] {label} sharpness thresholds...")
        cfg = _run_auto_calibration(input_dir, cfg)
        step_prefix = "[3/3]"
    else:
        step_prefix = "[1/1]"

    console.print(f"{step_prefix} Analyzing photos...\n")

    engine = Engine(cfg)

    _DECISION_STYLE = {"keep": "green", "review": "yellow", "reject": "red"}

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        TextColumn("· ETA"),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    ) as progress:
        task_id = progress.add_task("[dim]waiting…[/]", total=None)

        def on_progress(current: int, total: int, result: PhotoResult) -> None:
            style = _DECISION_STYLE.get(result.decision, "white")
            name = result.path.name
            if len(name) > 28:
                name = "…" + name[-27:]
            desc = f"[cyan]{name}[/]  [[{style}]{result.decision}[/]]"
            progress.update(task_id, completed=current, total=total, description=desc)

        import time as _time
        _t0 = _time.perf_counter()
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
            logging.getLogger(__name__).exception("Engine error")
            console.print(f"[red]Error:[/] {exc or type(exc).__name__}")
            raise typer.Exit(1)
        _elapsed = _time.perf_counter() - _t0

    _print_summary(result, output_dir, dry_run, _elapsed)


_MAX_LIST_ROWS = 25  # max filenames shown per bucket before truncating


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _print_bucket(title: str, style: str, photos: list) -> None:
    if not photos:
        return
    console.print(f"\n[bold {style}]{title}[/]  ({len(photos)} photos)")
    t = Table(box=None, show_header=False, padding=(0, 1))
    shown = photos[:_MAX_LIST_ROWS]
    for p in shown:
        reasons = ", ".join(p.reasons) if p.reasons else ""
        reason_text = f"[dim]— {reasons}[/]" if reasons else ""
        t.add_row(
            f"  [cyan]{p.path.name}[/]",
            f"[{style}]{p.score:.0f}[/]",
            reason_text,
        )
    if len(photos) > _MAX_LIST_ROWS:
        t.add_row(f"  [dim]… and {len(photos) - _MAX_LIST_ROWS} more[/]", "", "")
    console.print(t)


def _print_summary(result: AnalysisResult, output_dir: Path, dry_run: bool, elapsed_s: float = 0.0) -> None:
    total = len(result.photos)
    if total == 0:
        console.print("[yellow]No photos found.[/]")
        return

    # ── Timing ────────────────────────────────────────────────────────────────
    avg_ms = (elapsed_s * 1000 / total) if total else 0.0
    elapsed_str = _fmt_elapsed(elapsed_s)

    # ── Counts table ──────────────────────────────────────────────────────────
    console.rule()
    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_row("[green]✓  Keep[/]",    str(len(result.keep)),   f"({len(result.keep)/total:.0%})")
    tbl.add_row("[yellow]◎  Review[/]", str(len(result.review)), f"({len(result.review)/total:.0%})")
    tbl.add_row("[red]✗  Reject[/]",   str(len(result.reject)), f"({len(result.reject)/total:.0%})")
    tbl.add_row("", "", "")
    tbl.add_row("[dim]⏱  Time[/]", f"[dim]{elapsed_str}[/]",
                f"[dim](avg {avg_ms:.0f} ms/photo)[/]")
    console.print(tbl)
    console.rule()

    # ── Per-bucket file listings ───────────────────────────────────────────────
    _print_bucket("✓  Keep",   "green",  result.keep)
    _print_bucket("◎  Review", "yellow", result.review)
    _print_bucket("✗  Reject", "red",    result.reject)

    # ── Paths ─────────────────────────────────────────────────────────────────
    console.print()
    if not dry_run:
        console.print(f"Output  → [bold]{output_dir}[/]")
        console.print(f"Report  → [bold]{output_dir / 'report.html'}[/]\n")
    else:
        console.print("[yellow]Dry run — no files written[/]\n")


# ── Default command callback (show help when no args given) ───────────────────

@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Scan a directory of photos and sort into Keep / Review / Reject.

    When called without a subcommand, runs the scan directly:

        ss ./photos
    """
    if ctx.invoked_subcommand is not None:
        return
    console.print(ctx.get_help())
    raise typer.Exit(0)


# ── scan subcommand: ss scan ./photos  OR  ss ./photos ────────────────────────

@app.command(name="scan")
def scan(
    input_dir: Path = typer.Argument(..., help="Directory of photos to scan"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory"),
    config: Optional[Path] = typer.Option(None, "-c", "--config", help="Path to config file"),
    explain: bool = typer.Option(False, "-e", "--explain", help="Enable VLM explanation for Review photos"),
    dry_run: bool = typer.Option(False, "-n", "--dry-run", help="Analyze only, do not write files"),
    force: bool = typer.Option(False, "-f", "--force", help="Ignore cache, reanalyze all photos"),
    recalibrate: bool = typer.Option(False, "--recalibrate", help="Force redo sharpness threshold sampling"),
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
    from shuttersift.engine.downloader import MODEL_REGISTRY
    size = MODEL_REGISTRY["mediapipe_face_landmarker"]["size_hint"]
    console.print(f"      Downloading...  [dim]({size})[/]")
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

    _gpu_detail = {"cuda": "CUDA", "mps": "Apple Metal (MPS)", "cpu": "none"}.get(caps.gpu_device, caps.gpu_device)
    table.add_row("GPU", "[green]✓[/]" if caps.gpu else "[red]✗[/]", _gpu_detail)
    table.add_row("RAW decode", "[green]✓[/]" if caps.rawpy  else "[yellow]~[/]", "rawpy" if caps.rawpy else "Using Pillow fallback")
    table.add_row("MUSIQ",      "[green]✓[/]" if caps.musiq  else "[yellow]~[/]", "GPU aesthetic scoring" if caps.musiq else "BRISQUE fallback")
    table.add_row("Local VLM", "[green]✓[/]" if caps.gguf_vlm else "[red]✗[/]",
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
