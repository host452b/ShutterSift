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


# Custom group that routes bare paths/flags to the `scan` subcommand so that
# `ss ./photos -n` works just like `ss scan ./photos -n`.
class _DefaultToScan(typer.core.TyperGroup):
    _SUBCOMMANDS = {"scan", "setup", "info", "calibrate", "--help", "-h", "--version"}

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:  # type: ignore[override]
        non_opt = [a for a in args if not a.startswith("-")]
        if non_opt and non_opt[0] not in self._SUBCOMMANDS:
            args = ["scan"] + list(args)
        return super().parse_args(ctx, args)


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
