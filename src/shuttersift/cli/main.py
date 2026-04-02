from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

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

        try:
            result: AnalysisResult = engine.analyze(
                input_dir=input_dir,
                output_dir=output_dir,
                on_progress=on_progress,
                resume=not fresh,
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
        raise typer.Exit(1)

    if vlm:
        console.print("Downloading moondream2 GGUF (~1.7 GB)...")
        ok = download_gguf_vlm()
        if ok:
            console.print("[green]✓[/] GGUF VLM ready")
        else:
            console.print("[red]✗[/] GGUF download failed")
            raise typer.Exit(1)


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
