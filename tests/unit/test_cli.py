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


def test_default_command_no_args_shows_help():
    """ss with no args shows help, does not crash."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_default_command_rejects_missing_dir(tmp_path):
    missing = tmp_path / "nonexistent"
    result = runner.invoke(app, [str(missing)])
    assert result.exit_code != 0


def test_scan_dry_run_short_flag(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    result = runner.invoke(app, [str(photos), "-n"])
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
    # The calibrate subcommand should not appear in top-level help
    # Note: --recalibrate flag may appear; we only care about the subcommand
    lines = result.output.splitlines()
    command_lines = [l for l in lines if l.strip().startswith("calibrate")]
    assert len(command_lines) == 0, "calibrate subcommand should be hidden"


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
