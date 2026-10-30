"""Tests for run-log tee (ANSI strip + carriage-return coalescing)."""

from __future__ import annotations

from pathlib import Path

from ccd_pipeline.runlog import (
    _LogWriter,
    run_log_path,
    start_run_log,
    stop_run_log,
    strip_ansi,
)


def test_strip_ansi():
    assert strip_ansi("\033[1;32mOK\033[0m") == "OK"
    assert strip_ansi("plain") == "plain"


def test_log_writer_collapses_carriage_returns(tmp_path: Path):
    path = tmp_path / "ccd_pipeline.log"
    w = _LogWriter(path)
    w.feed("\033[35m  → bias [1/3] a.fits\033[0m")
    w.feed("\r")
    w.feed("\033[35m  → bias [2/3] b.fits\033[0m")
    w.feed("\r")
    w.feed("  ✓ calibrated 3 bias frame(s)\n")
    w.close()
    text = path.read_text()
    assert "a.fits" not in text
    assert "b.fits" not in text
    assert "calibrated 3 bias frame(s)" in text


def test_start_run_log_creates_file(tmp_path: Path):
    cfg = {"paths": {"output_dir": tmp_path / "reduced", "masters_dir": tmp_path / "reduced" / "masters"}}
    stop_run_log()
    path = start_run_log(cfg, command="test")
    assert path == run_log_path(cfg)
    assert path.exists()
    print("hello from test")
    stop_run_log()
    body = path.read_text()
    assert "ccd-pipeline" in body
    assert "hello from test" in body
