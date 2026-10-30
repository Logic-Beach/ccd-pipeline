"""Mirror pipeline terminal output into ``<output_dir>/ccd_pipeline.log``.

Installs a stdout/stderr tee that:

* keeps the live terminal unchanged (colors, in-place ``\\r`` progress)
* appends a plain-text copy to the night reduced folder
* collapses carriage-return updates so the log keeps only the latest
  progress/spinner text per line (no spam)
"""

from __future__ import annotations

import atexit
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

# ESC[ … m  and other common CSI sequences used for color / cursor
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

_active: "RunLog | None" = None


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class _LogWriter:
    """Accumulate teed text; write completed lines (after ``\\n``) to a file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("a", encoding="utf-8")
        self._line = ""  # current incomplete line (after last \\r / start)

    def feed(self, raw: str) -> None:
        text = strip_ansi(raw)
        if not text:
            return
        for ch in text:
            if ch == "\r":
                # In-place update: discard what was on this line so far
                self._line = ""
            elif ch == "\n":
                self._fh.write(self._line.rstrip() + "\n")
                self._fh.flush()
                self._line = ""
            else:
                self._line += ch

    def write_direct(self, text: str) -> None:
        """Write a full line (or block) without going through the tee buffer."""
        if self._line:
            self._fh.write(self._line.rstrip() + "\n")
            self._line = ""
        self._fh.write(text)
        if text and not text.endswith("\n"):
            self._fh.write("\n")
        self._fh.flush()

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        if self._line.strip():
            self._fh.write(self._line.rstrip() + "\n")
            self._line = ""
        self._fh.flush()
        self._fh.close()


class _TeeStream:
    """Proxy that mirrors writes to the original stream and the log writer."""

    def __init__(self, original: TextIO, log: _LogWriter) -> None:
        self._original = original
        self._log = log

    def write(self, s: str) -> int:  # type: ignore[override]
        if not isinstance(s, str):
            s = str(s)
        self._original.write(s)
        self._log.feed(s)
        return len(s)

    def flush(self) -> None:
        self._original.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._original, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._original.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class RunLog:
    def __init__(self, path: Path, *, command: str | None = None) -> None:
        self.path = path
        self._log = _LogWriter(path)
        self._stdout_orig = sys.stdout
        self._stderr_orig = sys.stderr
        stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        header = f"\n===== ccd-pipeline  {stamp}"
        if command:
            header += f"  ({command})"
        header += " =====\n"
        self._log.write_direct(header)
        sys.stdout = _TeeStream(self._stdout_orig, self._log)  # type: ignore[assignment]
        sys.stderr = _TeeStream(self._stderr_orig, self._log)  # type: ignore[assignment]

    def close(self) -> None:
        sys.stdout = self._stdout_orig
        sys.stderr = self._stderr_orig
        self._log.close()


def run_log_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["output_dir"]) / "ccd_pipeline.log"


def start_run_log(cfg: dict, *, command: str | None = None) -> Path:
    """Begin mirroring stdout/stderr into ``<output_dir>/ccd_pipeline.log``.

    Safe to call more than once for the same night: a new session header is
    appended. Returns the log path.
    """
    global _active
    from .config import ensure_dirs

    ensure_dirs(cfg)
    path = run_log_path(cfg)
    if _active is not None:
        # Already teeing — just mark a new section in the same file
        if _active.path.resolve() == path.resolve():
            stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            line = f"\n----- session  {stamp}"
            if command:
                line += f"  ({command})"
            line += " -----\n"
            _active._log.write_direct(line)
            return path
        stop_run_log()

    _active = RunLog(path, command=command)
    return path


def stop_run_log() -> None:
    global _active
    if _active is not None:
        _active.close()
        _active = None


def current_run_log() -> Path | None:
    return _active.path if _active is not None else None


@atexit.register
def _atexit_stop() -> None:
    stop_run_log()
