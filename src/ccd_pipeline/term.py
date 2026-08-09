"""Terminal styling helpers (ANSI colors).

Respects ``NO_COLOR`` and only colors when stdout is a TTY (or ``FORCE_COLOR``
is set). Safe to use when output is piped / redirected.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")


def color_enabled() -> bool:
    # https://no-color.org — any presence disables color
    if "NO_COLOR" in os.environ:
        return False
    fc = os.environ.get("FORCE_COLOR")
    if fc is not None:
        return fc.strip().lower() not in {"", "0", "false", "no"}
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


class S:
    """ANSI SGR codes."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_MAGENTA = "\033[95m"


def style(text: str, *codes: str) -> str:
    if not codes or not color_enabled():
        return text
    return f"{''.join(codes)}{text}{S.RESET}"


def bold(text: str) -> str:
    return style(text, S.BOLD)


def dim(text: str) -> str:
    return style(text, S.DIM)


def banner(title: str, width: int = 60) -> None:
    """Print a full-width banner block for the top of a run."""
    bar = "=" * width
    print()
    print(style(bar, S.BOLD, S.CYAN))
    print(style(title, S.BOLD, S.CYAN))
    print(style(bar, S.BOLD, S.CYAN))


def heading(title: str) -> None:
    """Major section heading, e.g. ``=== Master bias ===``."""
    text = title if title.startswith("=") else f"=== {title} ==="
    print()
    print(style(text, S.BOLD, S.BRIGHT_CYAN))


def subheading(title: str) -> None:
    """Secondary section, e.g. ``--- filter 'r' ---``."""
    text = title if title.startswith("-") else f"--- {title} ---"
    print()
    print(style(text, S.BOLD, S.YELLOW))


def step(label: str) -> None:
    """In-progress file / action line."""
    print()
    print(style(f"  → {label}", S.BOLD, S.MAGENTA))


def success(text: str) -> None:
    print(style(text, S.BOLD, S.BRIGHT_GREEN))


def warn(text: str) -> None:
    print(style(text, S.BOLD, S.BRIGHT_YELLOW))


def fail(text: str) -> None:
    print(style(text, S.BOLD, S.BRIGHT_RED))


def info(text: str) -> None:
    print(style(text, S.CYAN))


def status_tag(level: str) -> str:
    """Colored ``[OK]`` / ``[WARN]`` / ``[FAIL]`` tag."""
    level = level.lower()
    tags = {
        "ok": ("OK  ", S.BRIGHT_GREEN),
        "warn": ("WARN", S.BRIGHT_YELLOW),
        "fail": ("FAIL", S.BRIGHT_RED),
    }
    label, code = tags.get(level, (level.upper()[:4].ljust(4), S.WHITE))
    return style(f"[{label}]", S.BOLD, code)


def label_value(label: str, value: object) -> str:
    return f"{style(label, S.DIM)}{value}"


def run_with_spinner(
    message: str,
    fn: Callable[..., _T],
    /,
    *args,
    interval: float = 0.45,
    hint: str | None = None,
    **kwargs,
) -> _T:
    """Run ``fn`` while printing an animated ``...`` line with elapsed seconds.

    Intended for long CPU-bound steps (e.g. σ-clip combine of 4k CCD frames)
    that otherwise look hung. Falls back to a static message when stdout is
    not a TTY.

    If ``hint`` is set, it is shown next to the timer (e.g. a coffee nudge).
    """
    suffix = f" — {hint}" if hint else ""

    if not getattr(sys.stdout, "isatty", lambda: False)():
        print(dim(f"    {message} (this may take a few minutes)...{suffix}"))
        sys.stdout.flush()
        return fn(*args, **kwargs)

    stop = threading.Event()
    t0 = time.monotonic()
    widest = 0

    def _line(elapsed: int, dots: str = "...") -> str:
        return f"    {message}{dots:<3}  ({elapsed}s){suffix}"

    def _spin() -> None:
        nonlocal widest
        n = 0
        while not stop.wait(interval):
            dots = "." * ((n % 3) + 1)
            elapsed = int(time.monotonic() - t0)
            line = _line(elapsed, dots)
            widest = max(widest, len(line))
            print("\r" + dim(line), end="", flush=True)
            n += 1

    print(dim(_line(0, ".")), end="", flush=True)
    widest = len(_line(0, "."))
    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        return fn(*args, **kwargs)
    finally:
        stop.set()
        thread.join(timeout=1.0)
        elapsed = int(time.monotonic() - t0)
        # Clear the spinner line, then confirm done
        print("\r" + " " * max(widest, 80) + "\r", end="")
        print(dim(f"    {message} done ({elapsed}s)"))
