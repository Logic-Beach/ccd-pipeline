"""Terminal styling helpers (ANSI colors).

Respects ``NO_COLOR`` and only colors when stdout is a TTY (or ``FORCE_COLOR``
is set). Safe to use when output is piped / redirected.
"""

from __future__ import annotations

import os
import sys


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
