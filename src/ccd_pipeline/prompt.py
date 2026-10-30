"""Minimal interactive prompts with native macOS/tk path pickers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_INPUT_RULE = "-" * 55


def _ack_input() -> None:
    """Separate an accepted answer from the work that follows."""
    print(_INPUT_RULE)


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            _ack_input()
            return value
        if default is not None:
            _ack_input()
            return default
        print("  Please enter a value.")


def _escape_as(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def pick_directory(
    *,
    prompt: str = "Select a folder",
    start: str | Path | None = None,
) -> Path | None:
    """Open a native folder chooser. Returns None if cancelled."""
    start_path = Path(start).expanduser() if start else Path.home()
    if not start_path.exists():
        start_path = Path.home()

    if sys.platform == "darwin":
        script = (
            f'set theFolder to choose folder with prompt "{_escape_as(prompt)}" '
            f'default location POSIX file "{_escape_as(str(start_path))}"\n'
            f"return POSIX path of theFolder"
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            result = None
        if result is not None:
            if result.returncode != 0:
                return None
            chosen = result.stdout.strip()
            return Path(chosen) if chosen else None

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        chosen = filedialog.askdirectory(title=prompt, initialdir=str(start_path))
        root.destroy()
        return Path(chosen) if chosen else None
    except Exception:
        return None


def pick_file(
    *,
    prompt: str = "Select a file",
    start: str | Path | None = None,
    types: list[tuple[str, str]] | None = None,
) -> Path | None:
    """
    Open a native file chooser. Returns None if cancelled.

    types : optional list of (label, ext) e.g. [("YAML", "yaml"), ("YAML", "yml")]
    """
    start_path = Path(start).expanduser() if start else Path.home()
    if not start_path.exists():
        start_path = Path.home()

    if sys.platform == "darwin":
        type_clause = ""
        if types:
            # AppleScript: of type {"yaml", "yml"} — UTI/extension style varies;
            # using filename extensions via "with prompt" only is more reliable,
            # so we filter after selection if needed.
            exts = sorted({ext.lstrip(".").lower() for _, ext in types})
            # of type uses 4-char type codes historically; skip and filter in Python
            _ = exts
        script = (
            f'set theFile to choose file with prompt "{_escape_as(prompt)}" '
            f'default location POSIX file "{_escape_as(str(start_path))}"\n'
            f"return POSIX path of theFile"
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            result = None
        if result is not None:
            if result.returncode != 0:
                return None
            chosen = result.stdout.strip()
            return Path(chosen) if chosen else None

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        filetypes = [(label, f"*.{ext.lstrip('.')}") for label, ext in (types or [])]
        filetypes = filetypes or [("All files", "*.*")]
        chosen = filedialog.askopenfilename(
            title=prompt,
            initialdir=str(start_path),
            filetypes=filetypes + [("All files", "*.*")],
        )
        root.destroy()
        return Path(chosen) if chosen else None
    except Exception:
        return None


def ask_path(
    prompt: str,
    default: str | None = None,
    *,
    must_exist: bool = False,
    kind: str = "dir",
    start: str | Path | None = None,
    types: list[tuple[str, str]] | None = None,
) -> Path:
    """
    Ask for a path using a native picker (directory or file).

    kind : "dir" | "file"
    Falls back to typing if the picker is cancelled/unavailable.
    """
    if kind == "file":
        return ask_file(prompt, must_exist=must_exist, start=start, types=types)
    return ask_directory(prompt, must_exist=must_exist, start=start)


def ask_directory(
    prompt: str = "Select a directory",
    *,
    must_exist: bool = True,
    start: str | Path | None = None,
) -> Path:
    """Native folder picker; typed path fallback."""
    print(f"\n{prompt}")
    print("Opening folder picker...")
    chosen = pick_directory(prompt=prompt, start=start)
    if chosen is not None:
        if must_exist and not chosen.exists():
            print(f"  Selected path does not exist: {chosen}")
        elif must_exist and not chosen.is_dir():
            print(f"  Selected path is not a directory: {chosen}")
        else:
            print(f"  Selected: {chosen}")
            return chosen.resolve()

    print("No folder selected (or picker unavailable). Enter a path instead.")
    while True:
        raw = ask("Directory path")
        path = Path(raw).expanduser()
        if must_exist and not path.exists():
            print(f"  Path does not exist: {path}")
            continue
        if must_exist and not path.is_dir():
            print(f"  Not a directory: {path}")
            continue
        return path.resolve()


def ask_file(
    prompt: str = "Select a file",
    *,
    must_exist: bool = True,
    start: str | Path | None = None,
    types: list[tuple[str, str]] | None = None,
) -> Path:
    """Native file picker; typed path fallback."""
    print(f"\n{prompt}")
    print("Opening file picker...")
    chosen = pick_file(prompt=prompt, start=start, types=types)
    if chosen is not None:
        if must_exist and not chosen.exists():
            print(f"  Selected path does not exist: {chosen}")
        elif must_exist and not chosen.is_file():
            print(f"  Selected path is not a file: {chosen}")
        else:
            print(f"  Selected: {chosen}")
            return chosen.resolve()

    print("No file selected (or picker unavailable). Enter a path instead.")
    while True:
        raw = ask("File path")
        path = Path(raw).expanduser()
        if must_exist and not path.exists():
            print(f"  Path does not exist: {path}")
            continue
        if must_exist and not path.is_file():
            print(f"  Not a file: {path}")
            continue
        return path.resolve()


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_txt = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_txt}]: ").strip().lower()
        if not value:
            _ack_input()
            return default
        if value in {"y", "yes"}:
            _ack_input()
            return True
        if value in {"n", "no"}:
            _ack_input()
            return False
        print("  Please answer y or n.")


def ask_int(prompt: str, default: int | None = None, *, minimum: int | None = None) -> int:
    while True:
        raw = ask(prompt, None if default is None else str(default))
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter an integer.")
            continue
        if minimum is not None and value < minimum:
            print(f"  Must be >= {minimum}.")
            continue
        return value


def choose(
    prompt: str,
    options: list[tuple[str, str]],
    *,
    allow_quit: bool = False,
    default: str | None = None,
) -> str:
    """Present a numbered menu. Returns the chosen id."""
    from .term import S, style

    ids = [opt_id for opt_id, _ in options]
    if default is not None and default not in ids:
        raise ValueError(f"choose default {default!r} is not an option id")
    default_idx = ids.index(default) + 1 if default is not None else None

    print()
    print(style(prompt, S.BOLD, S.CYAN))
    for i, (_, label) in enumerate(options, start=1):
        print(f"  {i}) {label}")
    if allow_quit:
        print("  q) Quit")

    hint = f"Choice [{default_idx}]: " if default_idx is not None else "Choice: "
    while True:
        raw = input(hint).strip().lower()
        if not raw and default is not None:
            _ack_input()
            return default
        if allow_quit and raw in {"q", "quit"}:
            _ack_input()
            return "quit"
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(ids):
                _ack_input()
                return ids[idx - 1]
        print("  Invalid choice.")


def choose_many(
    prompt: str,
    options: list[tuple[str, str]],
    *,
    none_label: str = "None",
) -> list[str]:
    """Numbered menu that accepts several picks: ``2 4 6``, ``2,4,6``, ``2-6``, or ``all``.

    Returns the chosen ids in menu order; an empty list means none.
    """
    from .term import S, style

    print()
    print(style(prompt, S.BOLD, S.CYAN))
    for i, (_, label) in enumerate(options, start=1):
        print(f"  {i}) {label}")
    print(f"  {len(options) + 1}) {none_label}")
    print("  (several: 2 4 6  or  2,4,6  or  2-6  or  all)")

    ids = [opt_id for opt_id, _ in options]
    none_idx = len(options) + 1
    while True:
        raw = input("Choice: ").strip().lower().replace(",", " ")
        if not raw:
            print("  Invalid choice.")
            continue
        if raw in {"all", "a"}:
            _ack_input()
            return list(ids)
        picked: set[int] = set()
        ok = True
        for token in raw.split():
            if token.isdigit():
                picked.add(int(token))
            elif "-" in token:
                lo, _, hi = token.partition("-")
                if lo.isdigit() and hi.isdigit() and int(lo) <= int(hi):
                    picked.update(range(int(lo), int(hi) + 1))
                else:
                    ok = False
            else:
                ok = False
        if not ok or not picked or any(i < 1 or i > none_idx for i in picked):
            print("  Invalid choice.")
            continue
        if none_idx in picked:
            if len(picked) > 1:
                print(f"  {none_idx}) {none_label} cannot be combined with other picks.")
                continue
            _ack_input()
            return []
        _ack_input()
        return [ids[i - 1] for i in sorted(picked)]


def ask_index_picks(
    n: int,
    *,
    prompt: str = "Numbers",
    default: list[int] | None = None,
) -> list[int]:
    """Return 1-based indices in ``1..n``.

    Accepts ``1 3 5-8``, commas, or ``all``. Empty input uses ``default``
    (all indices when ``default`` is omitted).
    """
    if n < 1:
        return []
    fallback = list(default) if default is not None else list(range(1, n + 1))
    fallback_txt = " ".join(str(i) for i in fallback)
    while True:
        raw = input(f"{prompt} [{fallback_txt}]: ").strip().lower().replace(",", " ")
        if not raw:
            _ack_input()
            return fallback
        if raw in {"all", "a"}:
            _ack_input()
            return list(range(1, n + 1))
        picked: set[int] = set()
        ok = True
        for token in raw.split():
            if token.isdigit():
                picked.add(int(token))
            elif "-" in token:
                lo, _, hi = token.partition("-")
                if lo.isdigit() and hi.isdigit() and int(lo) <= int(hi):
                    picked.update(range(int(lo), int(hi) + 1))
                else:
                    ok = False
            else:
                ok = False
        if not ok or not picked or any(i < 1 or i > n for i in picked):
            print(f"  Enter numbers from 1 to {n}, a range (2-6), or all.")
            continue
        _ack_input()
        return sorted(picked)


def pause(message: str = "Press Enter to continue...") -> None:
    input(message)
    _ack_input()
