"""Minimal interactive prompts with native macOS/tk path pickers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
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
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
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


def choose(prompt: str, options: list[tuple[str, str]], *, allow_quit: bool = False) -> str:
    """Present a numbered menu. Returns the chosen id."""
    from .term import S, style

    print()
    print(style(prompt, S.BOLD, S.CYAN))
    for i, (_, label) in enumerate(options, start=1):
        print(f"  {i}) {label}")
    if allow_quit:
        print("  q) Quit")

    ids = [opt_id for opt_id, _ in options]
    while True:
        raw = input("Choice: ").strip().lower()
        if allow_quit and raw in {"q", "quit"}:
            return "quit"
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(ids):
                return ids[idx - 1]
        print("  Invalid choice.")


def pause(message: str = "Press Enter to continue...") -> None:
    input(message)
