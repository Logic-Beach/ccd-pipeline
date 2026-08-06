"""Interactive reduction flow: auto-config from raw dir, then step through masters.

Entry point for ``ccd`` with no subcommand. Prompts for a night folder (or uses
a path argument), writes a YAML config, then optionally runs bias → flats →
sanity → science.
"""

from __future__ import annotations

from pathlib import Path

from .config import find_project_root, load_config
from .inventory import format_inventory, inventory_night
from .masters import make_master_bias, make_master_flats
from .night_init import build_config_from_raw, format_inspection, resolve_night_paths
from .prompt import ask_directory, ask_file, ask_yes_no, choose
from .reduce import reduce_science
from .sanity import sanity_check_masters
from .term import banner, dim, heading, info, label_value, subheading, success, warn


def init_from_raw_dir(raw_dir: Path | None = None) -> Path:
    """Single-step config creation from a night raw directory."""
    project_root = find_project_root(Path.cwd())
    data_root = project_root.parent / "data"

    if raw_dir is None:
        start = data_root if data_root.exists() else project_root.parent
        raw_dir = ask_directory(
            "Select night folder or its raw/ subfolder",
            must_exist=True,
            start=start,
        )
    else:
        raw_dir = Path(raw_dir).expanduser().resolve()
        if not raw_dir.exists():
            raise SystemExit(f"Raw directory not found: {raw_dir}")

    night_id, raw_path, default_reduced, night_root = resolve_night_paths(
        raw_dir, project_root
    )
    print()
    print(label_value("Night     : ", night_id))
    print(label_value("Night root: ", night_root))
    print(label_value("Raw       : ", raw_path))
    print(label_value("Reduced   : ", default_reduced))

    if ask_yes_no(f"Use default reduced folder?\n  {default_reduced}", True):
        output_dir = default_reduced
    else:
        parent = ask_directory(
            "Select night folder (reduced/ will be created inside it)",
            must_exist=True,
            start=night_root if night_root.exists() else data_root,
        )
        output_dir = parent / "reduced" if parent.name != "reduced" else parent
        info(f"  Output will be: {output_dir}")

    info(f"\nScanning headers in {raw_path} ...")
    config_path, _cfg, info_dict = build_config_from_raw(
        raw_path,
        project_root=project_root,
        output_dir=output_dir,
    )
    print()
    print(format_inspection(info_dict, config_path))
    return config_path


def pick_config(project_root: Path) -> Path:
    configs_dir = project_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = ask_file(
        "Select a night config YAML",
        must_exist=True,
        start=configs_dir,
        types=[("YAML", "yaml"), ("YAML", "yml")],
    )
    return path


def step_through_reduction(config_path: Path) -> None:
    """Walk bias → flats → science with minimal prompts; list files used."""
    cfg = load_config(config_path)
    print()
    print(label_value("Using config: ", config_path))
    print(label_value("Raw         : ", cfg["paths"]["raw_dir"]))
    print(label_value("Output      : ", cfg["paths"]["output_dir"]))

    subheading("Inventory")
    print(format_inventory(inventory_night(cfg)))

    # Master bias
    heading("Step: master bias")
    if ask_yes_no("Build master bias", True):
        make_master_bias(cfg)
    else:
        warn("Skipped master bias.")

    # Master flats
    heading("Step: master flats")
    summary = inventory_night(cfg)
    available = sorted(summary.get("flats_by_filter", {}))
    if not available:
        warn("No flats found — skipping.")
    else:
        print(f"Filters with flats: {', '.join(available)}")
        mode = choose(
            "Build master flats:",
            [
                ("all", "All filters"),
                ("pick", "Choose filters"),
                ("skip", "Skip flats"),
            ],
        )
        if mode == "skip":
            warn("Skipped master flats.")
        else:
            filters = None
            if mode == "pick":
                from .prompt import ask

                raw = ask("Comma-separated filter names", ",".join(available))
                filters = [s.strip() for s in raw.split(",") if s.strip()]
            make_master_flats(cfg, filters=filters)

    # Sanity check masters before science
    heading("Step: sanity-check masters")
    masters_dir = Path(cfg["paths"]["masters_dir"])
    has_bias = (masters_dir / "master_bias.fits").exists()
    has_flats = any(masters_dir.glob("master_flat_*.fits"))
    if has_bias or has_flats:
        if ask_yes_no("Run sanity checks (stats + diagnostic plots)", True):
            report = sanity_check_masters(cfg, open_plots=True)
            if report["failures"] or report["warnings"]:
                if not ask_yes_no("Continue to science calibration anyway", False):
                    warn("Stopped before science calibration.")
                    success("\nDone.")
                    return
    else:
        warn("No masters found yet — skipping sanity checks.")

    # Science
    heading("Step: science calibration")
    if ask_yes_no("Calibrate science frames now", False):
        limit = None
        if ask_yes_no("Limit to first N frames (test)", False):
            from .prompt import ask_int

            limit = ask_int("N", 2, minimum=1)
        reduce_science(cfg, limit=limit)
    else:
        warn("Skipped science calibration.")

    success("\nDone.")


def wizard_main(argv: list[str] | None = None) -> None:
    """
    Entry for interactive use.

    Usage patterns:
      ccd
      ccd /path/to/data/<night>/raw
      ccd wizard /path/to/data/<night>
    """
    argv = list(argv or [])
    project_root = find_project_root(Path.cwd())

    raw_arg = None
    if argv and not argv[0].startswith("-"):
        raw_arg = Path(argv[0]).expanduser()

    banner("CCD reduction")

    if raw_arg is not None:
        config_path = init_from_raw_dir(raw_arg)
        if ask_yes_no("Continue to master bias / flats", True):
            step_through_reduction(config_path)
        return

    action = choose(
        "Start:",
        [
            ("new", "New night — select data/<night> or data/<night>/raw"),
            ("existing", "Use an existing config"),
        ],
        allow_quit=True,
    )
    if action == "quit":
        print(dim("Bye."))
        return
    if action == "new":
        config_path = init_from_raw_dir(None)
        if ask_yes_no("Continue to master bias / flats", True):
            step_through_reduction(config_path)
    else:
        config_path = pick_config(project_root)
        step_through_reduction(config_path)
