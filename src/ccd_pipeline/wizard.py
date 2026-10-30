"""Interactive reduction flow: auto-config from raw dir, then step through masters.

Entry point for ``ccd`` with no subcommand. Prompts for a night folder (or uses
a path argument), writes a YAML config, then optionally runs bias → flats →
sanity → science → WCS.
"""

from __future__ import annotations

from pathlib import Path

from .config import find_project_root, load_config
from .frame_qc import offer_exclude_frames, sanity_check_frames
from .fringe import apply_master_fringe, fringe_config, make_master_fringe
from .inventory import format_inventory, inventory_night
from .io import iter_fits_files
from .calibrate import darks_enabled
from .masters import make_master_bias, make_master_dark, make_master_flats
from .night_init import (
    build_config_from_raw,
    format_inspection,
    resolve_night_paths,
    update_config_filter_map,
)
from .prompt import ask_directory, ask_file, ask_yes_no, choose
from .reduce import reduce_science
from .runlog import start_run_log
from .sanity import offer_usable_overscan, sanity_check_masters
from .science_qc import sanity_check_science
from .term import (
    banner,
    dim,
    heading,
    info,
    label_value,
    show_ccd,
    subheading,
    success,
    warn,
)
from .wcs_solve import solve_science_wcs


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
    config_path, cfg, info_dict = build_config_from_raw(
        raw_path,
        project_root=project_root,
        output_dir=output_dir,
    )
    _prompt_unresolved_filters(config_path, cfg, info_dict)
    print()
    print(format_inspection(info_dict, config_path))
    return config_path


def _prompt_unresolved_filters(config_path: Path, cfg: dict, info_dict: dict) -> None:
    """Ask for a short filter name when OBJECT does not give one agreed band."""
    from .filters_discover import interactive_filter_map

    unresolved = [
        combo
        for combo in info_dict.get("filter_combos") or []
        if not combo.get("named_from_object")
    ]
    if not unresolved:
        return
    mapping = dict(info_dict.get("filter_map") or {})
    answered = interactive_filter_map(
        unresolved, prior_map=mapping, unresolved_only=True
    )
    mapping.update(answered)
    info_dict["filter_map"] = mapping
    update_config_filter_map(config_path, cfg, mapping)


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
    log_path = start_run_log(cfg, command="wizard")
    print()
    print(label_value("Using config: ", config_path))
    print(label_value("Raw         : ", cfg["paths"]["raw_dir"]))
    print(label_value("Output      : ", cfg["paths"]["output_dir"]))
    info(f"Logging to {log_path}")

    subheading("Inventory")
    show_ccd(cfg, "inventory")
    print(format_inventory(inventory_night(cfg)))

    # Raw overscan before any master is combined, so a light leak can be
    # excluded from the bias that flats and science will use.
    heading("Step: check overscan for light leaks")
    if ask_yes_no(
        "Check raw overscan for light leaks before building the master bias",
        True,
    ):
        report = sanity_check_masters(
            cfg, open_plots=True, include_bias=False, include_flats=False
        )
        offer_usable_overscan(
            cfg,
            config_path,
            report,
            open_plots=True,
            rebuild_bias=False,
            rebuild_flats=False,
        )

    # Whole-frame outliers (imstat habit): a bias that sits low or a noisy
    # dark is not caught by per-pixel σ-clipping inside the combine.
    heading("Step: check raw bias/dark frames")
    if ask_yes_no(
        "Show per-frame statistics for raw bias/dark frames and flag outliers",
        True,
    ):
        report = sanity_check_frames(cfg, kinds=("bias", "dark"))
        offer_exclude_frames(cfg, config_path, report)

    # Master bias
    heading("Step: master bias")
    if ask_yes_no("Build master bias", True):
        make_master_bias(cfg)
    else:
        warn("Skipped master bias.")

    # Overscan + master bias before any flat is combined. A light leak corrected
    # here is stored on the night config and used for flats and science too.
    heading("Step: check overscan and master bias")
    masters_dir = Path(cfg["paths"]["masters_dir"])
    has_bias = (masters_dir / "master_bias.fits").exists()
    if has_bias:
        if ask_yes_no("Check overscan and master bias before flats", True):
            report = sanity_check_masters(cfg, open_plots=True, include_flats=False)
            report = offer_usable_overscan(
                cfg, config_path, report, open_plots=True, rebuild_flats=False
            )
            if report["failures"] or report["warnings"]:
                if not ask_yes_no("Continue to master flats anyway", False):
                    warn("Stopped before master flats.")
                    success("\nDone.")
                    return
    else:
        warn("No master bias yet — skipping the overscan check.")

    # Master dark after the bias it depends on, and before flats that subtract it.
    if darks_enabled(cfg):
        heading("Step: master dark")
        if ask_yes_no("Build master dark (subtracted from flats and science)", True):
            make_master_dark(cfg)
        else:
            warn(
                "Skipped master dark. Flats and science in this run will not "
                "subtract dark current."
            )
            cfg.setdefault("darks", {})["enabled"] = False
    else:
        info("No darks for this night — dark subtraction stays off.")

    # Master flats
    heading("Step: master flats")
    summary = inventory_night(cfg)
    available = sorted(summary.get("flats_by_filter", {}))
    if not available:
        warn("No flats found — skipping.")
    else:
        print(f"Filters with flats: {', '.join(available)}")
        if ask_yes_no(
            "Show per-frame statistics for raw flats and flag outliers first", True
        ):
            report = sanity_check_frames(cfg, kinds=("flat",))
            offer_exclude_frames(cfg, config_path, report)
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

    # Flat sanity before science. Overscan is checked again only if a light leak
    # is still present; the region chosen above is already on the night config.
    heading("Step: sanity-check master flats")
    has_bias = (masters_dir / "master_bias.fits").exists()
    has_flats = any(masters_dir.glob("master_flat_*.fits"))
    if has_bias or has_flats:
        if ask_yes_no("Run sanity checks (stats + diagnostic plots)", True):
            report = sanity_check_masters(cfg, open_plots=True)
            report = offer_usable_overscan(
                cfg, config_path, report, open_plots=True
            )
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

    # Science sky QC + fringe (z/i)
    heading("Step: science sky QC / fringe")
    sci_dir = Path(cfg["paths"]["output_dir"]) / "science"
    n_sci = len(list(iter_fits_files(sci_dir))) if sci_dir.is_dir() else 0
    if n_sci == 0:
        warn("No calibrated science frames — skipping science QC / fringe.")
    else:
        if ask_yes_no("Check science sky flatness (sample frames)", True):
            sanity_check_science(cfg, open_plots=True)
        fcfg = fringe_config(cfg)
        fringe_filters = list(fcfg.get("filters") or ["z"])
        if ask_yes_no(
            f"Build master fringe for {', '.join(fringe_filters)} "
            "(star-masked sky stack)",
            True,
        ):
            make_master_fringe(cfg, filters=fringe_filters, interactive=True)
            if ask_yes_no("Apply scaled fringe to science frames now", False):
                apply_master_fringe(cfg, filters=fringe_filters)

    # WCS (offline astrometry.net)
    heading("Step: WCS plate-solve")
    n_sci = len(list(iter_fits_files(sci_dir))) if sci_dir.is_dir() else 0
    if n_sci == 0:
        warn("No calibrated science frames — skipping WCS.")
    elif ask_yes_no(f"Add WCS to science frames ({n_sci} found)", False):
        limit = None
        if ask_yes_no("Limit to first N frames (test)", False):
            from .prompt import ask_int

            limit = ask_int("N", 2, minimum=1)
        overwrite = ask_yes_no("Re-solve frames that already have WCS", False)
        solve_science_wcs(cfg, limit=limit, overwrite=overwrite)
    else:
        warn("Skipped WCS plate-solve.")

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
