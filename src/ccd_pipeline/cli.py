"""Command-line entry points.

``ccd`` with no subcommand launches the interactive wizard. Subcommands wrap
inventory / masters / sanity / reduce / wcs for scripting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .frame_qc import offer_exclude_frames, sanity_check_frames
from .fringe import apply_master_fringe, make_master_fringe
from .inventory import format_inventory, inventory_night
from .masters import build_all_masters, make_master_bias, make_master_dark, make_master_flats
from .reduce import reduce_science
from .runlog import start_run_log
from .sanity import offer_usable_overscan, sanity_check_masters
from .science_qc import sanity_check_science
from .term import info, show_ccd
from .wcs_solve import solve_science_wcs
from .wizard import wizard_main


def _config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to night/instrument YAML config",
    )


def _begin_logging(cfg: dict, command: str) -> None:
    path = start_run_log(cfg, command=command)
    info(f"Logging to {path}")


def inventory_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inventory raw CCD frames")
    _config_arg(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _begin_logging(cfg, "inventory")
    show_ccd(cfg, "inventory")
    summary = inventory_night(cfg)
    print(format_inventory(summary))


def masters_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build master bias and flats")
    _config_arg(parser)
    parser.add_argument(
        "--only",
        choices=["bias", "darks", "flats", "all"],
        default="all",
        help="Which masters to build",
    )
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        default=None,
        help="Only build master flat(s) for this filter name (repeatable), e.g. --filter r",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _begin_logging(cfg, "masters")
    if args.only == "bias":
        make_master_bias(cfg)
    elif args.only == "darks":
        make_master_dark(cfg)
    elif args.only == "flats":
        make_master_flats(cfg, filters=args.filters)
    else:
        build_all_masters(cfg, filters=args.filters)


def reduce_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate science frames")
    _config_arg(parser)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only reduce the first N science frames (for testing)",
    )
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        default=None,
        help="Only reduce science for this filter (repeatable), e.g. --filter z",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _begin_logging(cfg, "reduce")
    reduce_science(cfg, limit=args.limit, filters=args.filters)


def sanity_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check masters and/or calibrated science sky"
    )
    _config_arg(parser)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open diagnostic PNGs automatically",
    )
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        default=None,
        help="Only check these filters (repeatable)",
    )
    parser.add_argument(
        "--masters-only",
        action="store_true",
        help="Only check overscan + master bias/flats",
    )
    parser.add_argument(
        "--bias-only",
        action="store_true",
        help="Only check overscan + master bias (skip flats and science)",
    )
    parser.add_argument(
        "--overscan-only",
        action="store_true",
        help="Only check the raw overscan for a light leak, before a master bias exists",
    )
    parser.add_argument(
        "--frames-only",
        action="store_true",
        help=(
            "Only print per-frame statistics for raw bias/dark/flat frames "
            "(IRAF imstat style) and offer to exclude outliers"
        ),
    )
    parser.add_argument(
        "--science-only",
        action="store_true",
        help="Only check calibrated science sky flatness",
    )
    parser.add_argument(
        "--science-limit",
        type=int,
        default=2,
        help="Science frames to inspect per filter (default 2)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _begin_logging(cfg, "sanity")
    do_masters = not (args.science_only or args.frames_only)
    do_science = not (
        args.masters_only or args.bias_only or args.overscan_only or args.frames_only
    )
    if args.frames_only:
        report = sanity_check_frames(cfg, filters=args.filters)
        offer_exclude_frames(cfg, args.config, report)
    elif args.overscan_only:
        report = sanity_check_masters(
            cfg,
            open_plots=not args.no_open,
            include_bias=False,
            include_flats=False,
        )
        has_bias = (Path(cfg["paths"]["masters_dir"]) / "master_bias.fits").exists()
        offer_usable_overscan(
            cfg,
            args.config,
            report,
            open_plots=not args.no_open,
            rebuild_bias=has_bias,
            rebuild_flats=False,
        )
    elif do_masters:
        report = sanity_check_masters(
            cfg,
            open_plots=not args.no_open,
            filters=args.filters,
            include_flats=not args.bias_only,
        )
        offer_usable_overscan(
            cfg,
            args.config,
            report,
            open_plots=not args.no_open,
            rebuild_flats=not args.bias_only,
        )
    if do_science:
        sanity_check_science(
            cfg,
            filters=args.filters,
            limit_per_filter=args.science_limit,
            open_plots=not args.no_open,
        )


def imstat_main(argv: list[str] | None = None) -> None:
    """IRAF ``imstat`` equivalent: one line of statistics per FITS file."""
    import glob

    import numpy as np
    from astropy.io import fits

    from .frame_qc import STAT_COLUMNS, frame_statistics
    from .sanity import parse_fits_section
    from .term import dim, show_command

    parser = argparse.ArgumentParser(
        description=(
            "Per-file image statistics (astropy.stats.sigma_clipped_stats / mad_std), "
            "like IRAF imstat"
        )
    )
    parser.add_argument("files", nargs="+", help="FITS files or shell globs")
    parser.add_argument(
        "--section",
        default=None,
        metavar="[x1:x2,y1:y2]",
        help="FITS section (1-based, inclusive) to measure, e.g. [100:1000,100:1000]",
    )
    parser.add_argument(
        "--hdu", type=int, default=0, help="Image HDU index (default 0)"
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=None,
        metavar="KEY",
        help="Header keyword to print alongside the stats (repeatable)",
    )
    args = parser.parse_args(argv)

    paths: list[Path] = []
    for pattern in args.files:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    paths = [p for p in paths if not p.name.startswith("._")]
    show_command(["ccd", "imstat", *(["--section", args.section] if args.section else []), *[str(p) for p in paths]])
    print(
        dim(
            "    astropy.stats.sigma_clipped_stats(data, sigma=3)  →  mean*, std*;  "
            "astropy.stats.mad_std(data)  →  mad_std"
        )
    )
    section = parse_fits_section(args.section) if args.section else None
    keys = list(args.keyword or [])
    name_w = max(len(p.name) for p in paths) if paths else 4
    head = f"{'file':<{name_w}}"
    for key in keys:
        head += f"  {key:>10}"
    head += f"  {'npix':>9}"
    for _k, label in STAT_COLUMNS:
        head += f"  {label:>9}"
    print(head)
    for path in paths:
        try:
            with fits.open(path, memmap=False) as hdul:
                data = np.asarray(hdul[args.hdu].data, dtype=float)
                header = hdul[args.hdu].header
                if section is not None:
                    data = data[section]
        except Exception as exc:  # pragma: no cover - user input
            print(f"{path.name:<{name_w}}  ERROR: {exc}")
            continue
        stats = frame_statistics(data)
        line = f"{path.name:<{name_w}}"
        for key in keys:
            value = header.get(key, "")
            line += f"  {str(value)[:10]:>10}"
        line += f"  {stats['npix']:>9d}"
        for key, _label in STAT_COLUMNS:
            value = stats[key]
            if key == "n_hot":
                line += f"  {int(value):>9d}"
            else:
                line += f"  {value:>9.2f}"
        print(line)
    print(dim("mean*/std*: 3σ-clipped;  mad_std: robust σ;  hot: pixels > median + 10σ"))


def fringe_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build / apply additive master fringe frames (z, i, …)"
    )
    _config_arg(parser)
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        default=None,
        help="Filter(s) to process (default from config: z, i)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Subtract scaled fringe from science frames (after build unless --apply-only)",
    )
    parser.add_argument(
        "--apply-only",
        action="store_true",
        help="Only apply an existing master_fringe_*.fits",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-apply fringe to frames already marked FRINGCOR",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only apply to the first N science frames per filter",
    )
    parser.add_argument(
        "--select",
        choices=["blanks", "science", "all"],
        default=None,
        help=(
            "Frame selection for building the fringe master: "
            "blanks (blank/sky only), science (no blanks), "
            "all (similar airmass/time). Overrides config fringe.select."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the fringe-frame picker; use airmass/time/sky cuts as-is",
    )
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Show the UT / RA / Dec table and pick frames (default on a TTY)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _begin_logging(cfg, "fringe")
    if args.select:
        cfg.setdefault("fringe", {})["select"] = args.select
    interactive: bool | None
    if args.yes:
        interactive = False
    elif args.pick:
        interactive = True
    else:
        interactive = None
    if not args.apply_only:
        make_master_fringe(cfg, filters=args.filters, interactive=interactive)
    if args.apply or args.apply_only:
        apply_master_fringe(
            cfg,
            filters=args.filters,
            overwrite=args.overwrite,
            limit=args.limit,
        )


def wcs_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plate-solve calibrated science frames (offline astrometry.net)"
    )
    _config_arg(parser)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only solve the first N matching science frames (for testing)",
    )
    parser.add_argument(
        "--object",
        action="append",
        default=None,
        metavar="NAME",
        dest="objects",
        help="Only solve frames for this OBJECT (repeatable; matches filename)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-solve frames that already have a WCS",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _begin_logging(cfg, "wcs")
    solve_science_wcs(
        cfg,
        limit=args.limit,
        overwrite=args.overwrite or None,
        objects=args.objects,
    )


def main(argv: list[str] | None = None) -> None:
    """
    Top-level ``ccd`` command.

    Examples
    --------
    ccd
    ccd /path/to/raw/night
    ccd inventory --config configs/....yaml
    ccd masters --config ... --only bias
    ccd sanity --config ...
    ccd reduce --config ...
    ccd fringe --config ... --filter z --apply
    ccd wcs --config ...
    ccd imstat raw/*.fits --section [100:1000,100:1000]
    """
    argv = list(argv) if argv is not None else None
    import sys

    if argv is None:
        argv = sys.argv[1:]

    subcommands = {
        "inventory",
        "masters",
        "sanity",
        "reduce",
        "fringe",
        "wcs",
        "imstat",
        "wizard",
        "help",
    }
    if argv and argv[0] in subcommands:
        command = argv[0]
        rest = argv[1:]
        if command == "help":
            print(__doc__)
            return
        if command == "wizard":
            wizard_main(rest)
        elif command == "inventory":
            inventory_main(rest)
        elif command == "masters":
            masters_main(rest)
        elif command == "sanity":
            sanity_main(rest)
        elif command == "reduce":
            reduce_main(rest)
        elif command == "fringe":
            fringe_main(rest)
        elif command == "wcs":
            wcs_main(rest)
        elif command == "imstat":
            imstat_main(rest)
        return

    # No subcommand: interactive flow; optional raw-dir path as first arg
    wizard_main(argv)


if __name__ == "__main__":
    main()
