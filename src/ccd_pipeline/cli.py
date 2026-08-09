"""Command-line entry points.

``ccd`` with no subcommand launches the interactive wizard. Subcommands wrap
inventory / masters / sanity / reduce / wcs for scripting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .inventory import format_inventory, inventory_night
from .masters import build_all_masters, make_master_bias, make_master_flats
from .reduce import reduce_science
from .sanity import sanity_check_masters
from .wcs_solve import solve_science_wcs
from .wizard import wizard_main


def _config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to night/instrument YAML config",
    )


def inventory_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inventory raw CCD frames")
    _config_arg(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    summary = inventory_night(cfg)
    print(format_inventory(summary))


def masters_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build master bias and flats")
    _config_arg(parser)
    parser.add_argument(
        "--only",
        choices=["bias", "flats", "all"],
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
    if args.only == "bias":
        make_master_bias(cfg)
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
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    reduce_science(cfg, limit=args.limit)


def sanity_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sanity-check master bias and flats")
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
        help="Only check these flat filters (repeatable)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    sanity_check_masters(cfg, open_plots=not args.no_open, filters=args.filters)


def wcs_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plate-solve calibrated science frames (offline astrometry.net)"
    )
    _config_arg(parser)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only solve the first N science frames (for testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-solve frames that already have a WCS",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    solve_science_wcs(cfg, limit=args.limit, overwrite=args.overwrite or None)


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
    ccd wcs --config ...
    """
    argv = list(argv) if argv is not None else None
    import sys

    if argv is None:
        argv = sys.argv[1:]

    subcommands = {"inventory", "masters", "sanity", "reduce", "wcs", "wizard", "help"}
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
        elif command == "wcs":
            wcs_main(rest)
        return

    # No subcommand: interactive flow; optional raw-dir path as first arg
    wizard_main(argv)


if __name__ == "__main__":
    main()
