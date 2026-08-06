"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .inventory import format_inventory, inventory_night
from .masters import build_all_masters, make_master_bias, make_master_flats
from .reduce import reduce_science


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


if __name__ == "__main__":
    inventory_main()
