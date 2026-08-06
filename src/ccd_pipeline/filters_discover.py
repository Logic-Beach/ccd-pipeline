"""Discover filter-wheel position combinations from raw flats.

HDI (and similar) cameras encode the filter as ``FILTER1``/``FILTER2`` wheel
IDs. Dome flats often label the band in ``OBJECT`` (e.g. ``dflat-r``), which we
use to suggest short names for the night ``filter_map``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from astropy.io import fits

from .io import filter_key


def _guess_name_from_object(obj: str) -> str | None:
    obj = (obj or "").strip()
    lower = obj.lower()
    for prefix in ("dflat-", "flat-", "skyflat-", "sflat-"):
        if lower.startswith(prefix):
            return obj.split("-", 1)[1]
    return None


def discover_filter_combos(
    raw_dir: Path,
    *,
    obstype_key: str = "OBSTYPE",
    flat_value: str = "FLAT",
    object_key: str = "OBJECT",
    filter1_key: str = "FILTER1",
    filter2_key: str = "FILTER2",
    exptime_key: str = "EXPTIME",
) -> list[dict[str, Any]]:
    """
    Scan flats and return one record per unique FILTER1,FILTER2 pair.

    Each record includes counts, OBJECT name tallies, and a suggested short name
    (from dflat-* style OBJECT labels when available).
    """
    raw_dir = Path(raw_dir)
    by_key: dict[str, dict[str, Any]] = {}

    for path in sorted(raw_dir.glob("*.fits")):
        if path.name.lower() == "latest.fits":
            continue
        hdr = fits.getheader(path, 0)
        obstype = str(hdr.get(obstype_key, "")).strip().upper()
        if obstype != str(flat_value).strip().upper():
            continue

        key = filter_key(hdr, filter1_key, filter2_key)
        obj = str(hdr.get(object_key, "")).strip()
        exptime = hdr.get(exptime_key)

        rec = by_key.setdefault(
            key,
            {
                "key": key,
                "filter1": str(hdr.get(filter1_key, "")).strip(),
                "filter2": str(hdr.get(filter2_key, "")).strip(),
                "n": 0,
                "objects": Counter(),
                "exptimes": Counter(),
                "example_file": path.name,
            },
        )
        rec["n"] += 1
        rec["objects"][obj] += 1
        rec["exptimes"][exptime] += 1

    results = []
    for key, rec in sorted(by_key.items(), key=lambda kv: kv[0]):
        # Prefer the most common OBJECT for naming suggestion
        top_obj = rec["objects"].most_common(1)[0][0] if rec["objects"] else ""
        suggested = _guess_name_from_object(top_obj) or key.replace(",", "_")
        rec["suggested_name"] = suggested
        rec["top_object"] = top_obj
        results.append(rec)
    return results


def interactive_filter_map(
    combos: list[dict[str, Any]],
    *,
    prior_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Prompt the user to assign a short filter name to each wheel combination.
    """
    from .prompt import ask, ask_yes_no
    from .term import heading, style, S, subheading

    prior_map = prior_map or {}
    heading("Filter wheel mapping")
    print(
        "Each unique FILTER1,FILTER2 pair found in FLAT frames is listed below.\n"
        "Confirm or edit the short name used for master flats / science matching.\n"
        "(Wheel setups change — remapping per night is expected.)\n"
    )

    mapping: dict[str, str] = {}
    used_names: dict[str, str] = {}

    for rec in combos:
        key = rec["key"]
        objects = ", ".join(f"{n}× {o!r}" for o, n in rec["objects"].most_common(5))
        exps = ", ".join(str(e) for e, _ in rec["exptimes"].most_common(5))
        default = prior_map.get(key) or rec["suggested_name"]

        subheading(f"Wheel key {key}")
        print(f"Wheel key : {key}  (FILTER1={rec['filter1']}, FILTER2={rec['filter2']})")
        print(f"Flat count: {rec['n']}")
        print(f"OBJECT(s) : {objects}")
        print(f"EXPTIME(s): {exps}")
        print(f"Example   : {rec['example_file']}")

        while True:
            name = ask("Short filter name", default)
            if not name:
                continue
            if name in used_names and used_names[name] != key:
                print(
                    f"  Name {name!r} already used for wheel key {used_names[name]}. "
                    "Choose another, or re-enter the same to force."
                )
                if not ask_yes_no(f"Use {name!r} anyway (not recommended)", default=False):
                    continue
            mapping[key] = name
            used_names[name] = key
            break

    print()
    print(style("Filter map summary:", S.BOLD, S.CYAN))
    for key, name in mapping.items():
        print(f"  {key:>12}  →  {name}")
    if not ask_yes_no("Accept this filter map", default=True):
        return interactive_filter_map(combos, prior_map=mapping)
    return mapping
