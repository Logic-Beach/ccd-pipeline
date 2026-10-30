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

from .inventory import is_flat_obstype
from .io import _wheel_ids_look_valid, iter_fits_files


# Longer markers first so ``dflat-`` wins over the ``flat-`` inside it.
_FLAT_MARKERS = ("skyflat-", "dflat-", "tflat-", "sflat-", "flat-")


def _guess_name_from_object(obj: str) -> str | None:
    """Return the band token from a flat OBJECT name, if one is present.

    Matches a marker anywhere in the name, so ``dflat-r`` and ``oct12-dflat-u``
    both yield the band (``r``, ``u``). ``tflat-`` covers twilight flats.
    """
    text = (obj or "").strip()
    lower = text.lower()
    best: tuple[str, int] | None = None
    for marker in _FLAT_MARKERS:
        idx = lower.find(marker)
        if idx < 0:
            continue
        if best is None or idx < best[1] or (idx == best[1] and len(marker) > len(best[0])):
            best = (marker, idx)
    if best is None:
        return None
    marker, idx = best
    name = text[idx + len(marker) :].strip()
    return name or None


def _suggest_filter_name(objects: Counter, fallback: str) -> tuple[str, bool]:
    """Pick a short name from flat OBJECT labels.

    Returns ``(name, named_from_object)``. ``named_from_object`` is true only
    when every labeled flat agrees on one band. A wheel-key fallback
    (``7_2``) or a split vote needs the user to type the name.
    """
    votes: Counter[str] = Counter()
    for obj, n in objects.items():
        guessed = _guess_name_from_object(obj)
        if guessed:
            votes[guessed] += n
    if len(votes) == 1:
        return next(iter(votes)), True
    if votes:
        return votes.most_common(1)[0][0], False
    return fallback, False


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

    for path in iter_fits_files(raw_dir):
        hdr = fits.getheader(path, 0)
        obstype = str(hdr.get(obstype_key, "")).strip().upper()
        if not is_flat_obstype(obstype, flat_value):
            continue

        f1 = str(hdr.get(filter1_key, "")).strip()
        f2 = str(hdr.get(filter2_key, "")).strip()
        if not f1 and not f2:
            for card in ("FILTER", "FILTNAME", "FILTERS"):
                name = str(hdr.get(card, "")).strip()
                if name:
                    f1 = f2 = name
                    break
        key = f"{f1},{f2}"
        obj = str(hdr.get(object_key, "")).strip()
        exptime = hdr.get(exptime_key)

        rec = by_key.setdefault(
            key,
            {
                "key": key,
                "filter1": f1,
                "filter2": f2,
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
        top_obj = rec["objects"].most_common(1)[0][0] if rec["objects"] else ""
        f1 = rec["filter1"]
        f2 = rec["filter2"]
        if f1 and f1 == f2:
            fallback = f1
        else:
            fallback = key.replace(",", "_")
        suggested, named = _suggest_filter_name(rec["objects"], fallback)
        # FILTNAME / FILTER already holds the band (u, g, r…) — do not prompt.
        if (
            not named
            and f1
            and f1 == f2
            and not _wheel_ids_look_valid(key)
        ):
            suggested = f1
            named = True
        rec["suggested_name"] = suggested
        rec["named_from_object"] = named
        rec["top_object"] = top_obj
        results.append(rec)
    return results


def interactive_filter_map(
    combos: list[dict[str, Any]],
    *,
    prior_map: dict[str, str] | None = None,
    unresolved_only: bool = False,
) -> dict[str, str]:
    """
    Prompt the user to assign a short filter name to each wheel combination.
    """
    from .prompt import ask, ask_yes_no
    from .term import heading, style, S, subheading

    prior_map = prior_map or {}
    heading("Filter wheel mapping")
    if unresolved_only:
        print(
            "These wheel positions have no single filter name in OBJECT.\n"
            "Enter the short name used for master flats and science frames\n"
            "(for example u, g, r, i, z).\n"
        )
    else:
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
        fallback = key.replace(",", "_")
        default = prior_map.get(key) or rec["suggested_name"]
        # A bare wheel id (7_2) is not a filter. Make the user type one.
        if not rec.get("named_from_object", True) and default == fallback:
            default = None

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
        return interactive_filter_map(
            combos, prior_map=mapping, unresolved_only=unresolved_only
        )
    return mapping
