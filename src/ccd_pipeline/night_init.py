"""Auto-build a night config from a raw FITS directory.

Scans headers for telescope/instrument, frame-type counts, image HDU,
``BIASSEC``/``DATASEC``, and a filter map from flats — then writes YAML under
``configs/``. Data paths may be absolute (external drives) or relative to the
project root.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from astropy.io import fits

from .config import find_project_root, has_project_marker
from .filters_discover import discover_filter_combos


def _iter_fits(raw_dir: Path):
    for path in sorted(Path(raw_dir).glob("*.fits")):
        if path.name.lower() != "latest.fits":
            yield path


def _detect_image_hdu(path: Path) -> str | int:
    with fits.open(path, memmap=False) as hdul:
        names = [hdu.name for hdu in hdul]
        if "XY00" in names or "xy00" in names:
            # EXTNAME may be stored uppercased in the name list
            for hdu in hdul:
                if str(hdu.name).lower() == "xy00":
                    return hdu.name
            return "xy00"
        if hdul[0].data is not None:
            return 0
        if len(hdul) > 1 and hdul[1].data is not None:
            return 1
    return 0


def _pick_keyword(header, candidates: list[str], default: str) -> str:
    for key in candidates:
        if key in header and header[key] not in (None, ""):
            return key
    return default


def _default_wcs_block(instrument: str) -> dict[str, Any]:
    """Instrument-aware defaults for offline ``solve-field`` plate-solving."""
    inst = str(instrument or "").lower()
    # HDI: 0.43″/pix, ~29′ FOV (WIYN 0.9m ops manual)
    scale = 0.43 if "hdi" in inst else 0.43
    return {
        "enabled": True,
        "solver": "astrometry-net",
        "solve_field": "solve-field",
        "index_dir": None,
        "scale_arcsec_per_pix": scale,
        "scale_tol_frac": 0.05,
        "radius_deg": 1.0,
        "downsample": 2,
        "overwrite": False,
        "timeout_sec": 120,
    }


def _rel_to_root(path: Path, root: Path) -> str:
    """Prefer absolute data paths unless ``root`` is a real package/repo tree.

    Campaign folders (e.g. ``2017_JUN_JAS``) often have no ``pyproject.toml``;
    storing ``RAW/...`` relative to them breaks when ``load_config`` cannot find
    the same root. Absolute paths are unambiguous on external drives too.
    """
    path = path.resolve()
    root = root.resolve()
    if not has_project_marker(root):
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def auto_filter_map(combos: list[dict[str, Any]]) -> dict[str, str]:
    """Build FILTER1,FILTER2 → short name from flat OBJECT suggestions."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for rec in combos:
        name = rec["suggested_name"]
        base = name
        n = 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        mapping[rec["key"]] = name
    return mapping


def inspect_raw_dir(raw_dir: Path) -> dict[str, Any]:
    """Gather instrument/night facts from FITS headers."""
    raw_dir = Path(raw_dir).expanduser().resolve()
    files = list(_iter_fits(raw_dir))
    if not files:
        raise FileNotFoundError(f"No FITS files in {raw_dir}")

    sample = files[0]
    image_hdu = _detect_image_hdu(sample)
    primary = fits.getheader(sample, 0)
    if image_hdu not in (0, "PRIMARY"):
        try:
            image_hdr = fits.getheader(sample, image_hdu)
        except Exception:
            image_hdr = primary
    else:
        image_hdr = primary

    obstype_key = _pick_keyword(primary, ["OBSTYPE", "IMAGETYP", "IMGTYPE"], "OBSTYPE")
    exptime_key = _pick_keyword(primary, ["EXPTIME", "EXPOSURE", "EXPTIM"], "EXPTIME")
    object_key = _pick_keyword(primary, ["OBJECT", "OBJNAME", "TARGET"], "OBJECT")

    has_f1 = "FILTER1" in primary
    has_f2 = "FILTER2" in primary
    has_filter = "FILTER" in primary or "FILTERS" in primary
    if has_f1 and has_f2:
        filter1_key, filter2_key = "FILTER1", "FILTER2"
    elif has_filter:
        filter1_key = "FILTER" if "FILTER" in primary else "FILTERS"
        filter2_key = filter1_key  # single-filter instruments: key becomes "V,V" style — handled below
    else:
        filter1_key, filter2_key = "FILTER1", "FILTER2"

    biassec_key = _pick_keyword(image_hdr, ["BIASSEC"], "BIASSEC")
    datasec_key = _pick_keyword(image_hdr, ["DATASEC", "TRIMSEC", "CCDSEC"], "DATASEC")

    type_counts: Counter[str] = Counter()
    telescopes: Counter[str] = Counter()
    instruments: Counter[str] = Counter()
    for path in files:
        hdr = fits.getheader(path, 0)
        type_counts[str(hdr.get(obstype_key, "UNKNOWN")).strip().upper()] += 1
        if "TELESCOP" in hdr:
            telescopes[str(hdr["TELESCOP"]).strip()] += 1
        if "INSTRUME" in hdr:
            instruments[str(hdr["INSTRUME"]).strip()] += 1

    # Map common obstype strings
    def find_type(*names: str) -> str:
        for name in names:
            if name in type_counts:
                return name
        return names[0]

    obstypes = {
        "bias": find_type("BIAS", "ZERO"),
        "flat": find_type("FLAT", "DOME FLAT", "SKY FLAT", "FLATFIELD"),
        "science": find_type("OBJECT", "LIGHT", "SCIENCE"),
        "dark": find_type("DARK"),
    }
    darks_present = type_counts.get(obstypes["dark"], 0) > 0 and obstypes["dark"] in type_counts

    combos = discover_filter_combos(
        raw_dir,
        obstype_key=obstype_key,
        flat_value=obstypes["flat"],
        object_key=object_key,
        filter1_key=filter1_key,
        filter2_key=filter2_key,
        exptime_key=exptime_key,
    )
    # Single FILTER keyword: rediscover with duplicated key still works if FILTER1==FILTER2 path
    filter_map = auto_filter_map(combos)

    return {
        "raw_dir": raw_dir,
        "night_id": raw_dir.name,
        "n_files": len(files),
        "type_counts": dict(type_counts),
        "telescope": telescopes.most_common(1)[0][0] if telescopes else "unknown",
        "instrument": instruments.most_common(1)[0][0] if instruments else "unknown",
        "image_hdu": image_hdu,
        "keywords": {
            "obstype": obstype_key,
            "exptime": exptime_key,
            "object": object_key,
            "filter1": filter1_key,
            "filter2": filter2_key,
            "biassec": biassec_key,
            "datasec": datasec_key,
        },
        "obstypes": {
            "bias": obstypes["bias"],
            "flat": obstypes["flat"],
            "science": obstypes["science"],
        },
        "darks_enabled": bool(darks_present),
        "filter_combos": combos,
        "filter_map": filter_map,
        "has_biassec": biassec_key in image_hdr,
        "has_datasec": datasec_key in image_hdr,
    }


def _sibling_reduced_dir(campaign_root: Path, night_id: str) -> Path:
    """Prefer an existing REDUCED/reduced sibling; else create ``reduced/<night>``."""
    for name in ("reduced", "REDUCED"):
        sibling = campaign_root / name
        if sibling.is_dir():
            return sibling / night_id
    return campaign_root / "reduced" / night_id


def resolve_night_paths(raw_dir: Path, project_root: Path) -> tuple[str, Path, Path, Path]:
    """
    Map a selected path onto raw + reduced locations for a night.

    Accepts either:
      data/<night>/raw
      data/<night>              (uses ./raw if present)
      campaign/RAW/<night>      (HDI-style; reduced → campaign/reduced/<night>
                                 or campaign/REDUCED/<night> if that exists)
      any folder of FITS        (night_id = folder name)
    """
    raw_dir = Path(raw_dir).expanduser().resolve()
    data_root = (project_root.parent / "data").resolve()

    if raw_dir.name.lower() == "raw":
        night_root = raw_dir.parent
        night_id = night_root.name
        reduced = night_root / "reduced"
        return night_id, raw_dir, reduced, night_root

    # User selected the night folder itself
    if (raw_dir / "raw").is_dir():
        night_id = raw_dir.name
        return night_id, raw_dir / "raw", raw_dir / "reduced", raw_dir

    # campaign/RAW/<night> or campaign/raw/<night> (FITS directly in night folder)
    if raw_dir.parent.name.lower() == "raw":
        campaign_root = raw_dir.parent.parent
        night_id = raw_dir.name
        reduced = _sibling_reduced_dir(campaign_root, night_id)
        return night_id, raw_dir, reduced, campaign_root

    # Fallback: FITS live directly in this folder
    night_id = raw_dir.name
    # Prefer data/<night>/reduced when under data/
    try:
        raw_dir.relative_to(data_root)
        night_root = data_root / night_id
        return night_id, raw_dir, night_root / "reduced", night_root
    except ValueError:
        return night_id, raw_dir, raw_dir.parent / "reduced" / night_id, raw_dir.parent


def build_config_from_raw(
    raw_dir: Path,
    *,
    project_root: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """
    Create a night YAML from headers alone.

    Returns (config_path, cfg_dict, inspection_info).
    """
    project_root = project_root or find_project_root(Path.cwd())
    night_id, raw_dir, default_reduced, night_root = resolve_night_paths(
        raw_dir, project_root
    )
    info = inspect_raw_dir(raw_dir)
    # Prefer night folder name over literal "raw"
    info["night_id"] = night_id
    info["raw_dir"] = raw_dir
    info["night_root"] = night_root

    if output_dir is None:
        output_dir = default_reduced
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    masters_dir = output_dir / "masters"

    cfg: dict[str, Any] = {
        "night_id": night_id,
        "instrument": info["instrument"],
        "telescope": info["telescope"],
        "paths": {
            "raw_dir": _rel_to_root(raw_dir, project_root),
            "output_dir": _rel_to_root(output_dir, project_root),
            "masters_dir": _rel_to_root(masters_dir, project_root),
        },
        "fits": {"image_hdu": info["image_hdu"], "unit": "adu"},
        "keywords": info["keywords"],
        "obstypes": info["obstypes"],
        "overscan": {"median": True},
        "darks": {"enabled": info["darks_enabled"]},
        "combine": {
            "bias": {
                "method": "average",
                "sigma_clip": True,
                "sigma_clip_low": 5,
                "sigma_clip_high": 5,
            },
            "flat": {
                "method": "average",
                "scale": "inv_median",
                "sigma_clip": True,
                "sigma_clip_low": 5,
                "sigma_clip_high": 5,
            },
        },
        "filter_map": info["filter_map"],
        "science": {
            "include_objects": [],
            "exclude_object_substrings": ["rfocus", "junk", "focus"],
        },
        "wcs": _default_wcs_block(info["instrument"]),
    }

    safe_inst = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(info["instrument"]))
    config_path = project_root / "configs" / f"{safe_inst}_{night_id}.yaml".lower()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Auto-generated from {raw_dir}\n"
        f"# telescope={cfg['telescope']}  instrument={cfg['instrument']}\n"
    )
    with config_path.open("w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)

    return config_path, cfg, info


def format_inspection(info: dict[str, Any], config_path: Path) -> str:
    from .term import S, style

    lines = [
        f"Raw directory : {info['raw_dir']}",
        f"Night ID      : {info['night_id']}",
        f"Telescope     : {info['telescope']}",
        f"Instrument    : {info['instrument']}",
        f"Files         : {info['n_files']}",
        f"Image HDU     : {info['image_hdu']!r}",
        f"Config written: {config_path}",
        "",
        style("Frame counts:", S.BOLD, S.CYAN),
    ]
    for k, n in sorted(info["type_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {n:4d}  {k}")
    lines += ["", style("Filter map (from flats):", S.BOLD, S.CYAN)]
    if not info["filter_map"]:
        lines.append("  (none found)")
    for key, name in info["filter_map"].items():
        combo = next((c for c in info["filter_combos"] if c["key"] == key), None)
        n = combo["n"] if combo else "?"
        top = combo["top_object"] if combo else ""
        lines.append(f"  {key:>12}  →  {name:<12}  ({n} flats, OBJECT={top!r})")
    return "\n".join(lines)
