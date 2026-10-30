"""Auto-build a night config from a raw FITS directory.

Scans headers for telescope/instrument, frame-type counts, image HDU,
``BIASSEC``/``DATASEC``, and a filter map from flats — then writes YAML under
the night data folder's ``configs/``. When that folder is outside this
repository, the same file is also written to the repo ``configs/``. Data paths
may be absolute (external drives) or relative to the project root.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from astropy.io import fits

from .config import find_project_root, has_project_marker
from .filters_discover import discover_filter_combos
from .inventory import is_flat_obstype
from .io import iter_fits_files
from .wcs_solve import plate_scale_arcsec, unbinned_pixscal


def _iter_fits(raw_dir: Path):
    yield from iter_fits_files(raw_dir)


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


def _default_wcs_block(
    instrument: str,
    *,
    detector: str = "",
    telescope: str = "",
    ccdsum: str | None = None,
    pixscal_unbinned: float | None = None,
) -> dict[str, Any]:
    """Instrument-aware defaults for offline ``solve-field`` plate-solving."""
    scale = plate_scale_arcsec(
        detector=detector,
        instrument=instrument,
        telescope=telescope,
        ccdsum=ccdsum,
        pixscal_unbinned=pixscal_unbinned,
    )
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
        "timeout_sec": 300,
        "verbose": False,
        "show_solver_output": False,
        "include_objects": [],
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


_PLACEHOLDER_INSTRUMENTS = {"", "test", "unknown", "none", "n/a", "na"}


def camera_label(instrument: str, detector: str = "") -> str:
    """Name used in the config filename.

    ``INSTRUME`` is sometimes a placeholder (``test``). The detector card is
    the camera in that case.
    """
    inst = str(instrument or "").strip()
    det = str(detector or "").strip()
    if inst.lower() in _PLACEHOLDER_INSTRUMENTS and det:
        return det
    return inst or det or "unknown"


def night_config_paths(
    *,
    night_root: Path,
    project_root: Path,
    filename: str,
) -> list[Path]:
    """Paths for a night YAML: the data folder, then the repo if it differs."""
    primary = (Path(night_root) / "configs" / filename).resolve()
    paths = [primary]
    project_root = Path(project_root).resolve()
    if has_project_marker(project_root):
        repo_copy = (project_root / "configs" / filename).resolve()
        if repo_copy != primary:
            paths.append(repo_copy)
    return paths


def config_mirror_paths(config_path: Path, cfg: dict[str, Any] | None = None) -> list[Path]:
    """The file being edited, plus the repo copy and the night-folder copy."""
    config_path = Path(config_path).resolve()
    paths = [config_path]
    project_root = find_project_root(Path.cwd())
    name = config_path.name
    candidates: list[Path] = []
    if has_project_marker(project_root):
        candidates.append(project_root / "configs" / name)
    raw = ((cfg or {}).get("paths") or {}).get("raw_dir")
    if raw:
        _night_id, _raw_dir, _reduced, night_root = resolve_night_paths(
            Path(raw), project_root
        )
        candidates.append(Path(night_root) / "configs" / name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in paths:
            paths.append(resolved)
    return paths


def _write_yaml(path: Path, header: str, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(header)
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


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

    has_f1 = primary.get("FILTER1") not in (None, "")
    has_f2 = primary.get("FILTER2") not in (None, "")
    if has_f1 and has_f2:
        filter1_key, filter2_key = "FILTER1", "FILTER2"
    else:
        # Single-name instruments (Hall ``FILTNAME``, some ``FILTER``/``FILTERS``).
        filter1_key = filter2_key = "FILTER1"
        for card in ("FILTER", "FILTNAME", "FILTERS"):
            if primary.get(card) not in (None, ""):
                filter1_key = filter2_key = card
                break

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

    flat_keys = [k for k in type_counts if is_flat_obstype(k)]
    obstypes = {
        "bias": find_type("BIAS", "ZERO"),
        "flat": max(flat_keys, key=type_counts.__getitem__) if flat_keys else "FLAT",
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
        "detector": str(primary.get("DETECTOR", "")).strip(),
        "ccdsum": str(primary.get("CCDSUM", "")).strip() or None,
        "pixscal": unbinned_pixscal(primary),
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
            "dark": obstypes["dark"],
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
        "detector": info.get("detector") or "",
        "ccdsum": info.get("ccdsum") or "",
        "telescope": info["telescope"],
        "paths": {
            "raw_dir": _rel_to_root(raw_dir, project_root),
            "output_dir": _rel_to_root(output_dir, project_root),
            "masters_dir": _rel_to_root(masters_dir, project_root),
        },
        "fits": {"image_hdu": info["image_hdu"], "unit": "adu"},
        "keywords": info["keywords"],
        "obstypes": info["obstypes"],
        # median per row, then a Legendre line fitted along the rows (IRAF-style);
        # model: none would subtract each row's own noisy median.
        "overscan": {"median": True, "model": "legendre", "order": 1},
        "darks": {"enabled": info["darks_enabled"]},
        # Raw frames (file names) to leave out of every master and the science
        # list; the frame QC step fills this in.
        "exclude_files": [],
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
            "dark": {
                "method": "average",
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
        "fringe": {
            "filters": ["z", "i"],
            "min_frames": 3,
            "select": "all",  # blanks | science | all
            "blank_substrings": ["blank", "sky", "empty"],
            "blanks_fallback": "all",
            "airmass_tol": 0.15,
            "time_tol_hours": 1.5,
            "sky_tol_frac": 0.3,
            "min_sky": 40.0,
            "min_exptime": 20.0,
            "star_nsigma": 5.0,
            "wing_nsigma": 2.5,
            "dilate_pix": 16,
            "star_smooth_sigma": 0.0,
            "reject_high": 0,
            "hole_fill": "data",
            "residual_nsigma": 0.0,
        },
        "wcs": _default_wcs_block(
            info["instrument"],
            detector=str(info.get("detector") or ""),
            telescope=str(info.get("telescope") or ""),
            ccdsum=info.get("ccdsum"),
            pixscal_unbinned=info.get("pixscal"),
        ),
    }

    label = camera_label(str(info["instrument"]), str(info.get("detector") or ""))
    safe_inst = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    filename = f"{safe_inst}_{night_id}.yaml".lower()
    config_paths = night_config_paths(
        night_root=night_root, project_root=project_root, filename=filename
    )
    header = (
        f"# Auto-generated from {raw_dir}\n"
        f"# telescope={cfg['telescope']}  instrument={cfg['instrument']}"
        f"  detector={cfg['detector'] or '?'}  ccdsum={cfg['ccdsum'] or '?'}\n"
    )
    for config_path in config_paths:
        _write_yaml(config_path, header, cfg)
    info["config_paths"] = config_paths

    return config_paths[0], cfg, info


def save_overscan_biassec(
    config_path: Path,
    cfg: dict[str, Any],
    biassec: str | None,
    *,
    datasec: str | None = None,
    update_datasec: bool = False,
) -> None:
    """Store ``overscan.biassec`` (and optionally ``datasec``) in the night YAML.

    ``biassec=None`` removes the overscan override so later steps use the header
    ``BIASSEC``. Pass ``update_datasec=True`` to write or clear ``overscan.datasec``
    (the trim region) the same way.
    """
    over = cfg.setdefault("overscan", {})
    if biassec:
        over["biassec"] = biassec
    else:
        over.pop("biassec", None)
    if update_datasec:
        if datasec:
            over["datasec"] = datasec
        else:
            over.pop("datasec", None)

    config_path = Path(config_path)
    text = config_path.read_text()
    data = yaml.safe_load(text) or {}
    disk_over = dict(data.get("overscan") or {})
    if biassec:
        disk_over["biassec"] = biassec
    else:
        disk_over.pop("biassec", None)
    if update_datasec:
        if datasec:
            disk_over["datasec"] = datasec
        else:
            disk_over.pop("datasec", None)
    data["overscan"] = disk_over
    header = "".join(line for line in text.splitlines(True) if line.startswith("#"))
    for path in config_mirror_paths(config_path, cfg):
        _write_yaml(path, header, data)


def save_exclude_files(
    config_path: Path, cfg: dict[str, Any], files: list[str]
) -> list[str]:
    """Store ``exclude_files`` (frame-QC rejects) in the night YAML and in ``cfg``.

    Names are merged with any already listed, by basename, and written to
    every mirror of the config. Returns the merged, sorted list.
    """
    current = cfg.get("exclude_files") or []
    if isinstance(current, str):
        current = [current]
    merged = sorted({Path(str(f)).name for f in list(current) + list(files) if str(f).strip()})
    cfg["exclude_files"] = merged

    config_path = Path(config_path)
    text = config_path.read_text()
    data = yaml.safe_load(text) or {}
    data["exclude_files"] = merged
    header = "".join(line for line in text.splitlines(True) if line.startswith("#"))
    for path in config_mirror_paths(config_path, cfg):
        _write_yaml(path, header, data)
    return merged


def update_config_filter_map(
    config_path: Path, cfg: dict[str, Any], filter_map: dict[str, str]
) -> None:
    """Rewrite ``filter_map`` in an already-written night YAML."""
    cfg["filter_map"] = dict(filter_map)
    text = Path(config_path).read_text()
    header = "".join(line for line in text.splitlines(True) if line.startswith("#"))
    payload = {key: value for key, value in cfg.items() if not str(key).startswith("_")}
    for path in config_mirror_paths(config_path, cfg):
        _write_yaml(path, header, payload)


def format_inspection(info: dict[str, Any], config_path: Path) -> str:
    from .term import S, style

    lines = [
        f"Raw directory : {info['raw_dir']}",
        f"Night ID      : {info['night_id']}",
        f"Telescope     : {info['telescope']}",
        f"Instrument    : {info['instrument']}",
        f"Detector      : {info.get('detector') or '?'}",
        f"Binning       : {info.get('ccdsum') or '?'}",
        f"Files         : {info['n_files']}",
        f"Image HDU     : {info['image_hdu']!r}",
        f"Config written: {config_path}",
        *[
            f"Config copy   : {copy}"
            for copy in info.get("config_paths") or []
            if Path(copy).resolve() != Path(config_path).resolve()
        ],
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

    # Science FILTER card summary (helps when wheel IDs are missing/corrupt)
    sci_filters = _science_filter_card_counts(
        info["raw_dir"],
        obstype_key=info["keywords"]["obstype"],
        science_value=info["obstypes"]["science"],
    )
    flat_names = set(info["filter_map"].values())
    if sci_filters:
        lines += ["", style("Science FILTER card (may lack flats):", S.BOLD, S.CYAN)]
        for name, n in sorted(sci_filters.items(), key=lambda kv: (-kv[1], kv[0])):
            note = "" if name in flat_names else "  ⚠ no matching dome flat this night"
            lines.append(f"  {n:4d}  {name}{note}")
    return "\n".join(lines)


def _science_filter_card_counts(
    raw_dir: Path,
    *,
    obstype_key: str,
    science_value: str,
) -> dict[str, int]:
    """Count OBJECT/science frames by the FITS FILTER card (not wheel IDs)."""
    counts: Counter[str] = Counter()
    for path in _iter_fits(Path(raw_dir)):
        hdr = fits.getheader(path, 0)
        if str(hdr.get(obstype_key, "")).strip().upper() != str(science_value).upper():
            continue
        for card in ("FILTER", "FILTNAME", "FILTERS"):
            filt = hdr.get(card)
            if filt not in (None, ""):
                counts[str(filt).strip()] += 1
                break
    return dict(counts)
