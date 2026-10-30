"""Per-frame statistics and whole-frame rejection for raw calibration frames.

This is the Astropy equivalent of the IRAF habit ``imstat @ezero`` → look →
drop the odd frames before ``zerocombine``. Per-pixel σ-clipping inside the
combine handles cosmic rays; it does **not** notice a whole bias that sits
8 ADU low or a flat that saturated. That needs a per-frame table.

Stats come from Astropy: ``sigma_clipped_stats`` (IRAF ``imstat`` with
``nclip``) and ``mad_std`` (a robust standard deviation), computed on the
image area (``DATASEC``) of each raw frame.
https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html
https://docs.astropy.org/en/stable/api/astropy.stats.mad_std.html

Frames rejected here are written to ``exclude_files`` in the night config and
skipped by every master and by science calibration.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats

from .inventory import excluded_files, inventory_night, is_flat_obstype

# imstat-style QC thresholds; override under ``qc:`` in the night YAML.
DEFAULT_QC = {
    "nsigma": 3.0,  # how far (in robust σ of the group) a frame may stray
    "level_floor_adu": 5.0,  # bias/dark level deviation smaller than this is never flagged
    "noise_floor_frac": 0.10,  # noise must also be >10 % above the group to be flagged
    "saturation_adu": 60000.0,  # pixels at/above this are saturated
    "flat_sat_frac": 0.005,  # a flat with more than 0.5 % saturated pixels is flagged
    "flat_min_adu": 2000.0,  # a flat whose level is below this has too little signal
    "flat_structure_frac": 0.25,  # std*/level must differ >25 % from its set to be flagged
    "hot_nsigma": 10.0,  # pixels above median + 10σ count as hot / cosmic-ray hits
}

STAT_COLUMNS = (
    ("median", "median"),
    ("clipped_mean", "mean*"),
    ("clipped_std", "std*"),
    ("mad_std", "mad_std"),
    ("min", "min"),
    ("max", "max"),
    ("n_hot", "hot"),
)


def qc_settings(cfg: dict) -> dict[str, float]:
    out = dict(DEFAULT_QC)
    for key, value in (cfg.get("qc") or {}).items():
        if key in out and value is not None:
            out[key] = float(value)
    return out


def frame_statistics(
    data: np.ndarray, *, hot_nsigma: float = 10.0, saturation_adu: float = 60000.0
) -> dict[str, float]:
    """imstat-style numbers for one image (or image section).

    ``clipped_*`` use ``sigma_clipped_stats(sigma=3)``; ``mad_std`` is the
    MAD-based robust σ; ``n_hot`` counts pixels above median + ``hot_nsigma``·σ
    (cosmic rays, hot pixels, stars in a sky flat); ``frac_sat`` is the
    fraction of pixels at or above ``saturation_adu``.
    """
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        nan = float("nan")
        return {
            "npix": 0, "mean": nan, "median": nan, "std": nan, "mad_std": nan,
            "clipped_mean": nan, "clipped_median": nan, "clipped_std": nan,
            "min": nan, "max": nan, "n_hot": 0, "frac_sat": nan,
        }
    c_mean, c_med, c_std = sigma_clipped_stats(finite, sigma=3.0, maxiters=5)
    robust = float(mad_std(finite))
    sigma = float(c_std) if np.isfinite(c_std) and c_std > 0 else robust
    n_hot = int(np.sum(finite > c_med + hot_nsigma * sigma)) if sigma > 0 else 0
    return {
        "frac_sat": float(np.mean(finite >= saturation_adu)),
        "npix": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std()),
        "mad_std": robust,
        "clipped_mean": float(c_mean),
        "clipped_median": float(c_med),
        "clipped_std": float(c_std),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "n_hot": n_hot,
    }


def _image_area(data: np.ndarray, header, datasec_key: str) -> np.ndarray:
    """``DATASEC`` part of a raw frame (falls back to the whole array)."""
    from .sanity import parse_fits_section

    sec = header.get(datasec_key)
    if not sec:
        return data
    try:
        rows, cols = parse_fits_section(str(sec))
    except (ValueError, TypeError):
        return data
    return data[rows, cols]


def load_frame_section(path: Path, cfg: dict) -> tuple[np.ndarray, Any]:
    """Raw image-area pixels and header for one file."""
    image_hdu = cfg["fits"]["image_hdu"]
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[image_hdu if image_hdu not in (None, "PRIMARY") else 0]
        header = hdul[0].header.copy()
        if hdu is not hdul[0]:
            header.update(hdu.header)
        data = np.asarray(hdu.data, dtype=float)
    return _image_area(data, header, cfg["keywords"]["datasec"]), header


def raw_frame_rows(cfg: dict, kind: str, *, filters: list[str] | None = None) -> list[dict]:
    """One stats row per raw frame of ``kind`` (bias/dark/flat), excluded ones marked."""
    obstype = str((cfg.get("obstypes") or {}).get(kind, kind)).upper()
    settings = qc_settings(cfg)
    excluded = excluded_files(cfg)
    summary = inventory_night(cfg)
    rows: list[dict] = []
    for row in summary["rows"]:
        if kind == "flat":
            if not is_flat_obstype(row["obstype"], obstype):
                continue
        elif row["obstype"] != obstype:
            continue
        if filters is not None and row.get("filter") not in filters:
            continue
        data, _header = load_frame_section(Path(row["path"]), cfg)
        stats = frame_statistics(
            data,
            hot_nsigma=settings["hot_nsigma"],
            saturation_adu=settings["saturation_adu"],
        )
        rows.append(
            {
                "file": row["file"],
                "path": row["path"],
                "kind": kind,
                "object": row.get("object") or "",
                "filter": row.get("filter") or "",
                "exptime": row.get("exptime"),
                "excluded": row["file"] in excluded,
                **stats,
            }
        )
    return rows


def group_key(row: dict, kind: str) -> str:
    """Frames that should look alike: flats by filter *and* OBJECT (dome vs
    twilight sets), darks by exposure, biases all together."""
    if kind == "flat":
        return f"{row.get('filter') or '?'} {row.get('object') or '?'}"
    if kind == "dark":
        exp = row.get("exptime")
        return f"{exp:g}s" if isinstance(exp, (int, float)) else f"{exp}s"
    return "all"


def flat_sets_by_filter(rows: list[dict]) -> dict[str, dict[str, int]]:
    """``{filter: {OBJECT: n}}`` for the flats that are not excluded."""
    sets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row.get("excluded"):
            continue
        sets[row.get("filter") or "?"][row.get("object") or "?"] += 1
    return {f: dict(s) for f, s in sets.items()}


def _robust_center(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)), float(mad_std(arr)) if arr.size > 2 else 0.0


def flag_frame_outliers(rows: list[dict], kind: str, cfg: dict) -> dict[str, list[str]]:
    """Reasons to drop each frame, judged against the other frames in its group.

    Bias/dark: level and noise are compared with the group's median using a
    MAD-based σ (``qc.nsigma``) plus an absolute floor, so near-identical
    frames do not flag harmless 0.5 ADU differences.

    Flats: level differences inside a set are normal (twilight fades), so a
    flat is flagged only for saturation, too little signal, or a relative
    structure (std*/level) unlike the rest of its set — a cloud, a dome lamp
    that was off, or a twilight frame with a strong gradient.
    """
    settings = qc_settings(cfg)
    flags: dict[str, list[str]] = defaultdict(list)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("excluded"):
            continue
        groups[group_key(row, kind)].append(row)

    for _key, members in groups.items():
        if kind == "flat":
            for row in members:
                if row["frac_sat"] > settings["flat_sat_frac"]:
                    flags[row["file"]].append(
                        f"{100 * row['frac_sat']:.1f}% of pixels saturated "
                        f"(≥{settings['saturation_adu']:.0f} ADU)"
                    )
                if row["clipped_median"] < settings["flat_min_adu"]:
                    flags[row["file"]].append(
                        f"level {row['clipped_median']:.0f} ADU below qc.flat_min_adu "
                        f"{settings['flat_min_adu']:.0f}"
                    )
            if len(members) < 3:
                continue
            ratios = [
                m["clipped_std"] / m["clipped_median"] if m["clipped_median"] else float("nan")
                for m in members
            ]
            ratio_med, ratio_sig = _robust_center([r for r in ratios if np.isfinite(r)])
            tol = max(settings["nsigma"] * ratio_sig, settings["flat_structure_frac"] * ratio_med)
            for row, ratio in zip(members, ratios):
                if np.isfinite(ratio) and abs(ratio - ratio_med) > tol:
                    flags[row["file"]].append(
                        f"structure std*/level {100 * ratio:.2f}% vs set {100 * ratio_med:.2f}%"
                    )
            continue

        if len(members) < 3:
            continue
        levels = [m["clipped_median"] for m in members]
        noises = [m["clipped_std"] for m in members]
        level_med, level_sig = _robust_center(levels)
        noise_med, noise_sig = _robust_center(noises)
        level_tol = max(settings["nsigma"] * level_sig, settings["level_floor_adu"])
        noise_tol = max(settings["nsigma"] * noise_sig, settings["noise_floor_frac"] * noise_med)
        for row in members:
            d_level = row["clipped_median"] - level_med
            if abs(d_level) > level_tol:
                flags[row["file"]].append(
                    f"level {d_level:+.1f} ADU from the group median {level_med:.1f}"
                )
            d_noise = row["clipped_std"] - noise_med
            if d_noise > noise_tol:
                flags[row["file"]].append(
                    f"noise std* {row['clipped_std']:.2f} vs group {noise_med:.2f} ADU"
                )
    return dict(flags)


def format_frame_table(rows: list[dict], flags: dict[str, list[str]], *, kind: str) -> str:
    """imstat-like table, one raw frame per line, flagged frames marked."""
    if not rows:
        return f"  (no {kind} frames)"
    name_w = max(len(r["file"]) for r in rows)

    def _tag(row: dict) -> str:
        if kind == "bias":
            return row.get("object") or ""
        if kind == "dark":
            return f"{group_key(row, kind)} {row.get('object') or ''}".strip()
        return group_key(row, kind)

    tag_w = min(28, max(5, *(len(_tag(r)) for r in rows)))
    head = f"  {'file':<{name_w}}  {'group':<{tag_w}}"
    for _key, label in STAT_COLUMNS:
        head += f"  {label:>9}"
    head += "  note"
    lines = [head]
    for row in rows:
        tag = _tag(row)[:tag_w]
        line = f"  {row['file']:<{name_w}}  {tag:<{tag_w}}"
        for key, _label in STAT_COLUMNS:
            value = row[key]
            if key == "n_hot":
                line += f"  {int(value):>9d}"
            elif abs(value) >= 1e5 or (abs(value) < 1e-2 and value != 0):
                line += f"  {value:>9.3g}"
            else:
                line += f"  {value:>9.2f}"
        if row.get("excluded"):
            line += "  EXCLUDED (exclude_files)"
        elif row["file"] in flags:
            line += "  ⚠ " + "; ".join(flags[row["file"]])
        lines.append(line)
    lines.append(
        "  mean*/std*: 3σ-clipped (astropy.stats.sigma_clipped_stats);  "
        "mad_std: robust σ;  hot: pixels > median + 10σ"
    )
    return "\n".join(lines)


def sanity_check_frames(
    cfg: dict,
    *,
    kinds: tuple[str, ...] = ("bias", "dark", "flat"),
    filters: list[str] | None = None,
    show_command: bool = True,
) -> dict[str, Any]:
    """Print an imstat-style table per frame type and flag whole-frame outliers."""
    from .term import dim, heading, info, show_ccd, subheading, success, warn

    heading("Raw frame QC (per-frame statistics)")
    if show_command:
        flags_cli = ["--frames-only"]
        for filt_name in filters or []:
            flags_cli.extend(["--filter", filt_name])
        show_ccd(cfg, "sanity", *flags_cli)
    print(
        dim(
            "  IRAF habit: imstat @ezero, drop frames with an odd level or high std dev,\n"
            "  then zerocombine. Here: astropy.stats.sigma_clipped_stats on DATASEC of each\n"
            "  raw frame; a frame is flagged when its level or noise sits outside the group."
        )
    )
    report: dict[str, Any] = {"kinds": {}, "flagged": {}, "n_frames": 0, "flat_sets": {}}
    for kind in kinds:
        if kind == "dark" and not (cfg.get("darks") or {}).get("enabled", False):
            continue
        rows = raw_frame_rows(cfg, kind, filters=filters if kind == "flat" else None)
        flags = flag_frame_outliers(rows, kind, cfg)
        subheading(f"{kind} frames — {len(rows)}")
        print(format_frame_table(rows, flags, kind=kind))
        report["kinds"][kind] = {"rows": rows, "flags": flags}
        report["n_frames"] += len(rows)
        for name, reasons in flags.items():
            report["flagged"][name] = {"kind": kind, "reasons": reasons}
        if flags:
            warn(f"  {len(flags)} {kind} frame(s) stand out from the rest.")
        elif rows:
            success(f"  {kind} frames are consistent with each other.")
        if kind == "flat":
            sets = flat_sets_by_filter(rows)
            report["flat_sets"] = sets
            mixed = {f: s for f, s in sets.items() if len(s) > 1}
            for filt, members in sorted(mixed.items()):
                detail = ", ".join(f"{obj} ×{n}" for obj, n in sorted(members.items()))
                warn(
                    f"  filter '{filt}' has {len(members)} flat sets ({detail}). "
                    "The master flat combines them all; dome and twilight flats have "
                    "different illumination — exclude one set unless mixing is intended."
                )
    already = excluded_files(cfg)
    if already:
        info(f"  exclude_files already lists {len(already)} frame(s).")
    return report


def offer_exclude_frames(cfg: dict, config_path: Path, report: dict[str, Any]) -> list[str]:
    """Ask which flagged frames to drop; write ``exclude_files`` to the night YAML."""
    import sys

    flagged = report.get("flagged") or {}
    mixed = {f: s for f, s in (report.get("flat_sets") or {}).items() if len(s) > 1}
    if (not flagged and not mixed) or not sys.stdin.isatty():
        return []

    from .night_init import save_exclude_files
    from .prompt import choose, choose_many
    from .term import info

    options: list[tuple[str, str]] = []
    if flagged:
        options.append(("all", f"Exclude all {len(flagged)} flagged frame(s)"))
        options.append(("pick", "Choose which flagged frames to exclude"))
    if mixed:
        options.append(("set", "Exclude whole flat sets (by OBJECT name)"))
    options.append(("none", "Keep every frame"))
    choice = choose("Frame QC:", options)
    if choice == "none":
        return []

    names: list[str] = sorted(flagged)
    if choice == "pick":
        names = choose_many(
            "Flagged frames to exclude:",
            [(n, f"{n}  — {'; '.join(flagged[n]['reasons'])}") for n in names],
        )
    elif choice == "set":
        flat_rows = ((report.get("kinds") or {}).get("flat") or {}).get("rows") or []
        set_options: list[tuple[str, str]] = []
        for filt, members in sorted(mixed.items()):
            for obj, n in sorted(members.items()):
                set_options.append((f"{filt}\t{obj}", f"filter {filt}: {obj} ({n} frames)"))
        picked_sets = choose_many("Flat sets to exclude:", set_options)
        wanted = {tuple(p.split("\t", 1)) for p in picked_sets}
        names = sorted(
            r["file"]
            for r in flat_rows
            if ((r.get("filter") or "?"), (r.get("object") or "?")) in wanted
        )
        if names:
            info(f"  {len(names)} frame(s) in {len(wanted)} set(s)")
    if not names:
        return []
    merged = save_exclude_files(config_path, cfg, names)
    info(f"  exclude_files now has {len(merged)} frame(s); saved to {config_path}")
    return names


def check_master_noise(
    master_mad_std: float, raw_mad_std: float, n_combined: int | None
) -> list[tuple[str, str]]:
    """The notebook rule: a master must be quieter than one raw frame.

    Pure read noise averages down as 1/√N; a master well above that floor
    still carries fixed pattern (fine) or per-frame processing noise (not).
    """
    notes: list[tuple[str, str]] = []
    if not (np.isfinite(master_mad_std) and np.isfinite(raw_mad_std)) or raw_mad_std <= 0:
        return notes
    ratio = master_mad_std / raw_mad_std
    floor = raw_mad_std / np.sqrt(n_combined) if n_combined else float("nan")
    floor_txt = f"; pure-noise floor {floor:.2f} ADU for N={n_combined}" if n_combined else ""
    if ratio < 1.0:
        level = "ok" if (not n_combined or master_mad_std < 2.5 * floor) else "warn"
        notes.append(
            (
                level,
                f"master mad_std {master_mad_std:.2f} ADU vs one raw frame "
                f"{raw_mad_std:.2f} ADU (×{ratio:.2f}{floor_txt})",
            )
        )
    else:
        notes.append(
            (
                "fail",
                f"master mad_std {master_mad_std:.2f} ADU is not below one raw frame "
                f"({raw_mad_std:.2f} ADU) — combining did not reduce noise",
            )
        )
    return notes
