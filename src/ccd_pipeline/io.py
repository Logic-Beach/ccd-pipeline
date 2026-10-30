"""FITS I/O helpers for multi-extension (MEF) and single-HDU products.

WIYN HDI raw frames store the detector image in extension ``xy00``; pipeline
products are written as single-HDU FITS. Filter names are resolved from a
night ``filter_map`` (``FILTER1,FILTER2`` → short name such as ``r``).
"""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u


def skip_fits_name(name: str) -> bool:
    """True for names that match ``*.fits`` but must not be opened.

    macOS writes AppleDouble sidecars ``._frame.fits`` beside real frames on
    exFAT. Those start with ``.`` and are not FITS. ``latest.fits`` is a
    placeholder some cameras drop in the raw directory.
    """
    return name.startswith(".") or name.lower() == "latest.fits"


def iter_fits_files(directory: str | Path):
    """Yield real FITS files in ``directory``, skipping dotfiles and ``latest.fits``."""
    directory = Path(directory)
    for path in sorted(directory.glob("*.fits")):
        if skip_fits_name(path.name) or not path.is_file():
            continue
        yield path


def filter_key(header, filter1_key: str = "FILTER1", filter2_key: str = "FILTER2") -> str:
    """Return canonical 'FILTER1,FILTER2' string from a header."""
    f1 = str(header.get(filter1_key, "")).strip()
    f2 = str(header.get(filter2_key, "")).strip()
    return f"{f1},{f2}"


def _wheel_ids_look_valid(key: str) -> bool:
    """True if FILTER1,FILTER2 look like HDI wheel position codes (not corrupted)."""
    parts = [p.strip() for p in str(key).split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False
    bad = {"not available", "n/a", "none", "teldata", "nan"}
    for p in parts:
        pl = p.lower()
        if pl in bad or ":" in p or " " in p:
            return False
        # wheel IDs are small integers like 107, 204 — reject long/float junk
        try:
            v = float(p)
        except ValueError:
            return False
        if not v.is_integer() or abs(v) > 1000:
            return False
    return True


def resolve_filter_name(header, filter_map: dict[str, str], keywords: dict) -> str:
    """Resolve a short filter name for inventory / flats / science.

    Preference order:
      1. ``filter_map`` lookup on FILTER1,FILTER2 (when those IDs look valid)
      2. A name card: ``FILTER``, ``FILTNAME``, or ``FILTERS`` (Hall/NASACam
         uses ``FILTNAME``; HDI science often has ``FILTER`` when wheel IDs
         are corrupted)
      3. dome-flat OBJECT label ``dflat-*``
      4. raw ``FILTER1_FILTER2`` string
    """
    key = filter_key(header, keywords.get("filter1", "FILTER1"), keywords.get("filter2", "FILTER2"))
    if key in filter_map and _wheel_ids_look_valid(key):
        return filter_map[key]
    if key in filter_map:
        return filter_map[key]

    for card in ("FILTER", "FILTNAME", "FILTERS"):
        filt = header.get(card)
        if filt not in (None, ""):
            return str(filt).strip()

    # Fall back to dome-flat OBJECT label if present (dflat-r → r)
    obj = str(header.get(keywords.get("object", "OBJECT"), "")).strip()
    if obj.lower().startswith("dflat-"):
        return obj.split("-", 1)[1]

    if _wheel_ids_look_valid(key):
        return key.replace(",", "_")
    # Corrupted wheel IDs with no FILTER card — still return something stable
    return key.replace(",", "_")

# Single-letter photometric systems used at HDI (must not collide on macOS FS)
_JC_FILTERS = frozenset("UBVRI")  # Johnson–Cousins
_SDSS_FILTERS = frozenset("ugriz")  # Sloan / SDSS


def filter_filename_token(filter_name: str) -> str:
    """Filename token for a master flat (safe on case-insensitive disks).

    * Johnson–Cousins ``U B V R I`` → ``jc_I``, ``jc_R``, …
    * SDSS / Sloan ``u g r i z`` → ``sdss_i``, ``sdss_r``, …
    * everything else (``Wash-C``, ``OIII``, ``6580``, …) → unchanged

    So ``master_flat_jc_I.fits`` and ``master_flat_sdss_i.fits`` stay distinct
    on macOS, where bare ``I`` / ``i`` would overwrite each other.
    """
    name = str(filter_name).strip()
    if not name:
        return "unknown"
    safe = name.replace("/", "-").replace("\\", "-").replace(" ", "_")
    if safe in _JC_FILTERS:
        return f"jc_{safe}"
    if safe in _SDSS_FILTERS:
        return f"sdss_{safe}"
    return safe


def filter_name_from_token(token: str) -> str:
    """Inverse of ``filter_filename_token`` (also accepts older ``uc_`` / ``lc_`` names)."""
    for prefix in ("jc_", "sdss_", "uc_", "lc_", "mx_"):
        if token.startswith(prefix):
            return token[len(prefix) :]
    return token


def master_flat_path(masters_dir: str | Path, filter_name: str) -> Path:
    """Path for a master flat that will not collide on case-insensitive disks."""
    return Path(masters_dir) / f"master_flat_{filter_filename_token(filter_name)}.fits"


def filter_name_from_master_flat_path(path: str | Path, header=None) -> str:
    """Filter name for a master flat: prefer FITS ``FILTER``, else decode filename."""
    if header is not None and header.get("FILTER") not in (None, ""):
        return str(header.get("FILTER")).strip()
    stem = Path(path).stem
    token = stem.replace("master_flat_", "", 1)
    return filter_name_from_token(token)


def read_ccd(
    path: str | Path,
    *,
    image_hdu: str | int | None = "xy00",
    unit: str = "adu",
) -> CCDData:
    """
    Load an HDI MEF or a single-HDU pipeline product as CCDData.

    Raw HDI: empty primary + image extension (default ``xy00``).
    Pipeline products: single PrimaryHDU with image data — pass ``image_hdu=0``.
    """
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        if image_hdu in (0, "PRIMARY") or (image_hdu is None and hdul[0].data is not None):
            data = hdul[0].data
            header = hdul[0].header.copy()
        else:
            # MEF raw: merge primary metadata with image-extension header/data
            header = hdul[0].header.copy()
            image = hdul[image_hdu]
            data = image.data
            header.update(image.header)

    if data is None:
        raise ValueError(f"No image data found in {path} (hdu={image_hdu!r})")

    header["FILENAME"] = path.name
    header["ORIGFILE"] = str(path)
    return CCDData(data, meta=header, unit=u.Unit(unit))


def write_ccd(ccd: CCDData, path: str | Path, overwrite: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ccd.write(path, overwrite=overwrite)
    return path
