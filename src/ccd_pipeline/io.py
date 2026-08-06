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


def filter_key(header, filter1_key: str = "FILTER1", filter2_key: str = "FILTER2") -> str:
    """Return canonical 'FILTER1,FILTER2' string from a header."""
    f1 = str(header.get(filter1_key, "")).strip()
    f2 = str(header.get(filter2_key, "")).strip()
    return f"{f1},{f2}"


def resolve_filter_name(header, filter_map: dict[str, str], keywords: dict) -> str:
    key = filter_key(header, keywords.get("filter1", "FILTER1"), keywords.get("filter2", "FILTER2"))
    if key in filter_map:
        return filter_map[key]
    # Fall back to dome-flat OBJECT label if present (dflat-r → r)
    obj = str(header.get(keywords.get("object", "OBJECT"), "")).strip()
    if obj.lower().startswith("dflat-"):
        return obj.split("-", 1)[1]
    return key.replace(",", "_")


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
