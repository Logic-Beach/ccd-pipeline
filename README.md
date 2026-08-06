# CCD reduction pipeline

Astropy / [`ccdproc`](https://ccdproc.readthedocs.io/) reduction pipeline for
per-night CCD data (developed with WIYN 0.9m HDI MEF frames).

The calibration sequence follows the
[Astropy CCD Reduction and Photometry Guide](https://www.astropy.org/ccd-reduction-and-photometry-guide/):

| Step | Guide notebooks | What we do |
|------|-----------------|------------|
| Overscan + trim | [01-08](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-08-Overscan.html), [02-02](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-02-Calibrating-bias-images.html) | `BIASSEC` / `DATASEC` via `ccdproc.subtract_overscan` + `trim_image` |
| Master bias | [02-04](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-04-Combine-bias-images-to-make-master.html) | σ-clip average of overscan-calibrated biases |
| Master flats | [05-03](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-03-Calibrating-the-flats.html), [05-04](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html) | Bias-subtract, inv-median scale, combine |
| Science | [06-00](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/06-00-Reducing-science-images.html) | Overscan → trim → bias → flat |

Per-night configs are YAML (paths may be absolute — data can live on an
external drive). The interactive CLI auto-builds a config from FITS headers.

## Data layout

```
<data root>/
  <NIGHT>/
    raw/          # original FITS
    reduced/      # masters/, science/, masters/diagnostics/
ccd-pipeline/     # this repo (configs + code)
```

## Setup

```bash
conda activate astro   # needs astropy, ccdproc, numpy, pyyaml, matplotlib
cd /path/to/ccd-pipeline
pip install -e .
```

## Typical use

```bash
# Pass the night folder or its raw/ directory (any absolute path works)
ccd "/Volumes/Drive/astronomy/2017JUN29"
# or
ccd "/path/to/data/2017JUN29/raw"
```

Or run `ccd` and choose **New night** — folder/file pickers for raw data and
output (cancel falls back to typing a path).

### What auto-config reads from headers
- Telescope / instrument (`TELESCOP`, `INSTRUME`)
- Frame types (`OBSTYPE` / `IMAGETYP` counts)
- Image HDU (e.g. HDI `xy00` vs single-HDU)
- Overscan/trim keywords when present
- **Filter map** from FLAT `FILTER1`/`FILTER2` + `dflat-*` OBJECT names

## Non-interactive commands

```bash
ccd inventory --config configs/wiyn_hdi_2017jun29.yaml
ccd masters   --config configs/wiyn_hdi_2017jun29.yaml --only bias
ccd masters   --config configs/wiyn_hdi_2017jun29.yaml --only flats --filter r
ccd sanity    --config configs/wiyn_hdi_2017jun29.yaml
ccd reduce    --config configs/wiyn_hdi_2017jun29.yaml --limit 2
```

`ccd sanity` checks raw overscan (light leak / uniformity column cuts), then
master bias/flat statistics and row/column profiles. PNGs land under
`reduced/masters/diagnostics/` (`overscan_diag.png`, `master_*_diag.png`).

## Paths on external drives

Absolute paths in the night YAML are used as-is:

```yaml
paths:
  raw_dir: "/Volumes/MyDrive/2017JUN29/raw"
  output_dir: "/Volumes/MyDrive/2017JUN29/reduced"
  masters_dir: "/Volumes/MyDrive/2017JUN29/reduced/masters"
```

Relative paths are resolved from the `ccd-pipeline` project root.

## License / attribution

Reduction methods follow the Astropy CCD Reduction and Photometry Guide
(Astropy Project). This package is a thin, instrument-configurable wrapper
around `astropy` and `ccdproc`.
