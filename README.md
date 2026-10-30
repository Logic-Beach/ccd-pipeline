# CCD reduction pipeline
Built on
[Astropy](https://docs.astropy.org/) /
[ccdproc](https://ccdproc.readthedocs.io/), following the
[Astropy CCD Reduction Guide](https://www.astropy.org/ccd-reduction-and-photometry-guide/).

## Installation

Copy-paste these commands once. Then keep using `conda activate astro` in every
new terminal.

### Pipeline (required)

```bash
# 1. Install Miniconda 
#    https://docs.conda.io/en/latest/miniconda.html

# 2. Clone this repository
cd ~
git clone https://github.com/Logic-Beach/ccd-pipeline.git
cd ccd-pipeline

# 3. Conda env + packages
conda create -n astro python=3.12 -y
conda activate astro
conda install -c conda-forge astropy ccdproc numpy scipy matplotlib pyyaml -y

# If ccdproc / etc failed to install via conda:
#   pip install ccdproc / etc

# 4. Install this package (make sure you're in the right 'ccd-pipeline' directory)
pip install -e .

# Optional (tests):
#   pip install -e ".[dev]"

# 5. Check install is complete
ccd --help
```

After that, `ccd` works from any folder while `astro` environment is active. ```conda activate astro``` to activate the environment

### Image examination

[Ginga](https://ginga.readthedocs.io/) is installed with the pipeline (`ginga[qt6]` in `pyproject.toml`). It is the on-screen replacement for IRAF `imexamine`: star FWHM and radial profiles. The Qt extra is required or the window never opens. (Plugins > Pick > choose a star)

[SAOImage DS9](https://sites.google.com/cfa.harvard.edu/saoimageds9) is the frame display. 
To run from the terminal The Mac app will need to be in your `PATH` and linked to the conda astro env:

```bash
conda activate astro
ln -sf /Applications/SAOImageDS9.app/Contents/MacOS/ds9 "$CONDA_PREFIX/bin/ds9"
```

DS9 stores the scale it opens with in `~/.ds9/ds9.<version>.prf`. The best that matches our data is zscale for visibility. In `ds9.8.7.prf` that line was minmax:

```tcl
array set pscale { datasec 1 mode minmax scope local log 1000 type linear }
```
Change that to zscale:

```tcl
array set pscale { datasec 1 mode zscale scope local log 1000 type linear }
```

The same change from the DS9 window is **Scale → ZScale**, then **Edit → Preferences → Save Preferences**. One file, without changing the saved default: `ds9 -scale zscale <frame>.fits`.

### Install [Astrometry.net](https://astrometry.net/) offline WCS solver

Only needed for `ccd wcs`. `solve-field` is Homebrew `astrometry-net` (`/opt/homebrew/bin/solve-field`). The index files are the 301 FITS files in `~/astrometry-indexes`. Homebrew's `astrometry.cfg` already has `add_path /Users/null/astrometry-indexes`, and night configs leave `wcs.index_dir` null, so `ccd wcs` uses that directory.

Quad size below is `SCALE_L`–`SCALE_U` from the index headers, in arcminutes. The same two-digit code is the same angular size in every series.

| Files | Catalog | Quads |
| ----- | ------- | ----- |
| `index-4107` … `index-4119` | Tycho-2 | 22′–30′ (`4107`) through 1400′–2000′ (`4119`). One file each. |
| `index-4204` … `index-4219` | 2MASS | 8′–11′ (`4204`, 48 tiles) through 1400′–2000′ (`4219`). `4205`–`4207` are 12 tiles; `4208`–`4219` are one file each. |
| `index-5203` … `index-5206` | Gaia | 5.6′–8′ (`5203`), 8′–11′ (`5204`), 11′–16′ (`5205`), 16′–22′ (`5206`). 48 tiles each. |

There is no `index-4203` and no Tycho-2 index smaller than `4107`.

| Camera | Field | Plate scale | Index whose quads contain that field |
| ------ | ----- | ----------- | ------------------------------------ |
| HDI | ~29′ | 0.43″/pix | `4107`, `4207` (22′–30′) |
| Hall E2V, `CCDSUM 2 2` | ~25′ | **0.739″/pix** | `4107` (the solve that measured 0.739), `4207` |
| S2KB, unbinned | 20.5′ | 0.60″/pix | `4206`, `5206` (16′–22′) |
| S2KB, `CCDSUM 2 2` | 20.5′ | 1.20″/pix | `4206`, `5206` |

`4107` / `4207` start at 22′, which is larger than the S2KB chip, so those two do not solve S2KB. A scale window of 0.43″/pix rejects a correct S2KB match. Hall's header `PIXSCAL = 0.327` is unbinned and stale (×2 = 0.654), and the HDI default times binning (0.86) is also outside the real 0.739″/pix. Night init stores 0.369″/pix unbinned for Hall. `ccd wcs` widens the scale window when `PIXSCAL` disagrees with the YAML scale, so the real scale stays inside the search. Smaller quads already on disk, for a field that fails at the matching index, are `4206`/`5206`, then `4205`/`5205`, then `4204`/`5204`, then `5203`.

## Quick start

```bash
conda activate astro
cd /path/to/your/night/of/data   # e.g 2017JUN29 folder with RAW/ and configs/
ccd                          # interactive processing: New night → answer Y/n
```

Or you can run specific steps once a config exists:

```bash
ccd inventory --config configs/hdi_2017jun29.yaml
ccd masters   --config configs/hdi_2017jun29.yaml
ccd sanity    --config configs/hdi_2017jun29.yaml
ccd reduce    --config configs/hdi_2017jun29.yaml
ccd fringe    --config configs/hdi_2017jun29.yaml --filter z --filter i
ccd wcs       --config configs/hdi_2017jun29.yaml
```

Useful flags:

```bash
ccd reduce --config configs/hdi_2017jun29.yaml --limit 2
ccd sanity --config configs/hdi_2017jun29.yaml --science-only --filter z
ccd fringe --config configs/hdi_2017jun29.yaml --filter z --apply   # only after master looks clean
ccd fringe --config configs/hdi_2017jun29.yaml --filter z --apply-only --limit 2
ccd fringe --config configs/hdi_2017jun29.yaml --filter z           # TTY: UT / RA / Dec table, pick frames
ccd fringe --config configs/hdi_2017jun29.yaml --filter z --yes     # skip picker; use airmass/time/sky cuts
ccd wcs    --config configs/hdi_2017jun29.yaml --object APSExS-F-1 --overwrite
ccd sanity --config configs/hdi_2017jun29.yaml --frames-only            # per-frame stats, flag/exclude odd frames
ccd imstat RAW/2017JUN29/*.fits --section "[100:1000,100:1000]"         # IRAF imstat equivalent (quote the section in zsh)
```

IRAF habits and their Astropy equivalents here:

| IRAF                                   | Pipeline                                                                                      |
| -------------------------------------- | --------------------------------------------------------------------------------------------- |
| `imstat @ezero` → drop odd frames      | `ccd sanity --frames-only` (or the wizard step) → `exclude_files:` in the night YAML           |
| `imstat file [sec]`                    | `ccd imstat file --section [x1:x2,y1:y2]` — `astropy.stats.sigma_clipped_stats`, `mad_std`    |
| `ccdproc` overscan `function=legendre` | `overscan.model: legendre`, `overscan.order: 1` (degree; IRAF `order=1` constant is `order: 0`) |
| `zerocombine` / `combine`              | `ccd masters` — `ccdproc.combine` with σ-clipping                                              |
| `hedit … filter u add+`                | `filter_map:` in the night YAML                                                               |
| `implot :l 500 550` on a reduced frame | `ccd sanity --science-only` — row/column bands at 1/4, 1/2, 3/4 of the frame                   |

To check focus on a reduced frame, open it in Ginga and start **Plugins → Analysis → Pick**. Click a star. The **FWHM** tab fits a Gaussian to the horizontal and vertical cuts (widths are in pixels, in the plot title). The **Radial** tab is the radial profile. Click another star to measure that one. `a` and `c` are display keys (reset the stretch, recenter), not the old IRAF `imexamine` keys.

```bash
ginga reduced/<night>/science/<frame>.fits
```

## Pipeline steps


| Step         | Command                    | Output                                        |
| ------------ | -------------------------- | --------------------------------------------- |
| Inventory    | (wizard / `ccd inventory`) | Frame counts                                  |
| Frame QC     | `ccd sanity --frames-only` | Per-frame stats table; `exclude_files` in the night YAML |
| Master bias  | `ccd masters`              | `masters/master_bias.fits`                    |
| Master flats | `ccd masters`              | `masters/master_flat_jc_I.fits`, `master_flat_sdss_i.fits`, … |
| Sanity       | `ccd sanity`               | Diagnostics under `masters/diagnostics/`      |
| Science      | `ccd reduce`               | `science/p-{night}.{object}.{filter}[.NN].fits` |
| Fringe (i/z) | `ccd fringe`               | `master_fringe_*`; apply adds `f` (`p-` → `pf-`) |
| WCS          | `ccd wcs`                  | WCS in headers; adds `w` (`p-`/`pf-` → `pw-`/`pfw-`); `wcs_solve.log` |


## Data layout

```text
campaign/
  RAW/<night>/                 raw FITS
  reduced/<night>/
    masters/                   master_bias, master_flat_*, diagnostics/
    science/                   calibrated frames (p- processed, f fringe, w WCS)
    ccd_pipeline.log           mirror of terminal processing output
    wcs_solve.log
    wcs_logs/
  configs/<night>.yaml
```

A new night writes that YAML under the data folder's `configs/`. When you run from this repository and the data lives elsewhere, the same file is also copied to the repo `configs/`. Later edits (filter names, overscan region) update both. Absolute paths in the config YAML file are preferred (external drives work). Relative paths resolve from the parent of `configs/`.

## Methodology

Reference:
[Astropy CCD Reduction Guide](https://www.astropy.org/ccd-reduction-and-photometry-guide/)
([notebooks](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/)).
Fringe follows IRAF / NOIRLab / Astro-WISE (not in the Astropy guide).

### Order of operations

```text
PHASE A — build masters  (ccd masters)
  raw biases  → overscan + trim → combine           → master_bias
  raw darks   → overscan + trim → −bias → combine   → master_dark
              (only when the night has dark frames; scaled by exposure)
  raw flats   → overscan + trim → −bias → −dark → inv-median combine
              → master_flat_<filter>
  inspect with ccd sanity

PHASE B — calibrate science  (ccd reduce → fringe → wcs)
  raw science → overscan + trim → −bias → −dark → ÷ flat
              → [optional] − sky×fringe   (additive; after flat; i/z)
              → [optional] plate-solve WCS
```

Build masters **before** reducing science. Flat before fringe. Do not reorder.

### What each step does


| Step            | Practice                                                    | Guide / refs                                                                                                                                                                                                                         |
| --------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Frame QC        | Per-frame `sigma_clipped_stats` of raw bias/dark/flat; odd frames → `exclude_files` | [astropy.stats](https://docs.astropy.org/en/stable/stats/) |
| Overscan + trim | Median per row of `BIASSEC`, fit a Legendre line along the rows, subtract the fit; trim to `DATASEC` (biases too) | [01-08](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-08-Overscan.html), [02-02](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-02-Calibrating-bias-images.html), [ccdproc.subtract_overscan](https://ccdproc.readthedocs.io/en/stable/reduction_toolbox.html#overscan-subtraction) |
| Master bias     | σ-clip average; median ≈ 0                                  | [02-04](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-04-Combine-bias-images-to-make-master.html)                                                                                                    |
| Master dark     | −bias, scale to one exposure, σ-clip average. Used only when dark frames are present | [ccdproc.subtract_dark](https://ccdproc.readthedocs.io/en/stable/api/ccdproc.subtract_dark.html) |
| Master flats    | −bias, −dark when present, inv-median scale, combine; median ≈ 1 | [05-03](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-03-Calibrating-the-flats.html), [05-04](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html) |
| Science         | Overscan + trim, subtract master bias, subtract the dark scaled by exposure when darks exist, divide by the master flat for that filter. Prefix `p-` | [06-00](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/06-00-Reducing-science-images.html)                                                                                                               |
| Fringe          | Star-masked sky-scaled median → `science − sky×F`           | [IRAF rmfringe](https://iraf.readthedocs.io/en/latest/tasks/nfextern/msctools/rmfringe.html), [NOIRLab SOI](https://noirlab.edu/science/index.php/programs/ctio/instruments/soi/evaluating-and-reducing-soi-images)                  |
| WCS             | Offline `solve-field` + header pointing                     | [astrometry.net](https://astrometry.net/)                                                                                                                                                                                            |


Sanity expectations: overscan clean; master bias ~0 and quieter than one raw
bias (ideally close to raw/√N); master flat ~1; strong large-scale sky
structure in i/z often means fringing.

### Overscan fitting

`ccdproc.subtract_overscan` first takes one median per row of the overscan.
That median estimates the pedestal. With `overscan.model: none` the estimate
is subtracted as-is. Its uncertainty is about the read noise divided by
√(number of overscan columns), a bit larger for a median than a mean. S2KB's
usable overscan is 21 columns (`[1036:1056]`); at ~6 ADU read noise the
row-median error is ~1.5 ADU rms. The same error is left on every pixel of
that row, so one frame shows horizontal stripes. Each frame has its own
overscan, so the stripes change from frame to frame and subtracting the
master bias does not remove them. Stacking science frames does average them
down.

The default `overscan.model: legendre`, `overscan.order: 1` fits a straight
line to those row medians and subtracts the line (IRAF `ccdproc`/`colbias`
with `function=legendre`). The line follows a slow drift and does not follow
the per-row noise. Use `model: none` only when the overscan is wide enough
that one row's median is quiet, or when the bias really jumps from row to row.

### Frame QC and `exclude_files`

Per-pixel σ-clipping inside the combine removes cosmic rays; it does not
notice a whole bias that sits low or a saturated flat. `ccd sanity
--frames-only` (and the matching wizard steps) prints an `imstat`-style table
per frame type, flags level/noise outliers (bias, dark), saturation / low
signal / odd structure (flats), and warns when a filter mixes dome and
twilight flat sets. Frames you reject are written to `exclude_files:` in the
night YAML and skipped by every master and by `ccd reduce`.

`OBJECT` values matching `junk` / `rfocus` / `focus` (see
`science.exclude_object_substrings`) are skipped for bias, flats, and science.
`junk` is always excluded.

## Config snippets

```yaml
overscan:
  median: true                    # median (not mean) of each overscan row
  model: legendre                 # legendre | chebyshev | polynomial | none (raw per-row)
  order: 1                        # polynomial degree: 0 constant, 1 line
  biassec: '[1036:1056,1:1024]'   # optional: usable overscan when the header region leaks

exclude_files: []                 # raw frames rejected by frame QC (file names)

qc:                               # frame-QC thresholds (defaults shown)
  nsigma: 3.0
  level_floor_adu: 5.0
  saturation_adu: 60000
  flat_min_adu: 2000

science:
  include_objects: []
  exclude_object_substrings: [rfocus, junk, focus]

wcs:
  enabled: true
  solve_field: solve-field
  index_dir: null                 # or "/Users/you/astrometry-indexes"
  scale_arcsec_per_pix: 0.43
  radius_deg: 1.0
  timeout_sec: 300
  include_objects: []             # empty = all
```

Night init auto-fills paths, HDU (`xy00`), overscan keywords, and filter map
from flat `OBJECT` labels (`dflat-r`, …).

## Troubleshooting


| Problem                  | Fix                                                             |
| ------------------------ | --------------------------------------------------------------- |
| `ccd: command not found` | `conda activate astro`; from repo run `pip install -e .`        |
| Inventory 0 files        | Check `paths.raw_dir` with `ls`                                 |
| Combine “stuck”          | Large stacks take minutes — watch the spinner                   |
| WCS FAIL / low log-odds  | Check the plate scale, then `wcs_solve.log`. HDI ~29′: `4207`/`4107`. S2KB 20.5′: `4206`/`5206`, scale 0.60″/pix (1.20″ at 2×2). Hall 2×2: 0.739″/pix on `4107`. Next smaller quads on disk are `4206`/`5206`, `4205`/`5205`, `4204`/`5204`, `5203`. |
| Wrong folder             | `pwd` / `ls` — expect `RAW/` and `configs/`                     |


If you have issues, please open a github issue, I'll review as I am able.

## License

Methods follow the Astropy CCD Reduction Guide. This package wraps `astropy`
and `ccdproc` for HDI nights.
