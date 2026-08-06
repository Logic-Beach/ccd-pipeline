# CCD reduction pipeline

Astropy / `ccdproc` reduction pipeline tuned for **WIYN 0.9m HDI** multi-extension
FITS from the example night in `../Raw Data/2017_JUN_JAS`.

## What this night looks like

| Kind | Count | Header `OBSTYPE` | Notes |
|------|------:|------------------|-------|
| Bias | 36 | `BIAS` | filename `*b00.fits` |
| Flat | 120 | `FLAT` | dome flats; filter encoded in `OBJECT` (`dflat-r`, …) and `FILTER1`/`FILTER2` |
| Science | 239 | `OBJECT` | filename `*o00.fits` |
| Dark | 0 | — | **no darks this night** → bias + flat only |

Each file is a MEF: empty primary HDU (metadata) + image extension `xy00`
(4150×4150). Overscan is `BIASSEC=[4097:4150,*]`; useful data roughly
`DATASEC=[1:4096,1:4112]`.

## Layout

```
ccd-pipeline/
  configs/          night / instrument YAML
  src/ccd_pipeline/ library code
  scripts/          thin wrappers
  tests/
```

Raw / reduced FITS stay **outside** git (`../Raw Data`, `../Reduced Data`).

## Setup

Use the existing `astro` conda env (or any env with astropy + ccdproc):

```bash
conda activate astro
cd /Users/null/astronomy/ccd-pipeline
pip install -e .
```

## Usage

```bash
# 1. Summarize a night
ccd-inventory --config configs/wiyn_hdi_2017jun29.yaml

# 2. Build master bias + master flats
ccd-masters --config configs/wiyn_hdi_2017jun29.yaml

# 3. Calibrate science frames
ccd-reduce --config configs/wiyn_hdi_2017jun29.yaml
```

## Reduction steps (ccdproc)

1. Load image extension, merge primary header  
2. Subtract overscan (`BIASSEC`)  
3. Trim to `DATASEC`  
4. Subtract master bias  
5. Flat-field with matching filter master flat  
6. Write calibrated FITS under the configured output directory  

Plate-solving / alignment are intentionally **out of scope** for v0.1
(see unfinished guide §7; use astrometry.net / `reproject` later).

## Versioning

Git tracks code + configs only. Create commits as the pipeline evolves.
