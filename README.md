# CCD reduction pipeline

A terminal app that calibrates one night of CCD images (bias, flats, science)
and can add sky coordinates (WCS). Built for WIYN 0.9m **HDI** data using
Astropy / [`ccdproc`](https://ccdproc.readthedocs.io/), following the
[Astropy CCD Reduction and Photometry Guide](https://www.astropy.org/ccd-reduction-and-photometry-guide/).

You do **not** need to be a Linux expert. The usual workflow is: open a
terminal, go to the right folder, type a few commands, and answer Y/n
prompts.

---

## What it does (in order)

| Step | Command | Result |
|------|---------|--------|
| 1. Inventory | (automatic in wizard) | Counts bias / flat / science frames |
| 2. Master bias | `ccd masters …` | `masters/master_bias.fits` |
| 3. Master flats | `ccd masters …` | `masters/master_flat_<filter>.fits` |
| 4. Sanity check | `ccd sanity …` | Stats + PNG plots under `masters/diagnostics/` |
| 5. Science | `ccd reduce …` | Calibrated frames in `science/` |
| 6. WCS (optional) | `ccd wcs …` | Adds sky coordinates; log in `wcs_solve.log` |

Science files are named like `2017JUN29.SA_103-Z.r.01.fits`
(`night.object.filter`, with a number if there are several of the same).

---

## One-time setup (ask for help if needed)

1. Install [Miniconda](https://docs.conda.io/) (or Anaconda) if you do not have it.
2. Create/activate the astronomy environment (example name: `astro`) with
   `astropy`, `ccdproc`, `numpy`, `pyyaml`, `matplotlib`.
3. Install this package once:

```bash
conda activate astro
cd /home/<user>/path/to/ccd-pipeline
pip install -e .
```

After that, the `ccd` command works from **any** folder (as long as the
`astro` environment is active).

### Optional: plate-solving (WCS)

Only needed for `ccd wcs`:

```bash
sudo apt install astrometry.net astrometry-data-tycho2-07 astrometry-data-tycho2-08
ls /usr/share/astrometry/index-*.fits
```

Index series numbers are **quad angular scale**. Lower
number ⇒ smaller quads / denser stars; higher ⇒ larger quads / coarser sky.
Within a 2MASS series, `-00`…`-NN` are the **sky tiles** (HEALPix chunks).

| Package | Quad scale | Role for HDI (~29′ FOV, 0.43″/pix) |
|---------|------------|-------------------------------------|
| `astrometry-data-tycho2-07` … `09` | 22′–60′ | Light, good first try |
| `astrometry-data-2mass-07` | 22′–30′ | Matches FOV; denser than Tycho |
| `astrometry-data-2mass-06` / `05` | 16′–22′ / 11′–16′ | **Usually fixes stubborn HDI fields** |
| `astrometry-data-2mass-04` / `03` | 8′–11′ / 5.6′–8′ | Finer still (binning, sparse matches) |
| `astrometry-data-2mass-08-19` | 30′–2000′ | Coarser / wide-field recovery |
| `astrometry-data-tycho2-10-19` | 60′–2000′ | Coarse Tycho fallback (small download) |

Recommended add-ons once basics are in:

```bash
# Dense near-FOV:
sudo apt install astrometry-data-tycho2-09 \
  astrometry-data-2mass-07 astrometry-data-2mass-06 astrometry-data-2mass-05

# Optional: one step finer + coarser safety net
sudo apt install astrometry-data-2mass-04 astrometry-data-2mass-03 \
  astrometry-data-2mass-08-19 astrometry-data-tycho2-10-19
```

`2mass-00`…`02` are rarely worth it for full-frame HDI (huge, tiny quads).
A timeout with **best log-odds ~1** means “no match,” not “almost”, install a
denser series before raising `wcs.timeout_sec` further.

The pipeline works around a common NumPy quirk automatically. Default solve
timeout is **300 s** per frame (`wcs.timeout_sec` in the night YAML).

---

## Where your files live

A typical observing campaign looks like this:

```text
2017_JUN_JAS/                    ← campaign folder (example)
  RAW/
    2017JUN29/                   ← one night of raw FITS
  reduced/
    2017JUN29/                   ← pipeline output for that night
      masters/
      science/
      wcs_solve.log
  configs/
    hdi_2017jun29.yaml           ← night settings (paths, filters, …)
```

Older HDI nights may use `RAW/<night>` instead of `<night>/raw`. Both are fine.

---

## Where to be in the terminal

### Rule of thumb

1. Open a terminal.
2. Activate the environment: `conda activate astro`
3. Go to your **campaign** folder (the one that contains `RAW/`, `reduced/`,
   and `configs/`):

```bash
cd /home/<user>/path/to/DATA
```

(Use your own path, including an external drive `/media/<user>/drivename/` if applicable.)

From there, config files are short paths like `configs/hdi_2017jun29.yaml`.

You *can* run from any directory if you give the **full** path to the config,
for example:

```bash
ccd sanity --config /home/deskpop/astronomy/2017_JUN_JAS/configs/hdi_2017jun29.yaml
```

### Check you are in the right place

```bash
pwd          # prints current folder
ls           # should show RAW, configs, maybe reduced
ls RAW       # should list night folders such as 2017JUN29
```

---

## How to reduce a night (simple path)

### A. First time for a night (create config + run steps)

Still in the campaign folder (or any folder), with `astro` active:

```bash
conda activate astro
cd /home/<user>/path/to/data

ccd
```

Then:

1. Choose **New night**.
2. Pick the night’s raw folder (e.g. `RAW/2017JUN29`). A file picker usually
   appears; if it does not, type the path and press Enter.
3. Accept (or change) where reduced files will go.
4. Answer the Y/n questions for bias → flats → sanity → science → WCS.

Combining many large images can take several minutes. The terminal will show
a spinner and a timer so you know it is still working.

### B. Night already set up (config exists)

```bash
conda activate astro
cd /home/de<user>/path/to/data

ccd inventory --config configs/hdi_2017jun29.yaml
ccd masters   --config configs/hdi_2017jun29.yaml
ccd sanity    --config configs/hdi_2017jun29.yaml
ccd reduce    --config configs/hdi_2017jun29.yaml
ccd wcs       --config configs/hdi_2017jun29.yaml
```

Replace `hdi_2017jun29.yaml` with your night’s config name
(`ls configs` to see them).

Useful options while testing:

```bash
ccd reduce --config configs/hdi_2017jun29.yaml --limit 2
ccd wcs    --config configs/hdi_2017jun29.yaml --limit 2
ccd wcs    --config configs/hdi_2017jun29.yaml --object APSExS-F-1
ccd wcs    --config configs/hdi_2017jun29.yaml --overwrite
```

- `--limit 2` = only process 2 frames (quick test)
- `--object NAME` = only that science OBJECT (repeatable)
- `--overwrite` = re-solve WCS even if a solution is already present

Or reopen the interactive menu for an existing config:

```bash
ccd
# choose: Use an existing config → pick the YAML
```

---

## Where to look for results

After a successful run for night `2017JUN29`:

| What | Where |
|------|--------|
| Master bias / flats | `reduced/2017JUN29/masters/` |
| Diagnostic plots | `reduced/2017JUN29/masters/diagnostics/` |
| Calibrated science | `reduced/2017JUN29/science/` |
| WCS summary log | `reduced/2017JUN29/wcs_solve.log` |
| Per-frame solve-field text | `reduced/2017JUN29/wcs_logs/*.solve.log` |

Open the PNG diagnostic plots with your usual image viewer. For a good master
bias, the histogram should peak near **0**. For a master flat, near **1**.

---

## Common problems

| Symptom | Likely fix |
|---------|------------|
| `ccd: command not found` | `conda activate astro`, then reinstall with `pip install -e .` from `ccd-pipeline` |
| Inventory shows 0 files | Wrong `raw_dir` in the YAML; check the path with `ls` |
| Combine seems “stuck” | Wait — large HDI stacks take minutes; look for the spinner / seconds counter |
| WCS timeouts / some FAIL | If best log-odds ≪ 30: denser indexes (`2mass-05`→`03`); optional coarser `2mass-08-19`; else raise `timeout_sec` |
| WCS all FAIL | Install index packages (see setup); check `wcs_solve.log` |
| Permission / “No such file” | Typo in `cd` path; use `pwd` and `ls` |

---

## Technical notes (optional reading)

### Calibration steps vs the Astropy guide

| Step | Guide notebooks | What we do |
|------|-----------------|------------|
| Overscan + trim | 01-08, 02-02 | `BIASSEC` / `DATASEC` |
| Master bias | 02-04 | σ-clip average |
| Master flats | 05-03, 05-04 | Bias-subtract, inv-median scale, combine |
| Science | 06-00 | Overscan → trim → bias → flat |
| WCS | (post-cal) | Offline `solve-field` + pointing headers |

### Auto-config from headers

When you create a new night, the tool reads telescope/instrument, frame types,
image HDU (HDI `xy00`), overscan keywords, and a filter map from flat
`FILTER1`/`FILTER2` + `dflat-*` object names.

### Paths in the YAML

Absolute paths are preferred and work on external drives:

```yaml
paths:
  raw_dir: "/home/deskpop/astronomy/2017_JUN_JAS/RAW/2017JUN29"
  output_dir: "/home/deskpop/astronomy/2017_JUN_JAS/reduced/2017JUN29"
  masters_dir: "/home/deskpop/astronomy/2017_JUN_JAS/reduced/2017JUN29/masters"
```

Relative paths in a campaign `configs/` folder resolve from the **parent of
`configs/`** (the campaign folder).

### WCS config knobs (HDI defaults)

```yaml
wcs:
  enabled: true
  solver: "astrometry-net"
  solve_field: "solve-field"
  index_dir: null
  scale_arcsec_per_pix: 0.43
  scale_tol_frac: 0.05
  radius_deg: 1.0
  downsample: 2
  overwrite: false
  timeout_sec: 300
  verbose: true
  show_solver_output: false   # true = stream every solve-field line live
  include_objects: []         # e.g. [APSExS-F-1, NGC_6791]; empty = all
```

By default the terminal shows a spinner + OK/FAIL per frame. Full `solve-field`
output is always written to `wcs_logs/<frame>.solve.log` (and linked from
`wcs_solve.log`). Set `show_solver_output: true` if you want the live dump.

To solve only some targets:

```bash
ccd wcs --config configs/hdi_2017jun29.yaml --object APSExS-F-1 --object NGC_6791
```

---

## License / attribution

Reduction methods follow the Astropy CCD Reduction and Photometry Guide
(Astropy Project). This package is a thin, instrument-configurable wrapper
around `astropy` and `ccdproc`.
