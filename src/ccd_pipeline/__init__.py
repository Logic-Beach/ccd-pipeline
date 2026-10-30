"""CCD reduction pipeline built on Astropy / ccdproc.

Procedure follows the Astropy CCD Reduction and Photometry Guide:
https://www.astropy.org/ccd-reduction-and-photometry-guide/

Typical night workflow (do not reorder)
---------------------------------------
1. Overscan-subtract and trim (``BIASSEC`` / ``DATASEC``)
   — guide 01-08, 02-02
2. Build a master bias from overscan-calibrated bias frames
   — guide 02-04
3. Build master flats (bias-subtracted, inverse-median scaled)
   — guide 05-03, 05-04
4. Sanity-check overscan + masters (+ optional science sky QC)
5. Calibrate science: overscan → trim → bias → flat
   — guide 06-00, 06-03
6. Optional additive fringe correction (i/z; IRAF/NOIRLab practice —
   not covered in the Astropy guide)
7. Optional plate-solve science WCS (offline ``solve-field``)

See README “Methodology” for the full professional checklist and references.
"""

__version__ = "0.1.0"

# Canonical Astropy CCD Reduction Guide (dev notebooks)
GUIDE_URL = "https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/"

# Notebook deep-links used in module docs / comments
GUIDE = {
    "overscan": GUIDE_URL + "01-08-Overscan.html",
    "bias_calibrate": GUIDE_URL + "02-02-Calibrating-bias-images.html",
    "bias_combine": GUIDE_URL + "02-04-Combine-bias-images-to-make-master.html",
    "flat_about": GUIDE_URL + "05-01-about-flat-corrections.html",
    "flat_calibrate": GUIDE_URL + "05-03-Calibrating-the-flats.html",
    "flat_combine": GUIDE_URL + "05-04-Combining-flats.html",
    "science": GUIDE_URL + "06-00-Reducing-science-images.html",
    "science_examples": GUIDE_URL + "06-03-science-images-calibration-examples.html",
}

# Fringe is outside the Astropy guide; these are the professional refs we follow.
FRINGE_REFS = {
    "noirlab_soi": (
        "https://noirlab.edu/science/index.php/programs/ctio/instruments/"
        "soi/evaluating-and-reducing-soi-images"
    ),
    "iraf_mkfringecor": (
        "https://iraf.readthedocs.io/en/latest/tasks/noao/imred/ccdred/mkfringecor.html"
    ),
    "iraf_rmfringe": (
        "https://iraf.readthedocs.io/en/latest/tasks/nfextern/msctools/rmfringe.html"
    ),
    "astro_wise": "https://doc.astro-wise.org/man_howto_fringe.html",
}
