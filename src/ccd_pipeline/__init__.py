"""CCD reduction pipeline built on Astropy / ccdproc.

Procedure follows the Astropy CCD Reduction and Photometry Guide:
https://www.astropy.org/ccd-reduction-and-photometry-guide/

Typical night workflow
----------------------
1. Overscan-subtract and trim (``BIASSEC`` / ``DATASEC``)
2. Build a master bias from calibrated bias frames
3. Build master flats (bias-subtracted, inverse-median scaled)
4. Sanity-check overscan + masters
5. Calibrate science: overscan → trim → bias → flat
6. Plate-solve science WCS (offline ``solve-field``)
"""

__version__ = "0.1.0"

# Canonical guide URL (dev notebooks)
GUIDE_URL = "https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/"
