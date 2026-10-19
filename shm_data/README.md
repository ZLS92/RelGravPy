# Spherical Harmonic Models (experimental)

This directory is the local/offline coefficient cache used by `relgravpy.grav_analysis.shm`.
Large coefficient files are intentionally not embedded in the ordinary Git history.

## Sources, attribution, and licensing

- **EIGEN-6C4** (alias `EIGENC4`): Förste et al. (2014), ICGEM/GFZ,
  DOI: https://doi.org/10.5880/ICGEM.2015.1
  ICGEM catalog: https://icgem.gfz.de/tom_longtime
- **WHU-CASM-UGM2025_2159**: Liu, Wang & Jiang (2026), GFZ Data Services,
  DOI: https://doi.org/10.5880/icgem.2026.002
  License: CC BY 4.0, see the publisher's record:
  https://dataservices.gfz-potsdam.de/icgem/showshort.php?id=4baea009-3e16-11f1-a8d3-e388fc58c1bf
- **Earth2014 SUR2014** (topography): Hirt & Rexer (2015),
  DOI: https://doi.org/10.1016/j.jag.2015.03.001
  https://www.asg.ed.tum.de/iapg/forschung/topographie/earth2014/
  Consult the original Earth2014 README and redistribution terms before
  mirroring coefficient data. The Earth2014 binary SH coefficient format
  is NOT the same as a gravity-model ICGEM .gfc file.

The local cache is populated explicitly (requires internet only for download):

```python
from relgravpy import grav_analysis as ga
ga.sh_download_model("EIGEN-6C4")
ga.sh_download_model("WHU-CASM-UGM2025_2159")
model = ga.shm("EIGEN-6C4")          # offline after first download
result = model.synthesis(
    limits=[6, 14, 42, 48],
    step=1,
    fields=("grav_dist", "grav_gradient"),
    lmax=120,
    surface="sphere",
)
```

For a topographic SH surface, explicitly download Earth2014 first:

```python
ga.sh_download_topography()  # ~933 MB, one-time download to shm_data
model = ga.shm("EIGEN-6C4")   # offline
result = model.synthesis(
    limits=[8, 12, 44, 47], step=1.0, lmax=60,
    fields=("topo", "grav_dist"),
    surface="topography",
)
```

Earth2014 `.bshc` files are read using an efficient memmap and truncated
to the selected degree, although full-resolution synthesis is currently very
slow and memory intensive.

Important: this is a **reference kernel under development**, not a validated
ICGEM-compatible geodesy engine. Its current `grav_dist` field is the radial
contribution from degrees >=2 relative to the spherical monopole, **not yet the
physical gravity disturbance relative to WGS84 normal gravity**. Its
`grav_gradient` is the radial derivative of that degree>=2 field (Eötvös).
The functions for geoid, absolute gravity and free-air anomaly have not yet been implemented. Earth2014 topography
synthesis is available, but has not yet been numerically validated against
independent reference points. Do not treat
`surface="mean_sea_level"` as a precise geoid-height surface; it currently uses
the WGS84 ellipsoid radius. Regional least-squares coefficient analysis may
be ill-conditioned and is not equivalent to an exact global SH transform.

The numerical module does not depend on pyshtools or Numba. 
