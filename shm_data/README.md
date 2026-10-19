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

Important: this is a **reference kernel under development**, not a
numerically validated reproduction of ICGEM's calculation service.

Current synthesized gravity fields:
- `potential`: disturbing potential T, m2/s2, with WGS84 even-zonal reference;
- `geoid`: Bruns approximation T/gamma, m;
- `grav_dist`: radial gravity disturbance (spherical approximation), mGal;
- `free_air`: spherical gravity anomaly approximation, mGal;
- `abs_grav`: normal gravity at height plus radial disturbance, mGal;
- `grav_gradient`: radial derivative of the disturbance, Eotvos;
- `topo`: Earth2014 SUR relief synthesized from .bshc, m.

Important limitations:
- Exact ellipsoidal components and tide/zero-degree conventions are not yet
  reproduced. Do not use these results for precision geodesy or survey
  calibration without comparison to independently computed ICGEM checkpoints.
- `mean_sea_level` currently approximates the WGS84 ellipsoid, not an
  iteratively computed geoid surface.
- The Earth2014 SUR elevations are nominally relative to mean sea level;
  `surface="topography"` uses them as approximate ellipsoidal heights
  until a geoid-to-ellipsoid conversion is implemented.
- No performance claim is made for degree 10,800. Processing may be extremely
  slow or memory-intensive at that degree.
- Regional least squares analyses can be ill-conditioned and are not the
  equivalent of a global spectral inversion. For full global latitude/longitude
  grids use the FFT/latitude-quadrature option, which is approximate on
  equiangular latitude grids.
- These large model datasets are intentionally excluded from standard Git.
  An offline cache is available **after** the explicit downloads. The model
  binaries have not been committed to this repository.
