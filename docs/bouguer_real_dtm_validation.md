# Bouguer correction: full GDAL run on **real** open elevation datasets

Test date: 2026-11-01. Branch: `feat/bouguer-polyhedral-curvature`.

## Input and provenance

The reproducible data-acquisition script
`tests/probe_real_dem_sources.py` downloads geographic WGS84 GeoTIFF
subsets, not synthetic elevation fields:

1. **Dolomites (Cortina / Cadore region)**: the open Copernicus GLO-90
   public Cloud-Optimized GeoTIFF, tile
   `Copernicus_DSM_COG_30_N46_00_E012_00_DEM` (1° tile subset),
   downloaded from
   `https://copernicus-dem-90m.s3.amazonaws.com/`.
   The sample used for the near field has 780 × 720 cells and
   elevation range 302.5–3130.7 m.
2. **Wider Alpine field and Ligurian coastal field**:
   NOAA NCEI public `DEM_mosaics/DEM_global_mosaic/ImageServer`
   `exportImage` endpoint:
   `https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer`.
   The Ligurian near field has 700 × 700 cells and ranges from
   **−1541.5 m bathymetry to +1760.5 m terrain**. The far-field
   mosaic (800 × 800) spans −2804.0 to +4388.2 m.

Both source families are genuine georeferenced geospatial data,
but are **different elevation products**, with their own spatial
resolution, processing and height-reference lineage. A fully
geodetically consistent, seamless product should be used before
scientific deployment of the results.

## Actual complete entry point

`tests/benchmark_bgc_real_dtm.py` calls the public
`relgravpy._bouguer_correction.bgc()` function, not merely
`prism_grav` or `polydron_grav`. The run exercises:

- Geographic coordinates and GDAL dataset opening;
- Per-station height extraction from actual source raster;
- Local orthographic reprojection, near/far GDAL warps and resampling;
- Preservation of source data (verified **byte-for-byte SHA-256**);
- Construction of near Nagy prisms, variable-facet R1 transition and
  curvature-following outer polyhedra, with R3 point masses;
- GDAL terrain-sign land/sea assignment without coastline-driven
  subdivision or reclassification;
- True mixed land/sea station types, including a **sea-bottom** station;
- Results written to and read back from the public `bgc` text output;
- Station-wise `bgc_total ≈ bgc_near + bgc_far`;
- Integer-only `gs1, R1, gs2, R2` (nominal default R2=167,000 m).

For the explicit-mesh test, `gs1=300 m, R1=5250 m,
gs2=1500 m, R2=167250 m` after alignment, and
**production-default `R3=50000 m`**.

### Verified numerical outputs with explicit mesh

| Environment | Station lon / lat | Type | Source DTM elevation (m) | Near (mGal) | Far (mGal) | Total modeled gravity (mGal) |
|---|---|---|---:|---:|---:|---:|
| Dolomites | 12.35 / 46.57 | terrestrial | 1043.72 | 91.428449 | 5.279400 | **96.707849** |
| Dolomites | 12.45 / 46.65 | terrestrial | 1555.06 | 141.574232 | 21.585902 | **163.160134** |
| Ligurian coast | 9.63 / 44.13 | **sea-bottom** (`st_type=2`) | −35.26 | −2.414301 | −1.581180 | **−3.995481** |
| Ligurian coast | 9.63 / 44.20 | terrestrial (`st_type=0`) | +360.47 | 33.222793 | −0.482105 | **32.740688** |

These are forward-model **gravity mass effects**, not observed
Bouguer anomalies; sign is positive downwards. In particular, do not
interpret the sign of a sea-floor mass contribution as a universal
statement about the Bouguer correction's applied sign.

The two-station complete calls on the GitHub-hosted Linux runner
needed approximately **0.45 s** (Dolomites) and **0.36 s**
(Liguria), after source DEM download and GDAL installation.
Times depend on machine, source data, chosen mesh and caching.

### Automatic tuning with real DTM

The same script also exercises `bgc(gs1=None, R1=None,
gs2=None, target_error=0.05)` **once for both mountain stations**.
With the production default 50-km point-mass threshold, the optimizer
chose integer values
`gs1=94 m, R1=11985 m, gs2=510 m, R2=167025 m`.
The maximum morphology-based error proxy was **0.058817 mGal**,
above the user target (0.05 mGal): the intended `RuntimeWarning`
was emitted and the lowest-proxy-error candidate was used.
The run completed and produced two gravity results.

The proxy is **not a certified maximum difference** against
a converged physical reference. There is currently no independent
high-resolution full-extent reference for these particular real DEMs,
so this is an end-to-end processing verification, not confirmation
of the 0.05 mGal field accuracy goal.

## Defects found during real GDAL execution

Before this test, the high-level workflow tried to overwrite NoData
inside a GDAL dataset opened **read-only** from the original file:
`RuntimeError: Write operation not permitted`. This is now fixed:
`bgc()` works on a writable normalization copy. The real DTM
regression computes SHA-256 over both raw inputs before and after
the call to prevent regression to source mutation.

The station worker intentionally releases near/far prism arrays
before return to avoid retaining large survey-wide meshes. The
test respects this public return contract and checks numeric gravity
results and persisted GDAL products instead.

Older numerical benchmark results do **not** replace validation
of raster NoData, coast masking, elevation reference or consistent
near/far source blending on real data.

## How to repeat

On Linux with the project and GDAL's Python bindings installed:

```bash
python tests/probe_real_dem_sources.py
PYTHONPATH=. python tests/benchmark_bgc_real_dtm.py
```

Both commands are run on
`.github/workflows/bouguer-real-dtm.yml`.
The source DEM rasters are downloaded only in the CI job and are not
committed to GitHub.

**Pending before scientific use:** independent convergence on the
actual *same* real DEM, quantitative sensitivity to point-mass
threshold R3, control of NoData holes and near/far vertical datum
differences, realistic coastline classification masks where available,
and larger multi-station surveys.
