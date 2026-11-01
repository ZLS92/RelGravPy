# Bouguer correction: curvature-following polyhedra

This module is a research implementation on
`feat/bouguer-polyhedral-curvature`. It replaces
`relgravpy/topo_effect.py` with `relgravpy/_bouguer_correction.py`
and removes the runtime dependency on Harmonica.

## Coordinate system and physical conventions

Each station has a local orthographic projection, with the tangent plane at
the local reference sphere. Local axes are East / North / Up, and the station
is observed at `[0, 0, z]`. The reference sphere of radius `R` is tangent
at `z=0`. Gravity returned by this module is **down-positive in mGal**.

When used, the numerical routine `grav_modelling.gravicalc2_full` returns the Cartesian
`g_z` component **up-positive** for consistently outward-oriented faces.
Therefore `_polyhedral_gz` reverses its sign and also scales by
`grav_modelling.G / 6.6732e-11` to match the gravitational constant used in
`grav_modelling.nagy`.

**Important:** `bgc()` is the Bouguer-correction modelling entry point.
Its outputs (`bgc_near`, `bgc_far`, `bgc_tot`) are the modelled gravitational
mass contributions in mGal (**positive downward**). When removing this
modelled gravitational attraction from observed gravity, apply the opposite
sign. This function does not compute a complete Bouguer gravity anomaly:
the complete anomaly also depends on the other corrections and on the
chosen gravity sign convention.

## Geometry and kernels

- **Near field** (inside `R1`): original Nagy rectangular prisms.
  The existing `z_shift` correction approximates spherical curvature on a
  per-cell basis.
- **Far field** (`R1` to `R2`): for each DEM rectangle, build four
  vertices on the reference sphere at radial height `bottom`, and four
  at radial height `top`. Lower and upper quadrilateral patches are split
  into two planar triangles. Four radial side faces complete the closed
  polyhedron. For well-separated cells the gravitational acceleration is
  computed from the divergence-theorem boundary integral on outward-oriented
  triangular facets, evaluated with a vectorised 7-point (degree-5) quadrature.
  For very nearby cells we still reuse the deterministic `gravicalc2_full`
  kernel; the latter was found to suffer severe cancellation for small,
  distant cells.
- **Very far field** (beyond `R3`, when enabled): the existing
  box-to-point-mass reduction remains, but point mass gravity is now calculated
  directly in local Cartesian coordinates using Newton's law.

The underside uses **triangular chords with vertices on the sphere**, not
strictly tangent planes. This is intentional: adjacent grid cells share lower
vertices, and the approximation converges as the DEM mesh is refined.

## API

The main function is now **`bgc()`**, with corresponding wrappers
`run_bgc()` and `load_bgc()`. The former `te()`, `run_te()` and `load_te()`
names are intentionally not exposed on this experimental branch. The
former `tess=True` keyword has been replaced by `curved=True`.
Output file columns and loader dictionary keys are now `bgc_near`,
`bgc_far` and `bgc_tot` (formerly `te_near`, `te_far`, `te_tot`).
The default output name is `Bouguer_correction`; legacy files and
downstream code require migration. This rename does **not** change any
gravity sign or numerical algorithm.

```python
from relgravpy import _bouguer_correction as bc

# Existing data/DTM arguments apply.
# bc.bgc(..., curved=True, R3=50_000)
```

`curved=False` retains the old prism-only far-field alternative without
Harmonica.

## Verified so far

The automated tests in `tests/test_bouguer_polyhedra.py` verify:

- each curved cell's lower vertices lie on the reference sphere;
- neighbouring cells reuse the same lower-edge coordinates;
- the polyhedral field approaches the analytical Nagy prism result when
  the reference sphere radius becomes very large;
- negative densities reverse the sign, the point-mass kernel uses the
  intended vertical sign, and invalid cells are rejected;
- the empirical ring kernel returns a finite value.

CI compiles both Python modules and runs these tests with NumPy and pytest,
using AST extraction of numerical functions to avoid optional GIS imports.

## 31 October 2026: numerical seam, convergence and timing benchmark

Source: `tests/benchmark_bouguer_polyhedra.py` and workflow
`.github/workflows/bouguer-benchmark.yml`.

The first numerical verification found that the original `gravicalc2_full`
solver could return approximately **0.00003 µGal** for an 8-km-distant,
250 x 250 x 300 m element with a **0.37686 µGal** Nagy-prism reference
field. This motivated the stable far-cell surface-integral implementation,
while retaining the original analytic solver at close distance.

After that change, at **R1=8 km, gs1=250 m, gs2=1,000 m**, the diagnostic
cell returned **0.37675 µGal** (spherical polyhedron) compared with
**0.37686 µGal** (legacy near prism), a difference of **0.000113 µGal**.
The sampled legacy prism curvature shift differs from the spherical surface
at the seam by as much as **0.155727 m** vertically. With 300 m of relief
the radial expansion of a polyhedron's upper edge at 8 km is approximately
**0.377 m** horizontally. The seam is therefore NOT an exact watertight join;
a complete perimeter-wide gravity-error bound has not yet been obtained.

### Synthetic 20 x 20 km comparisons

Observation height 700 m, density 2,670 kg/m³, spherical radius 6,371 km.
Tesseroid bounds are built from approximately corresponding lat/lon limits
for the orthographic rectilinear polyhedral grid. Thus these residuals
include small *geometric grid-parametrisation differences*, not just
kernel error.

| Scenario | Cell width | Cells | Poly-Harmonica difference (µGal) | New poly time (s) | Harmonica time (s) |
|---|---:|---:|---:|---:|---:|
| Flat, elevation 250 m | 2,500 m | 64 | +5.683 | 0.0094 | 0.0004 |
| Flat, elevation 250 m | 1,250 m | 256 | +3.283 | 0.0097 | 0.0004 |
| Flat, elevation 250 m | 625 m | 1,024 | +1.546 | 0.0072 | 0.0006 |
| Synthetic rugged topography | 2,500 m | 64 | +7.336 | 0.0089 | 0.0003 |
| Synthetic rugged topography | 1,250 m | 256 | +4.238 | 0.0092 | 0.0004 |
| Synthetic rugged topography | 625 m | 1,024 | +2.410 | 0.0050 | 0.0006 |

The earlier pure `gravicalc2_full` implementation took **1.6765 s**
(flat, 1,024 cells) and **1.6795 s** (rugged, 1,024 cells), so the new
hybrid approach is roughly **230-330 times faster** in these particular
small, warmed-up synthetic tests. It remains around **8-12 times slower
than Harmonica** on the same 1,024-cell datasets. More reliable timing
will require repeated runs, isolated CPU benchmarks, and real DTM inputs.

**Interpretation:** the residual magnitude decreases as the mesh is
refined, but remains approximately 1.5-2.4 µGal at 625 m. This is not
a demonstration of sub-µGal equivalence for real microgravity surveys.

## 1 November 2026: fixed one-tesseroid, 20 m clearance benchmark

The previous many-tesseroid test can confound discretisation errors in the
reference and target meshes. A new independent test uses **one tesseroid**
for a uniform geocentric spherical wedge, while refining ONLY the
polyhedral mesh.

- Reference radius: 6,371,000 m
- Top height: 300 m above reference sphere; bottom: 0 m
- Density: 2,670 kg/m³
- Extent: approximately 20 km × 20 km, centered at lon/lat 0
- Calculation point: lon/lat 0, radius `R+320 m` (20 m above the top)
- `hm.tesseroid_gravity` uses a single input tesseroid with adaptive
  quadrature. This is a tesseroidal angular wedge, **not a circular cap**.
- Additional **independent volume quadrature** integrates the exact spherical
  volume in spherical coordinates (Gauss-Legendre in longitude, latitude and
  radius; denser sampling around the station and haversine distance
  stabilization). The reference is checked at two quadrature resolutions.

### Reference and convergence

| Method | Gravity (mGal) | Difference from direct integral (µGal) |
|---|---:|---:|
| Independent spherical volume quadrature | 33.104457071 | 0 |
| Harmonica, one tesseroid, default adaptive | 33.097690779 | -6.7663 |
| Harmonica, one tesseroid, radial-adaptive | 33.096834501 | -7.6226 |
| Polyhedral mesh 4 × 4 | 33.133188266 | +28.7312 |
| Polyhedral mesh 8 × 8 | 33.114322081 | +9.8650 |
| Polyhedral mesh 16 × 16 | 33.107357556 | +2.9005 |
| Polyhedral mesh 32 × 32 | 33.105117792 | +0.6607 |
| Polyhedral mesh 64 × 64 | 33.104552296 | +0.0952 |
| Polyhedral mesh 96 × 96 | 33.104499297 | +0.0422 |
| Polyhedral mesh 128 × 128 | 33.104476445 | +0.0194 |

The calculations show that the *single-Harmonica-tesseroid output* is
not a sufficiently accurate numerical reference at only 20 m above this
large source. The polygonal solution converges to the independent
high-resolution volume integral within 0.1 µGal by 64×64 cells and within
0.02 µGal by 128×128 cells. The comparison does **not** establish
universal accuracy for mountainous DEMs, other geometries or stations.

Geometry note: the tesseroid has constant longitude/latitude bounding
surfaces. The current polyhedra are generated from a planar orthographic
square, resulting in a maximum east-west footprint offset of about
0.0123 m at the north/south edges. This is small here but is not zero.

Reproduce with `python -u tests/benchmark_single_tesseroid.py`.
Harmonica remains an optional **benchmark-only** dependency; the
processing module does not import it.

## Verification priorities before production

**Primary benchmark**: `tests/benchmark_spherical_reference.py` computes
the gravitational effect by direct independent spherical volume quadrature.
It does not import Harmonica. Its CI step checks quadrature stability and
convergence to <0.1 microgal at 128x128 cells, for one uniform angular wedge
with 20-m observation clearance. The legacy single-tesseroid benchmark remains
available only as an auxiliary diagnostic.

Priority P0:
- Test the full high-level `bgc()` path on at least one real DTM (GDAL warp,
  georeferencing, elevation datums, nodata and output) with known results.
- Replace the approximate near-prism / far-polyhedron interface at `R1` or
  bound the full perimeter-wide gravity mismatch. Near prisms have a constant
  curvature shift whereas curved cells expand radially with height.
- Verify `R` handling: the caller-specified radius is overwritten inside
  `st_loop` by `utl.local_sph_raduis(yg)`, while prior raster projection
  uses the caller's `R`. Choose one consistent reference model.
- Verify the complete Bouguer effect using variable-thickness synthetic
  topography with a converged independent spherical reference, including
  negative height contrasts, density signs, surface and seabed stations.

Priority P1:
- Replace/tune empirical selection of `R1`, `R2`, `gs2` and the `R3`
  point-mass approximation with verified error-budget criteria.
- Benchmark the original `gravicalc2_full` / surface quadrature switching
  criterion and surface-face quadrature on near-singular geometries,
  demanding a smooth and stable gravity response.
- Validate projected vs geographic DTM pixel sizes; the default `gs1`
  currently uses `rt.raster_res(dtm1)` as a metric step without a CRS-unit
  check. Include tests for input height reference and spherical vs
  orthometric elevations.
- Check multiple stations, parallel execution, output correctness and
  performance. Current `st_loop` workers append to the same output path,
  even though the caller later rewrites the full result.

Priority P2:
- Define a public documented numerical API, with explicit output sign,
  gravitational *mass effect* versus *Bouguer correction* conventions,
  density units, station classification, and uncertainty diagnostics.
- Remove `eval()` from the CLI and replace it with typed parsing; clean
  up legacy parameter branches and test deterministic, reliable I/O.
- Provide regression test fixtures and notebooks for representative
  mountain, coastal and sea-floor DTM cases.

**Target proposal**, not a general validation claim: numerical kernel error
<0.1 microgal in controlled reference cases and total algorithmic error
<1 microgal over an agreed range of operational scenarios. These tolerances
must be evaluated separately from DTM elevation and rock-density uncertainty.

## Still to validate before scientific use



1. **Full DEM integration:** compare against the previous Harmonica
   tesseroid implementation at known stations and for land, sea surface,
   and sea-bottom station types.
2. **R1 seam:** the near-field prisms use a constant cell-wise curvature
   shift, while far-field polyhedra follow spherical geometry. The junction
   is approximate, not mathematically watertight. Quantify its maximum
   mismatch or introduce a dedicated transition band if necessary.
3. **Geometry convergence:** synthetic flat/rugged refinement shows residuals
   falling with cell width; repeat with exactly identical geographic
   footprints, different observation heights, latitude and real DEMs.
4. **R3 point-mass approximation:** quantify the result versus increasing
   `R3` and direct polyhedral integration.
5. **Performance:** the new far-field surface integral is vectorised and
   significantly faster than the previous cell-by-cell `gravicalc2_full`
   implementation, but performance on many real survey stations and DEMs
   is still unmeasured. Profile end-to-end before considering compiled code.
