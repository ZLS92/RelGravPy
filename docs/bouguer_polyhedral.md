# Bouguer terrain effect: curvature-following polyhedra

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

**Important:** `te()` still returns a modelled topographic mass effect. It
does not by itself calculate a complete Bouguer gravity anomaly; that requires
the other chosen anomaly corrections and a consistent sign convention.

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

Existing function names `te`, `run_te`, `load_te` are preserved for now.
The former `tess=True` keyword is renamed to `curved=True`.

```python
from relgravpy import _bouguer_correction as bc

# Existing data/DTM arguments apply.
# bc.te(..., curved=True, R3=50_000)
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
