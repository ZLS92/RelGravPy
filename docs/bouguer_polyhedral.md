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

The gravimetric solvers are `grav_modelling.prism_grav` (rectangular
prisms, analytic Nagy formulation) and `grav_modelling.polydron_grav`
(arbitrary polyhedra). Both produce downward-positive `gz` in mGal
and use the same gravitational constant.

**Important:** `bgc()` is the Bouguer-correction modelling entry point.
Its outputs (`bgc_near`, `bgc_far`, `bgc_tot`) are the modelled gravitational
mass contributions in mGal (**positive downward**). When removing this
modelled gravitational attraction from observed gravity, apply the opposite
sign. This function does not compute a complete Bouguer gravity anomaly:
the complete anomaly also depends on the other corrections and on the
chosen gravity sign convention.

## Geometry and kernels

- **Near field** (inside `R1`): **horizontal Nagy prisms** by default.
  The centre of each prism's bottom face lies on the spherical reference
  after an exact vertical `z_shift`. The prisms are not rotated.
- **Far field** (`R1` to `R2`): for each DEM rectangle, build four
  vertices on the reference sphere at radial height `bottom`, and four
  at radial height `top`. Lower and upper quadrilateral patches are split
  into two planar triangles. Four radial side faces complete the closed
  polyhedron. For well-separated cells the gravitational acceleration is
  computed from the divergence-theorem boundary integral on outward-oriented
  triangular facets, evaluated with a vectorised 7-point (degree-5) quadrature.
  For very nearby cells we use the deterministic analytic backend of
  `polydron_grav`; this kernel can suffer cancellation for small,
  distant cells.
- **Very far field** (beyond `R3`, when enabled): the existing
  box-to-point-mass reduction remains, but point mass gravity is now calculated
  directly in local Cartesian coordinates using Newton's law.

The underside uses **triangular chords with vertices on the sphere**, not
strictly tangent planes. All sides are triangulated with a **canonical
diagonal direction** so opposite cells generate the identical shared facet
when endpoints and top/bottom elevations coincide.

The outer faces of the Nagy prisms anchor the **arbitrary-facet inner
mosaic** of the new first-row bridge. The coarse far cells are **not
subdivided**: their outer four vertices and two outer triangles are
identical to those of the ordinary far polyhedron in the next ring.
Special corner bridge cells ensure identical shared triangles at all
four square R1 corners. The optional fully spherical near-field mode
retains its separate two-sided refinement.

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
Harmonica. For `curved=True`, the defaults are `near_geometry="auto"`
(**centre-shifted Nagy prisms**) and `refine_interface=True`.
The interface wedges then use the near-prism outer-face vertices.
`near_geometry="spherical"` is available only for controlled comparisons.
The new setting is forwarded through `run_bgc()`.

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

The first numerical verification found that the original analytic
polyhedral solver could return approximately **0.00003 µGal** for an 8-km-distant,
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

The earlier pure analytic polyhedral implementation took **1.6765 s**
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

## 1 November 2026: Nagy-prism-anchored transition at R1

The previous fully spherical *near-field* default has been superseded by
the specified hybrid geometry. The production-intended default now uses:

1. **Near R1: horizontal Nagy prisms**, never rotated. Their base is
   translated to the exact sphere elevation at each **prism centre**:
   `zshift = R - sqrt(R**2 - xc**2 - yc**2)`. Previously the implementation
   used the *farthest* cell corner, which is not the requested geometry.
   This is implemented in both `mesh_ls` and `mesh_sb`.
2. **Transition at R1:** one arbitrary-facet polyhedron per original
   coarse far cell, with an exact triangulated mosaic of Nagy prism
   exterior rectangles on its inner side. The mesh accommodates
   independent height steps without rotating or altering Nagy prisms.
3. **Beyond the first far row:** ordinary spherical eight-vertex
   polyhedra, without any split or extra mesh nodes. The outer face of
   each transition polyhedron retains the original coarse two-triangle
   face. Four separately triangulated corner cells close the junction.
4. `near_geometry="spherical"` still provides the old fully spherical
   experimental comparison mode, but no longer is the default.
   `near_geometry="auto"` or `"prism"` use the Nagy+bridge hybrid.

The transition is joined to the *actual* shifted-prism outer face, rather
than joining two independently computed spherical faces. The same
7-point stable surface integral is applied to the transitional polyhedron,
and ordinary far polyhedra retain the standard fast kernel.
When there is no unique matching prism/density on the R1 boundary, the
code **raises a ValueError** instead of silently creating a discontinuity;
marine/coastal interfaces need dedicated further handling.

### Independent gravimetric verification

Using a 20 x 20 km domain, near square R1=5 km, near step 125 m, far
step 312.5 m, density 2670 kg/m³, reference sphere radius 6371 km:

| Case | Direct spherical sharp-boundary reference (mGal) | Actual Nagy+bridge output (mGal) | Independent bridge volume mass effect (µGal) | Residual from geometry-adjusted reference (µGal) |
|---|---:|---:|---:|---:|
| Uniform 300 m; station at 320 m | 33.104457071 | 33.104429241 | +0.008360 | -0.036190 |
| 350 m inside R1 / 250 m outside; station at 370 m | 38.457963610 | 38.463290145 | +5.373918 | -0.047383 |

The **second case is not the same geometry as a discontinuous spherical
height-step reference**: the joined wedges add sloping material. Comparing
the output directly to the sharp-boundary reference incorrectly creates
an apparent +5.28 µGal error. The table explicitly corrects the
reference by the mass of those wedges, evaluated independently with
three-dimensional tetrahedral Gauss-Legendre *volume* integration.

The remaining <0.05 µGal error in these tests includes the geometrical
approximations of the centre-shifted Nagy prisms and of the orthographic
grid boundaries, not just kernel quadrature error. This is **not** an
end-to-end accuracy guarantee for arbitrary DEMs.

### Remaining limitations

- Distinct near/far DEM heights are joined by a *sloping mass transition*
  across the first far row. This intentionally changes the physical
  model with respect to a sharp vertical DTM discrepancy. A configurable
  transition width and explicit near/far DEM reconciliation are still needed.
- Independent raster inputs and their density compartments can be mismatched
  along R1. The current implementation checks for one compatible prism
  on the boundary and fails early if matching is ambiguous or impossible.
- The **outer** face of the transition strip now has exactly matching
  triangular facets with the next far ring (0 unmatched in the uniform
  benchmark). All individual transition polyhedra have closed oriented
  triangular surfaces; tests also confirm the four bridge/corner joints.
  The complete multi-cell 3D union still needs a rigorous global
  geometry check for variable real-DTM heights and compartment steps.
- Real DTM / sea / seabed / multi-station GDAL paths still require
  end-to-end tests. The station loop now consistently retains the
  radius supplied to `bgc()` for projection, prism shift and polyhedra;
  validating the geodetic reference and the height datum remains open.

## 1 November 2026: conservative `adj_mesh` strategy and outer-seam check

The original `adj_mesh` adjusted the far-grid step to fit a common
near/far boundary. It has now been revised to make the constraints
explicit and predictable.

With user-supplied `gs1, R1, gs2, R2`:
- accept only **positive integer metres**; reject fractional inputs instead
  of rounding them silently;
- require an **even `gs1`**. Because station-centred meshes have an
  odd number of cells, an integer common radius requires even integer
  steps on **both** grids; `gs2` is shifted to the nearest compatible
  even integer within the allowed tolerance;
- preserve `gs1` **exactly**;
- increase `R1` to the *first permissible* near-field cell boundary
  for which the far-field grid can also share that same boundary;
- choose the closest compatible `gs2`, allowing at most **5%** relative
  change by default;
- increase `R2` to the first far-grid boundary beyond both the input
  `R2` and the first entire far cell beyond `R1`;
- never decrease either radius; keep the central cell aligned with the
  station and an **odd number of cells** across each complete domain;
- default maximum R1 enlargement is `max(gs1, gs2/2)`. Raise a
  descriptive error rather than silently increasing R1 further;
  `adj_mesh` exposes explicit `max_gs2_change` and
  `max_R1_increase` override arguments.

Mathematically:
`2*R1 = (2*n+1)*gs1 = (2*m+1)*gs2_adjusted` and
`R2 = R1 + k*gs2_adjusted` for a positive integer k.
The correction is idempotent, so repeated calls do not drift.

**Integer-only example**: `adj_mesh(50, 5000, 250, 10000)`
returns `(50, 5125, 250, 10125)` **as Python ints**.
Indeed `2*5125/50=205` and `2*5125/250=41` (both odd),
and `2*10125/250=81` (odd). The former example involving
`gs1=125` and fractional `gs2=312.5` is now **invalid**:
odd `gs1` and an integer `R1` are mathematically incompatible
under the existing centred-mesh convention.

The far DTM crop takes place after alignment, and the near DTM crop
also uses the adjusted radius. Automatic tuning produces even integer
steps and integer candidate radii before calling `adj_mesh`; when
`R1` is automatic it may search a modestly wider growth interval
(up to two far grid steps) to find a genuinely integral shared face.
The default `R2=167000 m` remains the nominal requested radius;
the aligned outer boundary may be slightly larger.

The gravity arrays and polyhedron vertex coordinates remain
`float64` even when these four control parameters are ints:
the principal benefit is *exact discrete mesh bookkeeping* and
reproducibility, not a material reduction in the solver's CPU cost.

**The previous outer-seam defect has been resolved.** The original
refinement created 78 hanging vertices (1.84 mm maximum offset)
in a synthetic 5 km test. It has been replaced by
the **dedicated R1 transition section** of
`relgravpy/_bouguer_correction.py`: each original coarse far cell
is now a single closed polyhedron with an arbitrary number of facets
on its prism-facing inner boundary and exactly the two *original*
triangles on its outer boundary. Adjacent corner cells use diagonals
matching both incoming side bridges. On the adjusted example above,
the new diagnostic finds 132 side-transition parent cells, 4 corner
cells and **zero unmatched outer triangles**, without modifying
the next coarse far ring.

The latest geometry check includes 5 independent 3D direct-volume
integrals (one per side and a corner) with maximum discrepancy
**0.00000499 µGal** against the surface-integral implementation.
The full 20 km-wide synthetic R1 benchmark returns
**-0.0362 µGal** residual for uniform 300 m terrain and
**-0.0474 µGal** for a 350/250 m height contrast, after adding
the independently integrated mass of the sloping bridge geometry.

The updated `polydron_grav` solver preserves consistent outward
facet winding on concave step geometries, rather than individually
flipping normals relative to the centroid. The first-ring
point-mass reduction is blocked when `R3 <= R1 + gs2`.

### Limitations and verification still required

The far/near DTM and density compartments must be compatible along
R1. Unsupported water/land layer changes or missing near cells
trigger explicit errors instead of silently creating disconnected
material. The transition interpolates two DEM surfaces across the
first coarse cell, so it changes the physical mass model relative
to a sharp elevation discontinuity. Full end-to-end GDAL and
multi-station validation with real geographic data remains open.

## 1 November 2026: lightweight survey-wide auto-mesh (experimental)

The public function `optimize_bgc_mesh()` now selects **one** set of
near/far mesh parameters for **all input stations**. It is called by
`bgc()` exactly once before preprocessing the input DTM, and only when
at least one of `gs1`, `R1`, or `gs2` is `None`. When all three
are provided explicitly, the optimizer is completely bypassed and
`target_error` is ignored.

Nominal `R2=167000 m` by default, **not** optimized.
As before, `adj_mesh()` may enlarge R1/R2 slightly to align grid
boundaries and adjust gs2 by at most 5%. The existing `R3=50000 m`
point-mass option affects the runtime-cost estimate, not the R2 default.

Example:

```python
bgc(
    x, y, dtm1, dtm2=dtm2,
    gs1=None, R1=None, gs2=None,
    R2=167000., target_error=0.05
)
```

`target_error` is measured in mGal and is the **maximum** allowed
*estimated* error across all stations: `max(error_proxy[station])`,
not the average. A positive tolerance is required only if some mesh
parameter is automatic. The estimated error is derived from a
fixed-cost, multiscale analysis of two small source-DTM windows
(33 × 33 and 25 × 25 output pixels) at **each** input station. The
optimizer then compares at most 4 × 4 × 5 = 80 candidate meshes using
only cheap scalar arithmetic. It does not perform a full gravity
calculation during tuning, create a 167-km optimization raster, or
invoke a trial calculation once per candidate.

The station features capture gradients, high-frequency residual
topography, relief and absolute topographic thickness; the
heuristic gravity proxy models smoothing, first-ring far quadrature
and spherical curvature. The selection minimizes estimated runtime
among candidates meeting the requested maximum estimated error.
If none meet the target, the least-error candidate is used and a
`RuntimeWarning` is emitted, **without preventing the Bouguer
computation**. Missing DTM preview windows use deliberately rough
fallback features and trigger a separate warning.

**Research limitation: `target_error=0.05` is NOT a certified
bound on the actual discrepancy from a converged gravity model.**
The fast error proxy is provisional and needs calibration on
topographically complex mountain and coastal bathymetry test
cases. Both the constants in the proxy and its effectiveness as an
error predictor must be checked against real numerical error maps,
especially near cliffs and steep submarine slopes, before treating
the target as a guaranteed accuracy constraint.

`Radi1()`, `GrdStep2()` and `Radi2()` remain in the source for
comparison but are no longer invoked by the automatic branch of
`bgc()`. Legacy `errg` is retained in the signatures for backwards
compatibility but does not affect the new optimizer. Do not remove
these functions until the optimizer has been calibrated.


### Full synthetic DEM validation against refined gravity

`tests/benchmark_bgc_full_topography.py` runs actual near-prism, conformal
R1 bridge and far spherical-polyhedron gravity, using full synthetic
mountain and coastal/bathymetric surfaces, with separate stations.
An independent centre-shifted Nagy prism reference is integrated at
100, 50, 25 and **12.5 m** spacing over the compact source support.
The synthetic mass contrast is identically zero outside that support,
so the nominal 167 km R2 introduces no omitted mass for this benchmark.
The reference itself has finite discretisation and slight planar-vs-
spherical geometry differences; it is **not exact potential truth**.

The first optimizer version selected `gs1=60 m` in steep mountains:
estimated maximum `0.0382 mGal`, actual mismatch `0.1261 mGal`.
The 25-to-12.5 m reference difference was <=`0.0053 mGal`.
This falsified the original proxy's claim to predict the requested
0.05 mGal maximum. The optimizer now penalizes near-grid sampling
coarser than native DEM resolution when local terrain slopes are
large, and adds a modest 1.5 empirical margin to its proxy.
This is **provisional calibration**, not a universal bound.

The latest **integer-only mesh** benchmark chooses
`gs1=30 m, R1=4125 m, gs2=250 m, R2=167125 m` for the
synthetic mountain: maximum predicted mismatch `0.03759 mGal`,
measured mismatch `0.00800 mGal` against the 12.5 m reference.
For coast/bathymetry with `R1` explicitly requested at 2000 m,
the chosen compatible integer mesh is
`gs1=240 m, R1=2040 m, gs2=240 m, R2=167160 m`.
The error proxy reports `0.05282 mGal` (above the target and so
correctly emits a RuntimeWarning), whereas the measured mismatch
in this particular case is `0.03431 mGal` across two stations.
These are the results of the 1 November 2026 integer-only benchmark,
not universal numerical bounds.
These are four synthetic station cases, not a general calibration
proof. The `tests/benchmark_bgc_full_topography.py` script now
includes numerical regression assertions at 0.05 mGal for these
two supported geometries. The incompatible mixed-density R1 crossing
remains explicitly reported instead of being silently accepted.

Before integer-only alignment, the same mountain test used a
fractional far step of 242.727 m and gave a 0.00767 mGal mismatch.
That configuration is now invalid under the integer-metre contract;
the new exactly aligned geometry is used in the regression assertions.

For the coast/bathymetry synthetic surface with `R1` requested
at 2 km and all R1-adjacent material on land, the optimizer chooses
the other parameters once for both stations, and aligns the actual
boundary slightly upward to 2040 m. Direct gravity mismatch is
`0.03431 mGal` (reference convergence `0.00163 mGal`),
with a sea density contrast beyond R1.

If the coast intersects R1, `_transition_polyhedral_gz` currently
rejects mixed land/water density transition faces with a ValueError
rather than pretending the shared mass mesh is conformal.
**Do not use this auto-mesh as an accuracy guarantee for mixed-density
coastal interfaces** until that volumetric bridge is implemented and
validated.

The 0.05 mGal tests above cover a few specific synthetic morphologies
and station locations. Further independent rugged/high-relief, sharp
cliff, marine layer and real-DEM trials are required to establish a
generally conservative error predictor. The tuner remains intentionally
lightweight; the expensive independent forward-model comparison runs
only as an offline validation benchmark, not inside `bgc()`.

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
- Verify real-DEM height/density differences across R1 (including sea/land);
  choose and validate the width of the prism-anchored sloping bridge,
  and verify whole-domain face and mass continuity for real DEMs.
- The radius-propagation inconsistency has been corrected: `st_loop` now
  uses the caller's reference `R` consistently. Geodetic datum and
  height-reference conventions still require end-to-end validation.
- Verify the complete Bouguer effect using variable-thickness synthetic
  topography with a converged independent spherical reference, including
  negative height contrasts, density signs, surface and seabed stations.

Priority P1:
- Replace/tune empirical selection of `R1`, `R2`, `gs2` and the `R3`
  point-mass approximation with verified error-budget criteria.
- Benchmark the analytic / surface quadrature switching of `polydron_grav`
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
2. **R1 seam:** default now keeps near Nagy prisms centre-shifted onto
   the spherical reference and anchors special far polyhedra to their
   outer faces. Verify the transitional volume geometry and any
   real-DTM surface and density interfaces; separate physical smoothing
   effects from numerical integration errors.
3. **Geometry convergence:** synthetic flat/rugged refinement shows residuals
   falling with cell width; repeat with exactly identical geographic
   footprints, different observation heights, latitude and real DEMs.
4. **R3 point-mass approximation:** quantify the result versus increasing
   `R3` and direct polyhedral integration.
5. **Performance:** the new far-field surface integral is vectorised and
   significantly faster than the previous cell-by-cell analytic polyhedron calculation
   implementation, but performance on many real survey stations and DEMs
   is still unmeasured. Profile end-to-end before considering compiled code.
