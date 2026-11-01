# Polygrav and Nagy: the single-solver gravimetric API

The research implementation lives on
`feat/bouguer-polyhedral-curvature`. Do **not** merge into `main`
without end-to-end DEM/land-sea validation.

## Public gravity solvers in `relgravpy/grav_modelling.py`

- `Nagy(coordinates, prism, density).compute_gz()`: exact rectangular
  prism formula with the original `.gz()` method retained. Legacy
  `nagy` is an alias for the same class (not a duplicate implementation).
- `Polygrav(corners, faces, density).compute_gz(observations)`: one
  general polyhedral class. `corners` is shape `(n_vertices,3)`,
  `faces` contains arbitrary planar polygons in historical
  `[n, vertex_id_0, ..., vertex_id_(n-1)]` format (0-based indices).
  One observation produces a scalar; a list of observations produces
  an array.

Units: coordinates and heights in metres; density in kg/m³;
gravitational attraction `gz` in mGal **positive downward**, with
`G=6.6743e-11` SI in both solvers.

The Polygrav backend selects a corrected deterministic
Goetze–Lahmeyer/MATLAB line-solid-angle implementation for near
observations (`distance <= 5 * maximum body span`) and a
seven-point outward-triangle surface integral for distant bodies.
The private `method="analytic"` and `method="surface"` switches
are exposed for benchmarking. The edge-integral cache is now
recreated for each observation; it is **not** reused between stations.
Old duplicate `PolyGrav`, `GotzeGrav` and the top-level
`gravicalc2_full` have been removed from this branch.

The constructor checks finite coordinates/density, face indices,
nondegenerate polygons and that each undirected edge occurs twice.
It auto-orients faces outward based on body centroid; this orientation
heuristic is intended for convex / star-shaped DEM cells. Strongly
concave or self-intersecting surfaces need separate manifold and
winding tests. Nonplanar polygon faces should be triangulated.

## Validation

`tests/test_polygrav_nagy.py` compares both solvers on a Cartesian
product of five prism sizes/thicknesses, five horizontal station
offsets, and four densities (0, 1000, 2670, -1800 kg/m³):
**100 size-distance-density combinations**.

For these synthetic cases, the comparison requires
`rtol <= 1.5e-4` with `atol <= 2e-6 mGal` (0.002 µGal).
The absolute tolerance protects near-zero signals; it must not be
misreported as a global relative-accuracy guarantee.
The GitHub Actions benchmark reports a **maximum absolute discrepancy of
4.45 × 10⁻⁵ µGal** in these 100 cases. The maximum relative discrepancy
is **0.0522%**, occurring for a very small signal (~0.0853 µGal)
from a 10 km-wide / 200 m-thick prism at eight body widths offset.
This is a test-set statistic, not a universal error bound.
 Additional
tests check interior stations, stations below the body, multi-station
cache independence, density linearity, left/right symmetry,
arbitrary faces with extra collinear nodes, open mesh rejection, and
the distant point-mass asymptote.

The previous 7-point rule at intermediate range failed for broad,
shallow prisms by up to 12% in some cases. The automatic transition
between analytic and numerical kernels was moved from 1.75 to
5 times maximum body span. It was **not** resolved by relaxing tests.

The Bouguer curvature routine now uses `Polygrav` for its near
spherical-polyhedron fallback. Efficient vectorized batches are
still retained for ordinary distant spherical polyhedral cells;
the default within R1 remains the horizontally translated Nagy prism.

### Geometry remains a separate task

The R1 transition currently still uses 8-vertex wedges and splits
the first far ring. Tests
`tests/diagnose_r1_outer_seam.py` demonstrate nonmatching outer-ring
vertices. These geometry problems are not resolved by the solver
unification and should be fixed with an arbitrary-facet transition
builder, then independently validated for geometric closure and gravity.
