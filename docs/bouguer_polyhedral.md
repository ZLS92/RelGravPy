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

The numerical routine `grav_modelling.gravicalc2_full` returns the Cartesian
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
  polyhedron. The deterministic `gravicalc2_full` kernel calculates gravity.
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

## Still to validate before scientific use

1. **Full DEM integration:** compare against the previous Harmonica
   tesseroid implementation at known stations and for land, sea surface,
   and sea-bottom station types.
2. **R1 seam:** the near-field prisms use a constant cell-wise curvature
   shift, while far-field polyhedra follow spherical geometry. The junction
   is approximate, not mathematically watertight. Quantify its maximum
   mismatch or introduce a dedicated transition band if necessary.
3. **Geometry convergence:** systematically refine `gs1`/`gs2`; examine
   topographic ruggedness, latitude, grid size and large radii.
4. **R3 point-mass approximation:** quantify the result versus increasing
   `R3` and direct polyhedral integration.
5. **Performance:** `gravicalc2_full` executes Python loops for each
   polyhedron. Large far-field meshes will be slower than a fully vectorised
   or compiled implementation. Profile and batch/compile only after accuracy
   is verified.
