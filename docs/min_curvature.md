# Minimum curvature interpolation in RelGravPy

`min_curvature` is an experimental, dependency-light implementation of the
numerical core of the **GMT `surface`** algorithm. It is implemented as
functions in `relgravpy/utils.py`, without requiring the GMT executable.
`gmt_surf` remains unchanged and can be used as a reference.

## Scientific reference

**Smith, W. H. F., & Wessel, P. (1990).**
Gridding with continuous curvature splines in tension.
*Geophysics*, **55**(3), 293–305.
https://doi.org/10.1190/1.1442837

The implementation follows the GMT C implementation (including its
12-point finite-difference stencil, Briggs off-grid constraints, trend
removal/restoration, natural boundaries, multigrid and SOR):

https://github.com/GenericMappingTools/gmt/blob/master/src/surface.c

The GMT C implementation is LGPL-3.0-or-later; preserve the upstream
attribution and comply with that license when redistributing derived code.

## Example

```python
import numpy as np
from relgravpy import utils as utl

rng = np.random.default_rng(42)
x = rng.uniform(200, 800, 120)
y = rng.uniform(300, 700, 120)
z = (3.2 * np.exp(-((x - 450)**2 + (y - 550)**2) / 25000)
     + 0.001 * x)

gx, gy, surface, info = utl.min_curvature(
    x, y, z,
    region=(0, 1000, 0, 1000),
    spacing=20,
    tension=0.0,
    max_iterations=1000,
    return_info=True,
)

assert surface.shape == (51, 51)
assert np.isfinite(surface).all()
print(info["converged"])
```

The output arrays are `gx` (ascending x), `gy` (descending y,
GMT-style north-to-south row order) and `surface` (shape `ny, nx`).
Use projected Cartesian coordinates rather than raw longitude/latitude.

## Numerical characteristics

- `tension=0` gives minimum curvature (biharmonic); `0 < tension <= 1`
  gives splines in tension, with `tension=1` harmonic.
- The returned surface is computed on every grid node within `region`,
  including outside the convex hull of the observations. The result is an
  extrapolation in such areas; **absence of NaNs does not indicate accuracy**.
- No nearest-neighbour fill is performed after interpolation.
- Uses NumPy, already declared by the project. Numba is used when the existing
  optional `[fast]` extra is present, and is not a mandatory dependency.
- The implementation does not yet replicate all GMT options: automatic
  working-region expansion introduced in GMT 6.4+, pixel registration,
  periodic longitude, breaklines, upper/lower gridded bounds, masking,
  radius-based initialisation and GMT I/O are not implemented.
- Results depend on convergence, grid dimensions and the distribution of
  observations. Always check `info["converged"]` and inspect residuals.

## Tests

```bash
# No GMT installation required: analytical accuracy, coverage and constraints
python -m pytest tests/test_min_curvature.py -v

# Independent GMT reference: skips when gmt is not on PATH
python -m pytest tests/test_min_curvature_vs_gmt.py -v -s
```

The comparison test invokes GMT with `-Qr` (no expanded working region)
and compares all nodes, not only observations. It tests tensions 0, 0.25 and 1,
reporting RMSE, max error and convergence. These are **provisional acceptance
criteria**: execution against a real GMT installation and refinement of
tolerances are required before asserting numeric equivalence.

## Development status

**Experimental / review required.** Analytic Python tests have been run against
the pre-integration implementation (8 passed); independent GMT reference
tests have not been executed in this environment because GMT is absent.
The branch should only be merged into `main` after reference comparison,
potential discrepancy correction and performance profiling on typical
RelGravPy gravimetric surveys.
