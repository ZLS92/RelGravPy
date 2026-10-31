# Robust minimum-curvature gridding: gaps, padding and block preprocessing

Experimental implementation on branch `feature/min-curvature` of RelGravPy.
The existing `gmt_surf` remains unchanged. The new public entry point is
`relgravpy.utils.min_curvature`.

## Scientific scope

The `solver="gmt"` mode is a translation of the finite-difference
`GMT surface` algorithm of Smith & Wessel (1990), with a known numerical
discrepancy for extreme off-hull extrapolation at zero tension.

The new **`solver="robust"` mode** is a separate, regularized, discrete
minimum-bending-energy model. It is similar in *purpose* to the
minimum-curvature gridding and extension workflows in Oasis montaj, but
does not claim identical mathematics, boundary constraints, or outputs.

It minimizes the following quadratic functional in the detrended field
`u` on a rectangular grid:

```text
E(u) = data_weight * ||H*u - z_detrended||²
     + (1-tension) * [||Dxx*u||² + 2||Dxy*u||² + ||Dyy*u||²]
     + tension * [||Dx*u||² + ||Dy*u||²]
     + anchor * ||u||²
```

Here `H` is the bilinear interpolation operator mapping gridded model
values onto real, non-node-aligned station coordinates. The `D` operators
are sparse finite differences scaled for `dx/dy`.

A positive `anchor` penalizes residual departure from the fitted planar
background, suppressing poorly constrained low-frequency extrapolation
modes. It changes the model deliberately: it is **not** a free numerical
trick, and `anchor` sensitivity should be examined in the areas being
padded. Strict positive regularization makes the optimization well posed.

SciPy sparse direct LU is used for grids up to 7000 cells and Jacobi-
preconditioned conjugate gradients on larger grids (or via
`linear_solver="direct"` / `"cg"`). The numerical solution and observation
misfit are reported when `return_info=True`.

## Example 1: scattered XYZ including the outside of the convex hull

```python
import numpy as np
from relgravpy import utils as utl

rng = np.random.default_rng(2026)
x = rng.uniform(25, 75, 100)
y = rng.uniform(30, 70, 100)
z = 0.04*x - 0.02*y + 2*np.exp(-((x-48)**2+(y-55)**2)/300)

gx, gy, grid, info = utl.min_curvature(
    x, y, z,
    region=(0, 100, 0, 100), spacing=2,
    solver="robust", tension=0.0,
    anchor=0.01, data_weight=1e4,
    return_info=True,
)
assert grid.shape == (51, 51)
assert np.isfinite(grid).all()
print(info["observation_rmse"])
```

## Example 2: fill NaN holes and pad an existing grid

Supply coordinate vectors and a regular 2-D raster. The Y coordinate
vector may be north-to-south, and the input may contain holes and NaNs
on its exterior boundary; the full coordinate extent is preserved.

```python
x = np.arange(0., 101., 2.)
y = np.arange(100., -1., -2.)
X, Y = np.meshgrid(x, y)
raster = 0.1*X - 0.025*Y + np.exp(-((X-60)**2+(Y-40)**2)/320)
raster[10:22, 18:31] = np.nan

px, py, padded, info = utl.min_curvature(
    x, y, raster, spacing=2,
    solver="robust", padding=10,
    anchor=0.01, return_info=True,
)
# 51x51 original + 10 cells on each side in both directions.
assert padded.shape == (71, 71)
assert np.isfinite(padded).all()
```

No nearest-neighbor replacement is performed after gridding; the
computational domain is expanded and solved as a whole.

## Example 3: use the EXISTING block_m utility

`min_curvature` calls `utils.block_m` before gridding when requested.
It does **not** implement a competing binning method.

```python
_, _, gr_mean = utl.min_curvature(
    x, y, z, region=(0,100,0,100), spacing=2,
    solver="robust", block="mean", block_size=8,
)
_, _, gr_median = utl.min_curvature(
    x, y, z, region=(0,100,0,100), spacing=2,
    solver="robust", block="median", block_size=8,
)
```

Notes:
- Block **mean** reduces redundant/repeated measurements and averages
  independent random noise.
- Block **median** is more resistant to localized outliers, but is not
  always preferable for a clean, dense survey.
- Aggregation **does not universally improve conditioning**. In a
  21×21-node regularized test with 600 measurements at 100 repeated
  positions (6 per site), using 100 blocks reduces the estimated matrix
  condition number from roughly 155,000 to 26,000 (~6-fold).
- With a 2-m within-block positional spread, the same reduction instead
  raises the condition number from roughly 7,040 to 18,500 (~2.6-fold).
  Removing spatial constraints may leave the inverse problem worse
  conditioned. Reproduce with `tests/test_min_curvature_robust.py`.

## Synthetic validation (2026-10-31)

The following values compare the reconstructed robust surface with a
**known synthetic analytical reference**, in the original arbitrary field
units (not physical mGal). All evaluated raster cells were finite.

| Observation geometry | RMSE inside hull | RMSE outside hull |
| --- | ---: | ---: |
| 180 random observations spread over the region | 0.050 | 0.056 |
| 85 observations confined to central area | 0.020 | 2.022 |
| 140 observations along five parallel profiles | 0.236 | 0.233 |
| 546 observations around a large internal void | 0.250 | 0.005 |

These results emphasize the distinction between **completeness** (no NaNs)
and **accuracy**. The fit may be excellent near data and fail by orders
of magnitude where the field is unconstrained. In operational work,
check the observation convex hull, distance-to-data, residuals, changes
with padding and regularization, and reference data before interpreting
gravity gradients.

Validation in CI includes:

- Analytical exact plane, full-grid, off-node and SOR regression tests.
- Real GMT 6.5.0 comparison for `solver="gmt"` at T=0, 0.25, 1.
- Robust: full finite grid, accurate retention of measured grid nodes,
  NaN holes, anisotropic padding, sparse-center extrapolation, direct
  versus CG agreement, mean and median via the existing `block_m`.
- A strict expected-failure regression for the unresolved
  zero-tension GMT-vs-Python extreme extrapolation case.

## Parameters that control behavior

`tension`: weight between bending and gradient energy. Moderate positive
tension may damp oscillations but tends to suppress localized anomalies.

`anchor`: weight on residual-to-plane field, default 0.01. Increasing it
stabilizes far-field extrapolation but also imposes the planar background.

`data_weight`: squared data fidelity, default 1e4. High values reproduce
the original observations closely but may increase system conditioning.

`padding`: integer or pair of nonnegative numbers of cells per side.
The total rectangular output dimensions increase by twice that count.

`block`: `None`, `"mean"` or `"median"`; `block_size` is in the
same planar distance units as X/Y, defaulting to the smaller grid step.

## Reference and licensing

Smith, W. H. F., & Wessel, P. (1990). *Gridding with continuous curvature
splines in tension*. Geophysics **55**(3), 293–305.
https://doi.org/10.1190/1.1442837

GMT source: https://github.com/GenericMappingTools/gmt/blob/master/src/surface.c
(LGPL-3.0-or-later; preserve licensing attribution for derived portions).

Oasis montaj is a separate proprietary software package; no use of its
source code or claim of numerical equivalence is made here.
