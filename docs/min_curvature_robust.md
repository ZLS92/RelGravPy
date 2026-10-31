# Minimum-curvature interpolation and padding

RelGravPy provides one Python implementation of minimum-curvature gridding:
`min_curvature()` in `relgravpy/utils.py`. Its sparse numerical engine is
maintained separately in `relgravpy/_min_curvature_core.py` for readability.
No external gridding executable or Numba installation is required.

## Numerical formulation

For the detrended, normalized field `u`, the algorithm minimizes:

```text
E(u) = data_weight * ||H u - z||²
     + (1 - tension) * (||Dxx u||² + 2||Dxy u||² + ||Dyy u||²)
     + tension * (||Dx u||² + ||Dy u||²)
     + anchor * ||u||²
```

The sparse observation matrix `H` uses bilinear interpolation at actual
station coordinates; it does not snap measurements to nodes. Sparse first-
and second-order finite differences define the bending energy. Positive
`anchor` penalizes residual departures from a fitted planar trend, limiting
poorly constrained long-wavelength modes in padded regions.

The *mathematical model* is unique. `linear_solver="direct"` and
`linear_solver="cg"` are two numerical methods for solving the **same**
linear system; `"auto"` selects according to grid size.

The method pursues the same practical goal as commercial minimum-curvature
tools, but is not claimed to reproduce their proprietary implementation.

## Use with scattered gravity measurements

```python
from relgravpy import utils as utl

gx, gy, grid, info = utl.min_curvature(
    x, y, bouguer_mgal,
    region=(450000, 470000, 6400000, 6420000),
    spacing=400,
    tension=0.15,
    anchor=0.01,
    block="median",
    block_size=400,
    return_info=True,
)
```

Coordinates must be projected and the grid step expressed in the same
distance units. The output is finite over the entire rectangular region,
including areas without observations, but it remains an *extrapolation*.

Block aggregation reuses the existing `utils.block_m()` and accepts
`"mean"`, `"median"` or `None`. Depending on spatial sampling, reducing
observations can improve or worsen conditioning.

## Fill missing values in an existing grid

```python
filled = utl.fillnan(
    original_grid,
    xy=(x_coords, y_coords),
    method="min_curvature",
    tension=0.15,
    anchor=0.01,
)
```

Only originally missing (NaN) values are replaced. Finite observations
remain **exactly unchanged**.

## Expand the grid by padding

```python
padded, original_indices = utl.pad_array(
    original_grid,
    padw=10,
    ptype="cells",
    mode="min_curvature",
    xy=(x_coords, y_coords),
    tension=0.15,
    anchor=0.01,
)
```

The original function returns the padded array and an index tuple
`(row_start, row_end, col_start, col_end)`. If `xy` is absent the
grid uses unit-spaced pixel coordinates; for real datasets pass coordinate
vectors to preserve the physical aspect ratio.

For compatibility with previous calls, `method="surface"` in `fillnan`,
`mode="surface"` in `pad_array` and `method="surface"` in `xyz2grid`
now use this same SciPy interpolation; they no longer invoke an external
gridding executable. Other methods, including `nearest`, `gdal` and
`thin_plate`, retain their independent purposes.

## Limits of the numerical result

A sparse or incomplete survey does not constrain the gravity field at large
distances. Positive regularization makes the computation stable, **not**
necessarily physically accurate.

Reference tests were performed against published South Australian Bouguer
gravity grids, treating withheld gridded pixels as a numerical benchmark:

| Withheld reference values | RMSE vs reference |
| --- | ---: |
| Random 30% of interior cells | 0.0123 mGal |
| 17 × 17 cell central gap | 0.9064 mGal |
| Seven parallel profiles retained | 0.5728 mGal |
| Only a central station cluster retained | 10.3183 mGal |
| 8-cell (3.2 km) border extrapolated | 6.8159 mGal |

These errors are against an already gridded published product, not
independent original gravity readings. See
[the published benchmark](min_curvature_published_gravity_benchmark.md).

## Reproducibility

Run the interpolation, padding and preprocessing regression tests:

```bash
PYTHONPATH=tests:. python -m pytest \
    tests/test_min_curvature.py \
    tests/test_min_curvature_core.py \
    tests/test_min_curvature_integration.py
```

Recreate the synthetic sampling maps:

```bash
PYTHONPATH=tests:. python examples/min_curvature_gallery.py
```

Run the real published gravity benchmark:

```bash
PYTHONPATH=tests:. python examples/benchmark_published_gravity.py
```

## Background

Smith, W. H. F., & Wessel, P. (1990), *Gridding with continuous curvature
splines in tension*, Geophysics **55**(3), 293–305.
https://doi.org/10.1190/1.1442837

The RelGravPy solver uses a separately implemented sparse variational
formulation; this citation supplies related scientific background, not a
claim that the finite-difference formulations are numerically identical.
