# -*- coding: utf-8 -*-
"""
Analytical tests for relgravpy.utils.min_curvature.

The numerical section is loaded directly from utils.py rather than importing
the whole module: utils.py also imports optional geospatial packages that
are unrelated to this interpolator.
"""
from pathlib import Path
import sys
import types
import warnings

import numpy as np
import pytest


def _load_min_curvature():
    # -------------------------------------------------------------------------
    # Isolate the exact implementation committed to utils.py.
    # -------------------------------------------------------------------------
    path = Path(__file__).resolve().parents[1] / "relgravpy" / "utils.py"
    text = path.read_text(encoding="utf-8")
    marker = "# Minimum curvature: continuous-curvature splines in tension"
    if marker not in text:
        raise AssertionError("min_curvature implementation missing from utils.py")
    source = "# " + marker + text.split(marker, 1)[1]
    module = types.ModuleType("_relgravpy_min_curvature_test")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


mc = _load_min_curvature()
min_curvature = mc.min_curvature


@pytest.fixture
def stations():
    rng = np.random.default_rng(42)
    return rng.uniform(0, 100, 160), rng.uniform(0, 100, 160)


# -----------------------------------------------------------------------------
# Analytical accuracy
# -----------------------------------------------------------------------------
def test_recovers_exact_plane(stations):
    x, y = stations
    z = 13.5 + 0.27 * x - 0.36 * y
    gx, gy, grid, info = min_curvature(
        x, y, z, region=(0, 100, 0, 100), spacing=2,
        return_info=True
    )
    xx, yy = np.meshgrid(gx, gy)
    assert grid.shape == (51, 51)
    assert gy[0] == 100 and gy[-1] == 0
    assert np.isfinite(grid).all()
    np.testing.assert_allclose(grid, 13.5 + 0.27 * xx - 0.36 * yy,
                               rtol=0, atol=1e-10)
    assert info["converged"]


def test_sparse_observations_cover_entire_region():
    rng = np.random.default_rng(13)
    x = rng.uniform(30, 70, 60)
    y = rng.uniform(25, 75, 60)
    z = 2 * np.exp(-((x - 48)**2 + (y - 55)**2) / 120)
    for tension in (0.0, 0.25, 0.75, 1.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, _, grid = min_curvature(
                x, y, z, (0, 100, 0, 100), 2,
                tension=tension, max_iterations=600
            )
        assert grid.shape == (51, 51)
        assert np.isfinite(grid).all()
        assert np.isfinite(grid[[0, -1], :][:, [0, -1]]).all()


def test_non_finite_observations_are_ignored(stations):
    x, y = stations
    z = np.sin(x / 11) + np.cos(y / 17)
    x[0] = np.nan
    z[2] = np.nan
    z[3] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, grid, info = min_curvature(
            x, y, z, (0, 100, 0, 100), 2, return_info=True
        )
    assert np.isfinite(grid).all()
    assert info["discarded_observations"] >= 3


def test_off_node_constraints_and_multigrid(stations):
    x, y = stations
    z = np.sin(x / 12) + np.cos(y / 13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, grid, info = min_curvature(
            x, y, z, (0, 100, 0, 100), 2, return_info=True
        )
    assert np.isfinite(grid).all()
    assert info["retained_observations"] > 20
    assert info["strides"][-1] == 1


def test_rectangular_spacing(stations):
    x, y = stations
    z = np.exp(-((x - 50)**2 + (y - 50)**2) / 500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, grid = min_curvature(
            x, y, z, (0, 100, 0, 100), (2, 4),
            max_iterations=500
        )
    assert grid.shape == (26, 51)
    assert np.isfinite(grid).all()


# -----------------------------------------------------------------------------
# Validation and numerical consistency
# -----------------------------------------------------------------------------
def test_bad_grid_step_rejected(stations):
    x, y = stations
    with pytest.raises(ValueError):
        min_curvature(x, y, x, (0, 100, 0, 100), 3)


def test_invalid_tension_rejected(stations):
    x, y = stations
    with pytest.raises(ValueError):
        min_curvature(x, y, x, (0, 100, 0, 100), 2, tension=1.5)


def test_pure_python_and_numba_sor_consistent():
    accelerated = mc._gmt_surface_sor_step
    slow = getattr(accelerated, "py_func", None)
    if slow is None:
        pytest.skip("Numba is not installed; pure Python kernel is used")
    rng = np.random.default_rng(92)
    u = rng.normal(size=(12, 12)).astype(np.float32)
    status = np.zeros((8, 8), dtype=np.uint8)
    briggs = np.zeros((8, 8, 6), dtype=np.float64)
    coeff, _, a02 = mc._gmt_surface_coefficients(1., 0.)
    args = (status, briggs, coeff, mc._GMT_SURFACE_NEIGHBORS,
            mc._GMT_SURFACE_QUADRANTS, a02, 1.4)
    u1, u2 = u.copy(), u.copy()
    step1 = accelerated(u1, *args)
    step2 = slow(u2, *args)
    np.testing.assert_allclose(u1, u2, rtol=0, atol=1e-6)
    assert step1 == pytest.approx(step2, abs=1e-7)



# -----------------------------------------------------------------------------
# GMT multigrid forecast regression (south-to-north interpolation)
# -----------------------------------------------------------------------------
def test_multigrid_forecast_uses_gmt_south_to_north_fraction():
    """Regress GMT C's unusual i/previous_stride interpolation weights."""
    previous = np.zeros((10, 10), dtype=np.float32)
    previous[2:-2, 2:-2] = (
        10.0 * np.arange(6, dtype=np.float32)[:, None] +
        np.arange(6, dtype=np.float32)[None, :]
    )
    refined, frozen = mc._gmt_surface_bilinear(
        previous, nx=26, ny=26, factor=5, previous_stride=10
    )
    # GMT starts at the SOUTH edge of each coarse-grid cell. Since the
    # original rows are north-to-south, row +1 is weighted 0.4 top + 0.6
    # bottom for the first 10->2 refinement; naive np.interp gives 1.0.
    assert refined[3, 2] == pytest.approx(6.0, abs=1e-6)
    assert refined[6, 2] == pytest.approx(9.0, abs=1e-6)
    assert refined[2, 3] == pytest.approx(0.1, abs=1e-6)
    assert refined[3, 3] == pytest.approx(6.1, abs=1e-6)
    assert frozen[0, 0] == 5
    assert frozen[1, 1] == 0
