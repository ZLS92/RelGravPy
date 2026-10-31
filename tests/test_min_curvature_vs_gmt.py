# -*- coding: utf-8 -*-
"""
Reference comparison against the independent GMT surface executable.

Requirements for running this optional development test:
    GMT 6.4+ in PATH and the already-declared NumPy / pytest dependencies.

    python -m pytest tests/test_min_curvature_vs_gmt.py -v -s

GMT -Qr disables automatic working-region expansion, so both solvers use
the same exact requested region. Tests are skipped when GMT is unavailable.
They are NOT evidence of numerical agreement until executed with GMT.
"""
from pathlib import Path
import io
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import pytest

from test_min_curvature import min_curvature


# -----------------------------------------------------------------------------
# Comparison on irregular, non-grid-aligned observations
# -----------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("gmt") is None,
                    reason="GMT surface executable unavailable")
@pytest.mark.parametrize("tension", (0.0, 0.25, 1.0))
def test_min_curvature_matches_gmt_surface(tension):
    rng = np.random.default_rng(2026)
    x = rng.uniform(1.5, 98.5, 180)
    y = rng.uniform(1.5, 98.5, 180)
    z = (
        3.2 * np.exp(-((x - 35)**2 + (y - 61)**2) / 280)
        - 1.7 * np.exp(-((x - 73)**2 + (y - 33)**2) / 160)
        + 0.02 * x - 0.035 * y
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, calculated, info = min_curvature(
            x, y, z, (0, 100, 0, 100), 2,
            tension=tension, max_iterations=1000,
            return_info=True
        )

    # -------------------------------------------------------------------------
    # Obtain the independent GMT reference grid.
    # -------------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="min_curvature_gmt_") as tmpdir:
        folder = Path(tmpdir)
        samples = folder / "stations.xyz"
        output = folder / "reference.nc"
        np.savetxt(samples, np.column_stack((x, y, z)), fmt="%.16g")
        command = [
            "gmt", "surface", str(samples), "-R0/100/0/100", "-I2",
            f"-G{output}", f"-T{tension}", "-N1000", "-Z1.4", "-Qr"
        ]
        subprocess.run(command, cwd=folder, capture_output=True,
                       text=True, check=True)
        result = subprocess.run(
            ["gmt", "grd2xyz", str(output)], cwd=folder,
            capture_output=True, text=True, check=True
        )
        points = np.loadtxt(io.StringIO(result.stdout))
        reference = np.full_like(calculated, np.nan)
        ic = np.rint(points[:, 0] / 2).astype(int)
        ir = np.rint((100 - points[:, 1]) / 2).astype(int)
        reference[ir, ic] = points[:, 2]

    assert np.isfinite(reference).all()
    assert np.isfinite(calculated).all()
    diff = calculated - reference
    spread = np.ptp(reference)
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_error = float(np.max(np.abs(diff)))
    print(
        f"tension={tension}: rmse={rmse:.9g}, "
        f"max_abs={max_error:.9g}, span={spread:.9g}, "
        f"converged={info['converged']}"
    )

    # -------------------------------------------------------------------------
    # Preliminary acceptance limits, to be tightened after reference runs.
    # -------------------------------------------------------------------------
    assert spread > 0
    assert rmse / spread < 0.001
    assert max_error / spread < 0.002


# -----------------------------------------------------------------------------
# Severe extrapolation: all observations confined to a central area.
# Compare every node, including the outer corners well beyond their convex hull.
# -----------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("gmt") is None,
                    reason="GMT surface executable unavailable")
@pytest.mark.parametrize("tension", (0.0, 0.25))
def test_sparse_center_and_extrapolation_match_gmt(tension):
    rng = np.random.default_rng(2031)
    x = rng.uniform(30, 70, 85)
    y = rng.uniform(27, 73, 85)
    z = (
        2.5 * np.exp(-((x - 45)**2 + (y - 62)**2) / 200)
        - 1.0 * np.exp(-((x - 65)**2 + (y - 38)**2) / 180)
        + 0.016*x - 0.011*y
    )
    region = (0, 100, 0, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, calculated, info = min_curvature(
            x, y, z, region, 2, tension=tension,
            max_iterations=15000, return_info=True
        )

    with tempfile.TemporaryDirectory(prefix="min_curvature_sparse_") as tmpdir:
        folder = Path(tmpdir)
        samples = folder / "sparse.xyz"
        output = folder / "reference.nc"
        np.savetxt(samples, np.column_stack((x, y, z)), fmt="%.16g")
        subprocess.run([
            "gmt", "surface", str(samples), "-R0/100/0/100",
            "-I2", f"-G{output}", f"-T{tension}",
            "-N15000", "-Z1.4", "-Qr",
        ], cwd=folder, capture_output=True, text=True, check=True)
        raw = subprocess.run(
            ["gmt", "grd2xyz", str(output)], cwd=folder,
            capture_output=True, text=True, check=True
        ).stdout
        points = np.loadtxt(io.StringIO(raw))
    reference = np.full((51, 51), np.nan)
    col = np.rint(points[:, 0] / 2).astype(int)
    row = np.rint((100 - points[:, 1]) / 2).astype(int)
    reference[row, col] = points[:, 2]

    assert reference.shape == calculated.shape == (51, 51)
    assert np.isfinite(calculated).all() and np.isfinite(reference).all()
    error = calculated - reference
    rmse = np.sqrt(np.mean(error**2))
    peak_error = np.max(np.abs(error))
    scale = np.ptp(reference)
    assert scale > 0
    print(f"central-only tension={tension}: rmse={rmse:.9g}, "
          f"max_abs={peak_error:.9g}, span={scale:.9g}, "
          f"converged={info['converged']}")
    assert rmse / scale < 0.01
    assert peak_error / scale < 0.05
