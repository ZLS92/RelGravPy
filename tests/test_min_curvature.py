# -*- coding: utf-8 -*-
"""Tests for the single SciPy-based minimum-curvature implementation.

Import only the numerical functions of utils.py, avoiding unrelated optional
GIS dependencies of this large legacy module.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import scipy as sp


def _load_min_curvature():
    repo = Path(__file__).resolve().parents[1]
    package = sys.modules.get("relgravpy")
    if package is None:
        package = types.ModuleType("relgravpy")
        package.__path__ = [str(repo / "relgravpy")]
        sys.modules["relgravpy"] = package

    backend_path = repo / "relgravpy" / "_min_curvature_core.py"
    spec = importlib.util.spec_from_file_location(
        "relgravpy._min_curvature_core", backend_path
    )
    backend = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = backend
    spec.loader.exec_module(backend)

    path = repo / "relgravpy" / "utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"block_m", "min_curvature"}
    ]
    assert len(selected) == 2

    module = types.ModuleType("_relgravpy_min_curvature_test")
    module.__file__ = str(path)
    module.__package__ = "relgravpy"
    module.np = np
    module.sp = sp
    module._curvature_solver = backend._curvature_solver
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 str(path), "exec"), module.__dict__)
    return module


mc = _load_min_curvature()
min_curvature = mc.min_curvature


@pytest.fixture
def stations():
    rng = np.random.default_rng(42)
    return rng.uniform(0, 100, 160), rng.uniform(0, 100, 160)


# -----------------------------------------------------------------------------
# Analytical plane and complete rectangular domain
# -----------------------------------------------------------------------------


def test_recovers_exact_plane(stations):
    x, y = stations
    z = 13.5 + 0.27 * x - 0.36 * y
    gx, gy, grid, info = min_curvature(
        x, y, z, region=(0, 100, 0, 100), spacing=2, return_info=True
    )
    X, Y = np.meshgrid(gx, gy)
    np.testing.assert_allclose(grid, 13.5 + 0.27 * X - 0.36 * Y,
                               rtol=0, atol=1e-8)
    assert info["converged"]


def test_sparse_stations_cover_complete_grid():
    rng = np.random.default_rng(71)
    x, y = rng.uniform(30, 70, 50), rng.uniform(30, 70, 50)
    z = np.sin(x / 30) + np.cos(y / 30)
    gx, gy, grid = min_curvature(
        x, y, z, region=(0, 100, 0, 100), spacing=2
    )
    assert grid.shape == (51, 51)
    assert len(gx) == len(gy) == 51
    assert np.isfinite(grid).all()


def test_nonfinite_observations_are_discarded(stations):
    x, y = stations
    z = np.sin(x / 20) + np.cos(y / 30)
    z[:4] = np.nan
    _, _, grid, info = min_curvature(
        x, y, z, region=(0, 100, 0, 100), spacing=2, return_info=True
    )
    assert np.isfinite(grid).all()
    assert info["discarded_observations"] >= 4


def test_rectangular_spacing():
    x = np.array([1, 8, 20, 37, 49, 27, 5, 42], dtype=float)
    y = np.array([3, 12, 18, 6, 25, 22, 29, 15], dtype=float)
    z = np.sin(x / 10) - np.cos(y / 10)
    gx, gy, grid = min_curvature(
        x, y, z, region=(0, 50, 0, 30), spacing=(2, 3)
    )
    assert grid.shape == (11, 26)
    np.testing.assert_allclose(np.diff(gx), 2, atol=1e-10)
    np.testing.assert_allclose(np.diff(gy), -3, atol=1e-10)


def test_invalid_input_rejected(stations):
    x, y = stations
    z = np.sin(x)
    with pytest.raises(ValueError):
        min_curvature(x, y, z, (0, 100, 0, 100), spacing=-2)
    with pytest.raises(ValueError):
        min_curvature(x, y, z, (0, 100, 0, 100), spacing=2, tension=1.2)
    with pytest.raises(ValueError):
        min_curvature(x, y, z, (0, 100, 0, 100), spacing=2, anchor=0)
    with pytest.raises(ValueError):
        min_curvature(x, y, z, (0, 100, 0, 100), spacing=2, padding=-1)


# -----------------------------------------------------------------------------
# Source-style safeguards for the public interpolation implementation
# -----------------------------------------------------------------------------


def test_utils_has_one_curvature_function_and_module_level_imports():
    """Prevent a second gridding backend and local imports from reappearing."""
    path = Path(__file__).resolve().parents[1] / "relgravpy" / "utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    names = [node.name for node in functions]

    assert names.count("min_curvature") == 1
    assert not any("gmt" in name.lower() for name in names)

    first_function = min(node.lineno for node in functions)
    all_imports = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(node.lineno < first_function for node in all_imports)
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "math"
        for node in all_imports
    )

    # Multiple values must be assigned to a variable before returning.
    all_returns = [
        node for node in ast.walk(tree) if isinstance(node, ast.Return)
    ]
    assert not any(
        isinstance(node.value, (ast.Tuple, ast.List, ast.Dict))
        for node in all_returns
    )
