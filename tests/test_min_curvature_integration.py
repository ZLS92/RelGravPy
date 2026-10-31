# -*- coding: utf-8 -*-
"""Integration tests for minimum-curvature options in legacy utils helpers.

Load only the existing pad_array() and fillnan() function definitions to avoid
unrelated optional GIS dependencies in the monolithic utils.py module.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from test_min_curvature_robust import mc, min_curvature


# -----------------------------------------------------------------------------
# Execute the exact definitions from utils.py in the same isolated namespace
# used by the GMT/robust solver tests.
# -----------------------------------------------------------------------------
_source_file = Path(__file__).resolve().parents[1] / "relgravpy" / "utils.py"
_tree = ast.parse(_source_file.read_text(encoding="utf-8"))
_functions = [
    node for node in _tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in ("pad_array", "fillnan")
]
assert len(_functions) == 2
exec(compile(ast.Module(body=_functions, type_ignores=[]),
             str(_source_file), "exec"), mc.__dict__)
fillnan = mc.fillnan
pad_array = mc.pad_array


# -----------------------------------------------------------------------------
def _field(x, y):
    """Smooth planar plus Gaussian anomaly for numerical test grids."""
    return (1.5 + 0.02*x - 0.015*y
            + 1.8*np.exp(-((x-45)**2+(y-55)**2)/450.))


# -----------------------------------------------------------------------------
def test_fillnan_curvature_index_grid_preserves_original_data():
    x = np.arange(0., 101., 5.)
    y = np.arange(0., 101., 5.)  # ascending Y: tests orientation remap
    X, Y = np.meshgrid(x, y)
    known = _field(X, Y)
    missing = known.copy()
    missing[8:13, 8:13] = np.nan
    filled = fillnan(missing, method="min_curvature", tension=0.15)
    assert filled.shape == known.shape
    assert np.isfinite(filled).all()
    np.testing.assert_array_equal(filled[np.isfinite(missing)],
                                  known[np.isfinite(missing)])
    # The default coordinate system is pixel indices, not coordinates in m.
    assert np.max(np.abs(filled[8:13,8:13] - known[8:13,8:13])) < 0.5


# -----------------------------------------------------------------------------
def test_fillnan_georeferenced_xy_in_both_forms():
    x = 452904.697640 + np.arange(21)*400.
    y = 6426297.029215 - np.arange(21)*400.
    X,Y = np.meshgrid(x,y)
    field = _field((X-x[0])/80., (y[0]-Y)/80.)
    measured = field.copy()
    measured[7:13,8:14] = np.nan

    # Both XY axis vectors and full meshgrid arrays are accepted.
    a = fillnan(measured,xy=(x,y),method="min_curvature",
                tension=0.2,anchor=0.03)
    b = fillnan(measured,xy=(X,Y),method="robust",
                tension=0.2,anchor=0.03)
    assert a.shape == b.shape == field.shape
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-8)
    np.testing.assert_array_equal(a[np.isfinite(measured)],
                                  measured[np.isfinite(measured)])


# -----------------------------------------------------------------------------
def test_fillnan_reversed_x_and_y_axes():
    x = np.arange(100., -1., -5.)     # descending X
    y = np.arange(0.,101.,5.)        # ascending Y
    X,Y = np.meshgrid(x,y)
    original = _field(X,Y)
    damaged = original.copy()
    damaged[5:10,5:10] = np.nan
    completed = fillnan(damaged,xy=(x,y),method="min_curvature")
    assert completed.shape == original.shape
    assert np.isfinite(completed).all()
    np.testing.assert_array_equal(completed[np.isfinite(damaged)],
                                  damaged[np.isfinite(damaged)])


# -----------------------------------------------------------------------------
def test_pad_array_curvature_extended_domain():
    x = np.arange(0.,101.,5.)
    y = x[::-1]
    X,Y = np.meshgrid(x,y)
    original = _field(X,Y)
    out,idx = pad_array(original,padw=(3,5),ptype="cells",
                        mode="min_curvature",tension=0.25)
    assert out.shape == (27,31)
    assert idx == (3,24,5,26)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[3:24,5:26],original)
    assert np.max(np.abs(out))<100


# -----------------------------------------------------------------------------
def test_pad_array_curvature_georeferenced_anisotropic_xy():
    x=452904.697640+np.arange(12)*400.
    y=6426297.029215-np.arange(10)*800.
    X,Y=np.meshgrid(x,y)
    original = _field((X-x[0])/40,(y[0]-Y)/80)
    original[3:6,5:8] = np.nan
    result,idx=pad_array(original,padw=3,ptype="cells",
                         mode="min_curvature",xy=(x,y),
                         block=None, anchor=0.05)
    assert result.shape==(16,18)
    assert idx == (3,13,3,15)
    assert np.isfinite(result).all()
    finite=np.isfinite(original)
    core=result[3:13,3:15]
    np.testing.assert_array_equal(core[finite],original[finite])


# -----------------------------------------------------------------------------
def test_pad_array_curvature_block_median_and_old_edge():
    x = np.linspace(0,50,11)
    X,Y=np.meshgrid(x,x)
    a=_field(X,Y)
    a[3:5,6:8]=np.nan
    out,idx=pad_array(
        a,padw=2,ptype="cells",mode="robust",
        block="median",block_size=2.,anchor=.1)
    assert out.shape==(15,15) and np.isfinite(out).all()
    finite=np.isfinite(a)
    np.testing.assert_array_equal(out[2:13,2:13][finite],a[finite])
    # Regression: the original default edge mode remains unchanged.
    b,oldidx=pad_array(np.arange(12).reshape(3,4),
                       padw=1,ptype="cells")
    np.testing.assert_array_equal(b,np.pad(np.arange(12).reshape(3,4),1,
                                           mode="edge"))
    assert oldidx==(1,4,1,5)


# -----------------------------------------------------------------------------
def test_fillnan_curvature_rejects_nonrectilinear_xy():
    y,x=np.mgrid[0:11,0:11]
    a=np.cos(x+y)
    a[3:5,3:5]=np.nan
    xx = x.astype(float) + 0.1*y  # rotated/sheared grid
    with pytest.raises(ValueError,match="rectilinear"):
        fillnan(a,xy=(xx,y.astype(float)),method="min_curvature")


# -----------------------------------------------------------------------------
def test_fillnan_no_missing_values_remain_unchanged():
    original=np.arange(16).reshape(4,4).astype(float)
    filled=fillnan(original,method="min_curvature")
    np.testing.assert_array_equal(filled,original)
