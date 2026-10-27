"""Regression tests for the single fast spherical-harmonic synthesis engine.

The optional Numba path and always-available NumPy fallback must implement
the same degree/order recurrence, output units, and geographic geometry.
"""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from relgravpy import _sh_fast
from relgravpy import shm_tools as sh


def _model(lmax=8):
    """Make deterministic, realistic-size perturbation coefficients."""
    rng = np.random.default_rng(582)
    c = rng.normal(0, 2e-7, (lmax+1,lmax+1))
    s = rng.normal(0, 2e-7, (lmax+1,lmax+1))
    c = np.tril(c)
    s = np.tril(s)
    gm, radius = sh.GM_WGS84, sh.A_WGS84
    c[0,0] = 1.0
    s[:,0] = 0.0
    normal = sh.sh_wgs84_normal_zonals(gm,radius,lmax)
    for n in range(2,lmax+1,2):
        c[n,0] += normal[n]
    return dict(c=c,s=s,gm=gm,radius=radius,lmax=lmax)


@pytest.mark.parametrize("surface", ["sphere", "ellipsoid", "topography"])
def test_numba_and_numpy_agree_on_irregular_coordinates(monkeypatch,surface):
    """Check physical fields with differing longitude, latitude, and heights."""
    model = _model(8)
    lon = np.array([[5., 9., 14.], [16., 23., 29.]])
    lat = np.array([[30., 40., 50.], [31., 43., 55.]])
    topo = np.array([[50., 80., 120.],[250., 20., 30.]])
    fields = ("potential","geoid","grav_dist","free_air","abs_grav",
              "grav_gradient")
    opts = dict(lon=lon,lat=lat,limits=None,fields=fields,
                height=np.array([[0.,100.,250.],[50.,120.,25.]]),
                surface=surface,topo=topo,
                height_reference="ellipsoid",lmax=8)
    auto = sh.sh_synthesis_core(model,**opts)
    monkeypatch.setattr(_sh_fast,"_compiled_accumulator",None)
    numpy_only = sh.sh_synthesis_core(model,**opts)
    for field in fields:
        np.testing.assert_allclose(
            auto[field],numpy_only[field],rtol=3e-12,atol=1e-8)


def test_distinct_longitudes_on_each_row_are_used():
    """Catch the former reuse of first-row trig factors on a custom mesh."""
    model = _model(5)
    lon = np.array([[0., 30.], [70., 120.]])
    lat = np.array([[20., 20.], [40., 40.]])
    full = sh.sh_synthesis_core(
        model,lon=lon,lat=lat,fields=("grav_dist",),
        height_reference="ellipsoid",surface="ellipsoid",lmax=5)
    for ij in np.ndindex(lon.shape):
        single = sh.sh_synthesis_core(
            model,lon=np.array([lon[ij]]),lat=np.array([lat[ij]]),
            fields=("grav_dist",),height_reference="ellipsoid",lmax=5)
        np.testing.assert_allclose(full["grav_dist"][ij],
                                   single["grav_dist"][0],atol=1e-9)


def test_geoid_datum_internal_and_explicit_undulation_agree():
    """An implicit height-to-ellipsoid conversion must match explicit N."""
    model = _model(6)
    lon,lat = sh.sh_grid([8.,10.,44.,46.],1.)
    N = sh.sh_synthesis_core(
        model,lon=lon,lat=lat,fields=("geoid",),
        height_reference="ellipsoid")["geoid"]
    kwargs = dict(lon=lon,lat=lat,height=120.,
                  fields=("grav_dist","abs_grav"),surface="ellipsoid")
    automatic = sh.sh_synthesis_core(model,**kwargs)
    explicit = sh.sh_synthesis_core(
        model,**kwargs,geoid_undulation=N)
    for field in ("grav_dist","abs_grav"):
        np.testing.assert_allclose(automatic[field],explicit[field],
                                   rtol=1e-13,atol=1e-8)


def test_synthesis_wrapper_with_xyz_point_coordinates():
    """Public API must preserve the x/y/z point-array calling convention."""
    model = _model(5)
    lon = np.array([10.,12.,13.])
    lat = np.array([42.,44.,46.])
    z = np.array([0.,100.,250.])
    out = sh.sh_synthesis(
        model,x=lon,y=lat,z=z,fields=("grav_dist",),
        height_reference="ellipsoid",lmax=5)
    expected = sh.sh_synthesis_core(
        model,lon=lon,lat=lat,height=z,fields=("grav_dist",),
        height_reference="ellipsoid",lmax=5)
    np.testing.assert_allclose(out["grav_dist"],expected["grav_dist"])
    assert out["grav_dist"].shape == (3,)


def test_missing_numba_does_not_block_import_or_synthesis():
    """A fresh Python process must work without importable Numba."""
    script = (
        "import sys; sys.modules['numba'] = None; "
        "from relgravpy import _sh_fast, shm_tools as sh; "
        "import numpy as np; "
        "assert _sh_fast._compiled_accumulator is None; "
        "c=np.array([[1.0]]); "
        "model=dict(c=c,s=np.zeros_like(c),gm=sh.GM_WGS84,"
        "radius=sh.A_WGS84,lmax=0); "
        "result=sh.sh_synthesis_core(model,lon=np.array([0.]),"
        "lat=np.array([0.]),fields=('grav_dist',),"
        "height_reference='ellipsoid'); "
        "assert np.isfinite(result['grav_dist']).all()"
    )
    completed = subprocess.run(
        [sys.executable,"-c",script],text=True,capture_output=True,
        timeout=40,check=False)
    assert completed.returncode == 0, completed.stderr


def test_failed_jit_compilation_falls_back(monkeypatch):
    """Numba compilation failures are warnings, not fatal runtime errors."""
    def broken(*args):
        raise RuntimeError("simulated incompatible numba installation")
    monkeypatch.setattr(_sh_fast,"_compiled_accumulator",broken)
    monkeypatch.setattr(_sh_fast,"_numba_failed",False)
    with pytest.warns(RuntimeWarning,match="using NumPy"):
        result = sh.sh_synthesis_core(
            _model(3),lon=np.array([0.,10.]),lat=np.array([0.,20.]),
            fields=("grav_dist",),height_reference="ellipsoid")
    assert np.isfinite(result["grav_dist"]).all()
    assert _sh_fast._numba_failed
