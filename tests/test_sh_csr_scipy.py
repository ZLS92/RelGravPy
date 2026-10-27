"""Numerical comparison against independent SciPy Legendre functions.

The coefficients are a small, source-attributed CSR RL05 GRACE sample in
tests/data, converted from GRCOF2 to static ICGEM gfc form.
SciPy is required by CI tests only, NOT by the RelGravPy SH runtime.
The surface='sphere' test deliberately does not claim ellipsoidal ICGEM accuracy.
"""
import importlib.util
from math import factorial, sqrt
from pathlib import Path

import numpy as np
import pytest
from scipy.special import lpmv

from relgravpy import shm_tools as sh
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "csr_rl05_grace_degree6.gfc"

def independent_fully_normalized_legendre(n, m, latitude):
    """Unnormalized SciPy associated Legendre -> geodetic 4pi normalization."""
    x = np.sin(np.deg2rad(latitude))
    factor = sqrt((2*n+1)*(2 if m else 1)*factorial(n-m)/factorial(n+m))
    return factor*(-1)**m*lpmv(m, n, x)


def reference_sh_synthesis(model, lat, lon, height):
    """Direct degree/order sum using SciPy, not RelGravPy's Legendre recursion."""
    gm = model["gm"]
    rad0 = model["radius"]
    r = rad0 + height
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f*(2-f)
    gm_normal = 3.986004418e14
    j2 = 1.08262982131e-3
    accum = np.zeros(4, dtype=float)
    for n in range(7):
        for m in range(n+1):
            c = model["c"][n,m]
            s = model["s"][n,m]
            if n == 0 and m == 0:
                c -= gm_normal/gm
            if n > 0 and n % 2 == 0 and m == 0:
                k = n//2
                j2k = (-1)**(k+1) * (3*e2**k)/((2*k+1)*(2*k+3))*(
                    1-k + 5*k*j2/e2)
                c -= (-j2k/sqrt(2*n+1)) * (gm_normal/gm)*(a/rad0)**n
            P = independent_fully_normalized_legendre(n,m,lat)
            arg = np.deg2rad(m*lon)
            harmonic = P*(c*np.cos(arg)+s*np.sin(arg))*(rad0/r)**n
            accum += harmonic*np.array([1,n+1,(n-1),-(n+1)*(n+2)])
    return np.array([
        gm/r*accum[0],
        gm/r**2*accum[1]*1e5,
        gm/r**2*accum[2]*1e5,
        gm/r**3*accum[3]*1e9,
    ])


@pytest.mark.parametrize("height", [0., 1000.])
def test_csr_l6_spherical_fields_vs_scipy(height):
    model=sh.sh_read_gfc(DATA)
    assert model["lmax"]==6
    assert len([1 for n in range(7) for m in range(n+1)]) == 28
    assert model["c"][2,0]==pytest.approx(-0.484169060233e-3, rel=1e-12)
    limits=[-90.,90.,-60.,60.]
    fields=("potential","grav_dist","free_air","grav_gradient")
    result=sh.sh_synthesis_core(model,limits,step=(45,30),
                                lmax=6,fields=fields,height=height,
                                surface="sphere",height_reference="ellipsoid")
    keys=("potential","grav_dist","free_air","grav_gradient")
    deltas=[]
    for i,j in np.ndindex(result["lat"].shape):
        exp=reference_sh_synthesis(model, result["lat"][i,j],
                                       result["lon"][i,j], height)
        got=np.array([result[k][i,j] for k in keys])
        deltas.append(got-exp)
    deltas=np.array(deltas)
    max_abs=np.max(np.abs(deltas),axis=0)
    print("CSR degree-6 independent SciPy residuals at height",height)
    for key,err in zip(keys,max_abs):
        print(f"  {key} max |RelGravPy - SciPy| = {err:.12g}")
    assert max_abs[0] < 1e-7          # m2/s2
    assert max_abs[1] < 1e-8          # mGal
    assert max_abs[2] < 1e-8          # mGal
    assert max_abs[3] < 1e-8          # Eotvos


def test_requested_degree_above_available_warns():
    with pytest.warns(UserWarning,match="exceeds"):
        model=sh.sh_read_gfc(DATA,lmax=100)
    assert model["lmax"]==6
