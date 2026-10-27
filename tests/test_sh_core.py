"""Reference regression tests for the standalone SH numerical building blocks."""
import numpy as np
import pytest
# Exercise the actual public synthesis module, rather than the retired
# sh_core.py path. The import must succeed even with Numba absent.
from relgravpy import shm_tools as sh_core


def test_legendre_degree_two_at_equator():
    p = sh_core.sh_legendre(0.0, 2)
    assert p[0,0] == pytest.approx(1.0)
    assert p[1,0] == pytest.approx(0.0, abs=1e-14)
    assert p[1,1] == pytest.approx(np.sqrt(3.0))
    assert p[2,0] == pytest.approx(-np.sqrt(5.0)/2.0)
    assert p[2,2] == pytest.approx(np.sqrt(15.0)/2.0)


def test_gentle_taper():
    w = sh_core.sh_taper(10, 6)
    np.testing.assert_allclose(w[:7], 1.0)
    assert w[10] == pytest.approx(0.0)
    assert np.all(np.diff(w[6:]) <= 0)


def test_linear_scalar_analysis_recovers_degree_one():
    lon, lat = sh_core.sh_grid([-180, 150, -75, 75], 30)
    p10 = np.sqrt(3.) * np.sin(np.deg2rad(lat))
    p11 = np.sqrt(3.) * np.cos(np.deg2rad(lat))
    values = 5. + 2.*p10 + 3.*p11*np.cos(np.deg2rad(lon))
    result = sh_core.sh_analysis_core(lon, lat, values, lmax=1)
    assert result["rank"] == 4
    assert result["c"][0,0] == pytest.approx(5., abs=1e-10)
    assert result["c"][1,0] == pytest.approx(2., abs=1e-10)
    assert result["c"][1,1] == pytest.approx(3., abs=1e-10)


def test_quadrupole_radial_field():
    # Pure perturbation of normal quadrupole, checked against analytical
    # spherical derivatives; the reference C00 and even zonals are removed.
    gm=sh_core.GM_WGS84
    radius=sh_core.A_WGS84
    model = {
        "c": np.zeros((3,3)), "s": np.zeros((3,3)),
        "gm": gm, "radius": radius, "lmax": 2
    }
    model["c"][0,0] = 1.0
    model["c"][2,0] = sh_core.sh_wgs84_normal_zonals(gm,radius,2)[2]+1e-6
    result = sh_core.sh_synthesis_core(
        model, [-5,5,0,0], step=5, lmax=2,
        fields=("potential","grav_dist","grav_gradient"), surface="sphere",
        height_reference="ellipsoid"
    )
    expected_p20=-np.sqrt(5.)/2.
    expected_potential=gm/radius*1e-6*expected_p20
    np.testing.assert_allclose(result["potential"], expected_potential)
    np.testing.assert_allclose(
        result["grav_dist"], expected_potential*3/radius*1e5)
    np.testing.assert_allclose(
        result["grav_gradient"],
        -expected_potential*12/radius**2*1e9)


def test_wgs84_c20_reference():
    # Published fully normalized WGS84 second zonal coefficient.
    c20=sh_core.sh_wgs84_normal_zonals(
        sh_core.GM_WGS84, sh_core.A_WGS84, 2)[2]
    assert c20 == pytest.approx(-0.484166774985e-3, abs=5e-15)


def test_global_fft_analysis_low_degrees():
    # Independent closed-form l=0,1 reference field, on a 1-degree global grid.
    lon=np.arange(-180.,180.,1.)
    lat=np.linspace(-90.,90.,181)
    xx,yy=np.meshgrid(lon,lat)
    phi=np.deg2rad(yy)
    lam=np.deg2rad(xx)
    values=(5.+2.*np.sqrt(3.)*np.sin(phi)+
            3.*np.sqrt(3.)*np.cos(phi)*np.cos(lam))
    result=sh_core.sh_analysis_fft(xx,yy,values,lmax=1)
    assert result["c"][0,0] == pytest.approx(5.,abs=1e-10)
    assert result["c"][1,0] == pytest.approx(2.,abs=3e-4)
    assert result["c"][1,1] == pytest.approx(3.,abs=3e-4)
    assert result["s"][1,1] == pytest.approx(0.,abs=1e-10)


def test_legendre_degree_3_analytic_equator():
    p=sh_core.sh_legendre(0,3)
    # No Condon-Shortley phase, 4-pi geodetic normalization.
    assert p[3,0] == pytest.approx(0.,abs=1e-14)
    assert p[3,1] == pytest.approx(-np.sqrt(42.)/4,abs=1e-14)
