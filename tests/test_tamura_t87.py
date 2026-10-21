"""Regression and physical consistency tests for the dependency-free T87 tide."""
from datetime import datetime, timezone, timedelta
import numpy as np
import pytest
from relgravpy import grav_processing as gp
from relgravpy.earth_tides_t87 import _catalogue, _legendre, _terrestrial_minus_utc


def test_catalogue_complete():
    args, degree, coeffs = _catalogue()
    assert args.shape == (1200, 11)
    assert degree.shape == (1200,)
    assert coeffs.shape == (1200, 5)
    assert set(degree).issubset(set(range(2, 7)))
    assert np.all(np.isfinite(coeffs))


def test_legendre_grs80_normalization():
    theta = np.deg2rad(30)
    assert _legendre(2,0,theta) == pytest.approx(1.39754248593737, abs=1e-12)
    assert _legendre(2,1,theta) == pytest.approx(1.67705098312484, abs=1e-12)
    assert _legendre(2,2,theta) == pytest.approx(0.48412291827593, abs=1e-12)


def test_utc_leap_seconds():
    assert _terrestrial_minus_utc(np.array(["2016-12-31", "2017-01-01"],
                                            dtype="datetime64[ns]")).tolist() == [68.184, 69.184]


def test_single_and_range():
    at = "2026-10-21T00:00:00"
    one = gp.earth_tides_tsoft(45.5,10.2,150,DateTime=at)
    times, result = gp.earth_tides_tsoft(
        45.5,10.2,150,start=at,end="2026-10-21T23:00:00",step=3600)
    assert len(times) == result.size == 24
    assert result[0] == pytest.approx(one, abs=1e-10)
    assert np.all(np.isfinite(result))
    assert np.ptp(result) > 0.001
    assert np.max(np.abs(result)) < 1.0  # mGal; physical tide scale


def test_units_and_correction():
    kwargs = dict(lat=45.5,lon=10.2,z=150,DateTime="2026-10-21T12:00:00")
    nm = gp.earth_tides_tsoft(**kwargs,unit="nm/s2")
    micro = gp.earth_tides_tsoft(**kwargs,unit="uGal")
    milli = gp.earth_tides_tsoft(**kwargs,unit="mGal")
    assert nm == pytest.approx(10*micro,rel=1e-12)
    assert micro == pytest.approx(1000*milli,rel=1e-12)
    assert gp.earth_tides_tsoft(**kwargs,correction=True)==pytest.approx(-milli)


def test_wavegroup_identity_and_factor():
    kwargs = dict(lat=45.5,lon=10.2,z=150,DateTime="2026-10-21T12:00:00")
    ungrouped=gp.earth_tides_tsoft(**kwargs,delta=1.16)
    grouped=gp.earth_tides_tsoft(**kwargs,wavegroups=[[0.,10.,1.16,0.]])
    assert grouped == pytest.approx(ungrouped,abs=1e-10)
    rigid=gp.earth_tides_tsoft(**kwargs,delta=1.0)
    assert ungrouped == pytest.approx(1.16*rigid,abs=1e-10)


def test_utc_timezone_and_vectorization():
    utc=datetime(2026,10,21,10,tzinfo=timezone.utc)
    local=datetime(2026,10,21,12,tzinfo=timezone(timedelta(hours=2)))
    a=gp.earth_tides_tsoft(45.5,10.2,150,DateTime=utc)
    b=gp.earth_tides_tsoft(45.5,10.2,150,DateTime=local)
    assert a==pytest.approx(b,abs=1e-12)
    many=gp.earth_tides_tsoft([45.5,46.0],[10.2,11.0],150,DateTime=[utc,utc])
    assert many.shape==(2,)
    assert many[0] == pytest.approx(a,abs=1e-12)


@pytest.mark.parametrize("kwargs",[
    {"DateTime":"1971-12-31"},
    {"DateTime":"2026-10-21","delta":0},
    {"DateTime":"2026-10-21","unit":"gal"},
    {"DateTime":"2026-10-21","wavegroups":[[0,2,1,0],[1,3,1,0]]},
])
def test_reject_invalid(kwargs):
    with pytest.raises(ValueError):
        gp.earth_tides_tsoft(45.5,10.2,0,**kwargs)
