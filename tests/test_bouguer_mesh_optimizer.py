"""NumPy-only tests for the lightweight survey-wide mesh tuner.

The estimator is a morphology proxy; these tests do NOT assert that
the true discretisation error is rigorously bounded by target_error.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from test_bouguer_polyhedra import BC

R = 6371000.


def stations(n):
    return np.arange(n, dtype=float), np.zeros(n, dtype=float)


def terrain_examples():
    gx=np.linspace(-2,2,33)
    xx,yy=np.meshgrid(gx,gx)
    ux=np.linspace(-2,2,25)
    bx,by=np.meshgrid(ux,ux)
    # Mountain with asymmetric sharp high-frequency ridges.
    fine=1200 + 400*np.sin(4*xx)*np.cos(3*yy) + 160*np.exp(
        -((xx-.4)**2+(yy+.3)**2)/.02)
    broad=1200+500*np.sin(3*bx)*np.cos(2*by)+250*bx
    # Coast / bathymetry: land relief, shelf and submarine canyon.
    sea_fine=100*np.tanh(3*xx)-350+250*np.exp(-((yy-.5)**2)/.04)
    sea_broad=250*np.tanh(2*bx)-400-200*np.exp(-((by-.8)**2)/.05)
    return (fine,broad),(sea_fine,sea_broad)


def test_terrain_features_mountain_and_bathymetry_nonnegative():
    for fine,broad in terrain_examples():
        feat=BC._bgc_terrain_features((fine,broad),25.,200.)
        assert feat.shape==(4,)
        assert np.all(np.isfinite(feat))
        assert np.all(feat>=0)
        assert feat[2]>100.


def test_bgc_optimizer_manual_parameters_bypass_target_entirely():
    x,y=stations(3)
    with _no_warnings():
        result=BC.optimize_bgc_mesh(
            x,y,None,gs1=25.,R1=5000.,gs2=500.,
            target_error=-999, R2=167000.)
    assert result["optimized"] is False
    assert result["R2"]==167000.
    assert result["estimated_error_mgal"] is None


class _no_warnings:
    def __enter__(self):
        import warnings
        self.context=warnings.catch_warnings(record=True)
        self.captured=self.context.__enter__()
        warnings.simplefilter("always")
    def __exit__(self,*args):
        self.context.__exit__(*args)
        assert not self.captured


def test_optimizer_one_shared_mesh_and_maximum_all_stations():
    x,y=stations(4)
    feat=np.array([[.08,1.,80.,200.],
                   [.15,3.,200.,500.],
                   [1.20,20.,1500.,1800.],
                   [.35,7.,400.,700.]])
    # Loose tolerance gives a feasible candidate and makes the cost
    # criterion decisive. Check the maximum, NOT the average.
    result=BC.optimize_bgc_mesh(
        x,y,None,terrain_features=feat,raster_resolution=25.,
        gs1=None,R1=None,gs2=None,target_error=20.)
    expected=BC._bgc_mesh_error_proxy(
        feat,result["gs1"],result["R1"],result["gs2"],
        2670.,R,native_resolution=25.)
    assert result["optimized"] is True
    assert result["meets_target"] is True
    assert result["critical_station_index"]==int(np.argmax(expected))
    assert result["estimated_error_mgal"]==pytest.approx(max(expected))
    assert result["candidates"]<=80
    assert result["R2"]>=167000.
    assert result["gs1"]>=25.
    assert result["gs2"]>=result["gs1"]


def test_optimizer_tight_unattainable_target_warns_and_picks_minimum_error():
    x,y=stations(3)
    feat=np.array([[.1,1.,100.,200.],[1.8,40.,1800.,2000.],
                   [.2,2.,300.,400.]])
    with pytest.warns(RuntimeWarning,match="No mesh candidate reached"):
        strict=BC.optimize_bgc_mesh(
            x,y,None,terrain_features=feat,raster_resolution=25.,
            target_error=1e-15)
    with _no_warnings():
        relaxed=BC.optimize_bgc_mesh(
            x,y,None,terrain_features=feat,raster_resolution=25.,
            target_error=20.)
    assert strict["meets_target"] is False
    assert strict["estimated_error_mgal"]<=relaxed["estimated_error_mgal"]+1e-12
    assert strict["critical_station_index"]==1
    assert strict["R2"]>=167000.


def test_auto_parameter_subsets_keep_explicit_near_step():
    x,y=stations(2)
    feat=np.array([[.2,2.,200.,200.],[.8,10.,600.,800.]])
    result=BC.optimize_bgc_mesh(
        x,y,None,gs1=30.,R1=None,gs2=500.,
        terrain_features=feat,raster_resolution=10.,target_error=20.)
    assert result["gs1"]==30.
    assert abs(result["gs2"]/500.-1)<=.05000001
    assert result["R1"]>0


def test_default_bgc_radius_and_tuning_trigger_are_explicit():
    src=(Path(__file__).resolve().parents[1]/"relgravpy"/
         "_bouguer_correction.py").read_text()
    tree=ast.parse(src)
    fn=next(n for n in tree.body
            if isinstance(n,ast.FunctionDef) and n.name=="bgc")
    args={arg.arg:default for arg,default in zip(
        fn.args.args[-len(fn.args.defaults):],fn.args.defaults)}
    assert isinstance(args["R2"],ast.Constant)
    assert args["R2"].value==167000.
    assert args["target_error"].value==0.05
    snippet=ast.get_source_segment(src,fn)
    assert "if gs1 is None or R1 is None or gs2 is None:" in snippet
    assert "optimize_bgc_mesh(" in snippet
    assert "GrdStep2( R1" not in snippet
    assert "Radi2( R1" not in snippet


def test_invalid_tolerance_rejected_only_when_tuning():
    x,y=stations(2)
    feat=np.ones((2,4))
    with pytest.raises(ValueError,match="target_error"):
        BC.optimize_bgc_mesh(x,y,None,terrain_features=feat,
                             raster_resolution=25.,target_error=-1.)
    out=BC.optimize_bgc_mesh(x,y,None,gs1=30.,R1=5000.,gs2=400.,
                             target_error=-1.)
    assert out["optimized"] is False



def test_station_preview_reads_only_two_small_windows_per_station(monkeypatch):
    """No full gravity loop, no country-scale rewarping, no station subsample."""
    from types import SimpleNamespace
    class FakeBand:
        def __init__(self, array):
            self.array=array
            self.calls=0
        def GetNoDataValue(self):
            return None
        def ReadAsArray(self,xoff,yoff,win_xsize,win_ysize,
                        buf_xsize,buf_ysize):
            self.calls+=1
            assert win_xsize<=257 and win_ysize<=257
            a=self.array[yoff:yoff+win_ysize,xoff:xoff+win_xsize]
            ix=np.linspace(0,len(a[0])-1,buf_xsize).astype(int)
            iy=np.linspace(0,len(a)-1,buf_ysize).astype(int)
            return a[np.ix_(iy,ix)]
    class FakeDtm:
        def __init__(self,array):
            self.band=FakeBand(array)
            self.RasterXSize=array.shape[1]
            self.RasterYSize=array.shape[0]
        def GetRasterBand(self,i):
            assert i==1
            return self.band
        def GetGeoTransform(self):
            return (0.,30.,0.,0.,0.,-30.)
        def GetProjection(self):
            return "EPSG:32633"
    yy,xx=np.indices((520,520))
    mountain=1000.+350.*np.sin(xx/20)*np.cos(yy/17)
    dtm=FakeDtm(mountain)
    monkeypatch.setattr(BC,"utl",SimpleNamespace(
        prjxy=lambda src,dst,x,y:(np.asarray(x),np.asarray(y))),
        raising=False)
    x=np.array([3900.,7800.,10200.])
    y=np.array([-3300.,-6600.,-9300.])
    features,resolution=BC._bgc_station_terrain_features(
        dtm,x,y,"EPSG:32633","EPSG:32633")
    assert features.shape==(3,4)
    assert resolution==pytest.approx(30.)
    assert dtm.band.calls==2*len(x)
    assert np.all(np.isfinite(features))
    assert np.all(features[:,2]>0)



def test_rugged_terrain_prevents_unverified_coarse_near_mesh():
    """Real full-DEM benchmark showed gs1=60 m misses 0.05 mGal."""
    x,y=stations(2)
    features=np.array([[.883,.902,871.6,1376.8],
                       [.970,.898,958.6,1587.5]])
    selected=BC.optimize_bgc_mesh(
        x,y,None,gs1=None,R1=None,gs2=None,
        raster_resolution=30.,terrain_features=features,
        target_error=.05)
    assert selected["gs1"]<=30.,selected
    assert selected["meets_target"] is True
    assert selected["estimated_error_mgal"]<=.05
