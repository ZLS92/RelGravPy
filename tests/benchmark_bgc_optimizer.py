"""Fast auto-mesh decision benchmark on synthetic mountain and marine patches.

Only evaluates the optimizer's preview/error-proxy selection; does NOT
certify actual gravity accuracy. No GDAL and no full gravity forward run.
Usage: PYTHONPATH=tests:. python -u tests/benchmark_bgc_optimizer.py
"""
import json
import time
import warnings
import numpy as np

from test_bouguer_polyhedra import BC


def source_features(kind, scale):
    x=np.linspace(-2.,2.,33)
    y=np.linspace(-2.,2.,25)
    xf,yf=np.meshgrid(x,x)
    xb,yb=np.meshgrid(y,y)
    if kind=="mountain":
        fine=(1200+350*np.sin(scale*xf)*np.cos(3*yf)
              +170*np.exp(-((xf-.3)**2+(yf+.2)**2)/.04))
        broad=(1200+600*np.sin(3*xb)*np.cos(2*yb)
               +100*xb)
    else:
        # Land/ocean break, shallow shelf and underwater canyon.
        fine=(280*np.tanh(4*xf)-420
              -170*np.exp(-((yf+.4)**2)/.06)
              +60*np.sin(scale*yf)*np.cos(2*xf))
        broad=(300*np.tanh(2*xb)-450
               -350*np.exp(-((yb+.5)**2)/.15)
               +90*np.sin(scale*yb/2)*np.cos(xb))
    return BC._bgc_terrain_features((fine,broad),25.,200.)


def main():
    for terrain in ("mountain","coast_bathymetry"):
        feature=np.vstack([
            source_features("mountain" if terrain=="mountain" else "coast",s)
            for s in (2.,3.,5.,8.)
        ])
        x=np.arange(len(feature),dtype=float)
        y=np.zeros(len(feature))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            start=time.perf_counter()
            result=BC.optimize_bgc_mesh(
                x,y,None,gs1=None,R1=None,gs2=None,R2=167000.,
                target_error=.05,terrain_features=feature,
                raster_resolution=25.)
            elapsed=time.perf_counter()-start
        per_station=BC._bgc_mesh_error_proxy(
            feature,result["gs1"],result["R1"],result["gs2"],
            2670.,6371000.)
        report=dict(
            terrain=terrain,number_of_stations=len(feature),
            gs1_m=result["gs1"],R1_m=result["R1"],
            gs2_m=result["gs2"],R2_m=result["R2"],
            max_estimated_error_mgal=result["estimated_error_mgal"],
            worst_station=result["critical_station_index"],
            all_estimates_mgal=[float(x) for x in per_station],
            meets_target_estimate=result["meets_target"],
            candidate_count=result["candidates"],
            elapsed_s=elapsed,
            warnings=[str(item.message) for item in captured],
            caveat="morphology_proxy_not_certified"
        )
        print("BG_OPTIMIZER_CASE",json.dumps(report),flush=True)
        assert result["R2"] >= 167000.
        assert result["candidates"] <= 80
        assert np.isclose(result["estimated_error_mgal"],
                          max(per_station))
        assert elapsed<1.0

if __name__=="__main__":
    main()
