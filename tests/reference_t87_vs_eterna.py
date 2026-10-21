"""Reference-only numerical check against ETERNA/PyGTide T87.

This module is NOT imported by RelGravPy and the optional PyGTide
installation is restricted to a separate CI validation environment.
"""
import numpy as np
from relgravpy.earth_tides_t87 import earth_tides_tsoft

def run():
    import pygtide
    lat,lon,height=49.00937,8.40444,120.
    pt=pygtide.pygtide(msg=False)
    pt.set_wavegroup(np.array([[0.0,10.0,1.0,0.0]]))
    pt.predict(lat,lon,height,"2017-01-01",48,3600,
               tidalcompo=0,tidalpoten=4,poltidecor=0,lodtidecor=0)
    df=pt.results()
    print("Reference columns:",list(df.columns),flush=True)
    times=df["UTC"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    print("All ETERNA output sample:",df.head().to_string(index=False),flush=True)
    ref=df["Signal [nm/s**2]"].to_numpy(dtype=float)
    if not np.any(np.abs(ref)>1):
        raise RuntimeError("Reference engine returned zero tide")
    got=earth_tides_tsoft(lat,lon,height,DateTime=times,
                          unit="nm/s2",delta=1.0)
    d=got-ref
    print("First five rows (time, ETERNA, RelGravPy, difference):",flush=True)
    for t,a,b,c in zip(times[:5],ref[:5],got[:5],d[:5]):
        print(str(t),a,b,c,flush=True)
    print("reference RMSE [nm/s2]:",np.sqrt(np.mean(d*d)),flush=True)
    print("reference max abs [nm/s2]:",np.max(np.abs(d)),flush=True)
    print("reference bias [nm/s2]:",np.mean(d),flush=True)
    print("reference Pearson r:",np.corrcoef(got,ref)[0,1],flush=True)
    # Independent ETERNA reference numerical acceptance criterion (0.01 uGal).
    assert np.sqrt(np.mean(d*d)) < 0.10, (
        "T87 reference mismatch exceeds 0.1 nm/s2 (0.01 uGal)"
    )
    # Repeat with the published TSoft manual's WDD wave-group example.
    from relgravpy.earth_tides_t87 import (
        _catalogue, earth_tides_tsoft as predict_standalone,
    )
    # Reproduce the internal preset through one call, then hand the same
    # frequency bands and parameters to ETERNA as independent engine.
    groups=np.asarray([
        [0.000000,0.002427,1.00000,0.],
        [0.002428,0.249951,1.15787,0.],
        [0.721500,0.906315,1.15432,0.],
        [0.921941,0.940487,1.15430,0.],
        [0.958085,0.998028,1.14908,0.],
        [0.999853,1.000147,1.13448,0.],
        [1.005329,1.005623,1.27258,0.],
        [1.007595,1.011099,1.17072,0.],
        [1.013689,1.216397,1.15636,0.],
        [1.719381,2.182843,1.16200,0.],
        [2.753244,3.381478,1.07360,0.],
        [3.381379,4.347615,1.03900,0.],
    ])
    pt.set_wavegroup(groups)
    pt.predict(lat,lon,height,"2017-01-01",48,3600,
               tidalcompo=0,tidalpoten=4,poltidecor=0,lodtidecor=0)
    ref_wdd=pt.results()["Signal [nm/s**2]"].to_numpy(float)
    got_wdd=predict_standalone(lat,lon,height,DateTime=times,
                               wavegroups="tsoft_wdd",unit="nm/s2")
    d2=got_wdd-ref_wdd
    print("TSoft WDD preset RMSE [nm/s2]:",
          np.sqrt(np.mean(d2*d2)),flush=True)
    print("TSoft WDD preset max [nm/s2]:",
          np.max(np.abs(d2)),flush=True)
    assert np.sqrt(np.mean(d2*d2)) < 0.15, (
        "WDD preset fails independent ETERNA T87 numerical reference"
    )


if __name__=="__main__":
    run()
