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

if __name__=="__main__":
    run()
