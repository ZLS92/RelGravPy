"""Reproducible comparison of existing drift methods and overlap registration.

Read only two selected functions as AST nodes, to avoid unrelated heavy
geospatial imports when running the drift comparison in minimal CI.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy as sp
from relgravpy.drift_alignment import grav_drift_overlap


def _previous_methods():
    path = Path(__file__).resolve().parents[1] / "relgravpy/grav_processing.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    methods = [n for n in module.body if isinstance(n, ast.FunctionDef)
               and n.name in ("grav_drift_poly", "grav_drift_slope")]
    assert len(methods) == 2
    isolated = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = {"np": np, "utl": SimpleNamespace(sp=sp),
                 "__name__": "relgravpy.grav_processing",
                 "__package__": "relgravpy"}
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace["grav_drift_poly"], namespace["grav_drift_slope"]


def dispersion(g, st):
    valid = np.array([np.count_nonzero(st == s) >= 2 for s in st])
    tmp = g[valid]
    labels = st[valid]
    centered = np.array([
        v - np.mean(tmp[labels == s]) for v, s in zip(tmp, labels)
    ])
    return float(np.sqrt(np.mean(centered**2)))


old, slope = _previous_methods()
# Nonlinear synthetic with 3 repeatedly measured stations + one singleton.
times = np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.])
st = np.array(["A","B","C","A","B","C","Q","A","B","C"])
baseline = np.array([{"A":1000.,"B":1020.,"C":970.,"Q":1900.}[s] for s in st])
true = 0.015 * times + 0.0004 * times**2
g = baseline + true
_, da = grav_drift_overlap(st,g,times*3600,deg=2)
_, d_public = old(st,g,datetime=times*3600,
                  DateTimeFormat="seconds",deg=2,alignment="overlap")
np.testing.assert_allclose(da, d_public, atol=1e-12)
_, db = old(st,g,datetime=times*3600,DateTimeFormat="seconds",deg=2)
_, dc = slope(st,g,datetime=times*3600,DateTimeFormat="seconds",deg=1)
print("Quadratic synthetic (no noise, known true drift):")
for label,d in [("overlap",da),("sequential",db),("slope",dc)]:
    print(label,"RMSE (mGal):",round(float(np.sqrt(np.mean((d-true)**2))),8))

# Same synthetic 100 noise replicates, identical random noise for each method.
rng=np.random.default_rng(20261022)
stats={"overlap":[],"sequential":[],"slope":[]}
for _ in range(100):
    noisy=baseline+true+rng.normal(scale=0.001,size=len(g))
    for label, method in (
        ("overlap",lambda: grav_drift_overlap(st,noisy,times*3600,deg=2)[1]),
        ("sequential",lambda: old(st,noisy,datetime=times*3600,
                                  DateTimeFormat="seconds",deg=2)[1]),
        ("slope",lambda: slope(st,noisy,datetime=times*3600,
                              DateTimeFormat="seconds",deg=1)[1]),
    ):
        pred=method()
        stats[label].append(float(np.sqrt(np.mean((pred-true)**2))))
print("Quadratic synthetic with 0.001 mGal Gaussian observation noise, 100 seeds:")
for label, val in stats.items():
    print(label,"mean RMSE (mGal):",round(float(np.mean(val)),8),
          "median:",round(float(np.median(val)),8))
