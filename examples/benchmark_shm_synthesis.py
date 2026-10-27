"""Reproducible full-synthesis benchmark for the optional SH JIT accelerator.

Run from the repository root with::

    python -m examples.benchmark_shm_synthesis --degree 120 --points 400

Benchmarks use deterministic *synthetic* spherical harmonic coefficients, not
an observational validation data set. Both modes call the same public-facing
sh_synthesis_core() API. The first accelerated call is intentionally excluded
from timings to avoid charging JIT compilation to recurring evaluation time.
"""
import argparse
from statistics import median
from time import perf_counter

import numpy as np

from relgravpy import shm_tools as sh
from relgravpy import _sh_fast


def _synthetic_model(degree):
    """Return normalized synthetic coefficients with WGS84 zonal reference."""
    rng = np.random.default_rng(20261027)
    c = np.tril(rng.normal(0, 1e-7, (degree+1, degree+1)))
    s = np.tril(rng.normal(0, 1e-7, (degree+1, degree+1)))
    c[0,0] = 1.0
    s[:,0] = 0.0
    for n, coeff in enumerate(sh.sh_wgs84_normal_zonals(
            sh.GM_WGS84,sh.A_WGS84,degree)):
        if n and n % 2 == 0:
            c[n,0] += coeff
    return dict(c=c,s=s,gm=sh.GM_WGS84,
                radius=sh.A_WGS84,lmax=degree)


def _time_synthesis(model, lon, lat, repeats):
    """Return median wall time (seconds) for the complete synthesis function."""
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = sh.sh_synthesis_core(
            model,lon=lon,lat=lat,height_reference="ellipsoid",
            fields=("potential","grav_dist","grav_gradient"),
            surface="sphere")
        timings.append(perf_counter()-start)
    return median(timings),result


def main():
    """Compare Numba and NumPy and report numerical agreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree",type=int,default=120)
    parser.add_argument("--points",type=int,default=400)
    parser.add_argument("--repeats",type=int,default=3)
    args = parser.parse_args()
    if args.degree < 0 or args.points < 1 or args.repeats < 1:
        parser.error("degree >= 0, points >= 1 and repeats >= 1 are required")

    model = _synthetic_model(args.degree)
    rng = np.random.default_rng(42)
    lon = rng.uniform(-180,180,args.points)
    lat = rng.uniform(-75,75,args.points)

    compiled = _sh_fast._compiled_accumulator
    if compiled is not None:
        # Warm up the JIT before benchmarking. Compiled code is cached when
        # supported, but first-call compilation is machine/environment-specific.
        sh.sh_synthesis_core(
            model,lon=lon[:2],lat=lat[:2],height_reference="ellipsoid",
            fields=("potential",),surface="sphere")
        accelerated_sec,accelerated = _time_synthesis(
            model,lon,lat,args.repeats)
    else:
        accelerated_sec,accelerated = None,None

    try:
        # Simulate unavailable Numba without affecting the public API.
        _sh_fast._compiled_accumulator = None
        numpy_sec,numpy_only = _time_synthesis(model,lon,lat,args.repeats)
    finally:
        _sh_fast._compiled_accumulator = compiled

    print(f"Degree: {args.degree}; points: {args.points}")
    print(f"NumPy fallback median: {numpy_sec:.6f} s")
    if accelerated_sec is not None:
        print(f"Numba compiled median: {accelerated_sec:.6f} s")
        print(f"Speed ratio (NumPy / Numba): "
              f"{numpy_sec / accelerated_sec:.2f}x")
        for name in ("potential","grav_dist","grav_gradient"):
            error = np.max(np.abs(accelerated[name]-numpy_only[name]))
            print(f"Maximum absolute {name} difference: {error:.3e}")
    else:
        print("Numba not installed or unavailable: NumPy-only mode succeeded.")


if __name__ == "__main__":
    main()
