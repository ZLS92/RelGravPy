"""Fast spherical-harmonic summation with optional Numba acceleration.

This module is the single numerical engine used by gravity synthesis. It
computes degree/order expansions using O(N) work buffers for N evaluation
points, rather than allocating a Legendre matrix for each point. When Numba
cannot be imported or compiled, the SAME recurrences run with vectorized
NumPy blocks. A missing or incompatible Numba installation never prevents
importing or using RelGravPy.

All inputs and outputs are in float64. The 4-pi normalized associated
Legendre functions omit the Condon-Shortley phase.
"""
from math import cos, sin, sqrt, pi
import warnings
import numpy as np

try:
    from numba import njit, prange
except Exception:
    # Even a partially broken optional Numba installation must be harmless.
    njit = None
    prange = range


def _accumulate_points(lon, phi, radius, c, s, weight, model_radius):
    """Compute three dimensionless sums by fused degree/order recurrences.

    Parameters
    ----------
    lon, phi : ndarray, shape (N,)
        Longitude and GEOCENTRIC latitude of each sample, in degrees.
    radius : ndarray, shape (N,)
        Distance of each sample from the geocentre, in metres.
    c, s : ndarray, shape (L + 1, L + 1)
        Dimensionless fully normalized coefficients with normal-field terms
        already subtracted; triangular region m <= n is used.
    weight : ndarray, shape (L + 1,)
        Dimensionless spectral taper weight per degree.
    model_radius : float
        SH model reference radius in metres.

    Returns
    -------
    ndarray, shape (3, N)
        Dimensionless harmonic sums S0, S1 and S2, for conversion to
        disturbing potential T = GM*S0/r, radial disturbance
        delta_g = GM*S1/r**2, and its radial derivative
        d(delta_g)/dr = -GM*S2/r**3.

    Notes
    -----
    Summation is organized by order m, so each associated Legendre term is
    produced using the preceding two degrees without constructing the full
    degree/order matrix. No fast-math reassociation is enabled.
    """
    count = lon.size
    degree = weight.size - 1
    sums = np.zeros((3, count), dtype=np.float64)
    for k in prange(count):
        latitude = phi[k] * (pi / 180.0)
        longitude = lon[k] * (pi / 180.0)
        u, v = sin(latitude), cos(latitude)
        cos1, sin1 = cos(longitude), sin(longitude)
        cos_m, sin_m = 1.0, 0.0
        rho = model_radius / radius[k]
        radial_m, pmm = 1.0, 1.0
        sum0, sum1, sum2 = 0.0, 0.0, 0.0
        for m in range(degree + 1):
            # Compute sectorial P[m,m] using the 4-pi normalization.
            if m == 1:
                pmm = sqrt(3.0) * v
            elif m > 1:
                pmm *= sqrt((2.0*m+1.0)/(2.0*m)) * v
            if m:
                old_cos = cos_m
                cos_m = old_cos*cos1 - sin_m*sin1
                sin_m = sin_m*cos1 + old_cos*sin1

            value = (pmm * (c[m,m]*cos_m + s[m,m]*sin_m) *
                     weight[m] * radial_m)
            sum0 += value
            sum1 += (m+1.0)*value
            sum2 += (m+1.0)*(m+2.0)*value

            if m < degree:
                # First off-diagonal term P[m+1,m].
                pnm1 = sqrt(2.0*m+3.0)*u*pmm
                radial = radial_m*rho
                n = m+1
                value = (pnm1 * (c[n,m]*cos_m+s[n,m]*sin_m) *
                         weight[n]*radial)
                sum0 += value
                sum1 += (n+1.0)*value
                sum2 += (n+1.0)*(n+2.0)*value
                pnm2 = pmm

                # Remaining tesseral terms P[n,m], n >= m+2.
                for n in range(m+2, degree+1):
                    nn, mm = float(n), float(m)
                    a = sqrt(((2*nn+1)*(2*nn-1))/((nn-mm)*(nn+mm)))
                    b = sqrt(((2*nn+1)*(nn+mm-1)*(nn-mm-1)) /
                             ((2*nn-3)*(nn-mm)*(nn+mm)))
                    pnm = a*u*pnm1-b*pnm2
                    radial *= rho
                    value = (pnm * (c[n,m]*cos_m+s[n,m]*sin_m) *
                             weight[n]*radial)
                    sum0 += value
                    sum1 += (n+1.0)*value
                    sum2 += (n+1.0)*(n+2.0)*value
                    pnm2, pnm1 = pnm1, pnm
            radial_m *= rho
        sums[0,k], sums[1,k], sums[2,k] = sum0, sum1, sum2
    return sums


# Compile the same scalar recurrence in parallel across independent points.
_compiled_accumulator = (njit(cache=True, parallel=True)(_accumulate_points)
                         if njit is not None else None)
_numba_failed = False


def _accumulate_numpy(lon, phi, radius, c, s, weight, model_radius,
                      block_size=256):
    """Evaluate the identical SH recurrence in memory-bounded NumPy blocks.

    Parameters
    ----------
    lon, phi, radius, c, s, weight, model_radius
        Same meaning and physical units as in _accumulate_points().
    block_size : int, optional
        Maximum number of samples processed in one vectorized block.

    Returns
    -------
    ndarray, shape (3, N)
        S0, S1, S2 in exactly the units of _accumulate_points().

    Notes
    -----
    Only O(block_size) temporary arrays are required at any degree.
    In contrast to a Python loop per point, the recurrence operations
    act on complete blocks of coordinates.
    """
    count = lon.size
    degree = weight.size-1
    out = np.empty((3,count), dtype=np.float64)
    for start in range(0, count, block_size):
        stop = min(start+block_size, count)
        lam = np.deg2rad(lon[start:stop])
        ang = np.deg2rad(phi[start:stop])
        u, v = np.sin(ang), np.cos(ang)
        cm, sm = np.ones_like(u), np.zeros_like(u)
        c1, s1 = np.cos(lam), np.sin(lam)
        rho = model_radius/radius[start:stop]
        radial_m = np.ones_like(u)
        pmm = np.ones_like(u)
        t, dg, gradient = np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)
        for m in range(degree+1):
            if m == 1:
                pmm = sqrt(3.0)*v
            elif m > 1:
                pmm = pmm*sqrt((2.0*m+1.0)/(2.0*m))*v
            if m:
                cm, sm = cm*c1-sm*s1, sm*c1+cm*s1
            value = (pmm*(c[m,m]*cm+s[m,m]*sm) *
                     weight[m]*radial_m)
            t += value
            dg += (m+1)*value
            gradient += (m+1)*(m+2)*value
            if m < degree:
                p1 = sqrt(2*m+3)*u*pmm
                radial = radial_m*rho
                n = m+1
                value = p1*(c[n,m]*cm+s[n,m]*sm)*weight[n]*radial
                t += value
                dg += (n+1)*value
                gradient += (n+1)*(n+2)*value
                p2 = pmm
                for n in range(m+2,degree+1):
                    a = sqrt((2*n+1)*(2*n-1)/((n-m)*(n+m)))
                    b = sqrt((2*n+1)*(n+m-1)*(n-m-1) /
                             ((2*n-3)*(n-m)*(n+m)))
                    p = a*u*p1-b*p2
                    radial *= rho
                    value = p*(c[n,m]*cm+s[n,m]*sm)*weight[n]*radial
                    t += value
                    dg += (n+1)*value
                    gradient += (n+1)*(n+2)*value
                    p2, p1 = p1, p
            radial_m *= rho
        out[0,start:stop], out[1,start:stop], out[2,start:stop] = (
            t, dg, gradient)
    return out


def accumulate(lon, phi, radius, c, s, weight, model_radius):
    """Evaluate fast SH sums, using Numba if usable and otherwise NumPy.

    Parameters
    ----------
    lon, phi, radius : array_like
        Longitude and geocentric latitude (degrees), and distance from
        geocentre (metres). Each array must contain the same point count.
    c, s : array_like, shape (L + 1, L + 1)
        Dimensionless real harmonic coefficients (normal field removed).
    weight : array_like, shape (L + 1,)
        Degree-dependent cosine taper weights.
    model_radius : float
        Model reference radius, metres.

    Returns
    -------
    ndarray, shape (3, N)
        Dimensionless potential/radial derivative sums S0, S1, S2.

    Notes
    -----
    Neither this API nor any public RelGravPy method requires Numba.
    The JIT path is selected automatically. On an import or compilation
    failure, the NumPy implementation completes the calculation instead.
    No acceleration mode needs to be selected by the caller.
    """
    global _numba_failed
    lon = np.ascontiguousarray(np.asarray(lon,dtype=np.float64).ravel())
    phi = np.ascontiguousarray(np.asarray(phi,dtype=np.float64).ravel())
    radius = np.ascontiguousarray(np.asarray(radius,dtype=np.float64).ravel())
    c = np.ascontiguousarray(c,dtype=np.float64)
    s = np.ascontiguousarray(s,dtype=np.float64)
    weight = np.ascontiguousarray(weight,dtype=np.float64)
    if _compiled_accumulator is not None and not _numba_failed:
        try:
            return _compiled_accumulator(
                lon,phi,radius,c,s,weight,float(model_radius))
        except Exception as exc:
            _numba_failed = True
            warnings.warn("Numba SH acceleration unavailable (%s); "
                          "using NumPy instead" % exc,
                          RuntimeWarning,stacklevel=2)
    return _accumulate_numpy(lon,phi,radius,c,s,weight,float(model_radius))
