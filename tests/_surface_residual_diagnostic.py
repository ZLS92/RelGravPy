# -*- coding: utf-8 -*-
"""
Finite-difference residual diagnostic for GMT surface numerical comparisons.

This helper evaluates the *same discrete equations* independently from the
SOR convergence test. It is restricted to the 0..100 reference region, node
registration, and the synthetic datasets used by the GMT comparison tests.
"""
import numpy as np


def discrete_residual(mc, grid, x, y, z, tension, step=2):
    """Return (max absolute residual, RMS residual) in normalized z units."""
    ny, nx = grid.shape
    col = np.floor(x / step + 0.5).astype(int)
    row = ny - 1 - np.floor(y / step + 0.5).astype(int)
    dist = (x - col * step)**2 + (y - (step * (ny-1) - row*step))**2
    idx = np.lexsort((dist, col + nx*row))
    key = col[idx] + nx*row[idx]
    idx = idx[np.r_[True, key[1:] != key[:-1]]]
    x, y, z = x[idx], y[idx], z[idx]
    design = np.column_stack((np.ones(len(x)), x/step, y/step))
    plane = np.linalg.lstsq(design, z, rcond=None)[0]
    residual_z = z - design @ plane
    rms = np.sqrt(np.mean(residual_z**2))
    coeff, a01, a02 = mc._gmt_surface_coefficients(1., tension)
    trend = (plane[0] + plane[1]*np.arange(nx)[None, :] +
             plane[2]*np.arange(ny-1, -1, -1)[:, None])
    u = np.zeros((ny+4, nx+4), dtype=np.float32)
    u[2:-2, 2:-2] = ((grid - trend) / rms).astype(np.float32)
    fixed_u = np.zeros_like(u)
    status, b = mc._gmt_surface_constraints(
        x, y, residual_z/rms, nx, ny,
        (0, step*(nx-1), 0, step*(ny-1)), step, step,
        1., a01, a02, plane, rms, 1, fixed_u
    )
    mc._gmt_surface_boundary(u, nx, ny, 1., tension)
    values = u[2:-2, 2:-2].astype(np.float64)
    forecast = np.zeros_like(values)
    offnode = (status > 0) & (status < 5)
    frozen = status == 5
    for k, (dr, dc) in enumerate(mc._GMT_SURFACE_NEIGHBORS):
        v = u[2+dr:2+ny+dr, 2+dc:2+nx+dc].astype(np.float64)
        forecast[~offnode] += coeff[0, k] * v[~offnode]
        forecast[offnode] += (
            coeff[1, k] * b[:, :, 5][offnode] * v[offnode]
        )
    forecast[offnode] += a02 * b[:, :, 5][offnode] * b[:, :, 4][offnode]
    for rr, cc in np.argwhere(offnode):
        quadrant = status[rr, cc]
        for k in range(4):
            ind = mc._GMT_SURFACE_QUADRANTS[quadrant, k]
            dr, dc = mc._GMT_SURFACE_NEIGHBORS[ind]
            forecast[rr, cc] += (
                a02 * b[rr, cc, 5] * b[rr, cc, k] *
                u[rr+2+dr, cc+2+dc]
            )
    residual = values - forecast
    residual[frozen] = values[frozen] - fixed_u[2:-2, 2:-2][frozen]
    return float(np.max(np.abs(residual))), float(np.sqrt(np.mean(residual**2)))
