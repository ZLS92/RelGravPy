# -*- coding: utf-8 -*-
"""
Investigate conditioning of the discrete GMT-style minimum-curvature problem.

Run:
    python tests/diagnose_min_curvature_condition.py

Requires only existing RelGravPy dependencies (NumPy and SciPy).
The script does not require GMT and does not modify production files.

It constructs the *final-grid* linear stencil operator including the natural
boundary conditions and Briggs off-node constraints. Then it computes an
estimate of the 1-norm condition number with sparse LU solves.
This is a diagnostic, not a production interpolator.
"""
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from test_min_curvature import mc


# -----------------------------------------------------------------------------
# Assemble final-grid linear operator for zero-tension minimum curvature
# -----------------------------------------------------------------------------
def condition_estimate(x, y, z, spacing=2., grid_size=51, tension=0.):
    """Estimate ||A||_1 * ||A^{-1}||_1 for one synthetic survey."""
    nx = ny = grid_size
    col = np.floor(x / spacing + 0.5).astype(int)
    row = ny-1-np.floor(y / spacing + 0.5).astype(int)
    distances = ((x - col*spacing)**2 +
                 (y - (spacing*(ny-1) - row*spacing))**2)
    index = np.lexsort((distances, col + nx*row))
    key = col[index] + nx*row[index]
    index = index[np.r_[True, key[1:] != key[:-1]]]
    x, y, z = x[index], y[index], z[index]

    design = np.column_stack((np.ones(len(z)), x/spacing, y/spacing))
    plane = np.linalg.lstsq(design, z, rcond=None)[0]
    residual = z - design @ plane
    rms = np.sqrt(np.mean(residual**2))
    coeff, a01, a02 = mc._gmt_surface_coefficients(1.0, tension)

    u = np.zeros((ny+4, nx+4), dtype=np.float64)
    status, briggs = mc._gmt_surface_constraints(
        x, y, residual/rms, nx, ny,
        (0, spacing*(nx-1), 0, spacing*(ny-1)),
        spacing, spacing, 1., a01, a02, plane, rms, 1, u
    )

    # Effective linear weights for off-node constrained and free nodes.
    weights = np.empty((12, ny, nx))
    weights[:, :, :] = coeff[0, :, None, None]
    offnode = (status > 0) & (status < 5)
    weights[:, offnode] = (
        coeff[1, :, None] * briggs[:, :, 5][offnode][None, :]
    )
    for row_index, col_index in np.argwhere(offnode):
        quadrant = int(status[row_index, col_index])
        for k, node_index in enumerate(mc._GMT_SURFACE_QUADRANTS[quadrant]):
            weights[node_index, row_index, col_index] += (
                a02 * briggs[row_index, col_index, 5] *
                briggs[row_index, col_index, k]
            )
    weights[:, status == 5] = 0.

    # Apply A to each basis vector. The stencil and boundary conditions
    # have local support: the resulting matrix is sparse.
    rows, columns, values = [], [], []
    for index in range(nx*ny):
        r, c = divmod(index, nx)
        u.fill(0.)
        u[r+2, c+2] = 1.
        mc._gmt_surface_boundary(u, nx, ny, 1., tension)
        forecast = np.zeros((ny, nx), dtype=np.float64)
        for k, (dr, dc) in enumerate(mc._GMT_SURFACE_NEIGHBORS):
            forecast += (
                weights[k] *
                u[2+dr:ny+2+dr, 2+dc:nx+2+dc]
            )
        column = -forecast.ravel()
        column[index] += 1.
        nonzero = np.flatnonzero(np.abs(column) > 1e-12)
        rows.extend(nonzero)
        columns.extend([index] * len(nonzero))
        values.extend(column[nonzero])

    A = sparse.csr_matrix((values, (rows, columns)),
                          shape=(nx*ny, nx*ny))
    lu = spla.splu(A.tocsc())
    inverse = spla.LinearOperator(
        A.shape,
        matvec=lambda vector: lu.solve(vector),
        rmatvec=lambda vector: lu.solve(vector, trans="T"),
    )
    return float(spla.onenormest(A) * spla.onenormest(inverse))


# -----------------------------------------------------------------------------
# Controlled comparison: central-only versus widely distributed stations
# -----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(2031)
    central_x = rng.uniform(30., 70., 85)
    central_y = rng.uniform(27., 73., 85)
    central_z = (
        2.5*np.exp(-((central_x-45)**2+(central_y-62)**2)/200)
        - np.exp(-((central_x-65)**2+(central_y-38)**2)/180)
        + 0.016*central_x-0.011*central_y
    )
    central_condition = condition_estimate(central_x, central_y, central_z)
    central_tension_condition = condition_estimate(
        central_x, central_y, central_z, tension=0.25
    )

    rng = np.random.default_rng(2031)
    wide_x = rng.uniform(1.5, 98.5, 180)
    wide_y = rng.uniform(1.5, 98.5, 180)
    wide_z = (
        2.5*np.exp(-((wide_x-45)**2+(wide_y-62)**2)/200)
        - np.exp(-((wide_x-65)**2+(wide_y-38)**2)/180)
        + 0.016*wide_x-0.011*wide_y
    )
    wide_condition = condition_estimate(wide_x, wide_y, wide_z)

    print("Condition estimate — T=0, central observations:", central_condition)
    print("Condition estimate — T=0.25, central observations:", central_tension_condition)
    print("Condition estimate — T=0, spread observations:", wide_condition)
    print("Condition ratio (central / spread):", central_condition/wide_condition)
    assert np.isfinite([central_condition, wide_condition]).all()
    assert central_condition / wide_condition > 20.
    assert central_condition / central_tension_condition > 10.


if __name__ == "__main__":
    main()
