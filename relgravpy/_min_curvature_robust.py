# -*- coding: utf-8 -*-
"""Sparse, regularized minimum-curvature interpolation for RelGravPy.

This module contains one mathematical model. Direct sparse LU and conjugate
gradients are alternative linear-system solvers, not different interpolators.
"""

import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


# -----------------------------------------------------------------------------
# Finite-difference operators
# -----------------------------------------------------------------------------


def _mc_robust_difference(n, order):
    """Build a nonperiodic finite-difference operator along one grid axis."""
    if order == 1:
        if n < 2:
            raise ValueError("At least two nodes are needed for first differences")

        result = sparse.diags(
            [-np.ones(n - 1), np.ones(n - 1)],
            offsets=[0, 1],
            shape=(n - 1, n),
            format="csr",
        )
    elif order == 2:
        if n < 3:
            raise ValueError("At least three nodes are needed for second differences")

        result = sparse.diags(
            [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
            offsets=[0, 1, 2],
            shape=(n - 2, n),
            format="csr",
        )
    else:
        raise ValueError("Difference order must be 1 or 2")

    return result


# -----------------------------------------------------------------------------
# Observation constraints
# -----------------------------------------------------------------------------


def _mc_robust_observation_matrix(x, y, nx, ny, west, north, dx, dy):
    """Map each measurement onto its four surrounding grid nodes.

    The matrix rows sum to one. Bilinear interpolation retains the true
    observation position, avoiding the error introduced by node snapping.
    """
    x_index = np.clip((x - west) / dx, 0, nx - 1)
    y_index = np.clip((north - y) / dy, 0, ny - 1)

    left = np.minimum(np.floor(x_index).astype(int), nx - 2)
    top = np.minimum(np.floor(y_index).astype(int), ny - 2)
    x_fraction = x_index - left
    y_fraction = y_index - top

    weights = np.stack(
        (
            (1 - x_fraction) * (1 - y_fraction),
            x_fraction * (1 - y_fraction),
            (1 - x_fraction) * y_fraction,
            x_fraction * y_fraction,
        ),
        axis=1,
    )
    indices = np.stack(
        (
            top * nx + left,
            top * nx + left + 1,
            (top + 1) * nx + left,
            (top + 1) * nx + left + 1,
        ),
        axis=1,
    )

    observations = sparse.coo_matrix(
        (
            weights.ravel(),
            (np.repeat(np.arange(len(x)), 4), indices.ravel()),
        ),
        shape=(len(x), nx * ny),
    )
    result = observations.tocsr()

    return result


# -----------------------------------------------------------------------------
# Variational minimum-curvature solver
# -----------------------------------------------------------------------------


def _mc_robust_solver(
    x, y, z, region, spacing, *, tension=0.0,
    data_weight=1e4, anchor=1e-2, solver="auto",
    tolerance=1e-9, max_iterations=4000, return_info=False,
):
    """Minimize curvature with data fidelity and planar trend anchoring.

    For the normalized, detrended grid u, the minimized energy is

        data_weight * ||H u - observations||²
        + (1 - tension) * [||Dxx u||² + 2||Dxy u||² + ||Dyy u||²]
        + tension * [||Dx u||² + ||Dy u||²]
        + anchor * ||u||².

    Grid-node coordinates are in projected units. Derivative operators
    account for non-square spacing through the dx/dy aspect ratio.
    The weak positive trend anchor stabilizes poorly observed borders.
    """
    # -------------------------------------------------------------------------
    # Validate the data and the exact output grid geometry.
    # -------------------------------------------------------------------------
    x, y, z = (
        np.asarray(values, dtype=float).ravel()
        for values in (x, y, z)
    )
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, and z must have equal lengths")

    original_count = len(z)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]

    west, east, south, north = map(float, region)
    if np.isscalar(spacing):
        dx = dy = float(spacing)
    else:
        dx, dy = (float(value) for value in spacing)

    geometry = np.array([west, east, south, north, dx, dy])
    if (
        not np.isfinite(geometry).all()
        or east <= west
        or north <= south
        or min(dx, dy) <= 0
    ):
        raise ValueError("Invalid region or grid spacing")

    x_cells = (east - west) / dx
    y_cells = (north - south) / dy
    if not (
        np.isclose(x_cells, round(x_cells), rtol=0, atol=1e-8)
        and np.isclose(y_cells, round(y_cells), rtol=0, atol=1e-8)
    ):
        raise ValueError("Region extents must be multiples of spacing")

    nx = round(x_cells) + 1
    ny = round(y_cells) + 1
    if min(nx, ny) < 3:
        raise ValueError("Grid requires at least three nodes per axis")

    within_region = (
        (x >= west) & (x <= east)
        & (y >= south) & (y <= north)
    )
    x, y, z = x[within_region], y[within_region], z[within_region]

    if len(z) < 3:
        raise ValueError("At least three valid observations are required")
    if not (
        0 <= tension <= 1
        and data_weight > 0
        and anchor > 0
        and tolerance > 0
        and max_iterations >= 1
    ):
        raise ValueError("Invalid regularization or convergence parameters")

    grid_x = west + np.arange(nx) * dx
    grid_y = north - np.arange(ny) * dy

    # -------------------------------------------------------------------------
    # Remove the fitted plane to isolate the bending-energy component.
    # -------------------------------------------------------------------------
    design = np.column_stack(
        (
            np.ones(len(x)),
            (x - west) / dx,
            (y - south) / dy,
        )
    )
    plane = np.linalg.lstsq(design, z, rcond=None)[0]

    grid_col = np.arange(nx)[None, :]
    grid_row = np.arange(ny - 1, -1, -1)[:, None]
    trend = plane[0] + plane[1] * grid_col + plane[2] * grid_row

    residual = z - design @ plane
    scale = float(np.sqrt(np.mean(residual ** 2)))

    # A true plane has zero curvature: no sparse solve is necessary.
    if scale <= 1e-14 * max(1.0, float(np.max(np.abs(z)))):
        info = {
            "method": "variational",
            "converged": True,
            "iterations": 0,
            "retained_observations": len(z),
            "discarded_observations": original_count - len(z),
            "observation_rmse": 0.0,
            "trend_coefficients": plane,
        }
        result = (grid_x, grid_y, trend)
        if return_info:
            result = (grid_x, grid_y, trend, info)

        return result

    observations = _mc_robust_observation_matrix(
        x, y, nx, ny, west, north, dx, dy,
    )
    normalized_data = residual / scale

    # -------------------------------------------------------------------------
    # Construct the discrete thin-plate energy and gradient penalty.
    # -------------------------------------------------------------------------
    eye_x = sparse.eye(nx, format="csr")
    eye_y = sparse.eye(ny, format="csr")

    d1x = _mc_robust_difference(nx, 1)
    d1y = _mc_robust_difference(ny, 1)
    d2x = _mc_robust_difference(nx, 2)
    d2y = _mc_robust_difference(ny, 2)

    first_x = sparse.kron(eye_y, d1x, format="csr")
    first_y = sparse.kron(d1y, eye_x, format="csr")
    second_x = sparse.kron(eye_y, d2x, format="csr")
    second_y = sparse.kron(d2y, eye_x, format="csr")
    mixed_xy = sparse.kron(d1y, d1x, format="csr")

    ratio = dx / dy
    curvature = (
        second_x.T @ second_x
        + 2 * ratio ** 2 * (mixed_xy.T @ mixed_xy)
        + ratio ** 4 * (second_y.T @ second_y)
    )
    gradient = (
        first_x.T @ first_x
        + ratio ** 2 * (first_y.T @ first_y)
    )

    node_count = nx * ny
    regularization = (
        (1 - tension) * curvature + tension * gradient
    ).tocsr()
    system = (
        data_weight * (observations.T @ observations)
        + regularization
        + anchor * sparse.eye(node_count, format="csr")
    ).tocsr()
    right_hand_side = data_weight * (observations.T @ normalized_data)

    # -------------------------------------------------------------------------
    # Solve the same system by sparse LU or preconditioned conjugate gradients.
    # -------------------------------------------------------------------------
    if solver not in ("auto", "direct", "cg"):
        raise ValueError("linear_solver must be 'auto', 'direct', or 'cg'")

    method = solver
    if method == "auto":
        method = "direct" if node_count <= 7000 else "cg"

    iterations = 0
    if method == "direct":
        solution = spla.spsolve(system.tocsc(), right_hand_side)
        converged = bool(np.isfinite(solution).all())
    else:
        diagonal = system.diagonal()
        preconditioner = spla.LinearOperator(
            (node_count, node_count),
            matvec=lambda vector: vector / diagonal,
        )

        def count_iteration(_):
            nonlocal iterations
            iterations += 1

        solution, status = spla.cg(
            system,
            right_hand_side,
            rtol=tolerance,
            atol=0.0,
            M=preconditioner,
            maxiter=max_iterations,
            callback=count_iteration,
        )
        converged = bool(status == 0 and np.isfinite(solution).all())

    if not converged:
        warnings.warn(
            "Minimum-curvature linear solver did not converge",
            RuntimeWarning,
            stacklevel=2,
        )

    # -------------------------------------------------------------------------
    # Restore physical units and summarize the observation misfit.
    # -------------------------------------------------------------------------
    grid = solution.reshape(ny, nx) * scale + trend
    data_error = scale * (observations @ solution - normalized_data)
    linear_residual = (
        np.linalg.norm(system @ solution - right_hand_side)
        / max(np.linalg.norm(right_hand_side), 1e-30)
    )

    info = {
        "method": "variational",
        "linear_solver": method,
        "converged": converged,
        "iterations": iterations,
        "relative_system_residual": float(linear_residual),
        "retained_observations": len(z),
        "discarded_observations": original_count - len(z),
        "observation_rmse": float(np.sqrt(np.mean(data_error ** 2))),
        "anchor": float(anchor),
        "tension": float(tension),
        "data_weight": float(data_weight),
        "trend_coefficients": plane,
    }

    result = (grid_x, grid_y, grid)
    if return_info:
        result = (grid_x, grid_y, grid, info)

    return result
