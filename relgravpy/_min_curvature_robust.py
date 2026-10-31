# -*- coding: utf-8 -*-
"""Sparse variational minimum-curvature gridding.

This solves a regularized bending-energy minimization problem on a complete
rectangular grid. It is intentionally NOT a bitwise/numerical GMT surface port.

The implementation uses NumPy and SciPy already required by RelGravPy.
"""
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


# -----------------------------------------------------------------------------
def _mc_robust_difference(n, order):
    """Return non-periodic first-/second-order difference matrix."""
    if order == 1:
        if n < 2:
            raise ValueError("Grid requires at least two nodes per axis")
        return sparse.diags([-np.ones(n-1), np.ones(n-1)], [0,1],
                            shape=(n-1,n), format="csr")
    if order == 2:
        if n < 3:
            raise ValueError("Grid requires at least three nodes per axis")
        return sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)],
                            [0,1,2], shape=(n-2,n), format="csr")
    raise ValueError("Order must be 1 or 2")


# -----------------------------------------------------------------------------
def _mc_robust_observation_matrix(x, y, nx, ny, west, north, dx, dy):
    """Create sparse bilinear data constraints (never snap off-node samples)."""
    xc = np.clip((x-west)/dx, 0, nx-1)
    yc = np.clip((north-y)/dy, 0, ny-1)
    c0 = np.minimum(np.floor(xc).astype(int), nx-2)
    r0 = np.minimum(np.floor(yc).astype(int), ny-2)
    fx, fy = xc-c0, yc-r0
    weights = np.stack(((1-fx)*(1-fy), fx*(1-fy),
                        (1-fx)*fy, fx*fy), axis=1)
    indices = np.stack((r0*nx+c0, r0*nx+c0+1,
                        (r0+1)*nx+c0, (r0+1)*nx+c0+1), axis=1)
    return sparse.coo_matrix(
        (weights.ravel(),
         (np.repeat(np.arange(len(x)),4), indices.ravel())),
        shape=(len(x),nx*ny)).tocsr()


# -----------------------------------------------------------------------------
def _mc_robust_solver(x, y, z, region, spacing, *, tension=0.,
                      data_weight=1e4, anchor=1e-2, solver="auto",
                      tolerance=1e-9, max_iterations=4000,
                      return_info=False):
    """Fill the complete grid by minimizing a regularized curvature energy.

    In terms of the detrended field u, the objective is:

      E = data_weight * ||H u - data||^2
        + (1-tension) * (||Dxx u||^2+2||Dxy u||^2+||Dyy u||^2)
        + tension * (||Dx u||^2+||Dy u||^2)
        + anchor * ||u||^2.

    D matrices are physically scaled by dx/dy, with a weak residual-to-plane
    anchor for stabilizing unconstrained edges/padding. A strictly positive
    anchor makes the optimization mathematically well posed; the regularizer
    and its boundary behavior are NOT identical to GMT surface's natural BCs.
    """
    x,y,z = (np.asarray(a, dtype=float).ravel() for a in (x,y,z))
    if not (len(x)==len(y)==len(z)):
        raise ValueError("x, y and z must have equal lengths")
    n_original=len(z)
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)
    x,y,z=x[valid],y[valid],z[valid]
    west,east,south,north=map(float,region)
    dx,dy=((float(spacing),)*2 if np.isscalar(spacing)
           else map(float,spacing))
    if not (np.isfinite([west,east,south,north,dx,dy]).all()
            and east>west and north>south and dx>0 and dy>0):
        raise ValueError("Invalid region or spacing")
    qx,qy=(east-west)/dx,(north-south)/dy
    if not (np.isclose(qx,round(qx),atol=1e-8,rtol=0)
            and np.isclose(qy,round(qy),atol=1e-8,rtol=0)):
        raise ValueError("Region extents must be multiples of spacing")
    nx,ny=round(qx)+1,round(qy)+1
    if min(nx,ny)<3:
        raise ValueError("Minimum three grid nodes per direction")
    in_region=(x>=west)&(x<=east)&(y>=south)&(y<=north)
    x,y,z=x[in_region],y[in_region],z[in_region]
    if len(z)<3:
        raise ValueError("At least three valid observations inside region required")
    if not (0<=tension<=1 and data_weight>0 and anchor>0 and
            tolerance>0 and max_iterations>=1):
        raise ValueError("Invalid tension, data_weight, anchor, tolerance or iterations")
    # -------------------------------------------------------------------------
    # Remove planar trend; extrapolation defaults towards this reference plane.
    # -------------------------------------------------------------------------
    xs=west+np.arange(nx)*dx
    ys=north-np.arange(ny)*dy
    xy=np.column_stack((np.ones(len(x)),(x-west)/dx,(y-south)/dy))
    plane=np.linalg.lstsq(xy,z,rcond=None)[0]
    trend=(plane[0]+plane[1]*np.arange(nx)[None,:]
           +plane[2]*np.arange(ny-1,-1,-1)[:,None])
    residual=z-xy@plane
    scale=float(np.sqrt(np.mean(residual**2)))
    if scale<=1e-14*max(1.,float(np.max(abs(z)))):
        info=dict(method="variational",converged=True,iterations=0,
                  retained_observations=len(z),
                  discarded_observations=n_original-len(z),
                  observation_rmse=0.,trend_coefficients=plane)
        return (xs,ys,trend,info) if return_info else (xs,ys,trend)
    H=_mc_robust_observation_matrix(x,y,nx,ny,west,north,dx,dy)
    zn=residual/scale
    # -------------------------------------------------------------------------
    # Discrete thin-plate bending energy plus optional first-order tension.
    # -------------------------------------------------------------------------
    ix=sparse.eye(nx,format="csr")
    iy=sparse.eye(ny,format="csr")
    d1x,d1y=_mc_robust_difference(nx,1),_mc_robust_difference(ny,1)
    d2x,d2y=_mc_robust_difference(nx,2),_mc_robust_difference(ny,2)
    x1=sparse.kron(iy,d1x,format="csr")
    y1=sparse.kron(d1y,ix,format="csr")
    x2=sparse.kron(iy,d2x,format="csr")
    y2=sparse.kron(d2y,ix,format="csr")
    xy=sparse.kron(d1y,d1x,format="csr")
    alpha=dx/dy
    curvature=(x2.T@x2+2*alpha**2*(xy.T@xy)
               +alpha**4*(y2.T@y2))
    gradient=x1.T@x1+alpha**2*(y1.T@y1)
    n=nx*ny
    Q=((1-tension)*curvature+tension*gradient).tocsr()
    A=(data_weight*(H.T@H)+Q+anchor*sparse.eye(n,format="csr")).tocsr()
    b=data_weight*(H.T@zn)
    # -------------------------------------------------------------------------
    # Sparse direct solution for moderate grids, preconditioned CG otherwise.
    # -------------------------------------------------------------------------
    if solver not in ("auto","direct","cg"):
        raise ValueError("solver must be 'auto', 'direct' or 'cg'")
    method=("direct" if n<=7000 else "cg") if solver=="auto" else solver
    iterations=0
    if method=="direct":
        solution=spla.spsolve(A.tocsc(),b)
        converged=bool(np.isfinite(solution).all())
    else:
        diagonal=A.diagonal()
        preconditioner=spla.LinearOperator(
            (n,n),matvec=lambda vector:vector/diagonal)
        def counter(_):
            nonlocal iterations
            iterations+=1
        solution,flag=spla.cg(
            A,b,rtol=tolerance,atol=0.,M=preconditioner,
            maxiter=max_iterations,callback=counter)
        converged=bool(flag==0 and np.isfinite(solution).all())
    if not converged:
        warnings.warn("Variational minimum curvature solver did not converge",
                      RuntimeWarning,stacklevel=2)
    grid=solution.reshape(ny,nx)*scale+trend
    mismatch=scale*(H@solution-zn)
    relative_residual=(np.linalg.norm(A@solution-b)
                       /max(np.linalg.norm(b),1e-30))
    info=dict(method="variational",linear_solver=method,
              converged=converged,iterations=iterations,
              relative_system_residual=float(relative_residual),
              retained_observations=len(z),
              discarded_observations=n_original-len(z),
              observation_rmse=float(np.sqrt(np.mean(mismatch**2))),
              anchor=float(anchor),tension=float(tension),
              data_weight=float(data_weight),trend_coefficients=plane)
    return (xs,ys,grid,info) if return_info else (xs,ys,grid)
