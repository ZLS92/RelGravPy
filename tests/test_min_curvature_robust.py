# -*- coding: utf-8 -*-
"""Regression tests for robust gap filling, padding and block_m preprocessing."""
import ast
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import scipy as sp

from test_min_curvature import mc, min_curvature


# -----------------------------------------------------------------------------
# Isolate the exact repository block_m() implementation without importing
# unrelated optional OGR/GDAL modules from the huge legacy utils.py module.
# -----------------------------------------------------------------------------
repo = Path(__file__).resolve().parents[1]
package = sys.modules.get("relgravpy")
if package is None:
    package = types.ModuleType("relgravpy")
    package.__path__ = [str(repo/"relgravpy")]
    sys.modules["relgravpy"] = package
mc.__package__ = "relgravpy"

source = (repo/"relgravpy"/"utils.py").read_text(encoding="utf-8")
tree = ast.parse(source)
block_function = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name=="block_m")
block_module = ast.Module(body=[block_function], type_ignores=[])
mc.__dict__["sp"] = sp
exec(compile(block_module, str(repo/"relgravpy"/"utils.py"), "exec"),
     mc.__dict__)

def _truth(x,y):
    return (2.5*np.exp(-((x-38)**2+(y-61)**2)/310)
            -1.2*np.exp(-((x-70)**2+(y-29)**2)/190)
            +0.015*x -0.02*y)


# -----------------------------------------------------------------------------
def test_scattered_robust_full_grid_and_observation_mismatch():
    rng=np.random.default_rng(2026)
    x=rng.uniform(2,98,170)
    y=rng.uniform(2,98,170)
    z=_truth(x,y)
    gx,gy,grid,info=min_curvature(
        x,y,z,(0,100,0,100),2,solver="robust",
        tension=0,return_info=True)
    assert grid.shape==(51,51)
    assert np.isfinite(grid).all()
    assert info["converged"] and info["linear_solver"]=="direct"
    assert info["relative_system_residual"]<1e-8
    assert info["observation_rmse"]<0.01
    assert info["solver"]=="robust"


def test_grid_nan_holes_and_padding_are_completely_filled():
    x=np.arange(0.,101.,5.)
    y=np.arange(100.,-1.,-5.)
    xx,yy=np.meshgrid(x,y)
    data=_truth(xx,yy)
    data[7:15,7:15]=np.nan
    data[3:7,12:17]=np.nan
    gx,gy,grid,info=min_curvature(
        x,y,data,spacing=5,solver="robust",
        padding=(4,6),anchor=0.02,return_info=True)
    assert grid.shape==(33,29)
    assert gx[0]==-20 and gx[-1]==120
    assert gy[0]==130 and gy[-1]==-30
    assert np.isfinite(grid).all()
    # Existing measured grid cells remain accurately reproduced.
    original = grid[6:27,4:25]
    np.testing.assert_allclose(original[np.isfinite(data)],
                               data[np.isfinite(data)],atol=0.01,rtol=0)
    assert info["padding"]==(4,6)
    assert info["discarded_observations"]>=84


def test_robust_sparse_center_with_large_padding_is_stable():
    rng=np.random.default_rng(2031)
    x=rng.uniform(30,70,85)
    y=rng.uniform(27,73,85)
    z=_truth(x,y)
    gx,gy,grid,info=min_curvature(
        x,y,z,(20,80,20,80),2,solver="robust",
        padding=10,anchor=0.01,tension=0,return_info=True)
    assert grid.shape==(51,51)
    assert np.isfinite(grid).all()
    assert info["converged"]
    assert np.max(np.abs(grid))<15
    assert info["observation_rmse"]<0.01


def test_block_mean_median_call_existing_block_m():
    rng=np.random.default_rng(1337)
    n=500
    x=rng.uniform(0,100,n)
    y=rng.uniform(0,100,n)
    z=_truth(x,y)+rng.normal(0,0.04,n)
    # Positive outliers in dense clusters should be limited by medians.
    z[rng.choice(n,35,replace=False)]+=10.
    kw=dict(region=(0,100,0,100),spacing=2,solver="robust",
            tension=0.15,anchor=0.02,block_size=8,return_info=True)
    gx,gy,mean,mi=min_curvature(x,y,z,block="mean",**kw)
    _,_,median,mdi=min_curvature(x,y,z,block="median",**kw)
    xx,yy=np.meshgrid(gx,gy)
    truth=_truth(xx,yy)
    rmse_mean=np.sqrt(np.mean((mean-truth)**2))
    rmse_median=np.sqrt(np.mean((median-truth)**2))
    print(f"Block mean RMSE: {rmse_mean:.6g}, median RMSE: {rmse_median:.6g}")
    assert np.isfinite(mean).all() and np.isfinite(median).all()
    assert mi["block_method"]=="mean" and mdi["block_method"]=="median"
    assert mdi["observations_before_block"]==n
    assert mdi["observations_after_block"]<n
    assert rmse_median<rmse_mean


def test_cg_and_direct_agree_for_medium_grid():
    rng=np.random.default_rng(31)
    x=rng.uniform(0,100,85)
    y=rng.uniform(0,100,85)
    z=_truth(x,y)
    kw=dict(region=(0,100,0,100),spacing=2,solver="robust",
            tension=0.25,anchor=0.1,data_weight=500,return_info=True)
    _,_,direct,di=min_curvature(x,y,z,linear_solver="direct",**kw)
    _,_,cg,ci=min_curvature(x,y,z,linear_solver="cg",
                            max_iterations=4000,tolerance=1e-9,**kw)
    assert di["converged"] and ci["converged"]
    np.testing.assert_allclose(cg,direct,atol=1e-5,rtol=0)
    assert ci["relative_system_residual"]<1e-7


def test_block_and_solver_validation():
    x=np.array([0.,20.,0.,20.,10.])
    y=np.array([0.,0.,20.,20.,10.])
    z=_truth(x,y)
    with pytest.raises(ValueError):
        min_curvature(x,y,z,(0,20,0,20),2,block="sum")
    with pytest.raises(ValueError):
        min_curvature(x,y,z,(0,20,0,20),2,padding=1.5)
    with pytest.raises(ValueError):
        min_curvature(x,y,z,(0,20,0,20),2,solver="none")



# -----------------------------------------------------------------------------
# Block aggregation and conditioning are NOT monotonically related.
# -----------------------------------------------------------------------------
def test_block_condition_number_can_improve_or_worsen():
    """Show both directions with real block_m and the actual robust matrix."""
    from relgravpy._min_curvature_robust import (
        _mc_robust_observation_matrix, _mc_robust_difference
    )
    from scipy import sparse
    from scipy.sparse import linalg as spla

    nx=ny=21
    ix,iy=sparse.eye(nx,format="csr"),sparse.eye(ny,format="csr")
    d1x=_mc_robust_difference(nx,1)
    d1y=_mc_robust_difference(ny,1)
    d2x=_mc_robust_difference(nx,2)
    d2y=_mc_robust_difference(ny,2)
    x1=sparse.kron(iy,d1x,format="csr")
    y1=sparse.kron(d1y,ix,format="csr")
    x2=sparse.kron(iy,d2x,format="csr")
    y2=sparse.kron(d2y,ix,format="csr")
    xy=sparse.kron(d1y,d1x,format="csr")
    tension=0.15
    Q=((1-tension)*(x2.T@x2+2*xy.T@xy+y2.T@y2)
       +tension*(x1.T@x1+y1.T@y1))

    def condition_number(x,y):
        H=_mc_robust_observation_matrix(x,y,nx,ny,0,100,5,5)
        A=(1e4*H.T@H+Q+0.01*sparse.eye(nx*ny)).tocsc()
        lu=spla.splu(A)
        inverse=spla.LinearOperator(
            A.shape,matvec=lambda v:lu.solve(v),
            rmatvec=lambda v:lu.solve(v,trans="T"))
        return float(spla.onenormest(A)*spla.onenormest(inverse))

    centers=np.arange(5.,100.,10.)
    xc,yc=np.meshgrid(centers,centers)
    rng=np.random.default_rng(123)

    ratios=[]
    for jitter in (0.,0.5,2.):
        x=np.repeat(xc.ravel(),6)+rng.normal(0,jitter,xc.size*6)
        y=np.repeat(yc.ravel(),6)+rng.normal(0,jitter,yc.size*6)
        xb,yb,zb=mc.block_m(
            x,y,np.sin(x/30),wind_size=10,method="mean",
            lim=(0,100,0,100))
        raw_condition=condition_number(x,y)
        block_condition=condition_number(xb,yb)
        ratios.append(raw_condition/block_condition)
        print(f"jitter={jitter}: raw cond={raw_condition:.6g}, "
              f"block cond={block_condition:.6g}, "
              f"raw/block={ratios[-1]:.3f}")
    assert ratios[0]>3.0       # Repeated stations: averaging helps.
    assert ratios[2]<0.75      # Spatially spread: averaging can hurt.



# -----------------------------------------------------------------------------
# Gridded input with NaNs at the exterior must retain the ORIGINAL extent.
# -----------------------------------------------------------------------------
def test_all_nan_outer_border_does_not_shrink_region():
    x=np.arange(0.,51.,5.)
    y=np.arange(50.,-1.,-5.)
    X,Y=np.meshgrid(x,y)
    z=_truth(X,Y)
    z[0,:]=np.nan
    z[:,0]=np.nan
    gx,gy,grid,info=min_curvature(
        x,y,z,spacing=5,padding=2,solver="robust",
        return_info=True)
    assert grid.shape==(15,15)
    assert gx[0]==-10 and gx[-1]==60
    assert gy[0]==60 and gy[-1]==-10
    assert np.isfinite(grid).all()
    assert info["discarded_observations"]>=21


# -----------------------------------------------------------------------------
# On larger padded grids the AUTO backend selects CG and must converge.
# -----------------------------------------------------------------------------
def test_auto_sparse_cg_on_extended_padding_grid():
    x=np.arange(0.,61.,2.)
    y=x[::-1]
    X,Y=np.meshgrid(x,y)
    z=_truth(X,Y)
    z[10:19,12:20]=np.nan
    gx,gy,grid,info=min_curvature(
        x,y,z,spacing=2,padding=30,solver="robust",
        tension=0.25,anchor=0.05,return_info=True)
    assert grid.shape==(91,91)
    assert np.isfinite(grid).all()
    assert info["linear_solver"]=="cg"
    assert info["converged"]
    assert info["relative_system_residual"]<1e-6



# -----------------------------------------------------------------------------
# Real ERS rasters are often registered at non-integer projected origins.
# -----------------------------------------------------------------------------
def test_offset_projected_grid_origin_preserved():
    x=452904.697640 + np.arange(21)*400.
    y=6426297.029215 - np.arange(21)*400.
    X,Y=np.meshgrid(x,y)
    z=_truth((X-x[0])/400, (Y-y[-1])/400)
    z[9:12,8:13]=np.nan
    gx,gy,filled=min_curvature(
        x,y,z,spacing=400,solver="robust",anchor=0.01)
    assert filled.shape==(21,21)
    np.testing.assert_allclose(gx,x,rtol=0,atol=1e-8)
    np.testing.assert_allclose(gy,y,rtol=0,atol=1e-8)
    assert np.isfinite(filled).all()
    _,_,padded=min_curvature(
        x,y,z,spacing=400,solver="robust",padding=5)
    assert padded.shape==(31,31)
