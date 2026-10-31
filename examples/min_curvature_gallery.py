# -*- coding: utf-8 -*-
"""Reproduce synthetic minimum-curvature figures using RelGravPy.

Usage:
    python examples/min_curvature_gallery.py

Figures are saved to ./min_curvature_gallery/. This is a numerical
demonstration, not a geophysical validation on field data.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

from relgravpy import utils as utl


# -----------------------------------------------------------------------------
def synthetic_field(x, y):
    """Known analytical synthetic anomaly (arbitrary field units)."""
    return (2.5*np.exp(-((x-38)**2+(y-61)**2)/310)
            -1.2*np.exp(-((x-70)**2+(y-29)**2)/190)
            +0.015*x-0.02*y)


# -----------------------------------------------------------------------------
def build_station_scenarios(seed=2026):
    """Create four repeatable irregular station geometries."""
    rng=np.random.default_rng(seed)
    layouts=[]
    layouts.append(("Random coverage",rng.uniform(1,99,180),
                    rng.uniform(1,99,180)))
    layouts.append(("Central cluster and extrapolation",rng.uniform(29,71,85),
                    rng.uniform(29,71,85)))
    xl=np.concatenate([np.linspace(5,95,28) for _ in range(5)])
    yl=np.concatenate([np.full(28,yy)+rng.normal(0,.55,28)
                       for yy in (15,30,45,70,85)])
    layouts.append(("Parallel survey profiles",xl,yl))
    x=rng.uniform(1,99,650)
    y=rng.uniform(1,99,650)
    hole=((x-52)/23)**2+((y-53)/24)**2<1
    layouts.append(("Missing central area",x[~hole],y[~hole]))
    return layouts


# -----------------------------------------------------------------------------
def plot_scenarios(output_folder):
    """Plot sampling, reconstructed surface and reconstruction error."""
    fig,axes=plt.subplots(4,3,figsize=(14.4,17),layout="constrained")
    results=[]
    for row,(name,x,y) in enumerate(build_station_scenarios()):
        z=synthetic_field(x,y)
        gx,gy,grid,info=utl.min_curvature(
            x,y,z,region=(0,100,0,100),spacing=2,
            tension=0,anchor=0.01,return_info=True)
        X,Y=np.meshgrid(gx,gy)
        exact=synthetic_field(X,Y)
        difference=grid-exact
        inside=Delaunay(np.column_stack((x,y))).find_simplex(
            np.column_stack((X.ravel(),Y.ravel())))>=0
        inside=inside.reshape(X.shape)
        rmse_inside=np.sqrt(np.mean(difference[inside]**2))
        rmse_outside=np.sqrt(np.mean(difference[~inside]**2))
        results.append((name,len(x),rmse_inside,rmse_outside))
        color=dict(origin="upper",extent=(0,100,0,100),
                   cmap="coolwarm",vmin=-1.9,vmax=2.7)
        axes[row,0].imshow(exact,**color)
        axes[row,0].scatter(x,y,c="k",s=6,alpha=.7)
        axes[row,0].set_title(f"{name} (n={len(x)})")
        axes[row,1].imshow(grid,**color)
        hull=np.column_stack((x,y))[ConvexHull(np.column_stack((x,y))).vertices]
        axes[row,1].plot(np.r_[hull[:,0],hull[0,0]],
                         np.r_[hull[:,1],hull[0,1]],"k--",lw=1)
        axes[row,1].set_title("Robust minimum curvature\n(dashed: convex hull)")
        axes[row,2].imshow(difference,origin="upper",extent=(0,100,0,100),
                           cmap="RdBu_r",vmin=-1.5,vmax=1.5)
        axes[row,2].set_title(
            f"Error: inside {rmse_inside:.3f} | outside {rmse_outside:.3f}")
        for ax in axes[row]:
            ax.set(xlim=(0,100),ylim=(0,100),xlabel="x",ylabel="y")
            ax.set_aspect("equal")
    fig.suptitle("Sparse survey layouts: minimum curvature and error",
                 fontsize=16,fontweight="bold")
    fig.savefig(output_folder/"min_curvature_scenarios.png",dpi=160)
    plt.close(fig)
    return results


# -----------------------------------------------------------------------------
def plot_padding(output_folder):
    """Show filling NaN holes and expanding the original rectangular grid."""
    x=np.arange(0.,101.,2.)
    y=x[::-1]
    X,Y=np.meshgrid(x,y)
    observed=synthetic_field(X,Y)
    missing=((X-45)**2+(Y-55)**2<21**2)
    observed[missing]=np.nan
    _,_,filled=utl.min_curvature(
        x,y,observed,spacing=2,anchor=0.01)
    xx,yy,padded=utl.min_curvature(
        x,y,observed,spacing=2,padding=20,anchor=0.01)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    for ax,field,bounds,title in (
        (axes[0],observed,(0,100,0,100),"Original grid with NaNs"),
        (axes[1],filled,(0,100,0,100),"Filled grid"),
        (axes[2],padded,(xx[0],xx[-1],yy[-1],yy[0]),"20-cell padding"),
    ):
        ax.imshow(field,origin="upper",extent=bounds,
                  cmap="coolwarm",vmin=-2,vmax=3)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set(xlabel="x",ylabel="y")
    axes[2].plot([0,100,100,0,0],[0,0,100,100,0],"k--")
    fig.savefig(output_folder/"min_curvature_padding.png",dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
def main():
    output=Path("min_curvature_gallery")
    output.mkdir(exist_ok=True)
    for name,n,err_in,err_out in plot_scenarios(output):
        print(f"{name}: n={n}, RMSE inside={err_in:.5f}, "
              f"outside={err_out:.5f}")
    plot_padding(output)
    print(f"Figures written to {output.resolve()}")


if __name__=="__main__":
    main()
