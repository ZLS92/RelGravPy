"""Independent spherical benchmark for R1 near/far resolution transition.

Near field uses narrow spherical polyhedral cells, far field coarse
polyhedral cells. Coarse R1 boundary cells are subdivided tangentially to
align the *entire seam boundary-node grid*. Compare with the converged
uniform spherical-volume Gauss-Legendre integral (not Harmonica).

Run: PYTHONPATH=tests:. python -u tests/benchmark_r1_transition.py
"""
import json
import numpy as np

from test_bouguer_polyhedra import BC, GM
from benchmark_spherical_reference import (
    spherical_volume_gravity, RADIUS, DOMAIN, DENSITY,
    THICKNESS, OBS_Z,
)


def square_cells(extent, n):
    edges = np.linspace(-extent, extent, n+1)
    west, south = np.meshgrid(edges[:-1], edges[:-1], indexing="xy")
    east, north = np.meshgrid(edges[1:], edges[1:], indexing="xy")
    cells = np.column_stack((west.ravel(), east.ravel(),
                             south.ravel(), north.ravel(),
                             np.zeros(n*n), np.full(n*n,THICKNESS)))
    return cells, edges


def spherical_wedge_reference(horizontal_half_width, rbottom, rtop,
                              station_height, nangles=96, nradial=32):
    """Independent high-order radial/lat/lon integral for one spherical wedge.

    The angular footprint is longitude/latitude bounded, not an exactly
    orthographic square. The <centimetre edge discrepancy is a separate
    small geometry approximation in these 20-km tests.
    """
    angle=np.arcsin(horizontal_half_width/RADIUS)
    x,w=np.polynomial.legendre.leggauss(nangles)
    x, w=(x+1)/2, w/2
    a=angle*x**3
    aw=w*3*angle*x*x
    lat,lon=a[:,None],a[None,:]
    q=4*(np.sin(lat/2)**2+np.cos(lat)*np.sin(lon/2)**2)
    sw=4*aw[:,None]*aw[None,:]*np.cos(lat)
    xr,wr=np.polynomial.legendre.leggauss(nradial)
    radii=RADIUS+rbottom+(rtop-rbottom)*(xr+1)/2
    radial_weights=(rtop-rbottom)*wr/2
    ro=RADIUS+station_height
    total=0.
    for rs,weight in zip(radii,radial_weights):
        delta=ro-rs
        dist_sq=delta*delta+ro*rs*q
        total+=weight*rs*rs*np.sum(sw*(delta+rs*q/2)/dist_sq**1.5)
    return BC.gm.G*DENSITY*1e5*float(total)


def run_case(near_n=80, far_n=64, R1=5000.0,
             near_top=THICKNESS, far_top=THICKNESS, station_height=None):
    # Fine mesh [-R1, R1]^2; coarse mesh [-10km,10km]^2.
    # For far_n=64: R1 is an exact coarse grid line (16 cells from centre).
    near, near_edges = square_cells(R1, near_n)
    all_far, far_edges = square_cells(DOMAIN/2, far_n)
    cx = (all_far[:,0]+all_far[:,1])/2
    cy = (all_far[:,2]+all_far[:,3])/2
    exterior = (abs(cx)>=R1) | (abs(cy)>=R1)
    far = all_far[exterior]
    near[:,5] = near_top
    far[:,5] = far_top
    if station_height is None:
        station_height = max(near_top,far_top)+20.
    observation=[0,0,station_height]
    far_dense = np.full(len(far),DENSITY)
    fine_dense = np.full(len(near),DENSITY)
    refined, refined_density = BC._refine_far_interface(
        far, far_dense, near_edges,near_edges,R1
    )
    conformed_near, conformed_near_den = BC._refine_near_interface(
        near, fine_dense, far_edges, far_edges, R1
    )

    # The new seam: all near cells spherical, and BOTH sides of R1
    # subdivided at the union of fine and coarse horizontal grid knots.
    near_g = BC._polyhedral_gz(
        conformed_near,conformed_near_den,observation,RADIUS
    )
    far_g = BC._polyhedral_gz(refined,refined_density,observation,RADIUS)
    unrefined_far = BC._polyhedral_gz(far,far_dense,observation,RADIUS)

    # Legacy prism approximation: near-field cell shifted by the farthest
    # horizontal corner; far-field spherical unchanged.
    shifted = near.copy()
    xymax = np.maximum(abs(near[:,0]),abs(near[:,1]))**2
    xymax += np.maximum(abs(near[:,2]),abs(near[:,3]))**2
    shifted[:,4:] -= (xymax/(2*RADIUS))[:,None]
    near_nagy = float(GM.nagy([observation],shifted,fine_dense).gz()[0])

    if near_top == far_top == THICKNESS and station_height == OBS_Z:
        exact = spherical_volume_gravity(96,32)
    else:
        exact = spherical_wedge_reference(DOMAIN/2,0,far_top,station_height)
        if near_top > far_top:
            exact += spherical_wedge_reference(R1,far_top,near_top,station_height)
        elif near_top < far_top:
            exact -= spherical_wedge_reference(R1,near_top,far_top,station_height)
    data = {
        "R1_m":R1,
        "near_top_m":near_top,
        "far_top_m":far_top,
        "station_height_m":station_height,
        "near_cells_original":len(near),
        "near_cells_refined":len(conformed_near),
        "far_cells_original":len(far),
        "far_cells_refined":len(refined),
        "near_cell_width_m":2*R1/near_n,
        "far_cell_width_m":DOMAIN/far_n,
        "reference_mGal":exact,
        "new_combined_mGal":near_g+far_g,
        "legacy_combined_mGal":near_nagy+unrefined_far,
        "uncorrected_spherical_combined_mGal":near_g+unrefined_far,
        "new_error_uGal":1000*(near_g+far_g-exact),
        "legacy_error_uGal":1000*(near_nagy+unrefined_far-exact),
        "coarse_interface_difference_uGal":1000*(far_g-unrefined_far)
    }
    print("R1_BENCHMARK",json.dumps(data),flush=True)
    assert abs(data["new_error_uGal"]) < 1.0, data
    return data


if __name__ == "__main__":
    run_case()
    run_case(near_top=350.,far_top=250.,station_height=370.)
