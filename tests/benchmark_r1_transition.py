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


def run_case(near_n=80, far_n=64, R1=5000.0):
    # Fine mesh [-R1, R1]^2; coarse mesh [-10km,10km]^2.
    # For far_n=64: R1 is an exact coarse grid line (16 cells from centre).
    near, near_edges = square_cells(R1, near_n)
    all_far, far_edges = square_cells(DOMAIN/2, far_n)
    cx = (all_far[:,0]+all_far[:,1])/2
    cy = (all_far[:,2]+all_far[:,3])/2
    exterior = (abs(cx)>=R1) | (abs(cy)>=R1)
    far = all_far[exterior]
    far_dense = np.full(len(far),DENSITY)
    fine_dense = np.full(len(near),DENSITY)
    refined, refined_density = BC._refine_far_interface(
        far, far_dense, near_edges,near_edges,R1
    )

    # The new seam: all near cells spherical, including the R1 boundary.
    near_g = BC._polyhedral_gz(near,fine_dense,[0,0,OBS_Z],RADIUS)
    far_g = BC._polyhedral_gz(refined,refined_density,[0,0,OBS_Z],RADIUS)
    unrefined_far = BC._polyhedral_gz(far,far_dense,[0,0,OBS_Z],RADIUS)

    # Legacy prism approximation: near-field cell shifted by the farthest
    # horizontal corner; far-field spherical unchanged.
    shifted = near.copy()
    xymax = np.maximum(abs(near[:,0]),abs(near[:,1]))**2
    xymax += np.maximum(abs(near[:,2]),abs(near[:,3]))**2
    shifted[:,4:] -= (xymax/(2*RADIUS))[:,None]
    near_nagy = float(GM.nagy([[0,0,OBS_Z]],shifted,fine_dense).gz()[0])

    exact = spherical_volume_gravity(96,32)
    data = {
        "R1_m":R1,
        "near_cells":len(near),
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
