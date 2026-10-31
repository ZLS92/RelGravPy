"""Synthetic spherical-terrain benchmark and R1-seam diagnostic.

Run with: python tests/benchmark_bouguer_polyhedra.py
The numerical kernels are imported using the AST-based test harness to
avoid optional GDAL/OGR dependencies when using synthetic height fields.
"""
import json
import math
import time

import harmonica as hm
import numpy as np

from test_bouguer_polyhedra import BC, GM


R = 6_371_000.0
DENSITY = 2670.0
OBS_HEIGHT = 700.0
DOMAIN = 20_000.0  # total width in metres, 20 km

def heightfield(x, y, scenario):
    if scenario == "flat":
        return np.full_like(x, 250.0, dtype=float)
    # Smooth non-negative rolling relief. ~125-475 m above reference sphere.
    return (295.0
            + 105.0 * np.sin(2 * np.pi * x / 8000) * np.cos(2 * np.pi * y / 13000)
            + 55.0 * np.exp(-((x - 3600)**2 + (y + 2100)**2) / (2 * 2000**2))
            + 35.0 * np.sin(2 * np.pi * (x + y) / 9000))


def grid(n, scenario):
    edges = np.linspace(-DOMAIN / 2, DOMAIN / 2, n + 1)
    xw, ys = np.meshgrid(edges[:-1], edges[:-1], indexing="xy")
    xe, yn = np.meshgrid(edges[1:], edges[1:], indexing="xy")
    xc = (xw + xe) / 2
    yc = (ys + yn) / 2
    top = heightfield(xc, yc, scenario).ravel()
    bottom = np.zeros_like(top)
    cells = np.column_stack((xw.ravel(), xe.ravel(), ys.ravel(), yn.ravel(), bottom, top))
    dens = np.full(len(cells), DENSITY, dtype=float)

    # Spherical tesseroid approximation for the SAME nominal orthographic
    # grid. At nonzero latitude there is a small boundary-parameterisation
    # mismatch (orthographic rectilinear x/y vs spherical lon/lat).
    # Unlike a direct numerical integral over exactly the same solid,
    # this is an independent, closely matching conventional reference.
    # At the tangent point: lon ~= asin(x/R), lat = asin(y/R).
    lon_w = np.rad2deg(np.arcsin(xw.ravel() / R))
    lon_e = np.rad2deg(np.arcsin(xe.ravel() / R))
    lat_s = np.rad2deg(np.arcsin(ys.ravel() / R))
    lat_n = np.rad2deg(np.arcsin(yn.ravel() / R))
    tess = np.column_stack((lon_w, lon_e, lat_s, lat_n,
                            R + bottom, R + top))
    return cells, tess, dens


def timed(call, repeat=2):
    result = None
    durations = []
    for i in range(repeat):
        t0 = time.perf_counter()
        result = call()
        durations.append(time.perf_counter() - t0)
    return result, min(durations)


def run_terrain_cases():
    output = []
    # Warm up Harmonica's compiled tesseroid integrator, so its JIT startup
    # is excluded from the measured runtime.
    warm = np.array([[0.1, 0.11, 0.1, 0.11, R, R + 100]])
    hm.tesseroid_gravity((0.0, 0.0, R + 700), warm, density=[2670.0],
                         field="g_z", parallel=False)
    for case in ("flat", "rugged"):
        for n in (8, 16, 32):
            prisms, tesseroids, densities = grid(n, case)
            poly_g, poly_s = timed(
                lambda: BC._polyhedral_gz(prisms, densities,
                                           [0, 0, OBS_HEIGHT], R), repeat=1)
            tess_g, tess_s = timed(
                lambda: float(np.asarray(hm.tesseroid_gravity(
                    (0.0, 0.0, R + OBS_HEIGHT), tesseroids,
                    density=densities, field="g_z",
                    parallel=False)).ravel()[0]), repeat=1)
            # For a non-curved comparator, use the exact Nagy sum at the
            # same nominal orthographic x-y grid.
            nag_g, nag_s = timed(
                lambda: float(GM.nagy([[0, 0, OBS_HEIGHT]],
                                      prisms, densities).gz()[0]), repeat=1)
            row = dict(
                scenario=case,
                n=n,
                cells=n*n,
                cell_width_m=DOMAIN/n,
                poly_mGal=round(poly_g, 8),
                harmonica_mGal=round(tess_g, 8),
                nagy_flat_mGal=round(nag_g, 8),
                difference_uGal=round(1000*(poly_g-tess_g), 5),
                poly_s=round(poly_s, 4),
                harmonica_s=round(tess_s, 4),
                nagy_s=round(nag_s, 4),
                poly_per_cell_ms=round(1000*poly_s/(n*n), 4),
            )
            print("CASE", json.dumps(row), flush=True)
            output.append(row)
    return output


def seam_diagnostic():
    # Near grid: 250-m cell; far grid: 1000-m cell, both aligned at R1.
    # A near prism's z_shift in the legacy implementation is the sagitta
    # evaluated at the most distant corner of each cell.
    R1 = 8000.0
    gs1, gs2 = 250., 1000.
    yvals = np.linspace(-R1 + gs1/2, R1 - gs1/2, 64)
    gaps = []
    for ycenter in yvals:
        yfar = math.copysign(abs(ycenter) + gs1/2, ycenter)
        near_b = -(R1**2 + yfar**2) / (2*R)
        exact_at_seam = math.sqrt(R**2 - R1**2 - ycenter**2) - R
        gaps.append(near_b-exact_at_seam)
    gap = np.array(gaps)
    print("SEAM", json.dumps({
        "top_radial_horizontal_offset_m_at_300m": round(R1*300.0/R, 6),
        "R1_m": R1,
        "near_step_m": gs1,
        "far_step_m": gs2,
        "height_gap_min_m": round(float(np.min(gap)), 6),
        "height_gap_max_m": round(float(np.max(gap)), 6),
        "max_abs_height_gap_m": round(float(np.max(abs(gap))), 6),
    }), flush=True)

    # Gravitational diagnostic for two adjacent cells only. This is NOT a
    # fully sampled DEM; use it to isolate geometry effects at one seam.
    near = np.array([[R1-gs1, R1, -gs1/2, gs1/2, 0, 300.0]])
    far = np.array([[R1, R1+gs2, -gs2/2, gs2/2, 0, 300.0]])
    # Corresponding old near-field shift at its farthest corner:
    sag = (R1**2 + (gs1/2)**2)/(2*R)
    near_legacy = near.copy()
    near_legacy[:,4:] -= sag
    obs = [[0, 0, 700]]
    near_g = float(GM.nagy(obs, near_legacy, [DENSITY]).gz()[0])
    far_g = BC._polyhedral_gz(far, [DENSITY], obs[0], R)
    exact_near_g = BC._polyhedral_gz(near, [DENSITY], obs[0], R)
    mismatch_uGal = 1000*(near_g-exact_near_g)
    result = dict(
        near_prism_mGal=round(near_g, 8),
        near_spherical_mGal=round(exact_near_g, 8),
        near_difference_uGal=round(mismatch_uGal, 6),
        far_poly_mGal=round(far_g, 8),
        combined_mGal=round(near_g + far_g, 8),
    )
    print("SEAM_GRAVITY", json.dumps(result), flush=True)
    assert abs(result["near_difference_uGal"]) < 0.01, result
    return result


if __name__ == "__main__":
    seam_diagnostic()
    rows=run_terrain_cases()
    for row in rows:
        # These are cross-geometry benchmarks, not same-volume equalities.
        tolerance = 9.0 if row["n"] == 8 else (5.0 if row["n"] == 16 else 3.0)
        assert abs(row["difference_uGal"]) < tolerance, row
    print("RESULT_JSON", json.dumps(rows), flush=True)
