"""Primary RelGravPy reference benchmark: independent spherical volume integral.

No Harmonica and no GDAL/OGR dependency. Integrates a uniform spherical
longitude-latitude wedge directly in spherical coordinates using convergent
Gauss-Legendre cubature; compares to a refined orthographic polyhedral mesh.

Usage
-----
python -u tests/benchmark_spherical_reference.py
"""
import json

import numpy as np

from test_bouguer_polyhedra import BC

RADIUS = 6_371_000.0
DENSITY = 2_670.0
THICKNESS = 300.0
CLEARANCE = 20.0
DOMAIN = 20_000.0
ANGLE = np.arcsin(DOMAIN / (2.0 * RADIUS))
OBS_Z = THICKNESS + CLEARANCE


def spherical_volume_gravity(nangular=96, nradial=32):
    """Down-positive gravitational attraction of a uniform spherical wedge, mGal.

    Four-fold angular symmetry, Gauss-Legendre quadrature in radius and
    angular coordinates. Cubic angular transformation concentrates nodes
    near the station. A haversine-equivalent distance formula avoids
    cancellation for the 20-m clearance.
    """
    x, w = np.polynomial.legendre.leggauss(nangular)
    x, w = 0.5 * (x + 1), 0.5 * w
    angles = ANGLE * x**3
    angle_weights = 3 * ANGLE * x*x * w
    lat = angles[:, None]
    lon = angles[None, :]
    q = 4 * (np.sin(lat / 2)**2
             + np.cos(lat) * np.sin(lon / 2)**2)
    area_weights = (4 * angle_weights[:, None] *
                    angle_weights[None, :] * np.cos(lat))
    xr, wr = np.polynomial.legendre.leggauss(nradial)
    radii = RADIUS + THICKNESS * 0.5 * (xr + 1)
    radial_weights = THICKNESS * 0.5 * wr
    ro = RADIUS + OBS_Z
    integral = 0.
    for rs, weight in zip(radii, radial_weights):
        delta = ro - rs
        dist_sq = delta**2 + ro * rs * q
        radial_proj = delta + rs*q/2
        integral += (weight * rs**2 *
                     np.sum(area_weights * radial_proj / dist_sq**1.5))
    return float(BC.gm.G * DENSITY * 1e5 * integral)


def mesh(n):
    edge = np.linspace(-DOMAIN/2, DOMAIN/2, n+1)
    west, south = np.meshgrid(edge[:-1], edge[:-1], indexing="xy")
    east, north = np.meshgrid(edge[1:], edge[1:], indexing="xy")
    cells = np.column_stack((
        west.ravel(), east.ravel(), south.ravel(), north.ravel(),
        np.zeros(n*n), np.full(n*n, THICKNESS)
    ))
    return cells, np.full(n*n, DENSITY)


def main():
    reference = spherical_volume_gravity(96, 32)
    convergence_check = spherical_volume_gravity(64, 24)
    reference_disagreement_ugal = 1000*(reference-convergence_check)
    print("PRIMARY_REFERENCE", json.dumps({
        "method": "independent_spherical_volume_quadrature",
        "density_kg_m3": DENSITY,
        "reference_radius_m": RADIUS,
        "angular_half_extent_deg": float(np.rad2deg(ANGLE)),
        "thickness_m": THICKNESS,
        "clearance_m": CLEARANCE,
        "reference_mGal": reference,
        "quadrature_difference_uGal": reference_disagreement_ugal
    }), flush=True)
    assert abs(reference_disagreement_ugal) < 0.001

    results = []
    for n in (4, 8, 16, 32, 64, 96, 128):
        prisms, densities = mesh(n)
        poly = BC._polyhedral_gz(
            prisms, densities, [0., 0., OBS_Z], RADIUS
        )
        row = dict(
            cells=n*n,
            cell_width_m=DOMAIN/n,
            poly_mGal=poly,
            reference_mGal=reference,
            error_uGal=1000*(poly-reference)
        )
        results.append(row)
        print("CASE", json.dumps(row), flush=True)
    assert abs(results[-1]["error_uGal"]) < 0.1
    assert abs(results[-1]["error_uGal"]) < abs(results[-2]["error_uGal"])
    print("BENCHMARK_PASS", json.dumps({
        "finest_error_uGal": results[-1]["error_uGal"],
        "target_uGal": 0.1
    }), flush=True)


if __name__ == "__main__":
    main()
