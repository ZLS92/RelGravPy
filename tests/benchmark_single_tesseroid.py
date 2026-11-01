"""One-tesseroid reference vs refined curvature-following polyhedral mesh.

A constant-density spherical wedge (a *single* tesseroid) serves as a
fixed independent reference for every mesh resolution. One station is
located at the geographic centre, exactly 20 m above its top.

Run:
    python -u tests/benchmark_single_tesseroid.py

The orthographic rectilinear mesh is only an approximation to the
longitude/latitude boundaries: this is recorded as a systematic
footprint mismatch and must not be confused with the polygonal error.
"""
from __future__ import annotations

import json
from time import perf_counter
import math

import harmonica as hm
import numpy as np

from test_bouguer_polyhedra import BC

R = 6_371_000.0  # m
DENSITY = 2670.0  # kg/m^3
HEIGHT = 300.0  # top above reference sphere, m
CLEARANCE = 20.0  # elevation of station above top, m
DOMAIN = 20_000.0  # orthographic square extent, m
SCALES = (4, 8, 16, 32, 64, 96, 128)
ANGLE = np.rad2deg(np.arcsin(DOMAIN/(2.0*R)))
REF_TESS = np.array([[-ANGLE, ANGLE, -ANGLE, ANGLE, R, R + HEIGHT]], float)
OBS_CARTESIAN = np.array([0.0, 0.0, HEIGHT + CLEARANCE], float)
OBS_SPHERICAL = (0.0, 0.0, R + HEIGHT + CLEARANCE)


def generate_mesh(n: int):
    """Same orthographic domain at every n, with a fixed uniform density."""
    edges = np.linspace(-DOMAIN/2, DOMAIN/2, n+1, dtype=float)
    xw, ys = np.meshgrid(edges[:-1], edges[:-1], indexing="xy")
    xe, yn = np.meshgrid(edges[1:], edges[1:], indexing="xy")
    cells = np.column_stack([
        xw.ravel(), xe.ravel(), ys.ravel(), yn.ravel(),
        np.zeros(n*n), np.full(n*n, HEIGHT)
    ])
    return cells, np.full(n*n, DENSITY)


def compute_tesseroid(radial: bool):
    before = perf_counter()
    g = hm.tesseroid_gravity(
        OBS_SPHERICAL, REF_TESS, density=[DENSITY], field="g_z",
        radial_adaptive_discretization=radial,
        parallel=False
    )
    return float(np.asarray(g).ravel()[0]), perf_counter() - before


def tesseroid_vs_mesh():
    print("CONFIG", json.dumps({
        "reference": "single_tesseroid",
        "radius_m": R, "density_kg_m3": DENSITY,
        "height_m": HEIGHT, "clearance_m": CLEARANCE,
        "domain_m": DOMAIN,
        "tesseroid": REF_TESS.tolist()[0],
        "observation": list(OBS_SPHERICAL),
        "grid_geometry": "station-centered orthographic square",
    }), flush=True)

    # JIT warmup with a different, far source to avoid including compilation.
    warm = np.array([[0.1, 0.11, 0.1, 0.11, R, R+100]], float)
    hm.tesseroid_gravity((0, 0, R+1000), warm, density=[DENSITY],
                         field="g_z", parallel=False)
    reference, time_tess = compute_tesseroid(radial=False)
    ref_radial, time_radial = compute_tesseroid(radial=True)
    print("REFERENCE", json.dumps({
        "g_mGal": reference,
        "g_radial_adapt_mGal": ref_radial,
        "radial_mode_difference_uGal": 1000*(ref_radial-reference),
        "tess_s": time_tess, "tess_radial_s": time_radial,
    }), flush=True)

    output = []
    for n in SCALES:
        cells, density = generate_mesh(n)
        t0 = perf_counter()
        g = BC._polyhedral_gz(cells, density, OBS_CARTESIAN, R)
        elapsed = perf_counter()-t0
        result = {
            "n": n, "cells": n*n, "cell_width_m": DOMAIN/n,
            "g_poly_mGal": float(g),
            "g_one_tess_mGal": float(reference),
            "difference_uGal": 1000*(g-reference),
            "relative_difference_pct": 100*(g-reference)/reference,
            "elapsed_s": elapsed,
        }
        print("CASE", json.dumps(result), flush=True)
        output.append(result)

    # Diagnose the footprint mismatch independently of discretisation.
    # The side x=DOMAIN/2 maps to a longitude that varies with latitude,
    # while a true tesseroid has fixed longitude +/-ANGLE.
    # Its longitude side projects to x=R*cos(phi)*sin(ANGLE).
    lat_top = np.deg2rad(ANGLE)
    edge_offset = DOMAIN/2 - R*np.cos(lat_top)*np.sin(np.deg2rad(ANGLE))
    print("GEOMETRY", json.dumps({
        "max_eastwest_edge_offset_m": float(edge_offset),
        "note": "Nonzero asymptotic footprint mismatch: poly mesh is a square in orthographic x/y but tesseroid is bounded by constant longitude/latitude."
    }), flush=True)
    print("RESULT_JSON", json.dumps({
        "reference_mGal": reference,
        "reference_radial_mGal": ref_radial,
        "rows": output,
    }), flush=True)
    return output


if __name__ == "__main__":
    tesseroid_vs_mesh()
