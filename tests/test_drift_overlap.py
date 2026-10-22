"""Tests for non-sequential overlap registration of station drift curves."""
import numpy as np
import pytest
from relgravpy.drift_alignment import grav_drift_overlap


def synthetic_linear():
    # Every station has an independent, unknown level and repeat occupations.
    stations = np.array(["A", "B", "C", "A", "B", "C",
                         "Q", "A", "B", "C"])
    hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    station_levels = {"A": 1000.0, "B": 1050.0, "C": 990.0, "Q": 22.0}
    gobs = np.array([station_levels[s] for s in stations]) + 0.015 * hours
    return stations, hours * 3600, gobs


def test_linear_drift_exact_and_singleton_corrected():
    st, time, g = synthetic_linear()
    corrected, drift, details = grav_drift_overlap(
        st, g, time, deg=1, return_details=True)
    assert np.allclose(drift, 0.015 * time / 3600, atol=1e-10)
    for s in ("A", "B", "C", "Q"):
        np.testing.assert_allclose(corrected[st == s],
                                   corrected[st == s][0], atol=1e-10)
    assert len(details["edge_station_pairs"]) == 3
    assert np.max(np.abs(details["edge_registration_residual_mgal"])) < 1e-10


def test_order_independence_and_original_row_order():
    st, time, g = synthetic_linear()
    random_order = np.array([7, 0, 3, 9, 2, 6, 1, 5, 4, 8])
    _, straight = grav_drift_overlap(st, g, time)
    corrected, permuted = grav_drift_overlap(
        st[random_order], g[random_order], time[random_order])
    np.testing.assert_allclose(permuted, straight[random_order], atol=1e-10)
    np.testing.assert_allclose(corrected, (g - straight)[random_order], atol=1e-10)


def test_temporal_overlap_chain():
    # A overlaps B, B overlaps C, but A does not overlap C.
    stations = np.array(["A", "B", "A", "B", "C", "B", "C"])
    h = np.array([0, 3, 4, 6, 7, 8, 11], float)
    levels = {"A": 150., "B": 60., "C": 999.}
    values = np.array([levels[s] for s in stations]) + h * 0.021
    _, drift, details = grav_drift_overlap(
        stations, values, h * 3600, return_details=True)
    np.testing.assert_allclose(drift, 0.021 * h, atol=1e-10)
    assert len(details["edge_station_pairs"]) == 2


def test_disconnected_overlap_rejected_instead_of_extrapolation():
    stations = np.array(["A", "A", "B", "B"])
    times = np.array([0, 1, 5, 6]) * 3600.
    with pytest.raises(ValueError, match="disconnected overlap"):
        grav_drift_overlap(stations, np.array([11., 11.1, 80., 80.1]), times)


def test_tare_rejected_explicitly():
    st, times, g = synthetic_linear()
    with pytest.raises(ValueError, match="does not support tares"):
        grav_drift_overlap(st, g, times, tare=["2026-10-20"])


def test_quadratic_drift_is_reasonably_recovered():
    st, times, g = synthetic_linear()
    hours = times / 3600
    g = g + 0.0004 * hours**2
    _, drift = grav_drift_overlap(st, g, times, deg=2)
    exact = 0.015 * hours + 0.0004 * hours**2
    assert np.sqrt(np.mean((drift - exact)**2)) < 0.005


def test_one_repeated_station_needs_no_overlap_graph():
    st = np.array(["A", "B", "A", "A"])
    t = np.array([0, 1, 2, 4], float) * 3600
    g = np.array([100., 850., 100.02, 100.04])
    corrected, drift = grav_drift_overlap(st, g, t)
    np.testing.assert_allclose(drift, np.array([0., .01, .02, .04]), atol=1e-10)
    assert abs(corrected[1] - 849.99) < 1e-10


