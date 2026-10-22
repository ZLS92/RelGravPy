"""Station-curve overlap registration for relative-gravity meter drift.

This is a *pre-adjustment* drift reduction: each repeatedly occupied station
supplies its own time-varying observation curve. Only their shared time windows
are used to register curves vertically. No gravity values or network closures
are adjusted and no polynomial model enters the curve-registration stage.

Assumptions: stable stations, previously corrected physical temporal signals,
no meter tares, and locally near-linear meter drift between repeat visits.
"""
import numpy as np


def grav_drift_overlap(stations, gobs, time_sec, *, deg=1,
                       ignore_st=None, tare=None, plot=False,
                       plt_datetitle="", return_details=False,
                       datetime_axis=True):
    """Fit drift after pairwise registration of overlapping station histories.

    Pairwise offsets are obtained from the time-average difference of the
    piecewise-linear interpolants restricted to their **common** observation
    intervals. The offsets are jointly reconciled on a connected overlap
    graph, avoiding order-dependent propagation of local registration errors.
    This graph fit is NOT a gravity-network adjustment: it uses no readings
    from different stations at separate times, and no absolute constraints.

    On disjoint overlap graphs, raise rather than extrapolate unobserved
    station curves into the gaps. Tares are explicitly unsupported.
    """
    if tare is not None and len(tare):
        raise ValueError(
            "alignment='overlap' currently does not support tares. "
            "Use alignment='sequential' or process stable tare-free sessions."
        )
    stations = np.asarray(stations)
    gobs = np.asarray(gobs, dtype=float)
    time_sec = np.asarray(time_sec, dtype=float)
    if any(v.ndim != 1 for v in (stations, gobs, time_sec)):
        raise ValueError("stations, gobs, and time must be one-dimensional")
    if len(stations) != len(gobs) or len(gobs) != len(time_sec):
        raise ValueError("station, observation and time lengths differ")
    if len(gobs) < 3 or not np.all(np.isfinite(gobs)) or not np.all(np.isfinite(time_sec)):
        raise ValueError("at least three finite observations are required")
    if not isinstance(deg, (int, np.integer)) or deg < 1:
        raise ValueError("polynomial degree must be a positive integer")

    idx_sort = np.argsort(time_sec, kind="stable")
    inv_sort = np.argsort(idx_sort)
    sts, values, times = stations[idx_sort], gobs[idx_sort], time_sec[idx_sort]
    ignored = set() if ignore_st is None else set(ignore_st)
    histories = []
    for station in dict.fromkeys(sts.tolist()):
        if station in ignored:
            continue
        idx = np.flatnonzero(sts == station)
        if len(idx) < 2:
            continue
        st_times = times[idx]
        st_values = values[idx]
        # Multiple readings at the same timestamp are averaged for the
        # interpolation knots, but retained individually in polynomial fit.
        unique_t, inv_t = np.unique(st_times, return_inverse=True)
        if len(unique_t) < 2:
            continue
        unique_v = np.bincount(inv_t, weights=st_values) / np.bincount(inv_t)
        histories.append({
            "station": station,
            "idx": idx,
            "time": unique_t,
            "value": unique_v,
        })
    if not histories:
        raise ValueError("no repeatedly measured station with distinct times")

    edges = []
    # Edge equation: shift[i] - shift[j] = mean(g_i(t) - g_j(t)).
    # Trapezoidal integration is exact for differences of linear interpolants.
    for i, first in enumerate(histories):
        for j in range(i + 1, len(histories)):
            second = histories[j]
            lo = max(first["time"][0], second["time"][0])
            hi = min(first["time"][-1], second["time"][-1])
            if not hi > lo:
                continue
            knots = np.union1d(
                np.concatenate(([lo, hi],
                    first["time"][(first["time"] > lo) & (first["time"] < hi)])),
                second["time"][(second["time"] > lo) & (second["time"] < hi)]
            )
            diffs = np.interp(knots, first["time"], first["value"])
            diffs -= np.interp(knots, second["time"], second["value"])
            edge_difference = float(
                np.sum(np.diff(knots) * (diffs[1:] + diffs[:-1]) / 2) / (hi - lo)
            )
            # This relative precision heuristic favors longer, more densely
            # sampled overlapping windows; not a formal measurement variance.
            span_a = float(np.median(np.diff(first["time"])))
            span_b = float(np.median(np.diff(second["time"])))
            weight = float((hi - lo) / max(0.5 * (span_a + span_b), 1.0))
            edges.append((i, j, edge_difference, max(weight, 1.e-8), lo, hi))

    n = len(histories)
    if n > 1:
        incidence = np.zeros((len(edges), n - 1), dtype=float)
        rhs = np.zeros(len(edges), dtype=float)
        weights = np.zeros(len(edges), dtype=float)
        for k, (i, j, difference, weight, _, _) in enumerate(edges):
            if i:
                incidence[k, i - 1] = 1.0
            if j:
                incidence[k, j - 1] = -1.0
            rhs[k] = difference
            weights[k] = np.sqrt(weight)
        if np.linalg.matrix_rank(incidence) != n - 1:
            raise ValueError(
                "Repeated-station time windows form a disconnected overlap "
                "graph; vertical registration would require extrapolation. "
                "Use the legacy sequential method or change the survey design."
            )
        relative_shifts, _, _, _ = np.linalg.lstsq(
            incidence * weights[:, None], rhs * weights, rcond=None
        )
        shifts = np.r_[0.0, relative_shifts]
        edge_residuals = incidence @ relative_shifts - rhs
    else:
        shifts = np.zeros(1)
        edge_residuals = np.zeros(0)

    # Fit one polynomial to vertically registered station histories.
    fit_idx = np.concatenate([hist["idx"] for hist in histories])
    fit_values = np.concatenate([
        values[hist["idx"]] - shift for hist, shift in zip(histories, shifts)
    ])
    if len(fit_idx) <= deg:
        raise ValueError("not enough repeated-station readings for degree")
    t0 = times[0]
    scale = max(float(times[-1] - t0), 1.0)
    xfit = (times[fit_idx] - t0) / scale
    design = np.polynomial.polynomial.polyvander(xfit, deg)
    if np.linalg.matrix_rank(design) < deg + 1:
        raise ValueError("rank-deficient polynomial; reduce degree")
    coeff, _, _, _ = np.linalg.lstsq(design, fit_values, rcond=None)
    xall = (times - t0) / scale
    curve = np.polynomial.polynomial.polyval(xall, coeff) - coeff[0]
    corrected = values - curve

    details = {
        "station": [h["station"] for h in histories],
        "registration_shift_mgal": shifts,
        "edge_station_pairs": [
            (histories[i]["station"], histories[j]["station"])
            for i, j, *_ in edges
        ],
        "edge_overlap_seconds": np.array([hi - lo for *_, lo, hi in edges]),
        "edge_registration_residual_mgal": edge_residuals,
        "polynomial_coeff_scaled_time": coeff[1:],
        "time_scale_seconds": scale,
        "fit_residual_mgal": fit_values - design @ coeff,
        "fit_observation_indices": idx_sort[fit_idx],
        "reference_time_sec": t0,
        "note": "Registration shifts are nuisance offsets, not adjusted station gravities.",
    }
    if plot:
        import matplotlib.pyplot as plt
        import datetime as dt
        if datetime_axis:
            xaxis = lambda t: [dt.datetime.fromtimestamp(
                float(v), tz=dt.timezone.utc) for v in t]
        else:
            xaxis = lambda t: t
        fig, ax = plt.subplots(figsize=(10, 6))
        for hist, shift in zip(histories, shifts):
            idx = hist["idx"]
            ax.plot(xaxis(times[idx]), values[idx] - shift - coeff[0],
                    "o-", label=str(hist["station"]))
        ax.plot(xaxis(times), curve, "k--", linewidth=2, label="Overlap drift")
        ax.set(xlabel="Datetime" if datetime_axis else "Time [s]",
               ylabel="Relative drift [mGal]",
               title=f"Drift curve {plt_datetitle}".strip())
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()

    if return_details:
        return corrected[inv_sort], curve[inv_sort], details
    return corrected[inv_sort], curve[inv_sort]
