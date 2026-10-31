# -*- coding: utf-8 -*-
"""
Published-data benchmark for RelGravPy's minimum-curvature grid filling.

REFERENCE DATA
--------------
South Australia Government, Geological Survey of South Australia:
SA GRAVITY 2016 Bouguer anomaly compilation, Eyre Peninsula extract.

Official metadata:
https://pid.sarig.sa.gov.au/dataset/mesac4462

The numerical ER Mapper grid is a public extract in the open research repo:
https://github.com/Adelaide-Earth-Imaging/potential-fields/tree/main/data
  eyre_grav       : little-endian float32 pixels, top row first
  eyre_grav.ers   : header with georeferencing, nodata and units

Source grid: 400m cell size, GDA94 MGA53 (EPSG:28353), units mGal.
A full-reference subwindow is withheld in controlled experiments. It must
not be described as ground truth of the actual gravity field: it is the
independent pre-existing *published gridded product*.

Usage (from project root):
    PYTHONPATH=tests:. python examples/benchmark_published_gravity.py

Writes a small reference subset (NPZ), validation CSV, figures and metadata.
Requires NumPy, SciPy, Matplotlib. Does not require GMT or GDAL.
"""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
import sys
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

# Isolate the existing utils.py public function from legacy optional GIS
# imports; uses the exact in-repository implementation tested by CI.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests"))
sys.path.insert(0, str(_REPO))
from test_min_curvature_robust import min_curvature  # noqa: E402

_SOURCE = (
    "https://raw.githubusercontent.com/"
    "Adelaide-Earth-Imaging/potential-fields/main/data/"
)
_NROWS = 743
_NCOLS = 730
_STEP = 400.0
_X0 = 452904.697640
_Y0 = 6426297.029215
_NODATA = -99999.0
_GRID_SIZE = 67
_BORDER = 8
_OUTPUT = _REPO / "gravity_reference_benchmark"


# -----------------------------------------------------------------------------
def download_reference(folder: Path):
    """Fetch original float32 ERS raster and verify its size/metadata.

    Returns
    -------
    ndarray
        Top-to-bottom (743, 730) grid of Bouguer anomaly [mGal].
    str
        SHA256 of the original binary raster, for traceability.
    """
    folder.mkdir(exist_ok=True, parents=True)
    raster_path = folder / "eyre_grav"
    header_path = folder / "eyre_grav.ers"
    for path in (raster_path, header_path):
        if not path.is_file():
            with urllib.request.urlopen(_SOURCE + path.name, timeout=90) as r:
                path.write_bytes(r.read())
    header = header_path.read_text(encoding="utf-8")
    for expected in ("NrOfLines\t= 743", "NrOfCellsPerLine\t= 730",
                     "IEEE4ByteReal", "Xdimension\t= 400",
                     "Ydimension\t= 400", "mGal"):
        assert expected in header, f"Unexpected ERS header: {expected}"
    assert raster_path.stat().st_size == _NROWS*_NCOLS*4
    digest = hashlib.sha256(raster_path.read_bytes()).hexdigest()
    data = np.fromfile(raster_path, dtype="<f4").reshape(_NROWS, _NCOLS)
    data = data.astype(np.float64)
    data[(data == _NODATA) | (~np.isfinite(data))] = np.nan
    assert np.isfinite(data).any()
    return data, digest


# -----------------------------------------------------------------------------
def choose_reference_window(data: np.ndarray):
    """Locate a finite, geologically variable 67x67 subset reproducibly."""
    w = _GRID_SIZE
    candidates = []
    # Do not select patches at the file edge; use candidates from the land.
    for r in range(16, data.shape[0]-w-16, 11):
        for c in range(16, data.shape[1]-w-16, 11):
            patch = data[r:r+w, c:c+w]
            if np.isfinite(patch).all():
                spread = float(np.std(patch))
                candidates.append((spread, r, c))
    if not candidates:
        raise RuntimeError("No entirely finite reference grid window found")
    spread, row, col = max(candidates)
    return data[row:row+w, col:col+w].copy(), row, col, spread


# -----------------------------------------------------------------------------
def metrics(reference, reconstruction, selected):
    """Evaluate error only at withheld nodes, not reused observations."""
    difference = reconstruction[selected] - reference[selected]
    return {
        "withheld": int(selected.sum()),
        "rmse_mgal": float(np.sqrt(np.mean(difference**2))),
        "mae_mgal": float(np.mean(np.abs(difference))),
        "maxabs_mgal": float(np.max(np.abs(difference))),
    }


# -----------------------------------------------------------------------------
def benchmark(reference, source_row, source_col, folder, sha256):
    """Evaluate random holes, internal gaps, survey profiles, and padding."""
    n = _GRID_SIZE
    x = _X0 + (source_col + np.arange(n)) * _STEP
    y = _Y0 - (source_row + np.arange(n)) * _STEP
    core = reference[_BORDER:-_BORDER, _BORDER:-_BORDER]
    xc, yc = x[_BORDER:-_BORDER], y[_BORDER:-_BORDER]
    rng = np.random.default_rng(20261031)
    scenarios = {}

    # Strictly withheld reference cells for all tests.
    selected = np.ones(core.shape, dtype=bool)
    selected &= rng.random(core.shape) < 0.3
    scenarios["random_30pct"] = (~selected, selected)
    selected = np.zeros(core.shape, dtype=bool)
    selected[17:34, 17:34] = True
    scenarios["central_17x17"] = (~selected, selected)
    selected = np.ones(core.shape, dtype=bool)
    selected[::8, :] = False
    scenarios["parallel_profiles"] = (~selected, selected)
    selected = np.ones(core.shape, dtype=bool)
    selected[14:37, 14:37] = False
    scenarios["central_cluster"] = (~selected, selected)

    rows = []
    results = []
    for name, (keep, withheld) in scenarios.items():
        incomplete = core.copy()
        incomplete[~keep] = np.nan
        gx, gy, calculated, info = min_curvature(
            xc, yc, incomplete, spacing=_STEP, tension=0.15, anchor=0.01, data_weight=1e4,
            return_info=True,
        )
        assert calculated.shape == core.shape
        assert np.isfinite(calculated).all()
        assert info["converged"]
        observed = np.where(keep, core, np.nan)
        stats = metrics(core, calculated, withheld)
        rows.append(dict(scenario=name, observations=int(keep.sum()),
                         **stats))
        results.append((name, observed, calculated, calculated-core,
                        withheld, stats))

    # Padding: train on the 51x51 central subwindow, recover a 67x67 field.
    xpad, ypad, padded, diag = min_curvature(
        xc, yc, core, spacing=_STEP,
        padding=_BORDER,
        tension=0.15, anchor=0.01,
        data_weight=1e4, return_info=True,
    )
    assert padded.shape == reference.shape
    assert np.isfinite(padded).all()
    assert np.allclose(xpad, x) and np.allclose(ypad, y)
    outside = np.ones(reference.shape, dtype=bool)
    outside[_BORDER:-_BORDER, _BORDER:-_BORDER] = False
    padding_error = metrics(reference, padded, outside)
    rows.append(dict(scenario="padding_8cells",observations=int(core.size),
                     **padding_error))

    # -------------------------------------------------------------------------
    # Original numerical reference + reconstruction in an exchangeable format.
    # -------------------------------------------------------------------------
    np.savez_compressed(folder/"SA_GRAVITY_2016_eyre_reference_patch.npz",
        x_m=x, y_m=y, bouguer_mgal=reference.astype("f4"),
        padded_mgal=padded.astype("f4"), reference_crs="EPSG:28353",
        origin="SA GRAVITY 2016 - Eyre Peninsula",
        original_binary_sha256=sha256,
        source_url=_SOURCE+"eyre_grav",
        source_window_row=source_row,
        source_window_col=source_col)

    with (folder/"validation_metrics.csv").open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # -------------------------------------------------------------------------
    # Overview figure for real published gravity values.
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(2,3,figsize=(13.5,8),
                            constrained_layout=True)
    vmin,vmax=np.percentile(reference,[2,98])
    diff=padded-reference
    panels=[
        (reference,"Original SA GRAVITY 2016", "turbo",vmin,vmax),
        (core,"Original 51x51 interior", "turbo",vmin,vmax),
        (padded,"Reconstructed 67x67 (8-cell pad)", "turbo",vmin,vmax),
        (np.where(outside,reference,np.nan),"Published reference at pad", "turbo",vmin,vmax),
        (np.where(outside,padded,np.nan),"Predicted values in padded ring", "turbo",vmin,vmax),
        (np.where(outside,diff,np.nan),"Padding error (mGal)", "RdBu_r",
         -max(0.1,np.nanpercentile(np.abs(diff[outside]),98)),
          max(0.1,np.nanpercentile(np.abs(diff[outside]),98))),
    ]
    for ax,(arr,title,cmap,lo,hi) in zip(axs.flat,panels):
        im=ax.imshow(arr,cmap=cmap,vmin=lo,vmax=hi,origin="upper")
        ax.set_title(title,fontsize=10)
        fig.colorbar(im,ax=ax,shrink=.77)
    fig.suptitle("Measured Bouguer gravity grid (mGal): padding benchmark\n"
                 "SA GRAVITY 2016, Eyre Peninsula, cell 400m, EPSG:28353")
    fig.savefig(folder/"published_gravity_padding.png",dpi=160)
    plt.close(fig)

    fig,axs=plt.subplots(4,3,figsize=(12,14),constrained_layout=True)
    for axrow,(name,observed,recovered,difference,withheld,stats) in zip(axs,results):
        titles=(
            f"{name}: retained {np.isfinite(observed).sum()}",
            "Reconstructed Bouguer (mGal)",
            f"Difference: withheld RMSE {stats['rmse_mgal']:.3f} mGal",
        )
        for ax,field,title in zip(axrow,(observed,recovered,difference),titles):
            if title.startswith("Difference"):
                err= max(0.1,np.percentile(np.abs(difference[withheld]),98))
                im=ax.imshow(field,cmap="RdBu_r",vmin=-err,vmax=err)
            else:
                im=ax.imshow(field,cmap="turbo",vmin=vmin,vmax=vmax)
            ax.set_title(title,fontsize=10)
            fig.colorbar(im,ax=ax,shrink=.76)
    fig.suptitle("Published Bouguer grid: artificially withheld data and recovery")
    fig.savefig(folder/"published_gravity_gap_recovery.png",dpi=160)
    plt.close(fig)

    summary=(
        "# Published gravity reference benchmark\n\n"
        "Original: SA GRAVITY 2016 Bouguer gravity (South Australian Government).\n"
        "Extract from Adelaide-Earth-Imaging/potential-fields/data/eyre_grav.\n"
        "Source CRS EPSG:28353; sample spacing 400 m; unit mGal.\n"
        f"Window top-left raster (row,col): ({source_row}, {source_col}).\n"
        f"Original binary SHA256: {sha256}.\n\n"
        "Values compared against the **published numerical grid**, not\n"
        "independent field observations or the true physical gravity field.\n"
        "Artificially removed cells are NOT used as solver constraints.\n\n"
        "| Scenario | retained | withheld | RMSE (mGal) | MAE (mGal) | max (mGal) |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "".join(
            f"| {r['scenario']} | {r['observations']} | {r['withheld']} | "
            f"{r['rmse_mgal']:.4f} | {r['mae_mgal']:.4f} | "
            f"{r['maxabs_mgal']:.4f} |\n" for r in rows)
        + "\nThe extrapolated padding is poorly constrained relative to interior\n"
          "holes; tuning regularization and comparing multiple pad sizes is essential.\n")
    (folder/"REPORT.md").write_text(summary,encoding="utf-8")
    print(summary)
    return rows


# -----------------------------------------------------------------------------
def main():
    data,digest=download_reference(_OUTPUT)
    patch,r,c,std=choose_reference_window(data)
    print(f"Loaded real grid: shape={data.shape}, finite="
          f"{np.isfinite(data).sum()}, SHA256={digest}")
    print(f"Reference 67x67 window: row={r}, col={c}, std={std:.4f} mGal")
    rows=benchmark(patch,r,c,_OUTPUT,digest)
    print("Files saved in:",_OUTPUT.resolve())
    assert all(np.isfinite(row["rmse_mgal"]) for row in rows)


if __name__ == "__main__":
    main()
