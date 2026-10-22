"""Compare a TSoft theoretical gravity export with RelGravPy Tamura T87.

Example (CSV contains UTC datetime and tide in nm/s^2):
    python examples/compare_tsoft_t87.py tsoft.csv --lat 49.00937 \
        --lon 8.40444 --height 120 --wavegroups tsoft_wavegroups.txt

CSV headers: utc,tsoft_nm_s2
The optional wavegroup file uses the first four columns in the TSoft
Theotide | Edit parameters dialog: minimum_cpd maximum_cpd delta phase_lead.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from relgravpy.grav_processing import earth_tides_tsoft


def compare_tsoft(csv_path, lat, lon, height, wavegroups=None, plot=False):
    times, observed = [], []
    with Path(csv_path).open(newline="") as f:
        for row in csv.DictReader(f):
            times.append(row["utc"])
            observed.append(float(row["tsoft_nm_s2"]))
    if not times:
        raise ValueError("No TSoft records found")
    times = np.array(times)
    observed = np.array(observed)
    groups = None
    if wavegroups is not None:
        groups = np.loadtxt(wavegroups, usecols=(0,1,2,3), ndmin=2)
    predicted = earth_tides_tsoft(lat,lon,height,DateTime=times,
                                  wavegroups=groups,unit="nm/s2")
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(mask):
        raise ValueError("No valid measurements")
    delta = predicted[mask] - observed[mask]
    result = {
        "samples": int(np.count_nonzero(mask)),
        "bias_uGal": float(np.mean(delta)/10.),
        "rmse_uGal": float(np.sqrt(np.mean(delta**2))/10.),
        "max_abs_uGal": float(np.max(np.abs(delta))/10.),
    }
    if len(delta)>1 and np.std(observed[mask])>0 and np.std(predicted[mask])>0:
        result["r"] = float(np.corrcoef(observed[mask],predicted[mask])[0,1])
    for key,value in result.items():
        print(f"{key}: {value}")
    if plot:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,sharex=True,figsize=(10,6))
        axes[0].plot(times,observed,label="TSoft",lw=1.3)
        axes[0].plot(times,predicted,label="RelGravPy T87",lw=1)
        axes[0].set_ylabel("nm/s²")
        axes[0].legend()
        axes[1].plot(times,(predicted-observed)/10.,label="RelGravPy – TSoft")
        axes[1].set_ylabel("Difference (µGal)")
        axes[1].set_xlabel("UTC")
        fig.autofmt_xdate()
        plt.show()
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--lat",type=float,required=True)
    parser.add_argument("--lon",type=float,required=True)
    parser.add_argument("--height",type=float,default=0.)
    parser.add_argument("--wavegroups",default=None)
    parser.add_argument("--plot",action="store_true")
    kwargs=vars(parser.parse_args())
    kwargs["csv_path"]=kwargs.pop("csv_path")
    compare_tsoft(**kwargs)
