"""Report error statistics for unified polydron_grav against Nagy over 100 cases.

Usage: PYTHONPATH=tests:. python -u tests/benchmark_polygrav_nagy.py
"""
import json
import numpy as np
from test_bouguer_polyhedra import GM

def main():
    results=[]
    for length,thickness in ((20.,10.),(200.,100.),
                             (1000.,300.),(5000.,1500.),(10000.,200.)):
        prism=np.array([[-length/2,length/2,-length/2,length/2,-thickness,0.]])
        corners,faces=GM.prism_to_corners_faces(prism[0])
        for density in (0.,1000.,2670.,-1800.):
            solver=GM.polydron_grav(corners,faces,density)
            for factor in (0.,0.8,2.,8.,30.):
                station=np.array([length*factor,-length*factor*0.21,30.])
                g=solver.compute_gz(station)
                ref=float(GM.prism_grav([station],prism,density).compute_gz()[0])
                diff=1000.*(g-ref)
                results.append(dict(
                    prism_width_m=length,thickness_m=thickness,
                    distance_factor=factor,density_kg_m3=density,
                    poly_mGal=float(g),nagy_mGal=ref,
                    diff_uGal=float(diff),
                    rel_pct=float(100*diff/(1000*ref)) if abs(ref)>1e-10 else None))
    nonzero=[r for r in results if abs(r["nagy_mGal"])>1e-7]
    worst_u=max(results,key=lambda r:abs(r["diff_uGal"]))
    worst_pct=max(nonzero,key=lambda r:abs(r["rel_pct"]))
    report={
        "cases":len(results),
        "max_absolute_error_uGal":max(abs(r["diff_uGal"]) for r in results),
        "max_relative_error_percent_for_nonzero_signals":
            max(abs(r["rel_pct"]) for r in nonzero),
        "worst_absolute_case":worst_u,
        "worst_relative_case":worst_pct,
    }
    print("POLYGRAV_NAGY_BENCHMARK",json.dumps(report),flush=True)
    assert all(abs(r["poly_mGal"]-r["nagy_mGal"])
               <= 2e-6+1.5e-4*abs(r["nagy_mGal"]) for r in results)

if __name__=="__main__":
    main()
