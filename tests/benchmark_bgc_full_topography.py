"""Actual forward-model test of mesh tuning on full synthetic topography.

Two independent calculations:
  candidate: Nagy prisms within R1, arbitrary-facet R1 bridge,
             spherical far polyhedra
  reference: dense (100, 50 m) centre-shifted Nagy prism tessellation
             over the entire compact nonzero synthetic source domain

The reference is not an exact curved-Earth potential, so the residual
combines discretisation and remaining cross-kernel geometrical effects.
No 167-km input raster is required because height and density contrast
are identically zero outside the finite test support. The nominal R2
passed to the optimizer remains 167000 m.

Use: PYTHONPATH=tests:. python -u tests/benchmark_bgc_full_topography.py
"""
import json
import time
import warnings
import numpy as np
from test_bouguer_polyhedra import BC, GM

R = 6371000.
DC = 2670.
DW = 1030.
SUPPORT = 16000.
ROOT = 18000.


def terrain(x, y, kind):
    x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    r=np.maximum(abs(x),abs(y))
    window=np.clip((SUPPORT-r)/3000.,0,1.)**2
    if kind=="mountain":
        h=(1100.+620.*np.exp(-((x-1500)**2+(y+1100)**2)/(2*4200**2))
           +240*np.sin(x/950.)*np.cos(y/1100.)
           +160*np.sin(x/260.+y/220.)
           +290*np.tanh((x+.3*y-2100.)/600.))
    elif kind=="coastal_bathymetry":
        h=(900.*np.tanh((x+500.)/650.)
           +120.*np.sin(y/900.)*np.exp(-abs(x)/6000.)
           -450.*np.exp(-((x+2300.)/1100.)**2)
                     *np.exp(-(y/4200.)**2))
    else:
        raise ValueError(kind)
    return h*window


def patch_features(station,kind):
    x,y=station
    a=np.arange(-16,17)*30.
    b=np.arange(-12,13)*200.
    ax,ay=np.meshgrid(x+a,y+a)
    bx,by=np.meshgrid(x+b,y+b)
    return BC._bgc_terrain_features((terrain(ax,ay,kind),
                                     terrain(bx,by,kind)),30.,200.)


def mesh_from_step(station,step,extent):
    """Station-centred odd cell mesh, restricted to compact source."""
    x,y=station
    n=int(np.ceil(extent/step))+2
    edges=(np.arange(-n,n+1)+.5)*step
    w,s=np.meshgrid(edges[:-1],edges[:-1])
    e,north=np.meshgrid(edges[1:],edges[1:])
    cell=np.column_stack((w.ravel(),e.ravel(),s.ravel(),north.ravel()))
    h=terrain(x+(cell[:,0]+cell[:,1])/2,
              y+(cell[:,2]+cell[:,3])/2,kind=mesh_from_step.kind)
    return cell,h


def convert_cells(cell,h,station,near):
    """Signed land and seabed masses. Water replacement = rho_w-rho_c."""
    valid=np.abs(h)>1e-7
    cell,h=cell[valid],h[valid]
    bottom=np.minimum(0,h)
    top=np.maximum(0,h)
    if near:
        cx=(cell[:,0]+cell[:,1])/2
        cy=(cell[:,2]+cell[:,3])/2
        sag=R-np.sqrt(R**2-cx**2-cy**2)
        bottom-=sag
        top-=sag
    prisms=np.column_stack((cell,bottom,top))
    density=np.where(h>0,DC,DW-DC)
    return prisms,density


def grav_candidate(station,obs_z,kind,selection):
    gs1,R1,gs2=(selection[k] for k in ("gs1","R1","gs2"))
    mesh_from_step.kind=kind
    start=time.perf_counter()
    fine_cells,fine_h=mesh_from_step(station,gs1,R1)
    near,den=convert_cells(fine_cells,fine_h,station,near=True)
    obs=np.array([0.,0.,obs_z])
    near_g=float(GM.prism_grav([obs],near,den).compute_gz()[0])
    all_far,all_h=mesh_from_step(station,gs2,ROOT)
    fx=(all_far[:,0]+all_far[:,1])/2
    fy=(all_far[:,2]+all_far[:,3])/2
    include=(abs(fx)>=R1)|(abs(fy)>=R1)
    far,fd=convert_cells(all_far[include],all_h[include],
                         station,near=False)
    try:
        bridge_g,ordinary,ordinary_density=BC._transition_polyhedral_gz(
            far,fd,near,den,obs,R,R1)
        far_g=bridge_g+BC._polyhedral_gz(
            ordinary,ordinary_density,obs,R)
    except ValueError as exc:
        return dict(status="geometry_error",error=str(exc),
                    near_cells=len(near),far_cells=len(far),
                    run_s=time.perf_counter()-start)
    return dict(status="ok",gravity_mGal=near_g+far_g,
                near_mGal=near_g,far_mGal=far_g,
                near_cells=len(near),far_cells=len(far),
                run_s=time.perf_counter()-start)


def grav_reference(station,obs_z,kind,step):
    mesh_from_step.kind=kind
    start=time.perf_counter()
    cells,height=mesh_from_step(station,step,ROOT)
    prisms,density=convert_cells(cells,height,station,True)
    g=float(GM.prism_grav([[0.,0.,obs_z]],prisms,density).
            compute_gz()[0])
    return dict(value_mGal=g,step_m=step,
                cells=len(prisms),run_s=time.perf_counter()-start)


def run():
    cases=(
        ("mountain",[(0.,0.),(2200.,-1800.)]),
        ("coastal_bathymetry",[(1200.,0.),(380.,-2300.)]),
    )
    for kind,stations in cases:
        coords=np.asarray(stations)
        features=np.vstack([patch_features(pt,kind) for pt in coords])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0=time.perf_counter()
            choice=BC.optimize_bgc_mesh(
                coords[:,0],coords[:,1],None,
                target_error=.05,R2=167000.,
                terrain_features=features,raster_resolution=30.)
            tune_s=time.perf_counter()-t0
        out=[]
        for station in stations:
            station_height=float(terrain(*station,kind))
            obs_z=max(0.,station_height)+50.
            result=grav_candidate(station,obs_z,kind,choice)
            ref100=grav_reference(station,obs_z,kind,100.)
            ref50=grav_reference(station,obs_z,kind,50.)
            if result["status"]=="ok":
                result["actual_error_vs_50m_mGal"]=(
                    result["gravity_mGal"]-ref50["value_mGal"])
                result["abs_actual_error_mGal"]=abs(
                    result["actual_error_vs_50m_mGal"])
            result.update(station_xy_m=station,obs_z_m=obs_z,
                          reference_100m=ref100,reference_50m=ref50,
                          reference_convergence_mGal=abs(
                              ref100["value_mGal"]-ref50["value_mGal"]))
            out.append(result)
            print("BGC_FULL_STATION",json.dumps(
                dict(scenario=kind,**result)),flush=True)
        valid=[s for s in out if s["status"]=="ok"]
        summary=dict(
            scenario=kind,chosen=dict(
                (k,choice[k]) for k in
                ("gs1","R1","gs2","R2","estimated_error_mgal",
                 "meets_target","critical_station_index","candidates")),
            tune_s=tune_s,
            warnings=[str(w.message) for w in caught],
            stations=len(stations),
            successful_forward_stations=len(valid),
            max_abs_actual_error_mGal=(
                max(s["abs_actual_error_mGal"] for s in valid)
                if valid else None),
            max_reference_convergence_mGal=max(
                s["reference_convergence_mGal"] for s in out),
            caveat="50m prism mesh is a finite reference, not exact curved Earth")
        print("BGC_FULL_TOPOGRAPHY",json.dumps(summary),flush=True)
        assert choice["R2"] >= 167000.
        assert len(out)==len(stations)
        assert np.all(np.isfinite(
            [s["reference_50m"]["value_mGal"] for s in out]))

if __name__=="__main__":
    run()
