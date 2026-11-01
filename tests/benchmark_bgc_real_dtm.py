"""Complete public bgc() regression on real GDAL DEMs (mountain + coast).

Actual sources downloaded by tests/probe_real_dem_sources.py:
- Public Copernicus GLO-90 COG near the Dolomites
- NOAA NCEI DEM_global_mosaic for large-area topography/bathymetry.

The script uses 2 stations per area, including a real sea-bottom
station, and checks full public BGC execution, output text, actual
GDAL intermediate rasters and automatic tuning. The source DEMs have
different resolution/vertical lineage; these are I/O and physical
sanity tests, not a <0.05mGal external reference validation.
"""
import json,re,time,warnings
from pathlib import Path
import numpy as np
from osgeo import gdal
from relgravpy import raster_tools as rt
from relgravpy._bouguer_correction import bgc

ROOT=Path("real_dem_fixtures")
RUNS=[
    dict(name="dolomites",
         stations=[(12.35,46.57,0),(12.45,46.65,0)],
         near="dolomites_near.tif",far="dolomites_far.tif"),
    dict(name="ligurian_coast",
         # NOAA site (9.63,44.13) is BELOW sea level.
         # A second station farther inland is on land.
         stations=[(9.63,44.13,2),(9.63,44.20,0)],
         near="liguria_near.tif",far="liguria_far.tif"),
]


def check_raster(path):
    ds=gdal.Open(str(path))
    assert ds is not None, f"Cannot open GDAL source {path}"
    assert ds.RasterCount>=1 and ds.GetProjection()
    band=ds.GetRasterBand(1)
    arr=band.ReadAsArray()
    nd=band.GetNoDataValue()
    good=np.isfinite(arr)
    if nd is not None:good &= ~np.isclose(arr,nd)
    assert good.sum()>50,(path,float(good.sum()))
    valid=arr[good]
    return dict(path=str(path),width=ds.RasterXSize,
                height=ds.RasterYSize,
                min_m=float(np.min(valid)),max_m=float(np.max(valid)),
                mean_m=float(np.mean(valid)),nodata=nd,
                wkt=ds.GetProjection()[:90])


def run_case(config,automatic=False):
    name=config["name"]
    near=ROOT/config["near"]
    far=ROOT/config["far"]
    near_info,far_info=check_raster(near),check_raster(far)
    print("BGC_REAL_INPUT",json.dumps(
        {"name":name,"near":near_info,"far":far_info}),flush=True)
    assert near_info["min_m"]>=-5000.
    if name=="dolomites":assert near_info["max_m"]>1500.
    if name=="ligurian_coast":
        assert near_info["min_m"]<0 and near_info["max_m"]>0

    stations=config["stations"]
    x=[p[0] for p in stations]
    y=[p[1] for p in stations]
    st_type=[p[2] for p in stations]
    heights=np.asarray(rt.xy2rasterVal(str(near),x,y,prjcode=4326)[0],
                       dtype=float).ravel()
    assert len(heights)==len(stations) and np.all(np.isfinite(heights))
    if name=="ligurian_coast":
        assert heights[0]<0,heights
        assert heights[1]>0,heights

    label=name+("_auto" if automatic else "_manual")
    case_dir=ROOT/label
    began=time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result=bgc(
            x=x,y=y,dtm1=str(near),dtm2=str(far),
            z="dtm",st_type=st_type,
            gs1=None if automatic else 300,
            R1=None if automatic else 5250,
            gs2=None if automatic else 1500,
            # R2 NOT supplied: the default is 167 km.
            R3=12000,curved=True,prjcode_in=4326,
            cpu=1,new_dtm=True,adjust_dtm=False,
            output_file=str(case_dir),
            near_geometry="auto",refine_interface=True,
            target_error=.05 if automatic else .000001,
        )
    elapsed=time.monotonic()-began
    total,near_g,far_g,R1,R2,gs1,gs2,pr1,pr2=result
    assert len(total)==len(near_g)==len(far_g)==len(stations)
    assert np.isfinite(total).all()
    assert np.allclose(total,near_g+far_g,rtol=0,atol=1e-6)
    assert all(isinstance(v,(int,np.integer)) for v in (gs1,R1,gs2,R2))
    if not automatic:assert (gs1,gs2,R1)==(300,1500,5250)
    assert R2>=167000.
    # Deliberately discarded by st_loop to save memory on large surveys.
    assert pr1==tuple(None for _ in stations)
    assert pr2==tuple(None for _ in stations)
    out=case_dir/label
    assert out.is_file(),f"Missing output file: {out}"
    txt=out.read_text()
    assert "bgc_tot" in txt and "Total_time" in txt
    rows=[line for line in txt.splitlines()
          if re.match(r"^\s*[0-9]+\s+-?\d+\.\d+\s+-?\d+\.\d+",line)]
    assert len(rows)==len(stations),(rows,label)
    for intermediate in (case_dir/"dtm1"/"dtm1.tif",
                         case_dir/"dtm2"/"dtm2.tif"):
        assert intermediate.is_file(),intermediate
        assert gdal.Open(str(intermediate)) is not None
    captured=[str(item.message) for item in caught]
    assert not any("GDAL land/sea classification" in w for w in captured)
    record=dict(name=label,number_of_stations=len(stations),
        lat=[float(v) for v in y],lon=[float(v) for v in x],
        st_type=st_type,station_dtm_z_m=[float(v) for v in heights],
        real_raster_near=near_info,real_raster_far=far_info,
        gs1_m=gs1,gs2_m=gs2,R1_m=R1,R2_m=R2,
        bgc_total_mgal=[float(v) for v in total],
        bgc_near_mgal=[float(v) for v in near_g],
        bgc_far_mgal=[float(v) for v in far_g],
        near_grid_cells=(2*R1//gs1)**2,
        far_grid_cells=(2*R2//gs2)**2,
        total_runtime_s=elapsed,warnings=captured,
        output_text_path=str(out),
        source_provenance=(
            "Copernicus GLO-90 + NOAA NCEI ImageServer"
            if name=="dolomites" else "NOAA NCEI ImageServer topobathy"),
        assertion="complete_real_GDAL_BGC_not_accuracy_certificate")
    (ROOT/f"{label}_results.json").write_text(json.dumps(record,indent=2))
    print("BGC_REAL_END_TO_END",json.dumps(record),flush=True)
    return record


if __name__=="__main__":
    assert ROOT.is_dir(),"First run real DEM acquisition."
    for config in RUNS:run_case(config)
    # Important additional regression: the 3 None parameters are
    # jointly tuned ONCE for all stations using the real source DEM.
    run_case(RUNS[0],automatic=True)
