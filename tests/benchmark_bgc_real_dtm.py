"""End-to-end real, open DEM / GDAL / BGC regression.

Downloads are done by probe_real_dem_sources.py:
- Copernicus GLO-90 public COG, Dolomites near field;
- NOAA NCEI global elevation/bathymetry ImageServer, wide-scale far field
  and a Ligurian coast/sea DEM.

This test calls the *public* bgc() entry point, reads the text result,
checks intermediate GDAL rasters and station/cell classifications,
and verifies that outputs agree with returned near/far components.
R2 is omitted deliberately to test the nominal 167 km default.
The NOAA large-area mosaic is coarser than the near DEM, so this is a
pipeline / sanity test, NOT a claim of <0.05 mGal accuracy.
"""
import json, time
from pathlib import Path
import numpy as np
from osgeo import gdal
from relgravpy._bouguer_correction import bgc

ROOT=Path("real_dem_fixtures")
RUNS=[
    dict(name="dolomites", lon=12.35, lat=46.57,
         near="dolomites_near.tif",far="dolomites_far.tif"),
    dict(name="ligurian_coast", lon=9.63, lat=44.13,
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
    if nd is not None: good &= ~np.isclose(arr,nd)
    assert good.sum()>50,(path,float(good.sum()))
    valid=arr[good]
    return dict(path=str(path),width=ds.RasterXSize,
                height=ds.RasterYSize,
                min_m=float(np.min(valid)),max_m=float(np.max(valid)),
                mean_m=float(np.mean(valid)),nodata=nd,
                wkt=ds.GetProjection()[:90])


def run_case(config):
    name=config["name"]
    near=ROOT/config["near"]
    far=ROOT/config["far"]
    near_info,far_info=check_raster(near),check_raster(far)
    print("BGC_REAL_INPUT",json.dumps(
        {"name":name,"near":near_info,"far":far_info}),flush=True)
    assert near_info["min_m"] >= -5000.
    if name=="dolomites": assert near_info["max_m"]>1500.
    if name=="ligurian_coast":
        assert near_info["min_m"]<0 and near_info["max_m"]>0

    case_dir=ROOT/(name+"_full")
    began=time.monotonic()
    result=bgc(
        x=[config["lon"]],y=[config["lat"]],
        dtm1=str(near),dtm2=str(far),z="dtm",
        gs1=300,R1=5250,gs2=1500,
        # R2 NOT SPECIFIED => nominally 167000m.
        R3=12000, curved=True, prjcode_in=4326,
        cpu=1, new_dtm=True, adjust_dtm=False,
        output_file=str(case_dir),
        near_geometry="auto",refine_interface=True,
        target_error=0.000001, # ignored for manual mesh
    )
    elapsed=time.monotonic()-began
    total,near_g,far_g,R1,R2,gs1,gs2,pr1,pr2=result
    assert len(total)==len(near_g)==len(far_g)==1
    assert np.isfinite(total).all()
    assert np.allclose(total,near_g+far_g,rtol=0,atol=1e-6)
    assert all(isinstance(x,(int,np.integer)) for x in (gs1,R1,gs2,R2))
    assert (gs1,gs2,R1)==(300,1500,5250)
    assert R2>=167000
    assert len(pr1)==len(pr2)==1
    assert np.all(np.isfinite(pr1[0]))
    assert np.all(np.isfinite(pr2[0]))
    out=case_dir/f"{name}_full"
    assert out.is_file(),f"Missing output file: {out}"
    text=out.read_text()
    assert "bgc_tot" in text
    assert "Total_time" in text
    for intermediate in (case_dir/"dtm1"/"dtm1.tif",
                         case_dir/"dtm2"/"dtm2.tif"):
        assert intermediate.is_file(),intermediate
        assert gdal.Open(str(intermediate)) is not None
    record=dict(name=name,lat=config["lat"],lon=config["lon"],
        real_raster_near=near_info,real_raster_far=far_info,
        gs1_m=gs1,gs2_m=gs2,R1_m=R1,R2_m=R2,
        bgc_total_mgal=float(total[0]),
        bgc_near_mgal=float(near_g[0]),bgc_far_mgal=float(far_g[0]),
        number_of_near_prisms=len(pr1[0][0]),
        number_of_far_prisms=len(pr2[0][0]),
        total_runtime_s=elapsed,
        output_text_path=str(out),
        source_provenance=(
            "Copernicus GLO-90 + NOAA NCEI ImageServer" if
            name=="dolomites" else "NOAA NCEI ImageServer topobathy"),
        assertion="complete_BGC_real_GDAL_io_not_accuracy_certificate")
    (ROOT/f"{name}_results.json").write_text(json.dumps(record,indent=2))
    print("BGC_REAL_END_TO_END",json.dumps(record),flush=True)
    return record


if __name__=="__main__":
    assert ROOT.is_dir(),"First run the real DEM acquisition probe."
    for config in RUNS:
        run_case(config)
