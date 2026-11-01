"""Probe real publicly available georeferenced DEM inputs for end-to-end tests."""
import json, sys, urllib.parse, urllib.request, pathlib
from osgeo import gdal
import numpy as np

OUT=pathlib.Path("real_dem_fixtures")
OUT.mkdir(exist_ok=True,parents=True)

def fetch_noaa(label,bbox,size):
    endpoint="https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage"
    params={"bbox":",".join(map(str,bbox)), "bboxSR":"4326","imageSR":"4326",
            "size":f"{size},{size}","format":"tiff","pixelType":"F32",
            "interpolation":"+RSP_BilinearInterpolation","f":"image"}
    url=endpoint+"?"+urllib.parse.urlencode(params)
    target=OUT/f"{label}.tif"
    with urllib.request.urlopen(urllib.request.Request(
        url,headers={"User-Agent":"RelGravPy real DEM regression/1.0"}),
        timeout=150) as response:
        data=response.read()
        mime=response.headers.get("Content-Type")
    target.write_bytes(data)
    ds=gdal.Open(str(target))
    if ds is None: raise RuntimeError(f"NOAA export not geotiff {label}: {data[:400]!r}")
    arr=ds.GetRasterBand(1).ReadAsArray()
    print("DEM_PROBE",json.dumps({"name":label,"url":url,"bytes":len(data),
        "content_type":mime,"crs":ds.GetProjection()[:120],"geo":ds.GetGeoTransform(),
        "shape":arr.shape,"min":float(np.nanmin(arr)),"max":float(np.nanmax(arr)),
        "nodata":ds.GetRasterBand(1).GetNoDataValue()}),flush=True)

def fetch_copernicus():
    prefix="Copernicus_DSM_COG_30_N46_00_E012_00_DEM"
    url=f"https://copernicus-dem-90m.s3.amazonaws.com/{prefix}/{prefix}.tif"
    vsi=f"/vsicurl/{url}"
    ds=gdal.Open(vsi)
    if ds is None: raise RuntimeError("Copernicus tile not accessible")
    target=OUT/"dolomites_near.tif"
    gdal.Translate(str(target),ds,projWin=[12.10,46.85,12.75,46.25],format="GTiff",
                   creationOptions=["COMPRESS=DEFLATE","PREDICTOR=3"])
    d=gdal.Open(str(target))
    arr=d.GetRasterBand(1).ReadAsArray()
    print("DEM_PROBE",json.dumps({"name":"dolomites_near","url":url,
        "shape":arr.shape,"min":float(np.nanmin(arr)),"max":float(np.nanmax(arr)),
        "projection":d.GetProjection()[:100]}),flush=True)

if __name__=="__main__":
    fetch_copernicus()
    fetch_noaa("dolomites_far",(10.0,44.7,14.5,48.1),800)
    fetch_noaa("liguria_near",(9.15,43.75,10.0,44.65),700)
    fetch_noaa("liguria_far",(7.35,42.5,12.2,46.0),800)
