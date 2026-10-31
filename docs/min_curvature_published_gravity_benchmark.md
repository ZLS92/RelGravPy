# Real published gravity grid benchmark — SA GRAVITY 2016

Data source: Geological Survey of South Australia, *South Australian regional gravity grids and images, 2016*, catalogue **mesac4462**. The 2016 compilation used 531,047 gravity stations, based on the official catalog record. Data are subject to CC BY 4.0 attribution.

For a lightweight, reproducible benchmark we downloaded the independently published **Eyre Peninsula** numerical subset made available in [Adelaide-Earth-Imaging/potential-fields](https://github.com/Adelaide-Earth-Imaging/potential-fields/tree/main/data):

- source files: `eyre_grav` (little-endian float32 binary ER Mapper raster), `eyre_grav.ers` (georeferencing);
- original raster: 743 rows × 730 columns, 400 m cell, EPSG:28353 (GDA94/MGA53), units mGal;
- finite values in original raster: 541,713; missing cells: 677 (coastal nodata);
- SHA256 of binary source: `6c6ffb70f5fb345d96f6d0f00fb76daa9de688a6fbeac09750dbb879eb4ab2ea`;
- benchmark window: 67 × 67 cells, starting at original raster row 324, column 71;
- central measured window: 51 × 51; outer 8-cell ring contains 1,888 values withheld for padding evaluation;
- original reference range: approximately -46.29 to +5.34 mGal, standard deviation 17.64 mGal.

## Numerical results (verified 2026-10-31)

The published grid is taken as a **numerical reference**, not as the true gravity field. We remove values under controlled masks, reconstruct with `solver="robust"`, `tension=0.15`, `anchor=0.01`, `data_weight=10000`, then compare reconstructed values ONLY at withheld positions.

| Mask | Measured cells | Withheld cells | RMSE mGal | MAE mGal | Max abs. difference mGal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random 30% removal | 1843 | 758 | 0.0123 | 0.0061 | 0.1650 |
| Central square 17×17 missing | 2312 | 289 | 0.9064 | 0.6693 | 2.1429 |
| Seven horizontal profiles retained | 357 | 2244 | 0.5728 | 0.4586 | 1.7467 |
| Only central cluster retained | 529 | 2072 | 10.3183 | 7.9123 | 28.7531 |
| Padding: 8 cells per side | 2601 | 1888 | 6.8159 | 5.0510 | 21.6923 |

The region is approximately 26.8 km × 26.8 km overall. An 8-cell pad is **3.2 km per side**. This demonstrates how different the inference tasks are: local interpolation within a well-covered smooth grid can be highly accurate, while extrapolation into unconstrained zones becomes unreliable even when the output contains no NaNs and the linear solver converges.

## Reproduction

```bash
PYTHONPATH=tests:. python examples/benchmark_published_gravity.py
```

The script downloads the numerical ER Mapper binary and its header from GitHub, reads actual gravity values in mGal, chooses the same full-data land window by a deterministic variance criterion, generates all masks, reconstructs all five scenarios, and exports a small georeferenced NPZ patch, CSV metric table, two figures and report. It does not need GDAL or GMT.

Verified GitHub Actions run:
https://github.com/ZLS92/RelGravPy/actions/runs/41206303135

The run uploads an artifact named `sa-gravity-2016-eyre-min-curvature-benchmark` that contains the ORIGINAL raster, header, reference subset, validation arrays, plots and CSV.

## Scientific interpretation

1. The original SA GRAVITY product was already gridded from ground measurements; its 400 m pixels are spatially correlated, so withholding random pixels creates an *easier* problem than interpolating original irregular stations.
2. The published grid is a numerical reference, not an independent independent physical truth, and gridding method differences are not directly inferable from these results.
3. For gravity and microgravity, a pad of 3.2 km can produce errors of order 1–20 mGal relative to the published field; those errors are unacceptable in many high-precision applications.
4. In real projects, compare several values of `anchor`, `tension`, padding width and block size, and evaluate profile holdouts, central gaps and outer strips independently.
5. The original raster remains archived separately and untouched. Two optional, separately labeled GeoTIFFs may be generated from the NPZ reference and reconstruction with exact EPSG:28353 pixel-center registration.

## References

- Geological Survey of South Australia (2016), *South Australian regional gravity grids and images, 2016*. https://pid.sarig.sa.gov.au/dataset/mesac4462 . CC BY 4.0.
- Adelaide-Earth-Imaging (2026), numerical gravity subset from SA GRAVITY 2016, in the `potential-fields` teaching/research toolkit. https://github.com/Adelaide-Earth-Imaging/potential-fields .
