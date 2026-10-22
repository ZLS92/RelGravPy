# RelRelGravPy

**RelGravPy** is a Python package for processing **relative gravity data** for geophysical applications.  
It is primarily designed for **land-based gravity surveys**, but its architecture is intended to be extensible to **marine** and **airborne** gravimetry in the future.

The package covers the full processing chain: from **instrumental and campaign corrections**, through **gravity network adjustment**, to **Bouguer anomaly computation** and **geophysical analysis**.

---

## Goals

- Provide **reproducible and transparent** tools for relative gravity data processing
- Clearly separate the different stages of the gravimetric workflow
- Offer a flexible framework that can be extended to:
  - multi-day surveys
  - complex gravity networks
  - mobile platforms (marine and airborne gravimetry)

---

## Supported Workflow

A typical processing workflow includes:

1. **Pre-processing**
   - timestamp handling
   - organization of observations by station and session

2. **Instrumental corrections**
   - drift correction (linear or polynomial)
   - handling of repeated measurements
   - optional session- or day-wise processing

3. **Gravity network adjustment**
   - least-squares network adjustment
   - handling of constraints (absolute or relative reference stations)
   - multi-loop and multi-day networks

4. **Geophysical corrections**
   - free-air correction
   - Bouguer correction
      (total topographic and bathymetry mass effect using DTMs and coastlines)

5. **Anomaly analysis**
   - statistical analysis
   - profiles and maps
   - spectral decomposition and analysis
   - gravity grid filters and field-derivatives operators 
   - regional/local gravity field separation 
   - data preparation for geophysical modelling and interpretation

---

## Project status

The package is **under active development**.  
Current focus areas include:

- land-based relative gravimetry
- static and semi-static surveys
- spring-based and feedback gravimeters

Planned features:

- support for marine gravimetry
- support for airborne gravimetry
- integration with digital elevation models (DEM)
- extension of dynamic corrections

---

## Installation (conda)

```bash
git clone https://github.com/ZLS92/RelGravPy.git
cd RelGravPy
conda env create --file relgravpy.yaml -n relgravpy
conda activate relgravpy


### Tamura (1987) theoretical gravity tide — experimental

RelGravPy contains a standalone NumPy-only predictor for the 1,200-wave
Tamura T87 potential, as used in the **TSoft** theoretical gravity-tide
calculation. No PyGTide installation or executable is required.

```python
from relgravpy import grav_processing as gp

utc, gravity_tide = gp.earth_tides_tsoft(
    lat=45.5, lon=10.2, z=150,
    start="2026-10-21T00:00:00",
    end="2026-10-22T00:00:00", step=3600,
    unit="uGal",
)

# Correction to ADD to an uncorrected gravimeter reading:
to_add = gp.earth_tides_tsoft(45.5, 10.2, 150,
    DateTime="2026-10-21T10:00:00", correction=True)
```

Important conventions: inputs are **UTC** (timezone-naive values are
interpreted as UTC); longitude positive east; geodetic latitude (degrees);
elevation in metres above the reference ellipsoid; output is positive-down
gravity **effect** in mGal by default, or in uGal/nm/s2. The actual
correction to add to uncorrected measurements is the **negative** effect.

**TSoft manual discrepancy:** its printed WDD Example 3 calls
0.999853–1.000147 cycles/day the "K1" band; this is actually the
**S1** band. The physical K1 wave is around **1.00274 cycles/day**
and its omission previously led to underestimated diurnal tides.
The `"tsoft_wdd"` preset now explicitly includes K1
(1.001825–1.003651 cycles/day) with the theoretical WDD factor
1.13448. This correction is necessary for physically meaningful
predictions and agrees with TSoft's historical K1 bug notice.

The default `delta=1.16` applies a single amplitude factor to all
frequencies, while preserving internal Wahr–Dehant–Zschau elastic
frequency response (including the diurnal resonance). For **TSoft
settings from the published manual**, use the WDD frequency-band
preset, which retains their distinct amplitude factors and phase leads
(frequency in cycles/day):

```python
tide = gp.earth_tides_tsoft(
    45.5, 10.2, 150, DateTime="2026-10-21T10:00:00",
    wavegroups="tsoft_wdd", unit="nm/s2"
)
```

To reproduce a *specific* TSoft configuration, you may instead supply
your own four-column wave-group matrix:

```python
groups = [
    [0.0000, 0.249951, 1.16, 0.0],
    [0.7215, 1.216397, 1.15, 0.0],
    [1.719381, 2.182843, 1.162, 0.0],
    [2.753244, 4.347615, 1.0736, 0.0],
]
tide = gp.earth_tides_tsoft(
    45.5, 10.2, 150, DateTime="2026-10-21T10:00:00",
    wavegroups=groups, unit="nm/s2",
)
```

**Validation:** the standalone NumPy implementation was independently
compared against ETERNA/PyGTide run exclusively in a separate CI job
using the **same T87 catalogue** (not a RelGravPy dependency). On 49
hourly samples at Karlsruhe, Jan 1–3, 2017, with one unscaled group
and pole/LOD disabled, the resulting RMSE was **0.0282 nm/s²**
(**0.00282 µGal**) with maximum absolute difference **0.0549 nm/s²**
(**0.00549 µGal**). With the published **TSoft WDD** frequency
bands and delta factors applied to both models, the independent ETERNA
benchmark achieves **0.02761 nm/s² RMS (0.00276 µGal)** and a maximum
difference of **0.05138 nm/s² (0.00514 µGal)** over the same 49 samples.
These figures demonstrate agreement with ETERNA using TSoft parameters,
**not a direct TSoft output comparison**.

The implementation does not include UT1−UTC from IERS, pole/LOD tides,
ocean loading, or atmospheric loading. The previous `earth_tides()`
Longman method is unchanged. Complete TSoft compatibility also depends
on matching its tide parameter table and awaits a direct TSoft export.

To compare against TSoft, export a CSV file with two columns,
`utc,tsoft_nm_s2`, and run:

```bash
python -m examples.compare_tsoft_t87 my_tsoft.csv \
  --lat 45.5 --lon 10.2 --height 150 --plot
```

If non-default amplitude factors were used in TSoft, export its
four-column wave table and pass `--wavegroups groups.txt`. The
comparison reports mean bias, RMSE, maximum absolute difference
in **µGal**, and correlation. The T87 catalogue is vendored from
Wenzel's HW95-normalized table, with provenance and license text in
`relgravpy/data/`.

As a literature-only **reference target**, the original PyGTide
comparison to TSoft with identical T87 catalogues reported an RMSE
of 0.022 nm/s² and a maximum absolute discrepancy of 0.054 nm/s²
(Karlsruhe, January 2017). These are *not* measured results for
RelGravPy; see Rau (2018), PyGTide user guide, Figure 2.
