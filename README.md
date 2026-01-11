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
git clone https://github.com/yourusername/RelGravPy.git
cd RelGravPy
conda env create -f environment.yml
conda activate relgravpy
