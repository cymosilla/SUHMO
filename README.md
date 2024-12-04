# Welcome to SUHMO's preprocessing tools!

## Overview 

The objective of these tools are to convert external field data to a format SUHMO can read & output with for the sake of increasing the realism of SUHMO's simulations.

The three parameters that are being evaluated by these tools are bedrock elevation, ice thickness, & geothermal flux. 

Bedrock elevation & ice thickness data are derived from NASA MEaSUREs BedMachine. Geothermal flux is from the International Heat Flow Commission (IHFC) 2024.

NASA BedMachine stores Greenland data as a NetCDF (.nc) file. Both parameters are subregioned & interpolated into the same file as BedMachineGreenland-v5.nc.

Geothermal flux is stored as a .xslx (Excel spreadsheet) file. It is converted to .nc utilizing the pandas module. 

All three parameters will then be compiled as one .nc file that maps out Greenland's parameters, Greenland_params.nc. This is the file used for the subregion & interpolation.

# TODO: Figure out if we combine all three params as one file or subregion & interpolate them separately, then combine as one HDF file at the end.

## Documentation

This README is SOLELY for the preprocessing tools, which can be run without cloning SUHMO. Documentation for SUHMO is available [here](https://ennadelfen.github.io/SUHMO/). The SUHMO governing equations and core algorithm are described in: https://ennadelfen.github.io/SUHMO/Model. 

Before starting with preprocessing, ensure CHOMBO_HOME & SUHMO_HOME have exports from the instructions above.

## Getting started with preprocessing

These tools are meant to be automatic, that is, run a singular command with your desired parameters on the size of the plot. The tools take that input, take a subregion of said size & interpolate BedMachineGreenland-v5.nc, & then input it into SUHMO to output a simulation that can be visualized through the VisIt tool.

### Packages to install
- pathlib
- scipy
- numpy
- matplotlib
- netCDF4
- h5py
- xarray
- os (for SUHMO inputting)
- ncview (OPTIONAL for easy .nc file viewing)

The easiest method would be to install an Anaconda (or miniconda) environment with these packages. The most important packages are numpy, scipy, netCDF4, & h5py; the rest of them were used in test files & were not integrated (so far) into the main file.

### Files needed
BedMachineGreenland-v5.nc, which can be downloaded via NASA MEaSUREs.
Global Heat Flow Database, which can be downloaded via IHFC's website as a .zip file using the 2024 version. 

### What to run
```
cd SUHMO-Preprocessing-Tools
python preprocess.py
```
It will then ask you to input bounding box coordinates, as the parameters are such below.
```py
preprocess(x1, y1, x2, y2)
```
IMPORTANT: INPUT x, y values as floats or integers. x1, y1 are the lower-left coordinates & x2, y2 as the upper-right coordinates, forming a rectangle subregion. 

BedMachineGreenland-v5.nc's grid size is from x = 10,218 & y = 18,346

The preprocess command should automatically be subregioned & interpolated, then output one hdf5 file as greenland_FINAL.hdf

Specific coordinates for the Isunnguata Sermia glacier approximation: (-2740906, -375354) and (-2845385, -195722)

## Acknowledgements
- Mentors: Dan Martin, Anna Felden, Anjali Sandip
- Dr. Krti Tallam for lending her preprocessing tools
- Stephen for his BISICLES preprocessing tools
- NASA MEaSUREs BedMachine team for their NetCDF Greenland v5 file
- International Heatflow Commission for their Global Heat Database
- NOAA/NESDIS/STAR Aerosols and Atmospheric Composition Science Team
