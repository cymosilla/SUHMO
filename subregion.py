from pathlib import Path
import scipy
from scipy.ndimage import zoom
from pylab import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import netCDF4 as nc
import h5py
import xarray as xr
import subprocess

# command = ['cdo']
file_path = 'BedMachineGreenland-v5.nc'

def ISglacierSubregion():
    f = nc.Dataset(file_path)
    latbounds = [ 65 , 70 ]
    lonbounds = [ -55 , -50 ] # degrees east ? 
    lats = f.variables['x'][:] 
    lons = f.variables['y'][:]

    # latitude lower and upper index
    latli = np.argmin( np.abs( lats - latbounds[0] ) )
    latui = np.argmin( np.abs( lats - latbounds[1] ) ) 

    # longitude lower and upper index
    lonli = np.argmin( np.abs( lons - lonbounds[0] ) )
    lonui = np.argmin( np.abs( lons - lonbounds[1] ) )  
    thkSubset = f.variables['thickness'][ : , latli:latui , lonli:lonui ] 
    return thkSubset

ISglacierSubregion()
# def subregion(lower_left, upper_right):
#     rootgrp = nc.Dataset("test.nc", "a")
#     fcstgrp = rootgrp.createGroup("forecasts")
#     analgrp = rootgrp.createGroup("analyses")
#     print(rootgrp.groups)