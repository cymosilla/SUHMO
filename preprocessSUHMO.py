import scipy
from scipy.io import netcdf
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import netCDF4 as nc
import h5py

'''
Acknowledgements to NOAA/NESDIS/STAR Aerosols and Atmospheric Composition Science Team

All in the same .py file with different functions
0. Read
1. Cut file in NetCDF format
2. Coarsen OR interpolate (refinement) - Interpolate most likely
2.5 (OPTIONAL) Coarsen - final grid size based on user input
3. Dump cut file in NetCDF instead of creating new NetCDF file 
4. Conversion of .nc dumped file to HDF
5. Addition of boundary conditions

Order of development: 0, 1 (or 2), 3, 4, 2.5

Input: .nc file (for now)
Output: HDF5 file

To test step 1:
- Obtain full netCDF data (cut of original file)
- nc dump
- Use preprocess-thk_bed_btrc code to test if cut works (see below)

    #ouput netcdf
    print ('writing ...')
    ncout = Dataset(output_nc,'w')
    #dimensions
    xdim = ncout.createDimension('x',size=nx)
    ydim = ncout.createDimension('y',size=ny)
    #var defs
    xv = ncout.createVariable('x','f8',('x'))
    yv = ncout.createVariable('y','f8',('y'))
'''

# Step 0: Read data
def print_data_bedMachine(file_path):
    file_id = Dataset(file_path)
    print(file_id.variables['dataid'])
    print(file_id.variables['bed'])
    print(file_id.variables['thickness'])
    print(file_id.variables['errbed'])
    print(file_id.variables['surface'])
    # [:,:] obtains data from a 2d format
    bed = file_id.variables['bed'][:,:]
    id = file_id.variables['dataid'][:,:]
    thk = file_id.variables['thickness'][:,:]

###########################################################
# Step 1: Subregion (Cutting file)
def subregion(lower_left, upper_right):
    rootgrp = Dataset("test.nc", "a")
    fcstgrp = rootgrp.createGroup("forecasts")
    analgrp = rootgrp.createGroup("analyses")
    print(rootgrp.groups)



###########################################################
# Step 2: Interpolate (refine) data

def interpolate (x, y):
    x = np.random.rand(100) * 10
    y = np.random.rand(100) * 10
    z = np.sin(x) * np.cos(y)

    # Grid size
    grid_x, grid_y = np.mgrid[0:10:100j, 0:10:100j]

    # Griddata
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    plt.figure()
    plt.scatter(x, y, c=z, s=100, edgecolor='k', label='Original Points')
    plt.imshow(grid_z.T, extent=(0, 10, 0, 10), origin='lower', alpha=0.5)
    plt.colorbar(label='Interpolated Values')
    plt.legend()
    plt.show()


###########################################################
# Step 2.5: Coarsen data (from coarsen.py)
# (1km, 2km, 4km, 8km)
def coarsen_1D(u):
    """
    coarsen u(1D) by a factor of two
    """
    n = len(u)
    uc = 0.5 * ( u[0:n-1:2] + u[1:n:2] )
    #print ('u[1] - u[0] = {} {}'.format(type(u[0]), u[1]-u[0]))
    return uc

def coarsen_2D(u):
    """
    coarsen u(2D) by a factor of two
    """
    n,m = u.shape
    uc = 0.25 * ( u[0:n-1:2, 0:m-1:2]
                + u[1:n:2,   0:m-1:2]
                + u[0:n-1:2, 1:m:2]
                + u[1:n:2,   1:m:2])
    return uc

def coarsenc(name, fine_name):
    v_names = ['thk','topg','umod','btrc','umodc']
    fine_nc = Dataset(fine_name,'r')
    coarse_nc = Dataset(name, 'w')
    
    x_fine = fine_nc.variables['x'][:]
    y_fine = fine_nc.variables['y'][:]
    x_coarse, y_coarse = coarsen_1D(x_fine), coarsen_1D(y_fine)

    xdim = coarse_nc.createDimension('x',size=len(x_coarse))
    ydim = coarse_nc.createDimension('y',size=len(y_coarse))
    
    xv = coarse_nc.createVariable('x','f8',('x'))
    xv[:] = x_coarse

    yv = coarse_nc.createVariable('y','f8',('y'))
    yv[:] = y_coarse
   

    for v in v_names:
        vv = coarse_nc.createVariable(v  ,'f8',('y','x'))
        vv[:,:] = coarsen_2D(fine_nc.variables[v][:,:])


###########################################################
# Step 4: Converison to HDF
def conversionToHDF():
    # if ncdump -k == netCDF-4 classic model OR ___
    #   print("In HDF5")
    # else 
    #   print("Starting converison")

    return 0

def netCDFtoHDF(netcdf_file, hdf5_file):
    # Opens nc file & creates new hdf5 file
    with nc.Dataset(netcdf_file, 'r') as src:
        with h5py.File(hdf5_file, 'w') as dst:
            for name in src.ncattrs():
                dst.attrs[name] = src.getncattr(name)
            
            # Copy dims
            for name, dimension in src.dimensions.items():
                dst.create_dataset(name, data=np.arange(dimension.size), dtype='i4')
            
            # Copy vars
            for name, variable in src.variables.items():
                data = variable[:]
                dst.create_dataset(name, data=data, dtype=data.dtype)
                
                # Copy var attrs
                for attr_name in variable.ncattrs():
                    dst[name].attrs[attr_name] = variable.getncattr(attr_name)


###########################################################
# Step 5: Addition of Realistic BCs
# Geothermal flux, ice thickness, bed elevation


###########################################################
# main

directory_path = Path.cwd()  # Current working directory
bed_machine = 'BedMachineGreenland-v5.nc'
file_path = directory_path / bed_machine


print_data_bedMachine(bed_machine)

'''
# Examples below
# Read in fire radiative power (FRP) data
# Metadata indicates data are 2-dimensional so use [:,:] to extract all data in 2 dimensions
abi_frp = file_id.variables['Power'][:,:]

# Print max and min of FRP array to check data range
print('The maximum FRP value is', np.max(abi_frp), file_id.variables['dataid'].units)
print('The minimum FRP value is', np.min(abi_frp), file_id.variables['bed'].units)


# Print FRP array
# netCDF4 library automatically masks (--) invalid data & data outside of valid range
print(abi_frp)
'''