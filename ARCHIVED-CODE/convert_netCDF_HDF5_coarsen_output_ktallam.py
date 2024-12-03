import netCDF4 as nc
import h5py
import numpy as np
from scipy.ndimage import zoom

# Reading netCDF File:
# The read_netcdf function reads a netCDF file using the netCDF4 library and returns the dataset.

# Writing HDF5 File:
# The write_hdf5 function writes data to an HDF5 file using the h5py library.

# Coarsening Data:
# The coarsen_data function coarsens the data by the specified factor using scipy.ndimage.zoom.

# Interpolating Data:
# The interpolate_data function interpolates the data to the specified shape using scipy.ndimage.zoom.

# Main Workflow:
# The script reads the netCDF file, extracts the desired variable, coarsens or interpolates the data, and writes the resulting data to an HDF5 file.

# Note: # Replace 'input.nc', 'output.h5', and 'var_name' with the actual paths and variable names you are working with. Adjust the coarsening factor or the new shape as needed for your specific use case.

# Install Dependencies
# pip install netCDF4 h5py scipy numpy

# You can modify this script to match the specific structure of your netCDF and HDF5 files, including adjusting variable names and dimensions as needed


def read_netcdf(file_path):
    """
    Read a netCDF file and return the dataset.
    """
    ds = nc.Dataset(file_path, mode='r')
    return ds

def write_hdf5(file_path, data, dataset_name):
    """
    Write data to an HDF5 file.
    """
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(dataset_name, data=data)

def coarsen_data(data, factor):
    """
    Coarsen the data by the given factor.
    """
    return zoom(data, 1.0 / factor)

def interpolate_data(data, shape):
    """
    Interpolate the data to the given shape.
    """
    factors = [n / float(o) for n, o in zip(shape, data.shape)]
    return zoom(data, factors)

# Paths to the input netCDF file and output HDF5 file
netcdf_file_path = 'input.nc'
hdf5_file_path = 'output.h5'
dataset_name = 'dataset'

# Read netCDF file
ds = read_netcdf(netcdf_file_path)

# Extract data (assuming variable name 'var_name')
var_name = 'var_name'
data = ds.variables[var_name][:]

# Coarsen or interpolate the data
# Example: coarsening by a factor of 2
coarsened_data = coarsen_data(data, factor=2)

# Example: interpolating to a new shape
new_shape = (100, 100, 100)
interpolated_data = interpolate_data(data, shape=new_shape)

# Write coarsened or interpolated data to HDF5
write_hdf5(hdf5_file_path, interpolated_data, dataset_name)

print(f"Data has been converted and written to {hdf5_file_path}")