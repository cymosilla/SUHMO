import numpy as np
import netCDF4 as nc
import h5py
from scipy.interpolate import RectBivariateSpline

'''
This is the FINAL preprocessing .py file for everything in it. 

Acknowledgements:
- NOAA/NESDIS/STAR Aerosols and Atmospheric Composition Science Team
- Dr. Krti Tallam
- PT for debugging

Input: NASA BedMachine Greenland netCDF file
User input: Cartesian coordinates for subregion
Output: Subregioned & interpolated HDF5 file

0. Read
0.5. Take user input
1. Cut file in NetCDF format
2. Interpolation via Scipy (RectBivariateSpline)
3. Conversion of .nc dumped file to HDF

TODO: 3. Dump cut file in NetCDF instead of creating new NetCDF file 
As it stands, step 2 creates a new NetCDF file (easier for me to go back to original BM .nc)
We want to overwrite the BM file instead.
'''

file_path_BM_original = "BedMachineGreenland-v5.nc" # BedMachine file ORIGINAL
file_path_BM_preprocessed_NC = "greenland_bedmachine_bed_ice.nc" # BedMachine subregion --> interpolation (BEFORE HDF conversion)
file_path_BM_preprocessed_HDF = "greenland_bedmachine_bed_ice_FINAL.hdf5" # BedMachine final HDF5 file after all steps

###########################################################
# Step 0.5: User Input (absolute first step)
'''
    Allows user to input (x1, y1) & (x2, y2) as the corners of their bounding box
    Note: While the norm is lower left & upper right corners for bounding, interpolation should be capable for taking vice versa using min & max.
'''

# TODO: Implement read section
dataset = nc.Dataset("BedMachineGreenland-v5.nc", 'r')
x = dataset.variables['x'][:]
y = dataset.variables['y'][:]

# Get valid min/max values for x and y
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

def get_user_input():
    while True:
        try:
            print("Rectangular box coordinates are needed for the subregion\n\n")
            print(f"Range of x-coordinates is from {x_min} to {x_max}")
            print(f"Range of y-coordinates is from {y_min} to {y_max}")
            print("Enter the bounding box coordinates for the rectangular region:")
            x1 = float(input("x1 (lower-left x-coordinate): "))
            y1 = float(input("y1 (lower-left y-coordinate): "))
            x2 = float(input("x2 (upper-right x-coordinate): "))
            y2 = float(input("y2 (upper-right y-coordinate): "))
            if x1 == x2 or y1 == y2:
                raise ValueError("Bounding box cannot have zero width or height.")
            return x1, y1, x2, y2
        # TODO: If statements that check if x/y values are out of range
        except ValueError as e:
            print(f"Invalid input: {e}. Enter numeric (integer or floating point) values.")

###########################################################
# Step 1: Subregion (Cutting file)
'''
    Subregion from .nc --> .nc
    Input: BedMachineGreenland-v5.nc
    Output: Subsets of vars as arrays (x_subset, y_subset, bed_subset, thickness_subset)
'''

def preprocess_subregion_ncONLY(input_ncfile, output_ncfile, x1, y1, x2, y2):
    print("\n\n\nStarting subregion extraction")
    # Step 1: Read netCDF data
    dataset = nc.Dataset(input_ncfile, 'r')
    x = dataset.variables['x'][:]  
    y = dataset.variables['y'][:]  
    bed = dataset.variables['bed'][:]  
    thickness = dataset.variables['thickness'][:]  

    # Step 2: Define bounding box and extract the subset
    # TODO: Eliminate this for read function
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    # RElevant indices based on bounding box (user inputted) Cartesian coords
    x_indices = np.where((x >= x_min) & (x <= x_max))[0]
    y_indices = np.where((y >= y_min) & (y <= y_max))[0]
    
    # Sort indices (instead of sorting coords, sort by index to keep order)
    x_indices_sorted = np.sort(x_indices)
    y_indices_sorted = np.sort(y_indices)

    # Select the subsets using the sorted indices
    x_subset = x[x_indices_sorted]
    y_subset = y[y_indices_sorted]

    # np.ix_ constructs mesh from 1-D sequences, similar to cell_to_1D
    bed_subset = bed[np.ix_(y_indices_sorted, x_indices_sorted)]
    thickness_subset = thickness[np.ix_(y_indices_sorted, x_indices_sorted)]

    print("\n\n\n\nSubregion extraction complete")

    return x_subset, y_subset, bed_subset, thickness_subset

###########################################################
# Step 2: Interpolate (refine) data 
'''
    Interpolates data from step 1 dependent on user inputted coordinates
    Input: Subsets of vars as arrays (x_subset, y_subset, bed_subset, thickness_subset) from subregioning
    Output: NetCDF file that is both subregioned & interpolated
'''

def NC_interpolation(x_subset, y_subset, bed_subset, thickness_subset):
    print("\n\n\nStarting interpolation...")
    reso_factor = 2 # TODO: Figure out how to get the resolution factor based on user inputted coordinates
    print(f"First few x_subset values: {x_subset[:10]}")  # Print first 10 vals
    print(f"First few y_subset values: {y_subset[:10]}")  

    # Check for duplicates or non-monotonic sequences
    if np.any(np.diff(x_subset) <= 0):
        print("Warning: x_subset is not strictly increasing!")
    if np.any(np.diff(y_subset) <= 0):
        print("Warning: y_subset is not strictly increasing!")

    reso_factor = 2

    # If there are no duplicates and arrays are strictly increasing
    if np.any(np.diff(x_subset) <= 0) or np.any(np.diff(y_subset) <= 0):
        raise ValueError("x and y values must be strictly increasing for interpolation.")

    # Initiate a finer grid
    # np.linspace(start, stop, numArray)
    # Takes starting value of x/y arrays, takes stop value, then it produces evenly-spaced values based on resolution factor
    print("\n\n\nLinspace")
    x_new = np.linspace(x_subset.min(), x_subset.max(), len(x_subset) * reso_factor)
    y_new = np.linspace(y_subset.min(), y_subset.max(), len(y_subset) * reso_factor)
    
    print("\n\n\nRectBivariateSpline")
    # Init interpolators
    bed_interp = RectBivariateSpline(y_subset, x_subset, bed_subset, kx=1, ky=1)
    thickness_interp = RectBivariateSpline(y_subset, x_subset, thickness_subset, kx=1, ky=1)

    # Interpolate onto new grid
    interpolated_bed = bed_interp(y_new, x_new)
    interpolated_thickness = thickness_interp(y_new, x_new)

    # Create output NetCDF 
    with nc.Dataset(file_path_BM_preprocessed_NC, 'w', format='NETCDF4') as new_dataset:
        # Create dims
        new_dataset.createDimension('x', len(x_new))
        new_dataset.createDimension('y', len(y_new))

        # Create vars
        new_dataset.createVariable('x', 'f4', ('x',))
        new_dataset.createVariable('y', 'f4', ('y',))
        new_dataset.createVariable('bed', 'f4', ('y', 'x'))
        new_dataset.createVariable('thickness', 'f4', ('y', 'x'))

        # Write data to vars
        new_dataset.variables['x'][:] = x_new
        new_dataset.variables['y'][:] = y_new
        new_dataset.variables['bed'][:, :] = interpolated_bed
        new_dataset.variables['thickness'][:, :] = interpolated_thickness
    
    print("\n\n\n\Interpolation complete")

    print(f"NetCDF file successfully written as: {file_path_BM_preprocessed_NC}")

###########################################################
# Step 3: Save as HDF
'''
    Converts NetCDF file from interpolation to HDF5 format
    Input: NetCDF file
    Output: HDF5 file
'''

def NC_to_HDF5(input_ncfile, output_hdf5file):
    # Step 1: Open the interpolated NetCDF file
    # TODO: Figure out if having step 0 for reading files would be beneficial as a helper
    with nc.Dataset(input_ncfile, 'r') as nc_file:
        x = nc_file.variables['x'][:]
        y = nc_file.variables['y'][:]
        bed = nc_file.variables['bed'][:]
        thickness = nc_file.variables['thickness'][:]
    
    # Step 2: Create HDF5 file conversion
    with h5py.File(output_hdf5file, 'w') as hdf_file:
        # Create datasets for x, y, bed, and thickness
        hdf_file.create_dataset('x', data=x)
        hdf_file.create_dataset('y', data=y)
        hdf_file.create_dataset('bed', data=bed)
        hdf_file.create_dataset('thickness', data=thickness)
    
    print(f"HDF5 file successfully written as: {output_hdf5file}")

###########################################################
# All Together Now - Combination of all helper functions
def preprocess_BedMachine():
    # Step 0.5: User input
    x1, y1, x2, y2 = get_user_input() 
    # Step 1: Subregion
    x_subset, y_subset, bed_subset, thickness_subset = preprocess_subregion_ncONLY(file_path_BM_original, file_path_BM_preprocessed_NC, x1, y1, x2, y2) 
    # Step 2: Interpolation
    NC_interpolation(x_subset, y_subset, bed_subset, thickness_subset) 
    # Step 3: .nc --> hdf5
    NC_to_HDF5(file_path_BM_preprocessed_NC, file_path_BM_preprocessed_HDF)

def preprocess_geothermal():
    pass

def preprocess():
    # TODO: Get IHFC geothermal flux working & combine with BM here
    pass

preprocess_BedMachine()