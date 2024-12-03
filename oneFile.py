import netCDF4 as nc
import numpy as np
from scipy.interpolate import RectBivariateSpline
import h5py
import os
# from preprocess import NC_to_HDF5

def preprocess_bedmachine(input_ncfile, output_hdf5file, x1, y1, x2, y2):
    # Step 1: Read netCDF data
    dataset = nc.Dataset(input_ncfile, 'r')
    x = dataset.variables['x'][:]  
    y = dataset.variables['y'][:]  
    bed = dataset.variables['bed'][:]  
    thickness = dataset.variables['thickness'][:]  

    # Step 2: Define bounding box
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    x_indices = np.where((x >= x_min) & (x <= x_max))[0]
    y_indices = np.where((y >= y_min) & (y <= y_max))[0]
    
    x_subset = x[x_indices]
    y_subset = y[y_indices]
    
    # Check for empty or too small subsets
    if len(x_subset) < 2 or len(y_subset) < 2:
        raise ValueError("Insufficient data points for interpolation.")
    
    # Ensure sorted data for interpolation
    x_subset = np.sort(x_subset)
    y_subset = np.sort(y_subset)

    bed_subset = bed[np.ix_(y_indices, x_indices)]
    thickness_subset = thickness[np.ix_(y_indices, x_indices)]

    # Check if there's enough variation in the data
    if np.all(bed_subset == bed_subset[0, 0]) or np.all(thickness_subset == thickness_subset[0, 0]):
        raise ValueError("Data is constant, interpolation will not work properly.")

    # Step 3: Interpolate
    bed_interp = RectBivariateSpline(y_subset, x_subset, bed_subset)
    thickness_interp = RectBivariateSpline(y_subset, x_subset, thickness_subset)
    
    x_new = np.linspace(x_min, x_max, 100)  
    y_new = np.linspace(y_min, y_max, 100)
    bed_interp_data = bed_interp(y_new, x_new)
    thickness_interp_data = thickness_interp(y_new, x_new)

    # Step 4: Save as HDF5
    with h5py.File(output_hdf5file, 'w') as hdf_file:
        hdf_file.create_dataset('x', data=x_new)
        hdf_file.create_dataset('y', data=y_new)
        hdf_file.create_dataset('bed', data=bed_interp_data)
        hdf_file.create_dataset('thickness', data=thickness_interp_data)

    print(f"Data saved to {output_hdf5file}")

def preprocess_subregion(input_ncfile, output_hdf5file, x1, y1, x2, y2):
    # Step 1: Read netCDF data
    dataset = nc.Dataset(input_ncfile, 'r')
    x = dataset.variables['x'][:]  
    y = dataset.variables['y'][:]  
    bed = dataset.variables['bed'][:]  
    thickness = dataset.variables['thickness'][:]  

    # Step 2: Define bounding box and extract the subset
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    # Find indices for the bounding box
    x_indices = np.where((x >= x_min) & (x <= x_max))[0]
    y_indices = np.where((y >= y_min) & (y <= y_max))[0]
    
    # Extract the subsets of the data based on the indices
    x_subset = x[x_indices]
    y_subset = y[y_indices]
    bed_subset = bed[np.ix_(y_indices, x_indices)]
    thickness_subset = thickness[np.ix_(y_indices, x_indices)]

    # Step 3: Save the subregion as an HDF5 file
    with h5py.File(output_hdf5file, 'w') as hdf_file:
        hdf_file.create_dataset('x', data=x_subset)
        hdf_file.create_dataset('y', data=y_subset)
        hdf_file.create_dataset('bed', data=bed_subset)
        hdf_file.create_dataset('thickness', data=thickness_subset)

    print(f"Data saved to {output_hdf5file}")

################# TESTING #################
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

def preprocess_subregion_ncONLY(input_ncfile, output_ncfile, x1, y1, x2, y2):
    try:
        print("Starting subregion extraction...")

        # Read netCDF data
        dataset = nc.Dataset(input_ncfile, 'r')
        x = dataset.variables['x'][:]
        y = dataset.variables['y'][:]
        bed = dataset.variables['bed'][:]
        thickness = dataset.variables['thickness'][:]

        # Define bounding box
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Relevant indices based on bounding box
        x_indices = np.where((x >= x_min) & (x <= x_max))[0]
        y_indices = np.where((y >= y_min) & (y <= y_max))[0]

        # Subsetting
        x_subset = x[x_indices]
        y_subset = y[y_indices]
        bed_subset = bed[np.ix_(y_indices, x_indices)]
        thickness_subset = thickness[np.ix_(y_indices, x_indices)]

        # Debug: Check sizes of the subsets
        print(f"x_subset shape: {x_subset.shape}")
        print(f"y_subset shape: {y_subset.shape}")
        print(f"bed_subset shape: {bed_subset.shape}")
        print(f"thickness_subset shape: {thickness_subset.shape}")

        # Check if subsets are empty
        if x_subset.size == 0 or y_subset.size == 0:
            print("Error: Subregion extraction returned empty data.")
            return

        # Write the data to a new NetCDF file
        with nc.Dataset(output_ncfile, 'w') as new_dataset:
            new_dataset.createDimension('x', len(x_subset))
            new_dataset.createDimension('y', len(y_subset))

            new_dataset.createVariable('x', 'f4', ('x',))
            new_dataset.createVariable('y', 'f4', ('y',))
            new_dataset.createVariable('bed', 'f4', ('y', 'x'))
            new_dataset.createVariable('thickness', 'f4', ('y', 'x'))

            new_dataset.variables['x'][:] = x_subset
            new_dataset.variables['y'][:] = y_subset
            new_dataset.variables['bed'][:] = bed_subset
            new_dataset.variables['thickness'][:] = thickness_subset

            print(f"Subregion data saved to {output_ncfile}")
    except Exception as e:
        print(f"Error in preprocess_subregion_ncONLY: {e}")


def NC_to_HDF5(input_ncfile, output_hdf5file):
    try:
        print("Converting NetCDF to HDF5...")

        with nc.Dataset(input_ncfile, 'r') as nc_file:
            x = nc_file.variables['x'][:]
            y = nc_file.variables['y'][:]
            bed = nc_file.variables['bed'][:]
            thickness = nc_file.variables['thickness'][:]

        with h5py.File(output_hdf5file, 'w') as hdf_file:
            hdf_file.create_dataset('x', data=x)
            hdf_file.create_dataset('y', data=y)
            hdf_file.create_dataset('bed', data=bed)
            hdf_file.create_dataset('thickness', data=thickness)

        print(f"HDF5 file successfully written as: {output_hdf5file}")
    except Exception as e:
        print(f"Error in NC_to_HDF5: {e}")

# Main function
input_file = "BedMachineGreenland-v5.nc"
output_file = "output_subregion.hdf5"
output_file_netCDF = "output_subregion.nc"

# Ensure output directory exists
output_dir = os.path.dirname(output_file_netCDF)

# Check if the directory is empty (meaning the output file path does not have a directory part)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get user input for bounding box coordinates
# x1, y1, x2, y2 = get_user_input()
x1 = -100000
y1 =-2000000
x2 = 100000
y2 = -1500000

# Process the NetCDF and convert to HDF5
preprocess_subregion_ncONLY(input_file, output_file_netCDF, x1, y1, x2, y2)
NC_to_HDF5(output_file_netCDF, output_file)

with h5py.File('output_subregion.hdf5', 'r') as f:
    print("x shape:", f['x'].shape)
    print("y shape:", f['y'].shape)
    print("bed shape:", f['bed'].shape)
    print("thickness shape:", f['thickness'].shape)
    print("x data:", f['x'][:][:10])  # Print first 10 values of x
    print("y data:", f['y'][:][:10])  # Print first 10 values of y
    print("bed data:", f['bed'][:10, :10])  # Print first 10x10 block of bed data
    print("thickness data:", f['thickness'][:10, :10])  # First 10x10 block of thickness data

with h5py.File('output_subregion.hdf5', 'r') as f:
    for key in f.keys():
        print(key, f[key].shape)
