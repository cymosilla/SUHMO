from pathlib import Path
import scipy
# from scipy.ndimage import zoom
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import netCDF4 as nc
import h5py
import xarray as xr
# import subprocess
# import os 
'''
This is the messy test file. For preprocessing, refer to preprocess.py & the README.

Acknowledgements:
- NOAA/NESDIS/STAR Aerosols and Atmospheric Composition Science Team
- Dr. Krti Tallam

All in the same .py file with different functions
0. Read
1. Cut file in NetCDF format
2. Coarsen OR interpolate (refinement) - Interpolate most likely
2.5 (OPTIONAL) Coarsen - final grid size based on user input
NOTE: Should NOT be needed for BM, as it is already the full-size grid. 
A separate coarse function would need to be done if getting a small BM dataset to a large one.
3. Dump cut file in NetCDF instead of creating new NetCDF file 
4. Conversion of .nc dumped file to HDF
5. Read the HDF file inside SUHMO (not preprocess, extra step)

Order of development: 0, 2, 1, 3, 4, 2.5

Input: .nc file (for now)
Output: HDF5 file
'''

file_path_BM = "BedMachineGreenland-v5.nc" # BedMachine file
file_path_GF = 'test' #Geothermal Flux file

###########################################################
# Step 0: Read data

# BedMachine is in .nc format
def read_netCDF(file_path):
    dataset = xr.open_dataset(file_path)
    return dataset

# This reads BM .nc & gets bed elevation & ice thickness vars. Returns the dataset
def read_NC_BM(file_path):
    # Read a netCDF file and return the dataset.
    dataset = nc.Dataset(file_path, mode='r')
    x = dataset.variables['x'][:]  # X-coordinates
    y = dataset.variables['y'][:]  # Y-coordinates
    bed = dataset.variables['bed'][:]  # Bed elevation
    thickness = dataset.variables['thickness'][:]  # Ice thickness 
    return dataset

###########################################################
# Step 0.5: User Input (absolute first step)
def get_user_input():
    while True:
        try:
            print("Enter the bounding box coordinates for the rectangular region:")
            x1 = float(input("x1 (lower-left x-coordinate "))
            y1 = float(input("y1 (lower-left y-coordinate): "))
            x2 = float(input("x2 (upper-right x-coordinate): "))
            y2 = float(input("y2 (upper-right y-coordinate): "))
            if x1 == x2 or y1 == y2:
                raise ValueError("Bounding box cannot have zero width or height.")
            return x1, y1, x2, y2
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter numeric values.")

###########################################################
# Step 1: Subregion (Cutting file)
read_BM_output = read_NC_BM('BedMachineGreenland-v5.nc') # Test

def extract_subregion():
    pass
def extract_BM_CDO(input_file, x1, x2, y1, y2):
    output_extracted_file = "bed_elevation_subregion.nc" 

    bed_only_var = "bed" 
    # Grid size = 80000.00 200000.00 

    cdo_command = [
        "cdo", "sellonlatbox," + str(x1) + "," + str(x2) + "," + str(y1) + "," + str(y2),
        "selvar," + bed_only_var,
        input_file, output_extracted_file
    ]
    subprocess.run(cdo_command, check=True)
    print(f"Output saved to {output_extracted_file}")
    return output_extracted_file

def preprocess_subregion_ncONLY(input_ncfile, output_ncfile, x1, y1, x2, y2):
    # Step 1: Read netCDF data
    dataset = nc.Dataset(input_ncfile, 'r')
    x = dataset.variables['x'][:]  
    y = dataset.variables['y'][:]  
    bed = dataset.variables['bed'][:]  
    thickness = dataset.variables['thickness'][:]  

    # Step 2: Define bounding box and extract the subset
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    # RElevant indices based on bounding box (user inputted) Cartesian coords
    x_indices = np.where((x >= x_min) & (x <= x_max))[0]
    y_indices = np.where((y >= y_min) & (y <= y_max))[0]
    
    # Extract subsets from x/y_indices
    x_subset = x[x_indices]
    y_subset = y[y_indices]
    bed_subset = bed[np.ix_(y_indices, x_indices)]
    thickness_subset = thickness[np.ix_(y_indices, x_indices)]

    # Step 3: NetCDF to NetCDF conversion
    # Check with ncview
    with nc.Dataset(output_ncfile, 'w') as new_dataset:
        new_dataset.createDimension('x', len(x_subset))
        new_dataset.createDimension('y', len(y_subset))

        # Second parameter is how to store
        # F4 indicates to store by floating point values 4 points away
        new_dataset.createVariable('x', 'f4', ('x',))
        new_dataset.createVariable('y', 'f4', ('y',))
        new_dataset.createVariable('bed', 'f4', ('y', 'x'))
        new_dataset.createVariable('thickness', 'f4', ('y', 'x'))
        
        # Creates separate arrays per variable
        new_dataset.variables['x'][:] = x_subset
        new_dataset.variables['y'][:] = y_subset
        new_dataset.variables['bed'][:] = bed_subset
        new_dataset.variables['thickness'][:] = thickness_subset

    print(f"Subregion data saved to {output_ncfile}")

    
###########################################################
# Step 2: Interpolate (refine) data 
# interpolation_input = extract_BM_CDO(read_BM_output)
# interpolation_output = "bed_elevation_region_interpolated.nc"  

# For Bed thickness only
def interpolate_BM_bed_CDO(input_file):
    output_interpolated_file = "bed_elevation_subregion.nc" 
    variable_name = "bed"  
    x1, x2 = -50, 50   
    y1, y2 = -85, -60  

    grid_definition = "grid_description.txt"  

    # Grid definition file (explained by griddata)
    with open(grid_definition, 'w') as grid_file:
        grid_file.write(
            "gridtype = lonlat\n"
            "xsize = 100\n"      
            "ysize = 50\n"       
            "xfirst = -50\n"     
            "xinc = 1.0\n"       
            "yfirst = -85\n"     
            "yinc = 0.5\n"       
        )

    cdo_command = (
        f"cdo remapbil,{grid_definition} "  
        f"-selvar,{variable_name} "
        f"-sellonlatbox,{x1},{x2},{y1},{y2} "
        f"{input_file} {output_interpolated_file}"
    )

    os.system(cdo_command)

    print(f"Extracted and interpolated output saved to {output_interpolated_file}")

def interpolate_BM_bed_preprocess():
    C_MAX = 1.0e+4 # maximum value for C
    C_MIN = 1.0e+1 # minimum value for C

    #desired dimensions
    nx = 5120*2
    ny = 9216*2
    
    ncbm = Dataset(bedmachine_nc, 'r')
    xbm = ncbm.variables["x"][:]
    ybm = np.flipud(ncbm.variables["y"][:])
    
    nxbm,nybm = len(xbm),len(ybm)
    topg = np.zeros((ny, nx))
    thk = np.zeros((ny, nx))
    usrf_bm = np.zeros((ny, nx))
    mask = np.zeros((ny, nx))
    umod = np.zeros((ny, nx))

    print ('xbm[0] = {}, ybm[0] = {}'.format(xbm[0],ybm[0]))

    #bed machine data dimensions
    dx = xbm[1] - xbm[0]
    
    #desired data dimensions
    tol = 1.0e-10
    x = np.arange(xbm[0],xbm[0]+nx*dx+tol,dx)
    y = np.arange(ybm[0],ybm[0]+ny*dx+tol,dx)


    #bedmachine data
    topg[0:nybm,0:nxbm] =  np.flipud(ncbm.variables["bed"][:,:])
    thk[0:nybm,0:nxbm]  =  np.flipud(ncbm.variables["thickness"][:,:])
    usrf_bm[0:nybm,0:nxbm]  =  np.flipud(ncbm.variables["surface"][:,:])
    mask[0:nybm,0:nxbm]  = np.flipud(ncbm.variables["mask"][:,:])

    #speed data
    umod[0:nybm,0:nxbm]  = read_umod_mouginot(xbm,ybm,measures_nc)

    #thickness/bedrock mods 
    thk = remove_islands(thk,mask)
    
    #thk,topg = patch_holes(x,y,thk, topg, usrf_bm, mask)

    #raise ValueError('enough for now')
    
    #dependents
    eps = 1.0e-6
    rhoi = 917.0
    rhoo = 1027.0
    sg = topg + thk
    sf = (1.0 - rhoi/rhoo)*thk
    
    grounded = np.logical_and( thk > eps, sg + eps > sf)
    usrf = np.where( grounded, sg, sf )
        
    
    print ('umod c ...')
    #umodc is the weight w(x,y) in the misfit f_m(x,y) =  w (|u_model| - |u_obs|)^2
    umodc = np.where(umod > 1.0, 1.0, 0.0)
    umodc = np.where(thk > 10.0, umodc, 0.0)

    #surface gradient
    print ('grad s ...')
    usrf = ndimage.gaussian_filter(usrf, 4) # smooth
    grads = zeros_2D(x,y)
    grads[1:ny-1,1:nx-1] = 0.5 / dx *  np.sqrt(
        (usrf[1:ny-1,0:nx-2] - usrf[1:ny-1,2:nx])**2 + 
        (usrf[0:ny-2,1:nx-1] - usrf[2:ny,1:nx-1])**2 )
 
    #initial guess for C
    print ('btrc...')
    btrc = rhoi * 9.81 * grads * thk / (umod + 1.0)
    btrc = np.where(umod > 1, btrc, C_MAX)
    btrc = np.where(btrc < C_MAX, btrc, C_MAX)
    btrc = np.where(btrc > C_MIN, btrc, C_MIN)
    #smooth with slippy bias
    print ('    ...filtering')
    btrcs = ndimage.minimum_filter(btrc, 8)
    btrcs = ndimage.gaussian_filter(btrcs, 32)
    btrc = np.where(btrc < btrcs, btrc, btrcs) # retain slippy spots
    
    #no ice value for C
    btrc = np.where(thk > 0, btrc, 100.0)
    
    #ouput netcdf
    print ('writing ...')
    ncout = Dataset(output_nc,'w')
    #dimensions
    xdim = ncout.createDimension('x',size=nx)
    ydim = ncout.createDimension('y',size=ny)
    #var defs

    xv = ncout.createVariable('x','f8',('x'))
    yv = ncout.createVariable('y','f8',('y'))

    # add_projection_attr_greenland(ncout, xv, yv)

    def create2D(name):
        v = ncout.createVariable(name,'f8',('y','x'))
        v.setncattr('grid_mapping','crs')
        return v

    topgv = create2D('topg')
    thkv = create2D('thk')
    umodv = create2D('umod')
    umodcv = create2D('umodc')
    btrcv = create2D('btrc')
    
    #data
    xv[:] = x
    yv[:] = y
    topgv[:,:] = topg
    thkv[:,:] = thk
    umodv[:,:] = umod
    umodcv[:,:] = umodc
    btrcv[:,:] = btrc

    ncout.close()

    dx = x[1] - x[0]
    print( ' {} < x < {} '.format(np.min(x) - 0.5 * dx, np.max(x) + 0.5*dx))
    dy = y[1] - y[0]
    print( ' {} < y < {} '.format(np.min(y) - 0.5 * dy, np.max(y) + 0.5*dy))
    return 0

def interpolate_BM_bed(input_file, x1, x2, y1, y2):
    return 0

# Test function for interpolation with xarray 
def interpolate_BM_bed_xarray():
    file_path = 'BedMachineGreenland-v5.nc'
    ds = xr.open_dataset(file_path)
    print(ds)

    x_min, x_max = 1000, 2000 
    y_min, y_max = 1500, 2500  
    subregion = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))

    subregion = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))

    print(subregion)

    encoding = {var: {'dtype': 'float32'} for var in subregion.data_vars}

    subregion.to_netcdf('asNew.nc', encoding=encoding)

    ds.close()

###########################################################
# Step 2.5: Coarsen data (from coarsen.py)
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
    fine_nc = nc.Dataset(fine_name,'r')
    coarse_nc = nc.Dataset(name, 'w')
    
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
# Step 3: Write .nc file (still need?)

# This could probably be part of main down below
 
def write_hdf5(file_path, data, dataset_name):
    """
    Write data to an HDF5 file.
    """
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(dataset_name, data=data)

###########################################################
# Step 4: Converison to HDF

def netCDF_HDF_conversion(netcdf_file, hdf5_file):
    # Opens nc file & creates new hdf5 file
    with nc.Dataset(netcdf_file, 'r') as src:
        with h5py.File(hdf5_file, 'w') as dst:
            for name in src.ncattrs():
                dst.attrs[:qname] = src.getncattr(name)
            
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
    return hdf5_file

# def write_hdf5(file_path, data, dataset_name):
#     """
#     Write data to an HDF5 file.
#     """
#     with h5py.File(file_path, 'w') as f:
#         f.create_dataset(dataset_name, data=data)

###########################################################
# Step 5: Read HDF file into SUHMO

#NOT TESTEd
hdf_final_output = netCDF_HDF_conversion(netcdf, hdf_file)
def read_final_HDF(input_HDF):
    # From StackOverflow
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]      # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array

###########################################################
# All Together Now - Combination of all helper functions
def preprocess_BedMachine():
    get_user_input() # Step 0.5: User input
    # Step 1: Subregion
    # Step 2: Interpolate
    # Step 4: .nc --> hdf5

def preprocess_geothermal():
    pass

###########################################################
# # Testing

# # Step 0:
# file_path = 'BedMachineGreenland-v5.nc'
# xrTry = readNConXR(file_path)
# ncTry = read_netcdf(file_path)
# print(xrTry + "\n\n\n" + ncTry)

# # Step 1: Included in inteprolation

# # Step 2:
# interpolate_BM_test = interpolate_BM_bed_CDO(interpolation_input)