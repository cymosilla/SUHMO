import xarray as xr
import netCDF4 as nc

def readNConXR(file):
    dataset = xr.open_dataset(file)
    return dataset

def read_netcdf(file_path):
    """
    Read a netCDF file and return the dataset.
    """
    ds = nc.Dataset(file_path, mode='r')
    return ds

# xrTry = readNConXR('BedMachineGreenland-v5.nc')
# ncTry = read_netcdf('BedMachineGreenland-v5.nc')
# print(xrTry)
# print("\n\n\n\n\n\n\n\n")
# print(ncTry)
# Open the original BedMachine file
dataset = nc.Dataset("BedMachineGreenland-v5.nc", 'r')

# Extract the x and y variables
x = dataset.variables['x'][:]
y = dataset.variables['y'][:]

# Print the min and max values of x and y
print(f"x range: {x.min()} to {x.max()}")
print(f"y range: {y.min()} to {y.max()}")

    # print(file_id.variables['dataid'])
    # print(file_id.variables['bed'])
    # print(file_id.variables['thickness'])
    # print(file_id.variables['errbed'])
    # print(file_id.variables['surface'])
    # # [:,:] obtains data from a 2d format
    # bed = file_id.variables['bed'][:,:]
    # id = file_id.variables['dataid'][:,:]
    # thk = file_id.variables['thickness'][:,:]

'''To test step 1:
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