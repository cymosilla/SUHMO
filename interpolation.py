from pathlib import Path
import scipy
from scipy.ndimage import zoom
from scipy.interpolate import interp2d, interp1d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import netCDF4 as nc
import h5py
from scipy.interpolate import griddata
# import xarray as xr
# from osgeo import ogr, osr

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

def interpolate_data(data, shape):
    """
    Interpolate the data to the given shape.
    """
    factors = [n / float(o) for n, o in zip(shape, data.shape)]
    return zoom(data, factors)

def preprocess(output_nc, bedmachine_nc, measures_nc):
    C_MAX = 1.0e+4 # maximum value for C
    C_MIN = 1.0e+1 # minimum value for C

    #desired dimensions
    nx = 5120*2
    ny = 9216*2
    
    ncbm = nc.Dataset(bedmachine_nc, 'r')
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
    ncout = nc.Dataset(output_nc,'w')
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

# NEW test func

def preprocess_bedmachine(input_ncfile, output_ncfile, x1, y1, x2, y2, new_x_res=100, new_y_res=100):
    # Step 1: Read netCDF data
    dataset = nc.Dataset(input_ncfile, 'r')
    x = dataset.variables['x'][:]  # X-coordinates
    y = dataset.variables['y'][:]  # Y-coordinates
    bed = dataset.variables['bed'][:]  # Bed elevation
    thickness = dataset.variables['thickness'][:]  # Ice thickness

    # Step 2: Define bounding box
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    # Filter x and y to match the bounding box
    x_indices = np.where((x >= x_min) & (x <= x_max))[0]
    y_indices = np.where((y >= y_min) & (y <= y_max))[0]
    
    x_subset = x[x_indices]
    y_subset = y[y_indices]
    bed_subset = bed[np.ix_(y_indices, x_indices)]
    thickness_subset = thickness[np.ix_(y_indices, x_indices)]

    # Step 3: Create a new fine grid for interpolation
    x_new = np.linspace(x_min, x_max, new_x_res)
    y_new = np.linspace(y_min, y_max, new_y_res)

    # Step 4: Interpolate the bed elevation and ice thickness -  interp2d IS NOW LEGACY
    # bed_interp_func = interp2d(x_subset, y_subset, bed_subset, kind='linear', bounds_error=False, fill_value=None)
    # thickness_interp_func = interp2d(x_subset, y_subset, thickness_subset, kind='linear', bounds_error=False, fill_value=None)

    bed_interp_func_x = interp1d(x_subset, bed_subset, kind='linear', axis=1, fill_value='extrapolate')
    bed_interp_func_y = interp1d(y_subset, bed_subset, kind='linear', axis=0, fill_value='extrapolate')

    thickness_interp_func_x = interp1d(x_subset, thickness_subset, kind='linear', axis=1, fill_value='extrapolate')
    thickness_interp_func_y = interp1d(y_subset, thickness_subset, kind='linear', axis=0, fill_value='extrapolate')

    # Step 5: Interpolate data onto new grid
    bed_interp_data = bed_interp_func_y(bed_interp_func_x(x_new))
    thickness_interp_data = thickness_interp_func_y(thickness_interp_func_x(x_new))
    with nc.Dataset(output_ncfile, 'w') as new_dataset:
        # Create dimensions
        new_dataset.createDimension('x', len(x_new))
        new_dataset.createDimension('y', len(y_new))
        
        # Create variables
        x_var = new_dataset.createVariable('x', np.float32, ('x',))
        y_var = new_dataset.createVariable('y', np.float32, ('y',))
        bed_var = new_dataset.createVariable('bed', np.float32, ('y', 'x'))
        thickness_var = new_dataset.createVariable('thickness', np.float32, ('y', 'x'))

        # Assign data to variables
        x_var[:] = x_new
        y_var[:] = y_new
        bed_var[:] = bed_interp_data
        thickness_var[:] = thickness_interp_data

    print(f"Interpolated data saved to {output_ncfile}")

# Does not work
def interpolate_GT_latlon():
    dataset = nc.Dataset('Geothermal.nc')

    geothermal_flux = dataset.variables['gt'][:]
    lat = dataset.variables['lat'][:]
    lon = dataset.variables['lon'][:]

    roi_lat_min, roi_lat_max = 65, 66
    roi_lon_min, roi_lon_max = -40, -35

    # Subset the data for the Helheim glacier region
    roi_lat_mask = (lat >= roi_lat_min) & (lat <= roi_lat_max)
    roi_lon_mask = (lon >= roi_lon_min) & (lon <= roi_lon_max)
    roi_bed_elevation = geothermal_flux[roi_lat_mask, roi_lon_mask]

    # Define the grid for interpolation
    grid_lat, grid_lon = np.meshgrid(np.linspace(roi_lat_min, roi_lat_max, 100),
                                    np.linspace(roi_lon_min, roi_lon_max, 100))

    # Perform interpolation
    interpolated_bed_elevation = griddata((lat[roi_lat_mask], lon[roi_lon_mask]),
                                        roi_bed_elevation, (grid_lat, grid_lon), method='cubic')

    # Now interpolated_bed_elevation contains the interpolated data

# Example usage
input_file = "BedMachineGreenland-v5.nc"
output_file = "output_interpolated.nc"
x1, y1, x2, y2 = -500000, -500000, 500000, 500000  # Example bounding box
preprocess_bedmachine(input_file, output_file, x1, y1, x2, y2)