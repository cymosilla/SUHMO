import xarray as xr

file_path = 'BedMachineGreenland-v5.nc'
ds = xr.open_dataset(file_path)
print(ds)

x_min, x_max = 1000, 2000 
y_min, y_max = 1500, 2500  
subregion = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))

# Select the subregion
subregion = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))

# Inspect the subregion
print(subregion)

# Define encoding if needed
encoding = {var: {'dtype': 'float32'} for var in subregion.data_vars}

# Save the subregion to a new NetCDF file with encoding
subregion.to_netcdf('asNew.nc', encoding=encoding)

# Close the dataset
ds.close()

# subregion.to_netcdf('asNew.nc')
# test = 'asNew.nc'
# print(test)

