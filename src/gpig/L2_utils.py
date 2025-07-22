'''
Max Danenhower

This file provides methods to help retrieve Rrs data from the PACE Satellite, use that data to estimate chlorophyll a, cholorphyll b, 
chlorophyll c1+c2, and photoprotective carotenoids (PPC) concentrations using an inversion method, and plot a visualization of those 
pigment concentrations on a color map. These methods uses PACE's level 2 apparent optical properties (AOP) files, which include Rrs data
and their associate uncertainties. Level 2 files contain data from one swath of the PACE satellite, meaning the data are confined to 
the area of the swath. Level 2 files have 1km resolution. 
'''

import sys
import os
import re
from datetime import datetime, timedelta

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr
import earthaccess
import ray

from .rrs_inversion_pigments import rrs_inversion_pigments


def load_data(tspan, bbox):
    '''
    Downloads one L2 PACE apparent optical properties (AOP) file that intersects the coordinate box passed in, as well as 
    temperature and salinity files. Data files are saved to local folders named 'L2_data', 'sal_data', and 'temp_data'.

    Parameters:
    -----------
    tspan : tuple of str
        A tuple containing two strings both with format 'YYYY-MM-DD'. The first date in the tuple must predate the second date in the tuple.
    bbox : tuple of floats or ints
        A tuple representing spatial bounds in the form (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).

    Returns:
    --------
    L2_path : string
        A single file path to a PACE L2 AOP file.
    sal_path : string
        A single file path to a salinity file.
    temp_path : string
        A single file path to a temperature file.
    '''

    L2_results = earthaccess.search_data(
        short_name='PACE_OCI_L2_AOP_NRT',
        bounding_box=bbox,
        temporal=tspan,
        count=1
    )

    if (len(L2_results) > 0):
        L2_paths = earthaccess.download(L2_results, 'L2_data')
    else:
        raise Exception('No L2 PACE AOP data found')

    sal_results = earthaccess.search_data(
        short_name='SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5',
        temporal=tspan,
        count=1
    )
    if (len(sal_results) > 0):
        sal_paths = earthaccess.download(sal_results, 'sal_data')
    else:
        raise Exception('No salinity data found')

    temp_results = earthaccess.search_data(
        short_name='MUR-JPL-L4-GLOB-v4.1',
        temporal=tspan,
        count=1
    )
    if (len(temp_results) > 0):
        temp_paths = earthaccess.download(temp_results, 'temp_data')
    else:
        temp_paths = []
        raise Exception('No temperature data found')

    return L2_paths[0], sal_paths[0], temp_paths[0]

def interpolate_coords(rrs_path, sal_path, temp_path):
    '''
    Interpolate the salinity and temperature data coordinates onto the PACE L2 Rrs coordinates

    Parameters:
    -----------
    L2_path : str
        A single file path to a PACE L2 AOP file.
    sal_path : str
        A single file path to a salinity file.
    temp_path : str
        A single file path to a temperature file.

    Returns:
    --------
    rrs_box, rrs_unc_box, wavelength_coords, sal, temp : Xarrays all on the same lat/lon coordinates (except wavelength_coords, which is a 1D array)
    '''

    # define wavelengths
    sensor_band_params = xr.open_dataset(rrs_path, group='sensor_band_parameters')
    wavelength_coords = sensor_band_params.wavelength_3d.values
    
    dataset = xr.open_dataset(rrs_path, group='geophysical_data')
    rrs = dataset['Rrs']
    rrs_unc = dataset['Rrs_unc']

    # Add latitude and longitude coordinates to the Rrs and Rrs uncertainty datasets
    dataset = xr.open_dataset(rrs_path, group="navigation_data")
    dataset = dataset.set_coords(("longitude", "latitude"))
    dataset_r = xr.merge((rrs, dataset.coords))
    dataset_ru = xr.merge((rrs_unc, dataset.coords))

    n_bound = dataset_r.latitude.values.max()
    s_bound = dataset_r.latitude.values.min() 
    e_bound = dataset_r.longitude.values.max()
    w_bound = dataset_r.longitude.values.min()

    print('north',n_bound,'south',s_bound,'east',e_bound,'west',w_bound)

    rrs_box = dataset_r["Rrs"].where(
        (
            (dataset["latitude"] > s_bound) # southern boundary latitude
            & (dataset["latitude"] < n_bound) # northern boundary latitude
            & (dataset["longitude"] < e_bound) # eastern boundary latitude
            & (dataset["longitude"] > w_bound) # western boundary latitude
        ),
        drop=True,
    )

    rrs_unc_box = dataset_ru["Rrs_unc"].where(
        (
            (dataset["latitude"] > s_bound) # southern boundary latitude
            & (dataset["latitude"] < n_bound) # northern boundary latitude
            & (dataset["longitude"] < e_bound) # eastern boundary latitude
            & (dataset["longitude"] > w_bound) # western boundary latitude
        ),
        drop=True,
    )

    # Get the filename only
    filename = os.path.basename(rrs_path)

    # Extract the date and parse the month
    match = re.search(r"\.(\d{8})T", filename)
    if match:
        date_str = match.group(1)
        month = date_str[4:6]

    sss_key = 'sss' + month
    sst_key = 'data' + month

    # use climatology files
    sal = xr.open_dataset(sal_path)
    sal[sss_key] = sal[sss_key].assign_coords({
        'Number of Latitudes': sal['Latitude'],
        'Number of Longitudes': sal['Longitude']
    })

    sal = sal.rename({
        'Number of Latitudes': 'lat',
        'Number of Longitudes': 'lon'
    })

    # re-align longitude coords to -180 to 180 
    sal = sal.assign_coords({
        "lon": (((sal.lon + 180) % 360) - 180)
    })

    sal = sal.sortby('lon')

    sal = sal[sss_key].sel({"lat": slice(s_bound, n_bound), "lon": slice(w_bound, e_bound)})

    temp = xr.open_dataset(temp_path)
    temp_lat_dim = 2 * (int(month)-1)
    temp_lon_dim = temp_lat_dim + 1
    
    dim1 = 'fakeDim' + str(temp_lat_dim)
    dim2 = 'fakeDim' + str(temp_lon_dim)
    temp = temp.rename({dim1: 'Latitude', dim2: 'Longitude'})

    temp = temp[sst_key].sel({"Latitude": slice(n_bound, s_bound), "Longitude": slice(w_bound, e_bound)})

    # mesh salinity and temperature onto the same coordinate system as Rrs and Rrs uncertainty
    sal = sal.interp(lon=rrs_box.longitude, lat=rrs_box.latitude, method='nearest')
    temp = temp.interp(Longitude=rrs_box.longitude, Latitude=rrs_box.latitude, method='nearest')
    temp_slope = temp.slope
    temp_intercept = temp.intercept
    temp = temp*temp_intercept + temp_slope
    print(temp)

    rrs_box['chla'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size), np.nan))
    rrs_box['chlb'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size), np.nan))
    rrs_box['chlc'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size), np.nan))
    rrs_box['ppc'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size), np.nan))

    return rrs_box, rrs_unc_box, wavelength_coords, sal, temp

@ray.remote(num_cpus=1)
def run_batch(rrs_batch,rrs_unc_batch,wl,temp_batch,sal_batch):

    results = []
    
    for i in range(rrs_batch.shape[0]):
        if np.isnan(rrs_batch[i][0]) or np.isnan(sal_batch[i]) or np.isnan(temp_batch[i]):
            pigs = np.array([np.nan,np.nan,np.nan,np.nan])
            results.append(pigs)
        else:
            rrs = rrs_batch[i]
            rrs_unc = rrs_unc_batch[i]
            sal = sal_batch[i]
            temp = temp_batch[i]

            pigs = rrs_inversion_pigments(rrs,rrs_unc,wl,float(temp),float(sal))[0]
            results.append(pigs)

    return results


def plot_pigments(data, lower_bound, upper_bound, title):
    '''
    Plots the pigment data from an L2 file with lat/lon coordinates using a color map

    Paramaters:
    -----------
    data : Xarray data array
        Contains pigment values to be plotted.
    lower_bound : float
        The lowest value represented on the color scale.
    upper_bound : float
        The upper value represented on the color scale.
    '''

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, cmap.N))
    colors = np.vstack((np.array([1, 1, 1, 1]), colors)) 
    custom_cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(np.linspace(lower_bound, upper_bound, cmap.N)), ncolors=custom_cmap.N) 

    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels={"left": "y", "bottom": "x"})
    data.plot(x="longitude", y="latitude", cmap=custom_cmap, ax=ax, norm=norm)
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
    plt.title(title)
    plt.show()