'''
Max Danenhower

This file provides methods to help retrieve level 3 Rrs data from the PACE Satellite, use that data to estimate chlorophyll a, cholorphyll b, 
chlorophyll c1+c2, and photoprotective carotenoids (PPC) concentrations using an inversion method, and plot a visualization of those 
pigment concentrations on a color map. Also includes a method to estimate chlorophyll b, c1+c2, and PPC pigments by donwloading chlorophyll a
data from PACE and then applying a covariation method. 
'''

import os
import re

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr
import earthaccess
import ray

from .rrs_inversion_pigments import rrs_inversion_pigments

def load_data(tspan):
    '''
    Downloads Remote Sensing Reflectance (Rrs) data from the PACE Satellite, as well as salinity and temperature data (from different 
    missions), and saves the data files to local folders named 'rrs_data', 'sal_data', and 'temp_data'.

    Parameters:
    -----------
    tspan : tuple of str
        A tuple containing two strings both with format 'YYYY-MM-DD'. The first date in the tuple must predate the second date in the tuple.
    resolution : str
        The resolution of data being retrieved. Must be either '1deg', '0p1deg', or '4km'.

    Returns:
    --------
    rrs_paths : list
        A list containing the file path(s) to the downloaded Rrs PACE files.
    sal_paths : list
        A list containing the file path(s) to the downloaded salinity files.
    temp_paths : list
        A list containing the file path(s) to the downloaded temperature files.
    '''
    rrs_results = earthaccess.search_data(
        short_name='PACE_OCI_L3M_RRS_NRT',
        temporal=tspan,
        granule_name='*.DAY.*.Rrs.4km.*',
        count=1
    )
    if (len(rrs_results) > 0):
        rrs_paths = earthaccess.download(rrs_results, 'rrs_data')
    else:
        raise Exception('No L3 PACE Rrs data found')

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
        raise Exception('No temperature data found')

    return rrs_paths[0], sal_paths[0], temp_paths[0]

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

def plot_pigments(data, lower_bound, upper_bound, label):
    '''
    Plots the pigment data from an L3 file with lat/lon coordinates using a color map

    Paramaters:
    -----------
    data : Xarray data array
        Contains pigment values to be plotted.
    lower_bound : float
        The lowest value represented on the color scale.
    upper_bound : float
        The upper value represented on the color scale.
    label : string
        A label for the graph.
    '''

    data.attrs["long_name"] = label

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, cmap.N))
    colors = np.vstack((np.array([1, 1, 1, 1]), colors)) 
    custom_cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(np.linspace(lower_bound, upper_bound, cmap.N)), ncolors=custom_cmap.N) 

    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels={"left": "y", "bottom": "x"})
    data.plot(cmap=custom_cmap, ax=ax, norm=norm)
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
    plt.show()

def interpolate_data(rrs_paths, sal_paths, temp_paths, bbox):
    '''
    Creates an xarray data array with latitude and longitude coordinates. Each coordinate contains a hyperspectral Rrs spectra with 
    corresponding wavelenghts, salinity, and temperature. If more than one file for Rrs, salinity, or temperature are given, uses the 
    date averaged values. 

    Parameters:
    -----------
    rrs_paths : list or str
        A single file path to a PACE Rrs file or a list of file paths to PACE Rrs files.
    sal_paths : list or str
        A single file path to a salinity file or a list of file paths to salinity files.
    temp_paths : list or str
        A single file path to a temperature file or a list of file paths to temperature files.
    bbox : tuple of floats or ints
        A tuple representing spatial bounds in the form (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).

    Returns:
    --------
    Xarray data array
        A data array of Rrs values at each wavelength over a specified lat/lon box.

    Raises:
    -------
    TypeError 
        If rrs_paths, sal_paths, or temp_paths is not a string or list.
    '''

    n = bbox[3]
    s = bbox[1]
    e = bbox[2]
    w = bbox[0]
    
    # creates a dataset of rrs values of the given file
    rrs_data = xr.open_dataset(rrs_paths)
    rrs = rrs_data["Rrs"].sel({"lat": slice(n, s), "lon": slice(w, e)})

    rrs_unc = rrs*0.05

    # Get the filename only
    filename = os.path.basename(rrs_paths)

    # Extract the date and parse the month
    match = re.search(r"\.(\d{8})T", filename)
    if match:
        date_str = match.group(1)
        month = date_str[4:6]

    sss_key = 'sss' + month
    sst_key = 'data' + month

    # use climatology files
    sal = xr.open_dataset(sal_paths)
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

    sal = sal[sss_key].sel({"lat": slice(s, n), "lon": slice(w, e)})

    temp = xr.open_dataset(temp_paths)
    temp_lat_dim = 2 * (int(month)-1)
    temp_lon_dim = temp_lat_dim + 1
    
    dim1 = 'fakeDim' + str(temp_lat_dim)
    dim2 = 'fakeDim' + str(temp_lon_dim)
    temp = temp.rename({dim1: 'Latitude', dim2: 'Longitude'})

    temp = temp[sst_key].sel({"Latitude": slice(n, s), "Longitude": slice(w, e)})

    # mesh salinity and temperature onto the same coordinate system as Rrs and Rrs uncertainty
    sal = sal.interp(lon=rrs.lon, lat=rrs.lat, method='nearest')
    temp = temp.interp(Longitude=rrs.lon, Latitude=rrs.lat, method='nearest')
    temp = temp.slope * temp + temp.intercept

    return rrs, rrs_unc, rrs.wavelength, sal, temp


    



        

