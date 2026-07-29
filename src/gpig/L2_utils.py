'''
Max Danenhower

This file provides methods to help retrieve Rrs data from the PACE Satellite, use that data to estimate chlorophyll a, cholorphyll b, 
chlorophyll c1+c2, and photoprotective carotenoids (PPC) concentrations using an inversion method, and plot a visualization of those 
pigment concentrations on a color map. These methods uses PACE's level 2 apparent optical properties (AOP) files, which include Rrs data
and their associate uncertainties. Level 2 files contain data from one swath of the PACE satellite, meaning the data are confined to 
the area of the swath. Level 2 files have 1km resolution. 
'''

import sys

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr
import earthaccess

from .rrs_inversion_pigments import rrs_inversion_pigments


def load_data(tspan, bbox):
    '''
    Downloads one L2 PACE apparent optical properties (AOP) file that intersects the coordinate box passed in, as well as 
    temperature and salinity files. Data files are saved to local folders named 'L2_rrs_data', 'sal_data', and 'temp_data'.

    Parameters:
    -----------
    tspan : tuple of str
        A tuple containing two strings both with format 'YYYY-MM-DD'. The first date in the tuple must predate the second date in the tuple.
    bbox : tuple of floats or ints
        A tuple representing spatial bounds in the form (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).

    Returns:
    --------
    list
        A list of file paths to a PACE L2 AOP files intersecting the passed in bounding box.
    string
        A single file path to a salinity file.
    string
        A single file path to a temperature file.
    '''

    L2_results = earthaccess.search_data(
        short_name='PACE_OCI_L2_AOP',
        bounding_box=bbox,
        temporal=tspan
    )
    if (len(L2_results) > 0):
        L2_paths = earthaccess.download(L2_results, 'L2_rrs_data')
    else:
        L2_paths = []
        print('No L2 AOP data found')

    sal_results = earthaccess.search_data(
        short_name='SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5',
        temporal=tspan,
        count=1
    )
    if (len(sal_results) > 0):
        sal_paths = earthaccess.download(sal_results, 'sal_data')
    else:
        sal_paths = []
        print('No salinity data found')

    temp_results = earthaccess.search_data(
        short_name='MUR-JPL-L4-GLOB-v4.1',
        temporal=tspan,
        count=1
    )
    if (len(temp_results) > 0):
        temp_paths = earthaccess.download(temp_results, 'temp_data')
    else:
        temp_paths = []
        print('No temperature data found')

    return L2_paths, sal_paths[0], temp_paths[0]

def get_granule_boundary(L2_files):
    '''
    Print out boundaries for each L2 file

    Parameters:
    -----------
    L2_files : list of file paths
        A list of L2 file paths
    '''
    if len(L2_files) < 0:
        print('File list is empty.')
        return
    for i,L2_path in enumerate(L2_files):
        dataset = xr.open_dataset(L2_path, group="navigation_data")
        dataset = dataset.set_coords(("longitude", "latitude"))

        n_bound = dataset.latitude.values.max()
        s_bound = dataset.latitude.values.min() 
        e_bound = dataset.longitude.values.max()
        w_bound = dataset.longitude.values.min()

        print('file name:',L2_path)
        print('file index:',i)
        print(f'granule bounding box (w, s, e, n): ({w_bound}, {s_bound}, {e_bound}, {n_bound})')
        print('-----------')

def estimate_inv_pigments(L2_path, sal_path, temp_path, bbox=None):
    '''
    Uses the rrs_inversion_pigments algorithm to calculate chlorophyll a (Chla), chlorophyll b (Chlb), chlorophyll c1
    +c2 (Chlc12), and photoprotective carotenoids (PPC) given an Rrs spectra, salinity, and temperature. Relies on user input to 
    create a boundary box to estimate pigments for. Pigment values are in units of mg/m^3. 

    See rrs_inversion_pigments file for more information on the inversion estimation method.

    Parameters:
    -----------
    L2_path : str
        A single file path to a PACE L2 AOP file.
    sal_path : str
        A single file path to a salinity file.
    temp_path : str
        A single file path to a temperature file.
    bbox : tuple of floats or ints
        A tuple representing spatial bounds in the form (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).
        Default is None, which means the entire L2 granule will be processed

    Returns:
    --------
    xarray.Dataset
        Dataset containing the Chla, Chlb, Chlc, and PPC concentration at each lat/lon coordinate
    '''

    # define wavelengths
    sensor_band_params = xr.open_dataset(L2_path, group='sensor_band_parameters')
    wavelength_coords = sensor_band_params.wavelength_3d.values
    
    dataset = xr.open_dataset(L2_path, group='geophysical_data')
    rrs = dataset[['Rrs', 'Rrs_unc']]

    # Add latitude and longitude coordinates to the Rrs and Rrs uncertainty datasets
    dataset = xr.open_dataset(L2_path, group="navigation_data")
    dataset = dataset.set_coords(("longitude", "latitude"))
    rrs = xr.merge((rrs, dataset.coords))

    if bbox is not None:
        w,s,e,n = bbox[0],bbox[1],bbox[2],bbox[3]

        print(L2_path)
        print(w,s,e,n)

        try:
            rrs_box = rrs.where(
                (
                    (rrs["latitude"] > s)
                    & (rrs["latitude"] < n)
                    & (rrs["longitude"] < e)
                    & (rrs["longitude"] > w)
                ),
                drop=True,
            )
        except ValueError:
            print('Boundary box is outside of the granule boundary.')
            raise
    else:
        n = dataset.latitude.values.max()
        s = dataset.latitude.values.min() 
        e = dataset.longitude.values.max()
        w = dataset.longitude.values.min()

        rrs_box = rrs.where(
            (
                (dataset["latitude"] > s)
                & (dataset["latitude"] < n)
                & (dataset["longitude"] < e)
                & (dataset["longitude"] > w)
            ),
            drop=True,
        )

    # mesh salinity and temperature onto the same coordinate system as Rrs and Rrs uncertainty
    sal = xr.open_dataset(sal_path)
    sal = sal["smap_sss"]
    sal = sal.interp(longitude=rrs_box.longitude, latitude=rrs_box.latitude, method='nearest')    
    temp = xr.open_dataset(temp_path)
    temp = temp['analysed_sst'].squeeze() # get rid of extra time dimension
    temp = temp.interp(lon=rrs_box.longitude, lat=rrs_box.latitude, method='nearest')
    temp = temp - 273 # convert from kelvin to celcius
    
    rrs_box['tchla'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size),np.nan))
    rrs_box['tchlb'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size),np.nan))
    rrs_box['chlc12'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size),np.nan))
    rrs_box['ppc'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size),np.nan))

    progress = 1 # keeps track of how many pixels have been calculated
    pixels = rrs_box.number_of_lines.size * rrs_box.pixels_per_line.size

    # for each coordinate estimate the pigment concentrations
    for i in range(rrs_box.sizes["number_of_lines"]):
        for j in range(rrs_box.sizes["pixels_per_line"]):
            # prints total number of pixels and how many have been estimated already
            sys.stdout.write('\rProgress: ' + str(progress) + '/' + str(pixels))
            sys.stdout.flush()
            progress += 1

            r = rrs_box["Rrs"][i][j].to_numpy()
            ru = rrs_box["Rrs_unc"][i][j].to_numpy()
            sal_val = sal[i][j].values.item()
            temp_val = temp[i][j].values.item()
            if not (np.isnan(r[0]) or np.isnan(sal_val) or np.isnan(temp_val)):
                pigs = rrs_inversion_pigments(r, ru, wavelength_coords, temp_val, sal_val)[0]
                rrs_box['tchla'][i][j] = pigs[0]
                rrs_box['chlc12'][i][j] = pigs[1]
                rrs_box['tchlb'][i][j] = pigs[2]
                rrs_box['ppc'][i][j] = pigs[3]
    
    return rrs_box[['tchla', 'tchlb', 'chlc12', 'ppc']]

def plot_pigments(data, lower_bound, upper_bound):
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
    plt.show()
