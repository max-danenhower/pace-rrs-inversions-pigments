import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr
import earthaccess
import sys
import math
from rrs_inversion_pigments import rrs_inversion_pigments

'''

Max Danenhower
This file provides methods to help retrieve Rrs data from the PACE Satellite, use that data to calculate chlorophyll a, cholorphyll b, 
chlorophyll c, and PPC concentrations, and plot a visualization of those pigment concentrations. 

'''

def load_rrs(tspan, resolution):
    '''
    Downloads Remote Sensing Reflectance (RRS) data from the PACE Satellite and saves the nc file(s) to a folder named 'data'

    Parameters:
    tspan (tuple of strings): a tuple containing two strings both with format 'YYYY-MM-DD'. The first date in the tuple
    must predate the second date in the tuple.

    resolution (string): the resolution of data being retrieved. Either '1deg', '0.1deg' or '4km'

    Returns:
    An array containig the path(s) to the downloaded PACE files
    '''
    results = earthaccess.search_data(
        short_name='PACE_OCI_L3M_RRS_NRT',
        temporal=tspan,
        granule_name='*.DAY.*.Rrs.' + resolution + '.*'
    )

    paths = earthaccess.download(results, 'data')

    return paths

def create_rrs_dataset(paths, n, s, w, e):
    '''
    Creates an xarray data array containing the Rrs at each wavelength for a given set of lat/lon coordinates 

    Parameters:
    paths (array or string): a single file path to a PACE Rrs file or an array of file paths to PACE Rrs files
    n (float): northern boundary of the data array's lat
    s (float): southern boundary of the data array's lat
    w (float): western boundary of the data array's lon
    e (float): western boundary of the data array's lon

    Returns:
    A data array of Rrs values at each wavelength over a specified lat/lon box
    '''

    # returns the Rrs values from a single file
    if isinstance(paths, str):
        dataset = xr.open_dataset(paths)
        return dataset["Rrs"].sel({"lat": slice(n, s), "lon": slice(w, e)})

    # creates a date averaged dataset of Rrs values over the given files
    if isinstance(paths, list):
        dataset = xr.open_mfdataset(
            paths,
            combine="nested",
            concat_dim="date",
        )
        box = dataset["Rrs"].sel({"lat": slice(n, s), "lon": slice(w, e)}).mean('date')
        return box.compute()
    
    print('bad path')
            
def calculate_pigments(box):
    '''
    Uses the rrs_inversion_pigments algorithm to calculate chlorophyll a (Chla), chlorophyll b (Chlb), chlorophyll c1
    +c2 (Chlc12), and photoprotective carotenoids (PPC) given a remote sensing reflectance spectra. Calculates the pigment 
    values for each lat/lon coordinate in the box's range

    Parameters:
    box (xr dataarray): An xarray data array containing the Rrs for each wavelength at each lat/lon coordinate

    Returns:
    An xr dataset containing the Chla, Chlb, Chlc, and PPC concentration at each lat/lon coordinate
    '''
    progress = 1 # keeps track of how many pixels have been calculated
    pixels = box.lat.size * box.lon.size
    print('num pixels: ', pixels)

    chla = np.zeros((box.lat.size, box.lon.size))
    chlb = np.zeros((box.lat.size, box.lon.size))
    chlc = np.zeros((box.lat.size, box.lon.size))
    ppc = np.zeros((box.lat.size, box.lon.size))
    
    for lat in range(box.lat.size):
        for lon in range(box.lon.size):
            sys.stdout.write('\rProgress: ' + str(progress) + '/' + str(pixels))
            sys.stdout.flush()
            progress += 1
            wl = box.wavelength.to_numpy()
            Rrs = np.zeros(len(wl))
            Rrs_unc = np.zeros(len(wl))
            for w in range(len(wl)):
                rrs = box[lat][lon][w].values.item()
                if rrs == 0:
                    Rrs[w] = 0.000001
                    Rrs_unc[w] = 0.000001 * 0.05
                else:
                    Rrs[w] = rrs
                    Rrs_unc[w] = rrs * 0.05 # uncertainty is 5% of rrs value

            if not math.isnan(Rrs[0]):
                #TODO: use real temp and salinity values 
                vals = rrs_inversion_pigments(Rrs, Rrs_unc, wl, 22, 35)
                chla[lat][lon] = vals[0][0]
                chlb[lat][lon] = vals[0][1]
                chlc[lat][lon] = vals[0][2]
                ppc[lat][lon] = vals[0][3]

    print()
    lat_coord = box.lat.to_numpy()
    lon_coord = box.lon.to_numpy()
    chla = xr.DataArray(
        chla,
        dims=['lat', 'lon'],
        coords={'lat': lat_coord, 'lon': lon_coord},
        name='chla'
    )

    chlb = xr.DataArray(
        chlb,
        dims=['lat', 'lon'],
        coords={'lat': lat_coord, 'lon': lon_coord},
        name='chlb'
    )

    chlc = xr.DataArray(
        chlc,
        dims=['lat', 'lon'],
        coords={'lat': lat_coord, 'lon': lon_coord},
        name='chlc'
    )

    ppc = xr.DataArray(
        ppc,
        dims=['lat', 'lon'],
        coords={'lat': lat_coord, 'lon': lon_coord},
        name='ppc'
    )

    return xr.Dataset({'chla': chla, 'chlb': chlb, 'chlc': chlc, 'ppc': ppc})

def plot_pigments(data, lower_bound, upper_bound):
    '''
    Plots the pigment data with lat/lon coordinates using a color map

    Paramaters:
    data (xr data array): An array with pigment values at each lat/lon coordinate
    lower_bound (float): The lowest value represented on the color scale
    upper_bound (float): The upper value represented on the color scale
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
    data.plot(cmap=custom_cmap, ax=ax, norm=norm)
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
    plt.show()


    



        

