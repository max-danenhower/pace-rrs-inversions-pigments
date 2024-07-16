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

def load_data(tspan, resolution):
    '''
    Downloads Remote Sensing Reflectance (RRS) data from the PACE Satellite and saves the nc file(s) to a folder named 'data'

    Parameters:
    tspan (tuple of strings): a tuple containing two strings both with format 'YYYY-MM-DD'. The first date in the tuple
    must predate the second date in the tuple.

    resolution (string): the resolution of data being retrieved. Either '1deg', '0.1deg' or '4km'

    Returns:
    An list containig the path(s) to the downloaded PACE files
    '''

    rrs_results = earthaccess.search_data(
        short_name='PACE_OCI_L3M_RRS_NRT',
        temporal=tspan,
        granule_name='*.DAY.*.Rrs.' + resolution + '.*'
    )

    sal_results = earthaccess.search_data(
        short_name='SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5',
        temporal=tspan
    )

    temp_results = earthaccess.search_data(
        short_name='MUR-JPL-L4-GLOB-v4.1',
        temporal=tspan
    )

    if (len(rrs_results) > 0 and len(sal_results) > 0 and len(temp_results) > 0):
        rrs_paths = earthaccess.download(rrs_results, 'rrs_data')
        sal_paths = earthaccess.download(sal_results, 'sal_data')
        temp_paths = earthaccess.download(temp_results, 'temp_data')

        return rrs_paths, sal_paths, temp_paths
    
    print('Missing granules')
    return None

def create_dataset(rrs_paths, sal_paths, temp_paths, n, s, w, e):
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

    if (n < s):
        raise('northern boundary must be greater than southern boundary')
    
    if (e < w):
        raise('eastern boundary must be greater than western boundary')
    
    # creates a dataset of rrs values of the given file
    if isinstance(rrs_paths, str):
        rrs_data = xr.open_dataset(rrs_paths)
        rrs = rrs_data["Rrs"].sel({"lat": slice(n, s), "lon": slice(w, e)})
    elif isinstance(rrs_paths, list):
        # if given a list of files, create a date averaged dataset of Rrs values 
        rrs_data = xr.open_mfdataset(
            rrs_paths,
            combine="nested",
            concat_dim="date"
        )
        rrs = rrs_data["Rrs"].sel({"lat": slice(n, s), "lon": slice(w, e)}).mean('date')
        rrs = rrs.compute()
    else:
        raise TypeError('rrs_paths must be a string or list')

    # creates a dataset of sal and temp values of the given file
    if isinstance(sal_paths, str):
        sal = xr.open_dataset(sal_paths)
        sal = sal["smap_sss"].sel({"latitude": slice(n, s), "longitude": slice(w, e)})
    elif isinstance(sal_paths, list):
        # if given a list of files, create a date averaged dataset of salinity values 
        sal = xr.open_mfdataset(
            sal_paths,
            combine="nested",
            concat_dim="date"
        )
        sal = sal["smap_sss"].sel({"latitude": slice(n, s), "longitude": slice(w, e)}).mean('date')
        sal = sal.compute()
    else:
        raise TypeError('sal_paths must be a string or list')

    # creates a dataset of sal and temp values of the given file
    if isinstance(temp_paths, str):
        temp = xr.open_dataset(temp_paths)
        temp = temp['analysed_sst'].squeeze() # get rid of extra time dimension
        temp = temp.sel({"lat": slice(s, n), "lon": slice(w, e)})
    elif isinstance(temp_paths, list):
        # if given a list of files, create a date averaged dataset of temperature values 
        temp = xr.open_mfdataset(
            temp_paths,
            combine="nested",
            concat_dim="time"
        )
        temp = temp['analysed_sst'].sel({"lat": slice(s, n), "lon": slice(w, e)}).mean('time')
        temp = temp.compute()
    else:
        raise TypeError('temp_paths must be a string or list')

    # merge datasets to Rrs coordinates
    sal = sal.interp(longitude=rrs.lon, latitude=rrs.lat, method='nearest')
    temp = temp.interp(lon=rrs.lon, lat=rrs.lat, method='nearest')

    combined_ds = xr.Dataset(
        {
            "rrs": (["lat", "lon", 'wavelength'], rrs.data),
            'sal': (["lat", "lon"], sal.data),
            'temp': (["lat", "lon"], temp.data)
        },
        coords={
            "lat": rrs.lat,
            "lon": rrs.lon,
            'wavelength': rrs.wavelength
        }
    )

    return combined_ds

        
            
def calculate_pigments(box):
    '''
    Uses the rrs_inversion_pigments algorithm to calculate chlorophyll a (Chla), chlorophyll b (Chlb), chlorophyll c1
    +c2 (Chlc12), and photoprotective carotenoids (PPC) given a remote sensing reflectance spectra. Calculates the pigment 
    values for each lat/lon coordinate in the box's range

    Parameters:
    box (xr dataarray): An xarray data array containing the Rrs for each wavelength, salinity, and temperature at each lat/lon coordinate

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
                rrs = box['rrs'][lat][lon][w].values.item()
                if rrs == 0:
                    Rrs[w] = 0.000001
                    Rrs_unc[w] = 0.000001 * 0.05
                else:
                    Rrs[w] = rrs
                    Rrs_unc[w] = rrs * 0.05 # uncertainty is 5% of rrs value

            sal = box['sal'][lat][lon].data.item()
            temp = box['temp'][lat][lon].data.item() - 273 # convert from kelvin to celcius

            if not (math.isnan(Rrs[0]) or math.isnan(sal) or math.isnan(temp)):
                vals = rrs_inversion_pigments(Rrs, Rrs_unc, wl, temp, sal)
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

def plot_pigments(data, lower_bound, upper_bound, label):
    '''
    Plots the pigment data with lat/lon coordinates using a color map

    Paramaters:
    data (xr data array): An array with pigment values at each lat/lon coordinate
    lower_bound (float): The lowest value represented on the color scale
    upper_bound (float): The upper value represented on the color scale
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




    



        

