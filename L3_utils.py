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

This file provides methods to help retrieve level 3 Rrs data from the PACE Satellite, use that data to estimate chlorophyll a, cholorphyll b, 
chlorophyll c1+c2, and photoprotective carotenoids (PPC) concentrations using an inversion method, and plot a visualization of those 
pigment concentrations on a color map. Also includes a method to estimate chlorophyll b, c1+c2, and PPC pigments by donwloading chlorophyll a
data from PACE and then applying a covariation method. 
'''

def load_data(tspan, resolution):
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
        granule_name='*.DAY.*.Rrs.' + resolution + '.*'
    )
    if (len(rrs_results) > 0):
        rrs_paths = earthaccess.download(rrs_results, 'rrs_data')
    else:
        rrs_paths = []
        print('No Rrs data found')

    sal_results = earthaccess.search_data(
        short_name='SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5',
        temporal=tspan
    )
    if (len(sal_results) > 0):
        sal_paths = earthaccess.download(sal_results, 'sal_data')
    else:
        sal_paths = []
        print('No salinity data found')

    temp_results = earthaccess.search_data(
        short_name='MUR-JPL-L4-GLOB-v4.1',
        temporal=tspan
    )
    if (len(temp_results) > 0):
        temp_paths = earthaccess.download(temp_results, 'temp_data')
    else:
        temp_paths = []
        print('No temperature data found')

    return rrs_paths, sal_paths, temp_paths

def estimate_inv_pigments(rrs_paths, sal_paths, temp_paths, bbox):
    '''
    Uses the rrs_inversion_pigments algorithm to calculate chlorophyll a (Chla), chlorophyll b (Chlb), chlorophyll c1
    +c2 (Chlc12), and photoprotective carotenoids (PPC) given an Rrs spectra, salinity, and temperature. Calculates the pigment 
    values for each lat/lon coordinate in the box's range. Pigment values are in units of mg/m^3.
    
    Uses PACE L3 mapped data, which does not come with Rrs uncertainty values. A uniform value of 5% uncertainty is used because Rrs
    uncertainties are not included in level 3 data files. See L2_utils to run inversion method with real uncertainty values. 
    Each L3 mapped data file contains data for the entire globe. 

    See rrs_inversion_pigments file for more information on the inversion estimation method.

    Parameters:
    -----------
    rrs_paths : list or str
        A single file path to a PACE Rrs file or an list of file paths to PACE Rrs files.
    sal_paths : list or str
        A single file path to a salinity file or an list of file paths to salinity files.
    temp_paths : list or str
        A single file path to a temperature file or an list of file paths to temperature files.
    bbox : tuple of floats or ints
        A tuple representing spatial bounds in the form (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).

    Returns:
    --------
    Xarray dataset 
        Dataset containing the Chla, Chlb, Chlc, and PPC concentration at each lat/lon coordinate
    '''
    
    box = _create_dataset(rrs_paths, sal_paths, temp_paths, bbox)

    progress = 1 # keeps track of how many pixels have been calculated
    pixels = box.lat.size * box.lon.size
    print('num pixels: ', pixels)

    chla = np.zeros((box.lat.size, box.lon.size))
    chlb = np.zeros((box.lat.size, box.lon.size))
    chlc = np.zeros((box.lat.size, box.lon.size))
    ppc = np.zeros((box.lat.size, box.lon.size))
    
    # for each coordinate estimate pigment concentrations
    for lat in range(box.lat.size):
        for lon in range(box.lon.size):
            # prints total number of pixels and how many have been estimated already
            sys.stdout.write('\rProgress: ' + str(progress) + '/' + str(pixels))
            sys.stdout.flush()
            progress += 1
            
            wl = box.wavelength.to_numpy()
            Rrs = box['rrs'][lat][lon].to_numpy()
            Rrs[Rrs == 0] = 0.000001 # insure no zero values
            Rrs_unc = Rrs * 0.05 # 5% uncertianty used for all values
            sal = box['sal'][lat][lon].data.item()
            temp = box['temp'][lat][lon].data.item() - 273 # convert from kelvin to celcius

            if not (math.isnan(Rrs[0]) or math.isnan(sal) or math.isnan(temp)):
                vals = rrs_inversion_pigments(Rrs, Rrs_unc, wl, temp, sal)
                chla[lat][lon] = vals[0][0]
                chlb[lat][lon] = vals[0][1]
                chlc[lat][lon] = vals[0][2]
                ppc[lat][lon] = vals[0][3]

    pigments = xr.Dataset(
        {
            'chla': (['lat', 'lon'], chla),
            'chlb': (['lat', 'lon'], chlb),
            'chlc': (['lat', 'lon'], chlc),
            'ppc': (['lat', 'lon'], ppc)
        },
        coords={
            'lat': box.lat.to_numpy(),
            'lon': box.lon.to_numpy()
        }

    )

    return pigments

def estimate_cov_pigments(tspan, bbox):
    '''
    Uses the covarying relationship between chlorophyll a and chlorophyll b, c1+c2, and PPC to estimate pigment concentrations. 
    Relies on PACE's chlorophyll a tool to estimate chlorophyll b, c1+c2, and PPC.
    Covariation pigment estimation method can be found in the following publication:

        Chase, A., E. Boss, I. Cetinic, and W. Slade. 2017. "Estimation of Phytoplankton
        Accessory Pigments from Hyperspectral Reflectance Spectra: Toward a Global Algorithm."
        Journal of Geophysical Research: Oceans, doi: 10.1002/2017JC012859.

    Parameters:
    -----------
    tspan : tuple of str
        A tuple containing two strings both with format 'YYYY-MM-DD'. The first date in the tuple must predate the second date in the tuple.
    bbox : tuple of floats or ints
        A tuple representing spatial bounds in the form (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).

    Returns:
    --------
    Xarray dataset 
        Dataset containing the Chla, Chlb, Chlc, and PPC concentration at each lat/lon coordinate.
    '''
    
    chla_results = earthaccess.search_data(
        short_name='PACE_OCI_L3M_CHL_NRT',
        temporal=tspan,
        granule_name="*.DAY.*.0p1deg.*"
    )

    if (len(chla_results) < 1):
        print('No chlorophyll a data files found.')
    else:
        chla_paths = earthaccess.download(chla_results, 'chla_data')

        n = bbox[3]
        s = bbox[1]
        e = bbox[2]
        w = bbox[0]

        if len(chla_paths) == 1:
            chla_data = xr.open_dataset(chla_paths)
            chla_data = chla_data["chlor_a"].sel({"lat": slice(n, s), "lon": slice(w, e)})
        else:
            # if given a list of files, create a date averaged dataset of Rrs values 
            chla_data = xr.open_mfdataset(
                chla_paths,
                combine="nested",
                concat_dim="date"
            )
            chla_data = chla_data["chlor_a"].sel({"lat": slice(n, s), "lon": slice(w, e)}).mean('date')
            chla_data = chla_data.compute()

        chlb = np.zeros((chla_data.lat.size, chla_data.lon.size))
        chlc = np.zeros((chla_data.lat.size, chla_data.lon.size))
        ppc = np.zeros((chla_data.lat.size, chla_data.lon.size))

        # for each coordinate estimate pigment concentrations
        for lat in range(len(chla_data.lat)):
            for lon in range(len(chla_data.lon)):
                # PACE's chlorophyll a estimation is used
                chla = chla_data[lat][lon]
                
                # The following calculations are taken from Chase et. al. 2017 and describe the relationship of 
                # each accessory pigment with chlorophyll a
                chlb[lat][lon] = (chla/5.44)**(1/0.86) 
                chlc[lat][lon] = (chla/6.27)**(1/0.81)
                ppc[lat][lon] = (chla/11.10)**(1/1.44)

        pigments = xr.Dataset(
            {
                'chla': (['lat', 'lon'], chla_data.values),
                'chlb': (['lat', 'lon'], chlb),
                'chlc': (['lat', 'lon'], chlc),
                'ppc': (['lat', 'lon'], ppc)
            },
            coords={
                'lat': chla_data.lat,
                'lon': chla_data.lon
            }

        )

        return pigments

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

def _create_dataset(rrs_paths, sal_paths, temp_paths, bbox):
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
        raise ValueError('rrs_paths must be a string or a list containing at least one filepath')

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
        raise TypeError('temp_paths must be a string or list')
    
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


    



        

