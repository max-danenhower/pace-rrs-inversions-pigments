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
chlorophyll c, and PPC concentrations, and plot a visualization of those pigment concentrations on a color map. 
'''

def load_L3_data(tspan, resolution):
    '''
    Downloads Remote Sensing Reflectance (Rrs) data from the PACE Satellite, as well as salinity and temperature data (from different 
    missions), and saves the data files to local folders names 'rrs_data', 'sal_data', and 'temp_data'

    Parameters:
    tspan (tuple of strings): a tuple containing two strings both with format 'YYYY-MM-DD'. The first date in the tuple
    must predate the second date in the tuple.

    resolution (string): the resolution of data being retrieved. Either '1deg', '0.1deg' or '4km'

    Returns:
    rrs_paths (list): A list containig the file path(s) to the downloaded Rrs PACE files
    sal_paths (list): A list containig the file path(s) to the downloaded salinity files
    temp_paths (list): A list containig the file path(s) to the downloaded temperature files
    '''

    rrs_results = earthaccess.search_data(
        short_name='PACE_OCI_L3M_RRS_NRT',
        temporal=tspan,
        granule_name='*.DAY.*.Rrs.' + resolution + '.*'
    )
    if (len(rrs_results) > 0):
        rrs_paths = earthaccess.download(rrs_results, 'rrs_L3_data')
    else:
        rrs_paths = []
        print('No Rrs data found')

    sal_paths, temp_paths = _load_sal_temp_data(tspan)

    return rrs_paths, sal_paths, temp_paths

def load_L2_data(tspan, n, s, e, w):
    bbox = (e, s, w, n)

    rrs_results = earthaccess.search_data(
        short_name='PACE_OCI_L2_AOP_NRT',
        bounding_box=bbox,
        temporal=tspan,
        count=1
    )
    if (len(rrs_results) > 0):
        rrs_paths = earthaccess.download(rrs_results, 'rrs_L2_data')
    else:
        rrs_paths = []
        print('No L2 AOP data found')

    sal_paths, temp_paths = _load_sal_temp_data(tspan)

    return rrs_paths[0], sal_paths[0], temp_paths[0]

def _load_sal_temp_data(tspan):
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

    return sal_paths, temp_paths

def create_L3_dataset(rrs_paths, sal_paths, temp_paths, n, s, w, e):
    '''
    Creates an xarray data array with latitude and longitude coordinates. Each coordinate contains a hyperspectral Rrs spectra with 
    corresponding wavelenghts, salinity, and temperature. If more than one file for Rrs, salinity, or temperature are given, uses the 
    date averaged values. 

    Parameters:
    rrs_paths (array or string): a single file path to a PACE Rrs file or an array of file paths to PACE Rrs files
    sal_paths (array or string): a single file path to a salinity file or an array of file paths to salinity files
    temp_paths (array or string): a single file path to a temperature file or an array of file paths to temperature files
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

def calculate_pigments_L2_inv(L2_path, sal_path, temp_path):
    wl_coord = np.array([339., 341., 344., 346., 348., 351., 353., 356., 358., 361., 363., 366.,
       368., 371., 373., 375., 378., 380., 383., 385., 388., 390., 393., 395.,
       398., 400., 403., 405., 408., 410., 413., 415., 418., 420., 422., 425.,
       427., 430., 432., 435., 437., 440., 442., 445., 447., 450., 452., 455.,
       457., 460., 462., 465., 467., 470., 472., 475., 477., 480., 482., 485.,
       487., 490., 492., 495., 497., 500., 502., 505., 507., 510., 512., 515.,
       517., 520., 522., 525., 527., 530., 532., 535., 537., 540., 542., 545.,
       547., 550., 553., 555., 558., 560., 563., 565., 568., 570., 573., 575.,
       578., 580., 583., 586., 588., 591., 593., 596., 598., 601., 603., 605.,
       608., 610., 613., 615., 618., 620., 623., 625., 627., 630., 632., 635.,
       637., 640., 641., 642., 643., 645., 646., 647., 648., 650., 651., 652.,
       653., 655., 656., 657., 658., 660., 661., 662., 663., 665., 666., 667.,
       668., 670., 671., 672., 673., 675., 676., 677., 678., 679., 681., 682.,
       683., 684., 686., 687., 688., 689., 691., 692., 693., 694., 696., 697.,
       698., 699., 701., 702., 703., 704., 706., 707., 708., 709., 711., 712.,
       713., 714., 717., 719.])
    
    dataset = xr.open_dataset(L2_path, group='geophysical_data')
    rrs = dataset['Rrs']
    rrs_unc = dataset['Rrs_unc']

    dataset = xr.open_dataset(L2_path, group="navigation_data")
    dataset = dataset.set_coords(("longitude", "latitude"))
    dataset_r = xr.merge((rrs, dataset.coords))
    dataset_ru = xr.merge((rrs_unc, dataset.coords))

    n, s, e, w = _get_user_boundary(dataset_r.latitude.values.max(), dataset_r.latitude.values.min(), 
                                    dataset_r.longitude.values.max(), dataset_r.longitude.values.min())

    rrs_box = dataset_r["Rrs"].where(
        (
            (dataset["latitude"] > s)
            & (dataset["latitude"] < n)
            & (dataset["longitude"] < e)
            & (dataset["longitude"] > w)
        ),
        drop=True,
    )

    rrs_unc_box = dataset_ru["Rrs_unc"].where(
        (
            (dataset["latitude"] > s)
            & (dataset["latitude"] < n)
            & (dataset["longitude"] < e)
            & (dataset["longitude"] > w)
        ),
        drop=True,
    )

    sal = xr.open_dataset(sal_path)
    sal = sal["smap_sss"].sel({"latitude": slice(n, s), "longitude": slice(w, e)})

    temp = xr.open_dataset(temp_path)
    temp = temp['analysed_sst'].squeeze() # get rid of extra time dimension
    temp = temp.sel({"lat": slice(s, n), "lon": slice(w, e)})
    temp = temp - 273

    sal = sal.interp(longitude=rrs_box.longitude, latitude=rrs_box.latitude, method='nearest')
    temp = temp.interp(lon=rrs_box.longitude, lat=rrs_box.latitude, method='nearest')

    rrs_box['chla'] = (('number_of_lines', 'pixels_per_line'), np.zeros((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size)))
    rrs_box['chlb'] = (('number_of_lines', 'pixels_per_line'), np.zeros((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size)))
    rrs_box['chlc'] = (('number_of_lines', 'pixels_per_line'), np.zeros((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size)))
    rrs_box['ppc'] = (('number_of_lines', 'pixels_per_line'), np.zeros((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size)))

    progress = 1 # keeps track of how many pixels have been calculated
    pixels = rrs_box.number_of_lines.size * rrs_box.pixels_per_line.size

    for i in range(len(rrs_box.number_of_lines)):
        for j in range(len(rrs_box.pixels_per_line)):
            sys.stdout.write('\rProgress: ' + str(progress) + '/' + str(pixels))
            sys.stdout.flush()
            progress += 1

            r = rrs_box[i][j].to_numpy()
            ru = rrs_unc_box[i][j].to_numpy()
            if (not math.isnan(r[0])):
                sal_val = sal[i][j].values.item()
                temp_val = temp[i][j].values.item()
                if not (math.isnan(r[0]) or math.isnan(sal_val) or math.isnan(temp_val)):
                    pigs = rrs_inversion_pigments(r, ru, wl_coord, temp_val, sal_val)[0]
                    rrs_box['chla'][i][j] = pigs[0]
                    rrs_box['chlb'][i][j] = pigs[1]
                    rrs_box['chlc'][i][j] = pigs[2]
                    rrs_box['ppc'][i][j] = pigs[3]
    
    return rrs_box

def _get_user_boundary(n_box, s_box, e_box, w_box):
    print('The downloaded granule has boundaries:')
    print('north: ', n_box)
    print('south: ', s_box)
    print('east: ', e_box)
    print('west: ', w_box)
    print('Select a boundary box within these coordinates to calculate pigments for')
    n = float(input('north (between ' + str(n_box) + ' and ' + str(s_box) + '): '))
    s = float(input('south (between ' + str(n_box) + ' and ' + str(s_box) + '): '))
    e = float(input('east (between ' + str(w_box) + ' and ' + str(e_box) + '): '))
    w = float(input('west (between ' + str(w_box) + ' and ' + str(e_box) + '): '))

    return n, s, e, w

            
def calculate_pigments_L3_inv(box):
    '''
    Uses the rrs_inversion_pigments algorithm to calculate chlorophyll a (Chla), chlorophyll b (Chlb), chlorophyll c1
    +c2 (Chlc12), and photoprotective carotenoids (PPC) given an Rrs spectra, salinity, and temperature. Calculates the pigment 
    values for each lat/lon coordinate in the box's range. Pigment values are in units of mg/m^3.

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
            Rrs = box['rrs'][lat][lon].to_numpy()
            Rrs[Rrs == 0] = 0.000001 # insure no zero values
            Rrs_unc = Rrs * 0.05 # 5% uncertianty for all Rrs values
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

    pigments = xr.Dataset(
        {
            'chla': (['lat', 'lon'], chla),
            'chlb': (['lat', 'lon'], chlb),
            'chlc': (['lat', 'lon'], chlc),
            'ppc': (['lat', 'lon'], ppc)
        },
        coords={
            'lat': lat_coord,
            'lon': lon_coord
        }

    )

    return pigments

def calculate_pigements_cov(tspan, n, s, e, w):
    chla_results = earthaccess.search_data(
        short_name='PACE_OCI_L3M_CHL_NRT',
        temporal=tspan,
        granule_name="*.DAY.*.0p1deg.*"
    )

    chla_paths = earthaccess.download(chla_results, 'chla_data')

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

    for lat in range(len(chla_data.lat)):
        for lon in range(len(chla_data.lon)):
            chla = chla_data[lat][lon]
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

def plot_L3_pigments(data, lower_bound, upper_bound, label):
    '''
    Plots the pigment data with lat/lon coordinates using a color map

    Paramaters:
    data (xr data array): An array with pigment values at each lat/lon coordinate
    lower_bound (float): The lowest value represented on the color scale
    upper_bound (float): The upper value represented on the color scale
    label (string): A label for the graph
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

def plot_L2_pigments(data, lower_bound, upper_bound):
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




    



        

