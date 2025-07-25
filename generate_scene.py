from gpig import L2_utils, L3_utils
from datetime import date, timedelta
import ray
import time
import numpy as np
import xarray as xr
import sys

def main_L3(tspan_begin, tspan_end, n_boundary, s_boundary, e_boundary, w_boundary):

    tspan = (tspan_begin, tspan_end)

    bbox = (w_boundary, s_boundary, e_boundary, n_boundary)

    rrs_path,sss_path,sst_path = L3_utils.load_data(tspan)

    r,ru,wl,s,t = L3_utils.interpolate_data(rrs_path, sss_path, sst_path, bbox)

    Rrs_flat = r.stack(pix=('lat','lon'))         # shape: (n_pix, 172)
    Rrs_unc_flat = ru.stack(pix=('lat','lon'))
    temp_flat = t.stack(pix=('lat','lon'))
    sal_flat = s.stack(pix=('lat','lon')) 
    wl = wl.values

    n_pix = Rrs_flat.sizes['pix']
    print('number of pixels:',n_pix,'\n')

    Rrs_np = Rrs_flat.values.T       # shape: (n_pix, 172)
    Rrs_unc_np = Rrs_unc_flat.values.T
    temp_np = temp_flat.values.T
    sal_np = sal_flat.values.T

    batch_size = 10_000

    batches = [
        (
            Rrs_np[i:i+batch_size],
            Rrs_unc_np[i:i+batch_size],
            wl,
            temp_np[i:i+batch_size],
            sal_np[i:i+batch_size]
        )
        for i in range(0, len(Rrs_np), batch_size)
    ]

    start = time.time()

    ray.init(include_dashboard=True, log_to_driver=False)

    print('ray availble resources', ray.available_resources(),'\n')

    # Launch Ray tasks
    futures = [L3_utils.run_batch.remote(*b) for b in batches]
    results = ray.get(futures)  # list of lists, flatten if needed
    flat_results = [res for batch in results for res in batch]

    ray.shutdown()

    chla = np.full((r.lat.size, r.lon.size), np.nan)
    chlb = np.full((r.lat.size, r.lon.size), np.nan)
    chlc = np.full((r.lat.size, r.lon.size), np.nan)
    ppc = np.full((r.lat.size, r.lon.size), np.nan)

    n_lat = r.lat.size
    n_lon = r.lon.size

    # Convert to 3D array: (n_lines, n_pixels, 4)
    results_array = np.array(flat_results).reshape(n_lat, n_lon, -1)

    chla[:,:] = results_array[:,:,0]
    chlb[:,:] = results_array[:,:,1]
    chlc[:,:] = results_array[:,:,2]
    ppc[:,:] = results_array[:,:,3]

    pigments = xr.Dataset(
        {
            'chla': (['lat', 'lon'], chla),
            'chlb': (['lat', 'lon'], chlb),
            'chlc': (['lat', 'lon'], chlc),
            'ppc': (['lat', 'lon'], ppc)
        },
        coords={
            'lat': r.lat.to_numpy(),
            'lon': r.lon.to_numpy()
        }
    )

    print('total task runtime', time.time()-start,'\n')

    output_str = 'gpig-' + tspan_begin

    try:
        pigments.to_netcdf(output_str)
        print('successfully saved results to ', output_str, '\n')
    except:
        print('error loading results\n')


def main_L2(tspan_begin, tspan_end, n_boundary, s_boundary, e_boundary, w_boundary):

    tspan = (tspan_begin, tspan_end)

    bbox = (w_boundary, s_boundary, e_boundary, n_boundary)

    rrs_path,sss_path,sst_path = L2_utils.load_data(tspan,bbox)

    r,ru,wl,s,t = L2_utils.interpolate_coords(rrs_path,sss_path,sst_path)

    Rrs_flat = r.stack(pix=('number_of_lines', 'pixels_per_line'))         # shape: (n_pix, 172)
    Rrs_unc_flat = ru.stack(pix=('number_of_lines', 'pixels_per_line')) # same shape
    temp_flat = t.stack(pix=('number_of_lines', 'pixels_per_line'))       # shape: (n_pix,)
    sal_flat = s.stack(pix=('number_of_lines', 'pixels_per_line'))         # shape: (n_pix,)

    n_pix = Rrs_flat.sizes['pix']
    print('number of pixels:',n_pix,'\n')

    Rrs_np = Rrs_flat.values.T       # shape: (n_pix, 172)
    Rrs_unc_np = Rrs_unc_flat.values.T
    temp_np = temp_flat.values.T
    sal_np = sal_flat.values.T

    batch_size = 10_000

    batches = [
        (
            Rrs_np[i:i+batch_size],
            Rrs_unc_np[i:i+batch_size],
            wl,
            temp_np[i:i+batch_size],
            sal_np[i:i+batch_size]
        )
        for i in range(0, len(Rrs_np), batch_size)
    ]

    start = time.time()

    ray.init(include_dashboard=True, log_to_driver=False)

    print('ray availble resources', ray.available_resources(),'\n')

    # Launch Ray tasks
    futures = [L2_utils.run_batch.remote(*b) for b in batches]
    results = ray.get(futures)  # list of lists, flatten if needed
    flat_results = [res for batch in results for res in batch]

    ray.shutdown()

    # Get spatial dimensions from original data
    n_lines = r.sizes['number_of_lines']
    n_pixels = r.sizes['pixels_per_line']

    # Convert to 3D array: (n_lines, n_pixels, 4)
    results_array = np.array(flat_results).reshape(n_lines, n_pixels, -1)

    r['chla'].values[:, :] = results_array[:, :, 0]
    r['chlb'].values[:, :] = results_array[:, :, 1]
    r['chlc'].values[:, :] = results_array[:, :, 2]
    r['ppc'].values[:, :]  = results_array[:, :, 3]

    print('total task runtime', time.time()-start,'\n')

    output_str = 'gpig-' + str(date.today() - timedelta(days=7))

    try:
        r.to_netcdf(output_str)
        print('successfully saved results to ', output_str, '\n')
    except:
        print('error loading results\n')


if __name__ == "__main__":

    arg_names = ['temporal range (begin)', 'temporal range (end)', 'n boundary', 's boundary', 'e boundary', 'w boundary']

    if len(sys.argv) == 7:
        print(f"Script name: {sys.argv[0]}")
        print("Arguments received:")
        for i, arg in enumerate(sys.argv[1:]):
            print(f"  {arg_names[i]}: {arg}")

        main_L2(*sys.argv[1:])
    else:
        print("Must give 6 arguments.")

    


# im here


    