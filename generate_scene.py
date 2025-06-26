import L2_utils
import xarray as xr
import time
import numpy as np

if __name__ == "__main__":
    rrs_path = 'rrs_data/PACE_OCI.20250204T201251.L2.OC_AOP.V3_0.nc'
    sal_path = 'sal_data/SMAP_L3_SSS_20250204_8DAYS_V5.0.nc'
    temp_path = 'temp_data/20250204090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'

    start = time.time()

    result = L2_utils.estimate_inv_pigments(rrs_path, sal_path, temp_path)

    print()
    print('time:', time.time()-start)

    result.to_netcdf('/Users/mdanenhower/Desktop/pace/git/scene_generator/rrs_inversion_output')

    pigments = xr.open_dataset('rrs_inversion_output')

    # access pigments
    chla = pigments['chla']
    chlb = pigments['chlb']
    chlc = pigments['chlc']
    ppc = pigments['ppc']

    log_chla = np.log10(chla)

    L2_utils.plot_pigments(log_chla,-2,1, 'log10(Tchl a)')






    