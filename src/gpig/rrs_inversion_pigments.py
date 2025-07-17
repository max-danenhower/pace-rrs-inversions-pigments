'''
Originally developed in MatLab by Ali Chase and can be found here: https://github.com/alisonpchase/Rrs_inversion_pigments
Translated to Python by Max Danenhower, Charles Stern, and Ali Chase.
'''

from typing import Tuple, Union
from importlib.resources import files

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from numba import njit


G1 = 0.0949  # g1 and g2 are values from Gordon et al., 1988
G2 = 0.0794
LNOT = 400  # reference lambda wavelength (nm)

def rrs_inversion_pigments(Rrs, Rrs_unc, wl, temp, sal):
    '''
    Inversion algorithm to estimate phytoplankton pigments from Rrs spectra

    Ali Chase, University of Maine, 2017 - 2019

    See the following publication for details on the method:

        Chase, A., E. Boss, I. Cetinic, and W. Slade. 2017. "Estimation of Phytoplankton
        Accessory Pigments from Hyperspectral Reflectance Spectra: Toward a
        Global Algorithm."
        Journal of Geophysical Research: Oceans, doi: 10.1002/2017JC012859.

    NOTE: This code was developed to estimate phytoplankton pigments from
    hyperspectral remote-sensing reflectance (Rrs) data measured in situ at
    the ocean surface. The Rrs data used in this algorithm development were
    quality controlled, corrected for Raman scattering, normalized to
    eliminate the angular effect of the sun position in the sky relative to
    nadir. Please see above reference for details of these steps.

    Parameters:
    -----------
    Rrs : Numpy array
        Remote-sensing reflectance measurements, defined as Lw/Ed
    Rrs_unc : Numpy array
        Uncertainy values in Rrs measurements (e.g. could be the standard deviation
        in Rrs for a given five-minute sample collection), must be on the same 
        wavlength grid as the Rrs data
    wl : Numoy array
        wavelengths associated with Rrs (and Rrs_unc) data
    tem : int or float
        water temperature at the time of radiometery data collection
    sal : int or float
        water salinity at the time of radiometery data collection
                
    Returns:
    --------
    pigmedian : Numpy array
        Estimated pigment concentrations
    pig_unc : Numpy Array
        Uncertainties in estimated pigments,
        calculated using a Monte Carlo method that in turn uses the 
        reported uncertainties in the A and B coefficients reported
        in Chase et al. (2017). 
    vars_units : str
        The names and units of the estimated pigments:
        chlorophyll a (Chla), chlorophyll b (Chlb), chlorophyll c1
        +c2 (Chlc12), and photoprotective carotenoids (PPC) defined
        as abcarotene+zeaxanthin+alloxanthin+diadinoxanthin. All
        pigments and uncertainties are in mg m^-3.
    amps : Numpy array
        Amplitudes of Gaussian absorption functions
        representing Chla, Chlb, Chlc12, and PPC. Can be used to derive
        updated relationships between Gaussians and HPLC pigments.
    '''

    # cut off data < 400 or > 600 
    wlfull = wl
    Iwl = np.where((wlfull > 400) & (wlfull < 600) & (~np.isnan(wlfull)))[0]
    wl = wlfull[Iwl]

    Rrs = Rrs[Iwl]
    Rrs_unc = Rrs_unc[Iwl]

    # check for negatives
    if np.any(Rrs<0):
        return np.array([np.nan,np.nan,np.nan,np.nan]), 0,0,0

    # Get the absorption and backscattering by water for the temperature and
    # salinity measured coincidently with Rrs
    a_sw, bb_sw = get_water_iops(wl, temp, sal)

    # Calculate rrs from Rrs (method from Lee et al., 2002)
    rrs = Rrs / (0.52 + 1.7*Rrs)
    rrs_unc = Rrs_unc / (0.52 + 1.7*Rrs_unc)

    # Calculate the positive solution for U using rrs, where U = bb/(bb+a).
    # This model and g coefficients are from Gordon et al., 1988
    Upos = (-G1 + np.sqrt(G1**2 + 4*G2*rrs))/(2*G2)
    Uunc = (-G1 + np.sqrt(G1**2 + 4*G2*rrs_unc))/(2*G2)

    # Define the center peak locations (nm) and widths (nm) of the Gaussian functions
    # sig = sigma, where FWHM = sigma*2.355 and FWHM = full width at half max
    peaks = np.array([384,413,435,461,464,490,532,583,623,644,655,676], dtype=float)
    sig = np.array([23,9,14,11,19,19,20,20,15,12,12,9], dtype=float)

    #   Define the [lower bound, first guess, upper bound] for each parameter. These will be allowed to vary.
    s_nap = [.005, .011, .016]
    m_nap = [.0, .005, .05]
    s_cdom = [.005, .0185, .02]
    m_cdom = [.01, .1, .8]
    bbpbp_ratio = [.005, .01, .015]
    m_gaus = [.0, .01, 0.5]
    cpgam = [.0, 1.0, 1.3]
    m_cp = [.01, .1, 1.0]

    C_fluor = [.0, .001, .01]

    # first guess array
    Amp0 = [s_nap[1], m_nap[1], s_cdom[1], m_cdom[1], bbpbp_ratio[1], cpgam[1], m_cp[1],C_fluor[1]] + np.tile(m_gaus[1],len(peaks)).tolist() + peaks.tolist() + sig.tolist()
    
    # lower bound array
    LB = [s_nap[0], m_nap[0], s_cdom[0], m_cdom[0], bbpbp_ratio[0], cpgam[0], m_cp[0],C_fluor[0]] + np.tile(m_gaus[0],len(peaks)).tolist() + (peaks-1).tolist() + (sig-1).tolist()
    
    # upper bound array
    UB = [s_nap[2], m_nap[2], s_cdom[2], m_cdom[2], bbpbp_ratio[2], cpgam[2], m_cp[2],C_fluor[2]] + np.tile(m_gaus[2],len(peaks)).tolist() + (peaks+1).tolist() + (sig+1).tolist()

    # Run the inversion using a non-linear least squares inversion function
    result = least_squares(lsqnonlin_Amp_gen, np.array(Amp0), bounds=(LB, UB), ftol=1e-6, xtol=1e-6, gtol=1e-6, 
                           args=(Upos, Uunc, wl, bb_sw, a_sw, LNOT))

    Amp2 = result.x

    # Estimate pigment concentrations and their uncertainties with coefficients reported in
    # Chase et al., 2017 (JGR-Oceans) and a Monte Carlo method. 
    # NOTE: these coefficients have been recomputed using the same method in Chase et al., 2017 (JGR-Oceans) using new data
    # matrix:   A   A_unc   B   B_unc

    coeffs = np.array([
            [0.022, 0.008, 0.563, 0.068],
            [0.014, 0.009, 0.14, 0.059],
            [0.026, 0.013, 0.286, 0.074],
            [0.026, 0.024, 0.194, 0.105]])

    np.seterr(divide='ignore')

    pigmedian = np.zeros(4)
    pigunc = np.zeros(4)

    for ii in range(4):
        mc = np.random.randn(10000, 1) * coeffs[ii, [1, 3]]
        As = coeffs[ii, 0] + mc[:, 0]
        As[As < 0] = 0  # prevent imaginary pigment values
        Bs = coeffs[ii, 2] + mc[:, 1]
        pigest = (Amp2[10 + ii] / As) ** (1.0 / Bs)
        pigmedian[ii] = np.median(pigest)
        prc = np.percentile(pigest, [16, 84])
        pigunc[ii] = (prc[1] - prc[0]) / 2

    vars_units = 'Chla, Chlc1+c2, Chlb, PPC; all in mg m^-3'
    amps = Amp2[10:14]

    return pigmedian, pigunc, vars_units, amps

@njit(fastmath=True)
def lsqnonlin_Amp_gen(Amp0,Upos,Uunc,wvns,bb_sw_r,a_sw_r,lnot):
    '''
    The following function uses a non-linear least squares solver to minimize
    the difference between the measured Rrs and the modeled Rrs
    '''

    # Unpack parameters
    slope_nap, mag_nap = Amp0[0], Amp0[1]
    slope_cdom, mag_cdom = Amp0[2], Amp0[3]
    bbp_ratio = Amp0[4]
    slope_cp = Amp0[5]
    mag_cp = Amp0[6]
    fluor_amp = Amp0[7]

    amps = Amp0[8:20]                # Amplitudes of Gaussians
    peak_locs = Amp0[20:32]          # Centers
    sig = Amp0[32:44]                # Widths

    # Ensure broadcasting works
    wvns_col = wvns[:, np.newaxis]   # Shape (N, 1)
    peak_locs = peak_locs[np.newaxis, :]  # Shape (1, 12)
    sig = sig[np.newaxis, :]
    amps = amps[np.newaxis, :]

    # CDOM and NAP absorption
    CDOM = mag_cdom * np.exp(-slope_cdom * (wvns - lnot))
    NAP = mag_nap * np.exp(-slope_nap * (wvns - lnot))

    # Gaussian absorption (a_phi)
    gaus = np.exp(-0.5 * ((wvns_col - peak_locs) / sig)**2) * amps
    APHI = np.sum(gaus, axis=1)

    # Total particulate absorption
    AP = NAP + APHI

    # cp (total particle concentration)
    CP = mag_cp * (wvns / lnot) ** (-slope_cp)

    # BBP (backscatter from particles)
    BBP = bbp_ratio * (CP - AP)

    # Fluorescence Gaussian
    fluor = np.exp(-0.5 * ((wvns - 685) / 10.6) ** 2)
    F = fluor_amp * fluor

    # Modeled U = bb / (a + bb)
    denom = APHI + NAP + CDOM + a_sw_r + BBP + bb_sw_r
    numer = BBP + bb_sw_r + F
    Unew = numer / denom

    # Weighted residual
    spec_min = (Upos - Unew) / Uunc

    return spec_min

def RInw(
    lambda_: Union[int, float, np.ndarray],
    Tc: Union[int, float],
    S: Union[int, float],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Refractive index of air is from Ciddor (1996,Applied Optics)
    Refractive index of seawater is from Quan and Fry (1994, Applied Optics)
    """
    n_air = (
        1.0 + (5792105.0 / (238.0185 - 1 / (lambda_ / 1e3) ** 2)
        + 167917.0 / (57.362 - 1 / (lambda_ / 1e3) ** 2)) / 1e8
    )
    n0 = 1.31405
    n1 = 1.779e-4
    n2 = -1.05e-6
    n3 = 1.6e-8
    n4 = -2.02e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382
    n9 = 1.1455e6
    nsw = (
        n0 + (n1 + n2 * Tc + n3 * Tc ** 2) * S + n4 * Tc ** 2 
        + (n5 + n6 * S + n7 * Tc) / lambda_ + n8 / lambda_ ** 2 + n9
        / lambda_ ** 3
    )
    nsw = nsw * n_air
    dnswds = (n1 + n2 * Tc + n3 * Tc ** 2 + n6 / lambda_) * n_air

    return nsw, dnswds


def BetaT(Tc: Union[int, float], S: Union[int, float]) -> float:
    """Pure water secant bulk Millero (1980, Deep-sea Research).
    Isothermal compressibility from Kell sound measurement in pure water.
    Calculate seawater isothermal compressibility from the secant bulk.
    """
    kw = (
        19652.21 + 148.4206 * Tc - 2.327105 * Tc ** 2 + 1.360477e-2 
        * Tc ** 3 - 5.155288e-5 * Tc ** 4
    )
    Btw_cal = 1 / kw
    a0 = 54.6746 - 0.603459 * Tc + 1.09987e-2 * Tc ** 2-6.167e-5 * Tc ** 3
    b0 = 7.944e-2 + 1.6483e-2 * Tc - 5.3009e-4 * Tc ** 2

    Ks = kw + a0 * S + b0 * S ** 1.5
    IsoComp = 1 / Ks * 1e-5

    return IsoComp


def rho_sw(Tc: Union[int, float], S: Union[int, float]) -> float:
    """Density of water and seawater, unit is Kg/m^3, from UNESCO,38,1981.
    TODO: Compare to GSW Oceanographic Toolbox code (or other updated) density equations.
    """
    a0 = 8.24493e-1
    a1 = -4.0899e-3
    a2 = 7.6438e-5
    a3 = -8.2467e-7
    a4 = 5.3875e-9
    a5 = -5.72466e-3
    a6 = 1.0227e-4
    a7 = -1.6546e-6
    a8 = 4.8314e-4
    b0 = 999.842594
    b1 = 6.793952e-2
    b2 = -9.09529e-3
    b3 = 1.001685e-4
    b4 = -1.120083e-6
    b5 = 6.536332e-9

    # density for pure water
    density_w = b0 + b1 * Tc + b2 * Tc ** 2 + b3 * Tc ** 3 + b4 * Tc ** 4 + b5 * Tc ** 5
    # density for pure seawater
    density_sw = (
        density_w + ((a0 + a1 * Tc + a2 * Tc ** 2 + a3 * Tc ** 3 + a4 * Tc ** 4) * S
        + (a5 + a6 * Tc + a7 * Tc ** 2) * S ** 1.5 + a8 * S ** 2)
    )
    return density_sw


def dlnasw_ds(Tc: Union[int, float], S: Union[int, float]) -> float:
    """Water activity data of seawater is from Millero and Leung (1976,American
    Journal of Science,276,1035-1077). Table 19 was reproduced using
    Eqs.(14,22,23,88,107) then were fitted to polynominal equation.
    dlnawds is partial derivative of natural logarithm of water activity
    w.r.t.salinity.

    lnaw =  (-1.64555e-6-1.34779e-7*Tc+1.85392e-9*Tc.^2-1.40702e-11*Tc.^3)+......
            (-5.58651e-4+2.40452e-7*Tc-3.12165e-9*Tc.^2+2.40808e-11*Tc.^3).*S+......
            (1.79613e-5-9.9422e-8*Tc+2.08919e-9*Tc.^2-1.39872e-11*Tc.^3).*S.^1.5+......
            (-2.31065e-6-1.37674e-9*Tc-1.93316e-11*Tc.^2).*S.^2;
    
    density derivative of refractive index from PMH model
    """

    dlnawds = (
        (-5.58651e-4 + 2.40452e-7 * Tc - 3.12165e-9 * Tc ** 2 + 2.40808e-11 * Tc ** 3)
        + 1.5 * (1.79613e-5 - 9.9422e-8 * Tc + 2.08919e-9 * Tc ** 2 - 1.39872e-11 * Tc ** 3) * S ** 0.5
        + 2 * (-2.31065e-6 - 1.37674e-9 * Tc - 1.93316e-11 * Tc ** 2) * S
    )
    return dlnawds


def PMH(n_wat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    n_wat2 = n_wat ** 2
    n_density_derivative = (
        (n_wat2 - 1) * (1 + 2 / 3 * (n_wat2 + 2)
        * (n_wat / 3 - 1 / 3 / n_wat) ** 2)
    )
    return n_density_derivative


def betasw124_ZHH2009(lambda_, S, Tc, delta=0.039):
    """Scattering by pure seawater: Effect of salinity
    Xiaodong Zhang, Lianbo Hu, and Ming-Xia He, Optics Express, 2009, accepted
    lambda (nm): wavelength
    Tc: temperauter in degree Celsius, must be a scalar
    S: salinity, must be scalar
    delta: depolarization ratio, if not provided, default = 0.039 will be used (from Farinato and Roswell (1976))
    betasw: volume scattering at angles defined by theta. Its size is [x y],
    where x is the number of angles (x = length(theta)) and y is the number
    of wavelengths in lambda (y = length(lambda))
    beta90sw: volume scattering at 90 degree. Its size is [1 y]
    bw: total scattering coefficient. Its size is [1 y]
    for backscattering coefficients, divide total scattering by 2
    Xiaodong Zhang, March 10, 2009

    MODIFIED on 17/05/2011 to be able to process bbp profiles with coincident T and sal profiles
    MODIFIED on 05 Apr 2013 to use 124 degs instead of 117 degs
    """

    # TODO: Support S and Tc as column vectors
    for param in [S, Tc]:
        if type(param) not in (int, float) or isinstance(param, np.ndarray):
            raise NotImplementedError("S and Tc must be scalar.")

    # values of the constants
    Na = 6.0221417930e23  # Avogadro's constant
    Kbz = 1.3806503e-23  # Boltzmann constant
    Tk = Tc + 273.15  # Absolute temperature
    M0 = 18e-3  # Molecular weight of water in kg/mol

    theta = np.linspace(0.0, 180.0, 18_001)

    rad = theta * np.pi/180  # angle in radians as a 1-d array

    # nsw: absolute refractive index of seawater
    # dnds: partial derivative of seawater refractive index w.r.t. salinity
    nsw, dnds = RInw(lambda_, Tc, S)

    # isothermal compressibility is from Lepple & Millero (1971,Deep Sea-Research), pages 10-11
    # The error ~ +/-0.004e-6 bar^-1
    IsoComp = BetaT(Tc, S)

    # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
    density_sw = rho_sw(Tc, S)

    # water activity data of seawater is from Millero and Leung (1976,American
    # Journal of Science,276,1035-1077). Table 19 was reproduced using
    # Eq.(14,22,23,88,107) then were fitted to polynominal equation.
    # dlnawds is partial derivative of natural logarithm of water activity
    # w.r.t.salinity
    dlnawds = dlnasw_ds(Tc, S)

    # density derivative of refractive index from PMH model
    DFRI = PMH(nsw)  # PMH model

    # volume scattering at 90 degree due to the density fluctuation
    beta_df = (
        np.pi * np.pi / 2 * ((lambda_ * 1e-9) ** (-4)) * Kbz * Tk * IsoComp * DFRI ** 2
        * (6 + 6 * delta) / (6 - 7 * delta)
    )

    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = S * M0 * dnds ** 2 / density_sw / (-dlnawds) / Na
    beta_cf = (
        2 * np.pi * np.pi * ((lambda_ * 1e-9) ** (-4)) * nsw ** 2
        * (flu_con) * (6 + 6 * delta)/(6 - 7 * delta)
    )

    # total volume scattering at 90 degree
    beta90sw = beta_df + beta_cf
    bsw = 8 * np.pi/3 * beta90sw * (2 + delta) / (1 + delta)

    for i, value in enumerate(rad):  # TODO: Is there a better way to do this?
        if np.rad2deg(value) >= 124:
            rad124 = i
            break
    
    betasw124 = (
        beta90sw * (1 + ((np.cos(rad[rad124])) ** 2) * (1 - delta) / (1 + delta))
    )

    return betasw124, bsw, beta90sw, theta


def tempsal_corr(lambda_):
    """ """
    p = files("gpig").joinpath("resources/Sullivan_pure_water_temp_sal.csv")
    pwts = pd.read_csv(p)

    if lambda_.min() < 400 or lambda_.min() > 750:
        raise NotImplementedError(
            "Wavelengths < 400 or > 750 not currently supported because these "
            "values are outside the range for temperature salinitity correction "
            "provided in Sullivan et al 2006."
        )

    # next two lines changed from 'spline' on 14 Dec 2015
    psiT = np.interp(lambda_, pwts["lambda"], pwts["PsiT"])
    psiS = np.interp(lambda_, pwts["lambda"], pwts["PsiS"])

    return psiT, psiS


def get_water_iops(lambda_, T=22, S=35):
    """Function to obtain seawater absorption and backscattering spectra.
    Pure water absorption from Mason et al 2016 for 250-550, Pope and Frye for
    550-730 nm, and Smith and Baker for 730-800 nm salt water backscattering from Zhang et al 2009
    corrected for in situ temperature and salinity conditions Sullivan et al. 2006

    Parameters
    ----------
    wavel: (array)
        Wavelenghts (nm) for iops to be returned at.
    T: (float or int)
        Temperature (degC) of the water.
    S: (float or int)
        Salinity (psu) of the water.

    Returns
    -------
    a_sw: (array)
        Absorption of seawater at each wavelength in wavel for the specified temperature and salinity values.
    bb_sw: (array)
        Backscattering of seawater at each wavelenght in wavel for the specified temperature and salinity values.
    """
    lambda_water1 = [250, 260, 270, 280, 290] + list(range(300, 551, 2))
    lambda_water2 = list(np.linspace(552.5, 800.0, 100))
    lambda_water = lambda_water1 + lambda_water2

    aw1 = [
        58.7100, 51.5000, 43.5700, 22.3000, 9.3900, 4.6700, 4.3300, 3.5600, 3.1300, 2.7500, 2.3600, 2.0500, 
        1.8500, 1.7600, 1.6300, 1.4700, 1.3300, 1.2400, 1.1800, 1.1200, 1.0700, 1.0100, 0.9900, 0.9500, 
        0.9100, 0.8500, 0.8200, 0.8100, 0.8200, 0.8400, 0.8900, 0.9400, 0.9700, 0.9800, 0.9900, 1.0600, 
        1.1500, 1.2000, 1.2100, 1.2200, 1.2400, 1.2700, 1.2900, 1.3300, 1.3700, 1.4300, 1.4700, 1.5100, 
        1.5500, 1.6200, 1.7000, 1.7500, 1.8500, 1.9600, 2.0800, 2.2200, 2.3700, 2.4800, 2.5700, 2.5900, 
        2.6600, 2.7100, 2.8000, 2.8800, 3.0000, 3.1200, 3.2200, 3.3100, 3.4400, 3.5800, 3.7600, 3.9500, 
        4.1700, 4.4200, 4.8000, 5.2200, 5.7400, 6.2600, 6.9100, 7.5100, 8.0800, 8.4200, 8.6300, 8.7700, 
        8.9300, 9.0900, 9.3300, 9.5500, 9.7900, 9.9900, 10.3000, 10.6500, 11.0000, 11.3800, 11.7700, 12.1400, 
        12.5400, 12.9400, 13.3600, 13.9100, 14.6000, 15.4500, 16.4800, 17.7400, 19.2600, 20.7300, 22.4200, 24.2400, 
        26.6800, 29.7100, 33.0000, 35.6900, 37.3800, 38.2100, 38.7800, 39.1700, 39.6200, 40.1700, 40.8800, 41.6200, 
        42.4200, 43.3000, 44.3600, 45.4100, 46.4500, 47.5400, 48.8200, 50.4000, 52.2400, 54.2500, 56.2900,
    ]
    aw2 = [
        0.0593, 0.0596, 0.0606, 0.0619, 0.064, 0.0642, 0.0672, 0.0695, 0.0733, 0.0772, 0.0836, 0.0896, 0.0989, 0.11,
        0.122, 0.1351, 0.1516, 0.1672, 0.1925, 0.2224, 0.247, 0.2577, 0.2629, 0.2644, 0.2665, 0.2678, 0.2707, 0.2755,
        0.281, 0.2834, 0.2904, 0.2916, 0.2995, 0.3012, 0.3077, 0.3108, 0.322, 0.325, 0.335, 0.34, 0.358, 0.371, 0.393,
        0.41, 0.424, 0.429, 0.436, 0.439, 0.448, 0.448, 0.461, 0.465, 0.478, 0.486, 0.502, 0.516, 0.538, 0.559, 0.592,
        0.624, 0.663, 0.704, 0.756, 0.827, 0.914, 1.007, 1.119, 1.231, 1.356, 1.489, 1.678, 1.7845, 1.9333, 2.0822, 2.2311,
        2.3800, 2.4025, 2.4250, 2.4475, 2.4700, 2.4900, 2.5100, 2.5300, 2.5500, 2.5400, 2.5300, 2.5200, 2.5100, 2.4725,
        2.4350, 2.3975, 2.3600, 2.3100, 2.2600, 2.2100, 2.1600, 2.1375, 2.1150, 2.0925, 2.0700,
    ]
    a_water = list(np.asarray(aw1) * 1e-3) + aw2  # 1e-3 comes from Mason et al. 2016, Table 2

    a_pw = np.interp(lambda_, lambda_water, a_water)

    # use salt water scattering from Zhang et al 2009
    _, b_sw, _, _ = betasw124_ZHH2009(lambda_, S, T)

    bb_sw = b_sw/2 # divide total scattering by 2 to get the backscattering component 

    # use the temp and salinity corrections from Sullivan et al. 2006
    psiT, psiS = tempsal_corr(lambda_)

    # temperature and salinity corrections:
    T_norm = 22.0
    a_sw = (a_pw + psiT * (T - T_norm) + psiS * S)

    return a_sw, bb_sw
