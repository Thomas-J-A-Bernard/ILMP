import sys
import numpy as np
from scipy.interpolate import interp1d
from ncepatm_2 import NCEPatm_2
from era40atm import ERA40atm

def Stone2000(lat, P, Fsp):
    '''
    DESCRIPTION:
        calculate the geographic scaling factor for cosmogenic nuclide production as a function of site latitude and atmospheric 
        pressure according to Stone (2000), air pressure and cosmogenic isotope production, JGR.
    ----------
    PARAMETERS:
    lat : float
        latitude in decimal degree
    P : float
        pressure in hPa
    Fsp : float
        fraction (between 0 and 1) of production at sea level and high latitude due to spallation
    -------
    RETURNS
    out : float
        geographic scaling factor
    '''
    
    # check for error in the dataset
    if np.any(lat > 90):
        sys.exit("Latitudes have to be below 90 degrees")
    
    if np.shape(lat) != np.shape(P):
        sys.exit(" latitude and pressure arrays have to be the same dimension")
    
    # transform lat and P into arrays
    if not np.shape(lat) and not np.shape(P):
        lat = np.array([lat])
        P = np.array([P])
    
    # set default Fsp
    nargin = 1
    if nargin == 2:
        Fsp = 0.978
    
    # spallogenic production at index latitude
    # enter constants from table 1
    a = np.array([31.8518, 34.3699, 40.3153, 42.0983, 56.7733, 69.0720, 71.8733])
    b = np.array([250.3193, 258.4759, 308.9894, 512.6857, 649.1343, 832.4566, 863.1927])
    c = np.array([-0.083393, -0.089807, -0.106248, -0.120551, -0.160859, -0.199252, -0.207069])
    d = np.array([7.4260e-5, 7.9457e-5, 9.4508e-5, 1.1752e-4, 1.5463e-4, 1.9391e-4, 2.0127e-4])
    e = np.array([-2.2397e-8, -2.3697e-8, -2.8234e-8, -3.8809e-8, -5.0330e-8, -6.3653e-8, -6.6043e-8])
    ilats = [0, 10, 20, 30, 40, 50, 60]
    
    # calculate index latitude at given pressure
    lat0 = a[0] + (b[0]*np.exp(P/(-150))) + (c[0]*P) + (d[0]*(P**2)) + (e[0]*(P**3))
    lat10 = a[1] + (b[1]*np.exp(P/(-150))) + (c[1]*P) + (d[1]*(P**2)) + (e[1]*(P**3))
    lat20 = a[2] + (b[2]*np.exp(P/(-150))) + (c[2]*P) + (d[2]*(P**2)) + (e[2]*(P**3))
    lat30 = a[3] + (b[3]*np.exp(P/(-150))) + (c[3]*P) + (d[3]*(P**2)) + (e[3]*(P**3))
    lat40 = a[4] + (b[4]*np.exp(P/(-150))) + (c[4]*P) + (d[4]*(P**2)) + (e[4]*(P**3))
    lat50 = a[5] + (b[5]*np.exp(P/(-150))) + (c[5]*P) + (d[5]*(P**2)) + (e[5]*(P**3))
    lat60 = a[6] + (b[6]*np.exp(P/(-150))) + (c[6]*P) + (d[6]*(P**2)) + (e[6]*(P**3))
    
    # initialize output
    correction = np.zeros((np.shape(P)))
    
    # northernize southern-hemisphere inputs
    lat = abs(lat)
    
    # set hight latitude to 60
    lat[lat > 60] = 60
    
    # loop calculation
    S = np.zeros((max(lat.shape)))
    for i in range(0, max(lat.shape)):
        # interpolate for actual elevation
        f = interp1d(ilats, np.array([lat0[i], lat10[i], lat20[i], lat30[i], lat40[i], lat50[i], lat60[i]]))
        S[i] = f(lat[i])
    
    # production by muons
    # constants
    mk = np.array([0.587, 0.600, 0.678, 0.833, 0.933, 1.000, 1.000])
    
    # index latitudes at given pressure
    ml0 = mk[0]*np.exp((1013.25 - P)/242)
    ml10 = mk[1]*np.exp((1013.25 - P)/242)
    ml20 = mk[2]*np.exp((1013.25 - P)/242)
    ml30 = mk[3]*np.exp((1013.25 - P)/242)
    ml40 = mk[4]*np.exp((1013.25 - P)/242)
    ml50 = mk[5]*np.exp((1013.25 - P)/242)
    ml60 = mk[6]*np.exp((1013.25 - P)/242)
    
    # loop calculation
    M = np.zeros((max(lat.shape)))
    for i in range(0, max(lat.shape)):
        # interpolate for actual elevation
        f = interp1d(ilats, np.array([ml0[i], ml10[i], ml20[i], ml30[i], ml40[i], ml50[i], ml60[i]]))
        
    # Combine spallogenic and muogenic production
    Fm = 1 - Fsp
    out = ((S*Fsp) + (M*Fm))
    
    return out

if __name__ == '__main__':
    
    pressure = ERA40atm(60, 40, 1000)
    factor_stone = Stone2000(60, pressure, 0.9938)