import numpy as np
from matlab_extract import Matlab_Extract
from pathlib import Path
from sys import platform
from scipy.interpolate import interp1d, interpn
from scipy.interpolate import RegularGridInterpolator as rgi
from era40atm import ERA40atm
import math as mt
from lsd_scaling import LSDscaling

dirname = str(Path(__file__).parent.absolute())

class Sample:
    pass

def LSD(lat, lon, alt, atm, age, w, nuclide):
    '''
    DESCRIPTION: 
        This function calculates Lifton, Sato, and Dunai time-dependent scaling factors for a given set of inputs
    ----------
    PARAMETERS:
    lat : float
        sample latitude in deg N (negative values for S hemisphere)
    lon : float
        sample longitude in deg E (negative values for W longitudes, or 0-360 degrees E)
    alt : float
        sample altitude in m above sea level
    atm : int
        atmospheric model to use: 
        1 for U.S. Standard Atmosphere, 0 for ERA-40 Reanalysis
    age : float
        age of sample
    w : int
        gravimetric fractional water content:
        0.066 is default typically about 14% volumetric per Fred Phillips. -1 gives default value
    nuclide : int
        nuclide of interest: 
        26 for 26Al, 10 for 10Be, 14 for 14C, 3 for 3He, 0 for nucleon flux
    -------
    RETURNS
    LSDout : structure
        scaling factors
    '''
    
    consts = Matlab_Extract(dirname + "/cosmo-constants/consts_LSD.mat")
    
    sample = Sample()
    sample.lat = lat
    sample.lon = lon
    sample.alt = alt
    sample.atm = atm
    sample.age = age
    sample.nuclide = nuclide
    
    if nuclide == 14:
        is14 = 1
    elif nuclide == 10:
        is10 = 1
    elif nuclide == 26:
        is26 = 1
    elif nuclide == 3:
        is3 = 1
    else:
        isflux = 1
        
    if sample.atm == 1:
        stdatm = 1
        gmr = -0.03417      # assorted constant
        dtdz = 0.0065       # lapse rate from standard atmosphere
    else:
        stdatm = 0
    
    # make the time vector
    calFlag = 0
    
    # age relative to t0=210
    tv = np.append(np.append(np.arange(0, 60, 10), np.arange(60, 50160, 100)), np.arange(51060, 2001060, 1000))
    tv = np.append(tv, np.logspace(np.log10(2001060), 7, 200))
    LSDRc = np.zeros((len(tv)))
    
    # need solar modulation parameter
    this_SPhi = np.zeros(len(tv)) + float(consts['SPhiInf'])
    this_SPhi[0:120] = consts['SPhi']
    
    if w < 0:
        w = 0.066
        
    # interpolate an M for tv > 7000 ...
    f = interp1d(np.squeeze(consts['t_M']), np.squeeze(consts['M']), fill_value='extrapolate')
    temp_M = f(tv[76:])
    
    # Pressure correction
    if stdatm == 1:
        sample.pressure = 1013.25*np.exp((gmr/dtdz)*(np.log(288.15) - np.log(288.15 - (alt*dtdz))))
    else:
        sample.pressure = ERA40atm(sample.lat, sample.lon, sample.alt)
    
    # catch for negative longitude before Rc interpolation
    if sample.lon < 0:
        sample.lon = sample.lon + 360
    
    # make up the Rc vectors
    loni = np.zeros((1, 1, len(tv[0:76]))) + sample.lon
    lati = np.zeros((1, 1, len(tv[0:76]))) + sample.lat
    tvi = np.zeros((1, 1, len(tv[0:76]))) + tv[0:76]
    
    a = interpn((np.flip(np.squeeze(consts['lat_Rc'])), np.squeeze(consts['lon_Rc']), np.squeeze(consts['t_Rc'])), np.flip(consts['TTRc'], axis=0), np.array([lati, loni, tvi]).T)
    LSDRc[0:76] = np.squeeze(a)
    
    # Fit to trajectory-traced GAD dipole field as f(M/M0), as long-term average
    dd = np.array([6.89901,-103.241,522.061,-1152.15,1189.18,-448.004])
    b = temp_M*(dd[0]*mt.cos(mt.radians(sample.lat)) + dd[1]*mt.cos(mt.radians(sample.lat))**2 + dd[2]*mt.cos(mt.radians(sample.lat))**3 + dd[3]*mt.cos(mt.radians(sample.lat))**4 + dd[4]*mt.cos(mt.radians(sample.lat))**5 + dd[5]*mt.cos(mt.radians(sample.lat))**6)
    LSDRc[76:] = b
    
    # next shop tv
    clipindex = np.where(tv <= sample.age)[0][-1]
    tv2 = tv[0:clipindex+1]
    if tv2[-1] < sample.age:
        tv2 = np.append(tv2, sample.age)
    
    # now shorten the Rc's commensurately
    f = interp1d(tv, LSDRc)
    LSDRc = f(tv2)
    f = interp1d(tv, this_SPhi)
    LSDSPhi = f(tv2)
    
    LSDout = LSDscaling(sample.pressure, LSDRc, LSDSPhi, w, consts, nuclide)
    
    LSDout.tv = tv2
    LSDout.Rc = LSDRc
    LSDout.pressure = sample.pressure
    LSDout.alt = sample.alt
    
    return LSDout
    
if __name__ == '__main__':
    
    result = LSD(60, 40, 1000, 0, 0, -1, 10)
    factor_lsd = result.Be