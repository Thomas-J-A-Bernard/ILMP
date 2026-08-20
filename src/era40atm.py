import numpy as np
from scipy.interpolate import interp2d
from pathlib import Path
from sys import platform

from matlab_extract import Matlab_Extract

dirname = str(Path(__file__).parent.absolute())

def ERA40atm(site_lat, site_lon, site_elv):
    '''
    DESCRIPTION:
        Looks up mean sea level pressure and mean 1000 mb temp from ERA-40 reanalysis
        and calculates site atmospheric pressures using these as inputs to the
        standard atmosphere equation. 
    ----------
    PARAMETERS
    site_lat : floats
        latitude (DD). Southern hemisphere is negative.
    site_long : floats
        longitude (DD). Western hemisphere is negative.
    site_elv : floats
        elevation (m).
    -------
    RETURNS
    out : floats
        Returns site pressure in hPa
    '''
    
    # correct negative longitude
    site_lon = np.where(site_lon < 0, site_lon + 360, site_lon)
    
    # import ERA40 dataset
    if platform == 'win32':
        era = Matlab_Extract(dirname + "\\atmos-constants\\ERA40.mat")
    elif platform == 'linux' or platform == 'linux2':
        era = Matlab_Extract(dirname + "/atmos-constants/ERA40.mat")
        
    ERA40lat = np.squeeze(era['ERA40lat'])
    ERA40lon = np.squeeze(era['ERA40lon'])
    meanP = np.squeeze(era['meanP'])
    meanT = np.squeeze(era['meanT'])
    
    # interpolate sea level pressure and 1000-mb temperature from global reanalyses data grids
    f = interp2d(ERA40lon, ERA40lat, meanP)
    site_spl = np.diag(f(site_lon, site_lat))
    f = interp2d(ERA40lon, ERA40lat, meanT)
    site_T = np.diag(f(site_lon, site_lat))
    
    # parameter and constant
    gmr = -0.03417
    lr = np.array([-6.1517e-3, -3.1831e-6, -1.5014e-7, 1.8097e-9, 1.1791e-10, -6.5359e-14, -9.5209e-15])
    dtdz = -(lr[0] + lr[1]*site_lat + lr[2]*site_lat**2 + lr[3]*site_lat**3 + lr[4]*site_lat**4 + lr[5]*site_lat**5 + lr[6]*site_lat**6)
    
    out = site_spl*np.exp((gmr/dtdz)*(np.log(site_T) - np.log(site_T - (site_elv*dtdz))))
    
    return np.squeeze(out)

if __name__ == '__main__':
    
    pressure2 = ERA40atm(10, 10, 0)
    pressure = ERA40atm(np.array([10, 10]), np.array([10, 10]), np.array([1000, 0]))