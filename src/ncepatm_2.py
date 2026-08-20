import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline
from pathlib import Path
from sys import platform
from matlab_extract import Matlab_Extract

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

if platform == 'win32':
    ncep2 = Matlab_Extract(home_dirname + "\\data\\atmos-constants\\NCEP2.mat")
elif platform == 'linux' or platform == 'linux2':
    ncep2 = Matlab_Extract(home_dirname + "/data/atmos-constants/NCEP2.mat")

def NCEPatm_2(site_lat, site_long, site_elv):
    '''
    DESCRIPTION
        Looks up surface pressure and 1000 mb temp from NCEP reanalysis and calculates 
        site atmospheric pressures using these as inputs to the standard atmosphere equation.
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
    site_long = np.where(site_long < 0, site_long + 360, site_long)
    

    
    NCEPlon = np.squeeze(ncep2['NCEPlon'])
    NCEPlat = np.squeeze(ncep2['NCEPlat'])
    meant1000 = np.squeeze(ncep2['meant1000'])
    meanslp = np.squeeze(ncep2['meanslp'])
    
    if not np.all(np.diff(NCEPlat) > 0):
        NCEPlat = NCEPlat[::-1]
        meanslp = meanslp[::-1, :]
        meant1000 = meant1000[::-1, :]

    if not np.all(np.diff(NCEPlon) > 0):
        NCEPlon = NCEPlon[::-1]
        meanslp = meanslp[:, ::-1]
        meant1000 = meant1000[:, ::-1]
    
    # interpolate sea level pressure and 1000-mb temperature from global reanalyses data grids
    # f = interp2d(NCEPlon, NCEPlat, meanslp)
    f = RectBivariateSpline(NCEPlat, NCEPlon, meanslp)
    site_spl = np.diag(f(site_long, site_lat))
    # f = interp2d(NCEPlon, NCEPlat, meant1000)
    f = RectBivariateSpline(NCEPlat, NCEPlon, meant1000)
    site_T = np.diag(f(site_long, site_lat))
    site_T_degK = site_T + 273.15
    
    # additional parameters
    gmr = -0.03417
    dtdz = 0.0065
    
    out = site_spl*np.exp((gmr/dtdz)*(np.log(site_T_degK) - np.log(site_T_degK - (site_elv*dtdz))))
    
    return np.squeeze(out)

if __name__ == '__main__':
    
    pressure = NCEPatm_2(20, 20, 1000)
    pressure2 = NCEPatm_2(np.array([10, 20]), np.array([10, 20]), np.array([1000, 1000]))