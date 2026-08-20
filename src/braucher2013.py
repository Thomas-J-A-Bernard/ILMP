import numpy as np

def Braucher2013(P, Fsp):
    '''
    DESCRIPTION
        calculate the geographic scaling factor for cosmogenic-muon production as a function
        of site atmospheric pressure, according to Braucher et al., 2013
    ----------
    PARAMETERS
    P : floats
        pressure in hPa
    Fsp : floats
        fixe parameter
    RETURNS
    -------
    out : floats
        geographic scaling factor
    '''
    
    MP0 = 0.028
    MP0_sd = 0.004
    
    out = MP0*np.exp((1013.25 - P)/247)
    
    return out