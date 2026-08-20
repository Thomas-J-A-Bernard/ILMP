import numpy as np

def Thickness(zmax, L, rho):
    '''
    DESCRIPTION:
        Calculates the thickness correction for cosmogenic nuclide 
        production by spallation, in a sample of thickness zmax (cm)
        and density rho (g/cm3), with effective attenuation length 
        Lambda (g/cm2). 
    ----------
    PARAMETERS
    zmax : floats
        sample thickness
    L : floats
        DESCRIPTION.
    rho : float
        crustal density 
    -------    
    RETURNS
    out_l : floats
        thickness correction
    '''
    
    out_l = (L/(rho*zmax))*(1 - np.exp(((-1*rho*zmax)/L)))
    
    return out_l