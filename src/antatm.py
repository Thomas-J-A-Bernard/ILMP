import numpy as np

def Antatm(z):
    '''
    DESCRIPTION
        This function converts elevation to atmospheric pressure according 
        to a best-fit relationship for Antarctic stations.
    ----------
    PARAMETER
    z : floats
        Elevation in m.
    -------
    RETURN
    out_l : floats
        atmospheric pressure
    '''
    
    out_l = 989.1*np.exp(z/(-7588))
    
    return out_l