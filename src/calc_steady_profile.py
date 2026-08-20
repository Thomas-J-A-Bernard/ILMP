import numpy as np
from scipy.interpolate import interp1d

def Calc_Steady_Profile(c3, E, muon):
    '''
    DESCRIPTION
        calculate the cosmogenic nuclide steady state profile
    ----------
    PARAMETERS
    c3 : class
        nuclide specific parameters
    E : float
        erosion at the surface
    muon : float
        muon parameter
    -------
    RETURNS
    N_profile : numpy array of float
        cosmogenic nuclide steady state profile
    depth_array : numpy array of float
        time vector
    '''
    
    dt = 100
    t_target = np.arange(0, 1000000+dt, dt)
    depth_array = t_target*E                    # array in g/cm2
    
    if muon == 1 :
        P_mu_z_target = c3.P_mu*np.exp(-depth_array/c3.L_muon)
    else:
        f = interp1d(c3.z_mu, c3.P_mu_z, depth_array)
        P_mu_z_target = f(depth_array)
        P_mu_z_target[P_mu_z_target == np.nan] = 0
        
    P_sp_z_target = c3.P_sp_t*np.exp(-(depth_array)/c3.L)   # calculate nuclide production for modelled time vector (depth)
    N_profile = np.zeros(np.shape(P_mu_z_target))
    P_total = P_mu_z_target + P_sp_z_target
    P_total12 = np.append([P_total[1:]], [0])
    
    N_profile[-1] = P_total[-1]/(1-np.exp(-c3.l))
    
    N_profile = np.flip(np.cumsum(np.flip(P_total)*dt*np.exp(-c3.l*dt)))
    
    return N_profile, depth_array
    