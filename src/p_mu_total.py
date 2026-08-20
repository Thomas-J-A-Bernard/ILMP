import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate

class Out():
    '''
    store the complete breakdown of fluxes, stopping rates, production rates, etc ...
    '''

def P_mu_total(z, h, consts, dflag):
    '''
    DESCRIPTION
        calculate the production rate of Al-26 or Be-10 by muons as a function 
        of depth below the surface and site atmospheric pressure
    ----------  
    PARAMETERS
    z : floats
        depth below the surface in g/cm2
    h : floats
        site atmospheric pressure in hPa
    consts : class
        nuclide-specific constant
    dflag : bool
        return only the nuclide production rate from muon reaction if false
    -------
    RETURNS
    floats or class
        nuclide production rate and breakdown of fluxes, stopping rates, production rate, etc ...
    '''
    
    # version of the function
    ver = '1.1'
    
    # figure the atmospheric depth in g/m2
    H = (1013.25 - h)*1.019716
    # find the vertical flux at SLHL
    a = 258.5*(100**2.66)
    b = 75*(100**1.66)
    phi_vert_slhl = (a/((z+21000)*(((z + 1000)**1.66) + b)))*np.exp(-5.5e-6*z)
    
    # get the stopping rate of vertical muons at site
    R_vert_slhl = Rv0(z)
    
    # get the stopping rate of vertical muons at site
    R_vert_site = R_vert_slhl*np.exp(H/LZ(z))
    
    phi_vert_site = np.zeros(max(z.shape))
    # find the flux of vertical muons at site
    for a in range(0, max(z.shape)):
        tol = phi_vert_slhl[a]*1e-4
        func = lambda x: Rv0(x)*np.exp(H/LZ(x))
        temp, fcnt = integrate.quad(func, z[a], (2e5+1))
        phi_vert_site[a] = temp
        
    # invarient flux at 2e5 g/cm2 depth - constant of integration
    phi_200k = (a/((2e5+21000)*(((2e5+1000)**1.66) + b)))*np.exp(-5.5e-6*2e5)
    phi_vert_site = phi_vert_site + phi_200k
    
    # find the total flux of muons at site
    # angular distribution exponent
    nofz = 3.21 - 0.297*np.log((z+H)/100 + 42) + 1.21e-5*(z+H)
    # derivative of same
    dndz = (-0.297/100)/((z+H)/100 + 42) + 1.21e-5
    
    phi_temp = phi_vert_site*2*np.pi/(nofz + 1)
    # convert in muons/cm2/yrs
    phi = phi_temp*60*60*24*365
    
    # find the total stopping rate of muons at site
    R_temp = (2*np.pi/(nofz+1))*R_vert_site - phi_vert_site*(-2*np.pi*((nofz+1)**-2))*dndz
    # convert to negative muons/g/yr
    R = R_temp*0.44*60*60*24*365
    
    # calculate the production rate
    # depth-dependent parts of the fast muon reaction cross-section
    Beta = 0.846 - 0.015*np.log((z/100) + 1) + 0.003139*(np.log((z/100)+1)**2)
    Ebar = 7.6 + 321.7*(1 - np.exp(-8.059e-6*z)) + 50.7*(1-np.exp(-5.05e-7*z))
    
    # internally defined constants
    aalpha = 0.75
    sigma0 = consts.sigma190/(190**aalpha)
    
    # fast muon production
    P_fast = phi*Beta*(Ebar**aalpha)**sigma0**consts.Natoms
    
    # negative muon capture
    P_neg = R*consts.k_neg
    
    if dflag == False:
        return P_fast + P_neg
    elif dflag == True:
        out = Out()
        out.phi_vert_slhl = phi_vert_slhl
        out.R_vert_slhl = R_vert_slhl
        out.phi_vert_site = phi_vert_site
        out.R_vert_site = R_vert_site
        out.phi= phi
        out.R = R
        out.Beta = Beta
        out.Ebar = Ebar
        out.P_fast = P_fast
        out.P_neg = P_neg
        out.H = H
        out.LZ = LZ(z)
        out.ver = ver
        return out
    
def Rv0(z):
    '''
    DESCRIPTION
        calculate the stopping rate of vertically travelling muons as a function
        of depth z at sea level and high latitude
    ----------
    PARAMETERS
    z : floats
        depth at sea level
    -------
    RETURNS
    out : floats
        stooping rate

    '''

    a = np.exp(-5.5e-6*z)
    b = z + 21000
    c = (z + 1000)**1.66 + 1.567e5
    dadz = -5.5e-6*np.exp(-5.5e-6*z)
    dbdz = 1
    dcdz = 1.66*(z+1000)**0.66
    
    out = -5.401e7*(b*c*dadz - a*(c*dbdz + b*dcdz))/(b**2*c**2)
    
    return out

def LZ(z):
    '''
    DESCRIPTION
        returns the effective atmospheric attenuation length for muons
    ----------    
    PARAMETERS
    z : floats
        depth at sea level
    -------
    RETURNS
    out : floats
        effective atmospheric attenuation
    '''
    
    data = np.array([[4.704e1, 8.516e-1],
                     [5.616e1, 1.542e0],
                     [6.802e1, 2.866e0],
                     [8.509e1, 5.698e0],
                     [1.003e2, 9.145e0],
                     [1.527e2, 2.676e1],
                     [1.764e2, 3.696e1],
                     [2.218e2, 5.879e1],
                     [2.868e2, 9.332e1],
                     [3.917e2, 1.524e2],
                     [4.945e2, 2.115e2],
                     [8.995e2, 4.418e2],
                     [1.101e3, 5.534e2],
                     [1.502e3, 7.712e2],
                     [2.103e3, 1.088e3],
                     [3.104e3, 1.599e3],
                     [4.104e3, 2.095e3],
                     [8.105e3, 3.998e3],
                     [1.011e4, 4.920e3],
                     [1.411e4, 6.724e3],
                     [2.011e4, 9.360e3],
                     [3.011e4, 1.362e4],
                     [4.011e4, 1.776e4],
                     [8.011e4, 3.343e4],
                     [1.001e5, 4.084e4],
                     [1.401e5, 5.495e4],
                     [2.001e5, 7.459e4],
                     [3.001e5, 1.040e5],
                     [4.001e5, 1.302e5],
                     [8.001e5, 2.129e5]])
    
    # replace too low values
    z[z < 1] = 1
    
    # obtain momenta using log-linear interpolation
    f = interp1d(np.log(data[:,1]), np.log(data[:,0]))
    P_MeVc = np.exp(f(np.log(z)))
    
    # obtain attenuation lengths
    out = 263 + 150*(P_MeVc/1000)
    
    return out
    