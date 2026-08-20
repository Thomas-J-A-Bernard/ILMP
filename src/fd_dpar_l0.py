import numpy as np
import ctypes as ct
from sys import platform
from pathlib import Path

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

if platform == 'win32':
    ketch_functions = ct.WinDLL(home_dirname + "/bin/ketchlib.dll")
elif platform == 'linux' or platform == 'linux2':
    ketch_functions = ct.CDLL(home_dirname + "/bin/ketchlib.so")
else:
    print('I dont like Mac OS X system')

def Fd_Dpar_L0(time, temp):
    '''
    DESCRIPTION
        subroutine to transform python type variable to c type variable and 
        import the c function ketch main to calculate apatite fission track 
        ages, mean track lenght and density
    ----------
    PARAMETERS
    time : numpy arrays of floats
        time vector for the time-temperature path
    temp : numpy arrays of floats
        temperature vector for the time-temperature vector
    -------
    RETURNS
    age : floats
        apatite fission track ages
    MTL : floats
        apatite mean track length
    density : numpy arrays of floats
        mean track length histogramm density
    '''
    
    # transform time and temp numpy arrays into c arrays
    ls1 = max(time.shape)
    float_array = ct.c_float * ls1
    time_array = float_array(*time.tolist())
    temp_array = float_array(*temp.tolist())
    
    # define some c variables
    alo = ct.c_double(16.3)                 # initial fission track length
    kinetic_par = ct.c_double(1.83)         # Dpar kinetic parameter (Donelick et al., 1999)
    final_age = ct.c_double(0.0)        
    oldest_age = ct.c_double(0.0)
    ls2 = 200                               # number of bins for track length density
    double_array = ct.c_double * ls2
    fmean_array = double_array(*np.zeros(ls2).tolist())
    fdist_array = double_array(*np.zeros(ls2).tolist())
    
    # specified the required argument types of the c function
    ketch_functions.ketch_main.argtypes = ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_double, ct.c_double, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)
    # calculate the apatite fission track dataset
    ketch_functions.ketch_main(ls1, time_array, temp_array, alo, kinetic_par, final_age, oldest_age, fmean_array, fdist_array)
    
    # transform c variables into python variables
    age = final_age.value
    MTL = np.array(fmean_array)[0]
    density = np.array(fdist_array)
    
    return age, MTL, density