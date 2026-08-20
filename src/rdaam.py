import numpy as np
import ctypes as ct
from sys import platform
from pathlib import Path

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

if platform == 'win32':
    RDAAM_functions = ct.winDLL(home_dirname + "/bin/RDAAM.dll")
elif platform == 'linux' or platform == 'linux2':
    RDAAM_functions = ct.CDLL(home_dirname + "/bin/RDAAM.so")
else:
    print('I dont like Mac OS X system')
    
def RDAAM_Calculation(time, temp):
    '''
    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    time : TYPE
        DESCRIPTION.
    temp : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    
    ntime = max(time.shape)
    double_array = ct.c_double * ntime
    time_array = double_array(*time.tolist())
    temp_array = double_array(*temp.tolist())
    ntime = ct.c_int(ntime)
    ap_age = ct.c_double(1.0)
    ap_corrAge = ct.c_double(1.0)
    ap_totHe = ct.c_double(1.0)
    ap_optimize = ct.c_bool(True)
    
    RDAAM_functions.RDAAM_Calculate.argtypes = ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_bool
    
    RDAAM_functions.RDAAM_Calculate(ntime, time_array, temp_array, ap_age, ap_corrAge, ap_totHe, ap_optimize) # do not work (memory access problem when calling c++ function)
    
    age = ap_age.value
    
    return age

if __name__ == '__main__':
    
    time = np.array([0.0        , 6.70325203, 13.0903894,  17.88074243, 19.47752677, 25.86466414,
                     33.84858585, 37.04215453, 38.63893888, 54.6067823,  60.99391966, 65.78427269,
                     70.57462572, 75.36497874, 76.96176309, 78.55854743, 88.13925348])
    temp = np.array([12.89950144,  46.14366027,  77.15394312,  98.29391958, 102.96172553,
                     114.71381814, 128.91735419, 134.20877665, 136.32518392, 143.50474546,
                     146.71971636, 149.39992772, 152.44446371, 156.16903911, 157.79636317,
                     160.44421913, 197.38936274])
    
    tTpath2 = np.array([[0, 60, 65, 120], [10, 20, 140, 150]])
    
    age = RDAAM_Calculation(tTpath2[0], tTpath2[1])
    print(age)