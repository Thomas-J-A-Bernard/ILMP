import numpy as np
import matplotlib.pyplot as plt
import ctypes as ct
from sys import platform
import sys
from pathlib import Path

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

if platform == 'win32':
    RDAAM_functions = ct.winDLL(home_dirname + "/bin/RDAAM.dll")
elif platform == 'linux' or platform == 'linux2':
    RDAAM_functions = ct.CDLL(home_dirname + "/bin/RDAAM_profile.so")
else:
    print('I dont like Mac OS X system')

def RDAAM_Profile_Calc(time, temp, radius, ppm_U, ppm_Th, ppm_Sm):

    ntime = max(time.shape)
    rdim = max(ppm_U.shape)
    
    double_array = ct.c_double * ntime
    time_array = double_array(*time.tolist())
    temp_array = double_array(*temp.tolist())
    
    ntime = ct.c_int(ntime)
    radius = ct.c_double(radius)
    
    double_array = ct.c_double * rdim
    ppm_U_array = double_array(*ppm_U.tolist())
    ppm_Th_array = double_array(*ppm_Th.tolist())
    ppm_Sm_array = double_array(*ppm_Sm.tolist())
    
    helium_profile = np.zeros(rdim, dtype=np.float64)
    helium_profile_array = helium_profile.ctypes.data_as(ct.POINTER(ct.c_double))
    
    ap_age = ct.c_double(1.0)
    ap_corrAge = ct.c_double(1.0)
    ap_totHe = ct.c_double(1.0)
    ap_optimize = ct.c_bool(True)
    
    RDAAM_functions.RDAAM_Calculate.argtypes = ct.c_int, ct.c_double, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_bool
    RDAAM_functions.RDAAM_Calculate.restype = ct.c_int
    
    success = RDAAM_functions.RDAAM_Calculate(ntime, radius, ppm_U_array, ppm_Th_array, ppm_Sm_array, time_array, temp_array, ap_age, ap_corrAge, ap_totHe, helium_profile_array, ap_optimize)
    
    age = ap_age.value
    age_corr = ap_corrAge.value

    return age, age_corr, helium_profile

if __name__ == '__main__':
    
    tTpath1 = np.array([[0, 60, 65, 100], [10, 10, 100, 100]])
    tTpath2 = np.array([[0, 100], [10, 100]])
    tTpath3 = np.array([[0.0, 13.1, 14.7, 16.3, 37.0, 56.2, 69.0, 72.2, 73.8, 75.4, 77.0, 83.3, 89.7, 96.1, 102.5, 107.3], [13.3, 20.4, 21.2, 21.7, 24.5, 28.0, 30.6, 31.5, 32.4, 35.1, 41.9, 77.0, 110.3, 141.7, 171.3, 192.1]])
    
    ppm_U = np.ones((64))*10
    ppm_Th = np.ones((64))*10
    ppm_Sm = np.ones((64))*10
    radius = 60
    rdim = len(ppm_U)
    time = np.array([0.0        , 6.70325203, 13.0903894,  17.88074243, 19.47752677, 25.86466414,
                     33.84858585, 37.04215453, 38.63893888, 54.6067823,  60.99391966, 65.78427269,
                     70.57462572, 75.36497874, 76.96176309, 78.55854743, 88.13925348])
    temp = np.array([12.89950144,  46.14366027,  77.15394312,  98.29391958, 102.96172553,
                     114.71381814, 128.91735419, 134.20877665, 136.32518392, 143.50474546,
                     146.71971636, 149.39992772, 152.44446371, 156.16903911, 157.79636317,
                     160.44421913, 197.38936274])
    
    # age1, age_corr1, helium_profile1 = RDAAM_Profile_Calc(tTpath1[0], tTpath1[1], radius, ppm_U, ppm_Th, ppm_Sm)
    # age2, age_corr2, helium_profile2 = RDAAM_Profile_Calc(tTpath2[0], tTpath2[1], radius, ppm_U, ppm_Th, ppm_Sm)
    age1, age_corr2, helium_profile3 = RDAAM_Profile_Calc(time, temp, radius, ppm_U, ppm_Th, ppm_Sm)
    
    # fig = plt.figure(figsize=(6,6))
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    
    # ax1.plot(tTpath1[0], tTpath1[1], ls='-', marker='o', color='darkred', zorder=2)
    # ax1.plot(tTpath2[0], tTpath2[1], ls='-', marker='o', color='steelblue', zorder=2)
    # ax1.axhline(y=80, ls='--', lw=1, color='k', zorder=1)
    # ax1.axhline(y=40, ls='--', lw=1, color='k', zorder=1)
    
    # ax2.plot(np.arange(0, 64), helium_profile1, ls='-', marker='o', color='darkred')
    # ax2.plot(np.arange(0, 64), helium_profile2, ls='-', marker='o', color='steelblue')
    
    # ax1.invert_xaxis()
    # ax1.invert_yaxis()
    
    # ax1.set_xlabel('Time (Ma)')
    # ax1.set_ylabel ('Temperature (°C)')
    
    # ax2.set_xlabel ('Radius (\u03BCm)')
    # ax2.set_ylabel('Helium concentration (nmol/g)')
    
    # fig.tight_layout()