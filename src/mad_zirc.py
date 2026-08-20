import sys
import numpy as np
from scipy.interpolate import interp1d

def Mad_Zirc(time_i, temp_i, out_flag, param_flag):
    '''
    DESCRIPTIONS:
        subroutine to calculate zircon fission track age from a given thermal history. This subroutine is based 
        on the subroutine "ftmod.pas" provided by Peter van der Beek in December 1995. The algorithm is explained in
        Peter's PhD thesis and is based on the work by Lutz and Omar (1991). This adaptation of the program for zircon 
        fission-track annealing was written by Peter in August/September 2006 and is based on algorithms given by 
        Galbraith & Laslett (1997), Tagami et al. (1998) and Rahn et al. (2004).
    ----------
    PARAMETERS
    time_i : array of float
        the time values (in Myr) in descending order at which the thermal history is given (ex: 100,50,20,10,0). 
        The last value should always be 0; the first valueshould be smaller than 1000.
    temp_i : array of float
        the thermal history in degree Celsius
    out_flag : int
        0: only calculate fission track age
        1: also calculate track length distribution and statistics
    param_flag : int
        1: uses parameters for alpha-damaged zircon
        2: uses parameters for zero-damage zircon (parameters from Rahn et al. 2004)
    -------
    RETURNS
    fta : float
        fission track age in Myr
    ftld : array of float
        normalised track length distribution where ftld(k) is the percentage of track with length between k-0.5 and k+0.5 microns
    ftldmean : float
        mean track length in microns
    ftldsd : float
        track length standard deviation in microns
    '''
    
    r = np.zeros((1000))
    prob = np.zeros((101))
    
    if param_flag == 1:
        # alpha damage zircon
        a = -10.77
        b = 2.599e-4
        c = 1.026e-2
    else:
        # zero damage zirco
        a = -11.57
        b = 2.755e-4
        c = 1.075e-2
        
    # unannealed fission track length
    xind = 10.8
    # mean length of standards spontaneous tracks 
    xfct = 10.8
    # calculate the number of time steps
    nstep = int(np.floor(time_i[0]))
    if nstep > 2000:
        sys.exit("Fission track does not work very well for time spans greater than 1Byr")
    elif nstep > 100:
        nstep = 100
        time_interval = time_i[0]/100
    else:
        time_interval = 1
    deltat = time_interval*1e6*365.25*24*3600
    # calculate final temperature
    f = interp1d(time_i, temp_i)
    tempp = f(0) + 273
    rp = 0.5
    
    # begining of time stepping
    for i in range(0, nstep):
        time = (i+1)*time_interval
        
        # calculate temperature by linear interpolation
        f = interp1d(time_i, temp_i)
        temp = f(time) + 273
        
        # calculate mean temperature over the time step
        tempm = (temp + tempp)/2
        
        # calculate the "equivalent time"
        if i == 0:
            teq = 0
        else:
            teq = np.exp((np.log(1-rp) - a - (c*tempm))/(b*tempm))
        
        # calculate length reduction
        dt = teq + deltat
        gr = a + ((b*tempm)*np.log(dt)) + (c*tempm)
        r[i] = 1 - np.exp(gr)
        
        # update variable for next time step
        tempp = temp
        rp = r[i]
        
    # estimate the fission track age by simple summation
    sumdj = 0
    for i in range(0, nstep):
        if r[i] <= 0.4:
            dj = 0
        elif r[i] <= 0.66:
            dj = 2.15*r[i] - 0.76
        else:
            dj = r[i]
        sumdj = sumdj + dj
    
    fta = (xind/xfct)*sumdj*time_interval
    
    # statistic estimations
    if out_flag == 0:
        # first calculate probability density function using Luts and Omar (1991) method and assuming a Gaussian distribution
        sumprob = 0
        for j in range(0, 101):
            rr = j/100
            if rr <= 0.43:
                h = 2.53
            elif rr <= 0.67:
                h = 5.08 - 5.93*rr
            else:
                h = 1.39 - 0.61*rr
            
            fr = 0
            i = np.arange(0, nstep, 1)
            x = (rr - r[i])*xind/h
            fr = sum((0.39894228*np.exp(-(x**2/2)))/h)
            prob[j] = fr/nstep
            sumprob += prob[j]
        
        # rescale the track length distribution, mean and standard deviation
        ftld = np.zeros((11))
        ftld[10] = 100
        imin = 0
        
        for l in range(0, 10):
            imax = np.round((l+1)*100/xind)
            ftld[l] = 0
            for i in range(int(imin), int(imax)):
                ftld[l] = ftld[l] + prob[i]
            ftld[l] = (ftld[l]*100/sumprob)
            ftld[10] = ftld[10] - ftld[l]
            
            imin = imax + 1
        
        sumftld = 0
        for l in range(0, 11):
            sumftld = sumftld + ftld[l]*(l+1-0.5)
        
        ftldmean = sumftld/100
        devftld = 0
        for l in range(0, 11):
            devftld = devftld + ftld[l]*((l+1)-0.5 - ftldmean)**2
        
        ftldsd = np.sqrt(devftld/100)

    else:
        ftld = np.nan
        ftldmean = np.nan
        ftldsd = np.nan
    
    return fta, ftld, ftldmean, ftldsd