import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from rdaam_profile import RDAAM_Profile_Calc
# from scipy.ndimage import distance_transform_edt
from scipy.ndimage import convolve
# import skimage
# import sys
import warnings

#ignore specific warning messages
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")

def Prof1Dto3D(r):
    
    m, n, p = 2*r+1, 2*r+1, 2*r+1
    px, py, pz = np.meshgrid(np.arange(1, n+1), np.arange(1, m+1), np.arange(1, p+1))
    xc, yc, zc = r+1, r+1, r+1
    R = np.sqrt((px - xc)**2 + (py - yc)**2 + (pz - zc)**2)
    R = R / np.max(R[r, r, 0])
    R[R > 1] = 0
    
    return R

def GridCircle(sigma):
    
    sigma = int(np.ceil(sigma -0.5))
    neighborhood = []

    X = int(sigma)
    for i in range(-X, X + 1):
        Y = int(pow(sigma * sigma - i * i, 1/2))
        for j in range(-Y, Y + 1):
            neighborhood.append((sigma + i, sigma + j))
            
    points = neighborhood
    
    mask = np.zeros((sigma*2+1, sigma*2+1))
    for i in range(0, np.shape(points)[0]):
        mask[points[i]] = 1
            
    return mask

def FSpecialDisk(sigma):
    
    rad = sigma
    crad = np.ceil(rad - 0.5)
    x, y = np.meshgrid(np.arange(-crad, crad+1), np.arange(-crad, crad+1))
    maxxy = np.maximum(np.abs(x), np.abs(y))
    minxy = np.minimum(np.abs(x), np.abs(y))
    
    m1 = (rad**2 < (maxxy + 0.5)**2 + (minxy - 0.5)**2) * (minxy - 0.5) + (rad**2 >= (maxxy + 0.5)**2 + (minxy - 0.5)**2) * np.sqrt(rad**2 - (maxxy + 0.5)**2)
    m2 = (rad**2 > (maxxy - 0.5)**2 + (minxy + 0.5)**2) * (minxy + 0.5) + (rad**2 <= (maxxy - 0.5)**2 + (minxy + 0.5)**2) * np.sqrt(rad**2 - (maxxy - 0.5)**2)
    
    sgrid = (rad**2 * (0.5 * (np.arcsin(m2 / rad) - np.arcsin(m1 / rad)) + \
             0.25 * (np.sin(2 * np.arcsin(m2 / rad)) - np.sin(2 * np.arcsin(m1 / rad)))) - \
             (maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5)) * \
             ((((rad**2 < (maxxy + 0.5)**2 + (minxy + 0.5)**2) & \
             (rad**2 > (maxxy - 0.5)**2 + (minxy - 0.5)**2)) | \
             ((minxy == 0) & (maxxy - 0.5 < rad) & (maxxy + 0.5 >= rad))))
                 
    sgrid = sgrid + ((maxxy + 0.5)**2 + (minxy + 0.5)**2 < rad**2)
    sgrid[int(crad), int(crad)] = min(np.pi * rad**2, np.pi / 2)
    
    if (crad > 0) and (rad > crad - 0.5) and (rad**2 < (crad - 0.5)**2 + 0.25):
        
        m1 = np.sqrt(rad**2 - (crad - 0.5)**2)
        m1n = m1 / rad
        sg0 = 2 * (rad**2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))
        crad = int(crad)
        sgrid[2*crad, crad] = sg0
        sgrid[crad, 2*crad] = sg0
        sgrid[crad, 0] = sg0
        sgrid[0, crad] = sg0
        sgrid[2*crad-1,crad] = sgrid[2*crad-1,crad] - sg0
        sgrid[crad, 2*crad-1] = sgrid[crad, 2*crad-1] - sg0
        sgrid[crad, 1] = sgrid[crad, 1] - sg0
        sgrid[1, crad] = sgrid[1, crad] - sg0
        
    sgrid[int(crad), int(crad)] = min(sgrid[int(crad), int(crad)], 1)
    h = sgrid/np.nansum(sgrid)
    
    return(h)

def CalculateHeliumAgeUncorrected(mineral, He, He_SD, U238, U235, U_SD, Th232, Th_SD, Sm147, Sm_SD):
    
    # Constants
    lambda238 = 1.55125E-10
    lambda235 = 9.84500E-10
    lambda232 = 4.94750E-11
    lambda147 = 6.53900E-12
    
    uncor_age = np.zeros(1000)
    U238_rand = np.random.normal(U238, U238 * U_SD / 100, 1000)
    U235_rand = np.random.normal(U235, U235 * U_SD / 100, 1000)
    Th232_rand = np.random.normal(Th232, Th232 * Th_SD / 100, 1000)
    Sm147_rand = np.random.normal(Sm147, Sm147 * Sm_SD / 100, 1000)
    He_rand = np.random.normal(He, He_SD, 1000)
    
    if mineral == 'apatite':
        
        production = 8*lambda238*U238_rand + 7*lambda235*U235_rand + 6*lambda232*Th232_rand + 1*lambda147*Sm147_rand
        lambda_weight = (8*lambda238**2*U238_rand + 7*lambda235**2*U235_rand + 6*lambda232**2*Th232_rand + 1*lambda147**2*Sm147_rand)/production
        uncor_age = 1/lambda_weight*np.log(lambda_weight/production*He_rand + 1)/1e6
        uncor_age_mean = np.mean(uncor_age)
        uncor_age_SD = np.std(uncor_age)
        
    elif mineral == 'zircon':
        
        production = 8*lambda238*U238_rand + 7*lambda235*U235_rand + 6*lambda232*Th232_rand
        lambda_weight = (8*lambda238**2*U238_rand + 7*lambda235**2*U235_rand + 6*lambda232**2*Th232_rand)/production
        uncor_age = 1/lambda_weight*np.log(lambda_weight/production*He_rand + 1)/1e6
        uncor_age_mean = np.mean(uncor_age)
        uncor_age_SD = np.std(uncor_age)
        
    else:
        
        uncor_age_mean = 0
        uncor_age_SD = 0
        
    return uncor_age_mean, uncor_age_SD

def HeliumGrainAge(time, temp, radius=100, U=10, U_error=0, Th=10, Th_error=0, Sm=10, Sm_error=0, spot_size=15, spot_depth=7):

    zpos = radius
    
    Avg = 6.022e23
    ApRho = 3.2
    mm238 = 238.0508
    mm235 = 235.0439
    mm232 = 232.0380
    mm147 = 146.9149
    R235238 = 1/136.1384
    
    Uvec = np.ones(100)*U
    Thvec = np.ones(100)*Th
    Smvec = np.ones(100)*Sm
    
    rdim = 64
    f = interp1d(np.arange(0, len(Uvec)), Uvec, kind='linear', fill_value="extrapolate")
    Uvec = f(np.linspace(0, len(Uvec)-1, rdim))
    f = interp1d(np.arange(0, len(Thvec)), Thvec, kind='linear', fill_value="extrapolate")
    Thvec = f(np.linspace(0, len(Thvec)-1, rdim))
    f = interp1d(np.arange(0, len(Smvec)), Smvec, kind='linear', fill_value="extrapolate")
    Smvec = f(np.linspace(0, len(Smvec)-1, rdim))
    
    age , age_corr, HeProfile = RDAAM_Profile_Calc(time, temp, radius, Uvec, Thvec, Smvec)
    
    # SER = 3/2*radius
    # _, age_corr,_ = RDAAM_Profile_Calc(time, temp, SER, Uvec, Thvec, Smvec)
    
    f = interp1d(np.linspace(0,1,len(HeProfile)), HeProfile)
    HeProfile = f(np.linspace(0,1,101))
    
    dx = radius/(len(HeProfile)-1)
    HeGrain = HeProfile
    
    spot_area = np.pi*spot_size**2
    spot_volume = spot_area*spot_depth
    pixel_mass = dx*dx*dx*ApRho*1e-12
    spot_mass = spot_volume*ApRho*1e-12
    
    M = Prof1Dto3D(len(HeProfile)-1)
    M = np.round(M*(len(HeProfile) - 1))
    HeGrainAtGr = HeGrain*1e-9*Avg
    HeGrainAtPixel = HeGrainAtGr*pixel_mass
    
    HeGrain3D = np.zeros(np.shape(M))
    for i in range(0, len(HeGrain3D[0])):
        for j in range(0, len(HeGrain3D[1])):
            for k in range(0, len(HeGrain3D[2])):
                if M[i,j,k] > 0:
                    HeGrain3D[i,j,k] = HeGrainAtPixel[int(M[i,j,k])]
    
    disk_filter = FSpecialDisk(spot_size/dx)
    disk_filter = disk_filter/np.nanmax(disk_filter)
    
    HeGrainSpot = np.zeros((M.shape[0], M.shape[1]))
    for i in range(0, int(np.floor(spot_depth/dx))):
        HeGrainSpot = HeGrainSpot + convolve(HeGrain3D[int(round(zpos/dx)) + i - 2,:,:], disk_filter, mode='constant', cval=0.0)
    
    diffZ = spot_depth % dx
    if diffZ > 0:
        if np.floor(spot_depth/dx)<1:
            i=0
        HeGrainSpot = HeGrainSpot + diffZ/dx*convolve(HeGrain3D[int(round(zpos/dx)) + i - 2,:,:], disk_filter, mode='constant', cval=0.0)
        
    HeGrainSpotProfile = np.copy(HeGrainSpot[int(round(radius/dx)),:])
    HeGrainSpotProfile[:int(round(spot_size/dx))] = np.nan
    HeGrainSpotProfile[-int(round(spot_size/dx)):] = np.nan
    
    U238spot=U*1e-6*spot_mass*Avg/mm238
    U235spot=U*1e-6*R235238*spot_mass*Avg/mm235
    Th232spot=Th*1e-6*spot_mass*Avg/mm232
    Sm147spot=Sm*1e-6*spot_mass*Avg/mm147
    
    U238spoterror=U_error*1e-6*spot_mass*Avg/mm238
    Th232spoterror=Th_error*1e-6*spot_mass*Avg/mm232
    Sm147spoterror=Sm_error*1e-6*spot_mass*Avg/mm147
    
    transect_ages=np.zeros(np.shape(HeGrainSpotProfile));
    for i in range(0, len(HeGrainSpotProfile)):
        if HeGrainSpotProfile[i] > 0:
            transect_ages[i],_ = CalculateHeliumAgeUncorrected('apatite', HeGrainSpotProfile[i], 0, U238spot, U235spot, 0, Th232spot, 0, Sm147spot, 0)
    transect_ages[transect_ages <= 0] = np.nan
    
    return age, age_corr, HeProfile, transect_ages

    
if __name__ == '__main__':
    
    radius = 100
    tTpath1 = np.array([[0, 10, 100, 120],[10, 50, 70, 150]])
    tTpath2 = np.array([[0, 60, 65, 120], [10, 20, 140, 150]])
    tTpath3 = np.array([[0.0, 5.11, 8.30, 22.67, 25.86, 27.46, 30.66, 56.20, 60.99, 64.19, 65.78, 67.38, 68.98, 70.57, 76.96, 89.74, 100.91, 105.70, 108.90, 115.28],
                        [11.38, 14.23, 16.63, 30.41, 33.22, 34.42, 36.33, 49.43, 52.05, 54.28, 55.89, 58.07, 60.92, 64.29, 78.55, 106.13, 129.57, 140.23, 148.32, 165.88]])
    
    age1, age_corr1, helium_profile1, transect_ages1 = HeliumGrainAge(tTpath1[0], tTpath1[1], radius)
    # age2, age_corr2, helium_profile2, transect_ages2 = HeliumGrainAge(tTpath2[0], tTpath2[1], radius)
    # age3, age_corr3, helium_profile3, transect_ages3 = HeliumGrainAge(tTpath3[0], tTpath3[1], radius)
    
    age_length = len(transect_ages1)
    insitu_age1 = transect_ages1[int(age_length*0.125)]
    print(insitu_age1)
       
    #%% Plot time-temperature path
    
    # fig1 = plt.figure(figsize=(6,4))
    # ax1 = fig1.add_subplot(111)
    # ax1.plot(tTpath1[0], tTpath1[1], ls='-', marker='o', ms=5, color='darkred', zorder=3)
    # ax1.plot(tTpath2[0], tTpath2[1], ls='-', marker='o', ms=5, color='steelblue', zorder=3)
    # ax1.plot(tTpath3[0], tTpath3[1], ls='-', marker='o', ms=5, color='darkgreen', zorder=3)
    # ax1.axhline(y=80, ls='--', lw=1, color='k', zorder=2)
    # ax1.axhline(y=40, ls='--', lw=1, color='k', zorder=2)
    # ax1.fill_between(x=[-5,125], y1=[40,40], y2=[80,80], color='k', alpha=0.15, zorder=1)
    # ax1.axis([-5,125,0,175])
    # ax1.invert_xaxis()
    # ax1.invert_yaxis()
    # ax1.set_xlabel('Time (Ma)')
    # ax1.set_ylabel ('Temperature (°C)')
    # fig1.tight_layout()
    # # fig1.savefig('time-temperature_path.png', dpi=720)
    
    #%% Plot helium profile
    
    # fig2 = plt.figure(figsize=(6,4))
    # ax1 = fig2.add_subplot(111)
    # ax1.plot(np.linspace(radius, 0, len(helium_profile1)), helium_profile1, ls='-', marker='o', ms=5, color='darkred')
    # ax1.plot(np.linspace(radius, 0, len(helium_profile2)), helium_profile2, ls='-', marker='o', ms=5, color='steelblue')
    # ax1.plot(np.linspace(radius, 0, len(helium_profile3)), helium_profile3, ls='-', marker='o', ms=5, color='darkgreen')
    # ax1.set_xlabel ('Distance through grain radius (\u03BCm)')
    # ax1.set_ylabel('Helium concentration (nmol/g)')
    # # ax1.axis([-4,137,0,6.5])
    # ax1.invert_xaxis()
    # fig2.tight_layout()
    # # fig2.savefig('helium_profile.png', dpi=720)
    
    #%% Plot helium age transect
    
    dx = radius/(len(helium_profile1)-1)
    fig3 = plt.figure(figsize=(12,4))
    ax1 = fig3.add_subplot(111)
    ax1.plot(np.arange(len(transect_ages1)), transect_ages1, ls='-', marker='o', ms=5, color='darkred')
    # ax1.plot(np.arange(len(transect_ages2)), transect_ages2, ls='-', marker='o', ms=5, color='steelblue')
    # ax1.plot(np.arange(len(transect_ages3)), transect_ages3, ls='-', marker='o', ms=5, color='darkgreen')
    xtick = ax1.get_xticks()
    x = xtick*dx
    x = x[0:int(np.ceil(len(x)/2))]
    x = np.concatenate((x,np.flip(x[0:-1])))   
    ax1.set_xticklabels(np.round(x, decimals=1))
    ax1.set_xlabel ('Distance through grain radius (\u03BCm)')
    ax1.set_ylabel('Helium age (Ma)')
    fig3.tight_layout()
    # fig3.savefig('helium_age_transect.png', dpi=720)