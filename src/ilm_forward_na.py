import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from shapely.geometry import LineString
from pathlib import Path
from sys import platform

from mad_zirc import Mad_Zirc
from fd_dpar_l0 import Fd_Dpar_L0
from rdaam import RDAAM_Calculation
from InSituAge import HeliumGrainAge
from matlab_extract import Matlab_Extract
from ncepatm_2 import NCEPatm_2
from antatm import Antatm
from thickness import Thickness
from stone2000 import Stone2000
from braucher2013 import Braucher2013
from p_mu_total import P_mu_total
from calc_steady_profile import Calc_Steady_Profile
from general_functions import Find_Upstream_Index, Find_Downstream_Index
from lsd import LSD

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

if platform == 'win32':
    al_be_consts_v22 = Matlab_Extract(home_dirname + "\\data\\cosmo-constants\\al_be_consts_v22.mat")
elif platform == 'linux' or platform == 'linux2':
    al_be_consts_v22 = Matlab_Extract(home_dirname + "/data/cosmo-constants/al_be_consts_v22.mat")

class Thermo:
    '''
    store the time temperature paths of the different samples
    '''
    pass

class Thermo_Data:
    '''
    store the time ZFT, AHe, AFT ages and AFT mtl of the different samples
    '''
    pass

class Mc:
    '''
    store nuclide-specific constants information
    '''
    pass

class C3:
    '''
    store the nuclide-specific results
    '''
    pass

class Misfit:
    '''
    store the misfit results
    '''

class Capture:
    '''
    Store river capture information
    '''

class Modelled_Data():
    '''
    store the elevation, thermochronological ages and cosmogenic concentrations calculated by the model
    '''
    tcn = np.array([])
    zfta = np.array([])
    afta = np.array([])
    aftmtl = np.array([])
    ahea = np.array([])
    time = np.array([])
    temperature = np.array([])

def Size(arr):
    '''
    DESCRIOTION:
        Return dimension of a matrice in matlab style (always in two dimensions)
    ----------
    PARAMETERS
    arr : array of float
        DESCRIPTION.
    -------
    RETURNS
    Tuple
        dimension of the matrice
    '''
    
    if len(arr.shape) == 1:
        return 1, arr.shape[0]
    return arr.shape

def Length(arr):
    '''
    DESCRIPTION:
        Return length of a matrice in matlab style
    ----------    
    PARAMETERS
    arr : array of float
        DESCRIPTION.
    -------
    RETURNS
    int
        length of the matrice
    '''
    
    if np.size(arr) == 1:
        return 1
    else:
        return max(arr.shape)

def Ilm_Forward_Na(param, data, crn_calc=False, ahea_calc=False, afta_calc=False, aftmtl_calc=False, inverse=False):
    '''
    DESCRIPTION:
        Forward function for the prediction of topographic, cosmogenic and thermochronological data
    ----------
    PARAMETERS
    param : class object
        parameter for the forward calculation
    data : dict
        dataset of the basin
    crn_calc : bool
        calculate the cosmogenic radionuclide concentration if true
    ahea_calc: bool
        calculate apatite helium age if true
    afta_calc : bool
        calculate apatite fission track age if true
    aftmtl_calc: bool
        calculate apatite fission track mean track length
    inverse : bool
        do not write and do not return results if true
    -------
    RETURNS
    misfit : class object
        misfit between the observed and modelled dataset
    '''
    
    modelled_data = Modelled_Data()
    
    ### ======================= 1:PARAMETERS ============================== ###
    
    gg = param.gg                       # geothermal gradient
    T0 = param.T0                       # surface temperature
    lr = param.lr                       # atmospheric lapse rate
    TD = param.TD                       # thermal diffusivity
    rho_c = param.rho_c                 # crustal density
    hpc = param.hpc                     # crustal heat production
    cp_c = param.cp_c                   # specific heat capacity of granite
    
    tt = param.tt                       # model duration of Myr
    dt = param.dt                       # time step 
    dtr = param.dtr                     # laps time record in Myr
    start_dtr = param.start_dtr         # start of model record
    end_dtr = param.end_dtr             # end of model record
    
    U = param.U*1e-3                    # uplift rate converted to m/yr
    K = param.K                         # erodibility 
    m = param.m                         # area exponent
    n = param.n                         # slope exponent
    icflag = param.icflag               # slope shape
    islope = param.islope               # constant slope
    Ui = param.Ui                       # initial uplift rate (if icflag = 3)
    Ki = param.Ki                       # initial erodibility (if icflag = 3)
    ee = param.ee                       # elevation uncertainty
    pixel = param.pixel                 # area unit
    
    hl = param.hl                       # hillslope length
    hdn = param.hdn + 1                 # number of hillslope node
    hk = param.hk                       # hillslope diffusion rate 
    hm = param.hm                       # distance exponent
    hn = param.hn                       # slope exponent
    crit_slope = param.crit_slope       # criticall hillslope
    
    muon = param.muon                   # muon model production
    dx_cosmo = param.dx_cosmo           # distance between sample location for tcn calculation
    t_record = param.t_record           # time-span for tcn calculation
    
    initial_elevation = np.squeeze(data['initial_elevation'])           # observed river profile elevation
    x = np.squeeze(data['x'])                                           # node flow distance
    area = np.squeeze(data['area'])                                     # node area
    pairs = np.squeeze(data['pairs'])                                   # node pairs
    dx_dem = np.squeeze(data['dx_dem'])                                 # node resolution
    skipping_factor = np.squeeze(data['skipping_factor'])               # factor pr dem pixel (mostly ignored since we use area)
    capture = data['capture']                                           # river capture information
    drop = data['base_level_drop']                                      # base level information
    
    # initialize node indexes of thermo samples
    if (ahea_calc == True or afta_calc == True or aftmtl_calc == True):
        thermo_node = np.squeeze(data['thermo_meas']['node'])          
    
    # update parameters for change in uplift through time    
    if hasattr(param, 'ua'):            
        ua = param.ua
    else:
        ua = []    

    # update parameters for spatial and temporal variability in uplift
    if hasattr(param, 'nuv'):           
        nuv = param.nuv
    else:
        nuv = 0
    
    # update parameters for spatial variability in erodibility    
    if hasattr(param, 'kflag'):         
        kflag = param.kflag
    else:
        kflag = 0
        
    # update parameters for spatial variability in hillsope length
    if hasattr(param, 'hlflag'):         
        hlflag = param.hlflag
    else:
        hlflag = 0
    
    ### ================= 2:RIVER AND HILLSLOPE MODEL ===================== ###
    
    start_time = datetime.datetime.now()
    
    # generate initial river profiles according to icflag
    z0 = np.zeros(np.shape(initial_elevation))
    # generate an initial slope for rivers 
    if icflag == 1:                                 
        z0 = initial_elevation[0]+x*islope
    # generate initial flat plateau which have elevation of the more downstream river node
    elif icflag == 2:                               
        if 'drop1' in drop:
            z0[:] = drop['drop1']['initial_level']
        else:
            z0[:] = initial_elevation[0]
    # generate an initial steady-state river profile based on the initial uplift (Ui) and erodibility (Ki)
    elif icflag == 3:                               
        if pixel == 0:
            S = (Ui*1e-3/(Ki*((area*dx_dem**2)**m)))**(1/n)
        else:
            S = (Ui*1e-3/(Ki*(area**m)))**(1/n)        
        z0[0] = initial_elevation[0]
        z0[1:] = initial_elevation[np.transpose(pairs[:,0].astype(int)-1)]*S[1:]*dx_dem*skipping_factor
    
    # define an uniform or spatial variable hillslope length
    if hlflag == 0:
        hl = np.ones(np.shape(z0))*hl
        print('test1')
    elif hlflag == 1:
        hl = data['hillslope']['length']
        print('test2')
    
    hdx = hl/(hdn - 1)
    # generate initial hillslope profile according to icflag
    
    # change 26.07.13
    # hillx = np.linspace(hl, 0, hdn)
    hillx = np.linspace(1, 0, hdn)[:, None] * hl[None, :]
    
    
    hill0 = np.zeros((hdn, max(z0.shape)))
    
    # generate hillslope for an initial slope for rivers
    if icflag == 1:
        for i in range(0, max(z0.shape)):
            hill0[:,i] = initial_elevation[i]*hillx*islope
    # generate hillslope for flat plateau which have elevation of the more downstream river node
    elif icflag == 2:
        for i in range(0, max(z0.shape)):
            if 'drop1' in drop:
                hill0[:,i] = drop['drop1']['initial_level']
            else:
                hill0[:,i] = initial_elevation[0]    
    # generate hillslope for steady-state river profile based on the initial uplift (Ui) and erodibility (Ki)
    elif icflag == 3:
        z_ss = np.zeros(max(hillx.shape))
        if hm == 0 and hn == 1:
            z_ss = 0-Ui*1e-3*hillx**2/(2*hk)
            z_ss = z_ss - z_ss[-1]
        elif hm == 2 and hn == 2:
            z_ss = 0-2*np.sqrt(Ui*1e-3/hk)*hillx**0.5
            z_ss = z_ss -z_ss(-1)
        for i in range(0, max(z0.shape)):
            hill0[:, i] = z_ss + z0[i]
    
    # initialize river profile pre-capture
    for i in range(len(capture)):        
        # find upstream and downstream river nodes
        capture['river' + str(i+1)]['upstream_index'] = Find_Upstream_Index(capture['river' + str(i+1)]['node'], pairs)
        capture['river' + str(i+1)]['downstream_index'] = Find_Downstream_Index(capture['river' + str(i+1)]['node'], pairs)
        # find accumulation area of captured river
        capture['river' + str(i+1)]['capture_area'] = area[capture['river' + str(i+1)]['node']]
        # reduce area of connected trunk stream
        area[capture['river' + str(i+1)]['downstream_index']] = area[capture['river' + str(i+1)]['downstream_index']] - capture['river' + str(i+1)]['capture_area']
    
    # disconnect river network at capture node
    for i in range(len(capture)):
        pairs[capture['river' + str(i+1)]['node']-1, 1] = pairs[capture['river' + str(i+1)]['node']-1, 0]
    
    # initialize capture not done
    for i in range(len(capture)):
        capture['river' + str(i+1)]['capture_done'] = False

    # define a constant (kflag = 0) or spatial variable erodibility (kflag = 1)
    if kflag == 0:
        K = np.ones(np.shape(z0))*K
    elif kflag == 1:
        K = data['erodibility']
    
    # define an uniform uplift (nuv=0) or variable spatial and temporal uplift (nuv=1)
    U_initial = np.zeros((1, max(z0.shape)))
    if nuv == 0:
        U_initial[0,:] = U
    elif nuv == 1:
        U_initial = data['uplift']*1e-3
        
    U_final = U_initial[:,0]
    U_initial[:,0] = 0
        
    # number of steps per time interval (define by the change in uplift through time)
    if len(ua) == 0:
        TI = np.array([tt*1e6, 0])
    else:
        TI = np.flip(np.sort(ua))*1e6
        TI = np.insert(TI, [0, TI.size], [tt*1e6, 0])
        
    nsteps = np.round(-np.diff(TI)/dt)
    steps_cosmo = t_record/dt
    
    # initialize hillslope variables
    hillz = np.copy(hill0)
    
    # change 26.07.13
    # maxi = max(hillx.shape)
    maxi = hillx.shape[0]
    
    # change 26.07.13
    # hillx = np.transpose(np.tile(hillx, (max(z0.shape), 1))) # change 26.07.13
    
    qs = np.zeros(np.shape(hillx))
    xm = np.zeros(np.shape(hillx))
    k1xm = np.zeros(np.shape(hillx))
    k1 = -1*hk/(hdx**hn)
    k3 = dt/hdx
    
    for i in range(maxi):
        xm[i,:] = (i*hdx)**hm
    xm[0,:] = xm[0,:]+(0.2*hdx)
    k1xmm = k1*xm[maxi-1]
    k1xm = k1*xm
    # crit_elevation = np.tan(crit_slope)*hl/hdx
    crit_elevation = np.tan(crit_slope)*hdx
    
    # initialize outlet node elevation
    z1 = np.zeros(np.shape(z0), dtype=np.float64)
    if 'drop1' in drop:
        z1[0] = drop['drop1']['initial_level']
    else:
        z1[0] = initial_elevation[0]
    
    # initialize variables
    z00 = z0
    steps = 0
    o = 0
    s = 0
    p = 0
    
    # initialize model record
    erosion_hillslope = np.zeros((int(t_record/dt), np.shape(hillz)[0], np.shape(hillz)[1]))
    erosion_river = np.zeros((int(np.sum(nsteps)), max(z0.shape)))
    time_record = np.round(np.arange(start_dtr, end_dtr+dtr, dtr)*1e6)
    elevation_record = [np.zeros(len(z1)) for k in range(len(time_record))]
    
    # initialize thermo variables
    if (ahea_calc == True or afta_calc == True or aftmtl_calc == True):
        topo = np.zeros((Length(thermo_node), int(np.sum(nsteps))))
        exhumation = np.zeros((Length(thermo_node), int(np.sum(nsteps))))
    
    # number of uplift step
    us = len(ua) + 1
    
    # start time loops
    for l in range(0, us):
    
        for t in range(0, int(nsteps[l])):
            
            steps += 1
            s += 1
            time = dt*s
            
            U = np.copy(U_initial)
            U_hill = np.tile(U[l,:], (maxi, 1))
            U_hill[:, 0] = U_final[l]  
            
            # update uplift for river capture event
            for i in range(len(capture)):
                if time < (tt - capture['river' + str(i+1)]['time'])*1e6:
                    U_hill[:, capture['river' + str(i+1)]['upstream_index']] = capture['river' + str(i+1)]['initial_uplift']*1e-3
                    U[l, capture['river' + str(i+1)]['upstream_index']] = capture['river' + str(i+1)]['initial_uplift']*1e-3

            # update outlet elevation for base-level drop
            for i in range(len(drop)):
                if time > (tt - drop['drop'+str(i+1)]['time'])*1e6:                   
                    if i == len(drop) - 1:
                        z00[0] = initial_elevation[0]
                    else:
                        z00[0] = drop['drop'+str(i+2)]['initial_level']
                     
            # calculate the slope of stream and new river elevation
            if pixel == 0:
                S = np.diff(np.transpose(z00[pairs.astype(int)-1]), axis=0)/(dx_dem*skipping_factor)
            else:
                elevation_diff = np.diff(np.transpose(z00[pairs.astype(int)-1]), axis=0)
                distance_diff = np.diff(np.transpose(x[pairs.astype(int)-1]), axis=0)
                # remove low distance to avoid error in elevation calculation
                distance_diff[distance_diff <= 150] = 150
                S = elevation_diff/distance_diff
            S = np.insert(S, 0, 0)
            
            if pixel == 0:
                z1 = z00 + (U[l,:] - K*(area*dx_dem**2)**m*np.sign(S)*abs(S)**n)*dt
            else:
                z1 = z00 + (U[l,:] - K*area**m*np.sign(S)*abs(S)**n)*dt
 
            # calculate river segment alone or initiate the capture 
            for i in range(len(capture)):
                if capture['river' + str(i+1)]['capture_done'] == False:
                    if time < (tt - capture['river' + str(i+1)]['time'])*1e6:
                        z1[capture['river' + str(i+1)]['node']] = z1[capture['river' + str(i+1)]['node'] - 1]
                    else:
                        pairs[capture['river' + str(i+1)]['node'] - 1, 1] = pairs[capture['river' + str(i+1)]['node'], 0]
                        area[capture['river' + str(i+1)]['downstream_index']] = area[capture['river' + str(i+1)]['downstream_index']] + capture['river' + str(i+1)]['capture_area']
                        capture['river' + str(i+1)]['capture_done'] = True
            
            # calculate qs of hillslope
            qs[0,:] = k1*(hill0[1,:] - hill0[0,:])**hn
            qs[1:-1,:] = k1xm[1:-1,:]/2*(hill0[2:,:] - hill0[0:-2,:])**hn
            qs[-1,:] = k1xmm*(hill0[-1,:] - hill0[-2,:])**hn
            
            # calculate new hillslope elevation
            hillz[0,:] = hill0[0,:] - k3*(qs[0,:])
            hillz[1:-2,:] = hill0[1:-2,:] - (k3/2)*(qs[2:-1,:] - qs[0:-3,:])
            hillz[-2,:] = hill0[-2,:] - k3*(qs[-2,:] - qs[-3,:])
            hillz = hillz + U_hill*dt
            hillz[-1,:] = z1
            
            ## change 26.07.13 (need to fix for too steep hillslope)
            # S_hillz = -np.diff(hillz)
            # hillz[[S_hillz>crit_elevation, np.zeros((1, np.shape(hillz)[1]))] == 1] = hillz[[np.zeros((1, np.shape(hillz)[1])), S_hillz>crit_elevation] == 1]+crit_elevation
            # S_hillz = -np.diff(hillz, axis=0)
            # mask = S_hillz > crit_elevation[np.newaxis, :]
            # rows, cols = np.where(mask)
            # hillz[rows, cols] = hillz[rows + 1, cols] + crit_elevation[cols]
            
            
            # record elevation and exhumation at thermo sample nodes
            if (ahea_calc == True or afta_calc == True or aftmtl_calc == True):
                topo[:,s-1] = z1[thermo_node-1]
                exhumation[:,s-1] = U[l,thermo_node-1] - (z1[thermo_node-1] - z00[thermo_node-1])/dt
            
            # record fluvial erosion
            erosion_river[steps-1,:] = (U[l,:] + (z00 - z1)/dt)
            
            # record hillslope erosion for crnc calculation
            if sum(nsteps)-steps < steps_cosmo:
                # print(steps)
                o += 1                
                
                # do not work (don't know why)
                # erosion_hillslope[o-1,:,:] = (U_hill + (hill0 - hillz)/dt)
                
                erosion_hillslope[o-1,0,:] = qs[0,:]/hdx
                erosion_hillslope[o-1,1:-2,:] = (qs[2:-1,:] - qs[:-3,:])/(2*hdx)
                erosion_hillslope[o-1,-2,:] = (qs[-2,:] - qs[-3,:])/hdx
                erosion_hillslope[o-1,-1,:] = erosion_river[steps-1,:]
                
            z00 = z1
            hill0 = hillz
            
            if time in time_record:
                elevation_record[p] = z1
                p = p + 1
    
    erosion_river = erosion_river*1e3
    erosion_hillslope = erosion_hillslope*1e3            
    elevation_river = z1
    elevation_hillslope = hillz
    
    modelled_data.elevation_river = elevation_river
    modelled_data.erosion_river = erosion_river 
    modelled_data.elevation_hillslope = elevation_hillslope
    modelled_data.erosion_hillslope = erosion_hillslope
    
    # count the number of non-nan values
    n = np.count_nonzero(~np.isnan(elevation_river))
    # calculate the topographic misfit
    misfit_topo = np.nansum(np.log(2*np.pi)/2 + np.log(16) + 0.5*((z1 - initial_elevation)/16)**2)/n
    
    end_time = datetime.datetime.now()
    
    if not inverse:
        print('Log-likelihood topo misfit: ' + str(misfit_topo))
        print('Topographic calculation: {}'.format(end_time-start_time))
        print('-------------------')
    
    ### ======================= 3:COSMOGENIC MODEL ======================== ###
    
    start_time = datetime.datetime.now()
    
    if crn_calc == True:
        cosmo_meas = data['cosmo_meas']
        cosmo_meas_ind = cosmo_meas['ind']
        cosmo_thickness = param.cosmo_thickness
        cosmo_topocorr = param.cosmo_topocorr
        cosmo_aa = param.cosmo_aa
        cosmo_lat_river = np.squeeze(data['latitude'])
        cosmo_long_river = np.squeeze(data['longitude'])
        cosmo_meas_lat = cosmo_meas['latitude']
        cosmo_meas_type = cosmo_meas['type']
        cosmo_meas_tcn = np.squeeze(cosmo_meas['tcn'])
        cosmo_meas_tcn_error = np.squeeze(cosmo_meas['tcn_error'])
        
        nc = Length(cosmo_meas_tcn)
        node = np.unique(np.concatenate(cosmo_meas_ind))
        # TCN_hillslope = np.zeros((max(node), int(hl/hdx + 1)))
        TCN_hillslope = np.zeros((max(node), hdn))
        param.cosmo_pressure = 1
        
        nuclide = 10
        mc = Mc()
        c3 = C3()
        
        for o in range(0, max(node.shape)):
            delattr(param, 'cosmo_pressure')
            elevation = elevation_hillslope[:, node[o]]
            erosion = np.squeeze(erosion_hillslope[:, :, node[o]])
            param.cosmo_lat = np.squeeze(np.tile(cosmo_lat_river[node[o]], (1, max(elevation.shape))))
            param.cosmo_long = np.squeeze(np.tile(cosmo_long_river[node[o]], (1, max(elevation.shape))))
            
            if not hasattr(param, 'cosmo_pressure'):
                if cosmo_aa == 'std':
                    param.cosmo_pressure = NCEPatm_2(np.transpose(param.cosmo_lat), np.transpose(param.cosmo_long), elevation)
                elif cosmo_aa == 'ant':
                    param.cosmo_pressure = Antatm(elevation)
                    
            # check if cosmo_presure is correctly set
            if np.size(param.cosmo_pressure) == 0:
                sys.exit('sample pressure extant but empty on cosmo')
            elif param.cosmo_pressure.all() == 0:
                sys.exit('sample pressure equal to zero on sample')
                
            # convert cosmo thickness to g/cm2 and get thickness SF
            param.cosmo_thickgcm2 = cosmo_thickness*rho_c/1e3
            if cosmo_thickness > 0:
                param.cosmo_thickSF = Thickness(cosmo_thickness, int(al_be_consts_v22['Lsp']), rho_c/1e3)
            else:
                param.cosmo_thickSF = 1
                
            # negative longitude catch
            if param.cosmo_long.all() < 0:
                param.cosmo_long = param.cosmo_long + 360
            
            ### ============== 3.1:NUCLIDE-SPECIFIC ASSIGNMENTS =============== ###
            
            # nuclide specific assignments
            if nuclide == 10:
                mc.Natoms = float(al_be_consts_v22['Natoms10'])
                mc.sigma190 = float(al_be_consts_v22['sigma190_10'])
                mc.k_neg = float(al_be_consts_v22['k_neg10'])
                mc.delsigma190 = float(al_be_consts_v22['delsigma190_10'])
                mc.delk = float(al_be_consts_v22['delk_neg10'])
                l = float(al_be_consts_v22['l10'])
                L = float(al_be_consts_v22['Lsp'])
                P_ref_St, delP_ref_St = float(al_be_consts_v22['P10_ref_St']), float(al_be_consts_v22['delP10_ref_St'])
                P_ref_Du, delP_ref_Du = float(al_be_consts_v22['P10_ref_Du']), float(al_be_consts_v22['delP10_ref_Du'])
                P_ref_De, delP_ref_De = float(al_be_consts_v22['P10_ref_De']), float(al_be_consts_v22['delP10_ref_De'])
                P_ref_Li, delP_ref_Li = float(al_be_consts_v22['P10_ref_Li']), float(al_be_consts_v22['delP10_ref_Li'])
                P_ref_Lm, delP_ref_Lm = float(al_be_consts_v22['P10_ref_Lm']), float(al_be_consts_v22['delP10_ref_Lm'])
                nstring = 'BE-10'
            
            elif nuclide == 26:
                mc.Natoms = float(al_be_consts_v22['Natoms26'])
                mc.sigma190 = float(al_be_consts_v22['sigma190_26'])
                mc.k_neg = float(al_be_consts_v22['k_neg26'])
                mc.delsigma190 = float(al_be_consts_v22['delsigma190_26'])
                mc.delk = float(al_be_consts_v22['delk_neg26'])
                l = float(al_be_consts_v22['l26'])
                L = float(al_be_consts_v22['Lsp'])
                P_ref_St, delP_ref_St = float(al_be_consts_v22['P26_ref_St']), float(al_be_consts_v22['delP26_ref_St'])
                P_ref_Du, delP_ref_Du = float(al_be_consts_v22['P26_ref_Du']), float(al_be_consts_v22['delP26_ref_Du'])
                P_ref_De, delP_ref_De = float(al_be_consts_v22['P26_ref_De']), float(al_be_consts_v22['delP26_ref_De'])
                P_ref_Li, delP_ref_Li = float(al_be_consts_v22['P26_ref_Li']), float(al_be_consts_v22['delP26_ref_Li'])
                P_ref_Lm, delP_ref_Lm = float(al_be_consts_v22['P26_ref_Lm']), float(al_be_consts_v22['delP26_ref_Lm'])
                nstring = 'Al-26'
            
            ### ================ 3.2: GET THE EROSION RATE ==================== ###
            
            tv = np.append(np.append(np.append(np.arange(0, 7000, 500), 6900), np.arange(7500, 12500, 1000)), np.arange(12000, 801000, 1000))
            tv = np.append(tv, np.logspace(np.log10(810000), 7, 200))
            
            N = np.zeros(np.shape(erosion)[1])
            for i in range(0, np.shape(erosion)[1]):
                if muon == 1:
                    P_St = Stone2000(param.cosmo_lat[i], param.cosmo_pressure[i], 0.9938)*P_ref_St*param.cosmo_topocorr
                    P_mu = Braucher2013(param.cosmo_pressure[i], 0.9938)
                    c3.P_mu = P_mu
                else:
                    P_St = Stone2000(param.cosmo_lat[i], param.cosmo_pressure[i], 1)*P_ref_St*cosmo_topocorr
                    z_mu = np.append(np.array([0]), np.logspace(0, 5.3, 100)) + (param.cosmo_thickgcm2/2)
                    P_mu_z = np.zeros(np.shape(z_mu))
                    P_mu_z = P_mu_total(z_mu, param.cosmo_pressure[i], mc, False)
                    c3.z_mu = z_mu - (param.cosmo_thickgcm2/2)
                    c3.P_mu_z = P_mu_z
                
                c3.tv = tv
                c3.l = l
                c3.tsf = param.cosmo_thickSF
                c3.L = L
                c3.P_sp_t = P_St
                c3.P_mu = P_mu
                c3.L_muon = 4656                # attetuation length scale of muons in g/cm-2 according to Braucher et al., 2013
                
                ### ==== 3.3:FORWARD MODELLING OF SURFACE CN CONCENTRATION ==== ###
                
                E = erosion[:,i]
                E1 = E/10*rho_c/1e3
                if E1[0] == 0:
                    E1[0] == 1e-4
                
                # calculate steady state CN profile
                n_profile, depth_array = Calc_Steady_Profile(c3, E1[0], muon)
                
                # recalculate CN profile depending the difference in erosion
                # I = np.argwhere(abs(E1[:]) - abs(E1[0]) > 1e-10)                      # fixed erosion value
                I = np.argwhere(np.abs(E1[:] - E1[0]) > np.abs(E1[0])/1e3)              # percentage erosion
                
                if I.size > 0:
                    # print('Maximum erosion difference: ' + str(np.max(np.unique(np.abs(E1 - E1[0])))))
                    
                    I = I[0, 0]  # proper scalar extraction
                
                    if 0 <= I < E1.shape[0]:
                        depth_vector = np.cumsum(np.flip(erosion[I:, i], axis=0) * dt * rho_c / 1e4)
                        depth_vector = np.insert(depth_vector[:-1], 0, 0)
                
                        # interpolate depth_vector to dt = 100 years
                        xi = dt / 100
                        f = interp1d(np.arange(xi, depth_vector.shape[0]*xi + xi, xi), depth_vector)
                        depth_vector = f(np.arange(xi, depth_vector.shape[0]*xi + 1, 1))
                
                        # find CN of node that ends up at the surface
                        f = interp1d(depth_array, n_profile, fill_value='extrapolate')
                        N[i] = f(depth_vector[-1])
                
                        if depth_vector[-1] > depth_array[-1]:
                            N[i] = min(n_profile)
                
                        if muon == 1:
                            P_sp_z_target = c3.P_sp_t * np.exp(-depth_vector / c3.L)
                            P_m_z_target = c3.P_mu * np.exp(-depth_vector / c3.L_muon)
                            P_total = P_sp_z_target + P_m_z_target
                
                            for j in range(depth_vector.shape[0]):
                                N[i] = (N[i] + P_total[-1 - j] * dt / xi) * np.exp(-c3.l * dt / xi)
                
                        else:
                            f = interp1d(c3.z_mu, c3.P_mu_z)
                            P_mu_z_target = f(depth_vector)
                            P_sp_z_target = c3.P_sp_t * np.exp(-depth_vector / c3.L)
                
                            for j in range(1, depth_vector.shape[0]):
                                N[i] = (N[i] + P_mu_z_target[-1-j]*dt/10 + P_sp_z_target[-1-j]*dt/10)*np.exp(-c3.l*dt/10)
                
                else:
                    N[i] = n_profile[0]                
            
            # TCN_hillslope[node[o]-1,:] = N
            TCN_hillslope[node[o]-1,:] = N
        
        # calculate the mean tcn concenttration - for catchments the in-situ TCN are multiplied with the local erosion rate and the sum of all the products is divided by the sum of all local erosion rate
        tcn_mod = np.zeros(max(cosmo_meas_lat.shape))
        for i in range(0, max(cosmo_meas_lat.shape)):
            if cosmo_meas_type[i] == 1:
                tcn_mod[i] = sum(sum(TCN_hillslope[cosmo_meas_ind[i][cosmo_meas_ind[i][:]>=0]-1,:]*np.squeeze(erosion_hillslope[1999,:,cosmo_meas_ind[i][cosmo_meas_ind[i][:]>=0]-1])))/sum(sum(erosion_hillslope[1999,:,cosmo_meas_ind[i][cosmo_meas_ind[i][:]>=0]-1]))
            
            if cosmo_meas_type[i] == 2:
                tcn_mod[i] = TCN_hillslope[cosmo_meas_ind[i,0],0]
            
            if not inverse:
                print('TCN sample ' + str(i) + ' = ' + str(tcn_mod[i]))
        
        # calculate misfit for the cosmogenic information
        if 'tcn_error' in cosmo_meas:
            modelled_data.tcn = tcn_mod
            misfit_tcn = sum(np.log(2*np.pi/2) + np.log(cosmo_meas_tcn_error) + 0.5*((tcn_mod - cosmo_meas_tcn)/cosmo_meas_tcn_error)**2)/nc
            if not inverse:
                print('Log-likelihood cosmo misfit: ' + str(misfit_tcn))
        else:
            misfit_tcn = 0
    
        end_time = datetime.datetime.now()
        if not inverse:
            print('Cosmogenic calculation: {}'.format(end_time-start_time))
            print('-------------------')
    
    else:
        tcn_mod = np.nan
        misfit_tcn = 0
    
    ### =============== 4:THERMOCHRONOLOGICAL AGES MODEL ================== ###
    
    if (ahea_calc == True or afta_calc == True or aftmtl_calc == True):
        start_time = datetime.datetime.now()
        
        thermo_meas = data['thermo_meas']
        if np.size(thermo_meas['latitude']) == 1:
            thermo_lat = thermo_meas['latitude']
            thermo_elevation = thermo_meas['elevation']
        else:
            thermo_lat = np.squeeze(thermo_meas['latitude'])
            thermo_elevation = np.squeeze(thermo_meas['elevation'])
        
        nt = Length(thermo_lat)                                                                 # number of sample observations
        u = exhumation/(365.25*24*3600)                                                         # transform uplift in m/s
        steps = np.shape(u)[1]                                                                  # lenghth of input arrays                                            
        
        Ts = T0 - thermo_elevation*lr/1000                                                      # surface temperature in C
        h = 80000                                                                               # depth of basal boundary condition in m
        hc = h + thermo_elevation                                                               # crustal thickness in m
        nr_nodes = 80                                                                           # crustal nodes
        dx = hc/nr_nodes                                                                        # spatial resolution in m                                                   
        Tbase = h/1000*gg                                                                       # temperature of basal boundary condition in C
        hp_exp = 10000                                                                          # depth at which hp is 1/e
        zsteps = nr_nodes + 1                                                                   # crustal steps
        SC = 0.45                                                                               # CFL stability condition (<0.5)
        dt_thermo = SC*min(dx)**2/TD                                                            # determine new stable dt 
        
        time = np.linspace(tt, 0, steps+1)                                                      # time vector in Myrs
        time = time*(1e6*365.25*24*3600)                                                        # time vector in sec
        time = time[1:]   
        
        time_new = np.arange(np.floor(time[0]/dt_thermo)*dt_thermo, -dt_thermo, -dt_thermo)     # new time vector according to stable dt in sec
        tsteps = int(max(time_new.shape))                                                       # new time steps
        
        # resample topo and u for stable dt
        u_new = np.zeros((nt, tsteps))
        topo_new = np.zeros((nt, tsteps))
        for i in range(0, nt):
            f = interp1d(time, u[i,:], fill_value='extrapolate')
            u_new[i,:] = f(time_new)
            f = interp1d(time, topo[i,:], fill_value='extrapolate')
            topo_new[i,:] = f(time_new)
        
        # initialize temperature and heat production
        T = np.zeros((tsteps, nt, zsteps))
        hp = np.zeros((1, nt, zsteps))
        for i in range(0, nt):
            T[:,i,0] = Ts[i]
            T[:,i,-1] = Tbase
        for i in range(0, zsteps):
            hp[0,:,i] = hpc*np.exp(-(i*dx[:])/hp_exp)/(rho_c*cp_c)
            
        # calculate stable geotherm
        change = 1
        t = 1
        j = np.arange(1, zsteps-1)
        while change > 0.1 :
            t += 1
            # variable in the lithosphere
            T[t-1,:,j] = T[t-2,:,j] + SC*(T[t-2,:,j+1] - 2*T[t-2,:,j] + T[t-2,:,j-1]) + hp[0,:,j]*dt_thermo
            change = sum(abs(T[t-2,0,:] - T[t-1,0,:]))
        
        # time-varaible exhumation/burial history
        Tvar = np.zeros((tsteps, nt, zsteps))
        Tvar[0,:,:] = T[t-1,:,:]
        Tvar[:,:,0] = T[:,:,0]
        Tvar[:,:,-1] = T[:,:,-1]
        
        # calculate total exhumation/burial and original location of sample
        u_total = np.sum(np.transpose(u_new), axis=0)*dt_thermo
        u_total = np.where(u_total < 0, 0, u_total)
        location = u_total
        if tsteps > 1e2:
            step = np.floor(tsteps/1e2)
        else:
            step = 1
        tT = np.zeros((nt, int(np.ceil(tsteps/step))))
        z = np.zeros((nt, zsteps))
        for i in range(0, nt):
            z[i,:] = np.arange(0, hc[i]+dx[i], dx[i])
            f = interp1d(z[i,:], np.transpose(np.squeeze(Tvar[0,i,:])), fill_value='extrapolate')
            tT[i,0] = f(location[i])
            
        # model temperature evolution with time and record tTpath of sample
        j = np.round(j)
        u_matrix = np.zeros((tsteps, nt, zsteps))
        dx_matrix = np.zeros((tsteps, nt, zsteps))
        for o in range(0, zsteps):
            u_matrix[:,:,o] = np.transpose(u_new)
            for i in range(0, nt):
                dx_matrix[:,i,o] = dx[i]
        
        i = 0
        for t in range(1, tsteps):
            Tvar[t,:,j] = Tvar[t-1,:,j] + SC*(Tvar[t-1,:,j+1] - 2*Tvar[t-1,:,j] + Tvar[t-1,:,j-1]) + hp[0,:,j]*dt_thermo + (Tvar[t-1,:,j+1] - Tvar[t-1,:,j])*(u_matrix[t, :,j]*dt_thermo/dx_matrix[t,:,j])
            # find temperature at sample location
            location = location - (np.transpose(u_new[:,t])*dt_thermo)
            if t % step == 0:
                i += 1
                for o in range(0, nt):
                    f = interp1d(z[o,:], np.transpose(np.squeeze(Tvar[t,o,:])), fill_value='extrapolate')
                    tT[o,i] = f(location[o])
        
        # simplify t-T path
        time_i = time_new[0::int(step)]/(1e6*365.25*24*3600)
        if max(time_i.shape) < max(tT[0,:].shape):
            time_i = np.insert(time_i, time_i.size, 0)
        
        time_i = time_i[np.logical_not(np.isnan(sum(tT)))]
        tT = tT[:,np.logical_not(np.isnan(sum(tT)))]
        
        thermo = [Thermo() for i in range(nt)]
        for i in range(0, nt):
            f = np.vstack((tT[i,:], time_i)).T
            line = LineString(f)
            ps = line.simplify(0.1, preserve_topology=False)
            ps = np.array(ps.coords)
            thermo[i].temp = ps[:,0]
            thermo[i].time = ps[:,1]
        
        modelled_data.time = thermo[0].time
        modelled_data.temperature = thermo[0].temp
        
        # calculate thermochronological ages
        thermo_data = Thermo_Data()
        thermo_data.zfta = np.zeros(nt)
        thermo_data.afta = np.zeros(nt)
        thermo_data.aftmtl = np.zeros(nt)
        thermo_data.aftmtl_pdf = np.zeros((nt,200))
        thermo_data.ahea = np.zeros(nt)
        
        # calculate low-temperature isotopic system data for each sample
        for i in range(0, nt):
            #print('---------- thermo sample ' + str(i) + ' ----------')
            thermo[i].time[-1] = 0
            thermo[i].time = np.flip(thermo[i].time)
            thermo[i].temp = np.flip(thermo[i].temp)
            
            # calculate ZFT age (Tagami et al., 1998)
            # modelled_data.zfta[i], ftld, ftldmean, ftldsd = Mad_Zirc(thermo[i].time, thermo[i].temp, 0, 1)
            
            # reduce time-temperature path to ~20 nodes from 180°C to surface temperature in preparation for AFT data calculation
            if (afta_calc == True or aftmtl_calc == True):
                nn = np.where(thermo[i].temp > 180)
                dd = 1
                if not nn[0].any():
                    nn = max(thermo[i].temp.shape)
                else:
                    nn = nn[0][0]
                if nn > 20:
                    dd = int(np.round(nn*1.5/20))
                
                # calculate AFT age and track length distribution (Ketcham et al., 2007)
                if thermo_meas['afta'][i] == -1:
                    thermo_data.afta[i], thermo_data.aftmtl[i], thermo_data.aftmtl_pdf[i,:] = np.nan, np.nan, np.nan
                else:
                    thermo_data.afta[i], thermo_data.aftmtl[i], thermo_data.aftmtl_pdf[i,:] = Fd_Dpar_L0(thermo[i].time[0:nn][0::dd], thermo[i].temp[0:nn][0::dd])
            
            # remove time-temperature nodes above 200°C in preparation for AHe data calculation
            if ahea_calc == True:
                ind = np.where(thermo[i].temp < 200)
                thermo[i].time = np.delete(thermo[i].time[ind], [-1])
                thermo[i].temp = np.delete(thermo[i].temp[ind], [-1])
                
                # calculate AHe grain age (Flowers et al., 2009, Glotzbach and Ehlers, 2024) (RDAMM function will crash if some temperature points are too hot)
                if thermo_meas['ahea'][i] == -1:
                    thermo_data.ahea[i] = np.nan
                else:
                    if not 'model' in thermo_meas:
                        thermo_data.ahea[i] = RDAAM_Calculation(thermo[i].time, thermo[i].temp)
                    else:
                        if thermo_meas['model'][i] == 1:
                            thermo_data.ahea[i] = RDAAM_Calculation(thermo[i].time, thermo[i].temp)
                        elif thermo_meas['model'][i] == 2:
                            _, _, _, age_transect = HeliumGrainAge(thermo[i].time, thermo[i].temp, radius=thermo_meas['grain_radius'][i], U=thermo_meas['U'][i], Th=thermo_meas['Th'][i], Sm=thermo_meas['Sm'][i])
                            age_length = len(age_transect)
                            thermo_data.ahea[i] = age_transect[int(age_length*thermo_meas['pit_location'][i])]
                        else:
                            thermo_data.ahea[i] = np.nan
            
            if not inverse:
                print('AHe age ' + str(i) + ' = ' + str(thermo_data.ahea[i]))
                print('AFT age ' + str(i) + ' = ' + str(thermo_data.afta[i]))
                print('AFT mtl ' + str(i) + ' = ' + str(thermo_data.aftmtl[i]))
                # print('ZFT age ' + str(i) + ' = ' + str(modelled_data.zfta[i]))
        
        # calculate the misfit for the thermochronological data
        if ahea_calc == True:
            if 'ahea' in thermo_meas:
                
                ind = np.where(thermo_meas['ahea'] != -1)
                modelled_data.ahea = thermo_data.ahea[ind]
                
                thermo_ahea = np.squeeze(thermo_meas['ahea'])
                thermo_ahea_error = np.squeeze(thermo_meas['ahea_error'])
                misfit_ahea = sum(np.log(2*np.pi/2) + np.log(thermo_ahea_error[ind]) + 0.5*((thermo_ahea[ind] - thermo_data.ahea[ind])/thermo_ahea_error[ind])**2)/len(thermo_data.ahea[ind]);
                
                if not inverse:
                    print('Log-likelihood ahea misfit: ' + str(misfit_ahea))
        else:
            misfit_ahea = 0
        
        if afta_calc == True:
            if 'afta' in thermo_meas:
                
                ind = np.where(thermo_meas['afta'] != -1)
                modelled_data.afta = thermo_data.afta[ind]
                
                thermo_afta = np.squeeze(thermo_meas['afta'])
                thermo_afta_error = np.squeeze(thermo_meas['afta_error'])
                misfit_afta = sum(np.log(2*np.pi/2) + np.log(thermo_afta_error[ind]) + 0.5*((thermo_afta[ind] - thermo_data.afta[ind])/thermo_afta_error[ind])**2)/len(thermo_data.afta[ind])
                
                if not inverse:
                    print('Log-likelihood afta misfit: ' + str(misfit_afta))
            
            thermo_data.aftmtl_pdf[thermo_data.aftmtl_pdf < 1e-5] = 1e-5
        else:
            misfit_afta = 0
        
        if aftmtl_calc == True:
            if 'aftmtl' in thermo_meas:
                
                ind = np.where(thermo_meas['aftmtl'] != -1)
                modelled_data.aftmtl = thermo_data.aftmtl[ind]
                
                thermo_aftmtl = np.squeeze(thermo_meas['aftmtl'])
                thermo_aftmtl_sd = np.squeeze(thermo_meas['aftmtl_sd'])
                misfit_aftmtl = sum(np.log(2*np.pi/2) + np.log(thermo_aftmtl_sd[ind]) + 0.5*((thermo_aftmtl[ind] - thermo_data.aftmtl[ind])/thermo_aftmtl_sd[ind])**2)/len(thermo_data.aftmtl[ind])
                
                if not inverse:
                    print('Log-likelihood aftmtl misfit: ' + str(misfit_aftmtl))
        else:
            misfit_aftmtl = 0
            
        # if 'zhe' in thermo_meas:
        #     thermo_zhe = np.squeeze(thermo_meas['zhe'])
        #     thermo_zhe_error = np.squeeze(thermo_meas['zhe_error'])
        #     misfit_thermo = misfit_thermo + sum(np.log(2*np.pi/2) + np.log(thermo_zhe_error) + 0.5*((thermo_zhe - thermo_data.zhe)/thermo_zhe_error)**2)
        
        # if 'zft' in thermo_meas:
        #     modelled_data.zfta = thermo_data.zft
        #     thermo_zft = np.squeeze(thermo_meas['zft'])
        #     thermo_zft_error = np.squeeze(thermo_meas['zft_error'])
        #     misfit_thermo = misfit_thermo + sum(np.log(2*np.pi/2) + np.log(thermo_zft_error) + 0.5*((thermo_zft - thermo_data.zft)/thermo_zft_error)**2)
    
        end_time = datetime.datetime.now()
        
        if not inverse:
            print('Thermochronologic calculation: {}'.format(end_time-start_time))
            print('-------------------')
            
    else:
        misfit_ahea = 0
        misfit_afta = 0
        misfit_aftmtl = 0
    
    
    ### ================ 5:MISFIT RESUME OF THE MODEL ===================== ###    
    
    # store individual logL and global normalised misfit
    misfit = Misfit()
    misfit.logL_topo = misfit_topo;
    misfit.logL_tcn = misfit_tcn;
    misfit.logL_ahea = misfit_ahea;
    misfit.logL_afta = misfit_afta;
    misfit.logL_aftmtl = misfit_aftmtl;
    
    misfit.logL_tot = misfit_topo + misfit_tcn + misfit_ahea + misfit_afta + misfit_aftmtl
    
    modelled_data.topo_misfit = misfit_topo
    modelled_data.cosmo_misfit = misfit_tcn
    modelled_data.ahea_misfit = misfit_ahea
    modelled_data.afta_misfit = misfit_afta
    modelled_data.aftmtl_misfit = misfit_aftmtl
    modelled_data.global_misfit = (misfit_topo + misfit_tcn + misfit_ahea + misfit_aftmtl)/1
    
    # return result of the forward modelling
    if not inverse:
        print('Log-likelihood global misfit: ' + str(misfit.logL_tot))
    
    return modelled_data
    