import numpy as np
from parameters_class import Parameters
from geologic_functions import Lithology_to_Erodibility, Block_to_Uplift, Tilting_to_Uplift, Base_Level_Drop, River_Capture, Low_Temperature_Thermochronology, Cosmogenic_Nuclide, GetSpecificElevation

class Prior:
    def __init__(self, label, low, high):
        
        self.label = label
        self.low = low
        self.high = high

def CreateSimulation(sim_name, fp, basin_data):
    '''
    DESCRIPTION:
        Create and define the parameter and basin dataset for the different similation type
    ----------
    PARAMETERS:
    sim_name : string name
        name of the simulation
    fp : array of float
        free parameter of the prior
    basin_data : dictionnary
        information of the basin
    ----------
    RETURNS:
    param:
        parameters of the simulation
    '''
    
    # set model parameters and basin data
    if sim_name == 'test':
        param = Parameters(T0=15, lr=5, TD=1.5e-6, rho_c=2700, hpc=18.6e-10, cp_c=700,
                            tt=100, dt=1e3, dtr=0.2, start_dtr=10, end_dtr=15,
                            icflag=2, islope=0.01, Ui=0.05, Ki=1e-6, ee=16, pixel=1,
                            hl=100, hdx=10, hk=0.01, hm=0, hn=1, crit_slope=30,
                            muon=1, cosmo_thickness=0, cosmo_topocorr=1, cosmo_aa='std', dx_cosmo=1000, t_record=2e6)
        
        param.sample = np.array([fp[0], fp[1], 1.0, 0.5, 17.5])
    
    if sim_name == 'simple-case':
        param = Parameters(gg=17.5, T0=15, lr=5, TD=1.5e-6, rho_c=2700, hpc=18e-10, cp_c=700, 
                           tt=100, dt=1e3, dtr=0.2, start_dtr=23, end_dtr=25,
                           U=fp[0], K=fp[1], m=0.5, n=1, icflag=2, islope=0.01, Ui=0.05, Ki=1e-6, ee=16, pixel=1,
                           hl=100, hdx=10, hk=0.01, hm=0, hn=1, crit_slope=30,
                           muon=1, cosmo_thickness=0, cosmo_topocorr=1, cosmo_aa='std', dx_cosmo=1000, t_record=2e6)
        
    if sim_name == 'stream-power':
        param = Parameters(T0=15, lr=5, TD=1.5e-6, rho_c=2700, hpc=18.6e-10, cp_c=700,
                            tt=130, dt=1e3, dtr=0.2, start_dtr=23, end_dtr=25,
                            icflag=2, islope=0.01, Ui=0.05, Ki=1e-6, ee=16, pixel=1,
                            hl=100, hdx=10, hk=0.01, hm=0, hn=1, crit_slope=30,
                            muon=1, cosmo_thickness=0, cosmo_topocorr=1, cosmo_aa='std', dx_cosmo=1000, t_record=2e6)
        
        param.sample = np.array([fp[0], fp[1], fp[2], fp[3], 17.5])
    
    if sim_name == 'tilting-case':
        param = Parameters(T0=15, lr=5, TD=1.5e-6, rho_c=2700, hpc=18.6e-10, cp_c=700,
                            tt=140, dt=1e3, dtr=0.2, start_dtr=23, end_dtr=25,
                            icflag=2, islope=0.01, Ui=0.05, Ki=1e-6, kflag=0, ee=16, pixel=1,
                            hl=100, hdx=10, hk=0.01, hm=0, hn=1, crit_slope=30,
                            muon=1, cosmo_thickness=0, cosmo_topocorr=1, cosmo_aa='std', dx_cosmo=1000, t_record=2e6)
        
        param.sample = np.array([fp[0], fp[1], 1.0, 0.5, 17.5])
        
        Tilting_to_Uplift(basin_data, param, direction='degree', uplift=[fp[0]], gradient=[fp[2]], degree=[fp[3]], time=[])
    
    if sim_name == 'variable-uplift-1':
        param = Parameters(T0=15, lr=5, TD=1.5e-6, rho_c=2700, hpc=18.6e-10, cp_c=700,
                            tt=160, dt=1e3, dtr=0.2, start_dtr=23, end_dtr=25,
                            icflag=2, islope=0.01, Ui=0.05, Ki=1e-6, ee=16, pixel=1,
                            hl=100, hdx=10, hk=0.01, hm=0, hn=1, crit_slope=30,
                            muon=1, cosmo_thickness=0, cosmo_topocorr=1, cosmo_aa='std', dx_cosmo=1000, t_record=2e6)
        
        param.sample = np.array([fp[0], fp[6], 1.0, 0.5, 17.5])
        
        Tilting_to_Uplift(basin_data, param, direction='degree', uplift=[fp[0],fp[1],fp[2],fp[3],fp[4],fp[5]], gradient=[0,0,0,0,0,0], degree=[0,0,0,0,0,0], time=[120,80,40,20,10])
    
    if sim_name == 'variable-uplift-2':
        param = Parameters(T0=15, lr=5, TD=1.5e-6, rho_c=2700, hpc=18.6e-10, cp_c=700,
                            tt=120, dt=1e3, dtr=0.2, start_dtr=23, end_dtr=25,
                            icflag=2, islope=0.01, Ui=0.05, Ki=1e-6, ee=16, pixel=1,
                            hl=100, hdx=10, hk=0.01, hm=0, hn=1, crit_slope=30,
                            muon=1, cosmo_thickness=0, cosmo_topocorr=1, cosmo_aa='std', dx_cosmo=1000, t_record=2e6)
        
        param.sample = np.array([fp[0], fp[5], 1.0, 0.5, 17.5])
        
        Tilting_to_Uplift(basin_data, param, direction='degree', uplift=[fp[0],fp[1],fp[2],fp[3],fp[4]], gradient=[0,0,0,0,0], degree=[0,0,0,0,0], time=[80,40,20,10])
    
    if sim_name == 'variable-uplift+simple-lithology-1':
        param = Parameters(T0=15, lr=5, TD=1.5e-6, rho_c=2700, hpc=18.6e-10, cp_c=700,
                            tt=160, dt=1e3, dtr=0.2, start_dtr=23, end_dtr=25,
                            icflag=2, islope=0.01, Ui=0.05, Ki=1e-6, ee=16, pixel=1,
                            hl=100, hdx=10, hk=0.01, hm=0, hn=1, crit_slope=30,
                            muon=1, cosmo_thickness=0, cosmo_topocorr=1, cosmo_aa='std', dx_cosmo=1000, t_record=2e6)
        
        param.sample = np.array([fp[0], 1e-6, 1.0, 0.5, 17.5])
    
        Tilting_to_Uplift(basin_data, param, direction='degree', uplift=[fp[0],fp[1],fp[2],fp[3],fp[4],fp[5]], gradient=[0,0,0,0,0,0], degree=[0,0,0,0,0,0], time=[120,80,40,20,10], spatial=True, block_ind=[6,10], block_uplift=[0.025,0.035])
    
        Lithology_to_Erodibility(basin_data, param, carbonate=1.0e-06, claystone=1.0e-06, dolomite=1.0e-6, limestone=1.0e-06, mudstone=1.0e-06, quartzite=0.6e-6, plutonic=0.6e-6, metamorphic=0.6e-6, sand=1.0e-06, sandstone=1.0e-06, silt=1.0e-6, volcanic=1.0e-6)

    if sim_name == 'variable-uplift+simple-lithology-2':
        param = Parameters(T0=15, lr=5, TD=1.5e-6, rho_c=2700, hpc=18.6e-10, cp_c=700,
                            tt=120, dt=1e3, dtr=0.2, start_dtr=23, end_dtr=25,
                            icflag=2, islope=0.01, Ui=0.05, Ki=1e-6, ee=16, pixel=1,
                            hl=100, hdx=10, hk=0.01, hm=0, hn=1, crit_slope=30,
                            muon=1, cosmo_thickness=0, cosmo_topocorr=1, cosmo_aa='std', dx_cosmo=1000, t_record=2e6)
        
        param.sample = np.array([fp[0], 1e-6, 1.0, 0.5, 17.5])
    
        Tilting_to_Uplift(basin_data, param, direction='degree', uplift=[fp[0],fp[1],fp[2],fp[3],fp[4]], gradient=[0,0,0,0,0], degree=[0,0,0,0,0], time=[80,40,20,10], spatial=True, block_ind=[6,10], block_uplift=[0.025,0.035])
    
        Lithology_to_Erodibility(basin_data, param, carbonate=1.0e-06, claystone=1.0e-06, dolomite=1.0e-6, limestone=1.0e-06, mudstone=1.0e-06, quartzite=0.6e-6, plutonic=0.6e-6, metamorphic=0.6e-6, sand=1.0e-06, sandstone=1.0e-06, silt=1.0e-6, volcanic=1.0e-6)

    return param

def CreatePrior(sim_name):
    '''
    DESCRIPTION:
        Create free parameter priors of the different simulation
    ----------
    PARAMETERS:
    sim_name : string name
        name of the simulation
    ----------
    RETURNS:
    p:
        prior of the free parameter
    '''
    
    # define prior estimation of the free parameters
    if sim_name == 'test':     
        p = Prior(label=['Uplift', 'Erodibility'],
                  low=[0.0, 2e-7],
                  high=[0.2, 5e-6])
    
    if sim_name == 'simple-case':     
        p = Prior(label=['Uplift', 'Erodibility'],
                  low=[0.0, 2e-7],
                  high=[0.2, 5e-6])
    
    if sim_name == 'stream-power':       
        p = Prior(label=['Uplift', 'Erodibility', 'n', 'm'],
                  low=[0.0, 2e-7, 0.8, 0.4],
                  high=[0.2, 5e-6, 2, 1])
    
    if sim_name == 'tilting-case':        
        p = Prior(label=['Uplift', 'Erodibility', 'Gradient', 'Azimuth'],
                  low=[0.0, 2e-7, 0.0, 0.0],
                  high=[0.2, 5e-6, 1.0, 360.0])
        
    if sim_name == 'variable-uplift-1':       
        p = Prior(label=['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'K'],
                  low=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2e-7],
                  high=[0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 5e-6])
        
    if sim_name == 'variable-uplift-2':      
        p = Prior(label=['U1', 'U2', 'U3', 'U4', 'U5', 'K'],
                  low=[0.0, 0.0, 0.0, 0.0, 0.0, 2e-7],
                  high=[0.20, 0.20, 0.20, 0.20, 0.20, 5e-6])
    
    if sim_name == 'variable-uplift+simple-lithology-1':       
        p = Prior(label=['U1', 'U2', 'U3', 'U4', 'U5', 'U6'],
                  low=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  high=[0.20, 0.20, 0.20, 0.20, 0.20, 0.20])
        
    if sim_name == 'variable-uplift+simple-lithology-2':       
        p = Prior(label=['U1', 'U2', 'U3', 'U4', 'U5'],
                  low=[0.0, 0.0, 0.0, 0.0, 0.0],
                  high=[0.20, 0.20, 0.20, 0.20, 0.20])
        
    return p

def BuildModelledData(basin_data, data, rng, elevation='normal', tcn=True, ahea=True, afta=True, aftmtl=True):
    
    if elevation == 'normal':
        modelled = data.elevation + rng.normal(loc=0, scale=16/2)
    elif elevation == 'specific':
        modelled = GetSpecificElevation(basin_data, data.elevation + rng.normal(loc=0, scale=16/2))
    
    if tcn:
        if 'tcn' in basin_data['cosmo_meas']:
            modelled = np.concatenate((modelled, data.tcn + rng.normal(loc=0, scale=basin_data['cosmo_meas']['tcn_error']/2)))
    
    if ahea:
        if 'ahea' in basin_data['thermo_meas']:
            ind = np.where(basin_data['thermo_meas']['ahea'] != -1)
            modelled = np.concatenate((modelled, data.ahea + rng.normal(loc=0, scale=basin_data['thermo_meas']['ahea_error'][ind]/2)))
    
    if afta:
        if 'afta' in basin_data['thermo_meas']:
            ind = np.where(basin_data['thermo_meas']['afta'] != -1)
            modelled = np.concatenate((modelled, data.afta + rng.normal(loc=0, scale=basin_data['thermo_meas']['afta_error'][ind]/2)))

    if aftmtl:
        if 'aftmtl' in basin_data['thermo_meas']:
            ind = np.where(basin_data['thermo_meas']['aftmtl'] != -1)
            modelled = np.concatenate((modelled, data.aftmtl + rng.normal(loc=0, scale=basin_data['thermo_meas']['aftmtl_sd'][ind]/2)))
    
    return modelled
    
def BuildObservedData(basin_data, elevation='normal', tcn=True, ahea=True, afta=True, aftmtl=True):
    
    if elevation == 'normal':
        observed = basin_data['initial_elevation']
    elif elevation == 'specific':
        observed = GetSpecificElevation(basin_data, basin_data['initial_elevation'])
    
    if tcn:
        if 'tcn' in basin_data['cosmo_meas']:
            observed = np.concatenate((observed, basin_data['cosmo_meas']['tcn']))
            
    if ahea:
        if 'ahea' in basin_data['thermo_meas']:
            ind = np.where(basin_data['thermo_meas']['ahea'] != -1)
            observed = np.concatenate((observed, basin_data['thermo_meas']['ahea'][ind]))
            
    if afta:
        if 'afta' in basin_data['thermo_meas']:
            ind = np.where(basin_data['thermo_meas']['afta'] != -1)
            observed = np.concatenate((observed, basin_data['thermo_meas']['afta'][ind]))
    
    if aftmtl:
        if 'aftmtl' in basin_data['thermo_meas']:
            ind = np.where(basin_data['thermo_meas']['aftmtl'] != -1)
            observed = np.concatenate((observed, basin_data['thermo_meas']['aftmtl'][ind]))
    
    return observed