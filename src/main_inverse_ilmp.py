# import python packages
import datetime
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path
import torch
import multiprocessing as mp
import pickle
from tqdm import tqdm
import sys
import warnings
import signal
import time
from contextlib import contextmanager
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

# import sbi packages
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import NPE

# import ilmp functions
from parameters_class import Parameters
from corr_sample_size import corr_sample_size
from ilm_forward_na import Ilm_Forward_Na
from matlab_extract import Matlab_Extract
from lsd_to_ilm import Lsd_to_Ilm
from catchment_dictionnary_functions import DictionnaryImport
from geologic_functions import Lithology_to_Erodibility, Block_to_Uplift, Tilting_to_Uplift, Base_Level_Drop, River_Capture, GetSpecificElevation, Low_Temperature_Thermochronology, Low_Temperature_Thermochronology2, Low_Temperature_Thermochronology3, Cosmogenic_Nuclide, Cosmogenic_Nuclide2, Cosmogenic_Nuclide3
from plotting_result_functions import Misfit_2D_Plot, SBI_Probability_Multi_Dimension_Plot, SBI_Misfit_MultiRound_Simulation_Plot
from general_functions import Find_Upstream_Index, Find_Downstream_Index
from inverse_modelling_functions import CreateSimulation, CreatePrior, BuildModelledData, BuildObservedData

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())
# restore the rcparams from matplotlib's internal default style
plt.rcdefaults()
# do not display warning message
warnings.simplefilter("ignore")

#%% ================= PERFORM PARALLELIZED INVERSE MODELLING ===============%%#

def Prepare_Simulation_Params(basin_data, sim_name, theta, crn, ahea, afta, aftmtl):
    # Return the parameters as a tuple
    return (basin_data, sim_name, theta, crn, ahea, afta, aftmtl)

# define the parallel function with pebble package
def Simulation_Wrapper(sim_params, seed=None):
    
    basin_data, sim_name, fp, crn, ahea, afta, aftmtl = sim_params
    fp = np.array(fp)
    
    # pid = os.getpid()
    # print(f"Process ID: {pid}, fp: {fp}")
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # create the parameter for the simulation
    param = CreateSimulation(sim_name, fp, basin_data)

    # get the main model results from the forward model
    data = Ilm_Forward_Na(param, basin_data, crn_calc=crn, ahea_calc=ahea, afta_calc=afta, aftmtl_calc=aftmtl, inverse=True)
    
    # get the misfit
    misfit = data.global_misfit
    
    # get modelled results
    modelled = BuildModelledData(basin_data, data, rng, elevation='specific', tcn=crn, ahea=ahea, afta=afta, aftmtl=aftmtl)
                               
    return [misfit, modelled]

def Parallel_Inverve_Modeling(n_threads, num_simulation, prior, basin_data, crn, ahea, afta, aftmtl, sim_name, 
                                 timeout=240, print_error=False):
    
    print('Start inverse modeling:')
      
    # initialize misfit and modelled lists
    misfit = []
    modelled = []
    
    # sample the prior estimation
    theta = prior.sample((num_simulation,))
    
    # Prepare simulation parameters list for each theta sample
    sim_params_list = [Prepare_Simulation_Params(basin_data, sim_name, t, crn, ahea, afta, aftmtl) for t in theta]
    
    with ProcessPool(max_workers=n_threads) as pool:

        future = pool.map(Simulation_Wrapper, sim_params_list, timeout=240)
        iterator = future.result()

        # Get the total number of simulations for the progress bar
        total_simulations = len(theta)

        # Create a tqdm progress bar
        progress_bar = tqdm(total=total_simulations, desc="Simulations")

        while True:
            try:
                result = next(iterator)
                misfit.append(result[0])
                modelled.append(result[1])
                # Update the progress bar
                progress_bar.update(1)
            
            except StopIteration:
                break
            
            except TimeoutError as error:
                if print_error:
                    print("function took longer than %d seconds" % error.args[1])
                misfit.append(error)
                modelled.append(error)
                progress_bar.update(1)
            
            except ProcessExpired as error:
                if print_error:
                    print("%s. Exit code: %d" % (error, error.exitcode))
                misfit.append(error)
                modelled.append(error)
                progress_bar.update(1)
            
            except Exception as error:
                if print_error:
                    print("function raised %s" % error)
                    print(error.traceback)
                misfit.append(error)
                modelled.append(error)
                progress_bar.update(1)

        # Close and clean up the progress bar
        progress_bar.close()
    
    return theta, modelled, misfit

if __name__ == '__main__':
    
    start_time = datetime.datetime.now()
    basin = 'neckar'
    
    # import basin dataset
    basin_data = DictionnaryImport(home_dirname + '/data/basins/' + basin + '/' + basin + '-basin_r3_t20.pkl')
    Low_Temperature_Thermochronology3(basin_data, home_dirname + '/data/basins/' + basin + '/' + basin + '-basin_thermo.csv')
    Cosmogenic_Nuclide3(basin_data, home_dirname + '/data/basins/' + basin + '/' + basin + '-basin_cosmo.csv')
    
    # define simulation and file names
    sim_name = 'simple-case'
    filename = 'run-' + basin + '_' + sim_name + '_test1'
    
    # define dataset to calculate
    crn, ahea, afta, aftmtl = False, False, False, False

    # define number of threads and simulations
    n_threads = int(mp.cpu_count()/2)
    num_simulation = 4
    
    # define the prior parameters
    p = CreatePrior(sim_name)
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(p.low), high=torch.as_tensor(p.high))
    # save the prior 
    with open(home_dirname + '/data/file_results/' + filename + '_prior.pkl', 'wb') as pr:
        pickle.dump(p, pr)
        
    # build observed data and transform to tensor
    observed = BuildObservedData(basin_data, elevation='specific', tcn=crn, ahea=ahea, afta=afta, aftmtl=aftmtl)
    observed_t = torch.tensor(observed, dtype=torch.float)
    # save the observed
    torch.save(observed_t, home_dirname + '/data/file_results/' + filename + '_observed.pt')
    
    # initialize and define sbi inference 
    inference = NPE(prior=prior, density_estimator='mdn', device='cpu')
    posteriors = []                 
    proposal = prior

    # initialize and define inverse modeling
    num_rounds = 1                  # number of simulation based inference round
    scale = 1                       # scale for the number of iteration for last round   
    results = [None]*num_rounds     # list for the final results
    
    print('Start simulations with ' + str(n_threads) + ' threads and ' + str(num_simulation) + ' iterations:') 
    print('-------------------------------')
    for r in range(num_rounds):
        
        print('Round ' + str(num_rounds) + ':')
        # defined the number of simulation based on the round stage
        if r == num_rounds - 1:
            num_simulation = num_simulation*scale
        
        theta_list, modelled_list, misfit_list = Parallel_Inverve_Modeling(n_threads=n_threads, num_simulation=num_simulation, prior=prior, basin_data=basin_data, crn=crn, ahea=ahea, afta=afta, aftmtl=aftmtl, sim_name=sim_name)
        
        # sys.exit('stop after inverse modeling')
        
        end_time = datetime.datetime.now()
        print('\nModel duration time: {}'.format(end_time-start_time))
        
        # remove exception from theta, misfit and modelled list and transform to arrays
        ind = [i for i, x in enumerate(misfit_list) if isinstance(x, float)]
        theta = np.array(theta_list[ind])
        misfit = np.array([item for index, item in enumerate(misfit_list) if index in ind])
        modelled = np.array([item for index, item in enumerate(modelled_list) if index in ind])
        
        # remove nan value from theta, misfit and modelled array
        ind = np.argwhere(~np.isnan(misfit))
        theta = theta[np.squeeze(ind),:]
        misfit = misfit[np.squeeze(ind)]
        modelled = modelled[np.squeeze(ind),:]
        
        # pack all the data in one variable
        results[r] = [theta, misfit, modelled]
        # save the main results of the inverse modelling search
        with open(home_dirname + '/data/file_results/' + filename + '_results.pkl', 'wb') as re:
            pickle.dump(results, re)
        
        sys.exit('stop after inverse modeling')
        
#%% ========================= PERFORM SBI SAMPLING =========================%%#        
        
        # transform theta, misfit and modelled to tensor
        theta_t = torch.tensor(theta, dtype=torch.float)
        misfit_t = torch.tensor(np.array([misfit]).T, dtype=torch.float)
        modelled_t = torch.tensor(np.array(modelled), dtype=torch.float)
        
        # build observed data and transform to tensor
        observed = BuildObservedData(basin_data, elevation='specific', tcn=crn, ahea=ahea, afta=afta, aftmtl=aftmtl)
        observed_t = torch.tensor(observed, dtype=torch.float)
        # observed_t = torch.min(misfit_t)
        
        # build the posterior density estimator with sbi
        density_estimator = inference.append_simulations(theta_t, modelled_t, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(observed_t)
    
    # sys.exit('stop after sbi')
    
    # get the posterior sampling with sbi
    samples = posterior.sample((10000,), x=observed_t)

#%% =========================== SAVE SBI RESULT ============================%%#   

    # # save the prior 
    # with open(home_dirname + '/data/file_results/' + filename + '_prior.pkl', 'wb') as pr:
    #     pickle.dump(p, pr)
    # # save the observed
    # torch.save(observed_t, home_dirname + '/data/file_results/' + filename + '_observed.pt')
    # # save the posterior inference
    # with open(home_dirname + '/data/file_results/' + filename + '_posterior-inference.pkl', 'wb') as pi:
    #     pickle.dump(posterior, pi)
    # # save the sbi posterior sampling
    # torch.save(samples, home_dirname + '/data/file_results/' + filename + '_posterior-sampling.pt')
    # # save the main results text file
    # results2 = results[-1]
    # with open(home_dirname + '/data/file_results/' + filename + '_main_results.txt', 'w') as f:
    
    #     f.write('Inverse modelling name: ' + filename + ' - Simulation name: ' + sim_name + '\n')
    #     f.write('model duration time: {}'.format(end_time-start_time) + '\n')
    #     f.write('forward iterations: ' + str(num_simulation) + '\n')
    #     f.write('------------------------------------\n')
    #     f.write('Parameters:\n')
    #     for i in range(len(p.label)):
    #         f.write(p.label[i] + ': ' + str(p.low[i]) + ' - ' + str(p.high[i]) + '\n')
    #     f.write('------------------------------------\n')
    
    #     f.write('Misfit (best):\n')
    #     f.write('global: ' + str(np.min(results2[1])) + '\n')
    #     f.write('topo:\n')
    #     f.write('cosmo:\n')
    #     f.write('thermo:\n')
    #     f.write('------------------------------------\n')
    #     for i in range(len(p.label)):
    #         f.write(p.label[i] + ':\n')
    #         # n, m, p = plt.hist(samples[:,i], bins=100)
    #         # index = np.argmax(n)
    #         # mode = m[index] + (m[index + 1] - m[index])/2
    #         # f.write('mode: ' + str(mode) + '\n')
    #         f.write('best: ' + str(results2[0][np.argmin(results2[1]), i]) + '\n')
    #         f.write('median: ' + str(np.median(np.array(samples)[:,i])) + '\n')
    #         f.write('mean: ' + str(np.mean(np.array(samples)[:,i])) + '\n')
    #         f.write('sigma: ' + str(np.std(np.array(samples)[:,i])) + '\n')
    #         f.write('------------------------------------\n')
    # print('done')

#%% =========== CALCULATE MODEL RESULTS FOR POSTERIOR SAMPLING =============%%#
 
# # resample the posterior
# samples2 = posterior.sample((250,), x=observed)

# # initialize the posterior results
# posterior_results = []

# free_param = np.array(samples2)
# for i in range(len(samples2)):
#     print('Calculate model results with posterior sample - iteration: ' + str(i))
#     fp = free_param[i, :]
#     param = CreateSimulation(sim_name, fp, basin_data)
#     try:
#         data = Ilm_Forward_Na(param.sample, param, basin_data, cosmo_cal=True, thermo_cal=True, inverse=False)
#         modelled = data.elevation
#         if 'tcn' in basin_data['cosmo_meas']:
#             modelled = np.concatenate((modelled, data.tcn))
#         if 'ahea' in basin_data['thermo_meas']:
#             ind = np.where(basin_data['thermo_meas']['ahea'] != -1)
#             modelled = np.concatenate((modelled, data.ahea))
#         if 'afta' in basin_data['thermo_meas']:
#             ind = np.where(basin_data['thermo_meas']['afta'] != -1)
#             modelled = np.concatenate((modelled, data.afta))
#         if 'aftmtl' in basin_data['thermo_meas']:
#             ind = np.where(basin_data['thermo_meas']['aftmtl'] != -1)
#             modelled = np.concatenate((modelled, data.aftmtl))
            
#         posterior_results.append(modelled)
#     except:
#         print('Something wrong')

# posterior_results = np.array(posterior_results)
# with open(home_dirname + '/data/file_results/' + filename + '_posterior-results.pkl', 'wb') as e:
#     pickle.dump(posterior_results, e)

#%% ========== CALCULATE MODEL RESULTS FOR POSTERIOR SAMPLING 2 ============%%#

# # sys.exit('end of simulation')
# # define the parallel function with pebble package
# def SimulationWrapper(fp, seed=None):
    
#     if seed is not None:
#         rng = np.random.RandomState(seed)
#     else:
#         rng = np.random.RandomState()
    
#     # create the parameter for the simulation
#     param = CreateSimulation(sim_name, fp, basin_data)

#     # get the main model results from the forward model
#     data = Ilm_Forward_Na(param.sample, param, basin_data, cosmo_cal=True, thermo_cal=True, inverse=True)
    
#     # get the misfit
#     misfit = data.global_misfit
    
#     # get modelled results
#     modelled = BuildModelledData(basin_data, data, rng, elevation='normal')
                               
#     return [misfit, modelled]

# # open and resample the posterior
# print('re-sampled the posterior ...')
# # filename = 'run-regen_' + 'variable-uplift-1' + '_A'
# samples = torch.load(home_dirname + '/data/file_results/' + filename + '_posterior-sampling.pt')
# samples2 = samples[0:1000]
# print('done\n')

# misfit2 = []
# modelled2 = []

# n_threads = int(mp.cpu_count()/2)

# print('re-calculate modelled results ...')
# with ProcessPool(max_workers=n_threads) as pool:

#     future = pool.map(SimulationWrapper, np.array(samples2), timeout=240)
#     iterator = future.result()

#     # Get the total number of simulations for the progress bar
#     total_simulations = len(samples2)

#     # Create a tqdm progress bar
#     progress_bar = tqdm(total=total_simulations, desc="Simulations")

#     while True:
#         try:
#             result = next(iterator)
#             misfit2.append(result[0])
#             modelled2.append(result[1])
#             # Update the progress bar
#             progress_bar.update(1)
        
#         except StopIteration:
#             break
        
#         except TimeoutError as error:
#             print("function took longer than %d seconds" % error.args[1])
#             # misfit.append(error)
#             # modelled.append(error)
#             progress_bar.update(1)
        
#         except ProcessExpired as error:
#             print("%s. Exit code: %d" % (error, error.exitcode))
#             # misfit.append(error)
#             # modelled.append(error)
#             progress_bar.update(1)
        
#         except Exception as error:
#             print("function raised %s" % error)
#             # print(error.traceback)
#             # misfit.append(error)
#             # modelled.append(error)
#             progress_bar.update(1)

#     # Close and clean up the progress bar
#     progress_bar.close()
    
# print('done\n')

# posterior_results = np.array(modelled2)
# with open(home_dirname + '/data/file_results/' + filename + '_posterior-results.pkl', 'wb') as e:
#     pickle.dump(posterior_results, e)
    
#%% ========================= OPEN FILE RESULTS ============================%%#

# filename = 'run-main_variable-uplift+simple-lithology-2_A'
# filename = 'run-neckar_variable-uplift-2_AO'
# filename2 = 'run-neckar_variable-uplift-2_A'
# filename = 'run-naab_variable-uplift+simple-lithology-1_AO'
# filename2 = 'run-naab_variable-uplift+simple-lithology-1_A'
# filename = 'run-regen_variable-uplift+simple-lithology-1_AO'
# filename2 = 'run-regen_variable-uplift+simple-lithology-1_A'
# filename = 'run-weser_variable-uplift+simple-lithology-2_AO'
# filename2 = 'run-weser_variable-uplift+simple-lithology-2_A'
# filename = 'run-saale_variable-uplift+simple-lithology-1_AO'
# filename2 = 'run-saale_variable-uplift+simple-lithology-1_A'
# filename = 'run-mulde_variable-uplift+simple-lithology-1_AO'
# filename2 = 'run-mulde_variable-uplift+simple-lithology-1_A'

# # open the main results of the inverse modelling search
# with open(home_dirname + '/data/file_results/' + filename + '_results.pkl', 'rb') as re:
#     resultstest = pickle.load(re)
# theta, misfit, modelled = results[0][0], results[0][1], results[0][2]
# # open the prior
# with open(home_dirname + '/data/file_results/' + filename + '_prior.pkl', 'rb') as pr:
#     p = pickle.load(pr)
# # open the observed
# observed = torch.load(home_dirname + '/data/file_results/' + filename + '_observed.pt')
# # open the posterior inference
# with open(home_dirname + '/data/file_results/' + filename + '_posterior-inference.pkl', 'rb') as pi:
#     posterior = pickle.load(pi)
# # open the sbi posterior sampling
# samples = torch.load(home_dirname + '/data/file_results/' + filename + '_posterior-sampling.pt')

#%% ============================ RESULT PLOTTING ========================== ##%

# Misfit_2D_Plot(results[0], p, x=5, y=4, misfit_max=2, log=True, figsize=(4.5, 3.5), save=False, filename='test' + 'inverse-modelling-misfit')

# SBI_Probability_Multi_Dimension_Plot(samples, p, num_samples=10000, true_param=[], figsize=(7.125, 7.125), bins=75, save=True, filename=filename + '_inverse-modelling-sbi')

# SBI_Misfit_MultiRound_Simulation_Plot(param_list, low, high, label=label, num_round=4, x=3, y=0, numplotx=2, numploty=2, size_dot=10, same_axis=True, figsize=(7.125, 8), save_fig=False, filename='neckar-river_inverse-modelling_misfit_run-1A')
