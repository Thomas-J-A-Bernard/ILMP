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
from geologic_functions import Lithology_to_Erodibility, Block_to_Uplift, Tilting_to_Uplift, Base_Level_Drop, River_Capture, Low_Temperature_Thermochronology, Low_Temperature_Thermochronology2, Low_Temperature_Thermochronology3, Cosmogenic_Nuclide, Cosmogenic_Nuclide2, Cosmogenic_Nuclide3
from plotting_result_functions import Misfit_2D_Plot, SBI_Probability_Multi_Dimension_Plot, SBI_Misfit_MultiRound_Simulation_Plot
from general_functions import Find_Upstream_Index, Find_Downstream_Index
from inverse_modelling_functions import CreateSimulation, CreatePrior, GetSpecificElevation, BuildModelledData, BuildObservedData

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

sys.exit('end of simulation')

#%% ===================== IMPORT CATCHMENT DATASET =========================%%#
basin = 'neckar'
basin_data = DictionnaryImport(home_dirname + '/data/basins/' + basin + '/' + basin + '-basin_r3_t20.pkl')
Low_Temperature_Thermochronology3(basin_data, home_dirname + '/data/basins/' + basin + '/' + basin + '-basin_thermo.csv')
Cosmogenic_Nuclide3(basin_data, home_dirname + '/data/basins/' + basin + '/' + basin + '-basin_cosmo.csv')

#%% ================== IMPORT INVERSE MODELING RESULTS ======================%%#

sim_name = 'simple-case'
filename1 = 'run-' + basin + '_' + sim_name + '_test'
filename2 = 'run-' + basin + '_' + sim_name + '_test'
# open the main results of the inverse modelling search
with open(home_dirname + '/data/file_results/' + filename1 + '_results.pkl', 'rb') as re:
    results = pickle.load(re)
    
#%% ======================== PERFORM OPTIMIZATION ==========================%%#    

# define a threshold for the optimization
threshold = 40000
    
# sort misfit and modelled results
sort = np.argsort(results[0][1])
theta = results[0][0][sort]
misfit = results[0][1][sort]
modelled = results[0][2][sort]

# perform optimization with threshold
theta_o= theta[0:threshold+1]
misfit_o = misfit[0:threshold+1]
modelled_o = modelled[0:threshold+1]

# transform theta, misfit and modelled to tensor
theta_ot = torch.tensor(theta_o, dtype=torch.float)
misfit_ot = torch.tensor(np.array([misfit_o]).T, dtype=torch.float)
modelled_ot = torch.tensor(np.array(modelled_o), dtype=torch.float)

# open the prior and define sbi inference
with open(home_dirname + '/data/file_results/' + filename1 + '_prior.pkl', 'rb') as pr:
    p = pickle.load(pr)
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(p.low), high=torch.as_tensor(p.high))
inference = NPE(prior=prior, density_estimator='mdn', device='cpu')
proposal = prior
posteriors = []

# open the tensor observed data
observed_t = torch.load(home_dirname + '/data/file_results/' + filename1 + '_observed.pt', weights_only=False)

# build the posterior density estimator with sbi
density_estimator = inference.append_simulations(theta_ot, modelled_ot, proposal=proposal).train()
posterior = inference.build_posterior(density_estimator)
posteriors.append(posterior)
proposal = posterior.set_default_x(observed_t)

# sample the posterior
samples_ot = posterior.sample((10000,), x=observed_t)
samples_o = np.array(samples_ot)
samples_mean = np.mean(samples_o, axis=0)
samples_std = np.std(samples_o, axis=0)
samples_r = np.vstack((samples_mean, samples_std))

# save the optimize posterior and sample
torch.save(samples_ot, home_dirname + '/data/file_results/' + filename2 + '_posterior-sampling.pt')
with open(home_dirname + '/data/file_results/' + filename2 + '_posterior-inference.pkl', 'wb') as pi:
    pickle.dump(posterior, pi)

#%% =========== CALCULATE MODEL RESULTS FOR POSTERIOR SAMPLING =============%%#

def Prepare_Simulation_Params(basin_data, sim_name, theta, crn, ahea, afta, aftmtl):
    # Return the parameters as a tuple
    return (basin_data, sim_name, theta, crn, ahea, afta, aftmtl)

# define the parallel function with pebble package
def Simulation_Wrapper(sim_params, seed=None):
    
    basin_data, sim_name, fp, crn, ahea, afta, aftmtl = sim_params
    fp = np.asarray(fp)
    
    # pid = os.getpid()
    # print(f"Process ID: {pid}, fp: {fp}")
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # create the parameter for the simulation
    param = CreateSimulation(sim_name, fp, basin_data)

    # get the main model results from the forward model
    data = Ilm_Forward_Na(param.sample, param, basin_data, crn_calc=crn, ahea_calc=ahea, afta_calc=afta, aftmtl_calc=aftmtl, inverse=True)
    
    # get the misfit
    misfit = data.global_misfit
    
    # get modelled results
    modelled = BuildModelledData(basin_data, data, rng, elevation='specific', tcn=crn, ahea=ahea, afta=afta, aftmtl=aftmtl)
                               
    return [misfit, modelled]

# open the posterior sampling
samples_ot = torch.load(home_dirname + '/data/file_results/' + filename2 + '_posterior-sampling.pt', weights_only=False)[0:1000]

# define dataset to calculate
crn, ahea, afta, aftmtl = False, False, False, False

misfit2 = []
modelled2 = []
n_threads = int(mp.cpu_count()/2)
sim_params_list = [Prepare_Simulation_Params(basin_data, sim_name, t, crn, ahea, afta, aftmtl) for t in samples_ot]
print_error = True
with ProcessPool(max_workers=n_threads) as pool:

    future = pool.map(Simulation_Wrapper, sim_params_list, timeout=240)
    iterator = future.result()

    # Get the total number of simulations for the progress bar
    total_simulations = len(samples_ot)

    # Create a tqdm progress bar
    progress_bar = tqdm(total=total_simulations, desc="Simulations")

    while True:
        try:
            result = next(iterator)
            misfit2.append(result[0])
            modelled2.append(result[1])
            # Update the progress bar
            progress_bar.update(1)
        
        except StopIteration:
            break
        
        except TimeoutError as error:
            if print_error:
                print("function took longer than %d seconds" % error.args[1])
            misfit2.append(error)
            modelled2.append(error)
            progress_bar.update(1)
        
        except ProcessExpired as error:
            if print_error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            misfit2.append(error)
            modelled2.append(error)
            progress_bar.update(1)
        
        except Exception as error:
            if print_error:
                print("function raised %s" % error)
                print(error.traceback)
            misfit2.append(error)
            modelled2.append(error)
            progress_bar.update(1)

    # Close and clean up the progress bar
    progress_bar.close()

print('done\n')

# posterior_results = np.array(modelled2)
# with open(home_dirname + '/data/file_results/' + filename2 + '_posterior-results.pkl', 'wb') as e:
#     pickle.dump(posterior_results, e)

#%% =========================== SAVE MAIN RESULT ============================%%#
# open the prior and define sbi inference
with open(home_dirname + '/data/file_results/' + filename1 + '_prior.pkl', 'rb') as pr:
    p = pickle.load(pr)

with open(home_dirname + '/data/file_results/' + filename1 + '_results.pkl', 'rb') as re:
    results = pickle.load(re)
results2 = results[-1]

samples = torch.load(home_dirname + '/data/file_results/' + filename2 + '_posterior-sampling.pt', weights_only=False)

with open(home_dirname + '/data/file_results/' + filename2 + '_main_results.txt', 'w') as f:
    f.write('Inverse modelling name: ' + filename2 + ' - Simulation name: ' + sim_name + '\n')
    f.write('------------------------------------\n')
    f.write('Parameters:\n')
    for i in range(len(p.label)):
        f.write(p.label[i] + ': ' + str(p.low[i]) + ' - ' + str(p.high[i]) + '\n')
    f.write('------------------------------------\n')

    f.write('Misfit (best):\n')
    f.write('global: ' + str(np.min(results2[1])) + '\n')
    f.write('topo:\n')
    f.write('cosmo:\n')
    f.write('thermo:\n')
    f.write('------------------------------------\n')
    for i in range(len(p.label)):
        f.write(p.label[i] + ':\n')
        f.write('best: ' + str(results2[0][np.argmin(results2[1]), i]) + '\n')
        f.write('median: ' + str(np.median(np.array(samples)[:,i])) + '\n')
        f.write('mean: ' + str(np.mean(np.array(samples)[:,i])) + '\n')
        f.write('sigma: ' + str(np.std(np.array(samples)[:,i])) + '\n')
        f.write('------------------------------------\n')
print('done')
