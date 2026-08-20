# import python packages
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import pickle
import torch
import sys
import warnings
from pathlib import Path
from collections import defaultdict, deque

# import ilmp functions
from parameters_class import Parameters
from ilm_forward_na import Ilm_Forward_Na
from catchment_dictionnary_functions import DictionnaryImport, DictionnaryExport
from geologic_functions import Lithology_to_Erodibility, Block_to_Uplift, Tilting_to_Uplift, Base_Level_Drop, River_Capture, Low_Temperature_Thermochronology, Low_Temperature_Thermochronology2, Low_Temperature_Thermochronology3, Cosmogenic_Nuclide, Cosmogenic_Nuclide2, Cosmogenic_Nuclide3, Variable_Hillslope_lenght
from plotting_result_functions import River_Point_Map_Plot, River_Profile_Map_Points_Plot, River_Profile_Points_Plot, River_Map_Points_Plot, River_Profile_Obs_vs_Mod_Plot, Interpolation_Map_Plot
from general_functions import Find_Upstream_Index
from matlab_extract import Matlab_Extract

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())
# restore the rcparams from matplotlib's internal default style
plt.rcdefaults()

#%% ========================== CATCHMENT DATASET ===========================%%#
basin = 'neckar'
basin_data = DictionnaryImport(home_dirname + '/data/basins/' + basin + '/' + basin + '-basin_r3_t20.pkl')
Low_Temperature_Thermochronology3(basin_data, home_dirname + '/data/basins/' + basin + '/' + basin + '-basin_thermo.csv') 
Cosmogenic_Nuclide3(basin_data, home_dirname + '/data/basins/' + basin + '/' + basin + '-basin_cosmo.csv')

#%% ========================== MODEL PARAMETERS ============================%%#
# set model parameters
param = Parameters(gg=17.5, T0=15, lr=5, TD=1.5e-6, rho_c=2700, hpc=18e-10, cp_c=700, 
                   tt=60, dt=1e3, dtr=0.2, start_dtr=23, end_dtr=25,
                   U=0.05, K=1.0e-6, m=0.5, n=1, icflag=2, islope=0.01, Ui=0.05, Ki=1e-6, ee=16, pixel=1,
                   hl=100, hdn=10, hk=0.01, hm=0, hn=1, crit_slope=30,
                   muon=1, cosmo_thickness=0, cosmo_topocorr=1, cosmo_aa='std', dx_cosmo=1000, t_record=2e6)

# # set variable uplift
# Tilting_to_Uplift(basin_data, param, direction='degree', uplift=[0.05, 0.05], gradient=[0, 0], degree=[0, 0], time=[0], spatial=False, block_ind=[6,10], block_uplift=[0.025,0.035])

# set variable lithology erodibility
# k_sr, k_br = 1e-6, 0.6e-6
# Lithology_to_Erodibility(basin_data, param, carbonate=k_sr, claystone=k_sr, dolomite=k_sr, limestone=k_sr, mudstone=k_sr, quartzite=k_br, plutonic=k_br, metamorphic=k_br, sand=k_sr, sandstone=k_sr, silt=k_sr, volcanic=k_sr)

# # set river capture 
# River_Capture(basin_data, node=[], time=[], initial_uplift=[])

# # set base-level drop
# Base_Level_Drop(basin_data, initial_level=[], drop_time=[])

# set variable hillslope lenght
Variable_Hillslope_lenght(basin_data, param, random=True, minimum=100, maximum=200)

# sys.exit('...')

#%% ========================== FORWARD MODELLING ===========================%%#

start_time = datetime.datetime.now()
results = Ilm_Forward_Na(param, basin_data, crn_calc=True, ahea_calc=False, afta_calc=False, aftmtl_calc=False, inverse=False)
end_time = datetime.datetime.now()
print('Model duration time: {}'.format(end_time-start_time))

# sys.exit('...')

#%% ========================== FAST RESULT PLOTTING ======================= ##%

# plot map of catchment river point
River_Point_Map_Plot(basin_data, param, key='initial_elevation', label='Elevation (m)')

# sys.exit('...')

#%% ============================ PLOT: UPLIFT FIELD ======================= ##%

# filename = 'run-saale-2B'
# fig = plt.figure(figsize=(7.125, 7.125))
# ax1 = fig.add_subplot(111)
# u = ax1.scatter(basin_data['longitude'], basin_data['latitude'], c=basin_data['uplift'], s=3, cmap='jet')
# e = ax1.scatter(basin_data['longitude'], basin_data['latitude'], c=data.erosion_record[:,250], s=10, cmap='jet')
# ax1.xaxis.set_ticks_position('top')
# ax1.xaxis.set_label_position('top')
# ax1.axis('equal')
# ax1.set_xlabel('Longitude (°)')
# ax1.set_ylabel('Latitude (°)')
# fig.colorbar(e, orientation='horizontal', label='Uplift (mm.yr$^{-1}$)', pad=0.05)
# fig.tight_layout()
# fig.savefig(dirname + '/figure/' + filename + '_uplift_field' + '.pdf', dpi=720)
# fig.savefig(dirname + '/figure/' + filename + '_uplift_field' + '.png', dpi=720)

#%% ==================== PLOT: DISCREPANCY OBS VS PRE ===================== ##%

# dis_ele = basin_data['initial_elevation'] - data.elevation
# dis_thermo = basin_data['thermo_meas']['aft'] - data.afta
# # dis_thermo = basin_data['thermo_meas']['aft'] - np.nan
# dis_cosmo = basin_data['cosmo_meas']['tcn'] - data.tcn
# area = np.log10(basin_data['area'])
# distance = basin_data['x']/1000
# # rivers = [0,18,55,89,142,205,323,426,588,894,1229,1263,1310,1523,1407,1479,1602,1635]
# rivers = np.unique(basin_data['source'])
# pairs = basin_data['pairs']
# source = basin_data['source']

# fig = plt.figure(figsize=(7.125, 5.0))
# gs = fig.add_gridspec(5, 5)
# ax1 = fig.add_subplot(gs[0:3,0:5])
# ax2 = fig.add_subplot(gs[3:5,0:2])
# ax3 = fig.add_subplot(gs[3:5,2:5])

# # plot elevation discrepancy
# # A = ax1.scatter(area, dis_ele, c=distance, cmap='rainbow', edgecolors='black', vmin=np.min(distance), vmax=np.max(distance), zorder=2)
# for r in rivers:
#     ind = np.squeeze(np.argwhere(source == r))
#     # remove last index if main tributary
#     if r == 0:
#         ind = ind[:-1]
#     # remove last index and add connection if secondary tributaries
#     else:
#         ind = ind[:-1]
#         ind = np.insert(ind, 0, ind[0]-1)
        
#     for i in ind:
#         x = np.array([basin_data['x'][int(pairs[i,0]-1)], basin_data['x'][int(pairs[i,1]-1)]])
#         y = np.array([basin_data['initial_elevation'][int(pairs[i,0]-1)], basin_data['initial_elevation'][int(pairs[i,1]-1)]])
#         obs = ax1.plot(x, y, marker='', ls='-', lw=1, c='steelblue', zorder=1)
        
#         x = np.array([basin_data['x'][int(pairs[i,0]-1)], basin_data['x'][int(pairs[i,1]-1)]])
#         y = np.array([data.elevation[int(pairs[i,0]-1)], data.elevation[int(pairs[i,1]-1)]])
#         pre = ax1.plot(x, y, marker='', ls='-', lw=1, c='darkred', zorder=2)

# # plot thermo discrepancy
# B = ax2.scatter(np.arange(0, len(basin_data['thermo_meas']['aft'])), dis_thermo, c=distance[basin_data['thermo_meas']['ind']], marker='*', s=100, cmap='rainbow', edgecolors='black', vmin=np.min(distance), vmax=np.max(distance), zorder=2)

# # plot cosmo discrepancy
# C = ax3.scatter(np.arange(0, len(basin_data['cosmo_meas']['tcn'])), dis_cosmo/1000, c=basin_data['x'][basin_data['cosmo_meas']['node']]/1000, marker='D', s=40, cmap='rainbow', edgecolors='black', vmin=np.min(distance), vmax=np.max(distance), zorder=2)

# # plot zero horizontal lines
# # ax1.axhline(y=0, lw=0.8, ls='--', c='k', zorder=1)
# ax2.axhline(y=0, lw=0.8, ls='--', c='k', zorder=1)
# ax3.axhline(y=0, lw=0.8, ls='--', c='k', zorder=1)

# # plot elevation legend
# custom_legend = [Line2D([0], [0], color='steelblue', ls='-', lw=1),
#                   Line2D([0], [0], color='darkred', ls='-', lw=1)]
# ax1.legend(custom_legend, ['Observed elevation', 'Modelled elevation'], loc='best', fancybox=False, edgecolor='black')

# # elevation axis
# ax1.set_ylabel('Discrepancy (m)')
# # ax1.set_ylabel('Elevation (m)')
# # ax1.set_xlabel('Log$_{10}$ Area (m$^{2}$)', labelpad=10)
# ax1.set_xlabel('Flow distance (km)', labelpad=10)
# xmin, xmax, ymin, ymax = ax1.axis()
# ax1.axis([xmin, xmax, ymin, ymax])
# ax1.xaxis.set_label_position('top')
# ax1.xaxis.set_ticks_position('top')
# # ax1.invert_xaxis()
# ticks_x = plticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
# ax1.xaxis.set_major_formatter(ticks_x)

# # thermo axis
# ax2.set_ylabel('Discrepancy (Myrs)')
# ax2.set_xlabel('Sample index')
# xmin, xmax, ymin, ymax = ax2.axis()
# ax2.axis([xmin-0.33, xmax+0.33, ymin-1.5, ymax+1.5])
# loc = plticker.MultipleLocator(base=1.0) 
# ax2.xaxis.set_major_locator(loc)

# # cosmo axis
# ax3.set_ylabel('Discrepancy\n(10$^{3}$ atoms.g$^{-1}$)')
# ax3.set_xlabel('Sample index')
# xmin, xmax, ymin, ymax = ax3.axis()
# ax3.axis([xmin-0.33, xmax+0.33, ymin-5, ymax+5])
# loc = plticker.MultipleLocator(base=1.0) 
# ax3.xaxis.set_major_locator(loc)

# # colorbar
# cbax = fig.add_axes([0.87, 0.1, 0.025, 0.775])
# C1 = fig.colorbar(B, label='Flow distance (km)', cax=cbax)

# fig.subplots_adjust(top=0.875, bottom=0.1, left=0.125, right=0.85, wspace=3.0, hspace=0.20)
# # fig.savefig(dirname + '/figure/' + 'run-neckar-2A_' + 'data-observed-vs-best-predicted-V3' + '.png', dpi=720)
# # fig.savefig(dirname + '/figure/' + 'run-neckar-2A_' + 'data-observed-vs-best-predicted-V3' + '.pdf', dpi=720)

#%% ============== PLOT: RIVER MAP + COSMO POINT + RIVER NODE ============= ##%

# fig = plt.figure(figsize=(10, 10))
# ax1 = fig.add_subplot(111)

# node = []

# s = 6

# rn = ax1.plot(basin_data['longitude'], basin_data['latitude'], marker='o', ls='', c='steelblue', ms=5, zorder=1)
# rn = ax1.plot(basin_data['longitude'][node], basin_data['latitude'][node], marker='o', ls='', c='darkred', ms=5, zorder=3)

# ui = ax1.plot(basin_data['longitude'][basin_data['cosmo_meas']['ind'][s]], basin_data['latitude'][basin_data['cosmo_meas']['ind'][s]], marker='o', ls='', c='darkred', ms=5, zorder=2)
# cs = ax1.plot(basin_data['cosmo_meas']['longitude'][s], basin_data['cosmo_meas']['latitude'][s], marker='D', ls='', c='k', ms=5, zorder=4)

#%% ============== PLOT: SPECIFIC RIVER PROFILE + SPECIFIC NODE =========== ##%

# fig = plt.figure(figsize=(7.125, 3.5))
# ax1 = fig.add_subplot(111)

# pairs = basin_data['pairs']
# source = basin_data['source']
# rivers = np.unique(source)

# # key tributaries of the Neckar
# # rivers = [0,205,323,426,588,894,1229,1263,1310,   # Kocher
# #           0,214,731,1185,1256,1272,1403,          # Jagst
# #           0,18,55,78,85,89]                       # Neckar

# # simplified Neckar river profile 1 (kocher)
# # rivers = [0,18,55,89,142,205,323,426,588,894,1229,1263,1310,1523,1407,1479,1602,1635]

# # simplified Neckar river profile 2 (jagst)
# # rivers = [0,18,55,89,142,214,731,1185,1256,1272,1403,1523,1407,1479,1602,1635]

# for r in rivers:
#     ind = np.squeeze(np.argwhere(source == r))
#     # remove last index if main tributary
#     if r == 0:
#         ind = ind[:-1]
#     # remove last index and add connection if secondary tributaries
#     else:
#         ind = ind[:-1]
#         ind = np.insert(ind, 0, ind[0]-1)
        
#     for i in ind:
#         x = np.array([basin_data['x'][int(pairs[i,0]-1)], basin_data['x'][int(pairs[i,1]-1)]])
#         y = np.array([basin_data['initial_elevation'][int(pairs[i,0]-1)], basin_data['initial_elevation'][int(pairs[i,1]-1)]])
#         obs = ax1.plot(x, y, marker='', ls='-', lw=1.5, c='steelblue', zorder=1)
        
#         # x = np.array([basin_data['x'][int(pairs[i,0]-1)], basin_data['x'][int(pairs[i,1]-1)]])
#         # y = np.array([modelled_elevation[int(pairs[i,0]-1)], modelled_elevation[int(pairs[i,1]-1)]])
#         # pre = ax1.plot(x, y, marker='', ls='-', lw=1, c='darkred', zorder=2)
        
# # plot specific river node
# # if node:
# #     ax1.plot(basin_data['x'][node], basin_data['initial_elevation'][node], marker='o', ls='', c='steelblue', ms=4)
# #     ax1.plot(basin_data['x'][node], modelled_elevation[node], marker='o', ls='', c='darkred', ms=5)

# rn = ax1.plot(basin_data['x'], basin_data['initial_elevation'], ls='', marker='o', ms=3, color='steelblue')
# cs = ax1.plot(basin_data['x'][basin_data['cosmo_meas']['node']], basin_data['initial_elevation'][basin_data['cosmo_meas']['node']], ls='', marker='d', ms=8, mew=1.33, mfc='gold', mec='k', zorder=2)
# ts = ax1.plot(basin_data['x'][basin_data['thermo_meas']['ind'][0:7]], basin_data['thermo_meas']['elevation'][0:7], ls='', marker='*', ms=12, mew=1.33, mfc='slateblue', mec='k', zorder=2)
# for i in range(len(basin_data['cosmo_meas']['longitude'])):
#     cn = ax1.text(basin_data['x'][basin_data['cosmo_meas']['node']][i]+3000, basin_data['initial_elevation'][basin_data['cosmo_meas']['node']][i]-30, basin_data['cosmo_meas']['name'][i])
# for i in range(7):
#     tn = ax1.text(basin_data['x'][basin_data['thermo_meas']['ind'][i]]+3000, basin_data['thermo_meas']['elevation'][i]+20, basin_data['thermo_meas']['name'][i])  
    
# # plot elevation legend
# custom_legend = [Line2D([0], [0], color='steelblue', ls='-', marker='o', lw=1.5, ms=5),
#                  Line2D([0], [0], color='darkred', ls='', marker='d', ms=8, mew=1.33, mfc='gold', mec='k'),
#                  Line2D([0], [0], color='#ffadd2', ls='', marker='*', ms=12, mew=1.33, mfc='slateblue', mec='k')]
# ax1.legend(custom_legend, ['River nodes', 'Cosmo samples', 'Thermo samples'], loc='upper left', fancybox=False, edgecolor='black')
    
# ticks_x = plticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
# ax1.xaxis.set_major_formatter(ticks_x)
# ax1.set_xlabel('Flow distance (km)')
# ax1.set_ylabel('Elevation (m)')

# fig.tight_layout()    
# fig.savefig(dirname + '/figure/' + 'run-regen-3G_river+sample-profile' + '.png', dpi=720)
# fig.savefig(dirname + '/figure/' + 'run-regen-3G_river+sample-profile' + '.pdf', dpi=720)

#%% ================ PLOT: RIVER MAP SEGMENT + SPECIFIC NODE ============== ##%

# fig = plt.figure(figsize=(7.125, 5))
# ax1 = fig.add_subplot(111)

# pairs = basin_data['pairs']
# source = basin_data['source']
# rivers = np.unique(source)
# sp_rivers = []

# for r in rivers:
#     ind = np.squeeze(np.argwhere(source == r))
#     # remove last index if main tributary
#     if r == 0:
#         ind = ind[:-1]
#     # remove last index and add connection if secondary tributaries
#     else:
#         ind = ind[:-1]
#         ind = np.insert(ind, 0, ind[0]-1)
#     # plot the river map    
#     for i in ind:
#         x = np.array([basin_data['longitude'][int(pairs[i,0]-1)], basin_data['longitude'][int(pairs[i,1]-1)]])
#         y = np.array([basin_data['latitude'][int(pairs[i,0]-1)], basin_data['latitude'][int(pairs[i,1]-1)]])
#         if r in sp_rivers:
#             rs = ax1.plot(x, y, ls='-', lw=0.75, c='darkred', zorder=1) 
#         else:
#             rs = ax1.plot(x, y, ls='-', lw=1.5, c='steelblue', zorder=1) 

# rn = ax1.plot(basin_data['longitude'], basin_data['latitude'], ls='', marker='o', ms=3, color='steelblue')
# cs = ax1.plot(basin_data['longitude'][basin_data['cosmo_meas']['node']], basin_data['latitude'][basin_data['cosmo_meas']['node']], ls='', marker='d', ms=8, mew=1.33, mfc='gold', mec='k', zorder=2)
# ts = ax1.plot(basin_data['thermo_meas']['lon'][0:7], basin_data['thermo_meas']['lat'][0:7], ls='', marker='*', ms=12, mew=1.33, mfc='slateblue', mec='k', zorder=2)
# for i in range(len(basin_data['cosmo_meas']['longitude'])):
#     cn = ax1.text(basin_data['longitude'][basin_data['cosmo_meas']['node']][i]+0.02, basin_data['latitude'][basin_data['cosmo_meas']['node']][i]-0.02, basin_data['cosmo_meas']['name'][i])
# for i in range(7):
#     tn = ax1.text(basin_data['thermo_meas']['lon'][i]+0.03, basin_data['thermo_meas']['lat'][i]-0.008, basin_data['thermo_meas']['name'][i])

# # plot elevation legend
# custom_legend = [Line2D([0], [0], color='steelblue', ls='-', marker='o', lw=1.5, ms=5),
#                  Line2D([0], [0], color='darkred', ls='', marker='d', ms=8, mew=1.33, mfc='gold', mec='k'),
#                  Line2D([0], [0], color='#ffadd2', ls='', marker='*', ms=12, mew=1.33, mfc='slateblue', mec='k')]
# ax1.legend(custom_legend, ['River nodes', 'Cosmo samples', 'Thermo samples'], loc='lower left', fancybox=False, edgecolor='black')

# ax1.yaxis.set_label_position("right")
# ax1.yaxis.tick_right()
# ax1.set_xlabel('Longitude (°)')
# ax1.set_ylabel('Latitude (°)')

# fig.tight_layout()
# # fig.savefig(dirname + '/figure/' + 'run-regen-3G_river+sample-map' + '.png', dpi=720)
# # fig.savefig(dirname + '/figure/' + 'run-regen-3G_river+sample-map' + '.pdf', dpi=720)

#%% ============== PLOT: SPECIFIC RIVER PROFILE + SPECIFIC NODE =========== ##%

# fig = plt.figure(figsize=(7.125, 3.0))
# ax1 = fig.add_subplot(111)

# x = basin_data['x']
# y = basin_data['initial_elevation']

# s = ax1.scatter(x, y, c=np.arange(0, len(x)), cmap='jet_r', s=3)
# fig.colorbar(s)

#%% =================== PLOT: RIVER PROFILE THROUGH TIME ================== ##%

# fig = plt.figure(figsize=(7.125, 3.0))
# ax1 = fig.add_subplot(111)

# z = modelled_elevation_record
# pairs = basin_data['pairs']
# source = basin_data['source']
# rivers = [0,205,323,426,588,894,1229,1263,1310]
# time = np.round(np.arange(param.tt - param.start_dtr, param.tt - param.end_dtr - param.dtr, -param.dtr), 1)

# cmap = plt.cm.get_cmap('rainbow', len(z))
# c = 0
# for m in range(0, len(z)):
#     zi = z[m]
#     for r in rivers:
#         ind = np.squeeze(np.argwhere(source == r))
#         # remove last index if main tributary
#         if r == 0:
#             ind = ind[:-1]
#         # remove last index and add connection if secondary tributaries
#         else:
#             ind = ind[:-1]
#             ind = np.insert(ind, 0, ind[0]-1)
            
#         for i in ind:
#             x = np.array([basin_data['x'][int(pairs[i,0]-1)], basin_data['x'][int(pairs[i,1]-1)]])
#             y = np.array([zi[int(pairs[i,0]-1)], zi[int(pairs[i,1]-1)]])
#             pre = ax1.plot(x, y, marker='', ls='-', lw=1, c=cmap(c), zorder=m)
#     c = c + 1

# c = fig.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.5, len(z)-0.5)), ax=ax1, ticks=np.arange(len(z)), label='Time (Ma)')
# c.ax.set_yticklabels(time)

# ax1.axis([100000, 320000, 100, 600])
# ticks_x = plticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
# ax1.xaxis.set_major_formatter(ticks_x)
# ax1.set_xlabel('Flow distance (km)')
# ax1.set_ylabel('Elevation (m)')

# fig.tight_layout()
# # fig.savefig(dirname + '/figure/' + 'C2A_knickzone-migration' + '.png', dpi=720)
# # fig.savefig(dirname + '/figure/' + 'C2A_knickzone-migration' + '.pdf', dpi=720)

#%% ======================== PLOT: BASAL TEMPERATURE ====================== ##%

# def lithosphere_geotherm(gg, se, lh, hp_exp):

#     T0 = param.T0
#     lr = param.lr
#     rho_c = param.rho_c
#     cp_c = param.cp_c
#     hpc = param.hpc
#     TD = param.TD
#     SC = 0.38
#     # param.tt = 50
    
#     steps = int(param.tt * param.dt)
#     Tbase = lh/1000*gg
#     time = np.linspace(param.tt, 0, steps+1)
#     time = time*(1e6*365.25*24*3600)                                            # time vector in sec
#     time = time[1:]
#     Ts = T0 - se*lr/1000  
#     hc = lh + se                                                               # crustal thickness in C
#     nr_nodes = 80
#     # nr_nodes = int(h/1000)
#     dx = hc/nr_nodes
#     dt_thermo = SC*np.min(dx)**2/TD 
#     time_new = np.arange(np.floor(time[0]/dt_thermo)*dt_thermo, -dt_thermo, -dt_thermo) 
    
#     tsteps = int(max(time_new.shape))
#     # tsteps = 15000
#     zsteps = int(hc/dx+1)
#     T = np.zeros((tsteps, zsteps))
#     hp = np.zeros((1, zsteps))
    
#     T[:,0] = Ts
#     T[:,-1] = Tbase
#     for i in range(0, zsteps):
#         hp[0,i] = hpc*np.exp(-(i*dx)/hp_exp)

#     change = 1
#     t = 1
#     j = np.arange(2, zsteps)
#     while change > 0.1 :
#         t += 1
#         # variable in the lithosphere
#         T[t-1,j-1] = T[t-2,j-1] + SC*(T[t-2,j] - 2*T[t-2,j-1] + T[t-2,j-2]) + hp[0,j-1]*dt_thermo/(rho_c*cp_c)
#         change = sum(abs(T[t-2,:] - T[t-1,:]))
        
#     Tf = T[t-1,:]
#     Df = np.linspace(0, hc, zsteps)
    
#     return Tf, Df

# gg = 17.5
# se = 1000
# lh = 80000
# hp_exp = 10000

# TL, DL = lithosphere_geotherm(gg, se, lh, hp_exp)
# fig = plt.figure(figsize=(6, 6))
# ax1 = fig.add_subplot(111)
# ax1.plot(TL, (DL-se)*1e-3, ls='-', marker='o', color='steelblue')
# ax1.invert_yaxis()
# ax1.set_xlabel('Temperature (°C)')
# ax1.set_ylabel('Depth (km)')
# print('Geothermal gradient (1km): ' + str((TL[1] - TL[0])/1))
# print('Geothermal gradient (2km): ' + str((TL[2] - TL[0])/2))
# print('Geothermal gradient (4km): ' + str((TL[4] - TL[0])/4))

#%% ======================= PLOT: BASAL TEMPERATURE 2 ===================== ##%

# gg = param.sample[-1]               # geothermal gradient
# T0 = param.T0                             # surface temperature at sea level in C
# lr = param.lr                              # atmospheric lapse rate in C/km
# rho_c = param.rho_c                       # crustal density in kg/m3
# cp_c = param.cp_c                       # specific heat capacity of granite in J/kg*K
# hpc = param.hpc                        # crustal heat production in W/m3
# TD = param.TD                      # thermal diffusivity in m2/s
# hpc_exp = 10e3                      # depth at which hpc is 1/e of initial hpc

# tt = 10e6                          # total time
# dt = 1e3                            # time step
# tsteps = int(tt/dt)                 # number of time steps

# lh = 80e3                           # lower depth boundary condition 
# Tbase = lh/1000*gg                  # temperature at the base
# nodes = 80                          # number of depth steps
# se = 0                           # elevation
# Tsurface = T0 - se*lr/1000          # temperature at elevation
# hc = lh + se                        # crustal thickness
# dz = hc/nodes                       # depth step
# zsteps = nodes+1

# u = 0.05921768*1e-3/(365.25*24*3600)      # define uplift in m/s

# # calculate k*dt/dx**2
# TDD = (TD*dt*(365.25*24*3600))/(dz**2)

# # define temperature field and set initial conditions
# T1 = np.zeros((tsteps, zsteps))
# T1[:,0] = Tsurface
# T1[:,-1] = Tbase

# # calculate heat production
# hp = np.zeros((zsteps))
# for i in range(0, zsteps):
#     hp[i] = (hpc*np.exp(-(i*dz/hpc_exp)))/(rho_c*cp_c)

# # solve heat conduction equation through time
# z = np.arange(1, zsteps-1)
# for t in range(1, tsteps):
#     T1[t,z] = T1[t-1,z] + TDD*(T1[t-1,z+1] - 2*T1[t-1,z] + T1[t-1,z-1]) + dt*(365.25*24*3600)*hp[z]

# # solve heat conduction equation until steady-state condition
# T2 = np.zeros((2,zsteps))
# T2[:,0], T2[:,-1] = Tsurface, Tbase
# change = 1
# threshold = 0.01
# cpt = 0
# while change > threshold:
#     T2[1,z] = T2[0,z] + TDD*(T2[0,z+1] - 2*T2[0,z] + T2[0,z-1]) + dt*(365.25*24*3600)*hp[z]
#     change = sum(abs(T2[1,:] - T2[0,:]))
#     print(cpt)
#     if change > threshold:
#         T2[0,:] = T2[1,:]
#     cpt = cpt + 1

# T3 = np.zeros((tsteps, zsteps))
# T3[0,:] = T2[-1,:]
# T3[:,0] = T1[:,0]
# T3[:,-1] = T1[:,-1]

# for t in range(1, tsteps):
#     T3[t,z] = T3[t-1,z] + TDD*(T3[t-1,z+1] - 2*T3[t-1,z] + T3[t-1,z-1]) + dt*(365.25*24*3600)*hp[z] + (T3[t-1,z+1] - T3[t-1,z])*(u*dt*(365.25*24*3600)/dz)

#%% ======================== PLOT: POSTERIOR RESULTS ====================== ##%

# # open the posterior results and calculate metrics
# filename = 'run-main_variable-uplift+simple-lithology-2_A'
# filename = 'run-neckar_variable-uplift-2_AO'
# filename = 'run-naab_variable-uplift+simple-lithology-1_AO'
# filename = 'run-regen_variable-uplift+simple-lithology-1_AO'
# filename = 'run-weser_variable-uplift+simple-lithology-2_AO'
# filename = 'run-saale_variable-uplift+simple-lithology-1_AO'
# filename = 'run-mulde_variable-uplift+simple-lithology-1_AO'
# with open(home_dirname + '/data/file_results/' + filename + '_posterior-results.pkl', 'rb') as e:
#     posterior_results = pickle.load(e)

# elevation_end = len(basin_data['initial_elevation'])
# tcn_start, tcn_end = elevation_end, elevation_end + len(basin_data['cosmo_meas']['tcn'])
# ahea_start, ahea_end = tcn_end, tcn_end + np.sum((basin_data['thermo_meas']['ahea'] >= 0))
# afta_start, afta_end = ahea_end, ahea_end + np.sum((basin_data['thermo_meas']['afta'] >= 0))
# aftmtl_start = afta_end

# elevation_mean, elevation_std = np.mean(posterior_results[:,0:elevation_end], axis=0), np.std(posterior_results[:,0:elevation_end], axis=0)*1
# tcn_mean, tcn_std = np.mean(posterior_results[:,tcn_start:tcn_end], axis=0), np.std(posterior_results[:,tcn_start:tcn_end], axis=0)*1
# ahea_mean, ahea_std = np.mean(posterior_results[:,ahea_start:ahea_end], axis=0), np.std(posterior_results[:,ahea_start:ahea_end], axis=0)*1
# afta_mean, afta_std = np.mean(posterior_results[:,afta_start:afta_end], axis=0), np.std(posterior_results[:,afta_start:afta_end], axis=0)*1
# aftmtl_mean, aftmtl_std = np.mean(posterior_results[:,aftmtl_start:], axis=0), np.std(posterior_results[:,aftmtl_start:], axis=0)*1

# pairs = basin_data['pairs']
# source = basin_data['source']
# rivers = np.unique(source)

# fig = plt.figure(figsize=(7.125, 5.5))
# gs = fig.add_gridspec(5, 10)
# ax1 = fig.add_subplot(gs[0:3,0:7])          # elevation
# ax2 = fig.add_subplot(gs[3:5,0:2])          # tcn
# ax3 = fig.add_subplot(gs[3:5,2:7])          # ahea
# ax4 = fig.add_subplot(gs[3:5,7:10])         # afta
# ax5 = fig.add_subplot(gs[0:3,7:10])         # aftmtl

# # plot river profiles
# for r in rivers:
#     ind = np.squeeze(np.argwhere(source == r))
#     # remove last index if main tributary
#     if r == 0:
#         ind = ind[:-1]
#     # remove last index and add connection if secondary tributaries
#     else:
#         ind = ind[:-1]
#         ind = np.insert(ind, 0, ind[0]-1)
        
#     for i in ind:
#         # plot the superior one sigma curve
#         x = np.array([basin_data['x'][int(pairs[i,0]-1)], basin_data['x'][int(pairs[i,1]-1)]])
#         y1 = np.array([elevation_mean[int(pairs[i,0]-1)] + elevation_std[int(pairs[i,0]-1)], elevation_mean[int(pairs[i,1]-1)] + elevation_std[int(pairs[i,1]-1)]])
#         sup = ax1.plot(x, y1, marker='', ls='-', lw=1.25, c='#ffadd2', zorder=2)
        
#         # plot the inferior one sigma curve and fill
#         y2 = np.array([elevation_mean[int(pairs[i,0]-1)] - elevation_std[int(pairs[i,0]-1)], elevation_mean[int(pairs[i,1]-1)] - elevation_std[int(pairs[i,1]-1)]])
#         sdown = ax1.plot(x, y2, marker='', ls='-', lw=1.25, c='#ffadd2', zorder=2)
        
#         ax1.fill_between(x, y1, y2, color='#ffebf4', zorder=1)
        
#         # plot the observed data
#         y = np.array([basin_data['initial_elevation'][int(pairs[i,0]-1)], basin_data['initial_elevation'][int(pairs[i,1]-1)]])
#         obs = ax1.plot(x, y, marker='', ls='-', lw=1.5, c='steelblue', zorder=3)
        
#         # plot the mean posterior result
#         y = np.array([elevation_mean[int(pairs[i,0]-1)], elevation_mean[int(pairs[i,1]-1)]])
#         best = ax1.plot(x, y, marker='', ls='-', lw=1.5, c='darkred', zorder=4)
        
# # plot samples
# ax1.plot(basin_data['x'][basin_data['cosmo_meas']['node']], basin_data['initial_elevation'][basin_data['cosmo_meas']['node']], marker='o', ls='', mec='k', mfc='#FFE68C', ms=8.5, zorder=5)
# ax1.plot(basin_data['x'][basin_data['thermo_meas']['node']], basin_data['initial_elevation'][basin_data['thermo_meas']['node']], marker='D', ls='', mec='k', mfc='#8CBEFF', ms=7.5, zorder=5)

# # plot cosmogenic nuclide concentration
# ax2.errorbar(x=np.arange(0, len(basin_data['cosmo_meas']['tcn'])), y=basin_data['cosmo_meas']['tcn']/1000, yerr=basin_data['cosmo_meas']['tcn_error']/1000, marker='o', ls='', mfc='steelblue', mec='steelblue', ecolor='lightsteelblue', capsize=5, capthick=2, zorder=1)
# ax2.errorbar(x=np.arange(0, len(basin_data['cosmo_meas']['tcn'])), y=tcn_mean/1000, yerr=tcn_std/1000, marker='o', ls='', mfc='darkred', mec='darkred', ecolor='#ffadd2', capsize=5, capthick=2, zorder=2)

# # plot apatite helium ages
# ind1 = np.where(basin_data['thermo_meas']['ahea'] != -1)
# ax3.errorbar(x=np.arange(0, len(ind1[0])), y=basin_data['thermo_meas']['ahea'][ind1], yerr=basin_data['thermo_meas']['ahea_error'][ind1], marker='o', ls='', mfc='steelblue', mec='steelblue', ecolor='lightsteelblue', capsize=5, capthick=2, zorder=1)
# ax3.errorbar(x=np.arange(0, len(ind1[0])), y=ahea_mean, yerr=ahea_std, marker='o', ls='', mfc='darkred', mec='darkred', ecolor='#ffadd2', capsize=5, capthick=2, zorder=2)

# # plot apatite fission track ages
# ind2 = np.where(basin_data['thermo_meas']['afta'] != -1)
# ax4.errorbar(x=np.arange(0, len(ind2[0])), y=basin_data['thermo_meas']['afta'][ind2], yerr=basin_data['thermo_meas']['afta_error'][ind2], marker='o', ls='', mfc='steelblue', mec='steelblue', ecolor='lightsteelblue', capsize=5, capthick=2, zorder=1)
# ax4.errorbar(x=np.arange(0, len(ind2[0])), y=afta_mean, yerr=afta_std, marker='o', ls='', mfc='darkred', mec='darkred', ecolor='#ffadd2', capsize=5, capthick=2, zorder=2)

# # plot apatite fission track mean track length
# ind3 = np.where(basin_data['thermo_meas']['aftmtl'] != -1)
# ax5.errorbar(x=np.arange(0, len(ind3[0])), y=basin_data['thermo_meas']['aftmtl'][ind3], yerr=basin_data['thermo_meas']['aftmtl_sd'][ind3], marker='o', ls='', mfc='steelblue', mec='steelblue', ecolor='lightsteelblue', capsize=5, capthick=2, zorder=1)
# ax5.errorbar(x=np.arange(0, len(ind3[0])), y=aftmtl_mean, yerr=aftmtl_std, marker='o', ls='', mfc='darkred', mec='darkred', ecolor='#ffadd2', capsize=5, capthick=2, zorder=2)

# # plot elevation legend
# custom_legend = [Line2D([0], [0], marker='o', color='steelblue', ls='-', lw=1.5),
#                  Line2D([0], [0], marker='o', color='darkred', ls='-', lw=1.5),
#                  Line2D([0], [0], color='#ffadd2', ls='-', lw=1.5)]
# ax1.legend(custom_legend, ['Observed data', 'Modelled results', '1\u03C3 uncertainty'], loc='best', fancybox=False, edgecolor='black')

# # elevation axis
# ax1.set_ylabel('Elevation (m)')
# ax1.set_xlabel('Flow distance (km)', labelpad=10)
# xmin, xmax, ymin, ymax = ax1.axis()
# ax1.axis([xmin, xmax, ymin, np.max(basin_data['initial_elevation'])*1.1])
# ax1.xaxis.set_label_position('bottom')
# ax1.xaxis.set_ticks_position('bottom')
# ticks_x = plticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
# ax1.xaxis.set_major_formatter(ticks_x)
# ax1.xaxis.set_label_coords(0.5, 0.08)
# # cosmo axis
# ax2.set_ylabel('$^{10}$Be CRNC\n(10$^{3}$ atoms.g$^{-1}$)')
# xmin, xmax, ymin, ymax = ax2.axis()
# ax2.axis([xmin-0.33, xmax+0.33, ymin-5, ymax+5])
# loc = plticker.MultipleLocator(base=1.0) 
# ax2.xaxis.set_major_locator(loc)
# ax2.set_xticks(np.arange(0, len(basin_data['cosmo_meas']['tcn'])))
# rotation=66
# ax2.set_xticklabels(basin_data['cosmo_meas']['name'], rotation=rotation)
# # ahe axis
# ax3.set_ylabel('AHeA (Ma)')
# xmin, xmax, ymin, ymax = ax3.axis()
# ax3.axis([xmin-0.33, xmax+0.33, ymin-1.5, ymax+1.5])
# loc = plticker.MultipleLocator(base=1.0) 
# ax3.xaxis.set_major_locator(loc)
# ax3.set_xticks(np.arange(0, len(ind1[0])))
# ax3.set_xticklabels(basin_data['thermo_meas']['name'][ind1], rotation=rotation)
# # aft axis
# ax4.set_ylabel('AFTA (Ma)')
# xmin, xmax, ymin, ymax = ax4.axis()
# ax4.axis([xmin-0.33, xmax+0.33, ymin-1.5, ymax+1.5])
# loc = plticker.MultipleLocator(base=1.0) 
# ax4.xaxis.set_major_locator(loc)
# ax4.set_xticks(np.arange(0, len(ind2[0])))
# ax4.set_xticklabels(basin_data['thermo_meas']['name'][ind2], rotation=rotation)
# # aftmtl axis
# ax5.set_ylabel('AFTMTL (μm)')
# xmin, xmax, ymin, ymax = ax5.axis()
# ax5.axis([xmin-0.33, xmax+0.33, ymin-1.5, ymax+1.5])
# ax5.xaxis.set_label_position('bottom')
# ax5.xaxis.set_ticks_position('bottom')
# loc = plticker.MultipleLocator(base=1.0)
# ax5.xaxis.set_major_locator(loc)
# ax5.yaxis.set_major_locator(loc)
# ax5.set_xticks(np.arange(0, len(ind3[0])))
# ax5.set_xticklabels(basin_data['thermo_meas']['name'][ind3], rotation=rotation)
# ax5.tick_params(axis='x', labelbottom=False)

# fig.subplots_adjust(top=0.98, bottom=0.24, left=0.13, right=0.97, wspace=1000, hspace=0.5)
# # fig.tight_layout()
# fig.savefig(home_dirname + '/data/figures/' + filename + '_PUB_data-observed-vs-best-predicted-V3.png', dpi=720)
# fig.savefig(home_dirname + '/data/figures/' + filename + '_PUB_data-observed-vs-best-predicted-V3.pdf', dpi=720)

#%% ===================== PLOT: UPLIFT RATE THROUGH TIME ================== ##%

# filename = 'run-main_variable-uplift+simple-lithology-2_A'
# filename = 'run-neckar_variable-uplift-2_AO'
# filename = 'run-naab_variable-uplift+simple-lithology-1_AO'
# filename = 'run-regen_variable-uplift+simple-lithology-1_AO'
# filename = 'run-weser_variable-uplift+simple-lithology-2_AO'
# filename = 'run-saale_variable-uplift+simple-lithology-1_AO'
# filename = 'run-mulde_variable-uplift+simple-lithology-1_AO'

# samples2 = torch.load(home_dirname + '/data/file_results/' + filename + '_posterior-sampling.pt')

# pr = np.array(samples2)
# mr = np.zeros((2,np.shape(pr)[1]))
# for i in range(np.shape(pr)[1]):
#     mr[0,i] = np.mean(pr[:,i])
#     mr[1,i] = np.std(pr[:,i])

# x = np.array([160,120,80,40,20,10,0])                      # variable-uplift-1
# # x = np.array([120,80,40,20,10,0])                        # variable-uplift-2
# y1 = np.insert(mr[0,0:len(x)-1], 0, mr[0,0], axis=0)
# y2 = np.insert(mr[1,0:len(x)-1], 0, mr[1,0], axis=0)

# fig = plt.figure(figsize=(7.125, 2.75))
# ax1 = fig.add_subplot(111)

# # plot 1sigma patches
# for i in range(len(x)-1):
#     rect = Rectangle(xy=(x[i], y1[i+1]-y2[i+1]), width=-(x[i]-x[i+1]), height=y2[i+1]*2, fc='#ffadd2', alpha=0.5, zorder=2)
#     ax1.add_patch(rect)

# # plot curves
# ax1.step(x, y1, ls='-', lw=1.75, color='darkred', zorder=4)
# ax1.step(x, y1+y2*1, ls='-', lw=1.5, color='#CA4A7D', zorder=3)
# ax1.step(x, y1-y2*1, ls='-', lw=1.5, color='#CA4A7D', zorder=3)

# # plot legend
# custom_legend = [Line2D([0], [0], color='darkred', ls='-', lw=1.5),
#                   Line2D([0], [0], color='#CA4A7D', ls='-', lw=1.5)]
# ax1.legend(custom_legend, ['Mean', '1\u03C3'], loc='best', fancybox=False, edgecolor='black')

# # plot text
# # t = 'K = ' + str(np.round(mr[0][-1], 8)) + ' ± ' + str(np.round(mr[1][-1], 8))
# # ax1.text(0.05, 0.05, t, horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)

# # ax1.axis([0,160,0,ax1.axis()[3]])
# ax1.axis([x[-1],x[0],0,np.max(mr[0] + mr[1])*1.1])
# ax1.invert_xaxis()
# ax1.set_xlabel('Time (Ma)')
# ax1.set_ylabel('Uplift rate (mm.yr$^{-1}$)')
# ax1.xaxis.set_label_position('top')
# ax1.xaxis.set_ticks_position('top')
# fig.subplots_adjust(top=0.84, bottom=0.05, left=0.13, right=0.97)
# # fig.tight_layout()
# fig.savefig(home_dirname + '/data/figures/' + filename + '_PUB_uplift-rate-prediction.png', dpi=720)
# fig.savefig(home_dirname + '/data/figures/' + filename + '_PUB_uplift-rate-prediction.pdf', dpi=720)

#%% ================= PLOT: BOX-PLOT TOPO + ANALYTICAL DATA =============== ##%

# # import dataset
# basin_list = ['neckar', 'main', 'naab', 'regen', 'weser', 'saale', 'mulde']
# basin_mchi_list, basin_crn_list, basin_ahea_list, basin_afta_list  = [], [], [], []
# for i in basin_list:
#     # get basin data
#     basin_data = DictionnaryImport(home_dirname + '/data/basins/' + i + '/' + i + '-basin_r3_t20.pkl')
#     Low_Temperature_Thermochronology3(basin_data, home_dirname + '/data/basins/' + i + '/' + i + '-basin_thermo.csv') 
#     Cosmogenic_Nuclide3(basin_data, home_dirname + '/data/basins/' + i + '/' + i + '-basin_cosmo.csv')
#     basin_crn_list.append(basin_data['cosmo_meas']['tcn']/1000)
#     basin_ahea_list.append(np.delete(basin_data['thermo_meas']['ahea'], np.where(basin_data['thermo_meas']['ahea'] == -1)))
#     basin_afta_list.append(np.delete(basin_data['thermo_meas']['afta'], np.where(basin_data['thermo_meas']['afta'] == -1)))
    
#     # get basin mchi
#     basin_chi = pd.read_csv('/home/ubuntu/LSDTopoTools/data/Germany/' + i + '-basin/' + i + '_MChiSegmented.csv')
#     mchi = np.array(basin_chi['m_chi'])
#     basin_mchi_list.append(mchi)
    
# # plot figure
# fig = plt.figure(figsize=(7.125, 4))
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)

# meanlineprops = dict(linestyle='-', linewidth=1, color='darkred')
# ax1.boxplot(basin_mchi_list, showfliers=False, showmeans=True, meanline=True, meanprops=meanlineprops)
# ax2.boxplot(basin_crn_list, showfliers=True, showmeans=True, meanline=True, meanprops=meanlineprops)
# ax3.boxplot(basin_ahea_list, showfliers=True, showmeans=True, meanline=True, meanprops=meanlineprops)
# ax4.boxplot(basin_afta_list, showfliers=True, showmeans=True, meanline=True, meanprops=meanlineprops)

# # ksn legend
# ax1.set_xticklabels(labels=[])
# ax1.set_ylabel('k$_{sn}$')
# # crn legend
# ax2.set_xticklabels(labels=[])
# ax2.set_ylabel('$^{10}$Be CRNC')
# # ahea legend
# ax3.set_xticklabels(labels= basin_list, rotation=66)
# ax3.set_ylabel('AHeA (Ma)')
# # afta legend
# ax4.set_xticklabels(labels= basin_list, rotation=66)
# ax4.set_ylabel('AFTA (Ma)')

# fig.tight_layout()
# fig.savefig(home_dirname + '/data/figures/boxplot-topo+analytical-dataset.png', dpi=720)
# fig.savefig(home_dirname + '/data/figures/boxplot-topo+analytical-dataset.pdf', dpi=720)

#%% ===================== PLOT: CHI DATA + RESPONSE TIME ================== ##%

# parameters for the stream incision power model
x = basin_data['x']
N = len(x)
z = results.elevation_river
A = basin_data['area']
if 'erodibility' in basin_data:
    K = basin_data['erodibility']
else:
    K = np.ones(N)*param.K
pairs = basin_data['pairs'].astype(int)
m = param.m
n = param.n

# build network
children = defaultdict(list)
parent = np.full(N, -1, dtype=int)

for p, c in pairs:
    p -= 1
    c -= 1
    children[p].append(c)
    parent[c] = p

# outlet = node with no parent
outlet = np.where(parent == -1)[0]

if len(outlet) != 1:
    raise ValueError(f"Expected one outlet, found {len(outlet)}")
outlet = outlet[0]

# compute slope for each node
S = np.zeros(N)
for p, c in pairs:
    p -= 1
    c -= 1
    dx = x[c] - x[p]
    dz = z[c] - z[p]
    S[c] = abs(dz / dx)

# give outlet the same slope as its first upstream reach
first_child = children[outlet][0]
S[outlet] = S[first_child]
S = np.maximum(S, 1e-12)

# calculate response time
tau_river = np.zeros(N)
integrand = 1.0 / (K * A**m * S**(n-1))
queue = deque([outlet])

while queue:
    p = queue.popleft()
    for c in children[p]:
        dx = x[c] - x[p]
        dt = 0.5 * (integrand[p] + integrand[c]) * dx
        tau_river[c] = tau_river[p] + dt
        queue.append(c)

# convert to Myr if desired
tau_river = tau_river*1e-6

# parameters for hillslope erosion
hm = param.hm
hn = param.hn
hk = param.hk
if hasattr(param, 'hlflag'):
    hl = basin_data['hillslope']['length']
else:
    hl = np.ones(N)*param.hl
Sh = np.abs((results.elevation_hillslope[-1,:] - results.elevation_hillslope[0,:])/hl)
    
tau_hill = hl**(2 - hm)/(hk*Sh**(hn - 1))/1e6

# plot channel reponse time
fig = plt.figure(figsize=(7.125, 4.0))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

tr = ax1.scatter(basin_data['longitude'], basin_data['latitude'], c=tau_river, s=25, cmap='jet')
th = ax2.scatter(basin_data['longitude'], basin_data['latitude'], c=tau_hill, s=25, cmap='jet')
ts = ax3.scatter(basin_data['longitude'], basin_data['latitude'], c=tau_river+tau_hill, s=25, cmap='jet')
fig.colorbar(tr, orientation='horizontal', label='Fluvial Response\nTime (Myrs)', pad=0.05)
fig.colorbar(th, orientation='horizontal', label='Hillslope Response\nTime (Myrs)', pad=0.05)
fig.colorbar(ts, orientation='horizontal', label='System Response\nTime (Myrs)', pad=0.05)

ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')
ax1.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
ax1.xaxis.set_label_position('top') 
ax2.set_xlabel('Longitude (°)')
ax2.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
ax2.tick_params(axis='y', labelleft=False)
ax2.xaxis.set_label_position('top') 
ax3.set_xlabel('Longitude (°)')
ax3.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
ax3.tick_params(axis='y', labelleft=False)
ax3.xaxis.set_label_position('top') 

fig.tight_layout()
