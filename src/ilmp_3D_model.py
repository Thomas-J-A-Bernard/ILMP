import datetime
import pickle
import torch
import sys

import math as mt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.colors as mcolors
import mintpy.objects.colors as moc
import pandas as pd

from random import randint
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from landlab import RasterModelGrid
from landlab.components import Lithology
from osgeo import gdal
from pathlib import Path

from parameters_class import Parameters
from ilm_forward_na import Ilm_Forward_Na
from catchment_dictionnary_functions import DictionnaryImport, DictionnaryExport
from geologic_functions import Lithology_to_Erodibility, Block_to_Uplift, Tilting_to_Uplift, Base_Level_Drop, River_Capture, Low_Temperature_Thermochronology, Cosmogenic_Nuclide
from plotting_result_functions import River_Profile_Map_Points_Plot, River_Profile_Points_Plot, River_Map_Points_Plot, River_Profile_Obs_vs_Mod_Plot, Interpolation_Map_Plot
from matlab_extract import Matlab_Extract
from general_functions import Find_Upstream_Index, Find_Downstream_Index

# get absolute path of main directory
dirname = str(Path(__file__).parent.absolute())
# restore the rcparams from matplotlib's internal default style
plt.rcdefaults()

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
    
def Get_Raster_Data(path, remove_low_values=False, low=0, remove_high_values=False, high=0, replace_by=np.nan):
    '''
    DESCRIPTION: 
    ----------
    PARAMETERS
    path : TYPE
        DESCRIPTION.
    remove_low_values : TYPE, optional
        DESCRIPTION. The default is False.
    low : TYPE, optional
        DESCRIPTION. The default is 0.
    remove_high_values : TYPE, optional
        DESCRIPTION. The default is False.
    high : TYPE, optional
        DESCRIPTION. The default is 0.
    replace_by : TYPE, optional
        DESCRIPTION. The default is np.nan.
    -------
    RETURNS
    data : TYPE
        DESCRIPTION.
    geo : TYPE
        DESCRIPTION.
    '''
    
    dem = gdal.Open(path, gdal.GA_ReadOnly)
    band1 = dem.GetRasterBand(1)
    data = band1.ReadAsArray()
    if remove_low_values:
        data = np.where(data < low, replace_by, data)
    if remove_high_values:
        data = np.where(data > high, replace_by, data)
    geo = dem.GetGeoTransform()
    
    return data, geo

def haversine(latitude1, longitude1, latitude2, longitude2):
    
    lat1, lon1 = latitude1, longitude1
    lat2, lon2 = latitude2, longitude2
    
    R = 6371000 # radius of Earth in meters
    phi1 = mt.radians(lat1)
    phi2 = mt.radians(lat2)
    
    delta_phi = mt.radians(lat2 - lat1)
    delta_lambda = mt.radians(lon2 - lon1)
    
    a = mt.sin(delta_phi/2.0)**2 + mt.cos(phi1)*mt.cos(phi2)*mt.sin(delta_lambda/2.0)**2
    c = 2*mt.atan2(mt.sqrt(a), mt.sqrt(1 - a))
    
    distance = R*c
    
    return distance

def thickness_tilted(basin_data, azimuth=0, dip=0):
    
    lat = basin_data['latitude']
    lon = basin_data['longitude']

    lon_min = np.min(lon)
    lon_max = np.max(lon)

    lat_min = np.min(lat)
    lat_max = np.max(lat)

    lon_middle = (lon_min + lon_max)/2
    lat_middle = (lat_min + lat_max)/2

    radian = azimuth*np.pi/180
    
    lon_new = (lon - lon_middle)*np.cos(radian) - (lat - lat_middle)*np.sin(radian) + lon_middle
    lat_new = (lon - lon_middle)*np.sin(radian) + (lat - lat_middle)*np.cos(radian) + lat_middle
    
    lon_new_min = np.min(lon_new)
    lon_new_max = np.max(lon_new)
    lat_new_min = np.min(lat_new)
    lat_new_max = np.max(lat_new)
    lon_new_middle = (lon_new_min + lon_new_max)/2

    nn = len(basin_data['x'])
    distance, depth = np.zeros(nn), np.zeros(nn)
    for i in range(nn):
        distance[i] = haversine(lat_new_max, lon_new_middle, lat_new[i], lon_new_middle)
        depth[i] = mt.tan(mt.radians(dip))*distance[i]
        
    return depth

def Create_list_color(ncolors):
    
    color_list = []
    n = ncolors
    for i in range(n):
        color_list.append('#%06X' % randint(0, 0xFFFFFF))
        
    return color_list
    
#%% IMPORT CATCHMENT AND 3D GEOLOGY MODEL

# import catchment dataset
basin_data = DictionnaryImport(dirname + "/basins/neckar/neckar-basin_r3_t20.pkl")

# set model parameters
param = Parameters(nu=1, nux=1, nuv=0,
                    T0=15, lr=5, TD=1.5e-6, rho_c=2700, hpc=18e-10, cp_c=700, 
                    tt=45, dt=1e3, dtr=0.2, start_dtr=23, end_dtr=25,
                    icflag=2, islope=0.01, Ui=0.05, Ki=1e-6, kflag=0, ee=16, pixel=1,
                    hl=100, hdx=10, hk=0.01, hm=0, hn=1, crit_slope=30,
                    muon=1, cosmo_thickness=0, cosmo_topocorr=1, cosmo_aa='std', dx_cosmo=1000, t_record=2e6)

# set spatial constant model samples (uplift rate, erodibility, n, m, geothermal gradient)
param.sample = np.array([0.05, 1e-6, 1.0, 0.5, 17.5])

# set calculation
sample = param.sample
param = param
data = basin_data
cosmo_cal = False
thermo_cal = False
inverse = False

# # 3D geological model of Baden-Wuttemberg
# with open('3DM_geology/bw_thickness.pkl', 'rb') as re:
#     thickness_DEM = pickle.load(re) 
# TOPO_data, TOPO_geo = Get_Raster_Data('3DM_geology/bw_topo_3000m.tif')
# xd, yd = np.shape(TOPO_data)
# lat_dem = np.linspace(TOPO_geo[3] + 0.5*TOPO_geo[5], TOPO_geo[3] + 0.5*TOPO_geo[5] + TOPO_geo[5]*xd, xd)
# lon_dem = np.linspace(TOPO_geo[0] + 0.5*TOPO_geo[1], TOPO_geo[0] + 0.5*TOPO_geo[1] + TOPO_geo[1]*yd, yd)
# thickness = np.zeros((np.shape(thickness_DEM)[0], len(basin_data['x'])))
# for i in range(len(basin_data['x'])):
#     lat_river, lon_river = basin_data['latitude'][i], basin_data['longitude'][i]
#     ir, jr = np.argmin(abs(lat_river - lat_dem)), np.argmin(abs(lon_river - lon_dem))
#     thickness[:,i] = np.flip(thickness_DEM[:,ir*yd + jr])
# erodibility_id = np.array([1.2e-6, 1.0e-6, 0.9e-6, 1.0e-6, 0.9e-6, 1.3e-6, 0.9e-6, 0.5e-6]) # erodibility id at surface

# # flat layers 3D geological model
# thickness = np.zeros((3, len(basin_data['x']))) # thickness of the lithology layers at each nodes
# thickness[0,:], thickness[1,:], thickness[2,:] = 250, 250, 3000
# erodibility_id = np.array([1e-6, 1.1e-6, 0.7e-6]) # erodibility for each lithology id

# # dipping layers 3D geological model 1
# thickness1 = thickness_tilted(basin_data, azimuth=0, dip=0.4) + 100
# thickness2 = np.ones(len(basin_data['x']))*300
# thickness3 = np.ones(len(basin_data['x']))*600
# thickness4 = np.ones(len(basin_data['x']))*2000
# thickness = np.vstack((thickness1, thickness2, thickness3, thickness4))
# erodibility_id = np.array([1.0e-6, 1.5e-6, 0.7e-6, 0.6e-6]) # erodibility id at surface
# surface_id = np.zeros((len(basin_data['x'])), dtype=int) # lithology id at surface

# dipping layers 3D geological model 2
dip = 0.3
azimuth = 0
thickness1 = thickness_tilted(basin_data, azimuth=azimuth, dip=dip) + 500
thickness2 = np.ones(len(basin_data['x']))*5000
thickness = np.vstack((thickness1, thickness2))
erodibility_id = np.array([1.0e-6, 2e-6]) 
surface_id = np.zeros((len(basin_data['x'])), dtype=int) 

#%% RESOLVE STREAM POWER MODEL

# define parameters
nu = param.nu                       # number of uplift step
nux = param.nux                     # lateral zone uplift
nuv = param.nuv                     # lateral variable uplift

if hasattr(param, 'ua'):            # time for change in uplift
    ua = param.ua
else:
    ua = []                         

T0 = param.T0                       # surface temperature
lr = param.lr                       # atmospheric lapse rate
TD = param.TD                       # thermal diffusovity
rho_c = param.rho_c                 # crustal density
hpc = param.hpc                     # crustal heat production
cp_c = param.cp_c                   # specific heat capacity of granite

tt = param.tt                       # model duration of Myr
dtr = param.dtr                     # laps time record in Myr
start_dtr = param.start_dtr         # start of model record
end_dtr = param.end_dtr             # end of model record

icflag = param.icflag               # slope shape
islope = param.islope               # constant slope
Ui = param.Ui                       # initial uplift rate (if icflag = 3)
Ki = param.Ki                       # initial erodibility (if icflag = 3)
kflag = param.kflag                 # spatial variable erodibility
dt = param.dt                       # time step 
ee = param.ee                       # elevation uncertainty
pixel = param.pixel                 # area unit

hl = param.hl                       # hillslope length
hdx = param.hdx                     # node spacing
hk = param.hk                       # hillslope diffusion rate 
hm = param.hm                       # distance exponent
hn = param.hn                       # slope exponent
crit_slope = param.crit_slope       # criticall hillslope

muon = param.muon                   # muon model production
dx_cosmo = param.dx_cosmo           # distance between sample location for tcn calculation
t_record = param.t_record           # time-span for tcn calculation

n = sample[-3]                      # slope exponent
m = sample[-2]                      # area exponent
gg = sample[-1]                     # geothermal gradient

    
initial_elevation = np.squeeze(data['initial_elevation'])       # observed river profile elevation
x = np.squeeze(data['x'])                                       # node flow distance
area = np.squeeze(data['area'])                                 # node area
pairs = np.squeeze(data['pairs'])                               # node pairs
dx_dem = np.squeeze(data['dx_dem'])                             # node resolution
skipping_factor = np.squeeze(data['skipping_factor'])
capture = data['capture']                                       # river capture information
drop = data['base_level_drop']                                  # base level information

if kflag == 1:
    erodibility = data['erodibility']
if nuv == 1:
    uplift = data['uplift']

# define uplift steps and erosional parameter
if nu > 1:
    TI = sample[0:nu-1]
if nu == 1:
    UI = sample[0:nux**2]*1e-3
    k = sample[nux**2]
else:
    UI = sample[nu-1:nu*nu**2+nu-1]*1e-3
    k = sample[nu*nux**2+nu-1]


# start calculation time for the stream power model
start_time = datetime.datetime.now()

# generate initial river profiles according to icflag
z0 = np.zeros(np.shape(initial_elevation))
if icflag == 1:                                 # initial slope of rivers
    z0 = initial_elevation[0]+x*islope

elif icflag == 2:                               # initial plateau
    if 'drop1' in drop:
        z0[:] = drop['drop1']['initial_level']
    else:
        z0[:] = initial_elevation[0]

elif icflag == 3:                               # initial steady-state profiles
    if pixel == 0:
        S = (Ui*1e-3/(Ki*((area*dx_dem**2)**m)))**(1/n)
    else:
        S = (Ui*1e-3/(Ki*(area**m)))**(1/n)
    
    z0[0] = initial_elevation[0]
    z0[1:] = initial_elevation[np.transpose(pairs[:,0].astype(int)-1)]*S[1:]*dx_dem*skipping_factor
    
# generate initial hillslope profile according to icflag
hillx = np.linspace(hl, 0, int(hl/hdx+1))
hill0 = np.zeros((int(1+hl/hdx), max(z0.shape)))
if icflag == 1:
    for i in range(0, max(z0.shape)):
        hill0[:,i] = initial_elevation(i)*hillx*islope

elif icflag == 2:
    for i in range(0, max(z0.shape)):
        if 'drop1' in drop:
            hill0[:,i] = drop['drop1']['initial_level']
        else:
            hill0[:,i] = initial_elevation[0]    

elif icflag == 3:
    z_ss = np.zeros(max(hillx.shape))
    if hm == 0 and hn == 1:
        z_ss = 0-UI*1e-3*hillx**2/(2*hk)
        z_ss = z_ss - z_ss[-1]
    elif hm == 2 and hn == 2:
        z_ss = 0-2*np.sqrt(UI*1e-3/hk)*hillx**0.5
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

# define the constant (kflag = 0) or variable erodibility (kflag = 1)
if kflag == 0:
    K = np.ones(np.shape(z0))*k
else:
    K = erodibility

# define the temporal and lateral variable uplift field (version 3)
U_initial = np.zeros((nu, max(z0.shape)))
for i in range(0, nu):
    if nuv == 0:
        U_initial[i,:] = UI[i]
if nuv == 1:
    U_initial = uplift*1e-3
    
U_final = U_initial[:,0]
U_initial[:,0] = 0

# number of steps per time interval (version 2)
if len(ua) == 0:
    TI = np.array([tt*1e6, 0])
else:
    TI = np.flip(np.sort(ua))*1e6
    TI = np.insert(TI, [0, TI.size], [tt*1e6, 0])
    
nsteps = np.round(-np.diff(TI)/dt)
steps_cosmo = t_record/dt

# initialize hillslope variables
hillz = np.copy(hill0)
maxi = max(hillx.shape)
hillx = np.transpose(np.tile(hillx, (max(z0.shape), 1)))
qs = np.zeros(np.shape(hillx))
xm = np.zeros(np.shape(hillx))
k1xm = np.zeros(np.shape(hillx))
k1 = -1*hk/(hdx**hn)
k3 = dt/hdx
for i in range(0, maxi):
    xm[i,:] = (i*hdx)**hm
xm[0,:] = xm[0,:]+(0.2*hdx)
k1xm1 = k1*xm[1]
k1xmm = k1*xm[maxi-1]
k1xm = k1*xm
crit_elevation = np.tan(crit_slope)*hl/hdx

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
erosion_river = np.zeros((int(t_record/dt), max(z0.shape)))
erosion_hillslope = np.zeros((int(t_record/dt), np.shape(hillz)[0], np.shape(hillz)[1]))
erosion_record = np.zeros((max(z0.shape), int(np.sum(nsteps))))
elevation_record = np.zeros((max(z0.shape), int(np.sum(nsteps))))
hillslope_record = np.zeros((int(1+hl/hdx), max(z0.shape), int(np.sum(nsteps))))
surface_id_record = np.zeros((max(z0.shape), int(np.sum(nsteps))), dtype=int)
erosion_cum = np.zeros(len(x)) # cumulative erosion at nodes

# number of uplift step
us = len(ua) + 1

# start uplift step
for l in range(0, us):
    
    # start time step of uplift
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
        S_hillz = -np.diff(hillz)
        hillz[[S_hillz>crit_elevation, np.zeros((1, np.shape(hillz)[1]))] == 1] = hillz[[np.zeros((1, np.shape(hillz)[1])), S_hillz>crit_elevation] == 1]+crit_elevation
        
        # record processes through time step
        erosion_record[:,steps-1] = (U[l,:] - (z1 - z00)/dt)*1e3
        elevation_record[:,steps-1] = z1
        hillslope_record[:,:,steps-1] = hillz
        surface_id_record[:,steps-1] = surface_id
        erosion_cum = erosion_cum + (U[l,:] - (z1[:] - z00[:])/dt)*1e3

        # update erodibility based on cumulative erosion at river nodes
        for i in range(len(x)):
            if erosion_cum[i] > thickness[surface_id[i], i]:
                surface_id[i] = surface_id[i] + 1
                K[i] = erodibility_id[surface_id[i]]
                erosion_cum[i] = 0
        
        # record river erosion for cosmo
        if sum(nsteps)-steps < steps_cosmo:
            o += 1
            if o <= np.shape(erosion_river)[0]:
                erosion_river[o-1,:] = (U[l,:]+((z00-z1)/dt))*1e3
                erosion_river[o-1,0] = U_final[l-1]*1e3
                erosion_hillslope[o-1,:,:] = (U_hill+((hill0-hillz)/dt))*1e3
            
        z00 = z1
        hill0 = hillz
        
# save final results          
elevation_river = z1
elevation_hillslope = hillz
erosionatnode = erosion_record[1,:]

#%% CHANNEL HEAD PROPERTIES

# initiate variable for channel head erosion
n_nodes = len(basin_data['x'])
n_sources = len(np.unique(basin_data['source']))
channel_top_ind = np.zeros(n_sources).astype(int)
neighbor_top_ind = np.zeros(n_sources).astype(int)
flow_distance_dif = np.zeros(n_sources)
erosion_rate_diff = np.zeros((n_sources, int(np.sum(nsteps))))
distance_dif = np.zeros(n_sources)

# find index of the more upstream node of each tributaries
cpt1 = 0
for i in range(n_nodes-1):
    if basin_data['source'][i] != basin_data['source'][i+1]:
        channel_top_ind[cpt1] = i
        cpt1 = cpt1 + 1
channel_top_ind[-1] = n_nodes-1

# find index of the closest node
for i in range(n_sources):
    lat1 = basin_data['latitude'][channel_top_ind[i]]
    lat2 = np.delete(basin_data['latitude'][channel_top_ind], i)
    lon1 = basin_data['longitude'][channel_top_ind[i]]
    lon2 = np.delete(basin_data['longitude'][channel_top_ind], i)
    index1 = np.argmin(np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2))
    neighbor_top_ind[i] = np.delete(channel_top_ind, i)[index1]
    
# calculate difference in flow distance and erosion rate
flow_distance_dif = basin_data['x'][channel_top_ind] - basin_data['x'][neighbor_top_ind]
for i in range(n_sources):
    if flow_distance_dif[i] > 0:
        erosion_rate_diff[i] = erosion_record[channel_top_ind[i],:] - erosion_record[neighbor_top_ind[i], :]
    else:
        erosion_rate_diff[i] = erosion_record[neighbor_top_ind[i], :] - erosion_record[channel_top_ind[i], :]  
# erosion_rate_diff = erosion_record[channel_top_ind,:] - erosion_record[neighbor_top_ind,:]

# calculate difference in distance
for i in range(n_sources):
    distance_dif[i] = haversine(basin_data['latitude'][channel_top_ind[i]], basin_data['longitude'][channel_top_ind[i]], basin_data['latitude'][neighbor_top_ind[i]], basin_data['longitude'][neighbor_top_ind[i]])

# calculate timescale between channel haid pair
timescale = abs(np.argmax(erosion_rate_diff, axis=1) - np.argmin(erosion_rate_diff, axis=1))

# find hillslope top of channel head
hillslope_top  = np.flip(hillslope_record[0, np.append(channel_top_ind, 0), :], axis=0)
flow_top = np.flip(basin_data['x'][np.append(channel_top_ind, 0)])
sort = np.argsort(flow_top)
flow_top = flow_top[sort]
hillslope_top = hillslope_top[sort]


#%% FIGURE: CHANNEL HEAD EROSION DIFFERENCE

cn = 7 # channel head pair to plot
fig = plt.figure(figsize=(12,4))
gs = fig.add_gridspec(10,10)
ax1 = fig.add_subplot(gs[0:10,0:3])
ax2 = fig.add_subplot(gs[0:10,3:10])

ax1.plot(basin_data['longitude'], basin_data['latitude'], ls='', marker='o', ms=4, mew=0.75, mec='k', mfc='lightsteelblue', zorder=1)
ax1.plot(basin_data['longitude'][channel_top_ind], basin_data['latitude'][channel_top_ind], ls='', marker='o', ms=6, mec='k', mfc='steelblue', zorder=2)
ax1.plot(basin_data['longitude'][channel_top_ind[cn]], basin_data['latitude'][channel_top_ind[cn]], ls='', marker='o', ms=7, mec='k', mfc='darkred', zorder=3)
ax1.plot(basin_data['longitude'][neighbor_top_ind[cn]], basin_data['latitude'][neighbor_top_ind[cn]], ls='', marker='o', ms=7, mec='k', mfc='darkred', zorder=3)

ax2.axhline(y=sample[0], ls='--', color='k', zorder=1)
ax2.axhline(y=-sample[0], ls='--', color='k', zorder=1)
ax2.plot(np.arange(np.sum(nsteps))*dt/1e6, erosion_rate_diff[cn,:], color='darkred', zorder=2)

ax2.set_xlim([0, param.tt])
ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')
ax2.set_xlabel('Time (Myrs)')
ax2.set_ylabel('Erosion rate difference (mm.yr$^{-1}$)')

fig.tight_layout()
# fig.savefig(dirname + '/figure/' + 'channel-head-erosion-difference-n11' + '.png', dpi=720)
# fig.savefig(dirname + '/figure/' + 'channel-head-erosion-difference-n11' + '.pdf', dpi=720)

#%% FIGURE: CHANNEL HEAD EROSION TIMESCALE

index2 = np.argwhere(timescale < 10000)
timescale2, flow_distance_dif2, distance_dif2 = timescale[index2], flow_distance_dif[index2], distance_dif[index2]
fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
s1 = ax1.scatter(abs(flow_distance_dif2)/1000, timescale2/1000, c=distance_dif2/1000, s=65, edgecolor='k', cmap='jet')
fig.colorbar(s1, label='Distance (km)')

ax1.set_xlabel('Flow distance difference (km)')
ax1.set_ylabel('Channel head timescale (Myrs)')

fig.tight_layout()
# fig.savefig(dirname + '/figure/' + 'channel-head-timescale-vs-flow-distance' + '.png', dpi=720)
# fig.savefig(dirname + '/figure/' + 'channel-head-timescale-vs-flow-distance' + '.pdf', dpi=720)

#%% FIGURE: LITHO+ELEVATION+EROSION IN MAP VIEW + PROFILE

times = np.round(np.linspace(0, param.tt, int(np.sum(nsteps))), decimals=1)
colors = ['indianred', 'steelblue', 'peru', 'plum']
cmap = ListedColormap(colors)
cmap2 = moc.ColormapExt('wiki-schwarzwald-d050').colormap
erosion_scale = 0.75
norm1 = mcolors.PowerNorm(gamma=1.75, vmin=0, vmax=np.max(erosion_record))

# create figure frame
fig = plt.figure(figsize=(7.125, 7.125))
gs = fig.add_gridspec(20,20)
ax1 = fig.add_subplot(gs[0:12,0:9])
ax2 = fig.add_subplot(gs[0:12,9:20])
ax3 = fig.add_subplot(gs[12:20,0:20])

# plot legend for third axis
custom_legend = []
for i in range(len(erodibility_id)):
    custom_legend.append(Line2D([0], [0], color=colors[i], ls='', marker='o', mew=0.5, mec='k', label='Rock ID=' + str(i) + ', K=' + str(erodibility_id[i])))
ax3.legend(handles=custom_legend, loc='upper left', fancybox=False, edgecolor='black', fontsize=10, labelspacing=0.3)

# plot colorbar for second axis
fig.colorbar(ScalarMappable(cmap='rainbow', norm=plt.Normalize(0,np.max(erosion_record)*erosion_scale)), ax=ax2, label='Erosion (mm.yr$^{-1}$)')

# change first axis format
ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')
ax1.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('top')

# change second axis format
ax2.set_xlabel('Longitude (°)')
ax2.xaxis.set_label_position('top')
ax2.xaxis.set_ticks_position('top')
ax2.set_yticklabels([])

# change third axis format
ticks_x = plticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
ax3.xaxis.set_major_formatter(ticks_x)
ax3.set_xlabel('Flow distance (km)')
ax3.set_ylabel('Elevation (m)')
ax3.set_ylim([0,1000])

fig.subplots_adjust(top=0.925, bottom=0.085, left=0.115, right=0.925, wspace=1, hspace=1)

def update(frame):
    
    d = basin_data['x']
    x = basin_data['longitude']
    y = basin_data['latitude']
    z = elevation_record[:,frame]
    h = hillslope_top[:,frame]
    r = surface_id_record[:,frame]
    e = erosion_record[:,frame]
    t = times[frame]
    
    f1 = np.polyfit(x=flow_top, y=hillslope_top[:,frame], deg=8)
    t1 = np.poly1d(f1)
 
    scatter1 = ax1.scatter(x=x, y=y, c=r, s=35, marker='o', cmap=cmap, vmin=0, vmax=len(colors), linewidths=0.33, edgecolors='k')
    scatter2 = ax2.scatter(x=x, y=y, c=e, s=35, marker='o', cmap='rainbow', vmin=0, vmax=np.max(erosion_record)*erosion_scale, linewidths=0.33, edgecolors='k')
    scatter3 = ax3.scatter(x=d, y=z, c=r, s=35, marker='o', cmap=cmap, vmin=0, vmax=len(colors), linewidths=0.33, edgecolors='k')
    line = ax3.plot(flow_top, t1(flow_top), ls='-', color='k')[0]
    text = ax1.text(0.5, 0.03, s='Time: ' + str(t) + ' Myrs', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    
    return scatter1, scatter2, scatter3, line, text,

animation = FuncAnimation(fig, update, frames=np.arange(0, steps, 100), repeat=False, blit=True)
    
#%% FIGURE: LITHO+ELEVATION+EROSION IN MAP VIEW + PROFILE

# plt.ioff()
# j = 0
# for i in range(0, steps, 100):
    
#     times = np.round(np.linspace(0, param.tt, int(np.sum(nsteps))), decimals=1)
#     colors = ['indianred', 'steelblue', 'peru', 'plum']
#     cmap = ListedColormap(colors)
#     cmap2 = moc.ColormapExt('wiki-schwarzwald-d050').colormap

#     # create figure frame
#     fig = plt.figure(figsize=(7.125, 7.125))
#     gs = fig.add_gridspec(20,20)
#     ax1 = fig.add_subplot(gs[0:12,0:9])
#     ax2 = fig.add_subplot(gs[0:12,9:20])
#     ax3 = fig.add_subplot(gs[12:20,0:20])
    
#     d = basin_data['x']
#     x = basin_data['longitude']
#     y = basin_data['latitude']
#     z = elevation_record[:,i]
#     r = surface_id_record[:,i]
#     e = erosion_record[:,i]
#     t = times[i]
 
#     scatter1 = ax1.scatter(x=x, y=y, c=r, s=35, marker='o', cmap=cmap, vmin=0, vmax=len(colors), linewidths=0.33, edgecolors='k')
#     scatter2 = ax2.scatter(x=x, y=y, c=e, s=35, marker='o', cmap='rainbow', vmin=0, vmax=np.max(erosion_record)*0.66, linewidths=0.33, edgecolors='k')
#     scatter3 = ax3.scatter(x=d, y=z, c=r, s=35, marker='o', cmap=cmap, vmin=0, vmax=len(colors), linewidths=0.33, edgecolors='k')
#     text = ax1.text(0.5, 0.03, s='Time: ' + str(t) + ' Myrs', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    
#     # plot legend for third axis
#     custom_legend = []
#     for i in range(len(erodibility_id)):
#         custom_legend.append(Line2D([0], [0], color=colors[i], ls='', marker='o', mew=0.5, mec='k', label='Rock ID=' + str(i) + ', K=' + str(erodibility_id[i])))
#     ax3.legend(handles=custom_legend, loc='upper left', fancybox=False, edgecolor='black', fontsize=10, labelspacing=0.3)

#     # plot colorbar for second axis
#     fig.colorbar(ScalarMappable(cmap='rainbow', norm=plt.Normalize(0,np.max(erosion_record)*0.66)), ax=ax2, label='Erosion (mm.yr$^{-1}$)')

#     # change first axis format
#     ax1.set_xlabel('Longitude (°)')
#     ax1.set_ylabel('Latitude (°)')
#     ax1.xaxis.set_label_position('top')
#     ax1.xaxis.set_ticks_position('top')

#     # change second axis format
#     ax2.set_xlabel('Longitude (°)')
#     ax2.xaxis.set_label_position('top')
#     ax2.xaxis.set_ticks_position('top')
#     ax2.set_yticklabels([])

#     # change third axis format
#     ticks_x = plticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
#     ax3.xaxis.set_major_formatter(ticks_x)
#     ax3.set_xlabel('Flow distance (km)')
#     ax3.set_ylabel('Elevation (m)')
#     ax3.set_ylim([0,1000])

#     fig.subplots_adjust(top=0.925, bottom=0.085, left=0.115, right=0.925, wspace=1, hspace=1)
    
#     if j<10:
#         fig.savefig(dirname + '/figure/video/output_t00' + str(j) +'.png', dpi=420)
#     if j>=10 and j<100:
#         fig.savefig(dirname + '/figure/video/output_t0' + str(j) +'.png', dpi=420)
#     if j>=100:
#         fig.savefig(dirname + '/figure/video/output_t' + str(j) +'.png', dpi=420)
        
#     j=j+1