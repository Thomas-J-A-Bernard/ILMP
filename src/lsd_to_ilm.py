import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from  geologic_functions import Set_Lithology_Name, Set_Uplift

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

def Lsd_to_Ilm(filename, set_tectonic=False, set_lithology=False, resolution_distance=1000, threshold_method='distance', threshold_distance=10000, threshold_order=4):
    '''
    DESCRIPTION:
        Transform the topographic extraction from LSDTopoTools in order to be read by the ILMP code during creation of the basin dictionnary
    ----------
    PARAMETERS:
    filename : string
        name of the lsdtopotools file
    set_tectonic : bool
        apply a tectonic field if true
    set_lithology: bool
        apply a lithology name based on the lithology keys if true
    resolution_distance : float
        resolution for the point resampling in meter
    threshold_method : string
        apply a specific threshold method: 
            distance: select river tributaries based on their distances
            stream_order: select river tributaries based on their stream order (strahler stream order)
    threshold_distance : float
        minimum threshold distance of the tributaries in order to be kept in meter
    threshold_order :  int
        minimum stream order of the tributaries in order to be kept 
    -------
    RETURNS:
    latitude : array of float
        latitude of the points
    longitude : array of float
        longitude of the points
    flow_distance : array of float
        flow distance of the points
    elevation : array of float
        elevation of the points
    area : array of float
        area of the points
    source : array of float
        source (sub-basin tributaries) of the points
    lithology : array of float
        lithology key of the points
    block : array of float
        block key of the points
    pairs : array of float
        pairs (indicate where sub-basin tributaries are connected) of the points
    latitude_r : array of float
        latitude of the points after resampling
    longitude_r : array of float
        longitude of the points after resampling
    flow_distance_r : array of float
        flow distance of the points after resampling
    elevation_r : array of float
        elevation of the points after resampling
    area_r : array of float
        area of the points after resampling 
    source_r : array of float
        source (sub-basin tributaries) of the points after resampling
    lithology_r : array of float
        lithology key of the points after resampling
    block_r : array of float
        block key of the points after resampling
    pairs_r : array of float
        pairs (indicate where sub-basin tributaries are connected) of the points after resampling
    '''
    
    # read csv file
    print('Import LSDTopoTools file: ...')
    lsd_data = pd.read_csv(filename, engine='python')
    print('Done')
    
    if threshold_method == 'stream_order':
    
        # stream_order_max = np.max(lsd_data['stream_order'])
        # stream_order_remove = stream_order_max - threshold_order
        # lsd_data = lsd_data.drop(lsd_data[lsd_data.stream_order <= stream_order_remove].index)
        
        lsd_data = lsd_data.drop(lsd_data[lsd_data.stream_order < threshold_order].index)
    
    # define the different river segment keys of the catchment
    source_key = np.unique(lsd_data[['source_key']])
    n_skey = np.size(source_key)
    
    # initialize parameter arrays
    elevation = np.array([])
    flow_distance = np.array([])
    area = np.array([])
    latitude = np.array([])   
    longitude = np.array([])
    source = np.array([])
    lithology = np.array([])
    block = np.array([])
    sorder = np.array([])
    chi = np.array([])
    
    # intialize resampled parameter arrays
    elevation_r = np.array([])
    flow_distance_r = np.array([])
    area_r = np.array([])
    latitude_r = np.array([])
    longitude_r = np.array([])
    source_r = np.array([])
    lithology_r = np.array([])
    block_r = np.array([])
    sorder_r = np.array([])
    chi_r = np.array([])
    
    # build parameters array based on their segment source keys
    for i in source_key:
        # find data with the same segment key
        lsd_elevation = np.squeeze(np.array(lsd_data.loc[lsd_data['source_key'] == i][['elevation']]))
        lsd_flow_distance = np.squeeze(np.array(lsd_data.loc[lsd_data['source_key'] == i][['flow_distance']]))
        lsd_area = np.squeeze(np.array(lsd_data.loc[lsd_data['source_key'] == i][['drainage_area']]))
        lsd_latitude = np.squeeze(np.array(lsd_data.loc[lsd_data['source_key'] == i][['latitude']]))
        lsd_longitude = np.squeeze(np.array(lsd_data.loc[lsd_data['source_key'] == i][['longitude']]))
        lsd_source = np.squeeze(np.array(lsd_data.loc[lsd_data['source_key'] == i][['source_key']]))   
        if set_lithology:
            lsd_lithology = np.squeeze(np.array(lsd_data.loc[lsd_data['source_key'] == i][['lithology']]))
        if set_tectonic:
            lsd_block = np.squeeze(np.array(lsd_data.loc[lsd_data['source_key'] == i][['tectonic']]))
        lsd_sorder = np.squeeze(np.array(lsd_data.loc[lsd_data['source_key'] == i][['stream_order']]))
        lsd_chi = np.squeeze(np.array(lsd_data.loc[lsd_data['source_key'] == i][['chi']]))
        
        # set croissant elevation
        lsd_elevation = np.flip(lsd_elevation)
        lsd_flow_distance = np.flip(lsd_flow_distance)
        lsd_area = np.flip(lsd_area)
        lsd_latitude = np.flip(lsd_latitude)
        lsd_longitude = np.flip(lsd_longitude)
        lsd_source = np.flip(lsd_source)
        if set_lithology:
            lsd_lithology = np.flip(lsd_lithology)
        if set_tectonic:
            lsd_block = np.flip(lsd_block)
        lsd_sorder = np.flip(lsd_sorder)
        lsd_chi = np.flip(lsd_chi)
        
        # remove data if there is only one point in river segment
        if np.size(lsd_elevation) != 1:
            # calcul the distance of the river segment
            segment_distance = lsd_flow_distance[-1] - lsd_flow_distance[0]
            if threshold_method == 'distance':
                # remove data if river segment is too short
                if segment_distance > threshold_distance:
                    print('processing tributary: ' + str(i))
                    # resample the river profile data
                    flow_distance_new = np.arange(lsd_flow_distance[0], lsd_flow_distance[-1], resolution_distance)
                    f = interp1d(lsd_flow_distance, lsd_elevation)
                    elevation_new = f(flow_distance_new)
                    f = interp1d(lsd_flow_distance, lsd_area)
                    area_new = f(flow_distance_new)
                    f = interp1d(lsd_flow_distance, lsd_latitude)
                    latitude_new = f(flow_distance_new)
                    f = interp1d(lsd_flow_distance, lsd_longitude)
                    longitude_new = f(flow_distance_new)
                    f = interp1d(lsd_flow_distance, lsd_chi)
                    chi_new = f(flow_distance_new)
                    # no interpretation for the source, only set the same source number
                    source_new = np.zeros(np.shape(flow_distance_new))
                    source_new[:] = i
                    # no interpretation for the lithology, uplift and stream order, find the closest initial point and set the same lithology and uplift
                    if set_lithology:
                        lithology_new = np.zeros(np.shape(flow_distance_new))
                    if set_tectonic:
                        block_new = np.zeros(np.shape(flow_distance_new))
                    sorder_new = np.zeros(np.shape(flow_distance_new))
                    for j in range(np.size(flow_distance_new)):
                        loc = np.argmin(abs(lsd_flow_distance[:] - flow_distance_new[j]))
                        if set_lithology:
                            lithology_new[j] = lsd_lithology[loc]
                        if set_tectonic:
                            block_new[j] = lsd_block[loc]
                        sorder_new[j] = lsd_sorder[loc]
                    # store data
                    elevation = np.append(elevation, lsd_elevation)
                    flow_distance = np.append(flow_distance, lsd_flow_distance)
                    area = np.append(area, lsd_area)
                    latitude = np.append(latitude, lsd_latitude)
                    longitude = np.append(longitude, lsd_longitude)
                    source = np.append(source, lsd_source)
                    if set_lithology:
                        lithology = np.append(lithology, lsd_lithology)
                    if set_tectonic:
                        block = np.append(block, lsd_block)
                    sorder = np.append(sorder, lsd_sorder)
                    chi = np.append(chi, lsd_chi)
                    
                    elevation_r = np.append(elevation_r, elevation_new)
                    flow_distance_r = np.append(flow_distance_r, flow_distance_new)
                    area_r = np.append(area_r, area_new)
                    latitude_r = np.append(latitude_r, latitude_new)
                    longitude_r = np.append(longitude_r, longitude_new)
                    source_r = np.append(source_r, source_new)
                    if set_lithology:
                        lithology_r = np.append(lithology_r, lithology_new)
                    if set_tectonic:
                        block_r = np.append(block_r, block_new)
                    sorder_r = np.append(sorder_r, sorder_new)
                    chi_r = np.append(chi_r, chi_new)
                    
            else:
                if segment_distance > resolution_distance:
                    print('processing tributary: ' + str(i))
                    # resample the river profile data
                    flow_distance_new = np.arange(lsd_flow_distance[0], lsd_flow_distance[-1], resolution_distance)
                    f = interp1d(lsd_flow_distance, lsd_elevation)
                    elevation_new = f(flow_distance_new)
                    f = interp1d(lsd_flow_distance, lsd_area)
                    area_new = f(flow_distance_new)
                    f = interp1d(lsd_flow_distance, lsd_latitude)
                    latitude_new = f(flow_distance_new)
                    f = interp1d(lsd_flow_distance, lsd_longitude)
                    longitude_new = f(flow_distance_new)
                    f = interp1d(lsd_flow_distance, lsd_chi)
                    chi_new = f(flow_distance_new)
                    # no interpretation for the source, only set the same source number
                    source_new = np.zeros(np.shape(flow_distance_new))
                    source_new[:] = i
                    # no interpretation for the lithology, uplift and stream order, find the closest initial point and set the same lithology and uplift
                    if set_lithology:
                        lithology_new = np.zeros(np.shape(flow_distance_new))
                    if set_tectonic:
                        block_new = np.zeros(np.shape(flow_distance_new))
                    sorder_new = np.zeros(np.shape(flow_distance_new))
                    for j in range(np.size(flow_distance_new)):
                        loc = np.argmin(abs(lsd_flow_distance[:] - flow_distance_new[j]))
                        if set_lithology:
                            lithology_new[j] = lsd_lithology[loc]
                        if set_tectonic:
                            block_new[j] = lsd_block[loc]
                        sorder_new[j] = lsd_sorder[loc]
                    # store data
                    elevation = np.append(elevation, lsd_elevation)
                    flow_distance = np.append(flow_distance, lsd_flow_distance)
                    area = np.append(area, lsd_area)
                    latitude = np.append(latitude, lsd_latitude)
                    longitude = np.append(longitude, lsd_longitude)
                    source = np.append(source, lsd_source)
                    if set_lithology:
                        lithology = np.append(lithology, lsd_lithology)
                    if set_tectonic:
                        block = np.append(block, lsd_block)
                    sorder = np.append(sorder, lsd_sorder)
                    chi = np.append(chi, lsd_chi)
                    
                    elevation_r = np.append(elevation_r, elevation_new)
                    flow_distance_r = np.append(flow_distance_r, flow_distance_new)
                    area_r = np.append(area_r, area_new)
                    latitude_r = np.append(latitude_r, latitude_new)
                    longitude_r = np.append(longitude_r, longitude_new)
                    source_r = np.append(source_r, source_new)
                    if set_lithology:
                        lithology_r = np.append(lithology_r, lithology_new)
                    if set_tectonic:
                        block_r = np.append(block_r, block_new)
                    sorder_r = np.append(sorder_r, sorder_new)
                    chi_r = np.append(chi_r, chi_new)
    
    # start distance to zeros
    flow_distance = flow_distance - flow_distance[0]
    flow_distance_r = flow_distance_r - flow_distance_r[0]
    
    # initialize pairs array (where river points are connected between others)
    pairs = np.zeros((np.shape(elevation)[0]-1, 2))
    pairs_r = np.zeros((np.shape(elevation_r)[0]-1, 2))
    
    # build pairs array
    pairs[0,0], pairs[0,1] = 1, 2
    for i in range(1, np.shape(pairs)[0]):
        # if elevation increase (same river) point is connected with the following one
        pairs[i,0] = i+1
        pairs[i,1] = i+2
        # if elevation decrease (other river tributary) point is not connected to the following one
        if elevation[i] < elevation[i-1]:
            # set coordinate of the lowest point of the new river tributary
            latitude_temp, longitude_temp = latitude[i], longitude[i]
            # search closet point in the previous rivers 
            loc = np.argmin(abs(latitude[0:i-1] - latitude_temp) + abs(longitude[0:i-1] - longitude_temp))
            # loc new river tributary to previous river
            pairs[i-1,0] = loc+1
    
    # build resampled pairs array
    pairs_r[0,0], pairs_r[0,1] = 1, 2
    for i in range(1, np.shape(pairs_r)[0]):
        # if source is the same (same tributary) point is connected to the following one
        pairs_r[i,0] = i+1
        pairs_r[i,1] = i+2
        # if source is diferent (different tributary) point is not connected to the following one
        if source_r[i] != source_r[i-1]:
            
            # set coordinate of the lowest point of the new river tributary
            #latitude_temp_r, longitude_temp_r = latitude_r[i], longitude_r[i]
            # search closet point in the previous rivers 
            #loc = np.argmin(np.sqrt(abs(latitude_r[0:i-1] - latitude_temp_r)**2 + abs(longitude_r[0:i-1] - longitude_temp_r)**2))
            # loc new river tributary to previous river
            #pairs_r[i-1,0] = loc+1
            
            distance1 = 1000000
            for j in range(len(flow_distance_r)):
                if flow_distance_r[j] < flow_distance_r[i]:
                    distance2 = np.sqrt(abs(latitude_r[j] - latitude_r[i])**2 + abs(longitude_r[j] - longitude_r[i])**2)
                    if distance2 < distance1:
                        distance1 = distance2
                        loc = j
            
            pairs_r[i-1,0] = loc+1
            
    # if set_lithology == False:
        
            
    return latitude, longitude, flow_distance, elevation, area, source, lithology, block, sorder, chi, pairs, latitude_r, longitude_r, flow_distance_r, elevation_r, area_r, source_r, lithology_r, block_r, sorder_r, chi_r, pairs_r
   
if __name__ ==  '__main__':

    # get lsdtopotools data file
    basin = 'neckar'
    lsd_file = home_dirname + '/data/basins/' + basin + '/' + 'neckar_chi_data_map_burned.csv'
    
    # set values for resampling
    resolution_distance = 3000
    threshold_distance = 20000
    
    # transform lsdtopotools dataset
    latitude, longitude, flow_distance, elevation, area, source, lithology, block, sorder, chi, pairs, latitude_r, longitude_r, flow_distance_r, elevation_r, area_r, source_r, lithology_r, block_r, sorder_r, chi_r, pairs_r = Lsd_to_Ilm(lsd_file, set_tectonic=True, set_lithology=True, resolution_distance=resolution_distance, threshold_method='distance', threshold_distance=threshold_distance)
    
