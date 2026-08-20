import json
import pickle
import numpy as np
from pathlib import Path
from lsd_to_ilm import Lsd_to_Ilm
from geologic_functions import Set_Lithology_Name, Set_Uplift

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

def DictionnaryExport(path, dictionnary):
    '''
    DESCRIPTION:
        Export a dictionnary as a pickle file
    ----------
    PARAMETERS
    path : string
        path where the pickle file is save
    dictionnary : dict
        dictionnary to export
    -------
    RETURNS
    None.
    '''
    
    # export dictionnary as pkl file
    with open(path, 'wb') as f:
        pickle.dump(dictionnary, f)

def DictionnaryImport(path):
    '''
    DESCRIPTION:
        Import a pickle file as a dictionnary
    ----------
    PARAMETERS:
    path : string
        path where the pickle file is upload
    -------
    RETURNS:
    dictionnary : dict
        dictionnary imported
    '''
    # import dictionnary 
    with open(path, 'rb') as f:
        dictionnary = pickle.load(f)
        
    return dictionnary

def Remove_River_Litho(basin_data):
    '''
    DESCRIPTION:
        remove the river lithology and replace it by the other closest lithology
    ----------
    PARAMETERS
    basin_data : dict
        dictionnary with the basin dataset
    -------
    RETURNS
    None.
    '''
    
    litho = np.copy(basin_data['lithology'])
    new_litho = np.copy(basin_data['lithology'])
    lat = basin_data['latitude']
    lon = basin_data['longitude']
    litho_river_index = np.argwhere(litho == 'river')
    litho_not_river_index = np.argwhere(litho != 'river')

    for i in range(len(litho_river_index)):
        # 
        loc1 = np.argmin(np.sqrt(abs(lat[litho_not_river_index] - lat[litho_river_index[i]])**2 + abs(lon[litho_not_river_index] - lon[litho_river_index[i]])**2))
        index1 = litho_not_river_index[loc1]
        litho_rep = litho[index1]
        new_litho[litho_river_index[i]] = litho_rep
        
    basin_data['lithology'] = new_litho

def CatchmentDictionnaryCreation(lsd_file_name, set_tectonic=False, set_lithology=False, resolution_distance=1000, threshold_method='distance', threshold_distance=10000, threshold_order=4):
    '''
    DESCRIPTION: 
        create the basin dictionnary containing all the dataset from the lsdtopotools extraction
    ----------
    PARAMETERS
    lsd_file_name : string
        name of the lsdtopotools file
    set_tectonic: bool
        apply a tectonic field if true
    set_lithology : bool
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
    RETURNS
    catchment : dict
        catchment dataset
    '''
    
    # import lsd dataset
    latitude, longitude, x, elevation, area, source, lithology, block, sorder, chi, pairs, latitude_r, longitude_r, x_r, elevation_r, area_r, source_r, lithology_r, block_r, sorder_r, chi_r, pairs_r = Lsd_to_Ilm(lsd_file_name, set_tectonic, set_lithology, resolution_distance, threshold_method, threshold_distance, threshold_order)
    
    # set lithology
    if set_lithology:
        lithology_name_r = Set_Lithology_Name(lithology_r)
    
    # build dictionnary based on the 
    catchment = {}
    catchment['latitude'] = latitude_r
    catchment['longitude'] = longitude_r
    catchment['x'] = x_r
    catchment['initial_elevation'] = elevation_r
    catchment['area'] = area_r
    catchment['source'] = source_r
    if set_lithology:
        catchment['lithology'] = lithology_name_r
    if set_tectonic:
        catchment['block'] = block_r
    catchment['pairs'] = pairs_r
    catchment['cosmo_meas'] = {}
    catchment['thermo_meas'] = {}
    catchment['dx_dem'] = resolution_distance
    catchment['sorder'] = sorder_r
    catchment['skipping_factor'] = 1
    catchment['capture'] = {}
    catchment['base_level_drop'] = {}
    catchment['chi'] = chi_r
    
    # replace river lithology name
    if set_lithology:
        Remove_River_Litho(catchment)
    
    return catchment

if __name__ == '__main__':
    
    basin = 'neckar'
    
    catchment = CatchmentDictionnaryCreation(home_dirname + '/data/basins/' + basin + '/' + basin + '_chi_data_map_burned.csv',
                                             set_tectonic=True, set_lithology=True,
                                             resolution_distance=3000, threshold_method='distance', threshold_distance=20000, threshold_order=3)
    
    DictionnaryExport(home_dirname + '/data/basins/' + basin + '/' + basin + '-basin_r3_t20.pkl', catchment)
    