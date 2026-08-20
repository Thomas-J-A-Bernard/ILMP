import numpy as np
import pandas as pd
from pathlib import Path
from general_functions import Find_Upstream_Index, Find_Downstream_Index

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

def Set_Erodibility(lithology):
    '''
    DESCRIPTION:
        Apply an erodibility coefficient to points of the channel river based on their lithology key
    ----------
    PARAMETERS:
    lithology : array of int
        litholoy key of the points
    -------
    RETURNS
    erodibility : array of float
        erodibility of the points
    '''
    
    # import csv file for erodibility coefficient
    #erodibilitykey = pd.read_csv(dirname + "\\others\\erodibilitykey.csv", engine='python')
    erodibilitykey = pd.read_csv(home_dirname + "/data/keys/erodibilitykey.csv", engine='python')
    
    s = np.shape(lithology)[0]
    erodibility = np.zeros(s)
    for i in np.unique(lithology):
        loc = np.where(lithology == i)
        value = erodibilitykey[erodibilitykey['rocktype'] == i]['coefficient']
        erodibility[loc] = value
        
    return erodibility

def Set_Lithology_Name(lithology):
    '''
    DESCRIPTION:
        Apply a lithology name to points of the channel river based on their lithology key
    ----------
    PARAMETERS:
    lithology : array of int
        litholoy key of the points
    -------
    RETURNS
    lithology_name: array of float
        lithology name of the points
    '''
    
    # import csv file for erodibility coefficient
    #erodibilitykey = pd.read_csv(dirname + "\\others\\erodibilitykey.csv", engine='python')
    erodibilitykey = pd.read_csv(home_dirname + "/data/keys/GK1000-1GE_LCC_v1_new_lithokey.csv", engine='python')
    
    s = np.shape(lithology)[0]
    lithology_name = np.zeros(s).astype(str)
    for i in np.unique(lithology):
        loc = np.where(lithology == i)
        
        if i == 0:
            lithology_name[loc] = 'river'
        elif i == -9999:
            lithology_name[loc] = 'unknow'
        else:
            name = erodibilitykey[erodibilitykey['rocktype'] == i]['lithology']
            lithology_name[loc] = name.values[0]
        
    return lithology_name

def Set_Uplift(block):
    '''
    DESCRIPTION:
        Apply an uplift to points of the channel river based on their block key
    ----------
    PARAMETERS:
    lithology : array of int
        block key of the points
    -------
    RETURNS
    erodibility : array of float
        uplift of the points
    '''
    
    # import csv file for erodibility coefficient
    #upliftkey = pd.read_csv(dirname + "\\others\\upliftkey.csv", engine='python')
    upliftkey = pd.read_csv(home_dirname + "/data/keys/upliftkey.csv", engine='python')
    
    s = np.shape(block)[0]
    uplift = np.zeros(s)
    for i in np.unique(block):
        loc = np.where(block == i)
        value = upliftkey[upliftkey['block'] == i]['uplift']
        uplift[loc] = value
        
    return uplift

def Block_to_Uplift(basin_data, block1=0.05, block2=0.05, block3=0.0025, block4=0.05, block5=0.05, block6=0.05, 
                    block7=0.05, block8=0.05, block9=0.05, block10=0.05, block11=0.05):
    '''
    DESCRIPTION:
        create a new item of the basin dictionnary and set a default uplift to the different block of the basin
    ----------
    PARAMETERS
    basin_data : dictionnary
        information of the basin
    block1 : float, optional
        uplift of the block 1. The default is 0.05
    block2 : float, optional
        uplift of the block 2. The default is 0.05
    block3 : float, optional
        uplift of the block 3. The default is 0.0025
    block4 : float, optional
        uplift of the block 4. The default is 0.05
    block5 : float, optional
        uplift of the block 5. The default is 0.05
    block6 : float, optional
        uplift of the block 6. The default is 0.05
    block7 : float, optional
        uplift of the block 7. The default is 0.05
    block8 : float, optional
        uplift of the block 8. The default is 0.05
    block9 : float, optional
        uplift of the block 9. The default is 0.05
    block10 : float, optional
        uplift of the block 10. The default is 0.05
    block11 : float, optional
        uplift of the block 11. The default is 0.05
    -------
    RETURNS
    None
    '''
    
    block = basin_data['block']
    uplift = np.zeros(len(block))
    
    for i in range(len(block)):
        if block[i] == 1:
            uplift[i] = block1
        if block[i] == 2:
            uplift[i] = block2
        if block[i] == 3:
            uplift[i] = block3
        if block[i] == 4:
            uplift[i] = block4
        if block[i] == 5:
            uplift[i] = block5
        if block[i] == 6:
            uplift[i] = block6
        if block[i] == 7:
            uplift[i] = block7
        if block[i] == 8:
            uplift[i] = block8
        if block[i] == 9:
            uplift[i] = block9
        if block[i] == 10:
            uplift[i] = block10
    
    basin_data['uplift'] = uplift

def Tilting_to_Uplift(basin_data, param, direction='degree', uplift=[0.0], gradient=[0.0], degree=[0.0], time=[], spatial=False, block_ind=[], block_uplift=[]):
    '''
    DESCRIPTION:
        1) Apply tilting to the uplift field with a given gradient and direction. 2) You can change the uplift field 
        at specific time. In this case, the uplift, gradient and degree parameters have to be the same size while the
        time parameter have to be one size down. 3) You can change spatially the uplift depending you have a block key
        in your basin_data dictionnary
    ----------
    PARAMETERS:
    basin_data : dictionnary
        information of the basin and should contain 'latittude' and 'longitude' at least
    param : structure
        parameter for the model
    direction : string
        'east', 'west', 'north', 'south', or 'degree'. set the direction of the tilting
    uplift : list of float
        maximum uplift of the tilting
    gradient : float
        gradient of the tilting
    degree : float
        direction in degree if the tilting
    time : float
        time for the change of uplift
    RETURN:
    -------
    None
    '''
    
    s = len(uplift)
    
    lat = basin_data['latitude']
    lon = basin_data['longitude']

    lon_min = np.min(lon)
    lon_max = np.max(lon)
    lon_dif = lon_max - lon_min

    lat_min = np.min(lat)
    lat_max = np.max(lat)
    lat_dif = lat_max - lat_min
    
    U = np.zeros([s, len(lat)])

    if direction == 'east':
        a = abs((lon - lon_min)/lon_dif) * gradient
        U = uplift*a + uplift*abs(gradient - 1)
        
    if direction == 'west':
        a = abs((lon - lon_min)/lon_dif - 1) * gradient
        U = uplift*a + uplift*abs(gradient - 1)
        
    if direction == 'north':
        a = abs((lat - lat_min)/lat_dif) * gradient
        U = uplift*a + uplift*abs(gradient - 1)
        
    if direction == 'south':
        a = abs((lat - lat_min)/lat_dif - 1) * gradient
        U = uplift*a + uplift*abs(gradient - 1)
        
    if direction == 'degree':
        
        lon_middle = (lon_min + lon_max)/2
        lat_middle = (lat_min + lat_max)/2
        
        for i in range(s):
        
            radian = degree[i]*np.pi/180
        
            lon_new = (lon - lon_middle)*np.cos(radian) - (lat - lat_middle)*np.sin(radian) + lon_middle
            lat_new = (lon - lon_middle)*np.sin(radian) + (lat - lat_middle)*np.cos(radian) + lat_middle
            
            lat_new_min = np.min(lat_new)
            lat_new_max = np.max(lat_new)
            lat_new_dif = lat_new_max - lat_new_min
        
            a = abs((lat_new - lat_new_min)/lat_new_dif - 1) * gradient[i]
            u = uplift[i]*a + uplift[i]*abs(gradient[i] - 1)
            
            U[i,:] = u
    
    param.nuv = 1
    param.ua = time
    basin_data['uplift'] = U
    
    if spatial == True:
        
        block = basin_data['block']
        
        for i in range(len(block_ind)):
            arg = np.argwhere(block == block_ind[i])
            basin_data['uplift'][:,arg] = block_uplift[i]
            
    

def Lithology_to_Erodibility(basin_data, param, sand=1e-6, clay=1e-6, carbonate=1e-6, peat=1e-6, silt=1e-6, diamicton=1e-6, residual_material=1e-6,
                             conglomerate=1e-6, impact_generated_material=1e-6, sandstone=1e-6, gravel=1e-6, limestone=1e-6, mudstone=1e-6,
                             claystone=1e-6, dolomite=1e-6, shale=1e-6, quartzite=1e-6, wacke=1e-6, plutonic=1e-6, marble=1e-6, metamorphic=1e-6,
                             volcanic=1e-6):
    '''
    DESCRIPTION:
        create a new item of the basin dictionnary and set a default erodibility to the different lithology of the basin
    ----------
    PARAMETERS
    basin_data : dictionnary
        information of the basin
    sand : float, optional
        erodibility of the sand. The default is 10e-6.
    clay : float, optional
        erodibility of the clay. The default is 10e-6.
    carbonate : float, optional
        erodibility of the carbonate. The default is 5e-6.
    peat : float, optional
        erodibility of the peat. The default is 10e-6.
    silt : float, optional
        erodibility of the silt. The default is 10e-6.
    diamicton : float, optional
        erodibility of the diamicton. The default is 10e-6.
    residual_material : float, optional
        erodibility of the residual material. The default is 10e-6.
    conglomerate : float, optional
        erodibility of the conglomerate. The default is 10e-6.
    impact_generated_material : float, optional
        erodibility of the impact generated material. The default is 10e-6.
    sandstone : float, optional
        erodibility of the sandstone. The default is 1e-6.
    gravel : float, optional
        erodibility of the gravel. The default is 10e-6.
    limestone : float, optional
        erodibility of the limestone. The default is 1e-6.
    mudstone : float, optional
        erodibility of the mudstone. The default is 1e-6.
    claystone : float, optional
        erodibility of the claystone. The default is 1e-6.
    dolomite : float, optional
        erodibility of the dolomite. The default is 1e-6.
    shale : float, optional
        erodibility of the shale. The default is 5e-6.
    quartzite : float, optional
        erodibility of the quartzite. The default is 0.1e-6.
    wacke : float, optional
        erodibility of the wacke. The default is 5e-6.
    plutonic : float, optional
        erodibility of the plutonic. The default is 0.5e-6.
    marble : float, optional
        erodibility of the marble. The default is 1e-6.
    metamorphic : float, optional
        erodibility of the metamorphic. The default is 0.5e-6.
    volcanic : float, optional
        erodibility of the volcanic. The default is 0.75e-6.
    -------
    RETURNS
    None.
    '''
    
    lithology = basin_data['lithology']
    erodibility = np.zeros(len(lithology))
    
    for i in range(len(lithology)):
        if lithology[i] == "sand":
            erodibility[i] = sand
        elif lithology[i] == "clay":
            erodibility[i] = clay
        elif lithology[i] == "carbonate":
            erodibility[i] = carbonate
        elif lithology[i] == "peat":
            erodibility[i] = peat
        elif lithology[i] == "silt":
            erodibility[i] = silt
        elif lithology[i] == "diamicton":
            erodibility[i] = diamicton
        elif lithology[i] == "residual_material":
            erodibility[i] = residual_material
        elif lithology[i] == "conglomerate":
            erodibility[i] = conglomerate
        elif lithology[i] == "impact_generated_material":
            erodibility[i] = impact_generated_material
        elif lithology[i] == "sandstone":
            erodibility[i] = sandstone
        elif lithology[i] == "gravel":
            erodibility[i] = gravel
        elif lithology[i] == "limestone":
            erodibility[i] = limestone
        elif lithology[i] == "mudstone":
            erodibility[i] = mudstone
        elif lithology[i] == "claystone":
            erodibility[i] = claystone
        elif lithology[i] == "dolomite":
            erodibility[i] = dolomite
        elif lithology[i] == "shale":
            erodibility[i] = shale
        elif lithology[i] == "quartzite":
            erodibility[i] = quartzite
        elif lithology[i] == "wacke":
            erodibility[i] = wacke
        elif lithology[i] == "plutonic":
            erodibility[i] = plutonic
        elif lithology[i] == "marble":
            erodibility[i] = marble
        elif lithology[i] == "metamorphic":
            erodibility[i] = metamorphic
        elif lithology[i] == "volcanic":
            erodibility[i] = volcanic
    
    param.kflag = 1
    basin_data['erodibility'] = erodibility
    
def Base_Level_Drop(basin_data, initial_level, drop_time):
    '''
    DESCRIPTION:
        set base-level information in the basin dictionnary
    ----------
    PARAMETERS:
    basin_data : dictionnary
        information of the basin
    initial_level : list
        list of the initial elevation of the base level prior of the drop
    drop_time : list
        list of the time of the base level drop
    -------
    RETURNS
    None.
    '''
    
    s = len(initial_level)
    for i in range(s):
        basin_data['base_level_drop']['drop' + str(i+1)] = {}
        basin_data['base_level_drop']['drop' + str(i+1)]['initial_level'] = initial_level[i]
        basin_data['base_level_drop']['drop' + str(i+1)]['time'] = drop_time[i]
        
def River_Capture(basin_data, node, time, initial_uplift):
    '''
    DESCRIPTION:
        set river capture events information in the basin dictionnary
    ----------
    PARAMETERS:
    basin_data : dictionnary
        information of the basin
    node : list
        list of the node where the capture occures
    time : list
        list of the time of the river captures
    initial_uplift:
        list of the initial uplift of the upstream river section prior to the capture
    -------
    RETURNS
    None.
    '''
    
    s = len(node)
    for i in range(s):
        basin_data['capture']['river' + str(i+1)] = {}
        basin_data['capture']['river' + str(i+1)]['node'] = int(node[i])
        basin_data['capture']['river' + str(i+1)]['time'] = time[i]
        basin_data['capture']['river' + str(i+1)]['initial_uplift'] = initial_uplift[i]
        
def GetSpecificElevation(basin_data, elevation):
    
    e = []
    s = np.unique(basin_data['source'])
    for i in range(len(s)):
        ind = np.where(basin_data['source'] == s[i])
        ele = elevation[ind]
        e.append(ele[0])
        e.append(ele[-1])
    e = np.array(e)
    
    return e
        
def Low_Temperature_Thermochronology(basin_data, elevation, lat, lon, ahea=[], ahea_error=[], ahea_type=[], ahea_radius=[], ahea_loc=[], afta=[], afta_error=[], aftmtl=[], aftmtl_error=[], node=[], name=[]):
    
    if name:
        basin_data['thermo_meas']['name'] = np.array(name)
      
    if node:
        basin_data['thermo_meas']['ind'] = np.array(node)
    else:
        s = len(elevation)
        basin_data['thermo_meas']['ind'] = np.zeros((s))
        lat1 = basin_data['latitude']
        lon1 = basin_data['longitude']
        lat2 = lat
        lon2 = lon
        for i in range(s):
            index = np.argmin(np.sqrt((lat1 - lat2[i])**2 + (lon1 - lon2[i])**2))
            basin_data['thermo_meas']['ind'][i] = index
    basin_data['thermo_meas']['ind'] = basin_data['thermo_meas']['ind'].astype(int)
    
    if ahea:
        basin_data['thermo_meas']['ahe'] = np.array(ahea)
    if ahea_error:
        basin_data['thermo_meas']['ahe_error'] = np.array(ahea_error)
    if ahea_type:
        basin_data['thermo_meas']['ahe_type'] = np.array(ahea_type)
    if ahea_radius:
        basin_data['thermo_meas']['ahe_radius'] = np.array(ahea_radius)
    if ahea_type:
        basin_data['thermo_meas']['ahe_loc'] = np.array(ahea_loc)
    
    if afta:
        basin_data['thermo_meas']['aft'] = np.array(afta)
    if afta_error:
        basin_data['thermo_meas']['aft_error'] = np.array(afta_error)
        
    if aftmtl:
        basin_data['thermo_meas']['aft_length'] = np.array(aftmtl)
    if aftmtl_error:
        basin_data['thermo_meas']['aft_length_sd'] = np.array(aftmtl_error)
        
    basin_data['thermo_meas']['elevation'] = np.array(elevation)
    basin_data['thermo_meas']['lat'] = np.array(lat)
    basin_data['thermo_meas']['lon'] = np.array(lon)

def Low_Temperature_Thermochronology2(basin_data, filename):
    
    # open filename
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # add text line to the basin dictionnary
    for line in lines:
        ls = line.strip()
        # ignore line starting with a hash
        if not ls.startswith('#'):
            key, value = line.strip().split(':')
            if key == 'name':
                value = [str(x) for x in value.split(',')]
            elif key == 'node':
                value = [int(x) for x in value.split(',')]
            else:
                value = [float(x) for x in value.split(',')]
            basin_data['thermo_meas'][key.strip()] = np.array(value)
    
    # use latitude and longitude to locate sample if ind is not provided
    if not 'node' in basin_data['thermo_meas']:
        s = len(basin_data['thermo_meas']['afta'])
        basin_data['thermo_meas']['node'] = np.zeros((s))
        lat1 = basin_data['latitude']
        lon1 = basin_data['longitude']
        lat2 = basin_data['thermo_meas']['latitude']
        lon2 = basin_data['thermo_meas']['longitude']
        for i in range(s):
            index = np.argmin(np.sqrt((lat1 - lat2[i])**2 + (lon1 - lon2[i])**2))
            basin_data['thermo_meas']['node'][i] = index
        basin_data['thermo_meas']['node'] = basin_data['thermo_meas']['node'].astype(int)
        
def Low_Temperature_Thermochronology3(basin_data, filename):
    
    # open filename
    data = pd.read_csv(filename)
    data = data[data['inverse'] == 1]
    
    # Define column type mappings for conversion
    type_mapping = {'name': 'str',          
                    'node': 'int',
                    'model': 'int',}
    
    # Convert dataframe to dictionary with specified array types
    data_dict = {col: data[col].to_numpy(dtype=type_mapping[col]) if col in type_mapping 
                 else data[col].to_numpy() 
                 for col in data.columns}
    
    basin_data['thermo_meas'] = data_dict
    
    # use latitude and longitude to locate sample if ind is not provided
    if not 'node' in basin_data['thermo_meas']:
        s = len(basin_data['thermo_meas']['afta'])
        basin_data['thermo_meas']['node'] = np.zeros((s))
        lat1 = basin_data['latitude']
        lon1 = basin_data['longitude']
        lat2 = basin_data['thermo_meas']['latitude']
        lon2 = basin_data['thermo_meas']['longitude']
        for i in range(s):
            index = np.argmin(np.sqrt((lat1 - lat2[i])**2 + (lon1 - lon2[i])**2))
            basin_data['thermo_meas']['node'][i] = index
        basin_data['thermo_meas']['node'] = basin_data['thermo_meas']['node'].astype(int)
        
    if 'ahea' in basin_data['thermo_meas']:
        basin_data['thermo_meas']['ahea'] = np.round(basin_data['thermo_meas']['ahea'], 2)
    
        
def Cosmogenic_Nuclide(basin_data, lat, lon, tcn, error, typ, node=[], name=[]):
    
    if name:
        basin_data['cosmo_meas']['name'] = np.array(name)
    
    if node:
        basin_data['cosmo_meas']['node'] = np.array(node)
    else:
        s = len(tcn)
        basin_data['cosmo_meas']['node'] = np.zeros((s))
        lat1 = basin_data['latitude']
        lon1 = basin_data['longitude']
        lat2 = lat
        lon2 = lon
        for i in range(s):
            index = np.argmin(np.sqrt((lat1 - lat2[i])**2 + (lon1 - lon2[i])**2))
            basin_data['cosmo_meas']['node'][i] = index
    basin_data['cosmo_meas']['node'] = basin_data['cosmo_meas']['node'].astype(int)
    
    basin_data['cosmo_meas']['latitude'] = np.array(lat)
    basin_data['cosmo_meas']['longitude'] = np.array(lon)
    
    basin_data['cosmo_meas']['ind'] = [np.array([]) for i in range(len(basin_data['cosmo_meas']['node']))]
    for i in range(len(basin_data['cosmo_meas']['node'])):
        basin_data['cosmo_meas']['ind'][i] = Find_Upstream_Index(basin_data['cosmo_meas']['node'][i], basin_data['pairs'])
    
    basin_data['cosmo_meas']['tcn'] = np.array(tcn)
    basin_data['cosmo_meas']['tcn_error'] = np.array(error)
    basin_data['cosmo_meas']['type'] = np.array(typ)


def Cosmogenic_Nuclide2(basin_data, filename):
    
    # open filename
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # add text line to the basin dictionnary
    for line in lines:
        ls = line.strip()
        # ignore line starting with a hash
        if not ls.startswith('#'):
            key, value = line.strip().split(':')
            if key == 'name':
                value = [str(x) for x in value.split(',')]
            elif key == 'node' or key == 'type':
                value = [int(x) for x in value.split(',')]
            else:
                value = [float(x) for x in value.split(',')]
            basin_data['cosmo_meas'][key.strip()] = np.array(value)
    
    # use latitude and longitude to locate sample if ind is not provided
    if not 'node' in basin_data['cosmo_meas']:
        s = len(basin_data['cosmo_meas']['tcn'])
        basin_data['cosmo_meas']['node'] = np.zeros((s))
        lat1 = basin_data['latitude']
        lon1 = basin_data['longitude']
        lat2 = basin_data['cosmo_meas']['latitude']
        lon2 = basin_data['cosmo_meas']['longitude']
        for i in range(s):
            index = np.argmin(np.sqrt((lat1 - lat2[i])**2 + (lon1 - lon2[i])**2))
            basin_data['cosmo_meas']['node'][i] = index
        basin_data['cosmo_meas']['node'] = basin_data['cosmo_meas']['node'].astype(int)
    
    # find upstream index of the cosmo sample location (node)
    basin_data['cosmo_meas']['ind'] = [np.array([]) for i in range(len(basin_data['cosmo_meas']['node']))]
    for i in range(len(basin_data['cosmo_meas']['node'])):
        basin_data['cosmo_meas']['ind'][i] = Find_Upstream_Index(basin_data['cosmo_meas']['node'][i], basin_data['pairs'])

def Cosmogenic_Nuclide3(basin_data, filename):
    
    # open filename
    data = pd.read_csv(filename)
    
    # Define column type mappings for conversion
    type_mapping = {'name': 'str',          
                    'node': 'int',
                    'type': 'int'}
    
    # Convert dataframe to dictionary with specified array types
    data_dict = {col: data[col].to_numpy(dtype=type_mapping[col]) if col in type_mapping 
                 else data[col].to_numpy() 
                 for col in data.columns}
    
    basin_data['cosmo_meas'] = data_dict
    
    # use latitude and longitude to locate sample if ind is not provided
    if not 'node' in basin_data['cosmo_meas']:
        s = len(basin_data['cosmo_meas']['tcn'])
        basin_data['cosmo_meas']['node'] = np.zeros((s))
        lat1 = basin_data['latitude']
        lon1 = basin_data['longitude']
        lat2 = basin_data['cosmo_meas']['latitude']
        lon2 = basin_data['cosmo_meas']['longitude']
        for i in range(s):
            index = np.argmin(np.sqrt((lat1 - lat2[i])**2 + (lon1 - lon2[i])**2))
            basin_data['cosmo_meas']['node'][i] = index
        basin_data['cosmo_meas']['node'] = basin_data['cosmo_meas']['node'].astype(int)
    
    # find upstream index of the cosmo sample location (node)
    basin_data['cosmo_meas']['ind'] = [np.array([]) for i in range(len(basin_data['cosmo_meas']['node']))]
    for i in range(len(basin_data['cosmo_meas']['node'])):
        basin_data['cosmo_meas']['ind'][i] = Find_Upstream_Index(basin_data['cosmo_meas']['node'][i], basin_data['pairs'])
        
def Variable_Hillslope_lenght(basin_data, param, random=False, minimum=100, maximum=200):
    
    param.hlflag = 1
    
    if random:
        basin_data['hillslope'] = {}
        basin_data['hillslope']['length'] = np.random.randint(100, 200, len(basin_data['initial_elevation']))