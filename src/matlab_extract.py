import scipy.io as sio
from pathlib import Path

dirname = str(Path(__file__).parent.absolute())

def Matlab_Extract(filename=None, data={}, loaded=None):
    data = {}
    if filename:
        vrs = sio.whosmat(filename)
        name = vrs[0][0]
        loaded = sio.loadmat(filename,struct_as_record=True)
        loaded = loaded[name]
    whats_inside = loaded.dtype.fields
    if whats_inside is None:
        data = sio.loadmat(filename,struct_as_record=True)
    else:
        fields = list(whats_inside.keys())
        for field in fields:
            if len(loaded[0,0][field].dtype) > 0:
                data[field] = {}
                data[field] = Matlab_Extract(data=data[field], loaded=loaded[0,0][field])
            else: # it's a variable
                data[field] = loaded[0,0][field]
    return data

if __name__ == '__main__':
    
    # open LSDn constants
    LSDn = Matlab_Extract(dirname + "/cosmo-constants/consts_LSD.mat")
    # open cosmogenic nuclide constants
    al_be_constant_v22 = Matlab_Extract(dirname + "/cosmo-constants/al_be_consts_v22.mat")
    # open era40atm constants
    # ERA40 = Matlab_Extract(dirname + "/atmos-constants/ERA40.mat")

    