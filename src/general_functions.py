import numpy as np
from pathlib import Path

# get absolute path of main directory
dirname = str(Path(__file__).parent.absolute())

def Find_Upstream_Index(node, pairs):
    '''
    DESCRIPTION:
        This function will return the indices of river nodes above a given river node
    ----------
    PARAMETERS
    node : ind
        river node
    pairs : array of int
        pairs of the drainage network
    -------
    RETURNS
    ind : array of int
        river nodes upstream of the given river node
    '''
    
    ind = node + 1
    i = 1
    while i > 0:
        I = np.argwhere(np.isin(pairs[:,0], ind) == True)
        i = len(I) - np.size(ind) + 1
        ind = np.append(ind, pairs[I, 1])
        uni_ind = np.unique(ind)
        ind = uni_ind
    
    return ind.astype(int) - 1

def Find_Downstream_Index(node, pairs):
    '''
    DESCRIPTION:
        This function will return the indices of river nodes above a given river node
    ----------
    PARAMETERS
    node : ind
        river node
    pairs : array of int
        pairs of the drainage network
    -------
    RETURNS
    ind : array of int
        river nodes downstream of the given river node
    '''
    
    I = np.argwhere(np.isin(pairs[:,1], node + 1) == True)
    receiver = pairs[I, 0]
    ind = receiver
    while receiver > 1:
        I = np.argwhere(np.isin(pairs[:,1], receiver) == True)
        receiver = pairs[I, 0]
        ind = np.append(ind, receiver)
        
    return ind.astype(int) - 1
