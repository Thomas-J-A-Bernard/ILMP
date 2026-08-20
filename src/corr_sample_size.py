import sys

def corr_sample_size(sample, nu, nux, kflag, litho):
    '''
    DESCRIPTION
        check if the size of the free model parameters are correct
    ----------
    PARAMETERS
    sample : numpy arrays of floats
        parameters for the forward modelling
    nu : int
        number of uplift step
    nux : int
        lateral variable erosional efficiency
    kflag : int
        lateral variable erosional efficiency
    litho : numpy arrays of floats
        erosional efficiency for each lithologies
    -------
    RETURNS
    None
    '''
    
    model_length = len(sample)
    
    if kflag == 0:
        required_length = (nu-1)+nux**2*nu+4
    else:
        required_length = (1+(2*(nu-1)))*nux**2+kflag*len(litho)
        
    if model_length < required_length:
        sys.exit("Too few input parameters: "+str(required_length-model_length)+" missing!")
    elif model_length > required_length:
        sys.exit("Too many input parameters: "+str(required_length-model_length)+" missing!")
        