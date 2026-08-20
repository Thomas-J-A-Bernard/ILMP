import numpy as np
from matlab_extract import Matlab_Extract
from pathlib import Path
from sys import platform
from neutrons import Neutrons
from neutronslowe import NeutronsLowE
from protons import Protons
from muons import Muons

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

consts = Matlab_Extract(home_dirname + "/data/cosmo-constants/consts_LSD.mat")

class Site():
    pass

def LSDscaling(h, Rc, SPhi, w, consts, nuclide):
    '''
    DESCRIPTION:
        Implements the Lifton Sato Dunai scaling scheme for spallation.
    ----------
    PARAMETERS:
    h : float
        atmospheric pressure (hPa)
    Rc : array of float
        cutoff rigidity (GV)
    SPhi : array of float
        solar modulation potntial (Phi, see source paper)
    w : int
        fractional water content of ground (nondimensional)
    consts : dictionnary
        cosmogenic constants
    nuclide : int
        nuclide of interest: 
        26 for 26Al, 10 for 10Be, 14 for 14C, 3 for 3He, 0 for nucleon flux
    -------
    RETURNS
    site : structure
        spallation scaling
    '''
    
    mfluxRef = consts['mfluxRef']
    muRef = (mfluxRef['neg'] + mfluxRef['pos'])
    
    # Select reference values for nuclide of interest or flux
    if nuclide == 3:
        HeRef = consts['P3nRef'] + consts['P3pRef']
    elif nuclide == 10:
        BeRef = consts['P10nRef'] + consts['P10pRef']
    elif nuclide == 14:
        CRef = consts['P14nRef'] + consts['P14pRef']
    elif nuclide == 26:
        AlRef = consts['P26nRef'] + consts['P26pRef']
    else:
        SpRef = consts['P3nRef'] + consts['P3pRef']
        
    EthRef = consts['ethfluxRef']
    ThRef = consts['thfluxRef']
    
    # Site nucleon fluxes
    NSite = Neutrons(h, Rc, SPhi, w, consts, nuclide)
    ethflux, thflux = NeutronsLowE(h, Rc, SPhi, w)
    PSite = Protons(h, Rc, SPhi, consts, nuclide)
    
    # Site omnidirectional muon flux
    mflux = Muons(h, Rc, SPhi)          # Generates muon flux at site from Sato et al. (2008) model
    muSite = (mflux.neg + mflux.pos)
    
    site = Site()
    # Nuclide-specific scaling factors as f(Rc)
    if nuclide == 3:
        site.He = (NSite.P3n + PSite.P3p)/HeRef
    elif nuclide == 10:
        site.Be = (NSite.P10n + PSite.P10p)/BeRef
    elif nuclide == 14:
        site.C = (NSite.P14n + PSite.P14p)/CRef
    elif nuclide == 26:
        site.Al = (NSite.P26n + PSite.P26p)/AlRef
    else:    
        # Total nucleon flux scaling factors as f(Rc)
        site.sp = ((NSite.nflux + PSite.pflux))/SpRef       # Sato et al. (2008) Reference hadron flux integral >1 MeV
    
    site.E = NSite.E;               # Nucleon flux energy bins
    site.eth = ethflux/EthRef       # Epithermal neutron flux scaling factor as f(Rc)
    site.th = thflux/ThRef          # Thermal neutron flux scaling factor as f(Rc)

    # Differential muon flux scaling factors as f(Energy, Rc)
    site.muE = mflux.E              # Muon flux energy bins (in MeV)
    site.mup = mflux.p              # Muon flux momentum bins (in MeV/c)
    
    site.muSF = np.zeros((len(Rc), len(NSite.E)))
    for i in range(len(Rc)):
        site.muSF[i,:] = muSite[i,:]/muRef;
    
    # Integral muon flux scaling factors as f(Rc)
    site.muTotal = mflux.total/mfluxRef['total']                # Integral total muon flux scaling factor
    site.mn = mflux.nint/mfluxRef['nint']                       # Integral neg muon flux scaling factor
    site.mp = mflux.pint/mfluxRef['pint']                       # Integral pos muon flux scaling factor
    site.mnabs = mflux.nint                                     # Integral neg muon flux
    site.mpabs = mflux.pint                                     # Integral pos muon flux 
    
    return site
    