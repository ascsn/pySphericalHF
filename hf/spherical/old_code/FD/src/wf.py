import numpy as np
from scipy.special import genlaguerre
from scipy.special import assoc_laguerre
from init import *

sys.path.insert(0, '../../src/methods')
import utilities




'''
NOTE: HYDROGEN WAVEFUNCTIONS NOT WORKING
'''
def get_wfHO_radial(n,j):
    '''
    Parameters
    ----------
    n : Integer
        principle quantum number
    l : Integer
        orbital angular momentum quantum number.

    Returns
    -------
    psi : function
        radial wavefunction as a function of r.

    '''
    lag = genlaguerre(n,j)
    nu = .5
    def psi(r,shift):
        r_s = r - shift
        result = (r_s)**(j-.5) * lag(2*nu*(r_s)**2)  * np.exp(-nu*(r_s)**(2))
        return(result)

    return psi
def get_WfHydrogen_radial(n,l):
    '''
    Taken from Zettili pg 359. Distance scale is relative to charge radius of the proton.
    This means we take a_0 -> R_p

    Returns
    -------
    Radial part of the hydogen wavefunction.

    '''
    def psi(r):
        N = (2/(n*R_p))**(3/2) * np.sqrt(np.math.factorial((n - l - 1))/(2*n*np.math.factorial((n+l))**3))
        func = N* ((2*r/(n*R_p))**l) * np.exp(-r/(n*R_p))*assoc_laguerre(r,n-l-1,2*l +1)
        return func
    return psi


def initWfs(name='HO',shift=0):
    '''
    Function initializes wavefunctions for proton and neutron according to shell model

    Parameters
    ----------
    N : Integer
        Number of Neutrons.
    Z : Integer
        Number of Protons.
    name : string, optional
        Specify type of initial wavefunction. The default is 'HO'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    psi : TYPE
        DESCRIPTION.

    '''
    if name == 'HO':
        psi_dict = {}
        energy_dict = {}
        counter = 0
        for q in np.arange(0,2):
            for n in np.arange(0,nmax+1,1):
                lmax = n - 1
                jmax = lmax + .5
                for j in np.arange(.5,jmax+.5,1):
                    print(q,n,j)
                    psi_func = get_wfHO_radial(n,j)
                    eval_psi = psi_func(CPnts_mapped,shift)
                    norm = np.sqrt(utilities.innerProduct(eval_psi,eval_psi,int_weights))
                    psi_dict[f'{q}{n}{j}'] = eval_psi/norm
                    energy_dict[f'{q}{n}{j}'] = 0.0
                    counter += 1
        return psi_dict,energy_dict
    elif name == 'hydrogen':
        psi_dict = {}
        energy_dict = {}
        for n in np.arange(1,nmax+1,1):
            for l in np.arange(0,nmax,1):
                jmax = l + .5
                for j in np.arange(.5,jmax+.5,.5):
                    psi_func = get_wfHO_radial(n,j)
                    eval_psi = psi_func(grid,shift)
                    psi_dict[f'{n}{l}{j}'] = eval_psi
                    energy_dict[f'{n}{l}{j}'] = 0.0
        return psi_dict,energy_dict
    elif name=='test':
        psi_array = np.zeros((2,nmax,lmax+1,len(spin),len(grid)))
        energies_array = np.zeros((2,nmax,lmax+1,len(spin),1))
        return psi_array,energies_array
    else:
        raise ValueError('Only available wavfunctions are radial HO and hydrogen')
    return None

