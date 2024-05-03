import numpy as np
from numba import jit
from init import * 

@jit(nopython=True)
def rho(psi_array):
    '''
    Calculates the density at points r for protons and neutrons

    Parameters
    ----------
    psi : nd array
        array containing the wavefunctions for a particular q. should have size Z x len(grid) or N x len(grid)
    r : nd array
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    rho_p = np.zeros(len(grid))
    rho_n = np.zeros(len(grid))
    
    for n in range(nmax+1):
        for l in range(lmax+1):
            jArr = []
            for s in range(len(spin)):
                if l == 0:
                    j = .5
                    jArr.append(j)
                else:
                    j = l + spin[s]
                    jArr.append(j)
            jArr = set(jArr)
            for i,j in enumerate(jArr):
                #print('n',n,'l',l,'j',j)
                rho_p += ((2*j+1)*psi_array[0][n][l][i]**2) /(4*np.pi*grid**2)
                rho_n += ((2*j+1)*psi_array[1][n][l][i]**2) /(4*np.pi*grid**2)
    cutoff_p = rho_p[Rp_ind]
    cutoff_n = rho_n[Rp_ind]
    for k in range(0,Rp_ind):
        rho_p[k] = rho_p[Rp_ind]
        rho_n[k] = rho_n[Rp_ind]
    rho_tot = rho_p + rho_n
    return rho_tot,rho_p,rho_n

