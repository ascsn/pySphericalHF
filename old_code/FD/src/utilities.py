import numpy as np
from numba import jit
from init import *

#@jit(nopython=True)
def getNZ(rhoArr):
    nneu = 0.0
    npro = 0.0

    for i in np.arange(1,len(grid)-1):
        nneu += 4.0*np.pi*rhoArr[2][i] * grid[i]**2 * step_size
        npro += 4.0*np.pi*rhoArr[1][i] * grid[i]**2 * step_size
    return np.array([npro,nneu])

#@jit(nopython=True)
def getRadi(rhoArr):
    # Correction values taken from Alex Browns Nuclear structure notes (2020)
    Rcorr_sqr = 0.879**2 #+ (N/Z)*(-0.116)**2
    Rnsqr = 0.0
    Rpsqr = 0.0
    npro,nneu = getNZ(rhoArr)
    norm1 = np.linalg.norm(rhoArr[1])
    norm2 = np.linalg.norm(rhoArr[2])
    for i in np.arange(len(grid)):
        Rnsqr += 4*np.pi*(rhoArr[2][i] * grid[i]**4)*step_size
        Rpsqr += 4*np.pi*(rhoArr[1][i] * grid[i]**4)*step_size

    Rp = np.sqrt(Rnsqr/npro)
    Rn = np.sqrt(Rpsqr/nneu)
    Rch = np.sqrt(Rpsqr/npro + Rcorr_sqr )
    return Rp,Rn,Rch

@jit(nopython = True,fastmath=True)
def normalize(funcArr,weights):
    norm = 0.0
    for i in range(len(weights)-1):
        norm += weights[i]*funcArr[i]
    norm = np.sqrt(norm)
    return norm
