import numpy as np
from numba import jit
from init import *
import densities

aV0 = -166.9239#-166.9239 # strength of yukwawa potential (MeV)
### yukArr calculates the volume integral of a pure radial yukawa from [0,r]
@jit(nopython=True)
def yukArr(rhoArr):
    Vyuk = np.zeros(len(grid))
    for i,r in enumerate(grid):
        I = 0.0
        for j in range(i):
            rp = grid[j]
            I += rhoArr[j]*(np.exp(-abs(r-rp)/a)/abs(r-rp)) *rp** 2 * step_size
        Vyuk[i] = I
    return Vyuk*aV0#*4.0*np.pi

### integrates the full 3d yukawa by doing the dr' d\theta' from [0,r] and [0,pi] integral over spherical coordinates
@jit(nopython=True)
def yukArr2(rhoArr):
    Vyuk = np.zeros(len(grid))
    for i,r in enumerate(grid):
        I = 0.0
        for thetap in np.arange(0,np.pi,step_size):
            for j in range(i):
                rp = grid[j]
                num = np.exp(-np.sqrt(r**2 + rp**2 - 2*r*rp*np.cos(thetap))/a)*np.sin(thetap)*rp**2
                denom = np.sqrt(r**2 + rp**2 - 2*r*rp*np.cos(thetap))
                if denom < 10**-20:
                    I += 0.0
                else:
                    I += rhoArr[j]*(num/denom)*step_size
        Vyuk[i] = I
    return Vyuk*aV0*2*np.pi


# Integrates 3d yukawa r from [0,R_{box}] ( sub for [0,\inf)) and \theta' from [0,pi]
@jit(nopython=True)
def yukArr3(rhoArr):
    Vyuk = np.zeros(len(grid))
    ## pick a location r
    for i,r in enumerate(grid):
        I = 0.0
        ## loop for integrating over theta prime
        for thetap in np.arange(0,np.pi,step_size):
            #loop for integrating over r coordinate
            for j in range(len(grid)):
                rp = grid[j]
                denom = np.sqrt(r**2 + rp**2 - 2*r*rp*np.cos(thetap))
                num = np.exp(-np.sqrt(r**2 + rp**2 - 2*r*rp*np.cos(thetap))/a)*np.sin(thetap)*rp**2
                if denom < 10**-20:

                    I += 0.0
                else:
                    I += rhoArr[j]*(num/denom)*step_size
        Vyuk[i] = I
    return Vyuk*aV0*2*np.pi
# Integrates pure radial yukawa r from [0,R_{box}]
@jit(nopython=True)
def yukArr4(rhoArr):
    Vyuk = np.zeros(len(grid))
    for i,r in enumerate(grid):
        I = 0.0
        for j in range(len(grid)):
            rp = grid[j]
            if rp == r:
                I += 0.0
            else:
                I += rhoArr[j]*(np.exp(-abs(r-rp)/a)/abs(r-rp)) * rp**2 * step_size
        Vyuk[i] = I
    return Vyuk*aV0*4.0*np.pi

@jit(nopython=True)
def coulombArr(rhoArr):
    Vc12 = np.zeros(len(grid))
    Vcinf = 0.0
    for i,r in enumerate(grid):
        I1 = 0.0
        I2 = 0.0
        for j in range(i):
            rp = grid[j]
            I1 += (rhoArr[j]*(rp**2))*step_size
            I2 += rhoArr[j]*rp*step_size
        Vc12[i] = I1/r - I2
    for i in range(0,len(grid)):
        Vcinf += rhoArr[i]*grid[i]*step_size

    Vc = 4*np.pi*e2*(Vc12 + Vcinf) - e2*(3.0/np.pi)**(1.0/3.0)*rhoArr**(1.0/3.0) ## add in exchange term
    return Vc

@jit(nopython=True)
def centriforceArr(l):
    result = l*(l+1)/grid**2
    for k in range(0,10):
        result[k] = result[10]
    return result

'''
def externalV(r,t):

    return None
'''