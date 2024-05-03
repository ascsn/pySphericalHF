import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from scipy.integrate import solve_bvp
from scipy import special
import utilities
import math
'''
Contains: all the solvers needed to solve 1d single particle self consistent 
problems
'''


###############################################################################
'''

Contents: -1d self consistent solver using pseudospectal method.
          -1d unitary time dependent solver
              - in polynomial basis expansion
              - in chebyshev basis expansion
          - system of ODEs BVP solver using RK4
'''
###############################################################################
def MatrixSolve_SC_hermitian(H_func,psiArr,psiStarArr,int_weights,params):
    '''
    Self-consistent solve for the ground state of a non-linear system defined by H
    Parameters
    ----------
    H_func : function type
        DESCRIPTION.
    psiArr : ndarry
        Initial guess for ground state psi.
    psiStarArr : ndarray
        conjugate of psiArr.
    int_weights : ndarray
        integration weights defined on collocation points.
    params : dictionary
        interaction and system parms.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
    niter = params['niter']
    sigma = params['sigma']
    #since the problem is assumed to be a simple 1d, we store the self consistent 
    #iterations of the wavefunction

    seriesShape = np.concatenate([[niter],psiArr.shape])
    psiSeries = np.zeros(seriesShape)
    psiStarSeries = np.zeros(seriesShape)
    evalSeries = np.zeros(niter)
    '''
    # Array to store the mixing parameter
    sigmaSeries = np.zeros(niter) #Array to contain the mixing parameter
    sigmaSeries[0] = sigma
    rSeries = np.zeros(seriesShape) # residual array
    
    ## intialize the first iteration with psiArr and psiStar
    psiSeries[0] = psiArr
    psiStarSeries[0] = psiStarArr
    '''
    # main self-consistent loop
    for l in np.arange(1,niter):
        H = H_func(psiSeries[l-1],psiStarSeries[l-1],BC=True)
        evals,evects = sci.linalg.eig(H)
        idx = np.real(evals).argsort()
        evals = np.real(evals[idx])
        evects = evects[:,idx]
        evects = evects.T
        
        psi = evects[0]
        psi = np.concatenate([[0],psi,[0]]) # put the zeros back in since we cut them off by BCs
        psiStar = np.conjugate(psi)
        

        psi = sigma*psi + (1.0-sigma)*psiSeries[l-1]
        psiStar = sigma*psiStar + (1.0-sigma)*psiStarSeries[l-1]
        
        psi_norm = utilities.normalize(psi,int_weights)

        psiSeries[l] = psi/psi_norm
        psiStarSeries[l] = psiStar/psi_norm
        
        evalSeries[l] = evals[0]
        
        # self driving mixing doesn't work at the moment
        '''
        #mixing parameters give different results.
        #calculate residual
        rSeries[l] = psiSeries[l] - psiSeries[l-1]
        # compute the optimal mixing parameter
        if l > 1:
            r_norm = np.linalg.norm(rSeries[l] -rSeries[l-1])
            #print(r_norm)
            if r_norm < 10**(-4):
                #print('it vanished')
                sigmaSeries[l] = 0.0
            else:
                #print('it did not vanish')
                sigmaSeries[l] = -1.0*np.dot(rSeries[l-1],rSeries[l] -rSeries[l-1])/r_norm
                #print(np.dot(rSeries[l-1],rSeries[l] -rSeries[l-1]))
                #print(r_norm)
                #print(sigmaSeries[l])
        else: pass
        '''
        #print(f'GS Energy Iteration {l}: {evals[0]}')
    return evalSeries,psiSeries

def prop_cheb(psi,H,dt,prop_order,weights):
    psi = psi[1:len(psi)-1]
    dpsi = np.zeros(psi.shape,dtype='complex')
    dpsi_buffer = np.zeros((prop_order,len(psi)),dtype='complex')
    H_norm = np.linalg.norm(H,ord=1)
    H_normalized = H/H_norm
    tau = dt*H_norm
    dpsi_buffer[0] = psi
    dpsi_buffer[1] = np.matmul(H_normalized,psi)
    for k in np.arange(0,prop_order):
        if k  == 0:
            dpsi += (- 1.0j)**k * special.jv(k,tau)*dpsi_buffer[k]
        elif k ==1 :
            dpsi += 2.0*(- 1.0j)**k * special.jv(k,tau)*dpsi_buffer[k]
        else:
            dpsi_buffer[k] = 2.0*np.matmul(H_normalized,dpsi_buffer[k-1]) \
                - dpsi_buffer[k-2]
            dpsi += (2.0*(- 1.0j)**k) * special.jv(k,tau)*dpsi_buffer[k]
    norm = utilities.normalize(dpsi,weights)
    dpsi = dpsi/norm
    dpsi = np.concatenate([[0],dpsi,[0]])
    return dpsi

def prop(psi,H,dt,prop_order,weights):
    psi = psi[1:len(psi)-1]
    dpsi = np.zeros(psi.shape,dtype='complex')
    dpsi_buffer = np.zeros((prop_order,len(psi)),dtype='complex')
    H_norm = np.linalg.norm(H,ord=1)
    H_normalized = H/H_norm
    tau = dt*H_norm
    dpsi_buffer[0] = psi
    dpsi_buffer[1] = np.matmul(H_normalized,psi)
    for k in np.arange(0,prop_order):
        if k  == 0:
            dpsi += dpsi_buffer[k]
        elif k ==1 :
            dpsi += ((- 1.0j*tau)**k)*dpsi_buffer[k]
        else:
            dpsi_buffer[k] = np.matmul(H_normalized,dpsi_buffer[k-1]) #- dpsi_buffer[k-2]
            dpsi += ((- 1.0j*tau)**k)*dpsi_buffer[k]/math.factorial(k)
    norm = utilities.normalize(dpsi,weights)
    dpsi = dpsi/norm
    dpsi = np.concatenate([[0],dpsi,[0]])
    return dpsi


###############################################################################

## This section contains the 1d BVP solver for classical equations of motion

###############################################################################



def bvp(t,x,gradV,params):
    # x is a vector of all the independent variables
    x1, x2 = x
    dx1dt = x2
    dx2dt = gradV(x1,params)*np.exp(-2*1j*(params['theta']))
    return [dx1dt,dx2dt]
def bc(xa,xb,xi,xf):
    #note this is the value of the full sol x at the boundaries
    x1a, x2a = xa  # notion is solution of x1 at the a boundary (x1a), solution of x2 at the a boundary (x2a)
    x1b, x2b = xb
    return([x1a-xi,x1b-xf])






