import numpy as np
def V_HO(x,alpha):
    '''
    1-d harmonic Oscillator potential

    Parameters
    ----------
    x : float or nd array
        position.
    alpha : float
        oscillator length parameter.

    Returns
    -------
    float or ndarray
        value of potential evaluated at x.

    '''
    return alpha*x**2

def V_rho(psi,psiStar,q):
    '''

    Parameters
    ----------
    psi : ndArray
        wavefunction.
    psiStar : TYPE
        conjugate wavefunction
    q : TYPE
        coupling constant.

    Returns
    -------
    result : ndArray
           nonlinear part of the GP equation.

    '''
    #print(type(psiStar))
    #print(type(psi))
    #print(type(q))
    #print(type(q*psiStar*psi))
    return q*psiStar*psi

def V_quartic(x,alpha,g):
    '''
    1-d harmonic Oscillator potential

    Parameters
    ----------
    x : float or nd array
        position.
    alpha : float
        oscillator strength parameter.
    g: float 
        stength of anharmonic oscillator.

    Returns
    -------
    float or ndarray
        value of potential evaluated at x.

    '''
    return alpha*x**2 + g*x**4

def V_double_well(x,g,b):
    '''
    1-d harmonic Oscillator potential

    Parameters
    ----------
    x : float or nd array
        position.
    alpha : float
        oscillator strength parameter.
    b: float 
        adjusts where the zeros of the 

    Returns
    -------
    float or ndarray
        value of potential evaluated at x.

    '''
    return g*(x**2 -b)**2
