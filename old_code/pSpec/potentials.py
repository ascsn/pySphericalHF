import numpy as np


def HO(r,params):
    return params['kappa']*r**2
def ws(r,params):
    result = params['V0']/(1+np.exp((r-params['R'])/params['a']))
    return(result)
def coulomb(r,params):
    '''
    Uses the classical expression for the coulomb force with a cutoff radius
    (typically taken to the the proton radius Rp)

    V(r) = Z e^2 /2Rp  * ( 3 - r/Rp) for r <= Rp
         = Z e^2 /r for r > Rp

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    '''

    result = np.zeros(r.shape)
    for i,rVal in enumerate(r):
        if rVal < params['r_cutoff']:
            result[i] = params['Z_pro']*params['e2']/(2*params['r_cutoff']) * ( 3.0 - (r[i]/params['r_cutoff'])**(2) )
        else:
            result[i] = params['Z_pro']*params['e2']/r[i]
    return result
def spin_orbit(r,j,l,params):
    '''
    Uses phenomenlogical expression for the spin-orbit potential:

        V_{so}(r) = V_{ls} (j(j+1) - l(l+1) - 3/4) r_{0}^2 dF(r)/dr * 1/r

    where V_[ls} and r_{0} are parameters, s = 1/2, and F(r)is the fermi functions

        F(r) = 1/(1 + exp((r - R)/a))
    The derivative of the  fermi function is done analytically and subbed in here.

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    l : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    '''

    R,a,Vls,r0 = params['R'], params['a'],params['Vls'],params['r0']
    dfdr =  - np.exp((r - R)/a)/(a*(1+ np.exp((r-R)/a))**2)
    #check_zero = np.where(r < params['r_cutoff'])
    result = Vls*r0**(2) * (j*(j+1) - l*(l+1) - .75)*dfdr/r
    #result[check_zero] = 0
    return result
def centrifugal(r,l):
    #check_zero = np.where(r < params['r_cutoff'])
    result = l*(l+1)/r**2
    #result[check_zero] = 0
    return result
