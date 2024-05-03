import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy import special

def discrete_der(data,h):
    ## uses 2nd order finite difference scheme to calculate a derivative
    # assumes uniform spacing
    fp = np.zeros(data.shape,dtype='complex')
    for i in range(len(data)):
        # apply one sided derivatives on boundaries
        if i == 0:
            fp[i] = (1/(2*h))*(-3*data[i] + 4*data[i+1] - data[i+2])
        elif i == len(data)-1:
            fp[i] = (1/(2*h))*(3*data[i] - 4*data[i-1] + data[i-2])
        # apply central to the interior
        else:
            fp[i] = (data[i + 1] - data[i - 1])/(2*h)
    return fp

def cl_period(x,V,E,h):
    integrand = 1.0/(2.0*np.sqrt(E - V(x)))
    T = 2.0*np.trapz(integrand,dx=h)
    return T

def get_tp():
    return

def action(x,h,V,params):
    S = np.zeros(x.shape,dtype='complex')
    dxdt = discrete_der(x,h)
    integrand = np.exp(2*params['theta']*1j)*dxdt**2 - V(x,params)
    integral = np.trapz(integrand,dx=h)
    S = params['expansion_const']*np.exp(-1j*params['theta'])*integral
    return S

def normalize(funcArr,weights):
    norm = 0.0
    for i in range(len(weights)-1):
        norm += weights[i]*funcArr[i]*np.conj(funcArr[i])
    norm = np.sqrt(norm)
    return norm