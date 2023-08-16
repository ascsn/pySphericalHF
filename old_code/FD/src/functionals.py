import numpy as np
from init import *
from fields import *
import matplotlib.pyplot as plt
def h_BKN(rho):
    '''


    Parameters
    ----------
    rhoArr : nd array
        DESCRIPTION.
    r : float
        DESCRIPTION.

    Returns
    -------
    returns the functional value of h at a particular point r.

    '''
    a = 0.45979 # length parameter for yukawa potential (fm)

    t0 = -497.726#-1132.4 #MeVfm^3
    t3 = 17270.0 #23610.4 #MeVfm^6
    yuk = yukArr(rho)
    h = .75*t0*rho + 0.1875*t3*rho**2  + yuk

    return h

def h_Skyrme():
    return "No Skyrme interaction yet."

