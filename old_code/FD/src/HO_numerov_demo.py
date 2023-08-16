'''
Harmonic Oscillator using Numerov Demo
'''
import numpy as np
import scipy as sci
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
from numba import jit

def getPsi_x(n,k):
    '''
    Definition of exact HO wavefunction taken from Zettili page 240.
    
    Parameters
    ----------
    n : TYPE
        principle quantum number for SE equation
    k : TYPE
        harmonic oscillator parameter (mass*omega)^2 from the potential term \mu^2 \omega^{2} x^{2} in the SE.

    Returns
    -------
    wf : function
        1-d wavefunction for the 1d-harmonic oscillator as a function of position x. 
    '''
    herm = special.hermite(n)
    def wf(x):
        result = (1/np.sqrt(np.sqrt(np.pi)*2**(n)*np.math.factorial(n)*(k)**(.25)))*np.exp(-x**(2)*np.sqrt(k)/2)*herm(x*k**(.25))
        return(result)
    return wf
def getExactLambda(n,mass,alpha):
    '''
    Exact eigenvalues of the HO equation. -f''(x) + k x^2 f(x) = 2 m E f(x)
    Lambda = 2 m E  
    E = (n + .5) \omega 
    \alpha = m^2 \omega^2 
    \omega = \sqrt{alpha/m^2}
    Parameters
    ----------
    n : float
        principle quantum number. float integers
    omega : float
        oscillator frequency.

    Returns
    -------
    float
        oscillator energy 2mE. 
    '''
    return 2*mass*(.5 + n)*np.sqrt(alpha/mass**2)
def pot(x,alpha):
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
#@jit(nopython=True)
def solve_wf_Numerov(psi,g_array,init_side):
    nbox = len(g_array)
    if init_side == 'left':
        psi[0] = 0
        psi[1] = 10**-3
        for i in range(2,nbox):
            a1 = 1.0 + ((h**2)/12)*g_array[i]
            a2 = 2*(1 - ((5*h**2)/12)*g_array[i-1])
            a3 = -(1+ ((h**2)/12)*g_array[i-2])
            psi[i] = (a2*psi[i-1] + a3*psi[i-2])/a1
        #norm = 0.0
        #for i in range(0,nbox):
        #    norm += psi[i]**2
        #norm = np.sqrt(norm)
        #psi = psi/norm
        return psi
    elif init_side =='right':
        #psi = psi[::-1]
        psi[nbox-1] = 0
        psi[nbox-2] = 10**-3
        for i in np.arange(nbox-3,-1,-1):
            a1 = 1.0 + ((h**2)/12)*g_array[i]
            a2 = 2*(1 - ((5*h**2)/12)*g_array[i+1])
            a3 = -(1+ ((h**2)/12)*g_array[i+2])
            psi[i] = (a2*psi[i+1] + a3*psi[i+2])/a1
            
        #norm = 0.0
        #for i in range(0,nbox):
        #    norm += psi[i]**2
        #norm = np.sqrt(norm)
        return psi#/norm
    else:
        print('Invalid Wavefunction Initialization')

#@jit(nopython=True)
def solve_E(psi_init,E0,dE,V):
    '''
    Solves the ODE y''(x) + g(x,E)y(x) = 0 where g(x,E) = E - V(x). E is the eigenvalue of the diff operator.
    The function calls solve_wf_Numerov to solve the ODE given an array for the function g(x,E). To make sure
    the BCs are satisfied, we iterate the eigenvalue E such that psi satisfies the BC. This is done by 

    Parameters
    ----------
    psi_init : nd_array
        Guess for the initial wavfunction
    E0 : float
        Guess for initial energy.
    dE : float
        step size for incrementing energy.
    V : nd array
        the "potential" term evaluated everywhere in the domain.

    Returns
    -------
    E : float
        converged energy.
   psi: nd array
        convered wavefunction.

    '''
    # Construct g-function
    nbox = len(V)
    g_array = E0 - V
    
    psi_l_init = psi_init.copy()
    psi_l = solve_wf_Numerov(psi_l_init,g_array,init_side='left')
    
    
    P1 = psi_l[nbox-1] # grab right side boundary 
    dE_init = dE
    El = E0 + dE
    
    counter = 0
    ### evolve the left initilized one.
    while abs(dE) > 10**-12:
    #for g in range(0,2):
        g_array = El - V
        psi_l = solve_wf_Numerov(psi_l,g_array,init_side='left')
        P2 = psi_l[nbox -1] # grab right side boundary value again
        if P1*P2 < 0:
            dE = - dE/2.0
        El = El + dE
        P1 = P2 # reset boundary. #P1 is boundary value of n-1 iteration and P2 is the value of n interation
        counter += 1
    print(f'Left Converged in {counter} Iterations.')
    print('left ',El)
    
    ### Reinitialize the wavefunctions and arrays to solve for the right side initialized wavefunctions
    g_array = E0 - V
    
    psi_r_init = psi_init.copy()
    psi_r = solve_wf_Numerov(psi_r_init,g_array,init_side='right')
    
    P1 = psi_r[0] # grab left side boundary 
    dE = dE_init #reinit the dE iterator
    Er = E0 + dE
    counter = 0
    while abs(dE) > 10**-12:
        g_array = Er - V
        psi_r = solve_wf_Numerov(psi_r,g_array,init_side='right')
        P2 = psi_r[0] # grab left side boundary value again
        if P1*P2 < 0:
            dE = - dE/2.0
        Er = Er + dE
        P1 = P2 # reset boundary. #P1 is boundary value of n-1 iteration and P2 is the value of n interation
        counter += 1
    # Now merge the right and left solutions together.
    print(f'Right Converged in {counter} Iterations.')
    print('right ',Er)
    if abs(El-Er) < 10**-15:
        E = El
    else: 
        raise Exception('Right and Left Energies dont match. One of theme did not converge. Stopping Solve')
    psi = np.concatenate((psi_l[:int((nbox-1)/2)],psi_r[int((nbox-1)/2):]))
    norm = 0.0
    for i in range(0,nbox):
        norm += psi[i]**2
    norm = np.sqrt(norm)
    psi = psi/norm
    return E,psi


#First define global variables
h = 10**(-2) ### grid spacing for domain (Warning around 10**(-3) it starts to get slow).
### HO global parameters 
n = 0 # principle quantum number to solve in HO
mass = 1.0 # mass for the HO system
# define the domain boundaries
x_a = -10 # left boundary 
x_b = 10 # right boundary 
x_array = np.arange(x_a,x_b+h,h)
m = len(x_array) 
print('Number of grid points: ',m)

alpha = .5
E0 = .2
dE = 10**-2

V_array = pot(x_array,alpha)
k = E0 - V_array
psi_0 = np.zeros(m)

#psi_test = solve_wf(psi_0,k,m)
#print(m -1)
#print(psi_test[m-1])
psi_exact = getPsi_x(n,alpha)
E_0,psi_test = solve_E(psi_0,E0,dE,V_array)
print('Numerical E_0: ',E_0)
print('Exact: ',getExactLambda(0,mass,alpha))

plt.plot(x_array,psi_test)
plt.plot(x_array,psi_exact(x_array)/np.linalg.norm(psi_exact(x_array)))
plt.show()
